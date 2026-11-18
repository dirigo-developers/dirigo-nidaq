"""Test synchronization without invoking native NI or digitizer operations."""

from collections.abc import Iterator
from dataclasses import dataclass
from unittest.mock import Mock

import nidaqmx
import numpy as np
import pytest
from dirigo import units
from dirigo.hardware.capture import CaptureLayout, CaptureSettings
from dirigo.hardware.capture_session import CaptureState, DigitizerCaptureSession
from dirigo.hardware.raster_frame_sync import (
    RasterFrameSyncSession,
    RasterFrameSyncState,
)
from dirigo.hardware.scanner import ResonantScanner, ResonantScannerConfig
from dirigo.hardware.timing import DigitalRecordTrigger, DigitalStartTrigger
from dirigo.hardware.waveform import AngularWaveform

from dirigo_nidaq import synchronization as sync_module
from dirigo_nidaq.frame_clock import NIFrameClockSession
from dirigo_nidaq.retriggered_output import NIRetriggeredGalvoOutputSession
from dirigo_nidaq.scanner import NIGalvoScanner, NIGalvoScannerConfig
from dirigo_nidaq.synchronization import (
    NIRasterFrameSync,
    NIRasterFrameSyncConfig,
)


@dataclass
class Participants:
    """Expose the coordinator and its mocked child operations."""

    sync: RasterFrameSyncSession
    capture: Mock
    clock: Mock
    output: Mock
    events: list[str]


@pytest.fixture
def participants(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Participants]:
    """Bind real configuration models to passive mocked participants."""
    monkeypatch.setattr(
        nidaqmx,
        "Task",
        Mock(side_effect=AssertionError("Must not create a native task.")),
    )

    events: list[str] = []

    capture = Mock(spec=DigitizerCaptureSession)
    capture.state = CaptureState.NEW
    capture.settings = CaptureSettings(
        sample_rate=units.SampleRate("1 MS/s"),
        record_length=16,
        records_per_batch=2,
        batch_count=4,
        record_trigger=DigitalRecordTrigger(source="external"),
    )
    capture.layout = CaptureLayout(
        shape=(2, 16, 1),
        channel_indices=(0,),
        dtype=np.dtype(np.int16),
    )

    def capture_prepared() -> None:
        events.append("capture.prepare")
        capture.state = CaptureState.PREPARED

    def capture_started() -> None:
        events.append("capture.start")
        capture.state = CaptureState.STARTED

    def capture_stopped() -> None:
        events.append("capture.stop")
        capture.state = CaptureState.STOPPED

    capture.prepare.side_effect = capture_prepared
    capture.start.side_effect = capture_started
    capture.stop.side_effect = capture_stopped

    clock = Mock(spec=NIFrameClockSession)
    clock.trigger = DigitalStartTrigger(source="/Dev1/Ctr0InternalOutput")
    clock.prepare.side_effect = lambda: events.append("clock.prepare")
    clock.start.side_effect = lambda: events.append("clock.start")
    clock.stop.side_effect = lambda: events.append("clock.stop")
    clock.close.side_effect = lambda: events.append("clock.close")

    output = Mock(spec=NIRetriggeredGalvoOutputSession)
    output.prepare.side_effect = lambda: events.append("output.prepare")
    output.start.side_effect = lambda: events.append("output.start")
    output.stop.side_effect = lambda: events.append("output.stop")
    output.close.side_effect = lambda: events.append("output.close")

    monkeypatch.setattr(
        sync_module,
        "NIFrameClockSession",
        Mock(return_value=clock),
    )
    monkeypatch.setattr(
        sync_module,
        "NIRetriggeredGalvoOutputSession",
        Mock(return_value=output),
    )

    fast = Mock(spec=ResonantScanner)
    fast.config = ResonantScannerConfig(
        nominal_frequency=units.Frequency("1 kHz"),
        max_amplitude=units.Angle("20 deg"),
        settling_time=units.Time("0 s"),
        frequency_tolerance=0.0,
    )

    slow = Mock(spec=NIGalvoScanner)
    slow.config = NIGalvoScannerConfig(
        angle_control_channel="Dev1/ao0",
        angle_range=units.AngleRange("±10 deg"),
        command_voltage_range=units.VoltageRange("±10 V"),
    )

    backend = NIRasterFrameSync(
        NIRasterFrameSyncConfig(
            record_clock_input="/Dev1/PFI0",
            max_record_rate=units.SampleRate("1 kS/s"),
            frame_counter_channel="Dev1/ctr0",
        )
    )
    sync = backend.create_session(
        fast_scanner=fast,
        slow_scanner=slow,
        capture=capture,
    )

    try:
        sync.configure(
            records_per_frame=4,
            frame_count=2,
            waveforms={
                slow: AngularWaveform(
                    [0.0, 0.001, 0.0, -0.001],
                    sample_rate=units.SampleRate("10 kS/s"),
                ),
            },
        )

        yield Participants(sync, capture, clock, output, events)
    finally:
        try:
            sync.close()
        finally:
            backend.close()


def test_lifecycle_order_and_borrowed_capture_ownership(
    participants: Participants,
) -> None:
    """Coordinate lifecycle while leaving reading and capture closure outside."""
    p = participants

    assert p.sync.state is RasterFrameSyncState.CONFIGURED
    assert p.events == []

    p.sync.prepare()
    p.sync.start()
    p.sync.stop()
    p.sync.close()
    p.sync.close()

    assert p.events == [
        "capture.prepare",
        "clock.prepare",
        "output.prepare",
        "output.start",
        "clock.start",
        "capture.start",
        "output.stop",
        "clock.stop",
        "capture.stop",
        "output.close",
        "clock.close",
    ]
    assert p.sync.state is RasterFrameSyncState.CLOSED

    p.capture.read_into.assert_not_called()
    p.capture.close.assert_not_called()
    p.output.wait.assert_not_called()


def test_capture_eof_does_not_implicitly_stop_synchronization(
    participants: Participants,
) -> None:
    """Capture exhaustion remains independent of synchronization lifecycle."""
    p = participants
    p.sync.prepare()
    p.sync.start()

    p.capture.read_into.return_value = None
    samples = np.empty((2, 16, 1), dtype=np.int16)

    assert p.capture.read_into(samples, timeout=0) is None
    assert p.sync.state is RasterFrameSyncState.STARTED

    p.clock.stop.assert_not_called()
    p.output.stop.assert_not_called()
    p.output.wait.assert_not_called()
    p.capture.close.assert_not_called()


@pytest.mark.parametrize(
    "stage",
    [
        "capture.prepare",
        "clock.prepare",
        "output.prepare",
        "output.start",
        "clock.start",
        "capture.start",
    ],
)
def test_failure_attempts_all_cleanup_and_preserves_primary_error(
    participants: Participants,
    stage: str,
) -> None:
    """Release owned sessions even when an operation and cleanup both fail."""
    p = participants
    failure = RuntimeError("primary operation failed")

    participant_name, operation_name = stage.split(".")
    participant = {
        "capture": p.capture,
        "clock": p.clock,
        "output": p.output,
    }[participant_name]
    operation = getattr(participant, operation_name)
    operation.side_effect = failure

    p.output.close.side_effect = RuntimeError("output cleanup failed")

    if operation_name == "start":
        p.sync.prepare()

    run = p.sync.prepare if operation_name == "prepare" else p.sync.start

    with pytest.raises(RuntimeError) as caught:
        run()

    assert caught.value is failure
    assert p.sync.state is RasterFrameSyncState.CLOSED

    p.output.stop.assert_called_once_with()
    p.clock.stop.assert_called_once_with()
    p.capture.stop.assert_called_once_with()
    p.output.close.assert_called_once_with()
    p.clock.close.assert_called_once_with()

    p.capture.close.assert_not_called()
    p.capture.read_into.assert_not_called()

    assert any(
        "output cleanup failed" in note
        for note in getattr(caught.value, "__notes__", ())
    )
