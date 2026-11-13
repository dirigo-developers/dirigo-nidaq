"""Test NI coordination using real session lifecycles and recording hooks."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from unittest.mock import Mock

import nidaqmx
import numpy as np
import pytest
from dirigo import units
from dirigo.hardware.capture import (
    CaptureInfo,
    CaptureLayout,
    CaptureSettings,
    SampleArray,
)
from dirigo.hardware.capture_session import CaptureState
from dirigo.hardware.output import OutputState
from dirigo.hardware.scanner import GalvoScanner
from dirigo.hardware.timing import (
    DigitalRecordTrigger,
    DigitalStartTrigger,
    ExternalSampleClock,
)
from dirigo.hardware.waveform import AngularWaveform

from dirigo_nidaq import output as ni_output
from dirigo_nidaq.digitizer import (
    NICaptureSession,
    NICaptureSettings,
    NIDigitizer,
    NIDigitizerConfig,
)
from dirigo_nidaq.output import NIGalvoOutputSession
from dirigo_nidaq.scanner import NIGalvoScanner, NIGalvoScannerConfig
from dirigo_nidaq.synchronized import NISynchronizedScan


class RecordingCapture(NICaptureSession):
    """Keep capture states and reservations while replacing native operations."""

    def __init__(
        self,
        digitizer: NIDigitizer,
        settings: NICaptureSettings,
        events: list[str],
        failures: dict[str, Exception],
    ) -> None:
        super().__init__(digitizer, settings)
        self.events = events
        self.failures = failures
        self.reported_rate: units.SampleRate | None = settings.sample_rate

    def _step(self, event: str) -> None:
        self.events.append(event)
        if event in self.failures:
            raise self.failures[event]

    @property
    def sample_rate(self) -> units.SampleRate | None:
        """Return the test-controlled realized rate."""
        return self.reported_rate

    @property
    def sample_clock_terminal(self) -> str:
        """Advertise the AI terminal used in the routing assertion."""
        return "/Dev1/ai/SampleClock"

    def _prepare_capture(self) -> CaptureLayout:
        self._step("ai.prepare")
        settings = self._settings
        return CaptureLayout(
            shape=(
                settings.records_per_batch,
                settings.record_length,
                len(settings.channel_indices),
            ),
            channel_indices=settings.channel_indices,
            dtype=np.dtype(np.int16),
        )

    def _start_capture(self) -> None:
        self._step("ai.start")

    def _read_into(
        self,
        destination: SampleArray,
        *,
        timeout: float | None,
    ) -> CaptureInfo:
        self._step("ai.read")
        destination.fill(self.batches_acquired + 1)
        return CaptureInfo()

    def _stop_capture(self) -> None:
        self._step("ai.stop")

    def _close_capture(self) -> None:
        self._step("ai.close")


class RecordingOutput(NIGalvoOutputSession):
    """Keep NI binding validation and reservations while recording output hooks."""

    def __init__(
        self,
        waveforms: Mapping[GalvoScanner, AngularWaveform],
        *,
        external_clock: ExternalSampleClock,
        events: list[str],
        failures: dict[str, Exception],
    ) -> None:
        super().__init__(waveforms, external_clock=external_clock)
        self.events = events
        self.failures = failures
        self.complete = True
        self.wait_timeouts: list[float | None] = []

    def _step(self, event: str) -> None:
        self.events.append(event)
        if event in self.failures:
            raise self.failures[event]

    def _prepare_output(self) -> None:
        self._step("ao.prepare")

    def _start_output(self) -> None:
        self._step("ao.start")

    def _wait_output(self, timeout: float | None) -> bool:
        self._step("ao.wait")
        self.wait_timeouts.append(timeout)
        return self.complete

    def _stop_output(self) -> None:
        self._step("ao.stop")

    def _close_output(self) -> None:
        self._step("ao.close")


@dataclass(frozen=True)
class Rig:
    """Own test devices and expose sessions created for each fresh scan."""

    digitizer: NIDigitizer
    axis: NIGalvoScanner
    settings: NICaptureSettings
    events: list[str]
    failures: dict[str, Exception]
    captures: list[RecordingCapture]
    outputs: list[RecordingOutput]
    capture_factory: Mock
    output_factory: Mock
    scans: list[NISynchronizedScan]


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    """Use production bookkeeping while forbidding native task creation.

    Patch the public session factories. Recording subclasses retain the core
    state machines and reservations, so failure and reuse assertions exercise
    production ownership behavior rather than a second implementation.
    """
    monkeypatch.setattr(
        nidaqmx,
        "Task",
        Mock(side_effect=AssertionError("Unexpected NI task")),
    )
    digitizer = NIDigitizer(NIDigitizerConfig(physical_channels="Dev1/ai0"))
    axis = NIGalvoScanner(
        NIGalvoScannerConfig(
            angle_control_channel="Dev1/ao0",
            angle_range=units.AngleRange("-10 deg", "10 deg"),
            command_voltage_range=units.VoltageRange("-5 V", "5 V"),
        )
    )
    settings = NICaptureSettings(
        sample_rate=units.SampleRate("1 kS/s"),
        record_length=4,
        batch_count=2,
        input_voltage_range=units.VoltageRange("-5 V", "5 V"),
    )
    events: list[str] = []
    failures: dict[str, Exception] = {}
    captures: list[RecordingCapture] = []
    outputs: list[RecordingOutput] = []

    def make_capture(settings: CaptureSettings) -> NICaptureSession:
        assert isinstance(settings, NICaptureSettings)
        capture = RecordingCapture(digitizer, settings, events, failures)
        captures.append(capture)
        return capture

    def make_output(
        waveforms: Mapping[GalvoScanner, AngularWaveform],
        *,
        external_clock: ExternalSampleClock,
    ) -> NIGalvoOutputSession:
        events.append("ao.construct")
        if "ao.construct" in failures:
            raise failures["ao.construct"]

        output = RecordingOutput(
            waveforms,
            external_clock=external_clock,
            events=events,
            failures=failures,
        )
        outputs.append(output)
        return output

    capture_factory = Mock(side_effect=make_capture)
    output_factory = Mock(side_effect=make_output)
    monkeypatch.setattr(digitizer, "create_capture", capture_factory)
    monkeypatch.setattr(ni_output, "NIGalvoOutputSession", output_factory)

    result = Rig(
        digitizer=digitizer,
        axis=axis,
        settings=settings,
        events=events,
        failures=failures,
        captures=captures,
        outputs=outputs,
        capture_factory=capture_factory,
        output_factory=output_factory,
        scans=[],
    )
    try:
        yield result
    finally:
        failures.clear()
        for scan in reversed(result.scans):
            scan.close()
        axis.close()
        digitizer.close()


def _waveform(count: int = 8, rate: float = 1000.0) -> AngularWaveform:
    """Build a finite zero-angle waveform with controlled geometry."""
    return AngularWaveform([0.0] * count, sample_rate=units.SampleRate(rate))


def _scan(
    rig: Rig,
    *,
    settings: NICaptureSettings | None = None,
    waveform: AngularWaveform | None = None,
) -> NISynchronizedScan:
    """Create a fresh scan and register it for fixture cleanup."""
    scan = NISynchronizedScan(
        rig.digitizer,
        rig.settings if settings is None else settings,
        {rig.axis: _waveform() if waveform is None else waveform},
    )
    rig.scans.append(scan)
    return scan


def test_clock_routing_start_order_and_device_reuse(rig: Rig) -> None:
    """Route AO from prepared AI timing and arm its consumer first.

    The input trigger is retained only on capture. Closing an unfinished scan
    must clean up AO before AI, without parking or closing either device.
    A subsequent preparation proves both devices remain open and unreserved.
    """
    settings = replace(
        rig.settings,
        start_trigger=DigitalStartTrigger(source="/Dev1/PFI1"),
    )
    scan = _scan(rig, settings=settings)
    assert rig.events == []
    rig.capture_factory.assert_called_once_with(settings)

    scan.prepare()
    output = rig.outputs[0]
    assert output.external_clock == ExternalSampleClock(
        source="/Dev1/ai/SampleClock",
        max_rate=units.SampleRate("1 kS/s"),
    )
    assert output.start_trigger is None
    assert scan.layout.shape == (1, 4, 1)
    assert rig.events == ["ai.prepare", "ao.construct", "ao.prepare"]

    scan.start()
    assert rig.events[-2:] == ["ao.start", "ai.start"]

    scan.close()
    scan.close()
    assert rig.events[-4:] == ["ao.stop", "ao.close", "ai.stop", "ai.close"]

    retry = _scan(rig)
    retry.prepare()


def test_finite_exhaustion_waits_for_ao_without_redelivering_data(rig: Rig) -> None:
    """Deliver each AI batch once and keep EOF retryable until AO completes.

    Completion waiting must use no more than the caller's remaining budget.
    Retrying EOF must neither consume another batch nor overwrite the last
    valid data. Confirmed EOF is idempotent.
    """
    scan = _scan(rig)
    scan.prepare()
    scan.start()
    output = rig.outputs[0]
    output.complete = False
    samples = np.empty(scan.layout.shape, dtype=scan.layout.dtype)

    assert scan.read_into(samples, timeout=0.01) is not None
    np.testing.assert_array_equal(samples, 1)
    assert scan.read_into(samples, timeout=0.01) is not None
    np.testing.assert_array_equal(samples, 2)

    with pytest.raises(TimeoutError, match="AO output"):
        scan.read_into(samples, timeout=0.01)

    assert scan.state is CaptureState.STARTED
    assert rig.captures[0].state is CaptureState.COMPLETED
    waited = output.wait_timeouts[0]
    assert waited is not None
    assert 0 <= waited <= 0.01

    output.complete = True
    assert scan.read_into(samples, timeout=0) is None
    assert scan.state is CaptureState.COMPLETED
    assert scan.read_into(samples, timeout=0) is None
    assert output.wait_timeouts == [waited, 0.0]
    assert rig.events.count("ai.read") == 2
    np.testing.assert_array_equal(samples, 2)


@pytest.mark.parametrize(
    "stage",
    ["ai.prepare", "ao.construct", "ao.prepare", "ao.start", "ai.start"],
)
def test_failure_closes_all_owned_sessions_and_releases_devices(
    rig: Rig,
    stage: str,
) -> None:
    """Roll back preparation and startup failures without replacing their cause.

    Child sessions may clean themselves up before the coordinator sees an
    error. Group cleanup must remain safe and release every reservation so a
    fresh scan can prepare on the same devices.
    """
    failure = RuntimeError(f"{stage} failed")
    rig.failures[stage] = failure
    scan = _scan(rig)

    if stage.endswith("start"):
        scan.prepare()
        invoke = scan.start
    else:
        invoke = scan.prepare

    with pytest.raises(RuntimeError) as caught:
        invoke()

    assert caught.value is failure
    assert scan.state is CaptureState.CLOSED
    assert rig.captures[0].state is CaptureState.CLOSED
    assert all(output.state is OutputState.CLOSED for output in rig.outputs)

    rig.failures.clear()
    _scan(rig).prepare()


@pytest.mark.parametrize("during", ["start", "close"])
def test_cleanup_attempts_both_sessions_and_preserves_primary_error(
    rig: Rig,
    during: str,
) -> None:
    """Attempt both cleanups and preserve the operation or first cleanup error."""
    scan = _scan(rig)
    scan.prepare()
    rig.failures["ao.close"] = RuntimeError("AO cleanup failed")
    rig.failures["ai.close"] = RuntimeError("AI cleanup failed")

    if during == "start":
        primary = RuntimeError("AI start failed")
        rig.failures["ai.start"] = primary
        invoke = scan.start
    else:
        scan.start()
        primary = rig.failures["ao.close"]
        invoke = scan.close

    with pytest.raises(RuntimeError) as caught:
        invoke()

    assert caught.value is primary
    details = str(primary) + " ".join(getattr(primary, "__notes__", ()))
    assert "AO cleanup failed" in details
    assert "AI cleanup failed" in details
    assert rig.events.count("ao.close") == rig.events.count("ai.close") == 1

    scan.close()
    rig.failures.clear()
    _scan(rig).prepare()


def test_capture_timeout_and_argument_error_leave_scan_usable(rig: Rig) -> None:
    """Keep retryable failures open, then close AO after a fatal AI error."""
    scan = _scan(rig)
    scan.prepare()
    scan.start()
    samples = np.empty(scan.layout.shape, dtype=scan.layout.dtype)

    rig.failures["ai.read"] = TimeoutError("capture timed out")
    with pytest.raises(TimeoutError, match="capture timed out"):
        scan.read_into(samples, timeout=0)
    assert scan.state is CaptureState.STARTED

    with pytest.raises(ValueError, match="shape"):
        scan.read_into(np.empty((1,), dtype=scan.layout.dtype))
    assert scan.state is CaptureState.STARTED

    failure = RuntimeError("capture failed")
    rig.failures["ai.read"] = failure
    with pytest.raises(RuntimeError) as caught:
        scan.read_into(samples)

    assert caught.value is failure
    assert scan.state is CaptureState.CLOSED
    assert rig.outputs[0].state is OutputState.CLOSED


def test_stop_retains_reservations_until_close(rig: Rig) -> None:
    """Stop the consumer first while retaining both device reservations."""
    scan = _scan(rig)
    scan.prepare()
    scan.start()
    scan.stop()
    scan.stop()

    assert scan.state is CaptureState.STOPPED
    assert rig.events[-2:] == ["ao.stop", "ai.stop"]

    with pytest.raises(RuntimeError, match="reserved"):
        rig.digitizer.close()
    with pytest.raises(RuntimeError, match="reserved"):
        rig.axis.close()

    scan.close()
    _scan(rig).prepare()


@pytest.mark.parametrize(("count", "rate"), [(7, 1000.0), (8, 999.0)])
def test_mismatched_waveform_is_rejected_before_capture_creation(
    rig: Rig,
    count: int,
    rate: float,
) -> None:
    """Reject incompatible geometry and rates during passive construction."""
    with pytest.raises(ValueError, match="Waveform"):
        _scan(rig, waveform=_waveform(count, rate))

    rig.capture_factory.assert_not_called()


def test_different_device_is_rejected_before_capture_creation(rig: Rig) -> None:
    """Enforce the one-card scope before reserving any input resource."""
    other = NIGalvoScanner(replace(rig.axis.config, angle_control_channel="Dev2/ao0"))
    try:
        with pytest.raises(ValueError, match="same NI device"):
            NISynchronizedScan(rig.digitizer, rig.settings, {other: _waveform()})

        rig.capture_factory.assert_not_called()
    finally:
        other.close()


def test_coerced_capture_rate_is_rejected_before_output_creation(rig: Rig) -> None:
    """Roll back AI when its realized rate would retime the waveforms."""
    scan = _scan(rig)
    rig.captures[0].reported_rate = units.SampleRate("1.25 kS/s")

    with pytest.raises(ValueError, match="Prepared AI rate"):
        scan.prepare()

    assert scan.state is CaptureState.CLOSED
    rig.output_factory.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_count": None},
        {
            "external_clock": ExternalSampleClock(
                source="/Dev1/PFI0",
                max_rate=units.SampleRate("1 kS/s"),
            )
        },
    ],
)
def test_unsupported_clock_or_delivery_mode_is_rejected(
    rig: Rig,
    changes: dict[str, object],
) -> None:
    """Reject unsupported modes before creating or reserving child sessions."""
    with pytest.raises(NotImplementedError):
        _scan(rig, settings=replace(rig.settings, **changes))

    rig.capture_factory.assert_not_called()


def test_completion_wait_failure_closes_capture(rig: Rig) -> None:
    """Release AI when observing AO completion fails instead of timing out."""
    scan = _scan(rig)
    scan.prepare()
    scan.start()
    samples = np.empty(scan.layout.shape, dtype=scan.layout.dtype)
    assert scan.read_into(samples) is not None
    assert scan.read_into(samples) is not None

    failure = RuntimeError("AO completion failed")
    rig.failures["ao.wait"] = failure
    with pytest.raises(RuntimeError) as caught:
        scan.read_into(samples)

    assert caught.value is failure
    assert scan.state is CaptureState.CLOSED
    assert rig.captures[0].state is CaptureState.CLOSED

    rig.failures.clear()
    _scan(rig).prepare()


def test_stop_failure_attempts_both_children_and_releases_devices(rig: Rig) -> None:
    """Preserve the first stop error while attempting input stop and all cleanup."""
    scan = _scan(rig)
    scan.prepare()
    scan.start()

    failure = RuntimeError("AO stop failed")
    rig.failures["ao.stop"] = failure
    rig.failures["ai.stop"] = RuntimeError("AI stop failed")

    with pytest.raises(RuntimeError) as caught:
        scan.stop()

    assert caught.value is failure
    assert any("AI stop failed" in note for note in failure.__notes__)
    assert rig.events.count("ao.stop") == rig.events.count("ai.stop") == 1
    assert scan.state is CaptureState.CLOSED

    rig.failures.clear()
    _scan(rig).prepare()


def test_triggered_records_are_rejected_before_capture_creation(rig: Rig) -> None:
    """Keep triggered-record timing outside the current NI coordinator's scope."""
    with pytest.raises(NotImplementedError, match="consecutive records"):
        _scan(
            rig,
            settings=replace(
                rig.settings,
                record_trigger=DigitalRecordTrigger(source="external"),
            ),
        )

    rig.capture_factory.assert_not_called()
