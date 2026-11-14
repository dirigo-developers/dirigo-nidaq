"""Exercise triggered-frame coordination with real child session lifecycles."""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import cast
from unittest.mock import Mock

import nidaqmx
import numpy as np
import pytest
from dirigo import units
from dirigo.hardware.capture import CaptureSettings, SampleArray
from dirigo.hardware.capture_session import CaptureState, DigitizerCaptureSession
from dirigo.hardware.output import OutputState
from dirigo.hardware.scanner import GalvoScanner
from dirigo.hardware.simulated_digitizer import (
    SimulatedDigitizer,
    SimulatedDigitizerConfig,
)
from dirigo.hardware.timing import DigitalStartTrigger
from dirigo.hardware.waveform import AngularWaveform

from dirigo_nidaq import output as ni_output
from dirigo_nidaq import triggered_frame as frame_module
from dirigo_nidaq.output import NIGalvoOutputSession
from dirigo_nidaq.scanner import NIGalvoScanner, NIGalvoScannerConfig
from dirigo_nidaq.triggered_frame import NITriggeredFrame


@dataclass
class Rig:
    """Retain devices, real sessions, and recording hooks for each test."""

    digitizer: SimulatedDigitizer
    axis: NIGalvoScanner
    settings: CaptureSettings
    monkeypatch: pytest.MonkeyPatch
    events: list[str]
    failures: dict[str, Exception]
    captures: list[DigitizerCaptureSession]
    outputs: list[NIGalvoOutputSession]
    frames: list[NITriggeredFrame]
    waits: list[float | None]
    complete: bool = True


def _step(rig: Rig, event: str) -> None:
    """Record an operation and raise its selected test failure, if any."""
    rig.events.append(event)
    if event in rig.failures:
        raise rig.failures[event]


def _record(
    rig: Rig,
    event: str,
    original: Callable[..., object],
) -> Callable[..., object]:
    """Wrap a real capture hook without replacing its lifecycle bookkeeping."""

    def invoke(*args: object, **kwargs: object) -> object:
        _step(rig, event)
        return original(*args, **kwargs)

    return invoke


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    """Use a core simulator and NI axes while forbidding native task creation.

    Capture retains its real data generation, validation, and reservations.
    NI output retains its real constructor and lifecycle; only driver hooks
    are replaced. The simulator supplies no physical trigger, so these tests
    prove sequencing and ownership rather than electrical synchronization.
    """
    task_factory = Mock(side_effect=AssertionError("Unexpected NI task"))
    monkeypatch.setattr(nidaqmx, "Task", task_factory)

    digitizer = SimulatedDigitizer(SimulatedDigitizerConfig(channel_count=1))
    axis = NIGalvoScanner(
        NIGalvoScannerConfig(
            angle_control_channel="Dev1/ao0",
            angle_range=units.AngleRange("-10 deg", "10 deg"),
            command_voltage_range=units.VoltageRange("-5 V", "5 V"),
        )
    )
    result = Rig(
        digitizer=digitizer,
        axis=axis,
        settings=CaptureSettings(
            sample_rate=units.SampleRate("100 MS/s"),
            record_length=4,
            records_per_batch=2,
            batch_count=2,
        ),
        monkeypatch=monkeypatch,
        events=[],
        failures={},
        captures=[],
        outputs=[],
        frames=[],
        waits=[],
    )

    def make_output(
        waveforms: Mapping[GalvoScanner, AngularWaveform],
        *,
        start_trigger: DigitalStartTrigger,
    ) -> NIGalvoOutputSession:
        """Keep the real NI constructor while recording its driver operations."""
        output = NIGalvoOutputSession(
            waveforms,
            start_trigger=start_trigger,
        )
        for hook, event in (
            ("_prepare_output", "ao.prepare"),
            ("_start_output", "ao.start"),
            ("_stop_output", "ao.stop"),
            ("_close_output", "ao.close"),
        ):
            monkeypatch.setattr(
                output,
                hook,
                Mock(side_effect=_record(result, event, lambda: None)),
            )

        def wait(timeout: float | None) -> bool:
            """Record the remaining budget and control observed completion."""
            _step(result, "ao.wait")
            result.waits.append(timeout)
            return result.complete

        monkeypatch.setattr(output, "_wait_output", wait)
        result.outputs.append(output)
        return output

    monkeypatch.setattr(ni_output, "NIGalvoOutputSession", make_output)
    try:
        yield result
    finally:
        result.failures.clear()
        for frame in reversed(result.frames):
            frame.close()
        for capture in reversed(result.captures):
            capture.close()
        axis.close()
        digitizer.close()
        task_factory.assert_not_called()


def _capture(rig: Rig) -> DigitizerCaptureSession:
    """Create a fresh capture and wrap its real driver hooks for recording."""
    capture = rig.digitizer.create_capture(rig.settings)
    rig.captures.append(capture)
    for hook, event in (
        ("_prepare_capture", "capture.prepare"),
        ("_start_capture", "capture.start"),
        ("_read_into", "capture.read"),
        ("_stop_capture", "capture.stop"),
        ("_close_capture", "capture.close"),
    ):
        rig.monkeypatch.setattr(
            capture,
            hook,
            _record(rig, event, getattr(capture, hook)),
        )
    return capture


def _frame(rig: Rig) -> NITriggeredFrame:
    """Register a frame whose AO grid differs from the capture sample grid."""
    frame = NITriggeredFrame(
        _capture(rig),
        {
            rig.axis: AngularWaveform(
                [0.0] * 20,
                sample_rate=units.SampleRate("20 kS/s"),
            )
        },
        start_trigger=DigitalStartTrigger(source="/Dev1/PFI0"),
    )
    rig.frames.append(frame)
    return frame


def _samples(frame: NITriggeredFrame) -> SampleArray:
    """Allocate caller-owned storage from the prepared capture layout."""
    return np.empty(frame.layout.shape, dtype=frame.layout.dtype)


def test_passive_construction_start_order_and_device_reuse(rig: Rig) -> None:
    """Arm NI before capture and release sessions without closing either device.

    The capture uses a non-NI core implementation. Its ADC grid contains
    sixteen samples per channel across two batches, while AO has twenty
    samples at a different rate. No equality between those grids is imposed.
    """
    frame = _frame(rig)
    assert frame.state is CaptureState.NEW
    assert frame.batches_acquired == 0
    assert rig.events == []

    frame.prepare()

    assert rig.events == ["capture.prepare", "ao.prepare"]
    assert rig.outputs[0].start_trigger == DigitalStartTrigger(source="/Dev1/PFI0")
    assert rig.outputs[0].external_clock is None
    assert frame.layout.shape == (2, 4, 1)

    frame.start()
    assert rig.events[-2:] == ["ao.start", "capture.start"]

    frame.close()
    assert rig.events[-4:] == [
        "ao.stop",
        "ao.close",
        "capture.stop",
        "capture.close",
    ]
    frame.close()

    assert rig.events.count("ao.close") == rig.events.count("capture.close") == 1
    assert rig.axis.angle is None
    assert frame.layout.shape == (2, 4, 1)
    _frame(rig).prepare()


def test_frame_spans_batches_and_eof_wait_is_retryable(rig: Rig) -> None:
    """Keep EOF retryable until AO finishes, without redelivering capture data.

    Two batches belong to the same finite frame. An AO timeout after the last
    batch leaves the frame started. Retrying EOF observes AO completion while
    preserving the last batch's samples and the delivered-batch count.
    """
    frame = _frame(rig)
    frame.prepare()
    frame.start()
    samples = _samples(frame)

    assert frame.read_into(samples, timeout=0) is not None
    assert frame.read_into(samples, timeout=0) is not None
    assert frame.batches_acquired == 2
    expected = samples.copy()

    rig.complete = False
    with pytest.raises(TimeoutError, match="AO output"):
        frame.read_into(samples, timeout=0)

    assert frame.state is CaptureState.STARTED
    assert rig.captures[0].state is CaptureState.COMPLETED

    rig.complete = True
    assert frame.read_into(samples, timeout=0) is None
    assert frame.state is CaptureState.COMPLETED
    assert frame.read_into(samples, timeout=0) is None
    assert rig.waits == [0.0, 0.0]
    assert rig.events.count("capture.read") == 2
    np.testing.assert_array_equal(samples, expected)

    frame.close()
    assert frame.batches_acquired == 2


@pytest.mark.parametrize(
    ("timeout", "expected"),
    [(None, None), (0.5, 0.375), (0.0625, 0.0)],
)
def test_ao_wait_uses_only_the_remaining_budget(
    rig: Rig,
    timeout: float | None,
    expected: float | None,
) -> None:
    """Share one EOF budget, polling AO if capture-call time has exhausted it."""
    frame = _frame(rig)
    frame.prepare()
    frame.start()
    samples = _samples(frame)
    frame.read_into(samples)
    frame.read_into(samples)
    rig.monkeypatch.setattr(
        frame_module,
        "monotonic",
        Mock(side_effect=[10.0, 10.125]),
    )

    assert frame.read_into(samples, timeout=timeout) is None
    assert rig.waits == [expected]


def test_unused_close_owns_sessions_and_leaves_devices_usable(rig: Rig) -> None:
    """Close both transferred sessions without preparing or parking the devices."""
    frame = _frame(rig)
    frame.close()
    frame.close()

    assert frame.state is CaptureState.CLOSED
    assert rig.captures[0].state is CaptureState.CLOSED
    assert rig.outputs[0].state is OutputState.CLOSED
    assert rig.axis.angle is None
    _frame(rig).prepare()


def test_frame_lifecycle_rejects_out_of_order_calls(rig: Rig) -> None:
    """Reject premature start, repeated preparation/start, and reuse after close."""
    frame = _frame(rig)
    with pytest.raises(RuntimeError, match="Frame must be"):
        frame.start()

    frame.prepare()
    with pytest.raises(RuntimeError, match="Frame must be"):
        frame.prepare()

    frame.start()
    with pytest.raises(RuntimeError, match="Frame must be"):
        frame.start()

    samples = _samples(frame)
    frame.close()
    with pytest.raises(RuntimeError, match="Frame must be"):
        frame.read_into(samples, timeout=0)
    frame.stop()


def test_capture_timeout_and_argument_error_retain_usability(rig: Rig) -> None:
    """Preserve active sessions after a retryable timeout or invalid destination."""
    frame = _frame(rig)
    frame.prepare()
    frame.start()
    samples = _samples(frame)

    rig.failures["capture.read"] = TimeoutError("capture waiting")
    with pytest.raises(TimeoutError, match="capture waiting"):
        frame.read_into(samples, timeout=0)
    with pytest.raises(ValueError, match="shape"):
        frame.read_into(np.empty((1,), dtype=frame.layout.dtype), timeout=0)

    assert frame.state is CaptureState.STARTED
    assert frame.batches_acquired == 0
    assert rig.outputs[0].state is OutputState.STARTED

    rig.failures.clear()
    assert frame.read_into(samples, timeout=0) is not None


@pytest.mark.parametrize(
    "stage",
    [
        "capture.prepare",
        "ao.prepare",
        "ao.start",
        "capture.start",
        "capture.read",
        "ao.wait",
    ],
)
def test_fatal_child_failure_closes_both_and_allows_reuse(
    rig: Rig,
    stage: str,
) -> None:
    """Roll back partial preparation, start, read, or completion failures.

    A child can close itself before the coordinator receives the error. Group
    cleanup must preserve the original exception, close its companion, and
    release both reservations so a new frame can use the same devices.
    """
    frame = _frame(rig)
    if stage in {"ao.start", "capture.start", "capture.read", "ao.wait"}:
        frame.prepare()

    if stage in {"capture.read", "ao.wait"}:
        frame.start()
        samples = _samples(frame)
        if stage == "ao.wait":
            frame.read_into(samples)
            frame.read_into(samples)

        def read_batch() -> object:
            """Read or observe finite completion using the prepared storage."""
            return frame.read_into(samples, timeout=0)

        operation: Callable[[], object] = read_batch
    else:
        operation = frame.start if stage.endswith("start") else frame.prepare

    failure = RuntimeError(f"{stage} failed")
    rig.failures[stage] = failure

    with pytest.raises(RuntimeError, match=f"{stage} failed") as caught:
        operation()

    assert caught.value is failure
    assert frame.state is CaptureState.CLOSED
    assert rig.outputs[0].state is OutputState.CLOSED
    assert rig.captures[0].state is CaptureState.CLOSED

    rig.failures.clear()
    _frame(rig).prepare()


@pytest.mark.parametrize("during", ["start", "stop", "close"])
def test_cleanup_attempts_both_children_and_preserves_errors(
    rig: Rig,
    during: str,
) -> None:
    """Attempt every cleanup and retain operation errors ahead of cleanup errors.

    Both close hooks fail. The start variant also fails capture startup; the
    stop variant fails both stop hooks. All paths must close each child once,
    retain every failure as context, and release the logical device claims.
    """
    frame = _frame(rig)
    frame.prepare()
    rig.failures["ao.close"] = RuntimeError("AO cleanup failed")
    rig.failures["capture.close"] = RuntimeError("capture cleanup failed")

    if during == "start":
        primary = RuntimeError("capture start failed")
        rig.failures["capture.start"] = primary
        operation = frame.start
    else:
        frame.start()
        if during == "stop":
            primary = RuntimeError("AO stop failed")
            rig.failures["ao.stop"] = primary
            rig.failures["capture.stop"] = RuntimeError("capture stop failed")
            operation = frame.stop
        else:
            primary = rig.failures["ao.close"]
            operation = frame.close

    with pytest.raises(RuntimeError) as caught:
        operation()

    assert caught.value is primary
    details = str(primary) + " ".join(getattr(primary, "__notes__", ()))
    assert "AO cleanup failed" in details
    assert "capture cleanup failed" in details
    if during == "stop":
        assert "capture stop failed" in details

    assert rig.events.count("ao.close") == rig.events.count("capture.close") == 1
    assert frame.state is CaptureState.CLOSED
    frame.close()

    rig.failures.clear()
    _frame(rig).prepare()


def test_stop_retains_reservations_until_close(rig: Rig) -> None:
    """Stop both children once and retain device reservations until closure."""
    frame = _frame(rig)
    frame.prepare()
    frame.start()
    frame.stop()
    frame.stop()

    assert frame.state is CaptureState.STOPPED
    assert rig.events[-2:] == ["ao.stop", "capture.stop"]
    with pytest.raises(RuntimeError, match="reserved"):
        rig.axis.close()
    with pytest.raises(RuntimeError, match="reserved"):
        rig.digitizer.close()

    frame.close()
    _frame(rig).prepare()


@pytest.mark.parametrize("closed", [False, True])
def test_rejected_non_new_capture_stays_caller_owned(
    rig: Rig,
    closed: bool,
) -> None:
    """Leave a prepared or closed caller session unchanged on constructor failure."""
    capture = _capture(rig)
    if closed:
        capture.close()
    else:
        capture.prepare()

    expected = capture.state
    with pytest.raises(ValueError, match="must be NEW"):
        NITriggeredFrame(
            capture,
            {},
            start_trigger=DigitalStartTrigger(source="/Dev1/PFI0"),
        )

    assert capture.state is expected
    assert rig.outputs == []


def test_invalid_waveforms_leave_new_capture_caller_owned(rig: Rig) -> None:
    """Keep ownership with the caller when NI waveform validation fails."""
    capture = _capture(rig)
    with pytest.raises(ValueError, match="At least one"):
        NITriggeredFrame(
            capture,
            {},
            start_trigger=DigitalStartTrigger(source="/Dev1/PFI0"),
        )

    assert capture.state is CaptureState.NEW
    assert rig.events == []


@pytest.mark.parametrize(
    ("invalid", "error_type", "message"),
    [
        ("capture", TypeError, "DigitizerCaptureSession"),
        ("unlimited", ValueError, "finite capture"),
        ("trigger", TypeError, "DigitalStartTrigger"),
    ],
)
def test_constructor_rejects_invalid_contract_arguments(
    rig: Rig,
    invalid: str,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject wrong interfaces and unlimited capture before creating NI output.

    Use a session-spec mock only for unlimited delivery, which the real core
    simulator deliberately does not support. No mock lifecycle is exercised.
    """
    capture = _capture(rig)
    if invalid == "capture":
        capture = cast(DigitizerCaptureSession, object())
    elif invalid == "unlimited":
        capture = cast(
            DigitizerCaptureSession,
            Mock(
                spec=DigitizerCaptureSession,
                state=CaptureState.NEW,
                batch_count=None,
            ),
        )

    trigger = (
        cast(DigitalStartTrigger, object())
        if invalid == "trigger"
        else DigitalStartTrigger(source="/Dev1/PFI0")
    )
    with pytest.raises(error_type, match=message):
        NITriggeredFrame(capture, {}, start_trigger=trigger)

    assert rig.outputs == []
    assert rig.events == []
