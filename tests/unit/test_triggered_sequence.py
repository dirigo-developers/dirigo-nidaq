"""Test frame-sequence coordination with real child lifecycles and fake NI calls."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from functools import partial
from typing import cast
from unittest.mock import Mock, PropertyMock

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
from dirigo.hardware.capture_session import CaptureState, DigitizerCaptureSession
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.frame_plan import RecordFramePlan
from dirigo.hardware.simulated_digitizer import (
    SimulatedDigitizer,
    SimulatedDigitizerConfig,
)
from dirigo.hardware.timing import DigitalStartTrigger, ExternalSampleClock, SignalEdge
from dirigo.hardware.waveform import AngularWaveform
from nidaqmx import constants, errors, stream_writers

from dirigo_nidaq import retriggered_output as output_module
from dirigo_nidaq import triggered_sequence as sequence_module
from dirigo_nidaq.scanner import NIGalvoScanner, NIGalvoScannerConfig
from dirigo_nidaq.triggered_sequence import NITriggeredFrameSequence


@dataclass(frozen=True, slots=True, kw_only=True)
class DriverSettings(CaptureSettings):
    """Stand in for a backend-specific settings subclass without another plugin."""

    backend_tag: str = "preserve_me"


@dataclass
class FakeTime:
    """Advance deadlines deterministically without sleeping in real time."""

    now: float = 0.0
    slept: float = 0.0

    def monotonic(self) -> float:
        """Return the shared simulated clock."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Account for AO polling delays against the same deadline clock."""
        self.now += seconds
        self.slept += seconds


@dataclass
class Rig:
    """Keep real devices and bookkeeping while replacing only native NI calls."""

    digitizer: SimulatedDigitizer
    axis: NIGalvoScanner
    plan: RecordFramePlan
    monkeypatch: pytest.MonkeyPatch
    timer: FakeTime
    factory: Mock
    writer: Mock
    events: list[str] = field(default_factory=list)
    failures: dict[str, Exception] = field(default_factory=dict)
    requests: list[CaptureSettings] = field(default_factory=list)
    captures: list[DigitizerCaptureSession] = field(default_factory=list)
    sequences: list[NITriggeredFrameSequence] = field(default_factory=list)
    pending: list[Mock] = field(default_factory=list)
    created: list[Mock] = field(default_factory=list)
    clock_waits: list[float] = field(default_factory=list)
    clock_complete: bool = True
    generated: int = 40
    capture_cost: float = 0.0
    clock_cost: float = 0.0


def _step(rig: Rig, event: str) -> None:
    """Record an operation and inject its selected failure, if any."""
    rig.events.append(event)
    if event in rig.failures:
        raise rig.failures[event]


def _record(
    rig: Rig, event: str, original: Callable[..., object]
) -> Callable[..., object]:
    """Record a capture hook while retaining its real implementation."""

    def invoke(*args: object, **kwargs: object) -> object:
        _step(rig, event)
        return original(*args, **kwargs)

    return invoke


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    """Run all three child lifecycles using a simulator and public NI mocks.

    Capture still performs real validation, reservations, and synthetic data
    generation. Both NI sessions still configure, preload, start, wait, stop,
    and close through their implementations. Mock tasks provide observations;
    the simulator supplies no electrical record clock or frame triggers.
    """
    digitizer = SimulatedDigitizer(SimulatedDigitizerConfig(channel_count=1))
    axis = NIGalvoScanner(
        NIGalvoScannerConfig(
            angle_control_channel="Dev1/ao0",
            angle_range=units.AngleRange("-10 deg", "10 deg"),
            command_voltage_range=units.VoltageRange("-5 V", "5 V"),
        )
    )
    timer = FakeTime()
    factory = Mock()
    writer = Mock()
    writer.write_many_sample.return_value = 20
    result = Rig(
        digitizer=digitizer,
        axis=axis,
        plan=RecordFramePlan(
            capture_settings=DriverSettings(
                sample_rate=units.SampleRate("100 MS/s"),
                record_length=4,
                records_per_batch=2,
                batch_count=3,
            ),
            records_per_frame=3,
        ),
        monkeypatch=monkeypatch,
        timer=timer,
        factory=factory,
        writer=writer,
    )

    def make_task() -> Mock:
        """Hand each prepared NI session its queued native task."""
        task = result.pending.pop(0)
        result.created.append(task)
        return task

    original_create = digitizer._create_capture

    def create(settings: CaptureSettings) -> DigitizerCaptureSession:
        """Record the exact request and instrument real capture operations."""
        result.requests.append(settings)
        capture = original_create(settings)
        result.captures.append(capture)
        for hook, event in (
            ("_prepare_capture", "capture.prepare"),
            ("_start_capture", "capture.start"),
            ("_read_into", "capture.read"),
            ("_stop_capture", "capture.stop"),
            ("_close_capture", "capture.close"),
        ):
            monkeypatch.setattr(
                capture, hook, _record(result, event, getattr(capture, hook))
            )
        original_read = capture.read_into

        def read(
            destination: SampleArray, *, timeout: float | None = None
        ) -> CaptureInfo | None:
            """Include capture-call overhead even when validating repeated EOF."""
            try:
                return original_read(destination, timeout=timeout)
            finally:
                timer.now += result.capture_cost

        monkeypatch.setattr(capture, "read_into", read)
        return capture

    factory.side_effect = make_task
    monkeypatch.setattr(nidaqmx, "Task", factory)
    monkeypatch.setattr(
        stream_writers, "AnalogMultiChannelWriter", Mock(return_value=writer)
    )
    monkeypatch.setattr(digitizer, "_create_capture", create)
    monkeypatch.setattr(sequence_module, "monotonic", timer.monotonic)
    monkeypatch.setattr(output_module, "time", timer)
    try:
        yield result
    finally:
        result.failures.clear()
        for sequence in reversed(result.sequences):
            sequence.close()
        axis.close()
        digitizer.close()


def _sequence(rig: Rig) -> NITriggeredFrameSequence:
    """Create a fresh run and queue its clock/AO tasks for later preparation."""
    clock = Mock()
    output = Mock()
    clock.do_channels.add_do_chan.side_effect = lambda **_: _step(rig, "clock.prepare")
    output.ao_channels.add_ao_voltage_chan.side_effect = lambda **_: _step(
        rig, "ao.prepare"
    )

    def write_clock(data: object, *, auto_start: bool) -> int:
        """Acknowledge the initial level or every buffered record-clock sample."""
        return data.size if isinstance(data, np.ndarray) else 1

    clock.write.side_effect = write_clock
    clock.start.side_effect = partial(_step, rig, "clock.start")
    output.start.side_effect = partial(_step, rig, "ao.start")
    clock.stop.side_effect = lambda: _step(
        rig, "clock.stop" if clock.start.called else "clock.idle_stop"
    )
    output.stop.side_effect = partial(_step, rig, "ao.stop")
    clock.close.side_effect = partial(_step, rig, "clock.close")
    output.close.side_effect = partial(_step, rig, "ao.close")

    def wait_clock(*, timeout: float) -> None:
        """Report bounded clock timeout or completion after simulated elapsed time."""
        _step(rig, "clock.wait")
        rig.clock_waits.append(timeout)
        rig.timer.now += rig.clock_cost
        if not rig.clock_complete:
            raise errors.DaqError("record edges still missing", -200560)

    def poll_output() -> bool:
        """Surface native AO errors while leaving retriggering armed."""
        _step(rig, "ao.wait")
        return False

    clock.wait_until_done.side_effect = wait_clock
    output.is_task_done.side_effect = poll_output
    rig.monkeypatch.setattr(
        type(output.out_stream),
        "total_samp_per_chan_generated",
        PropertyMock(side_effect=lambda: rig.generated),
        raising=False,
    )
    rig.pending[:] = [clock, output]
    sequence = NITriggeredFrameSequence(
        rig.digitizer,
        rig.plan,
        {
            rig.axis: AngularWaveform(
                [0.0] * 20, sample_rate=units.SampleRate("20 kS/s")
            )
        },
        record_clock=ExternalSampleClock(
            source="/Dev1/PFI0", max_rate=units.SampleRate("8 kS/s")
        ),
        frame_clock_channel="Dev1/port0/line0",
        frame_trigger=DigitalStartTrigger(source="/Dev1/PFI1"),
    )
    rig.sequences.append(sequence)
    return sequence


def _drain(sequence: NITriggeredFrameSequence) -> SampleArray:
    """Deliver all three batches while leaving final NI completion unobserved."""
    samples = np.empty(sequence.layout.shape, dtype=sequence.layout.dtype)
    for _ in range(3):
        assert sequence.read_into(samples, timeout=0) is not None
    return samples


def test_construction_is_passive_and_uses_the_plan_request(rig: Rig) -> None:
    """Use the exact immutable settings object without native tasks or reservations."""
    sequence = _sequence(rig)
    assert rig.requests == [rig.plan.capture_settings]
    assert rig.requests[0] is rig.plan.capture_settings
    assert isinstance(rig.requests[0], DriverSettings)
    assert rig.requests[0].backend_tag == "preserve_me"
    assert sequence.state is CaptureState.NEW
    assert sequence.frame_count == 2
    assert sequence.batches_acquired == 0
    assert rig.events == []
    sequence.close()
    sequence.close()
    assert rig.captures[0].state is CaptureState.CLOSED
    rig.factory.assert_not_called()
    _sequence(rig).prepare()


def test_prepare_start_and_close_follow_dependency_order(rig: Rig) -> None:
    """Establish a low frame line before arming AO, then enable capture last.

    The ADC and AO grids deliberately differ. Six records form two frames
    across three transfer batches, while AO preloads only one twenty-sample
    frame. Normal cleanup disarms AO before stopping the clock or capture.
    """
    sequence = _sequence(rig)
    sequence.prepare()
    assert rig.events == [
        "capture.prepare",
        "clock.prepare",
        "clock.idle_stop",
        "ao.prepare",
    ]
    assert sequence.layout.shape == (2, 4, 1)
    clock, output = rig.created
    np.testing.assert_array_equal(
        clock.write.call_args_list[1].args[0], [True, False, False, True, False, False]
    )
    assert rig.writer.write_many_sample.call_args.args[0].shape == (1, 20)
    sequence.start()
    assert rig.events[-3:] == ["ao.start", "clock.start", "capture.start"]
    sequence.close()
    assert rig.events[-6:] == [
        "ao.stop",
        "ao.close",
        "clock.stop",
        "clock.close",
        "capture.stop",
        "capture.close",
    ]
    clock.close.assert_called_once_with()
    output.close.assert_called_once_with()
    assert rig.axis.angle is None
    assert sequence.layout.shape == (2, 4, 1)
    _sequence(rig).prepare()


def test_eof_requires_both_outputs_without_redelivering_batches(rig: Rig) -> None:
    """Retry clock and AO completion independently after capture is exhausted.

    Frame boundaries cross transfer batches. First the frame clock times out,
    then AO remains incomplete, then both complete. None of these EOF retries
    may overwrite the last batch, restart children, or deliver duplicate data.
    """
    sequence = _sequence(rig)
    sequence.prepare()
    sequence.start()
    samples = np.empty(sequence.layout.shape, dtype=sequence.layout.dtype)
    delivered = []
    for _ in range(3):
        assert sequence.read_into(samples, timeout=0) is not None
        delivered.append(samples.copy())
    expected = (np.arange(24, dtype=np.int32) - 32768).astype(np.int16).reshape(6, 4, 1)
    np.testing.assert_array_equal(np.concatenate(delivered), expected)
    last = samples.copy()
    rig.clock_complete = False
    rig.generated = 0
    with pytest.raises(TimeoutError, match="frame clock"):
        sequence.read_into(samples, timeout=0)
    assert "ao.wait" not in rig.events
    rig.clock_complete = True
    with pytest.raises(TimeoutError, match="AO output"):
        sequence.read_into(samples, timeout=0)
    assert sequence.state is CaptureState.STARTED
    rig.generated = 40
    assert sequence.read_into(samples, timeout=0) is None
    assert sequence.state is CaptureState.COMPLETED
    events = rig.events.copy()
    assert sequence.read_into(samples, timeout=0) is None
    assert rig.events == events
    assert rig.events.count("capture.read") == 3
    assert len(rig.clock_waits) == 2
    assert sequence.batches_acquired == 3
    np.testing.assert_array_equal(samples, last)
    with pytest.raises(ValueError, match="shape"):
        sequence.read_into(np.empty((1,), dtype=sequence.layout.dtype), timeout=0)
    with pytest.raises(ValueError, match="timeout"):
        sequence.read_into(samples, timeout=float("nan"))
    assert sequence.state is CaptureState.COMPLETED


def test_eof_waits_share_one_deadline(rig: Rig) -> None:
    """Subtract capture overhead and clock waiting before giving AO its budget."""
    sequence = _sequence(rig)
    sequence.prepare()
    sequence.start()
    samples = _drain(sequence)
    rig.capture_cost = 0.125
    rig.clock_cost = 0.25
    rig.generated = 0
    with pytest.raises(TimeoutError, match="AO output"):
        sequence.read_into(samples, timeout=0.5)
    assert rig.clock_waits == [0.375]
    assert rig.timer.now == 0.5
    assert abs(rig.timer.slept - 0.125) < 1e-12
    rig.capture_cost = rig.clock_cost = 0.0
    rig.generated = 40
    assert sequence.read_into(samples, timeout=0) is None


@pytest.mark.parametrize("timeout", [None, 0.0625])
def test_unbounded_budget_or_exhausted_budget_can_observe_completion(
    rig: Rig, timeout: float | None
) -> None:
    """Forward unbounded waiting or poll both outputs after capture uses the budget."""
    sequence = _sequence(rig)
    sequence.prepare()
    sequence.start()
    samples = _drain(sequence)
    rig.capture_cost = 0.125
    assert sequence.read_into(samples, timeout=timeout) is None
    assert rig.clock_waits == [constants.WAIT_INFINITELY if timeout is None else 0.0]
    assert rig.timer.slept == 0


def test_capture_timeout_and_argument_errors_preserve_usability(rig: Rig) -> None:
    """Keep all children usable after a capture timeout or invalid caller storage."""
    sequence = _sequence(rig)
    sequence.prepare()
    sequence.start()
    samples = np.empty(sequence.layout.shape, dtype=sequence.layout.dtype)
    rig.failures["capture.read"] = TimeoutError("capture waiting")
    with pytest.raises(TimeoutError, match="capture waiting"):
        sequence.read_into(samples, timeout=0)
    with pytest.raises(ValueError, match="shape"):
        sequence.read_into(np.empty((1,), dtype=sequence.layout.dtype), timeout=0)
    assert sequence.state is CaptureState.STARTED
    assert sequence.batches_acquired == 0
    assert not any(event.endswith("close") for event in rig.events)
    rig.failures.clear()
    assert sequence.read_into(samples, timeout=0) is not None


@pytest.mark.parametrize(
    "stage",
    [
        "capture.prepare",
        "clock.prepare",
        "ao.prepare",
        "ao.start",
        "clock.start",
        "capture.start",
        "capture.read",
        "clock.wait",
        "ao.wait",
    ],
)
def test_fatal_child_failure_closes_every_session(rig: Rig, stage: str) -> None:
    """Roll back partial preparation, startup, read, or completion failures.

    Child implementations may close themselves before propagating an error.
    The coordinator must close their companions without replacing that error
    or double-closing native tasks. A fresh run must reuse both devices.
    """
    sequence = _sequence(rig)
    if not stage.endswith("prepare"):
        sequence.prepare()
    if stage in {"capture.read", "clock.wait", "ao.wait"}:
        sequence.start()
        samples = (
            _drain(sequence)
            if stage.endswith("wait")
            else np.empty(sequence.layout.shape, dtype=sequence.layout.dtype)
        )
        operation: Callable[[], object] = partial(
            sequence.read_into, samples, timeout=0
        )
    else:
        operation = sequence.prepare if stage.endswith("prepare") else sequence.start
    failure = RuntimeError(f"{stage} failed")
    rig.failures[stage] = failure
    with pytest.raises(RuntimeError) as caught:
        operation()
    assert caught.value is failure
    assert sequence.state is CaptureState.CLOSED
    assert rig.captures[0].state is CaptureState.CLOSED
    for task in rig.created:
        task.close.assert_called_once_with()
    rig.failures.clear()
    _sequence(rig).prepare()


@pytest.mark.parametrize("during", ["start", "stop", "close"])
def test_cleanup_attempts_every_child_and_preserves_all_errors(
    rig: Rig, during: str
) -> None:
    """Retain primary operation errors while attempting every failing cleanup.

    All close operations fail. Startup can also fail, or all three stops can
    fail. The original error must survive with the remaining failures noted,
    and logical reservations must be released for a subsequent run.
    """
    sequence = _sequence(rig)
    sequence.prepare()
    for child in ("ao", "clock", "capture"):
        rig.failures[f"{child}.close"] = RuntimeError(f"{child} cleanup failed")
    if during == "start":
        primary = RuntimeError("capture start failed")
        rig.failures["capture.start"] = primary
        operation = sequence.start
    else:
        sequence.start()
        if during == "stop":
            for child in ("ao", "clock", "capture"):
                rig.failures[f"{child}.stop"] = RuntimeError(f"{child} stop failed")
            primary = rig.failures["ao.stop"]
            operation = sequence.stop
        else:
            primary = rig.failures["ao.close"]
            operation = sequence.close
    with pytest.raises(RuntimeError) as caught:
        operation()
    assert caught.value is primary
    details = str(primary) + " ".join(getattr(primary, "__notes__", ()))
    for child in ("ao", "clock", "capture"):
        assert f"{child} cleanup failed" in details
        assert rig.events.count(f"{child}.close") == 1
    assert sequence.state is CaptureState.CLOSED
    sequence.close()
    rig.failures.clear()
    _sequence(rig).prepare()


def test_stop_retains_reservations_until_close(rig: Rig) -> None:
    """Disarm in dependency order once while retaining both device reservations."""
    sequence = _sequence(rig)
    sequence.prepare()
    sequence.start()
    sequence.stop()
    sequence.stop()
    assert sequence.state is CaptureState.STOPPED
    assert rig.events[-3:] == ["ao.stop", "clock.stop", "capture.stop"]
    with pytest.raises(RuntimeError, match="reserved"):
        rig.axis.close()
    with pytest.raises(RuntimeError, match="reserved"):
        rig.digitizer.close()
    assert not any(event.endswith("close") for event in rig.events)
    sequence.close()
    _sequence(rig).prepare()


def test_out_of_order_lifecycle_calls_are_rejected(rig: Rig) -> None:
    """Reject premature calls, repeated prepare/start, and use after closure."""
    sequence = _sequence(rig)
    with pytest.raises(RuntimeError, match="Sequence must be"):
        sequence.start()
    sequence.prepare()
    with pytest.raises(RuntimeError, match="Sequence must be"):
        sequence.prepare()
    sequence.start()
    with pytest.raises(RuntimeError, match="Sequence must be"):
        sequence.start()
    samples = np.empty(sequence.layout.shape, dtype=sequence.layout.dtype)
    sequence.close()
    with pytest.raises(RuntimeError, match="Sequence must be"):
        sequence.read_into(samples, timeout=0)
    sequence.stop()


@pytest.mark.parametrize(
    "invalid",
    ["digitizer", "plan", "trigger", "edge", "clock", "channel", "frames", "waveforms"],
)
def test_constructor_rejects_invalid_inputs_without_hardware(
    rig: Rig, invalid: str
) -> None:
    """Validate coordinator and child arguments before constructing a capture."""
    trigger = DigitalStartTrigger(
        source="/Dev1/PFI1",
        active_edge=SignalEdge.FALLING if invalid == "edge" else SignalEdge.RISING,
    )
    plan = replace(rig.plan, records_per_frame=1) if invalid == "frames" else rig.plan
    expected = (
        TypeError
        if invalid in {"digitizer", "plan", "trigger", "clock"}
        else ValueError
    )
    waveform = AngularWaveform([0.0] * 20, sample_rate=units.SampleRate("20 kS/s"))
    clock = ExternalSampleClock(
        source="/Dev1/PFI0", max_rate=units.SampleRate("8 kS/s")
    )
    with pytest.raises(expected):
        NITriggeredFrameSequence(
            cast(Digitizer, object()) if invalid == "digitizer" else rig.digitizer,
            cast(RecordFramePlan, object()) if invalid == "plan" else plan,
            {} if invalid == "waveforms" else {rig.axis: waveform},
            record_clock=(
                cast(ExternalSampleClock, object()) if invalid == "clock" else clock
            ),
            frame_clock_channel="bad" if invalid == "channel" else "Dev1/port0/line0",
            frame_trigger=(
                cast(DigitalStartTrigger, object()) if invalid == "trigger" else trigger
            ),
        )
    assert rig.requests == []
    rig.factory.assert_not_called()


def test_unexpected_prepared_record_batching_is_rejected(rig: Rig) -> None:
    """Reject an effective record count inconsistent with the digital frame plan."""
    sequence = _sequence(rig)
    rig.monkeypatch.setattr(
        rig.captures[0],
        "_prepare_capture",
        Mock(
            return_value=CaptureLayout(
                shape=(1, 4, 1), channel_indices=(0,), dtype=np.dtype(np.int16)
            )
        ),
    )
    with pytest.raises(ValueError, match="batching"):
        sequence.prepare()
    assert sequence.state is CaptureState.CLOSED
    rig.factory.assert_not_called()
    _sequence(rig).prepare()
