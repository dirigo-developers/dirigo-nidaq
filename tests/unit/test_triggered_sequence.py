"""Test frame-sequence coordination with real child lifecycles and fake NI calls."""

from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from threading import Event, get_ident
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
from dirigo.hardware.hardware import Hardware
from dirigo.hardware.simulated_digitizer import (
    SimulatedDigitizer,
    SimulatedDigitizerConfig,
)
from dirigo.hardware.timing import ExternalSampleClock
from dirigo.hardware.waveform import AngularWaveform
from dirigo.pipeline.exceptions import EndOfStream
from dirigo.pipeline.pipeline import Pipeline
from dirigo.pipeline.product import Product
from dirigo.pipeline.sink import Sink
from dirigo.records import RecordBatch
from dirigo.sources.frame_acquisition import FrameAcquisitionWorker
from nidaqmx import stream_writers

from dirigo_nidaq import retriggered_output as output_module
from dirigo_nidaq import triggered_sequence as sequence_module
from dirigo_nidaq.acquisition import nidaq_triggered
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
    generated: int = 40
    capture_cost: float = 0.0


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
                records_per_batch=4,
                batch_count=3,
            ),
            records_per_frame=6,
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


def _queue_tasks(rig: Rig) -> None:
    """Queue fake NI tasks for preparation by a sequence or worker."""
    clock = Mock()
    output = Mock()

    def configure_counter(**parameters: object) -> Mock:
        """Record counter setup and echo timing for the post-commit checks."""
        _step(rig, "clock.prepare")
        channel = Mock()
        channel.co_pulse_ticks_initial_delay = parameters["initial_delay"]
        channel.co_pulse_high_ticks = parameters["high_ticks"]
        channel.co_pulse_low_ticks = parameters["low_ticks"]
        return channel

    clock.co_channels.add_co_pulse_chan_ticks.side_effect = configure_counter
    output.ao_channels.add_ao_voltage_chan.side_effect = lambda **_: _step(
        rig, "ao.prepare"
    )
    clock.start.side_effect = partial(_step, rig, "clock.start")
    output.start.side_effect = partial(_step, rig, "ao.start")
    clock.stop.side_effect = partial(_step, rig, "clock.stop")
    output.stop.side_effect = partial(_step, rig, "ao.stop")
    clock.close.side_effect = partial(_step, rig, "clock.close")
    output.close.side_effect = partial(_step, rig, "ao.close")

    def poll_output() -> bool:
        """Surface native AO errors while leaving retriggering armed."""
        _step(rig, "ao.wait")
        return False

    clock.wait_until_done.side_effect = AssertionError("Must not await counter")
    output.is_task_done.side_effect = poll_output
    rig.monkeypatch.setattr(
        type(output.out_stream),
        "total_samp_per_chan_generated",
        PropertyMock(side_effect=lambda: rig.generated),
        raising=False,
    )
    rig.pending[:] = [clock, output]


def _sequence(rig: Rig) -> NITriggeredFrameSequence:
    """Create a fresh run and queue its clock/AO tasks for later preparation."""

    _queue_tasks(rig)

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
        frame_counter_channel="Dev1/ctr0",
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
    """Commit a counter and arm AO before enabling acquired-record events.

    The ADC and AO grids deliberately differ. Twelve records form two frames
    across three transfer batches, while AO preloads only one twenty-sample
    frame. AO receives the counter's internal signal without a PFI jumper.
    Normal cleanup disarms AO before stopping the clock or capture.
    """
    sequence = _sequence(rig)
    sequence.prepare()
    assert rig.events == ["capture.prepare", "clock.prepare", "ao.prepare"]
    assert sequence.layout.shape == (4, 4, 1)
    clock, output = rig.created
    counter_arguments = clock.co_channels.add_co_pulse_chan_ticks.call_args.kwargs
    trigger_arguments = (
        output.triggers.start_trigger.cfg_dig_edge_start_trig.call_args.kwargs
    )
    assert counter_arguments["initial_delay"] == 0
    assert trigger_arguments["trigger_source"] == "/Dev1/Ctr0InternalOutput"
    clock.write.assert_not_called()
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
    assert sequence.layout.shape == (4, 4, 1)
    _sequence(rig).prepare()


def test_eof_waits_for_ao_then_stops_clock_without_redelivering_batches(
    rig: Rig,
) -> None:
    """Complete capture and AO before stopping the continuous counter.

    Frame boundaries cross transfer batches. After capture EOF, incomplete AO
    makes the read retryable while leaving the counter armed. Once all AO
    samples are generated, EOF stops the counter without native waiting.
    Retries must not overwrite or redeliver the last batch, restart children,
    or repeat the counter stop.
    """
    sequence = _sequence(rig)
    sequence.prepare()
    sequence.start()
    clock, output = rig.created

    samples = np.empty(sequence.layout.shape, dtype=sequence.layout.dtype)
    delivered: list[SampleArray] = []
    rig.generated = 0

    for _ in range(3):
        assert sequence.read_into(samples, timeout=0) is not None
        delivered.append(samples.copy())

    expected = (
        (np.arange(48, dtype=np.int32) - 32768).astype(np.int16).reshape(12, 4, 1)
    )
    np.testing.assert_array_equal(np.concatenate(delivered), expected)
    last = samples.copy()

    with pytest.raises(TimeoutError, match="AO output"):
        sequence.read_into(samples, timeout=0)

    assert sequence.state is CaptureState.STARTED
    clock.stop.assert_not_called()
    output.stop.assert_not_called()

    rig.generated = 40

    assert sequence.read_into(samples, timeout=0) is None
    assert sequence.state is CaptureState.COMPLETED
    assert rig.events[-2:] == ["ao.stop", "clock.stop"]

    events = rig.events.copy()
    assert sequence.read_into(samples, timeout=0) is None
    assert rig.events == events
    assert rig.events.count("capture.read") == 3
    assert sequence.batches_acquired == 3
    np.testing.assert_array_equal(samples, last)

    with pytest.raises(ValueError, match="shape"):
        sequence.read_into(
            np.empty((1,), dtype=sequence.layout.dtype),
            timeout=0,
        )

    with pytest.raises(ValueError, match="timeout"):
        sequence.read_into(samples, timeout=float("nan"))

    assert sequence.state is CaptureState.COMPLETED

    sequence.close()

    clock.start.assert_called_once_with()
    output.start.assert_called_once_with()
    clock.stop.assert_called_once_with()
    clock.wait_until_done.assert_not_called()
    clock.close.assert_called_once_with()
    output.close.assert_called_once_with()


def test_eof_ao_wait_uses_budget_remaining_after_capture(
    rig: Rig,
) -> None:
    """Subtract capture overhead before giving AO the remaining timeout budget.

    Capture uses 0.125 seconds of the 0.5-second budget. Incomplete AO consumes
    the remaining 0.375 seconds while the counter stays armed. A later
    zero-timeout poll can observe AO completion and stop the counter.
    """
    sequence = _sequence(rig)
    sequence.prepare()
    sequence.start()
    samples = _drain(sequence)

    rig.capture_cost = 0.125
    rig.generated = 0

    with pytest.raises(TimeoutError, match="AO output"):
        sequence.read_into(samples, timeout=0.5)

    assert rig.timer.now == 0.5
    assert abs(rig.timer.slept - 0.375) < 1e-12
    rig.created[0].stop.assert_not_called()

    rig.capture_cost = 0.0
    rig.generated = 40

    assert sequence.read_into(samples, timeout=0) is None


@pytest.mark.parametrize("timeout", [None, 0.0625])
def test_unbounded_budget_or_exhausted_budget_can_observe_completion(
    rig: Rig,
    timeout: float | None,
) -> None:
    """Finish AO and stop the counter with an unbounded or exhausted budget.

    AO has already generated every expected sample. Even if capture consumes
    the bounded budget, a zero-timeout AO poll can observe completion without
    sleeping. Counter shutdown must not invoke native completion waiting.
    """
    sequence = _sequence(rig)
    sequence.prepare()
    sequence.start()
    samples = _drain(sequence)

    rig.capture_cost = 0.125

    assert sequence.read_into(samples, timeout=timeout) is None
    rig.created[0].wait_until_done.assert_not_called()
    rig.created[0].stop.assert_called_once_with()
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
        "clock.stop",
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
    if stage in {"capture.read", "clock.stop", "ao.wait"}:
        sequence.start()
        samples = (
            _drain(sequence)
            if stage != "capture.read"
            else np.empty(
                sequence.layout.shape,
                dtype=sequence.layout.dtype,
            )
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
    if stage == "clock.stop":
        rig.created[0].stop.assert_called_once_with()
        assert "ao.stop" in rig.events
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
    ["digitizer", "plan", "clock", "counter", "frames", "waveforms"],
)
def test_constructor_rejects_invalid_inputs_without_hardware(
    rig: Rig, invalid: str
) -> None:
    """Validate coordinator and child arguments before constructing a capture."""
    plan = replace(rig.plan, records_per_frame=1) if invalid == "frames" else rig.plan
    expected = TypeError if invalid in {"digitizer", "plan", "clock"} else ValueError
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
            frame_counter_channel="bad" if invalid == "counter" else "Dev1/ctr0",
        )
    assert rig.requests == []
    rig.factory.assert_not_called()


def test_unexpected_prepared_record_batching_is_rejected(rig: Rig) -> None:
    """Reject an effective record count inconsistent with the counter frame plan."""
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


class CollectRecordBatches(Sink[RecordBatch]):
    """Retain unpooled products without copying their sample arrays."""

    def __init__(self, upstream: FrameAcquisitionWorker) -> None:
        super().__init__(upstream=upstream)
        self.products: list[Product[RecordBatch]] = []

    def _prepare(self) -> None:
        """Allocate no external resources."""

    def _work(self) -> None:
        """Keep original products until the upstream publisher reaches EOF."""
        while True:
            try:
                with self._receive() as product:
                    self.products.append(product)
            except EndOfStream:
                return

    def _close(self) -> None:
        """Retain collected products for assertions after completion."""


@contextmanager
def _running_acquisition(
    rig: Rig,
) -> Generator[tuple[Pipeline, CollectRecordBatches]]:
    """Run real pipeline/session lifecycles using the existing fake NI tasks.

    Finish worker threads before closing the hardware collection. Device
    closure is idempotent, allowing the existing rig finalizer to run.
    """
    _queue_tasks(rig)
    hardware = Hardware({"digitizer": rig.digitizer, "galvo": rig.axis})
    source = FrameAcquisitionWorker(
        rig.plan,
        sequence="nidaq_triggered",
        factories={"nidaq_triggered": nidaq_triggered},
        sequence_config={
            "digitizer": "digitizer",
            "record_clock_input": "/Dev1/PFI0",
            "max_record_rate": "8 kS/s",
            "frame_counter_channel": "Dev1/ctr0",
        },
        waveforms={
            "galvo": AngularWaveform(
                [0.0] * 20,
                sample_rate=units.SampleRate("20 kS/s"),
            )
        },
        read_timeout=0.01,
    )
    sink = CollectRecordBatches(source)
    pipeline = Pipeline(sink, hardware=hardware)

    rig.factory.assert_not_called()

    try:
        pipeline.start()
        yield pipeline, sink
    finally:
        try:
            if not pipeline.done:
                pipeline.abort()
                pipeline.wait(timeout=2)
        finally:
            hardware.close()


def test_acquisition_pipeline_publishes_owned_positioned_batches(
    rig: Rig,
) -> None:
    """Preserve raw batches across publication, closure, and device reuse.

    The sink keeps original products after receive() exits. Comparing their
    samples after the source finishes detects accidental buffer recycling.
    Four-record batches cross six-record logical frame boundaries:
    first_record describes records, while sequence_index counts batches.
    Native task creation must occur on the acquisition thread.
    """
    caller_thread = get_ident()
    task_threads: list[int] = []
    original_factory = cast(Callable[[], Mock], rig.factory.side_effect)

    def make_task() -> Mock:
        """Observe task creation while retaining the existing fake tasks."""
        task_threads.append(get_ident())
        return original_factory()

    rig.factory.side_effect = make_task

    with _running_acquisition(rig) as (pipeline, sink):
        pipeline.wait(timeout=2)

        assert pipeline.exception is None
        assert task_threads == [caller_thread, caller_thread]

        products = sink.products
        batches = [product.payload for product in products]

        assert [product.metadata.sequence_index for product in products] == [0, 1, 2]
        assert [batch.first_record for batch in batches] == [0, 4, 8]
        assert all(batch.records_per_frame == 6 for batch in batches)
        assert all(batch.layout.shape == (4, 4, 1) for batch in batches)
        assert all(isinstance(batch.info, CaptureInfo) for batch in batches)
        assert all(not batch.samples.flags.writeable for batch in batches)

        for index, batch in enumerate(batches):
            assert batch.samples.flags.owndata
            for earlier in batches[:index]:
                assert not np.shares_memory(
                    batch.samples,
                    earlier.samples,
                )

        expected = (
            (np.arange(48, dtype=np.int32) - 32768).astype(np.int16).reshape(12, 4, 1)
        )
        np.testing.assert_array_equal(
            np.concatenate([batch.samples for batch in batches]),
            expected,
        )

        assert rig.captures[0].state is CaptureState.CLOSED
        assert not rig.digitizer.is_active
        for task in rig.created:
            task.close.assert_called_once_with()
        rig.created[0].wait_until_done.assert_not_called()

        retry = _sequence(rig)
        try:
            retry.prepare()
        finally:
            retry.close()


def test_acquisition_pipeline_cancels_while_waiting_for_records(
    rig: Rig,
) -> None:
    """Abort a waiting read without a hardware error or published data.

    The capture hook signals entry and waits only for its supplied timeout.
    Abort runs through Pipeline and the worker's next polling boundary.
    Cleanup must release both NI tasks and the digitizer reservation.
    """
    waiting = Event()
    wake = Event()
    original_create = rig.digitizer._create_capture

    def create(settings: CaptureSettings) -> DigitizerCaptureSession:
        """Use the real session with a bounded, waiting transfer hook."""
        capture = original_create(settings)

        def wait_for_record(
            destination: SampleArray,
            *,
            timeout: float | None,
        ) -> CaptureInfo:
            """Model a quiet trigger input using a bounded wait."""
            assert timeout is not None
            assert timeout > 0
            waiting.set()
            wake.wait(timeout=timeout)
            raise TimeoutError("No record yet.")

        rig.monkeypatch.setattr(
            capture,
            "_read_into",
            wait_for_record,
        )
        return capture

    rig.monkeypatch.setattr(
        rig.digitizer,
        "_create_capture",
        create,
    )

    with _running_acquisition(rig) as (pipeline, sink):
        try:
            assert waiting.wait(timeout=2)
            pipeline.abort()
            wake.set()
            pipeline.wait(timeout=2)
        finally:
            wake.set()

        assert pipeline.exception is None
        assert sink.products == []
        assert rig.captures[0].state is CaptureState.CLOSED
        assert not rig.digitizer.is_active

        for task in rig.created:
            task.stop.assert_called_once_with()
            task.close.assert_called_once_with()


def test_acquisition_pipeline_propagates_capture_failure(
    rig: Rig,
) -> None:
    """Report the original capture error after cleanup releases hardware."""
    failure = RuntimeError("Capture failed.")
    rig.failures["capture.read"] = failure

    with _running_acquisition(rig) as (pipeline, sink):
        with pytest.raises(RuntimeError) as caught:
            pipeline.wait(timeout=2)

        assert caught.value is failure
        assert sink.products == []
        assert rig.captures[0].state is CaptureState.CLOSED
        assert not rig.digitizer.is_active

        for task in rig.created:
            task.close.assert_called_once_with()
