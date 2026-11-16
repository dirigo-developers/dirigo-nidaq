"""Test finite frame-clock output without native NI drivers or equipment."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import cast
from unittest.mock import Mock, call

import nidaqmx
import numpy as np
import pytest
from dirigo import units
from dirigo.hardware.capture import CaptureSettings
from dirigo.hardware.frame_plan import RecordFramePlan
from dirigo.hardware.output import OutputState
from dirigo.hardware.timing import ExternalSampleClock, SignalEdge
from nidaqmx import constants, errors

from dirigo_nidaq.frame_clock import NIFrameClockSession


@dataclass(frozen=True)
class FakeNI:
    """Expose the patched constructor and its recording task."""

    factory: Mock
    task: Mock


@pytest.fixture(autouse=True)
def ni(monkeypatch: pytest.MonkeyPatch) -> FakeNI:
    """Patch the public Task constructor and acknowledge static/buffered writes."""
    task = Mock()

    def write(data: object, *, auto_start: bool) -> int:
        """Acknowledge a scalar level or every sample in an output array."""
        return data.size if isinstance(data, np.ndarray) else 1

    task.write.side_effect = write
    factory = Mock(return_value=task)
    monkeypatch.setattr(nidaqmx, "Task", factory)
    return FakeNI(factory, task)


@pytest.fixture
def plan() -> RecordFramePlan:
    """Put two three-record frames across three two-record transfer batches."""
    return RecordFramePlan(
        capture_settings=CaptureSettings(
            sample_rate=units.SampleRate("100 MS/s"),
            record_length=4096,
            records_per_batch=2,
            batch_count=3,
        ),
        records_per_frame=3,
    )


@pytest.fixture
def clock() -> ExternalSampleClock:
    """Describe an acquired-record clock independently of the ADC sample rate."""
    return ExternalSampleClock(
        source="/Dev1/PFI0",
        max_rate=units.SampleRate("8 kS/s"),
    )


@pytest.fixture
def session(
    plan: RecordFramePlan,
    clock: ExternalSampleClock,
) -> Iterator[NIFrameClockSession]:
    """Close each borrowing session even if an assertion or driver operation fails."""
    output = NIFrameClockSession(
        plan,
        output_channel="Dev1/port0/line0",
        record_clock=clock,
    )
    try:
        yield output
    finally:
        output.close()


def test_unused_session_is_passive_and_single_use(
    session: NIFrameClockSession,
    ni: FakeNI,
) -> None:
    """Construct and close without a task, then reject reusing the closed session."""
    assert session.state is OutputState.NEW
    session.close()
    session.close()
    with pytest.raises(RuntimeError, match="Session must be"):
        session.prepare()

    assert session.state is OutputState.CLOSED
    ni.factory.assert_not_called()


@pytest.mark.parametrize(
    ("records_per_frame", "expected"),
    [
        (2, (True, False, True, False, True, False)),
        (3, (True, False, False, True, False, False)),
        (6, (True, False, False, False, False, False)),
    ],
)
@pytest.mark.parametrize("edge", [SignalEdge.RISING, SignalEdge.FALLING])
def test_prepare_establishes_low_before_preloading_frame_zero(
    plan: RecordFramePlan,
    clock: ExternalSampleClock,
    ni: FakeNI,
    records_per_frame: int,
    expected: tuple[bool, ...],
    edge: SignalEdge,
) -> None:
    """Drive low before loading a pattern whose first active sample is record zero.

    Golden patterns cover one frame, adjacent two-record frames, and a frame
    boundary inside a transfer batch. Input clock polarity must affect NI edge
    selection while output frame pulses remain high at the same record indices.
    The timing rate is the record-clock bound, not the 100 MS/s ADC rate.
    """
    output = NIFrameClockSession(
        replace(plan, records_per_frame=records_per_frame),
        output_channel="/Dev1/port0/line0",
        record_clock=replace(clock, active_edge=edge),
    )
    try:
        output.prepare()

        assert output.state is OutputState.PREPARED
        ni.factory.assert_called_once_with()
        ni.task.do_channels.add_do_chan.assert_called_once_with(
            lines="/Dev1/port0/line0",
            line_grouping=constants.LineGrouping.CHAN_PER_LINE,
        )
        assert [entry[0] for entry in ni.task.mock_calls] == [
            "do_channels.add_do_chan",
            "write",
            "stop",
            "control",
            "timing.cfg_samp_clk_timing",
            "write",
            "control",
        ]
        baseline, buffered = ni.task.write.call_args_list
        assert baseline == call(False, auto_start=True)
        assert buffered.kwargs == {"auto_start": False}
        samples = buffered.args[0]
        np.testing.assert_array_equal(samples, expected)
        assert samples.dtype == np.bool_
        assert samples.shape == (6,)
        assert samples.flags.c_contiguous
        ni.task.timing.cfg_samp_clk_timing.assert_called_once_with(
            rate=8000.0,
            source="/Dev1/PFI0",
            active_edge=(
                constants.Edge.RISING
                if edge is SignalEdge.RISING
                else constants.Edge.FALLING
            ),
            sample_mode=constants.AcquisitionType.FINITE,
            samps_per_chan=6,
        )
        assert ni.task.control.call_args_list == [
            call(constants.TaskMode.TASK_UNRESERVE),
            call(constants.TaskMode.TASK_COMMIT),
        ]
        ni.task.start.assert_not_called()
    finally:
        output.close()

    # Only the on-demand setup phase needed a stop; buffered output never ran.
    ni.task.stop.assert_called_once_with()
    ni.task.close.assert_called_once_with()


def test_timeout_is_retryable_and_completion_retains_task(
    session: NIFrameClockSession,
    ni: FakeNI,
) -> None:
    """Retain an armed task across timeout and observe finite completion once."""
    session.prepare()
    session.start()
    ni.task.wait_until_done.side_effect = [errors.DaqError("waiting", -200560), None]

    assert session.wait(timeout=0) is False
    assert session.state is OutputState.STARTED
    ni.task.close.assert_not_called()
    assert session.wait() is True
    assert session.state is OutputState.COMPLETED
    ni.task.wait_until_done.assert_called_with(timeout=constants.WAIT_INFINITELY)
    assert session.wait(timeout=0) is True
    assert ni.task.wait_until_done.call_count == 2
    session.stop()
    session.close()
    session.close()

    ni.task.start.assert_called_once_with()
    ni.task.stop.assert_called_once_with()
    ni.task.close.assert_called_once_with()
    assert ni.task.write.call_count == 2


def test_stop_retains_task_and_adds_no_static_write(
    session: NIFrameClockSession,
    ni: FakeNI,
) -> None:
    """Stop buffered output once, retaining resources without commanding a level."""
    session.prepare()
    session.start()
    session.stop()
    session.stop()

    assert session.state is OutputState.STOPPED
    assert session.wait(timeout=0) is False
    assert ni.task.stop.call_count == 2
    assert ni.task.write.call_count == 2
    ni.task.close.assert_not_called()
    ni.task.wait_until_done.assert_not_called()
    session.close()
    ni.task.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("stage", "cleanup_fails"),
    [
        (stage, False)
        for stage in (
            "create",
            "channel",
            "idle_write",
            "idle_stop",
            "unreserve",
            "timing",
            "buffer_write",
            "commit",
            "start",
            "wait",
            "stop",
            "close",
        )
    ]
    + [(stage, True) for stage in ("idle_write", "start", "wait", "stop")],
)
def test_native_failure_closes_partial_resources(
    session: NIFrameClockSession,
    ni: FakeNI,
    stage: str,
    cleanup_fails: bool,
) -> None:
    """Clean up every stage, including a failed baseline auto-start or stop.

    Preparation briefly uses on-demand output before buffered configuration.
    Failures during that phase must still stop possible activity and close
    the task. Later failures exercise cleanup of armed buffered output. A
    failed close must remain secondary to the original operation error.
    """
    if stage in {"start", "wait", "stop", "close"}:
        session.prepare()
    if stage in {"wait", "stop", "close"}:
        session.start()
    failure = RuntimeError(f"{stage} failed")
    targets = {
        "create": ni.factory,
        "channel": ni.task.do_channels.add_do_chan,
        "idle_write": ni.task.write,
        "idle_stop": ni.task.stop,
        "unreserve": ni.task.control,
        "timing": ni.task.timing.cfg_samp_clk_timing,
        "start": ni.task.start,
        "wait": ni.task.wait_until_done,
        "stop": ni.task.stop,
        "close": ni.task.close,
    }
    if stage == "buffer_write":
        ni.task.write.side_effect = [1, failure]
    elif stage == "commit":
        ni.task.control.side_effect = [None, failure]
    else:
        targets[stage].side_effect = failure
    if cleanup_fails:
        ni.task.close.side_effect = RuntimeError("cleanup failed")
    operation = getattr(
        session,
        stage if stage in {"start", "wait", "stop", "close"} else "prepare",
    )

    with pytest.raises(RuntimeError, match=f"{stage} failed") as caught:
        operation()

    assert caught.value is failure
    assert session.state is OutputState.CLOSED
    if cleanup_fails:
        assert any("cleanup failed" in note for note in failure.__notes__)
    if stage == "create":
        ni.task.close.assert_not_called()
    else:
        ni.task.close.assert_called_once_with()
    if stage in {"create", "channel"}:
        expected_stops = 0
    elif stage in {"start", "wait", "stop", "close"}:
        expected_stops = 2
    else:
        expected_stops = 1
    assert ni.task.stop.call_count == expected_stops
    session.close()


@pytest.mark.parametrize(
    ("counts", "message"),
    [((0,), "inactive level"), ((1, 5), "every frame-clock sample")],
)
def test_incomplete_write_prevents_prepared_state(
    session: NIFrameClockSession,
    ni: FakeNI,
    counts: tuple[int, ...],
    message: str,
) -> None:
    """Require complete acknowledgement of both baseline and pattern writes."""
    ni.task.write.side_effect = list(counts)
    with pytest.raises(RuntimeError, match=message):
        session.prepare()

    assert session.state is OutputState.CLOSED
    ni.task.start.assert_not_called()
    ni.task.close.assert_called_once_with()
    assert call(constants.TaskMode.TASK_COMMIT) not in ni.task.control.call_args_list


@pytest.mark.parametrize(("code", "timeout"), [(-200452, 0.01), (-200560, None)])
def test_other_ni_wait_errors_are_fatal(
    session: NIFrameClockSession,
    ni: FakeNI,
    code: int,
    timeout: float | None,
) -> None:
    """Preserve other NI errors and an unexpected unbounded completion timeout."""
    session.prepare()
    session.start()
    failure = errors.DaqError("native failure", code)
    ni.task.wait_until_done.side_effect = failure

    with pytest.raises(errors.DaqError) as caught:
        session.wait(timeout=timeout)

    assert caught.value is failure
    assert session.state is OutputState.CLOSED
    ni.task.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("timeout", "error_type"),
    [
        (True, TypeError),
        ("0", TypeError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_invalid_timeout_retains_usability(
    session: NIFrameClockSession,
    ni: FakeNI,
    timeout: object,
    error_type: type[Exception],
) -> None:
    """Reject invalid budgets before waiting or altering the active session."""
    session.prepare()
    session.start()
    with pytest.raises(error_type, match="timeout"):
        session.wait(timeout=cast(float, timeout))

    assert session.state is OutputState.STARTED
    ni.task.wait_until_done.assert_not_called()
    ni.task.close.assert_not_called()


def test_lifecycle_rejects_repeated_or_premature_calls(
    session: NIFrameClockSession,
) -> None:
    """Reject start/wait before preparation and repeated preparation or start."""
    for operation in (session.start, session.wait):
        with pytest.raises(RuntimeError, match="Session must be"):
            operation()
    session.prepare()
    with pytest.raises(RuntimeError, match="Session must be"):
        session.prepare()
    session.start()
    with pytest.raises(RuntimeError, match="Session must be"):
        session.start()
    session.close()
    session.stop()


def test_plan_can_be_reused_by_a_fresh_session(
    session: NIFrameClockSession,
    plan: RecordFramePlan,
    clock: ExternalSampleClock,
    ni: FakeNI,
) -> None:
    """Reuse immutable planning data while allocating a fresh native task per run."""
    session.prepare()
    session.close()
    next_task = Mock()
    next_task.write.side_effect = [1, 6]
    ni.factory.side_effect = [next_task]
    retry = NIFrameClockSession(
        plan, output_channel="Dev1/port0/line0", record_clock=clock
    )
    try:
        retry.prepare()
        assert retry.state is OutputState.PREPARED
        assert ni.factory.call_count == 2
    finally:
        retry.close()

    ni.task.close.assert_called_once_with()
    next_task.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        ("plan", object(), TypeError, "RecordFramePlan"),
        ("record_clock", object(), TypeError, "ExternalSampleClock"),
        ("output_channel", None, TypeError, "output_channel"),
        ("output_channel", "Dev1/port1/line0", ValueError, "Port 0"),
        ("output_channel", "Dev1/port0", ValueError, "Port 0"),
        ("output_channel", "Dev1/port0/line0:1", ValueError, "Port 0"),
    ],
)
def test_constructor_rejects_invalid_arguments(
    plan: RecordFramePlan,
    clock: ExternalSampleClock,
    ni: FakeNI,
    field: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject wrong models and unsupported channel syntax before allocating a task."""
    with pytest.raises(error_type, match=message):
        NIFrameClockSession(
            cast(RecordFramePlan, value) if field == "plan" else plan,
            output_channel=(
                cast(str, value) if field == "output_channel" else "Dev1/port0/line0"
            ),
            record_clock=(
                cast(ExternalSampleClock, value) if field == "record_clock" else clock
            ),
        )

    ni.factory.assert_not_called()


def test_single_record_frames_are_rejected_by_this_backend(
    plan: RecordFramePlan,
    clock: ExternalSampleClock,
    ni: FakeNI,
) -> None:
    """Keep the core grouping valid while rejecting pulses lacking a low interval."""
    with pytest.raises(ValueError, match="at least two"):
        NIFrameClockSession(
            replace(plan, records_per_frame=1),
            output_channel="Dev1/port0/line0",
            record_clock=clock,
        )

    ni.factory.assert_not_called()
