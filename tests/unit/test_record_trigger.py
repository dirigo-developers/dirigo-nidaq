"""Test finite NI record-trigger output through mocked DAQmx calls."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from unittest.mock import Mock

import nidaqmx
import numpy as np
import pytest
from dirigo import units
from dirigo.hardware.capture import CaptureSettings
from dirigo.hardware.output import OutputState
from dirigo.hardware.timing import DigitalRecordTrigger, SignalEdge
from dirigo.hardware.waveform import AngularWaveform
from nidaqmx import constants, errors

from dirigo_nidaq.record_trigger import NIRecordTriggerOutputSession
from dirigo_nidaq.timing import RecordTriggerPlan


@dataclass(frozen=True)
class FakeNI:
    """Expose the task factory and its recording task."""

    factory: Mock
    task: Mock


@pytest.fixture(autouse=True)
def ni(monkeypatch: pytest.MonkeyPatch) -> FakeNI:
    """Replace task creation so no NI driver or physical card is needed."""
    task = Mock()
    task.write.return_value = 12
    factory = Mock(return_value=task)
    monkeypatch.setattr(nidaqmx, "Task", factory)
    return FakeNI(factory, task)


@pytest.fixture
def plan() -> RecordTriggerPlan:
    """Place two record edges at AO samples two and six in a 12-sample run."""
    return RecordTriggerPlan(
        capture_settings=CaptureSettings(
            sample_rate=units.SampleRate("100 MS/s"),
            record_length=4096,
            records_per_batch=2,
            record_trigger=DigitalRecordTrigger(source="external"),
        ),
        waveform=AngularWaveform(
            [0.0] * 12,
            sample_rate=units.SampleRate("1 kS/s"),
        ),
        interval_samples=4,
        first_trigger_sample=2,
    )


@pytest.fixture
def session(plan: RecordTriggerPlan) -> Iterator[NIRecordTriggerOutputSession]:
    """Provide a fresh session and release its task after each test."""
    output = NIRecordTriggerOutputSession(
        plan,
        output_channel="Dev1/port0/line0",
        sample_clock_source="/Dev1/ao/SampleClock",
        pulse_width_samples=2,
    )
    try:
        yield output
    finally:
        output.close()


def test_construction_and_unused_close_are_passive(
    session: NIRecordTriggerOutputSession,
    ni: FakeNI,
) -> None:
    """Construct and repeatedly close an unused session without making a task."""
    assert session.state is OutputState.NEW
    session.close()
    session.close()

    assert session.state is OutputState.CLOSED
    ni.factory.assert_not_called()


@pytest.mark.parametrize("edge", [SignalEdge.RISING, SignalEdge.FALLING])
def test_prepare_preloads_complete_pulses_on_the_ao_clock(
    plan: RecordTriggerPlan,
    ni: FakeNI,
    edge: SignalEdge,
) -> None:
    """Preload both pulse polarities without starting output.

    Golden samples specify two active intervals, at indices 2:4 and 6:8.
    Inverting the array gives falling-edge pulses with identical edge times.
    Configuration must use the AO rate, finite length, and shared AO clock.
    """
    plan = replace(
        plan,
        capture_settings=replace(
            plan.capture_settings,
            record_trigger=DigitalRecordTrigger(
                source="external",
                active_edge=edge,
            ),
        ),
    )
    output = NIRecordTriggerOutputSession(
        plan,
        output_channel="Dev1/port0/line0",
        sample_clock_source="/Dev1/ao/SampleClock",
        pulse_width_samples=2,
    )
    try:
        output.prepare()

        assert output.state is OutputState.PREPARED
        ni.task.do_channels.add_do_chan.assert_called_once_with(
            lines="Dev1/port0/line0",
            line_grouping=constants.LineGrouping.CHAN_PER_LINE,
        )
        ni.task.timing.cfg_samp_clk_timing.assert_called_once_with(
            rate=1000.0,
            source="/Dev1/ao/SampleClock",
            active_edge=constants.Edge.RISING,
            sample_mode=constants.AcquisitionType.FINITE,
            samps_per_chan=12,
        )
        samples = ni.task.write.call_args.args[0]
        expected = np.array(
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            dtype=np.bool_,
        )
        if edge is SignalEdge.FALLING:
            expected = ~expected

        np.testing.assert_array_equal(samples, expected)
        assert samples.dtype == np.bool_
        assert samples.flags.c_contiguous
        assert ni.task.write.call_args.kwargs == {"auto_start": False}
        ni.task.control.assert_called_once_with(constants.TaskMode.TASK_COMMIT)
        ni.task.start.assert_not_called()
    finally:
        output.close()

    ni.task.write.assert_called_once()
    ni.task.stop.assert_not_called()
    ni.task.close.assert_called_once_with()


def test_timeout_is_retryable_and_completion_retains_task(
    session: NIRecordTriggerOutputSession,
    ni: FakeNI,
) -> None:
    """Keep an armed task after timeout and observe its later completion.

    Only a bounded NI completion timeout returns False. A subsequent unbounded
    wait succeeds; repeated completion checks avoid another SDK wait. The
    completed task stays owned until explicit closure.
    """
    session.prepare()
    session.start()
    ni.task.wait_until_done.side_effect = [
        errors.DaqError("waiting", -200560),
        None,
    ]

    assert session.wait(timeout=0.01) is False
    assert session.state is OutputState.STARTED
    ni.task.stop.assert_not_called()
    ni.task.close.assert_not_called()

    assert session.wait() is True
    ni.task.wait_until_done.assert_called_with(timeout=constants.WAIT_INFINITELY)
    assert session.state is OutputState.COMPLETED
    assert session.wait(timeout=0) is True
    assert ni.task.wait_until_done.call_count == 2

    session.stop()
    session.close()
    session.close()

    ni.task.stop.assert_not_called()
    ni.task.close.assert_called_once_with()


def test_stop_is_idempotent_and_retains_task_until_close(
    session: NIRecordTriggerOutputSession,
    ni: FakeNI,
) -> None:
    """Stop an armed run once without adding an unrequested static write."""
    session.prepare()
    session.start()
    session.stop()
    session.stop()

    assert session.state is OutputState.STOPPED
    assert session.wait(timeout=0) is False
    ni.task.stop.assert_called_once_with()
    ni.task.close.assert_not_called()
    ni.task.write.assert_called_once()

    session.close()
    ni.task.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("stage", "cleanup_fails"),
    [
        (stage, False)
        for stage in (
            "create",
            "channel",
            "timing",
            "write",
            "commit",
            "start",
            "wait",
            "stop",
            "close",
        )
    ]
    + [(stage, True) for stage in ("write", "start", "wait", "stop")],
)
def test_native_failure_closes_partial_resources(
    session: NIRecordTriggerOutputSession,
    ni: FakeNI,
    stage: str,
    cleanup_fails: bool,
) -> None:
    """Preserve operation errors and close every successfully created task.

    Failures before start close the prepared or partial task directly. Start,
    wait, stop, and close failures exercise cleanup of a potentially active
    task. A failure creating the task leaves no returned task to release.
    A second failure closing the task must become a note on the original error.
    """
    if stage in {"start", "wait", "stop", "close"}:
        session.prepare()
    if stage in {"wait", "stop", "close"}:
        session.start()

    targets = {
        "create": ni.factory,
        "channel": ni.task.do_channels.add_do_chan,
        "timing": ni.task.timing.cfg_samp_clk_timing,
        "write": ni.task.write,
        "commit": ni.task.control,
        "start": ni.task.start,
        "wait": ni.task.wait_until_done,
        "stop": ni.task.stop,
        "close": ni.task.close,
    }
    failure = RuntimeError(f"{stage} failed")
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
    if cleanup_fails:
        assert any("cleanup failed" in note for note in failure.__notes__)

    if stage in {"start", "wait", "stop", "close"}:
        ni.task.stop.assert_called_once_with()
    else:
        ni.task.stop.assert_not_called()

    assert session.state is OutputState.CLOSED
    if stage == "create":
        ni.task.close.assert_not_called()
    else:
        ni.task.close.assert_called_once_with()

    session.close()


def test_incomplete_preload_closes_task(
    session: NIRecordTriggerOutputSession,
    ni: FakeNI,
) -> None:
    """Reject partial buffer acceptance before committing or starting output."""
    ni.task.write.return_value = 11

    with pytest.raises(RuntimeError, match="every digital sample"):
        session.prepare()

    assert session.state is OutputState.CLOSED
    ni.task.control.assert_not_called()
    ni.task.start.assert_not_called()
    ni.task.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("first", "width", "message"),
    [
        (0, 1, "before the first"),
        (2, 4, "inactive gap"),
        (5, 3, "after the final"),
    ],
)
def test_rejects_pulses_without_inactive_samples(
    plan: RecordTriggerPlan,
    ni: FakeNI,
    first: int,
    width: int,
    message: str,
) -> None:
    """Reject a missing leading sample, overlapping pulses, or a truncated tail."""
    with pytest.raises(ValueError, match=message):
        NIRecordTriggerOutputSession(
            replace(plan, first_trigger_sample=first),
            output_channel="Dev1/port0/line0",
            sample_clock_source="/Dev1/ao/SampleClock",
            pulse_width_samples=width,
        )

    ni.factory.assert_not_called()


@pytest.mark.parametrize(
    ("records", "first", "width"),
    [(2, 5, 2), (1, 2, 5)],
)
def test_accepts_exact_tail_and_single_record_widths(
    plan: RecordTriggerPlan,
    ni: FakeNI,
    records: int,
    first: int,
    width: int,
) -> None:
    """Allow the return edge on the last sample and wide isolated single pulses."""
    plan = replace(
        plan,
        capture_settings=replace(
            plan.capture_settings,
            records_per_batch=records,
        ),
        first_trigger_sample=first,
    )
    output = NIRecordTriggerOutputSession(
        plan,
        output_channel="/Dev1/port0/line0",
        sample_clock_source="dev1/ao/SampleClock",
        pulse_width_samples=width,
    )
    try:
        output.prepare()
        samples = ni.task.write.call_args.args[0]

        assert samples[0] == samples[-1] == np.False_
        assert np.count_nonzero(samples) == records * width
    finally:
        output.close()
