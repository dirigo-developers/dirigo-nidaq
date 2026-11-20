"""Test counter frame clocks without NI equipment or native driver libraries."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import cast
from unittest.mock import Mock

import nidaqmx
import pytest
from dirigo import units
from dirigo.hardware.output import OutputState
from dirigo.hardware.timing import DigitalStartTrigger, ExternalSampleClock, SignalEdge
from nidaqmx import constants

from dirigo_nidaq.frame_clock import NIFrameClockSession


@dataclass(frozen=True)
class FakeNI:
    """Expose the task factory, task, and configured counter channel."""

    factory: Mock
    task: Mock
    channel: Mock


@pytest.fixture(autouse=True)
def ni(monkeypatch: pytest.MonkeyPatch) -> FakeNI:
    """Patch the public NI boundary and echo requested ticks on readback."""
    task, channel = Mock(), Mock()

    def configure(
        *,
        counter: str,
        source_terminal: str,
        idle_state: constants.Level,
        initial_delay: int,
        low_ticks: int,
        high_ticks: int,
    ) -> Mock:
        """Preserve programmed values for the driver's post-commit checks."""
        channel.co_pulse_ticks_initial_delay = initial_delay
        channel.co_pulse_low_ticks = low_ticks
        channel.co_pulse_high_ticks = high_ticks
        return channel

    task.co_channels.add_co_pulse_chan_ticks.side_effect = configure
    factory = Mock(return_value=task)
    monkeypatch.setattr(nidaqmx, "Task", factory)
    return FakeNI(factory, task, channel)


@pytest.fixture
def clock() -> ExternalSampleClock:
    """Describe the record-event source independently of ADC sample timing."""
    return ExternalSampleClock(source="/Dev1/PFI0", max_rate=units.SampleRate("8 kS/s"))


@pytest.fixture
def session(clock: ExternalSampleClock) -> Iterator[NIFrameClockSession]:
    """Close the counter session even if a test assertion fails."""
    output = NIFrameClockSession(
        records_per_frame=6,
        counter_channel="Dev1/ctr0",
        record_clock=clock,
    )
    try:
        yield output
    finally:
        output.close()


def test_construction_and_unused_close_are_passive(
    session: NIFrameClockSession, ni: FakeNI
) -> None:
    """Expose internal routing without allocating a task or reserving PFI pins."""
    assert session.state is OutputState.NEW
    assert session.trigger == DigitalStartTrigger(source="/Dev1/Ctr0InternalOutput")
    session.close()
    session.close()
    ni.factory.assert_not_called()


@pytest.mark.parametrize("edge", [SignalEdge.RISING, SignalEdge.FALLING])
@pytest.mark.parametrize("export", [None, "/Dev1/PFI1"])
def test_prepare_programs_continuous_counter_and_internal_route(
    clock: ExternalSampleClock,
    ni: FakeNI,
    edge: SignalEdge,
    export: str | None,
) -> None:
    """Program a divider with default initial delay and optional export."""
    output = NIFrameClockSession(
        records_per_frame=6,
        counter_channel="/Dev1/ctr2",
        record_clock=replace(clock, active_edge=edge),
        export_terminal=export,
    )
    try:
        output.prepare()
        assert output.state is OutputState.PREPARED
        assert output.trigger.source == "/Dev1/Ctr2InternalOutput"
        ni.task.co_channels.add_co_pulse_chan_ticks.assert_called_once_with(
            counter="/Dev1/ctr2",
            source_terminal="/Dev1/PFI0",
            idle_state=constants.Level.LOW,
            initial_delay=0,
            low_ticks=4,
            high_ticks=2,
        )
        assert ni.channel.co_pulse_term == (export or "")
        assert ni.channel.co_ctr_timebase_active_edge is (
            constants.Edge.RISING
            if edge is SignalEdge.RISING
            else constants.Edge.FALLING
        )
        ni.task.timing.cfg_implicit_timing.assert_called_once_with(
            sample_mode=constants.AcquisitionType.CONTINUOUS,
        )
        ni.task.control.assert_called_once_with(constants.TaskMode.TASK_COMMIT)
        ni.task.start.assert_not_called()
        ni.task.write.assert_not_called()
        ni.task.do_channels.add_do_chan.assert_not_called()
    finally:
        output.close()
    ni.task.stop.assert_not_called()
    ni.task.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("co_pulse_high_ticks", 3),
        ("co_pulse_low_ticks", 5),
    ],
)
def test_coerced_tick_counts_fail_preparation(
    session: NIFrameClockSession, ni: FakeNI, attribute: str, value: int
) -> None:
    """Reject changes to pulse widths that would alter the requested frame period."""
    ni.task.control.side_effect = lambda _: setattr(ni.channel, attribute, value)
    with pytest.raises(RuntimeError, match="coerced"):
        session.prepare()
    assert session.state is OutputState.CLOSED
    ni.task.start.assert_not_called()
    ni.task.stop.assert_not_called()
    ni.task.close.assert_called_once_with()


@pytest.mark.parametrize("effective_delay", [0, 2])
def test_initial_delay_readback_does_not_reject_preparation(
    session: NIFrameClockSession, ni: FakeNI, effective_delay: int
) -> None:
    """Request zero delay while permitting NI's minimum startup delay.

    Model either unchanged readback or a two-tick minimum applied at commit.
    Preparation must succeed in both cases, retaining checks of the high and
    low pulse widths without treating the requested zero as an exact phase.
    """
    ni.task.control.side_effect = lambda _: setattr(
        ni.channel, "co_pulse_ticks_initial_delay", effective_delay
    )
    session.prepare()
    assert session.state is OutputState.PREPARED
    arguments = ni.task.co_channels.add_co_pulse_chan_ticks.call_args.kwargs
    assert arguments["initial_delay"] == 0
    assert ni.channel.co_pulse_ticks_initial_delay == effective_delay


def test_close_stops_running_clock_without_waiting(
    session: NIFrameClockSession,
    ni: FakeNI,
) -> None:
    """Stop and release a running divider without requiring more record edges.

    A continuous counter has no natural completion. Make native waiting fail
    immediately if called, so cleanup must succeed using stop and close only.
    Repeated closure must not issue another native operation.
    """
    ni.task.wait_until_done.side_effect = AssertionError("Must not await counter")

    session.prepare()
    session.start()

    assert session.state is OutputState.STARTED

    session.close()
    session.close()

    assert session.state is OutputState.CLOSED
    ni.task.start.assert_called_once_with()
    ni.task.stop.assert_called_once_with()
    ni.task.close.assert_called_once_with()
    ni.task.wait_until_done.assert_not_called()


def test_stop_retains_task_until_close(
    session: NIFrameClockSession, ni: FakeNI
) -> None:
    """Cancel once, keep resources reserved, and avoid extra digital commands."""
    session.prepare()
    session.start()
    session.stop()
    session.stop()
    assert session.state is OutputState.STOPPED
    ni.task.stop.assert_called_once_with()
    ni.task.close.assert_not_called()
    session.close()
    session.close()
    ni.task.close.assert_called_once_with()
    ni.task.write.assert_not_called()


@pytest.mark.parametrize(
    ("stage", "cleanup_fails"),
    [
        (stage, False)
        for stage in (
            "create",
            "configure",
            "commit",
            "start",
            "stop",
            "close",
        )
    ]
    + [(stage, True) for stage in ("configure", "commit", "start", "stop")],
)
def test_native_failure_preserves_primary_and_closes_once(
    session: NIFrameClockSession,
    ni: FakeNI,
    stage: str,
    cleanup_fails: bool,
) -> None:
    """Release partial preparation or generation while retaining the first error.

    A failed start may have partially armed the task, so cleanup attempts
    stop and close. A failed stop is not retried. Failure to close must not
    replace the original error, and repeated closure performs no more work.
    """
    if stage in {"start", "stop", "close"}:
        session.prepare()
    if stage in {"stop", "close"}:
        session.start()

    failure = RuntimeError(f"{stage} failed")
    targets = {
        "create": ni.factory,
        "configure": ni.task.co_channels.add_co_pulse_chan_ticks,
        "commit": ni.task.control,
        "start": ni.task.start,
        "stop": ni.task.stop,
        "close": ni.task.close,
    }
    targets[stage].side_effect = failure

    if cleanup_fails:
        ni.task.close.side_effect = RuntimeError("cleanup failed")

    operation = getattr(
        session,
        stage if stage in {"start", "stop", "close"} else "prepare",
    )

    with pytest.raises(RuntimeError) as caught:
        operation()

    assert caught.value is failure
    assert session.state is OutputState.CLOSED

    if stage == "create":
        ni.task.close.assert_not_called()
    else:
        ni.task.close.assert_called_once_with()

    assert ni.task.stop.call_count == int(stage in {"start", "stop", "close"})

    if cleanup_fails:
        assert any("cleanup failed" in note for note in failure.__notes__)

    session.close()


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("records_per_frame", None, TypeError),
        ("records_per_frame", True, TypeError),
        ("records_per_frame", False, TypeError),
        ("records_per_frame", 6.0, TypeError),
        ("records_per_frame", "6", TypeError),
        ("records_per_frame", -1, ValueError),
        ("records_per_frame", 0, ValueError),
        ("records_per_frame", 1, ValueError),
        ("records_per_frame", 2, ValueError),
        ("records_per_frame", 3, ValueError),
        ("record_clock", object(), TypeError),
        ("counter_channel", None, TypeError),
        ("counter_channel", "Dev1/port0/line0", ValueError),
        ("export_terminal", 1, TypeError),
        ("export_terminal", "", ValueError),
        ("export_terminal", " /Dev1/PFI1", ValueError),
    ],
)
def test_constructor_rejects_invalid_arguments(
    clock: ExternalSampleClock,
    ni: FakeNI,
    field: str,
    value: object,
    error_type: type[Exception],
) -> None:
    """Reject invalid divider periods and connection descriptions passively."""
    with pytest.raises(error_type):
        NIFrameClockSession(
            records_per_frame=(cast(int, value) if field == "records_per_frame" else 6),
            counter_channel=(
                cast(str, value) if field == "counter_channel" else "Dev1/ctr0"
            ),
            record_clock=(
                cast(ExternalSampleClock, value) if field == "record_clock" else clock
            ),
            export_terminal=(
                cast(str | None, value) if field == "export_terminal" else None
            ),
        )

    ni.factory.assert_not_called()


@pytest.mark.parametrize("records", [4, 6, 12])
def test_period_follows_records_per_frame(
    clock: ExternalSampleClock,
    ni: FakeNI,
    records: int,
) -> None:
    """Set the divider period independently of capture batching or run length."""
    output = NIFrameClockSession(
        records_per_frame=records,
        counter_channel="Dev1/ctr0",
        record_clock=clock,
    )
    try:
        output.prepare()

        assert ni.channel.co_pulse_high_ticks == 2
        assert ni.channel.co_pulse_low_ticks == records - 2
        ni.task.timing.cfg_implicit_timing.assert_called_once_with(
            sample_mode=constants.AcquisitionType.CONTINUOUS,
        )
    finally:
        output.close()


def test_lifecycle_rejects_repeated_or_premature_calls(
    session: NIFrameClockSession,
) -> None:
    """Reject start/wait before preparation and repeated preparation or start."""
    with pytest.raises(RuntimeError, match="Session must be"):
        session.start()
    session.prepare()
    with pytest.raises(RuntimeError, match="Session must be"):
        session.prepare()
    session.start()
    with pytest.raises(RuntimeError, match="Session must be"):
        session.start()
    session.close()
    session.stop()
