"""Divide acquired-record events into internally routable counter frame pulses."""

import re

import nidaqmx
from dirigo.hardware.frame_plan import RecordFramePlan
from dirigo.hardware.output import OutputState
from dirigo.hardware.timing import DigitalStartTrigger, ExternalSampleClock, SignalEdge
from nidaqmx import constants

_WAIT_UNTIL_DONE_TIMEOUT = -200560
_NI_EDGES = {
    SignalEdge.RISING: constants.Edge.RISING,
    SignalEdge.FALLING: constants.Edge.FALLING,
}


class NIFrameClockSession:
    """Divide acquired-record events into continuous frame-trigger pulses.

    counter_channel selects one counter, e.g. Dev1/ctr0. record_clock supplies
    one active edge per acquired record. Its max_rate describes record events,
    not ADC samples. Counter timing uses these events directly as ticks.

    Each pulse is high for two ticks and low for records_per_frame - 2 ticks.
    At least four records per frame are required. Preparation requests zero
    initial delay but permits NI's hardware minimum. A PCIe-6323 has reported
    two record-clock ticks after committing that request. The first trigger
    is therefore not guaranteed to coincide with the first acquired record.
    Subsequent triggers remain records_per_frame ticks apart.

    Logical frame grouping remains defined by the plan. Downstream frame
    processing accounts for line alignment, including galvo response delay.
    This session does not provide an adjustable alignment parameter.

    trigger identifies CtrNInternalOutput for same-device AO routing without
    a jumper. export_terminal optionally also exports pulses to a physical
    terminal. None disables the default PFI export.

    Construction accesses no hardware. Arm AO, start this session, then start
    capture. The record-clock source must remain quiet outside acquisition.

    The plan supplies the divider period; frame_count does not limit this
    continuous task. There is no wait() or natural COMPLETED state. The owner
    explicitly stops the counter after capture and AO finish, or during
    cancellation. stop() retains resources until close() and may interrupt a
    pulse. No parking commands or static digital writes are performed.
    """

    def __init__(
        self,
        plan: RecordFramePlan,
        *,
        counter_channel: str,
        record_clock: ExternalSampleClock,
        export_terminal: str | None = None,
    ) -> None:
        """Validate the finite plan and connection descriptions without NI calls."""
        if not isinstance(plan, RecordFramePlan):
            raise TypeError("plan must be a RecordFramePlan.")
        if plan.records_per_frame < 4:
            raise ValueError(
                "Counter frame clock requires at least four records per frame."
            )
        if not isinstance(record_clock, ExternalSampleClock):
            raise TypeError("record_clock must be an ExternalSampleClock.")
        if not isinstance(counter_channel, str):
            raise TypeError("counter_channel must be a string.")
        match = re.fullmatch(
            r"/?([A-Za-z0-9_-]+)/ctr([0-9]+)", counter_channel, flags=re.IGNORECASE
        )
        if match is None:
            raise ValueError(
                "counter_channel must name one NI counter, e.g. Dev1/ctr0."
            )
        if export_terminal is not None:
            if not isinstance(export_terminal, str):
                raise TypeError("export_terminal must be a string or None.")
            if not export_terminal or export_terminal != export_terminal.strip():
                raise ValueError(
                    "export_terminal must be nonempty without edge whitespace."
                )

        self._plan = plan
        self._counter_channel = counter_channel
        self._record_clock = record_clock
        self._export_terminal = export_terminal
        self._trigger = DigitalStartTrigger(
            source=f"/{match[1]}/Ctr{int(match[2])}InternalOutput"
        )
        self._task: nidaqmx.Task | None = None
        self._running = False
        self._state = OutputState.NEW

    @property
    def state(self) -> OutputState:
        """Last observed state; STARTED can mean waiting for record-clock edges."""
        return self._state

    @property
    def trigger(self) -> DigitalStartTrigger:
        """Rising-edge trigger on the counter's internally routable output."""
        return self._trigger

    def prepare(self) -> None:
        """Commit continuous pulses and reject changes to the divider period."""
        self._require_state(OutputState.NEW)
        try:
            task = nidaqmx.Task()
            self._task = task
            high = 2
            low = self._plan.records_per_frame - high
            channel = task.co_channels.add_co_pulse_chan_ticks(
                counter=self._counter_channel,
                source_terminal=self._record_clock.source,
                idle_state=constants.Level.LOW,
                initial_delay=0,
                low_ticks=low,
                high_ticks=high,
            )
            channel.co_ctr_timebase_active_edge = _NI_EDGES[
                self._record_clock.active_edge
            ]
            channel.co_pulse_term = self._export_terminal or ""
            task.timing.cfg_implicit_timing(
                sample_mode=constants.AcquisitionType.CONTINUOUS,
            )
            task.control(constants.TaskMode.TASK_COMMIT)
            actual = (
                channel.co_pulse_high_ticks,
                channel.co_pulse_low_ticks,
            )
            if actual != (high, low):
                raise RuntimeError("NI coerced the requested frame-clock tick counts.")
        except BaseException as failure:
            self._close_after_failure(failure)
            raise
        self._state = OutputState.PREPARED

    def start(self) -> None:
        """Arm the counter before capture starts supplying record events."""
        self._require_state(OutputState.PREPARED)
        task = self._task
        assert task is not None
        self._state = OutputState.STARTED
        self._running = True
        try:
            task.start()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def stop(self) -> None:
        """Stop once while retaining the prepared task until close()."""
        try:
            self._stop_unfinished()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def _stop_unfinished(self) -> None:
        """Stop possible activity after a complete or partially failed start."""
        if self._state in (
            OutputState.STOPPED,
            OutputState.CLOSED,
        ):
            return
        self._state = OutputState.STOPPED
        if self._running:
            self._running = False
            task = self._task
            assert task is not None
            task.stop()

    def close(self) -> None:
        """Attempt stop and close once, preserving the first failure."""
        if self._state is OutputState.CLOSED:
            return
        failure: BaseException | None = None
        try:
            self._stop_unfinished()
        except BaseException as exc:
            failure = exc
        task, self._task = self._task, None
        self._state = OutputState.CLOSED
        if task is not None:
            try:
                task.close()
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(f"Frame-clock cleanup also failed: {exc!r}")
        if failure is not None:
            raise failure

    def _require_state(self, expected: OutputState) -> None:
        """Reject an operation outside its required lifecycle state."""
        if self._state is not expected:
            raise RuntimeError(
                f"Session must be {expected.value}; it is {self._state.value}."
            )

    def _close_after_failure(self, failure: BaseException) -> None:
        """Release resources without replacing the primary operation error."""
        try:
            self.close()
        except BaseException as cleanup:
            failure.add_note(f"Frame-clock cleanup also failed: {cleanup!r}")
            for note in getattr(cleanup, "__notes__", ()):
                failure.add_note(note)
