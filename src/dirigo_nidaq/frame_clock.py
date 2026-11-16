"""Generate finite frame-start pulses from acquired-record clock edges."""

import math
import re
from numbers import Real

import nidaqmx
import numpy as np
from dirigo.hardware.frame_plan import RecordFramePlan
from dirigo.hardware.output import OutputState
from dirigo.hardware.timing import ExternalSampleClock, SignalEdge
from nidaqmx import constants, errors

_WAIT_UNTIL_DONE_TIMEOUT = -200560
_NI_EDGES = {
    SignalEdge.RISING: constants.Edge.RISING,
    SignalEdge.FALLING: constants.Edge.FALLING,
}


class NIFrameClockSession:
    """Own a finite Port 0 digital pattern advancing once per acquired record.

    output_channel names one line, e.g. Dev1/port0/line0. record_clock names
    the NI input receiving one active edge per acquired record. Its max_rate
    is the maximum expected record-event rate, not the ADC sample rate.
    NI validates the physical line, clock route, and rate during preparation.

    Sample zero is high. Later samples are high exactly at the plan's frame
    starts and low otherwise. Pulses last one record-clock interval. Require
    at least two records per frame so pulses have intervening low samples and
    the final generated sample is low. Output pulses always have rising edges;
    record_clock.active_edge selects which input edges advance the pattern.

    Construction is passive. prepare() drives the line low with an on-demand
    write, then configures and preloads buffered output. Prepare before arming
    the receiving AO task. Arm AO, call this session's start(), then start the
    capture supplying record events. The clock must remain quiet until then.
    Wire the output line to the AO trigger input used by the receiving task.

    COMPLETED means the digital pattern finished; it establishes neither DMA
    delivery nor completion of AO waveforms. Completion and stop retain the
    task until close(). Cancellation can interrupt a high pulse; stop() and
    close() issue no additional static write. Call methods serially and use
    bounded waits for cooperative cancellation.
    """

    def __init__(
        self,
        plan: RecordFramePlan,
        *,
        output_channel: str,
        record_clock: ExternalSampleClock,
    ) -> None:
        """Validate the finite plan and connection descriptions without NI calls."""
        if not isinstance(plan, RecordFramePlan):
            raise TypeError("plan must be a RecordFramePlan.")
        if plan.records_per_frame < 2:
            raise ValueError("Frame clock requires at least two records per frame.")
        if not isinstance(record_clock, ExternalSampleClock):
            raise TypeError("record_clock must be an ExternalSampleClock.")
        if not isinstance(output_channel, str):
            raise TypeError("output_channel must be a string.")
        if (
            re.fullmatch(
                r"/?[A-Za-z0-9_-]+/port0/line[0-9]+",
                output_channel,
                flags=re.IGNORECASE,
            )
            is None
        ):
            raise ValueError("output_channel must name one NI Port 0 digital line.")

        self._plan = plan
        self._output_channel = output_channel
        self._record_clock = record_clock
        self._task: nidaqmx.Task | None = None
        self._running = False
        self._state = OutputState.NEW

    @property
    def state(self) -> OutputState:
        """Last observed state; STARTED can mean waiting for record-clock edges."""
        return self._state

    def prepare(self) -> None:
        """Drive low, then preload finite output, closing partial resources on error."""
        self._require_state(OutputState.NEW)
        try:
            samples = np.zeros(self._plan.record_count, dtype=np.bool_)
            samples[:: self._plan.records_per_frame] = True
            task = nidaqmx.Task()
            self._task = task
            task.do_channels.add_do_chan(
                lines=self._output_channel,
                line_grouping=constants.LineGrouping.CHAN_PER_LINE,
            )

            # A fresh task is on-demand. Track possible activity before the
            # write because a failed auto-start may still have started it.
            self._running = True
            if task.write(False, auto_start=True) != 1:
                raise RuntimeError("NI task did not accept the inactive level.")
            self._running = False
            task.stop()
            task.control(constants.TaskMode.TASK_UNRESERVE)

            clock = self._record_clock
            task.timing.cfg_samp_clk_timing(
                rate=float(clock.max_rate),
                source=clock.source,
                active_edge=_NI_EDGES[clock.active_edge],
                sample_mode=constants.AcquisitionType.FINITE,
                samps_per_chan=samples.size,
            )
            if task.write(samples, auto_start=False) != samples.size:
                raise RuntimeError("NI task did not accept every frame-clock sample.")
            task.control(constants.TaskMode.TASK_COMMIT)
        except BaseException as failure:
            self._close_after_failure(failure)
            raise
        self._state = OutputState.PREPARED

    def start(self) -> None:
        """Arm the preloaded pattern before the source starts supplying records."""
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

    def wait(self, timeout: float | None = None) -> bool:
        """Return True for finite completion or False for timeout or prior stop."""
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, Real):
                raise TypeError("timeout must be a real number of seconds or None.")
            timeout = float(timeout)
            if not math.isfinite(timeout) or timeout < 0:
                raise ValueError("timeout must be finite and nonnegative.")
        if self._state is OutputState.COMPLETED:
            return True
        if self._state is OutputState.STOPPED:
            return False
        self._require_state(OutputState.STARTED)
        task = self._task
        assert task is not None
        try:
            task.wait_until_done(
                timeout=constants.WAIT_INFINITELY if timeout is None else timeout
            )
        except BaseException as failure:
            if (
                isinstance(failure, errors.DaqError)
                and timeout is not None
                and failure.error_code == _WAIT_UNTIL_DONE_TIMEOUT
            ):
                return False
            self._close_after_failure(failure)
            raise
        self._running = False
        self._state = OutputState.COMPLETED
        return True

    def stop(self) -> None:
        """Stop once while retaining the prepared task until close()."""
        try:
            self._stop_unfinished()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def _stop_unfinished(self) -> None:
        """Stop possible activity during baseline setup or buffered generation."""
        if self._state in (
            OutputState.COMPLETED,
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
