"""Finite record-trigger pulses clocked by NI analog output."""

import math
import re

import nidaqmx
import numpy as np
from dirigo.hardware.output import OutputState
from dirigo.hardware.timing import SignalEdge
from nidaqmx import constants, errors

from dirigo_nidaq.timing import RecordTriggerPlan

_WAIT_UNTIL_DONE_TIMEOUT = -200560


class NIRecordTriggerOutputSession:
    """Own one finite digital pulse train synchronized to an AO sample grid.

    output_channel names one line, e.g. Dev1/port0/line0. sample_clock_source
    names the same device's AO sample clock, e.g. /Dev1/ao/SampleClock.
    The caller must configure AO with the plan's waveform rate and length.
    Device support and routing are checked by NI during preparation.

    Pulses use the capture's record-trigger edge. Require an inactive first
    sample, an inactive gap between pulses, and an inactive final sample.
    first_trigger_sample must therefore be at least one. Pulse width is in
    AO sample intervals; the digitizer's sampling rate does not retime it.

    Construction is passive. prepare() commits and preloads a task. start()
    arms it before the caller starts AO; it does not start the clock source.
    Methods must be called serially. Use bounded wait() calls for cancellation.
    Completion and stop retain resources until close(). Cancellation may
    interrupt a pulse; stop() and close() issue no additional static write.
    """

    def __init__(
        self,
        plan: RecordTriggerPlan,
        *,
        output_channel: str,
        sample_clock_source: str,
        pulse_width_samples: int = 1,
    ) -> None:
        """Validate pulse geometry and NI identifiers without creating a task."""
        if not isinstance(plan, RecordTriggerPlan):
            raise TypeError("plan must be a RecordTriggerPlan.")
        if not isinstance(output_channel, str):
            raise TypeError("output_channel must be a string.")
        if not isinstance(sample_clock_source, str):
            raise TypeError("sample_clock_source must be a string.")

        line = re.fullmatch(
            r"/?([A-Za-z0-9_-]+)/port[0-9]+/line[0-9]+",
            output_channel,
            flags=re.IGNORECASE,
        )
        clock = re.fullmatch(
            r"/?([A-Za-z0-9_-]+)/ao/SampleClock",
            sample_clock_source,
            flags=re.IGNORECASE,
        )
        if line is None:
            raise ValueError("output_channel must name one NI digital line.")
        if clock is None:
            raise ValueError("sample_clock_source must name an NI AO sample clock.")
        if line[1].casefold() != clock[1].casefold():
            raise ValueError("Digital output and AO clock must use the same device.")

        if not isinstance(pulse_width_samples, int) or isinstance(
            pulse_width_samples, bool
        ):
            raise TypeError("pulse_width_samples must be an integer.")
        if pulse_width_samples <= 0:
            raise ValueError("pulse_width_samples must be positive.")
        if plan.first_trigger_sample == 0:
            raise ValueError("An inactive sample is required before the first pulse.")
        if plan.record_count > 1 and pulse_width_samples >= plan.interval_samples:
            raise ValueError("Pulses require an inactive gap between active edges.")
        if plan.sample_indices[-1] + pulse_width_samples >= plan.waveform.samples.size:
            raise ValueError("An inactive sample is required after the final pulse.")

        self._plan = plan
        self._output_channel = output_channel
        self._sample_clock_source = sample_clock_source
        self._pulse_width_samples = pulse_width_samples
        self._task: nidaqmx.Task | None = None
        self._state = OutputState.NEW

    @property
    def state(self) -> OutputState:
        """Last observed state; STARTED can mean waiting for AO clock edges."""
        return self._state

    def prepare(self) -> None:
        """Preload and commit output, closing partial resources on failure."""
        self._require_state(OutputState.NEW)
        try:
            plan = self._plan
            trigger = plan.capture_settings.record_trigger
            assert trigger is not None

            active = trigger.active_edge is SignalEdge.RISING
            samples = np.full(
                plan.waveform.samples.size,
                not active,
                dtype=np.bool_,
            )
            for index in plan.sample_indices:
                samples[index : index + self._pulse_width_samples] = active

            task = nidaqmx.Task()
            self._task = task
            task.do_channels.add_do_chan(
                lines=self._output_channel,
                line_grouping=constants.LineGrouping.CHAN_PER_LINE,
            )
            task.timing.cfg_samp_clk_timing(
                rate=float(plan.waveform.sample_rate),
                source=self._sample_clock_source,
                active_edge=constants.Edge.RISING,
                sample_mode=constants.AcquisitionType.FINITE,
                samps_per_chan=samples.size,
            )
            if task.write(samples, auto_start=False) != samples.size:
                raise RuntimeError("NI task did not accept every digital sample.")
            task.control(constants.TaskMode.TASK_COMMIT)
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

        self._state = OutputState.PREPARED

    def start(self) -> None:
        """Arm once before AO starts generating the shared sample clock."""
        self._require_state(OutputState.PREPARED)
        task = self._task
        assert task is not None

        self._state = OutputState.STARTED
        try:
            task.start()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def wait(self, timeout: float | None = None) -> bool:
        """Return completion or False on timeout, retaining the task."""
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
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
                timeout=(constants.WAIT_INFINITELY if timeout is None else timeout)
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

        self._state = OutputState.COMPLETED
        return True

    def stop(self) -> None:
        """Stop an unfinished run once; keep its task until close()."""
        try:
            self._stop_unfinished()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def _stop_unfinished(self) -> None:
        """Stop a potentially active task, including a partially failed start."""
        if self._state in (
            OutputState.COMPLETED,
            OutputState.STOPPED,
            OutputState.CLOSED,
        ):
            return

        started = self._state is OutputState.STARTED
        self._state = OutputState.STOPPED
        if started:
            task = self._task
            assert task is not None
            task.stop()

    def close(self) -> None:
        """Stop and release once, preserving the first failure if both fail."""
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
                    failure.add_note(f"Trigger output cleanup also failed: {exc!r}")

        if failure is not None:
            raise failure

    def _require_state(self, expected: OutputState) -> None:
        """Reject an operation outside its required lifecycle state."""
        if self._state is not expected:
            raise RuntimeError(
                f"Session must be {expected.value}; it is {self._state.value}."
            )

    def _close_after_failure(self, failure: BaseException) -> None:
        """Release resources without replacing an operation's original error."""
        try:
            self.close()
        except BaseException as cleanup:
            failure.add_note(f"Trigger output cleanup also failed: {cleanup!r}")
            for note in getattr(cleanup, "__notes__", ()):
                failure.add_note(note)
