"""Repeat one grouped galvo waveform on hardware frame triggers."""

import time
from collections.abc import Mapping

import nidaqmx
from dirigo.hardware.scanner import GalvoScanner
from dirigo.hardware.timing import DigitalStartTrigger
from dirigo.hardware.waveform import AngularWaveform
from nidaqmx import constants

from dirigo_nidaq.output import NIGalvoOutputSession

_POLL_INTERVAL = 0.005


class NIRetriggeredGalvoOutputSession(NIGalvoOutputSession):
    """Replay one frame per accepted trigger using the onboard AO sample clock.

    Each supplied waveform describes one frame. Axes share a sample rate and
    sample count, and must belong to the same NI device. Construction is
    passive. prepare() reserves the axes, enables regeneration and hardware
    retriggering, preloads one frame, and commits the task without starting it.
    NI rejects unsupported hardware or routes during preparation.

    start() arms once. The trigger source must supply exactly frame_count
    frame triggers, separated enough for each finite waveform to finish and
    rearm. Frame planning must leave timing margin; this session cannot infer
    trigger spacing from an opaque terminal name. Triggers arriving while AO
    is busy can be ignored by NI hardware.

    wait() polls NI's generated-samples-per-channel count, rather than using
    wait_until_done() on a task that rearms indefinitely. Once all expected
    samples are reported generated, wait() stops the task and returns True.
    A timeout retains the armed task and axis reservations. frame_count is a
    software completion target, not a hardware limit on accepted triggers.
    Observed excess generation raises an error; polling cannot prevent extra
    triggers or prove a particular frame-to-trigger correspondence.

    Completion and stop retain resources until close(). No static angle or
    parking command is issued. Call session methods serially. Hardware must
    outlive the session; bounded waits allow cooperative cancellation.
    """

    def __init__(
        self,
        waveforms: Mapping[GalvoScanner, AngularWaveform],
        *,
        start_trigger: DigitalStartTrigger,
        frame_count: int,
    ) -> None:
        """Validate one-frame waveforms and the finite target without NI calls."""
        if not isinstance(start_trigger, DigitalStartTrigger):
            raise TypeError("start_trigger must be a DigitalStartTrigger.")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise TypeError("frame_count must be an integer.")
        if frame_count <= 0:
            raise ValueError("frame_count must be positive.")
        super().__init__(waveforms, start_trigger=start_trigger)
        self._frame_count = frame_count
        self._expected_samples = self._sample_count * frame_count
        self._generated_samples = 0
        self._armed = False

    @property
    def frame_count(self) -> int:
        """Expected number of complete one-frame waveform generations."""
        return self._frame_count

    @property
    def completed_frames(self) -> int:
        """Complete frames from the latest successful wait() progress observation.

        This cached value performs no NI calls. A partially generated frame
        is excluded. Zero before the first observation does not establish
        that hardware has remained idle.
        """
        return self._generated_samples // self._sample_count

    def _configure_output_task(self, task: nidaqmx.Task) -> None:
        """Enable repeated hardware starts and reuse of the one-frame buffer."""
        task.triggers.start_trigger.retriggerable = True
        task.out_stream.regen_mode = constants.RegenerationMode.ALLOW_REGENERATION

    def _prepare_output(self) -> None:
        """Preload through the shared NI implementation, then commit the task."""
        super()._prepare_output()
        self._require_task().control(constants.TaskMode.TASK_COMMIT)

    def _start_output(self) -> None:
        """Track possible activity even if NI partially starts before raising."""
        self._armed = True
        super()._start_output()

    def _wait_output(self, timeout: float | None) -> bool:
        """Observe generated samples until the finite target or timeout is reached."""
        deadline = None if timeout is None else time.monotonic() + timeout
        task = self._require_task()
        while True:
            # A retriggered task normally reports False. Query it to surface
            # asynchronous NI errors, not as the frame-completion criterion.
            done = task.is_task_done()
            generated = task.out_stream.total_samp_per_chan_generated
            if generated < self._generated_samples:
                raise RuntimeError("NI generated-sample count moved backwards.")
            if generated > self._expected_samples:
                raise RuntimeError("NI generated more than the expected frame samples.")
            self._generated_samples = generated
            if generated == self._expected_samples:
                self._stop_output()
                return True
            if done:
                raise RuntimeError("NI output ended before all frames were generated.")
            if deadline is None:
                delay = _POLL_INTERVAL
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                delay = min(_POLL_INTERVAL, remaining)
            time.sleep(delay)

    def _stop_output(self) -> None:
        """Disarm once, including after a partial start or a failed stop attempt."""
        if self._armed:
            self._armed = False
            super()._stop_output()
