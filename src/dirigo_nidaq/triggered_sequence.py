"""Coordinate finite record-framed capture and retriggered NI galvo output."""

from collections.abc import Callable, Mapping
from time import monotonic

from dirigo.hardware.capture import CaptureInfo, CaptureLayout, SampleArray
from dirigo.hardware.capture_session import CaptureState
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.frame_plan import RecordFramePlan
from dirigo.hardware.scanner import GalvoScanner
from dirigo.hardware.timing import ExternalSampleClock
from dirigo.hardware.waveform import AngularWaveform

from dirigo_nidaq.frame_clock import NIFrameClockSession
from dirigo_nidaq.retriggered_output import NIRetriggeredGalvoOutputSession


def _run_all(*actions: Callable[[], None]) -> None:
    """Attempt every action, preserving the first error and noting later errors."""
    failure: BaseException | None = None
    for action in actions:
        try:
            action()
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note(f"Sequence cleanup also failed: {exc!r}")
                for note in getattr(exc, "__notes__", ()):
                    failure.add_note(note)
    if failure is not None:
        raise failure


class NITriggeredFrameSequence:
    """Own capture, frame-clock, and AO sessions for one finite frame sequence.

    Devices are borrowed and must outlive this single-use coordinator. The
    capture is created from plan.capture_settings, preserving backend-specific
    settings subclasses. Construction accesses no hardware or reservations.

    record_clock names the NI input receiving one active edge per acquired
    record. frame_counter_channel selects the counter generating frame pulses.
    AO uses its CtrNInternalOutput signal, with no physical jumper on the same
    NI device. frame_clock_export_terminal optionally also exports the pulses.
    The record-clock source must remain quiet until capture starts. For Alazar,
    configure acquisition-active AUX output in the capture settings; this
    coordinator neither imports Alazar nor configures that connection itself.

    The counter requests zero initial delay and permits NI's hardware minimum.
    A PCIe-6323 has reported two record-clock ticks after committing that
    request. First-pulse phase is not inferred from logical record indices.
    Downstream frame processing accounts for line alignment and galvo response
    delay; this coordinator has no adjustable alignment offset.

    Each waveform describes one frame. AO uses its own onboard sample clock;
    no equality with the ADC sample grid is required. The caller plans waveform
    duration and rearm margin, positions axes, and allows settling. The
    coordinator does not start or set the amplitude of a resonant scanner.

    prepare() prepares capture, commits the idle-low counter, then prepares
    AO. start() arms AO, arms the frame clock, then starts capture. Transfer
    batches and frame boundaries may cross.

    read_into() delivers complete batches. It returns None only after capture
    EOF, every expected AO frame, and successful explicit frame-clock stop.
    The frame counter runs continuously and is never awaited for completion.
    The record-clock source must remain quiet after capture finishes.

    Capture reading and AO waiting share one timeout budget. Timeouts and
    caller argument errors retain usability; fatal child failures close all
    three sessions. stop() retains resources until close(). Cleanup attempts
    AO first, then frame clock, then capture, without closing devices or
    issuing parking commands. Call methods serially and use bounded reads
    for cooperative cancellation.
    """

    def __init__(
        self,
        digitizer: Digitizer,
        plan: RecordFramePlan,
        waveforms: Mapping[GalvoScanner, AngularWaveform],
        *,
        record_clock: ExternalSampleClock,
        frame_counter_channel: str,
        frame_clock_export_terminal: str | None = None,
    ) -> None:
        """Validate connections and create owned sessions without hardware calls."""
        if not isinstance(digitizer, Digitizer):
            raise TypeError("digitizer must be a Digitizer.")
        if not isinstance(plan, RecordFramePlan):
            raise TypeError("plan must be a RecordFramePlan.")
        clock = NIFrameClockSession(
            plan,
            counter_channel=frame_counter_channel,
            record_clock=record_clock,
            export_terminal=frame_clock_export_terminal,
        )
        output = NIRetriggeredGalvoOutputSession(
            waveforms,
            start_trigger=clock.trigger,
            frame_count=plan.frame_count,
        )
        capture = digitizer.create_capture(plan.capture_settings)
        self._plan = plan
        self._capture = capture
        self._clock = clock
        self._output = output
        self._state = CaptureState.NEW

    @property
    def state(self) -> CaptureState:
        """Sequence state; COMPLETED includes AO completion and counter stop."""
        return self._state

    @property
    def layout(self) -> CaptureLayout:
        """Prepared capture batch layout, retained after closure."""
        return self._capture.layout

    @property
    def batches_acquired(self) -> int:
        """Complete batches delivered so far, retained after closure."""
        return self._capture.batches_acquired

    @property
    def frame_count(self) -> int:
        """Number of complete frames requested by the immutable plan."""
        return self._plan.frame_count

    def prepare(self) -> None:
        """Prepare capture, frame clock, then AO; roll back partial preparation."""
        self._require_state(CaptureState.NEW)
        try:
            self._capture.prepare()
            if (
                self.layout.records_per_batch
                != self._plan.capture_settings.records_per_batch
            ):
                raise ValueError(
                    "Capture record batching does not match the frame plan."
                )
            self._clock.prepare()
            self._output.prepare()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise
        self._state = CaptureState.PREPARED

    def start(self) -> None:
        """Arm both NI consumers before starting the source of record events."""
        self._require_state(CaptureState.PREPARED)
        self._state = CaptureState.STARTED
        try:
            self._output.start()
            self._clock.start()
            self._capture.start()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def read_into(
        self, destination: SampleArray, *, timeout: float | None = None
    ) -> CaptureInfo | None:
        """Deliver a batch or finish AO and stop the frame counter before EOF.

        Capture validates destination and timeout even on repeated EOF.
        Completing a transfer batch does not establish AO completion.

        An AO timeout is retryable without redelivering the last batch or
        restarting any child. The counter remains armed while AO is incomplete;
        its record-clock source must be quiet after capture finishes.

        Counter stop failure closes every child and prevents successful EOF.
        None permits an unbounded AO wait. The timeout does not bound native
        cleanup calls.
        """
        if self._state is not CaptureState.COMPLETED:
            self._require_state(CaptureState.STARTED)

        started = monotonic()
        try:
            info = self._capture.read_into(destination, timeout=timeout)
        except BaseException as failure:
            if self._capture.state is CaptureState.CLOSED:
                self._close_after_failure(failure)
            raise

        if info is not None or self._state is CaptureState.COMPLETED:
            return info

        remaining = (
            None
            if timeout is None
            else max(0.0, float(timeout) - (monotonic() - started))
        )
        try:
            complete = self._output.wait(timeout=remaining)
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

        if not complete:
            raise TimeoutError("Sequence AO output has not completed.")

        try:
            self._clock.stop()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

        self._state = CaptureState.COMPLETED
        return None

    def stop(self) -> None:
        """Stop AO, frame clock, then capture, retaining resources until close()."""
        if self._state in (CaptureState.STOPPED, CaptureState.CLOSED):
            return
        self._state = CaptureState.STOPPED
        try:
            _run_all(self._output.stop, self._clock.stop, self._capture.stop)
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def close(self) -> None:
        """Close every child in order, preserving the first cleanup failure."""
        if self._state is CaptureState.CLOSED:
            return
        self._state = CaptureState.CLOSED
        _run_all(self._output.close, self._clock.close, self._capture.close)

    def _require_state(self, expected: CaptureState) -> None:
        """Reject an operation outside its required coordinator state."""
        if self._state is not expected:
            raise RuntimeError(
                f"Sequence must be {expected.value}; it is {self._state.value}."
            )

    def _close_after_failure(self, failure: BaseException) -> None:
        """Attempt every child cleanup without replacing an operation error."""
        try:
            self.close()
        except BaseException as cleanup:
            failure.add_note(f"Sequence cleanup also failed: {cleanup!r}")
            for note in getattr(cleanup, "__notes__", ()):
                failure.add_note(note)
