"""Coordinate one finite capture with externally started NI galvo output."""

from collections.abc import Callable, Mapping
from time import monotonic

from dirigo.hardware.capture import CaptureInfo, CaptureLayout, SampleArray
from dirigo.hardware.capture_session import CaptureState, DigitizerCaptureSession
from dirigo.hardware.scanner import GalvoScanner
from dirigo.hardware.timing import DigitalStartTrigger
from dirigo.hardware.waveform import AngularWaveform

from dirigo_nidaq import output as ni_output


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
                failure.add_note(f"Frame cleanup also failed: {exc!r}")
                for note in getattr(exc, "__notes__", ()):
                    failure.add_note(note)
    if failure is not None:
        raise failure


class NITriggeredFrame:
    """Own one finite capture and one externally started NI waveform run.

    Successful construction takes ownership of a NEW capture session. Its
    digitizer and the waveform axes remain borrowed devices. Construction
    accesses no hardware. A constructor error leaves capture caller-owned.

    The caller must configure and wire start_trigger to an acquisition-related
    signal that remains inactive until capture starts. For Alazar, request
    ACQUISITION_ACTIVE AUX output and connect it to the named NI input.
    Arm AO before starting capture so its first acquisition event starts AO.

    The caller supplies waveform duration and phase, positions the axes at
    their starting angles, and allows settling before prepare(). AO uses its
    internal sample clock. Capture and waveform rates and lengths can differ;
    this coordinator establishes a start relationship, not a shared clock.

    One frame can span several capture batches. No record-to-image-line
    mapping is inferred. read_into() returns None only after capture EOF and
    observed AO completion. Timeouts are retryable. stop() retains resources;
    close() releases both sessions without closing or parking devices.
    Session methods must be called serially.
    """

    def __init__(
        self,
        capture: DigitizerCaptureSession,
        waveforms: Mapping[GalvoScanner, AngularWaveform],
        *,
        start_trigger: DigitalStartTrigger,
    ) -> None:
        """Validate the run and accept ownership without reserving resources."""
        if not isinstance(capture, DigitizerCaptureSession):
            raise TypeError("capture must be a DigitizerCaptureSession.")
        if capture.state is not CaptureState.NEW:
            raise ValueError("capture must be NEW.")
        if capture.batch_count is None:
            raise ValueError("Frame requires a finite capture.")
        if not isinstance(start_trigger, DigitalStartTrigger):
            raise TypeError("start_trigger must be a DigitalStartTrigger.")

        output = ni_output.NIGalvoOutputSession(
            waveforms,
            start_trigger=start_trigger,
        )
        self._capture = capture
        self._output = output
        self._state = CaptureState.NEW

    @property
    def state(self) -> CaptureState:
        """Frame state; COMPLETED includes observed AO completion."""
        return self._state

    @property
    def layout(self) -> CaptureLayout:
        """Prepared batch layout, retained after closure."""
        return self._capture.layout

    @property
    def batches_acquired(self) -> int:
        """Complete batches already delivered, retained after closure."""
        return self._capture.batches_acquired

    def prepare(self) -> None:
        """Prepare capture then AO, closing both sessions on failure."""
        self._require_state(CaptureState.NEW)
        try:
            self._capture.prepare()
            self._output.prepare()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise
        self._state = CaptureState.PREPARED

    def start(self) -> None:
        """Arm AO before starting the source of its acquisition-related trigger."""
        self._require_state(CaptureState.PREPARED)
        self._state = CaptureState.STARTED
        try:
            self._output.start()
            self._capture.start()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def read_into(
        self,
        destination: SampleArray,
        *,
        timeout: float | None = None,
    ) -> CaptureInfo | None:
        """Read a batch, or observe AO completion before returning capture EOF.

        timeout is one total waiting budget in seconds. Capture validates it
        and the destination, including on repeated EOF. Caller argument errors
        and timeouts retain usability. Fatal child failures close the frame.
        An AO completion timeout can be retried without redelivering a batch.
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
            raise TimeoutError("Frame AO output has not completed.")
        self._state = CaptureState.COMPLETED
        return None

    def stop(self) -> None:
        """Stop AO then capture, retaining resources until close()."""
        if self._state in (CaptureState.STOPPED, CaptureState.CLOSED):
            return
        self._state = CaptureState.STOPPED
        try:
            _run_all(self._output.stop, self._capture.stop)
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def close(self) -> None:
        """Close AO then capture, attempting both even if one cleanup fails."""
        if self._state is CaptureState.CLOSED:
            return
        self._state = CaptureState.CLOSED
        _run_all(self._output.close, self._capture.close)

    def _require_state(self, expected: CaptureState) -> None:
        """Reject lifecycle operations outside their required state."""
        if self._state is not expected:
            raise RuntimeError(
                f"Frame must be {expected.value}; it is {self._state.value}."
            )

    def _close_after_failure(self, failure: BaseException) -> None:
        """Attempt both cleanups without replacing the primary error."""
        try:
            self.close()
        except BaseException as cleanup:
            failure.add_note(f"Frame cleanup also failed: {cleanup!r}")
            for note in getattr(cleanup, "__notes__", ()):
                failure.add_note(note)
