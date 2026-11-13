"""Finite NI analog-input and galvo-output coordination on one DAQ device."""

from collections.abc import Callable, Iterable, Mapping
from time import monotonic

from dirigo.hardware.capture import CaptureInfo, CaptureLayout, SampleArray
from dirigo.hardware.capture_session import CaptureState
from dirigo.hardware.scanner import GalvoScanner
from dirigo.hardware.timing import ExternalSampleClock
from dirigo.hardware.waveform import AngularWaveform

from dirigo_nidaq import output as ni_output
from dirigo_nidaq.digitizer import NICaptureSettings, NIDigitizer
from dirigo_nidaq.scanner import NIGalvoScanner


def _run_cleanup(actions: Iterable[Callable[[], None]]) -> None:
    """Attempt every action, preserving the first error and noting later ones."""
    failure: BaseException | None = None

    for action in actions:
        try:
            action()
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note(f"Scan cleanup also failed: {exc!r}")
                for note in getattr(exc, "__notes__", ()):
                    failure.add_note(note)

    if failure is not None:
        raise failure


class NISynchronizedScan:
    """Own one finite AI/AO run while borrowing its hardware devices.

    Use internally clocked NI input and galvo outputs on the same NI device,
    with equal sample rates and total sample counts. Waveforms cover the
    entire capture, across every record and batch. Rate coercion that would
    retime the waveforms is rejected during preparation.

    prepare() prepares AI, then configures AO to consume its sample clock.
    start() arms AO before starting AI. An optional input start trigger gates
    the AI run; AO waits for its sample-clock edges without another trigger.

    read_into() delivers complete capture batches. None establishes both
    capture exhaustion and observed AO completion. Capture and completion
    timeouts retain the run for retry.

    stop() retains reservations; close() releases both sessions, leaving the
    borrowed devices open. Calls must be serial. No parking command is issued.
    """

    def __init__(
        self,
        digitizer: NIDigitizer,
        settings: NICaptureSettings,
        waveforms: Mapping[GalvoScanner, AngularWaveform],
    ) -> None:
        """Validate and copy the run description without reserving resources."""
        if not isinstance(digitizer, NIDigitizer):
            raise TypeError("digitizer must be an NIDigitizer.")
        if not isinstance(settings, NICaptureSettings):
            raise TypeError("settings must be NICaptureSettings.")

        batch_count = settings.batch_count
        if (
            settings.external_clock is not None
            or settings.record_trigger is not None
            or batch_count is None
        ):
            raise NotImplementedError(
                "Scan requires finite, internally clocked AI with consecutive records."
            )

        if not isinstance(waveforms, Mapping):
            raise TypeError("waveforms must be a mapping.")
        bindings = dict(waveforms)
        if not bindings:
            raise ValueError("At least one scanner waveform is required.")

        count = settings.record_length * settings.records_per_batch * batch_count
        device = digitizer.config.expanded_channels[0].split("/", 1)[0].casefold()

        for axis, waveform in bindings.items():
            if not isinstance(axis, NIGalvoScanner):
                raise TypeError("Scan axes must be NIGalvoScanner instances.")

            axis.validate_waveform(waveform)
            channel_device = (
                axis.config.angle_control_channel.lstrip("/")
                .split("/", 1)[0]
                .casefold()
            )
            if channel_device != device:
                raise ValueError("AI and AO must use the same NI device.")
            if waveform.sample_rate != settings.sample_rate:
                raise ValueError("Waveform and capture sample_rate must match.")
            if waveform.samples.size != count:
                raise ValueError(
                    "Waveform length must equal the total capture samples."
                )

        self._settings = settings
        self._waveforms = bindings
        self._capture = digitizer.create_capture(settings)
        self._output: ni_output.NIGalvoOutputSession | None = None
        self._state = CaptureState.NEW

    @property
    def state(self) -> CaptureState:
        """Run state; COMPLETED includes observed AO completion."""
        return self._state

    @property
    def layout(self) -> CaptureLayout:
        """AI batch layout, available after AI preparation and retained on close."""
        return self._capture.layout

    def prepare(self) -> None:
        """Prepare AI and its clock consumer, rolling back partial preparation."""
        self._require_state(CaptureState.NEW)

        try:
            self._capture.prepare()
            rate = self._capture.sample_rate
            if rate is None or rate != self._settings.sample_rate:
                raise ValueError(
                    "Prepared AI rate must equal the waveform sample_rate."
                )

            self._output = ni_output.NIGalvoOutputSession(
                self._waveforms,
                external_clock=ExternalSampleClock(
                    source=self._capture.sample_clock_terminal,
                    max_rate=rate,
                ),
            )
            self._output.prepare()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

        self._state = CaptureState.PREPARED

    def start(self) -> None:
        """Arm AO before starting or arming the AI clock producer."""
        self._require_state(CaptureState.PREPARED)
        output = self._output
        assert output is not None
        self._state = CaptureState.STARTED

        try:
            output.start()
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
        """Read a batch, or confirm AO completion before returning finite EOF.

        Capture timeouts and caller argument errors retain the capture's
        usability. A fatal capture error also closes output.

        At capture EOF, the remaining timeout budget applies to AO completion.
        An AO completion timeout raises TimeoutError and can be retried with
        another read.
        """
        if self._state is not CaptureState.COMPLETED:
            self._require_state(CaptureState.STARTED)

        started = monotonic()
        try:
            info = self._capture.read_into(destination, timeout=timeout)
        except BaseException as failure:
            # The child owns argument validation and its failure-cleanup policy.
            if self._capture.state is CaptureState.CLOSED:
                self._close_after_failure(failure)
            raise

        if info is not None or self._state is CaptureState.COMPLETED:
            return info

        output = self._output
        assert output is not None
        remaining = (
            None if timeout is None else max(0.0, timeout - (monotonic() - started))
        )

        try:
            complete = output.wait(timeout=remaining)
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

        if not complete:
            raise TimeoutError("Synchronized AO output has not completed.")

        self._state = CaptureState.COMPLETED
        return None

    def stop(self) -> None:
        """Stop AO then AI, retaining both reservations until close()."""
        if self._state in (CaptureState.STOPPED, CaptureState.CLOSED):
            return

        self._state = CaptureState.STOPPED
        actions: list[Callable[[], None]] = [self._capture.stop]
        if self._output is not None:
            actions.insert(0, self._output.stop)

        try:
            _run_cleanup(actions)
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def close(self) -> None:
        """Close AO then AI, attempting both even when one cleanup fails."""
        if self._state is CaptureState.CLOSED:
            return

        self._state = CaptureState.CLOSED
        actions: list[Callable[[], None]] = [self._capture.close]
        if self._output is not None:
            actions.insert(0, self._output.close)

        _run_cleanup(actions)

    def _require_state(self, expected: CaptureState) -> None:
        """Reject lifecycle calls outside their required state."""
        if self._state is not expected:
            raise RuntimeError(
                f"Scan must be {expected.value}; it is {self._state.value}."
            )

    def _close_after_failure(self, failure: BaseException) -> None:
        """Attempt all owned cleanup without replacing an operation's error."""
        try:
            self.close()
        except BaseException as cleanup:
            failure.add_note(f"Scan cleanup also failed: {cleanup!r}")
            for note in getattr(cleanup, "__notes__", ()):
                failure.add_note(note)
