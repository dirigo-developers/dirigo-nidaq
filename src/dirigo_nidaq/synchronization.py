"""Record-triggered raster synchronization using NI counter and AO tasks."""

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from dirigo import units
from dirigo.hardware.capture import CaptureSettings
from dirigo.hardware.capture_session import CaptureState, DigitizerCaptureSession
from dirigo.hardware.raster_frame_sync import (
    RasterFrameSync,
    RasterFrameSyncSession,
    RasterFrameSyncState,
)
from dirigo.hardware.scanner import (
    BeamScanner,
    GalvoScanner,
    ResonantScanner,
)
from dirigo.hardware.timing import ExternalSampleClock, SignalEdge
from dirigo.hardware.waveform import AngularWaveform

from dirigo_nidaq.frame_clock import NIFrameClockSession
from dirigo_nidaq.retriggered_output import NIRetriggeredGalvoOutputSession
from dirigo_nidaq.scanner import NIGalvoScanner

_COUNTER_PATTERN = re.compile(
    r"/?([A-Za-z0-9_-]+)/ctr([0-9]+)",
    flags=re.IGNORECASE,
)


def _run_all(*actions: Callable[[], None]) -> None:
    """Attempt every action, preserving the first failure."""
    failure: BaseException | None = None

    for action in actions:
        try:
            action()
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note(f"Synchronization cleanup also failed: {exc!r}")
                for note in getattr(exc, "__notes__", ()):
                    failure.add_note(note)

    if failure is not None:
        raise failure


@dataclass(frozen=True, slots=True, kw_only=True)
class NIRasterFrameSyncConfig:
    """Describe the installed NI record-divider connections.

    record_clock_input receives one edge per acquired record and must remain
    quiet outside capture. This implementation assumes one record per
    resonant-scanner period.

    max_record_rate is a conservative upper bound on those events, not the
    ADC sampling rate.

    frame_clock_output optionally exports the counter signal physically.
    Same-device AO uses the counter's internal route.
    """

    record_clock_input: str
    max_record_rate: units.SampleRate
    frame_counter_channel: str
    record_clock_edge: str = "rising"
    frame_clock_output: str = ""

    def __post_init__(self) -> None:
        """Validate installation descriptions without accessing NI hardware."""
        for name in (
            "record_clock_input",
            "frame_counter_channel",
            "record_clock_edge",
            "frame_clock_output",
        ):
            value = getattr(self, name)

            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string.")

            if value != value.strip():
                raise ValueError(f"{name} must not have edge whitespace.")

        if not self.record_clock_input:
            raise ValueError("record_clock_input must not be empty.")

        if _COUNTER_PATTERN.fullmatch(self.frame_counter_channel) is None:
            raise ValueError(
                "frame_counter_channel must name an NI counter, e.g. Dev1/ctr0."
            )

        SignalEdge(self.record_clock_edge)

        if not isinstance(self.max_record_rate, units.SampleRate):
            raise TypeError("max_record_rate must be a SampleRate.")

        rate = float(self.max_record_rate)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("max_record_rate must be finite and positive.")


class NIRasterFrameSync(RasterFrameSync):
    """Create record-triggered synchronization sessions using NI timing.

    The initial supported arrangement is a resonant fast scanner, an NI
    slow galvo, and a digitizer accepting independently triggered records.

    This backend does not import or configure a vendor digitizer SDK.
    Capture settings must establish the required record-event output and
    trigger inputs.

    Scanner amplitude, initial positioning, settling, and waveform planning
    are performed before starting synchronization.
    """

    config_type = NIRasterFrameSyncConfig

    def __init__(self, config: NIRasterFrameSyncConfig) -> None:
        """Store installation settings without hardware operations."""
        if not isinstance(config, NIRasterFrameSyncConfig):
            raise TypeError("config must be an NIRasterFrameSyncConfig.")

        self._config = config
        self._closed = False

    def create_session(
        self,
        *,
        fast_scanner: BeamScanner,
        slow_scanner: BeamScanner,
        capture: DigitizerCaptureSession,
    ) -> RasterFrameSyncSession:
        """Validate this arrangement and return a new passive session."""
        if self._closed:
            raise RuntimeError("Synchronization backend is closed.")

        if not isinstance(fast_scanner, ResonantScanner):
            raise NotImplementedError(
                "NI raster synchronization currently requires a resonant fast scanner."
            )

        if not isinstance(slow_scanner, NIGalvoScanner):
            raise TypeError(
                "NI raster synchronization requires an NIGalvoScanner "
                "for the slow axis."
            )

        if not isinstance(capture, DigitizerCaptureSession):
            raise TypeError("capture must be a DigitizerCaptureSession.")

        if capture.state is not CaptureState.NEW:
            raise RuntimeError("Capture must be NEW when binding synchronization.")

        match = _COUNTER_PATTERN.fullmatch(self._config.frame_counter_channel)
        assert match is not None

        counter_device = match[1].casefold()
        output_device = (
            slow_scanner.config.angle_control_channel.lstrip("/")
            .split("/", 1)[0]
            .casefold()
        )

        if counter_device != output_device:
            raise ValueError(
                "The frame counter and slow scanner must use the same "
                "NI device for internal routing."
            )

        expected_maximum = float(fast_scanner.config.nominal_frequency) * (
            1 + fast_scanner.config.frequency_tolerance
        )

        if float(self._config.max_record_rate) < expected_maximum:
            raise ValueError(
                "max_record_rate must cover the resonant scanner's "
                "declared frequency tolerance."
            )

        return _RecordTriggeredRasterFrameSyncSession(
            config=self._config,
            slow_scanner=slow_scanner,
            capture=capture,
        )

    def close(self) -> None:
        """Prevent new sessions; existing sessions must already be closed."""
        self._closed = True


class _RecordTriggeredRasterFrameSyncSession(RasterFrameSyncSession):
    """Own counter and AO sessions while borrowing capture and scanners.

    prepare(): capture, counter, then AO.
    start(): AO, counter, then capture.
    stop(): AO, counter, then capture.
    close(): stop participants and close AO and counter resources.

    Capture is never read or closed here. The worker closes it separately.

    Counter startup phase and scanner response delay are not inferred from
    logical record indices. Nominal waveform duration is checked, but the
    caller must plan the required retrigger margin and image alignment.
    """

    def __init__(
        self,
        *,
        config: NIRasterFrameSyncConfig,
        slow_scanner: NIGalvoScanner,
        capture: DigitizerCaptureSession,
    ) -> None:
        """Retain borrowed participants without touching hardware."""
        self._config = config
        self._slow_scanner = slow_scanner
        self._capture = capture

        self._state = RasterFrameSyncState.NEW
        self._records_per_frame = 0
        self._frame_count: int | None = None
        self._clock: NIFrameClockSession | None = None
        self._output: NIRetriggeredGalvoOutputSession | None = None
        self._capture_preparation_attempted = False

    @property
    def state(self) -> RasterFrameSyncState:
        """Current synchronization state, independent of capture EOF."""
        return self._state

    def configure(
        self,
        *,
        records_per_frame: int,
        frame_count: int | None,
        waveforms: Mapping[GalvoScanner, AngularWaveform],
    ) -> None:
        """Construct passive counter and output sessions for one run."""
        self._require_state(RasterFrameSyncState.NEW)

        if type(records_per_frame) is not int:
            raise TypeError("records_per_frame must be an integer.")
        if records_per_frame < 4:
            raise ValueError("records_per_frame must be at least four.")

        if frame_count is not None:
            if type(frame_count) is not int:
                raise TypeError("frame_count must be an integer or None.")
            if frame_count <= 0:
                raise ValueError("frame_count must be positive.")

        if not isinstance(waveforms, Mapping):
            raise TypeError("waveforms must be a mapping.")

        bindings = dict(waveforms)
        if len(bindings) != 1 or self._slow_scanner not in bindings:
            raise ValueError(
                "This arrangement requires exactly one waveform "
                "targeting the bound slow scanner."
            )

        waveform = bindings[self._slow_scanner]
        self._slow_scanner.validate_waveform(waveform)

        duration = waveform.samples.size / float(waveform.sample_rate)
        minimum_interval = records_per_frame / float(self._config.max_record_rate)

        if duration >= minimum_interval:
            raise ValueError(
                "The slow-scanner waveform must finish before the "
                "earliest next frame trigger."
            )

        clock = NIFrameClockSession(
            records_per_frame=records_per_frame,
            counter_channel=self._config.frame_counter_channel,
            record_clock=ExternalSampleClock(
                source=self._config.record_clock_input,
                max_rate=self._config.max_record_rate,
                active_edge=SignalEdge(self._config.record_clock_edge),
            ),
            export_terminal=self._config.frame_clock_output or None,
        )
        output = NIRetriggeredGalvoOutputSession(
            bindings,
            start_trigger=clock.trigger,
            frame_count=frame_count,
        )

        self._records_per_frame = records_per_frame
        self._frame_count = frame_count
        self._clock = clock
        self._output = output
        self._state = RasterFrameSyncState.CONFIGURED

    def _validate_capture_geometry(self, settings: CaptureSettings) -> None:
        """Check frame grouping before reserving any hardware."""
        if settings.record_trigger is None:
            raise ValueError(
                "Record-triggered synchronization requires a capture record_trigger."
            )

        batches = settings.batch_count
        frames = self._frame_count

        if frames is None:
            if batches is not None:
                raise ValueError(
                    "Unlimited synchronization requires unlimited capture."
                )
            return

        if batches is None:
            raise ValueError("Finite synchronization requires finite capture.")

        records = settings.records_per_batch * batches
        expected = self._records_per_frame * frames

        if records != expected:
            raise ValueError(
                "Capture record count does not match the configured frames."
            )

    def prepare(self) -> None:
        """Prepare capture, counter, and AO without starting acquisition."""
        self._require_state(RasterFrameSyncState.CONFIGURED)

        if self._capture.state is not CaptureState.NEW:
            raise RuntimeError("Capture must be NEW before preparation.")

        # Missing or incompatible capture configuration remains correctable.
        settings = self._capture.settings
        self._validate_capture_geometry(settings)

        clock = self._clock
        output = self._output
        assert clock is not None
        assert output is not None

        try:
            self._capture_preparation_attempted = True
            self._capture.prepare()

            layout = self._capture.layout
            if (
                layout.records_per_batch != settings.records_per_batch
                or layout.record_length != settings.record_length
            ):
                raise ValueError(
                    "Prepared capture layout differs from configured geometry."
                )

            clock.prepare()
            output.prepare()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

        self._state = RasterFrameSyncState.PREPARED

    def start(self) -> None:
        """Arm AO and the counter before starting capture."""
        self._require_state(RasterFrameSyncState.PREPARED)

        clock = self._clock
        output = self._output
        assert clock is not None
        assert output is not None

        self._state = RasterFrameSyncState.STARTED

        try:
            output.start()
            clock.start()
            self._capture.start()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def _stop_participants(self) -> None:
        """Attempt every relevant stop in dependency order."""
        actions: list[Callable[[], None]] = []

        if self._output is not None:
            actions.append(self._output.stop)

        if self._clock is not None:
            actions.append(self._clock.stop)

        if self._capture_preparation_attempted:
            actions.append(self._capture.stop)

        _run_all(*actions)

    def stop(self) -> None:
        """Cancel activity without waiting for output completion."""
        if self._state in (
            RasterFrameSyncState.STOPPED,
            RasterFrameSyncState.CLOSED,
        ):
            return

        self._state = RasterFrameSyncState.STOPPED

        try:
            self._stop_participants()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def close(self) -> None:
        """Stop participants and release owned sessions, leaving capture open."""
        if self._state is RasterFrameSyncState.CLOSED:
            return

        already_stopped = self._state is RasterFrameSyncState.STOPPED
        self._state = RasterFrameSyncState.CLOSED

        actions: list[Callable[[], None]] = []

        if not already_stopped:
            actions.append(self._stop_participants)

        if self._output is not None:
            actions.append(self._output.close)

        if self._clock is not None:
            actions.append(self._clock.close)

        _run_all(*actions)

    def _require_state(self, expected: RasterFrameSyncState) -> None:
        """Reject lifecycle operations without changing a usable session."""
        if self._state is not expected:
            raise RuntimeError(
                f"Synchronization must be {expected.value}; it is {self._state.value}."
            )

    def _close_after_failure(self, failure: BaseException) -> None:
        """Attempt cleanup without replacing the primary operation failure."""
        try:
            self.close()
        except BaseException as cleanup:
            failure.add_note(f"Synchronization cleanup also failed: {cleanup!r}")
            for note in getattr(cleanup, "__notes__", ()):
                failure.add_note(note)
