"""NI counter-triggered sequence factory for the core acquisition worker."""

from collections.abc import Mapping
from dataclasses import dataclass

from dirigo import units
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.frame_plan import RecordFramePlan
from dirigo.hardware.scanner import GalvoScanner
from dirigo.hardware.timing import ExternalSampleClock, SignalEdge
from dirigo.hardware.waveform import AngularWaveform
from dirigo.system import System

from dirigo_nidaq.triggered_sequence import NITriggeredFrameSequence


@dataclass(frozen=True, slots=True, kw_only=True)
class NITriggeredSequenceConfig:
    """Identify borrowed devices and record-clock/counter wiring.

    record_clock_input receives one active edge per acquired record.
    Capture settings must request any required digitizer trigger output.

    An empty frame_clock_export_terminal disables physical PFI export.
    Same-device AO uses the counter's internal output in either case.
    """

    digitizer: str
    record_clock_input: str
    max_record_rate: units.SampleRate
    frame_counter_channel: str
    record_clock_edge: str = "rising"
    frame_clock_export_terminal: str = ""


class NITriggeredSequenceFactory:
    """Construct passive NI-triggered runs satisfying the core contract."""

    config_type = NITriggeredSequenceConfig

    def __call__(
        self,
        config: NITriggeredSequenceConfig,
        /,
        *,
        system: System,
        plan: RecordFramePlan,
        waveforms: Mapping[str, AngularWaveform],
    ) -> NITriggeredFrameSequence:
        """Resolve borrowed devices and preserve the exact capture request."""
        digitizer = system.require_device(config.digitizer, Digitizer)
        bindings: dict[GalvoScanner, AngularWaveform] = {}

        for name, waveform in waveforms.items():
            axis = system.require_device(name, GalvoScanner)
            if axis in bindings:
                raise ValueError("Waveforms must target distinct scanners.")
            bindings[axis] = waveform

        return NITriggeredFrameSequence(
            digitizer,
            plan,
            bindings,
            record_clock=ExternalSampleClock(
                source=config.record_clock_input,
                max_rate=config.max_record_rate,
                active_edge=SignalEdge(config.record_clock_edge),
            ),
            frame_counter_channel=config.frame_counter_channel,
            frame_clock_export_terminal=(config.frame_clock_export_terminal or None),
        )


nidaq_triggered = NITriggeredSequenceFactory()
