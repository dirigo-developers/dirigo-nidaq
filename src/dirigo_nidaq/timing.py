"""Plan record-trigger edges on a finite NI analog-output sample grid."""

from dataclasses import dataclass

from dirigo.hardware.capture import CaptureSettings
from dirigo.hardware.waveform import AngularWaveform


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordTriggerPlan:
    """Place equally spaced record-trigger edges within an AO waveform.

    capture_settings supplies the finite record count. waveform supplies the
    AO sample rate and length. interval_samples and first_trigger_sample use
    AO samples, independently of the digitizer's sample rate or record length.

    Edge i coincides nominally with AO sample
    first_trigger_sample + i * interval_samples. Index zero denotes the first
    waveform sample. Every requested edge must lie within the waveform.

    This is a pure timing description. It neither reserves resources nor
    configures routing, pulse widths, or signal polarity. It does not establish
    measured timestamps, guarantee digitizer readiness between triggers, or
    require the final record to finish before AO output ends.
    """

    capture_settings: CaptureSettings
    waveform: AngularWaveform
    interval_samples: int
    first_trigger_sample: int = 0

    def __post_init__(self) -> None:
        """Require a finite triggered capture and integer edges within AO."""
        if not isinstance(self.capture_settings, CaptureSettings):
            raise TypeError("capture_settings must be CaptureSettings.")
        if not isinstance(self.waveform, AngularWaveform):
            raise TypeError("waveform must be an AngularWaveform.")
        if self.capture_settings.batch_count is None:
            raise ValueError("Record trigger planning requires a finite capture.")
        if self.capture_settings.record_trigger is None:
            raise ValueError("Capture settings must specify a record_trigger.")

        for name in ("interval_samples", "first_trigger_sample"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer.")

        if self.interval_samples <= 0:
            raise ValueError("interval_samples must be positive.")
        if self.first_trigger_sample < 0:
            raise ValueError("first_trigger_sample must be nonnegative.")

        last = (
            self.first_trigger_sample + (self.record_count - 1) * self.interval_samples
        )
        if last >= self.waveform.samples.size:
            raise ValueError("Every record trigger must lie within the waveform.")

    @property
    def record_count(self) -> int:
        """Total trigger count across all requested batches."""
        batch_count = self.capture_settings.batch_count
        assert batch_count is not None
        return self.capture_settings.records_per_batch * batch_count

    @property
    def sample_indices(self) -> range:
        """Active-edge positions in AO samples, without allocating an array."""
        return range(
            self.first_trigger_sample,
            self.first_trigger_sample + self.record_count * self.interval_samples,
            self.interval_samples,
        )
