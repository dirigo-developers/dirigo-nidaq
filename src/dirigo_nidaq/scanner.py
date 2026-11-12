from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata

import nidaqmx
from dirigo import units
from dirigo.hardware.analog import angle_voltage_coefficients
from dirigo.hardware.metadata import DriverIdentity
from dirigo.hardware.scanner import ResonantScanner, ResonantScannerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class NIResonantScannerConfig(ResonantScannerConfig):
    """Add NI connection and vendor command specifications to scanner parameters.

    All inherited fields remain at the same configuration-table level.

    Attributes:
        amplitude_control_channel: Nonempty NI analog-output channel name.
            Hardware availability is checked by the concrete driver.
        command_voltage_range: Vendor-specified scanner controller input range.
            Its minimum maps to zero amplitude and maximum to max_amplitude.

    Raises:
        TypeError: An initialization field has an inappropriate type.
        ValueError: Inherited parameters are invalid, the channel name is blank,
            or finite nominal scaling coefficients cannot be represented.

    Validation does not import an SDK or access hardware.
    """

    amplitude_control_channel: str = field(
        metadata={
            "description": "NI analog-output channel controlling scan amplitude.",
        }
    )
    command_voltage_range: units.VoltageRange = field(
        metadata={
            "description": "Vendor-specified scanner amplitude-command voltage range.",
            "display_unit": "V",
        }
    )

    def __post_init__(self) -> None:
        """Validate inherited parameters and the NI analog-output configuration."""
        ResonantScannerConfig.__post_init__(self)

        if not isinstance(self.amplitude_control_channel, str):
            raise TypeError("amplitude_control_channel must be a string.")
        if not self.amplitude_control_channel.strip():
            raise ValueError("amplitude_control_channel must not be blank.")

        angle_voltage_coefficients(
            units.AngleRange("0 deg", self.max_amplitude),
            self.command_voltage_range,
        )


@contextmanager
def _output_task(config: NIResonantScannerConfig) -> Generator[nidaqmx.Task]:
    """Configure an AO task and release it when the operation finishes."""
    task = nidaqmx.Task()

    try:
        task.ao_channels.add_ao_voltage_chan(
            physical_channel=config.amplitude_control_channel,
            min_val=float(config.command_voltage_range.min),
            max_val=float(config.command_voltage_range.max),
        )
        yield task
    except BaseException as failure:
        try:
            task.close()
        except BaseException as cleanup_failure:
            failure.add_note(f"NI task cleanup also failed: {cleanup_failure!r}")
        raise
    else:
        task.close()


class NIResonantScanner(ResonantScanner):
    """Control resonant optical amplitude through one NI analog-output channel.

    Construction configures and clears a task without writing a sample.
    Each amplitude command creates, writes, and clears its own on-demand task.

    No settling wait or idle-amplitude policy is applied. Output behavior
    after task release depends on the NI device and driver.

    Calls must not overlap. NI driver exceptions propagate to the caller.
    """

    config_type = NIResonantScannerConfig

    def __init__(self, config: NIResonantScannerConfig) -> None:
        """Store nominal scaling and check channel configuration.

        Args:
            config: Scanner specifications and NI connection parameters.

        Raises:
            TypeError: config is not an NIResonantScannerConfig.

        NI task creation, configuration, and cleanup errors propagate.
        """
        if not isinstance(config, NIResonantScannerConfig):
            raise TypeError("config must be an NIResonantScannerConfig.")

        super().__init__(config)
        self._ni_config = config
        self._gain, self._offset = angle_voltage_coefficients(
            units.AngleRange("0 deg", config.max_amplitude),
            config.command_voltage_range,
        )

        with _output_task(config):
            pass

    @property
    def config(self) -> NIResonantScannerConfig:
        """Physical scanner specifications and NI connection parameters."""
        return self._ni_config

    @property
    def driver_identity(self) -> DriverIdentity:
        """Identify the implementation and installed driver distribution."""
        try:
            version = importlib_metadata.version("dirigo-nidaq")
        except importlib_metadata.PackageNotFoundError:
            version = None

        return DriverIdentity(
            implementation=super().driver_identity.implementation,
            distribution="dirigo-nidaq",
            version=version,
        )

    def _write_amplitude(self, value: units.Angle) -> None:
        """Write one validated amplitude using nominal voltage scaling."""
        voltage = float(value) * self._gain + self._offset

        # The base class already validated the amplitude. Bound endpoint
        # roundoff to the configured command-voltage interval.
        voltage = min(
            max(voltage, float(self.config.command_voltage_range.min)),
            float(self.config.command_voltage_range.max),
        )

        with _output_task(self.config) as task:
            if task.write(voltage, auto_start=True) != 1:
                raise RuntimeError("NI task did not accept exactly one sample.")

    def _close(self) -> None:
        """Finish closure; individual commands release their own tasks."""
        return

    def _snapshot_settings(self) -> dict[str, object]:
        """Include the NI connection and nominal command-voltage range."""
        settings = super()._snapshot_settings()
        settings.update(
            {
                "amplitude_control_channel": (self.config.amplitude_control_channel),
                "command_voltage_range": {
                    "min": self.config.command_voltage_range.min,
                    "max": self.config.command_voltage_range.max,
                },
            }
        )
        return settings
