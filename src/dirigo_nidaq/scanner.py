from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata

import nidaqmx
from dirigo import units
from dirigo.hardware.analog import angle_voltage_coefficients
from dirigo.hardware.metadata import DriverIdentity
from dirigo.hardware.scanner import (
    GalvoScanner,
    GalvoScannerConfig,
    ResonantScanner,
    ResonantScannerConfig,
)


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
def _output_task(
    channel: str,
    command_voltage_range: units.VoltageRange,
) -> Generator[nidaqmx.Task]:
    """Own one scalar AO task and preserve operation errors during cleanup."""
    task = nidaqmx.Task()
    try:
        task.ao_channels.add_ao_voltage_chan(
            physical_channel=channel,
            min_val=float(command_voltage_range.min),
            max_val=float(command_voltage_range.max),
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

        with _output_task(
            self.config.amplitude_control_channel,
            self.config.command_voltage_range,
        ) as task:
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


@dataclass(frozen=True, slots=True, kw_only=True)
class NIGalvoScannerConfig(GalvoScannerConfig):
    """Add NI connection and nominal voltage mapping to one galvo axis.

    All fields remain in one configuration table. With inverted=False,
    angle_range.min maps to command_voltage_range.min and angle_range.max
    maps to command_voltage_range.max. Inversion swaps those voltage endpoints.

    These are full-scale vendor specifications, independent of the angular
    extent selected for a particular scan.
    """

    angle_control_channel: str = field(
        metadata={
            "description": "NI analog-output channel controlling optical angle.",
        }
    )
    command_voltage_range: units.VoltageRange = field(
        metadata={
            "description": "Vendor-specified galvo command-voltage range.",
            "display_unit": "V",
        }
    )
    inverted: bool = field(
        default=False,
        metadata={
            "description": "Reverse the nominal optical-angle-to-voltage mapping.",
        },
    )

    def __post_init__(self) -> None:
        """Validate inherited optical limits and the NI mapping parameters."""
        GalvoScannerConfig.__post_init__(self)
        if not isinstance(self.angle_control_channel, str):
            raise TypeError("angle_control_channel must be a string.")
        if not self.angle_control_channel.strip():
            raise ValueError("angle_control_channel must not be blank.")
        if not isinstance(self.inverted, bool):
            raise TypeError("inverted must be a bool.")

        angle_voltage_coefficients(
            self.angle_range,
            self.command_voltage_range,
            inverted=self.inverted,
        )


class NIGalvoScanner(GalvoScanner):
    """Command static optical angles through one NI analog-output channel.

    Construction stores configuration and scaling without creating tasks.
    Each static command creates, writes, and closes an on-demand task.
    Timed grouped output is owned by a separate output session.

    The base class enforces optical limits, reservations, and command history.
    No settling wait or automatic parking command is applied.
    """

    config_type = NIGalvoScannerConfig

    def __init__(self, config: NIGalvoScannerConfig) -> None:
        """Store NI specifications without accessing hardware.

        Raises:
            TypeError: config is not an NIGalvoScannerConfig.
        """
        if not isinstance(config, NIGalvoScannerConfig):
            raise TypeError("config must be an NIGalvoScannerConfig.")

        super().__init__(config)
        self._ni_config = config
        self._gain, self._offset = angle_voltage_coefficients(
            config.angle_range,
            config.command_voltage_range,
            inverted=config.inverted,
        )

    @property
    def config(self) -> NIGalvoScannerConfig:
        """Physical optical limits and NI connection parameters."""
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

    def _write_angle(self, value: units.Angle) -> None:
        """Convert a validated static angle and write one voltage sample."""
        voltage = float(value) * self._gain + self._offset

        # Optical limits were already checked. Bound endpoint roundoff to
        # the configured command-voltage interval.
        voltage = min(
            max(voltage, float(self.config.command_voltage_range.min)),
            float(self.config.command_voltage_range.max),
        )
        with _output_task(
            self.config.angle_control_channel,
            self.config.command_voltage_range,
        ) as task:
            if task.write(voltage, auto_start=True) != 1:
                raise RuntimeError("NI task did not accept exactly one sample.")

    def _close(self) -> None:
        """Release no persistent tasks and issue no parking command."""
        return

    def _snapshot_settings(self) -> dict[str, object]:
        """Include connection, nominal voltage mapping, and inversion settings."""
        settings = super()._snapshot_settings()
        settings.update(
            {
                "angle_control_channel": self.config.angle_control_channel,
                "command_voltage_range": {
                    "min": self.config.command_voltage_range.min,
                    "max": self.config.command_voltage_range.max,
                },
                "inverted": self.config.inverted,
            }
        )
        return settings
