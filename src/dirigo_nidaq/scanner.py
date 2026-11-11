"""Initialization parameters for NI analog-output control of resonant scanners."""

from dataclasses import dataclass, field

from dirigo import units
from dirigo.hardware.analog import angle_voltage_coefficients
from dirigo.hardware.scanner import ResonantScannerConfig


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
