"""Test flat NI resonant configuration without importing an NI SDK."""

from dataclasses import replace
from pathlib import Path

import pytest
from dirigo import units
from dirigo.hardware.config import load_hardware
from dirigo.hardware.device import Device
from dirigo.hardware.scanner import ResonantScannerConfig

from dirigo_nidaq.scanner import NIResonantScannerConfig


@pytest.fixture
def config() -> NIResonantScannerConfig:
    """Provide representative vendor and connection specifications."""
    return NIResonantScannerConfig(
        nominal_frequency=units.Frequency("7.91 kHz"),
        max_amplitude=units.Angle("26 deg"),
        settling_time=units.Time("100 ms"),
        frequency_tolerance=0.003,
        amplitude_control_channel="Dev1/ao0",
        command_voltage_range=units.VoltageRange("0 V", "5 V"),
    )


def test_loads_flat_toml_with_inherited_parameters(tmp_path: Path) -> None:
    """Load generic and NI-specific fields into one validated configuration."""

    class ConfigOnlyDevice(Device):
        """Receive configuration without accessing hardware."""

        config_type = NIResonantScannerConfig

        def __init__(self, config: NIResonantScannerConfig) -> None:
            self.config = config

        def close(self) -> None:
            return

    path = tmp_path / "system_config.toml"
    path.write_text(
        """
        [devices.fast_scanner]
        driver = "ni_resonant"

        [devices.fast_scanner.config]
        amplitude_control_channel = "Dev1/ao0"
        command_voltage_range = {min = "0 V", max = "5 V"}
        nominal_frequency = "7.91 kHz"
        max_amplitude = "26 deg"
        settling_time = "100 ms"
        frequency_tolerance = 0.003
        """,
        encoding="utf-8",
    )
    hardware = load_hardware(path, drivers={"ni_resonant": ConfigOnlyDevice})

    try:
        config = hardware.require("fast_scanner", ConfigOnlyDevice).config
        assert isinstance(config, ResonantScannerConfig)
        assert config.amplitude_control_channel == "Dev1/ao0"
        assert config.command_voltage_range == units.VoltageRange("0 V", "5 V")
        assert config.nominal_frequency == units.Frequency("7910 Hz")
        assert config.max_amplitude == units.Angle("26 deg")
        assert config.settling_time == units.Time("0.1 s")
        assert config.frequency_tolerance == 0.003
    finally:
        hardware.close()


@pytest.mark.parametrize("value", [None, True, 1])
def test_channel_must_be_a_string(
    config: NIResonantScannerConfig,
    value: object,
) -> None:
    """Reject incorrectly typed channel names in direct model construction."""
    with pytest.raises(TypeError, match="amplitude_control_channel"):
        replace(config, amplitude_control_channel=value)


@pytest.mark.parametrize("value", ["", "   "])
def test_channel_must_not_be_blank(
    config: NIResonantScannerConfig,
    value: str,
) -> None:
    """Require a meaningful channel name before driver construction."""
    with pytest.raises(ValueError, match="amplitude_control_channel"):
        replace(config, amplitude_control_channel=value)


def test_inherited_validation_still_runs(config: NIResonantScannerConfig) -> None:
    """Chain generic physical-parameter validation through the subclass."""
    with pytest.raises(ValueError, match="nominal_frequency"):
        replace(config, nominal_frequency=units.Frequency("-1 Hz"))

    with pytest.raises(ValueError, match="max_amplitude"):
        replace(config, max_amplitude=units.Angle("0 deg"))


def test_command_range_requires_voltage_bounds(
    config: NIResonantScannerConfig,
) -> None:
    """Reject a range of a different physical dimension."""
    with pytest.raises(TypeError, match="command_voltage_range"):
        replace(config, command_voltage_range=units.AngleRange("-1 deg", "1 deg"))


def test_command_range_must_support_finite_scaling(
    config: NIResonantScannerConfig,
) -> None:
    """Reject unbounded command-voltage specifications."""
    with pytest.raises(ValueError, match="endpoints"):
        replace(config, command_voltage_range=units.VoltageRange("0 V", float("inf")))
