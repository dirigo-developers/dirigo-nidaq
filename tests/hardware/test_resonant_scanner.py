import math
import tomllib
from pathlib import Path

import pytest
from dirigo import units
from dirigo.hardware.config import load_hardware

from dirigo_nidaq.scanner import NIResonantScanner


@pytest.mark.hardware
def test_ni_resonant_scanner_commands() -> None:
    """Exercise configured NI output through the installed driver."""
    config_path = Path(__file__).with_name("hardware_local.toml")

    with config_path.open("rb") as stream:
        settings = tomllib.load(stream)["test"]

    scanner_name = settings["scanner"]
    test_amplitude = units.Angle(settings["amplitude"])
    final_amplitude = units.Angle(settings["final_amplitude"])

    hardware = load_hardware(config_path)
    try:
        scanner = hardware.require(scanner_name, NIResonantScanner)

        # Validate both targets before issuing either command.
        for name, amplitude in (
            ("test_amplitude", test_amplitude),
            ("final_amplitude", final_amplitude),
        ):
            value = float(amplitude)
            maximum = float(scanner.config.max_amplitude)
            if not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(
                    f"{name} must be finite and between zero and "
                    f"max_amplitude ({scanner.config.max_amplitude})."
                )

        try:
            scanner.amplitude = test_amplitude
            assert scanner.amplitude == test_amplitude
        finally:
            scanner.amplitude = final_amplitude
    finally:
        hardware.close()
