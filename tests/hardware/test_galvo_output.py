"""Exercise finite galvo output on explicitly configured NI hardware."""

import math
import tomllib
from contextlib import ExitStack
from pathlib import Path

import pytest
from dirigo import units
from dirigo.hardware.config import load_hardware
from dirigo.hardware.scanner import GalvoScanner
from dirigo.hardware.waveform import AngularWaveform

from dirigo_nidaq.output import NIGalvoOutputSession
from dirigo_nidaq.scanner import NIGalvoScanner


@pytest.mark.hardware
def test_ni_galvo_output() -> None:
    """Run a configured finite waveform group through the installed NI driver.

    Load scanner aliases, angular samples, timing, and final static angles from
    hardware_local.toml. Validate the complete group and every final angle
    before creating a task. This exercises driver integration and cleanup;
    it makes no voltage measurements or claims about physical mirror motion.

    Cleanup attempts session closure first, then every final static command,
    then hardware closure. Each cleanup is attempted even if another fails.
    """
    config_path = Path(__file__).with_name("hardware_local.toml")
    with config_path.open("rb") as stream:
        document = tomllib.load(stream)

    settings = document.get("test", {}).get("galvo")
    if settings is None:
        pytest.skip("Configure [test.galvo] in hardware_local.toml to run this test.")

    sample_rate = units.SampleRate(settings["sample_rate"])
    timeout = float(units.Time(settings["timeout"]))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("The galvo test timeout must be finite and positive.")

    with ExitStack() as cleanup:
        hardware = load_hardware(config_path)
        cleanup.callback(hardware.close)

        waveforms: dict[GalvoScanner, AngularWaveform] = {}
        final_angles: list[tuple[NIGalvoScanner, units.Angle]] = []

        for entry in settings["axes"]:
            scanner = hardware.require(entry["scanner"], NIGalvoScanner)
            if scanner in waveforms:
                raise ValueError("Each scanner may appear only once in [test.galvo].")

            waveform = AngularWaveform(
                [float(units.Angle(value)) for value in entry["samples"]],
                sample_rate=sample_rate,
            )
            final_angle = units.Angle(entry["final_angle"])

            # Reuse optical-limit validation without issuing a static command.
            scanner.validate_waveform(
                AngularWaveform([float(final_angle)], sample_rate=sample_rate)
            )
            waveforms[scanner] = waveform
            final_angles.append((scanner, final_angle))

        # Construction validates waveform limits, common timing, and channels.
        session = NIGalvoOutputSession(waveforms)

        # Callbacks execute in reverse registration order. Register only after
        # all settings have passed validation, so invalid settings cause no output.
        for scanner, final_angle in reversed(final_angles):
            cleanup.callback(setattr, scanner, "angle", final_angle)
        cleanup.callback(session.close)

        session.prepare()
        session.start()
        assert session.wait(timeout), "NI galvo output did not complete before timeout."
