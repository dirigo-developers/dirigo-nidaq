"""Smoke-test finite NI synchronization on explicitly configured hardware."""

import math
import tomllib
from contextlib import ExitStack
from pathlib import Path
from time import monotonic

import numpy as np
import pytest
from dirigo import units
from dirigo.hardware.capture_session import CaptureState
from dirigo.hardware.config import load_hardware
from dirigo.hardware.waveform import AngularWaveform
from nidaqmx.constants import TerminalConfiguration

from dirigo_nidaq.digitizer import NICaptureSettings, NIDigitizer
from dirigo_nidaq.scanner import NIGalvoScanner
from dirigo_nidaq.synchronized import NISynchronizedScan


@pytest.mark.hardware
def test_ni_synchronized_scan() -> None:
    """Run finite capture with constant galvo output through installed DAQmx.

    Read device aliases and run parameters from [test.synchronized] in
    hardware_local.toml. Validate both commanded and final angles before
    issuing output. Check batch geometry, integer storage, batch count,
    and joint completion without inspecting measured signal values.

    Close the run and prepare another scan on the same devices, without
    starting it, to verify resource reuse. Cleanup attempts session closure,
    the configured final angle, and Hardware closure in that order.
    """
    config_path = Path(__file__).with_name("hardware_local.toml")
    if not config_path.is_file():
        pytest.skip("Create hardware_local.toml to run this hardware test.")

    with config_path.open("rb") as stream:
        document = tomllib.load(stream)

    parameters = document.get("test", {}).get("synchronized")
    if parameters is None:
        pytest.skip("Configure [test.synchronized] in hardware_local.toml.")

    timeout = float(units.Time(parameters["timeout"]))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("The synchronized test timeout must be finite and positive.")

    requested_range = parameters["input_voltage_range"]
    request = NICaptureSettings(
        sample_rate=units.SampleRate(parameters["sample_rate"]),
        record_length=parameters["record_length"],
        records_per_batch=parameters["records_per_batch"],
        batch_count=parameters["batch_count"],
        channel_indices=tuple(parameters["channel_indices"]),
        input_voltage_range=units.VoltageRange(
            requested_range["min"],
            requested_range["max"],
        ),
        terminal_config=TerminalConfiguration[parameters["terminal_config"]],
    )
    batch_count = request.batch_count
    if batch_count is None:
        raise ValueError("The synchronized hardware test requires finite batch_count.")

    command_angle = units.Angle(parameters["command_angle"])
    final_angle = units.Angle(parameters["final_angle"])
    sample_count = request.records_per_batch * request.record_length * batch_count

    waveform = AngularWaveform(
        np.full(sample_count, float(command_angle)),
        sample_rate=request.sample_rate,
    )
    final_waveform = AngularWaveform(
        [float(final_angle)],
        sample_rate=request.sample_rate,
    )

    with ExitStack() as cleanup:
        hardware = load_hardware(config_path)
        cleanup.callback(hardware.close)

        digitizer = hardware.require(parameters["digitizer"], NIDigitizer)
        scanner = hardware.require(parameters["scanner"], NIGalvoScanner)
        scanner.validate_waveform(final_waveform)
        scan = NISynchronizedScan(digitizer, request, {scanner: waveform})

        # Register physical cleanup after every command target and the
        # complete scan description have passed passive validation.
        cleanup.callback(setattr, scanner, "angle", final_angle)
        cleanup.callback(scan.close)

        scan.prepare()
        layout = scan.layout
        expected_shape = (
            request.records_per_batch,
            request.record_length,
            len(request.channel_indices),
        )
        assert layout.shape == expected_shape
        assert layout.channel_indices == request.channel_indices
        assert layout.dtype.kind in "iu"
        assert layout.dtype.isnative
        samples = np.empty(layout.shape, dtype=layout.dtype)

        deadline = monotonic() + timeout
        scan.start()
        received = 0

        while True:
            remaining = deadline - monotonic()
            assert remaining > 0, "Synchronized scan exceeded its configured timeout."

            info = scan.read_into(samples, timeout=remaining)
            if info is None:
                break

            received += 1
            assert received <= batch_count, "Capture delivered too many batches."
            assert samples.shape == expected_shape
            assert samples.dtype == layout.dtype

        assert received == batch_count
        assert scan.state is CaptureState.COMPLETED

        scan.close()
        assert scan.state is CaptureState.CLOSED

        retry = NISynchronizedScan(digitizer, request, {scanner: waveform})
        cleanup.callback(retry.close)
        retry.prepare()
        assert retry.state is CaptureState.PREPARED
