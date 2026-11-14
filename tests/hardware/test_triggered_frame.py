"""Smoke-test Alazar capture with acquisition-triggered NI galvo output."""

import math
import tomllib
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import pytest
from dirigo import units
from dirigo.hardware.capture import CaptureSettings
from dirigo.hardware.capture_session import CaptureState
from dirigo.hardware.config import load_hardware
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.timing import (
    DigitalRecordTrigger,
    DigitalStartTrigger,
    SignalEdge,
)
from dirigo.hardware.waveform import AngularWaveform

from dirigo_nidaq.scanner import NIGalvoScanner
from dirigo_nidaq.triggered_frame import NITriggeredFrame


@pytest.mark.hardware
def test_alazar_triggered_frame() -> None:
    """Capture one finite frame and observe completion of triggered constant AO.

    Read [test.triggered_frame] from hardware_local.toml. The operator supplies
    running TTL record triggers to Alazar and connects its software-configurable
    acquisition-active AUX output to the configured NI start-trigger input.
    Import Alazar settings only after this test has been explicitly configured.

    Validate both galvo targets before output, position and settle the axis,
    then prepare and run the coordinator within one acquisition waiting budget.
    Check geometry, batch count, joint completion, and session/device reuse.
    Signal amplitudes, mirror motion, and electrical phase are not measured.

    Reprepare a second frame without starting it. Cleanup closes every session
    before applying the explicit final angle and closing the borrowed devices.
    """
    config_path = Path(__file__).with_name("hardware_local.toml")
    if not config_path.is_file():
        pytest.skip("Create hardware_local.toml to run this hardware test.")

    with config_path.open("rb") as stream:
        document = tomllib.load(stream)
    parameters = document.get("test", {}).get("triggered_frame")
    if parameters is None:
        pytest.skip("Configure [test.triggered_frame] in hardware_local.toml.")

    timeout = float(units.Time(parameters["timeout"]))
    settling_time = float(units.Time(parameters["settling_time"]))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("The frame test timeout must be finite and positive.")
    if not math.isfinite(settling_time) or settling_time < 0:
        raise ValueError("settling_time must be finite and nonnegative.")

    command_angle = units.Angle(parameters["command_angle"])
    final_angle = units.Angle(parameters["final_angle"])
    ao_rate = units.SampleRate(parameters["ao_sample_rate"])
    waveform = AngularWaveform(
        np.full(parameters["ao_sample_count"], float(command_angle)),
        sample_rate=ao_rate,
    )
    final_waveform = AngularWaveform(
        [float(final_angle)],
        sample_rate=ao_rate,
    )
    start_trigger = DigitalStartTrigger(source=parameters["acquisition_active_input"])

    # Keep this optional integration dependency out of normal NI test imports
    # and type checking. Missing or outdated configured plugins must fail.
    alazar = import_module("dirigo_alazar.settings")
    voltage_range = parameters["input_voltage_range"]
    request: CaptureSettings = alazar.AlazarCaptureSettings(
        sample_rate=units.SampleRate(parameters["sample_rate"]),
        record_length=parameters["record_length"],
        records_per_batch=parameters["records_per_batch"],
        batch_count=parameters["batch_count"],
        channel_indices=tuple(parameters["channel_indices"]),
        record_trigger=DigitalRecordTrigger(
            source="external",
            active_edge=SignalEdge(parameters["trigger_edge"]),
        ),
        input_voltage_range=units.VoltageRange(
            voltage_range["min"],
            voltage_range["max"],
        ),
        input_impedance=units.Resistance(parameters["input_impedance"]),
        input_coupling=alazar.Coupling(parameters["input_coupling"]),
        dma_buffer_count=parameters["dma_buffer_count"],
        aux_output_mode=alazar.AuxOutputMode.ACQUISITION_ACTIVE,
    )
    batch_count = request.batch_count
    if batch_count is None:
        raise ValueError("The frame hardware test requires finite batch_count.")

    with ExitStack() as cleanup:
        hardware = load_hardware(config_path)
        cleanup.callback(hardware.close)

        digitizer = hardware.require(parameters["digitizer"], Digitizer)
        scanner = hardware.require(parameters["scanner"], NIGalvoScanner)
        scanner.validate_waveform(final_waveform)

        capture = digitizer.create_capture(request)
        cleanup.callback(capture.close)
        frame = NITriggeredFrame(
            capture,
            {scanner: waveform},
            start_trigger=start_trigger,
        )

        # All passive target/run validation precedes either static command.
        cleanup.callback(setattr, scanner, "angle", final_angle)
        cleanup.callback(frame.close)

        scanner.angle = command_angle
        sleep(settling_time)
        frame.prepare()

        layout = frame.layout
        expected_shape = (
            request.records_per_batch,
            request.record_length,
            len(request.channel_indices),
        )
        assert layout.shape == expected_shape
        assert layout.channel_indices == request.channel_indices
        assert layout.dtype.kind == "i"
        assert layout.dtype.isnative
        samples = np.empty(layout.shape, dtype=layout.dtype)

        deadline = monotonic() + timeout
        frame.start()
        received = 0

        while True:
            remaining = deadline - monotonic()
            assert remaining > 0, "Frame exceeded its configured waiting budget."

            try:
                info = frame.read_into(samples, timeout=remaining)
            except TimeoutError as exc:
                exc.add_note(
                    "Check TTL record triggers and the Alazar AUX-to-NI connection."
                )
                raise

            if info is None:
                break

            received += 1
            assert received <= batch_count, "Capture delivered too many batches."
            assert samples.shape == expected_shape
            assert samples.dtype == layout.dtype

        assert received == frame.batches_acquired == batch_count
        assert frame.state is CaptureState.COMPLETED

        frame.close()
        assert frame.state is CaptureState.CLOSED
        assert not digitizer.is_active

        retry_capture = digitizer.create_capture(request)
        cleanup.callback(retry_capture.close)
        retry = NITriggeredFrame(
            retry_capture,
            {scanner: waveform},
            start_trigger=start_trigger,
        )
        cleanup.callback(retry.close)

        retry.prepare()
        assert retry.state is CaptureState.PREPARED
        assert retry.layout == layout
