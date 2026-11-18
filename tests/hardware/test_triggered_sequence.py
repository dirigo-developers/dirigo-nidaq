"""Exercise Alazar capture, NI frame-clock output, and retriggered galvo output."""

import math
import tomllib
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import pytest
from dirigo import load_system, units
from dirigo.hardware.capture import CaptureSettings
from dirigo.hardware.capture_session import CaptureState
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.frame_plan import RecordFramePlan
from dirigo.hardware.timing import (
    DigitalRecordTrigger,
    ExternalSampleClock,
    SignalEdge,
)
from dirigo.hardware.waveform import AngularWaveform

from dirigo_nidaq.scanner import NIGalvoScanner
from dirigo_nidaq.triggered_sequence import NITriggeredFrameSequence


@pytest.mark.hardware
def test_alazar_triggered_frame_sequence() -> None:
    """Acquire multiple frames and require completion of both NI output tasks.

    Read [test.triggered_sequence] from hardware_local.toml. The operator
    supplies running TTL record triggers to Alazar. Connect its configurable
    acquisition-active AUX output to record_clock_input. The selected NI
    counter routes its internal output to AO without another jumper.
    Import Alazar settings only after the test has been explicitly configured.

    Hold a caller-selected constant angle during acquisition. Preflight both
    angle targets and nominal timing before commanding output. Require at
    least two frames, consume every capture batch, and observe the combined
    completion within one budget beginning immediately before start(). This
    exercises our use of NI's generated-sample count across retriggers; it
    does not measure output voltage, mirror position, or electrical phase.

    Close the sequence and prepare another using the same devices, without
    starting it. All sessions close before the explicit final angle is
    commanded and before the borrowed hardware is closed. Configuration and
    SDK errors fail the test; only an absent file or test table causes a skip.
    """
    config_path = Path(__file__).with_name("hardware_local.toml")
    if not config_path.is_file():
        pytest.skip("Create hardware_local.toml to run this hardware test.")
    with config_path.open("rb") as stream:
        document = tomllib.load(stream)
    parameters = document.get("test", {}).get("triggered_sequence")
    if parameters is None:
        pytest.skip("Configure [test.triggered_sequence] in hardware_local.toml.")

    timeout = float(units.Time(parameters["timeout"]))
    settling_time = float(units.Time(parameters["settling_time"]))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("The sequence test timeout must be finite and positive.")
    if not math.isfinite(settling_time) or settling_time < 0:
        raise ValueError("settling_time must be finite and nonnegative.")
    command_amplitude = units.Angle(parameters["command_amplitude"])
    final_angle = units.Angle(parameters["final_angle"])
    ao_rate = units.SampleRate(parameters["ao_sample_rate"])
    waveform = AngularWaveform(
        float(command_amplitude) * np.linspace(-1, 1, parameters["ao_sample_count"]),
        sample_rate=ao_rate,
    )
    final_waveform = AngularWaveform([float(final_angle)], sample_rate=ao_rate)
    record_clock = ExternalSampleClock(
        source=parameters["record_clock_input"],
        max_rate=units.SampleRate(parameters["max_record_rate"]),
    )

    # Keep the optional Alazar dependency out of normal NI test imports and
    # type checking. A configured but unavailable plugin must fail the test.
    alazar = import_module("dirigo_alazar.settings")
    voltage_range = parameters["input_voltage_range"]
    request: CaptureSettings = alazar.AlazarCaptureSettings(
        sample_rate=units.SampleRate(parameters["sample_rate"]),
        record_length=parameters["record_length"],
        records_per_batch=parameters["records_per_batch"],
        batch_count=parameters["batch_count"],
        channel_indices=tuple(parameters["channel_indices"]),
        record_trigger=DigitalRecordTrigger(
            source="external", active_edge=SignalEdge(parameters["trigger_edge"])
        ),
        input_voltage_range=units.VoltageRange(
            voltage_range["min"], voltage_range["max"]
        ),
        input_impedance=units.Resistance(parameters["input_impedance"]),
        input_coupling=alazar.Coupling(parameters["input_coupling"]),
        dma_buffer_count=parameters["dma_buffer_count"],
        aux_output_mode=alazar.AuxOutputMode.ACQUISITION_ACTIVE,
    )
    plan = RecordFramePlan(
        capture_settings=request, records_per_frame=parameters["records_per_frame"]
    )
    batch_count = request.batch_count
    assert batch_count is not None
    if plan.frame_count < 2:
        raise ValueError("The sequence hardware test requires at least two frames.")
    waveform_duration = waveform.samples.size / float(ao_rate)
    shortest_frame_interval = plan.records_per_frame / float(record_clock.max_rate)
    if waveform_duration >= shortest_frame_interval:
        raise ValueError("AO waveform must be shorter than the minimum frame interval.")

    with ExitStack() as cleanup:
        system = load_system(config_path)
        cleanup.callback(system.close)
        digitizer = system.require_device(parameters["digitizer"], Digitizer)
        scanner = system.require_device(parameters["scanner"], NIGalvoScanner)
        scanner.validate_waveform(final_waveform)
        sequence = NITriggeredFrameSequence(
            digitizer,
            plan,
            {scanner: waveform},
            record_clock=record_clock,
            frame_counter_channel=parameters["frame_counter_channel"],
        )

        # Passive validation precedes output. Reverse callback order closes
        # sessions before applying the final angle and closing the devices.
        cleanup.callback(setattr, scanner, "angle", final_angle)
        cleanup.callback(sequence.close)
        scanner.angle = command_amplitude
        sleep(settling_time)
        sequence.prepare()

        layout = sequence.layout
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
        received = 0
        try:
            sequence.start()
            while True:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Sequence exceeded its configured waiting budget."
                    )
                info = sequence.read_into(samples, timeout=remaining)
                if info is None:
                    break
                received += 1
                assert received <= batch_count, "Too many capture batches."
                assert samples.shape == expected_shape
                assert samples.dtype == layout.dtype
            if monotonic() > deadline:
                raise TimeoutError("Sequence completion exceeded its waiting budget.")
        except TimeoutError as failure:
            failure.add_note(
                "Check Alazar record triggers, AUX-to-record-clock wiring, "
                "counter-to-AO routing, and AO timing margin."
            )
            raise

        assert received == sequence.batches_acquired == batch_count
        assert received * layout.records_per_batch == plan.record_count
        assert sequence.frame_count == plan.frame_count
        assert sequence.state is CaptureState.COMPLETED
        sequence.close()
        assert sequence.state is CaptureState.CLOSED
        assert not digitizer.is_active

        retry = NITriggeredFrameSequence(
            digitizer,
            plan,
            {scanner: waveform},
            record_clock=record_clock,
            frame_counter_channel=parameters["frame_counter_channel"],
        )
        cleanup.callback(retry.close)
        retry.prepare()
        assert retry.state is CaptureState.PREPARED
        assert retry.layout == layout
