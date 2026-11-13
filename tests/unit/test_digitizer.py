"""Test NI input preparation without native DAQmx libraries."""

from dataclasses import dataclass, replace
from importlib import metadata as importlib_metadata
from typing import cast
from unittest.mock import Mock, call

import nidaqmx
import numpy as np
import pytest
from dirigo import units
from dirigo.hardware.capture import CaptureSettings, SampleArray
from dirigo.hardware.capture_session import CaptureState
from dirigo.hardware.config import parse_config
from dirigo.hardware.timing import (
    DigitalStartTrigger,
    ExternalSampleClock,
    SignalEdge,
)
from nidaqmx import constants, errors, stream_readers

from dirigo_nidaq.digitizer import (
    NICaptureSession,
    NICaptureSettings,
    NIDigitizer,
    NIDigitizerConfig,
)


@dataclass(frozen=True)
class FakeNI:
    """Expose patched SDK factories, their objects, and native channel metadata."""

    factory: Mock
    task: Mock
    channel: Mock
    reader_factory: Mock
    reader: Mock


@pytest.fixture(autouse=True)
def ni(monkeypatch: pytest.MonkeyPatch) -> FakeNI:
    """Replace task and reader creation without invoking native DAQmx."""
    channel = Mock()
    channel.ai_raw_samp_size = 16
    channel.ai_resolution = 16.0
    channel.ai_resolution_units = constants.ResolutionType.BITS
    channel.ai_raw_samp_justification = constants.DataJustification.LEFT
    channel.ai_raw_data_compression_type = constants.RawDataCompressionType.NONE
    channel.ai_rng_low = -5.0
    channel.ai_rng_high = 5.0

    task = Mock()
    task.ai_channels.add_ai_voltage_chan.return_value = channel
    task.timing.samp_clk_rate = 1000.0
    task.timing.samp_clk_term = "/Dev1/ai/SampleClock"

    factory = Mock(return_value=task)
    reader = Mock()
    reader_factory = Mock(return_value=reader)

    monkeypatch.setattr(nidaqmx, "Task", factory)
    monkeypatch.setattr(stream_readers, "AnalogUnscaledReader", reader_factory)

    return FakeNI(factory, task, channel, reader_factory, reader)


def _device() -> NIDigitizer:
    """Expose two nonconsecutive physical channels as input indices zero and one."""
    return NIDigitizer(NIDigitizerConfig(physical_channels="Dev1/ai2,Dev1/ai0"))


def _settings() -> NICaptureSettings:
    """Request five two-record batches through reversed input indices."""
    return NICaptureSettings(
        sample_rate=units.SampleRate("1 kS/s"),
        record_length=4,
        records_per_batch=2,
        batch_count=5,
        channel_indices=(1, 0),
        input_voltage_range=units.VoltageRange("-5 V", "5 V"),
        terminal_config=constants.TerminalConfiguration.DIFF,
    )


def test_config_parsing_and_passive_construction(ni: FakeNI) -> None:
    """Parse a TOML-compatible connection model without probing hardware.

    Range expansion supplies the exposed channel count. ADC format properties
    remain unavailable until preparation, while connection snapshots and
    driver identity are already available without creating a task.
    """
    config = parse_config(
        NIDigitizerConfig,
        {"physical_channels": "Dev1/ai0:1"},
    )
    digitizer = NIDigitizer(config)

    try:
        assert digitizer.config is config
        assert config.expanded_channels == ("Dev1/ai0", "Dev1/ai1")
        assert digitizer.channel_count == 2

        with pytest.raises(RuntimeError, match="successful preparation"):
            _ = digitizer.bit_depth
        with pytest.raises(RuntimeError, match="successful preparation"):
            _ = digitizer.data_range

        snapshot = digitizer.snapshot()
        assert snapshot["settings"]["physical_channels"] == [
            "Dev1/ai0",
            "Dev1/ai1",
        ]
        assert snapshot["driver"]["distribution"] == "dirigo-nidaq"
    finally:
        digitizer.close()

    ni.factory.assert_not_called()


@pytest.mark.parametrize(
    ("channels", "message"),
    [
        ("", "least one physical channel"),
        ("Dev1/ao0", "must name an NI AI channel"),
        ("Dev1/ai0,dev1/ai00", "channels must be distinct"),
        ("Dev1/ai0,Dev2/ai0", "channels must belong to one"),
    ],
)
def test_invalid_channel_mapping(channels: str, message: str, ni: FakeNI) -> None:
    """Reject empty, non-AI, duplicated, or multiple-device selections."""
    with pytest.raises(ValueError, match=message):
        NIDigitizerConfig(physical_channels=channels)

    ni.factory.assert_not_called()


def test_invalid_configuration_types() -> None:
    """Reject incorrect model and connection-field types before SDK access."""
    with pytest.raises(TypeError, match="physical_channels"):
        NIDigitizerConfig(physical_channels=cast(str, 123))

    with pytest.raises(TypeError, match="config"):
        NIDigitizer(cast(NIDigitizerConfig, object()))


@pytest.mark.parametrize(
    ("changes", "error_type"),
    [
        ({"input_voltage_range": "5 V"}, TypeError),
        ({"input_voltage_range": units.VoltageRange("0 V", "5 V")}, ValueError),
        ({"terminal_config": "differential"}, TypeError),
        ({"record_length": 0}, ValueError),
    ],
)
def test_invalid_capture_settings(
    changes: dict[str, object],
    error_type: type[Exception],
) -> None:
    """Apply NI voltage/wiring validation and inherited geometry checks."""
    with pytest.raises(error_type):
        replace(_settings(), **changes)


def test_factory_validation_and_unused_close_are_passive(ni: FakeNI) -> None:
    """Require NI run settings and finite delivery without allocating a task."""
    digitizer = _device()

    try:
        with pytest.raises(TypeError, match="NICaptureSettings"):
            digitizer.create_capture(
                CaptureSettings(
                    sample_rate=units.SampleRate("1 kS/s"),
                    record_length=4,
                )
            )

        with pytest.raises(NotImplementedError, match="finite"):
            digitizer.create_capture(replace(_settings(), batch_count=None))

        session = digitizer.create_capture(_settings())
        session.close()
        session.close()
        assert session.state is CaptureState.CLOSED
    finally:
        digitizer.close()

    ni.factory.assert_not_called()


def test_prepare_preserves_mapping_and_commits_without_starting(ni: FakeNI) -> None:
    """Register inputs in delivery order and reserve one finite AI task.

    Configured input zero maps to ai2 and input one to ai0. Selecting (1, 0)
    must therefore register ai0 before ai2. NI's per-channel sample count covers
    every record in every batch; CaptureLayout describes just one batch.
    """
    digitizer = _device()
    session = digitizer.create_capture(_settings())
    ni.factory.assert_not_called()

    try:
        session.prepare()

        assert session.state is CaptureState.PREPARED
        ni.factory.assert_called_once_with()
        assert ni.task.ai_channels.add_ai_voltage_chan.call_args_list == [
            call(
                physical_channel=name,
                terminal_config=constants.TerminalConfiguration.DIFF,
                min_val=-5.0,
                max_val=5.0,
            )
            for name in ("Dev1/ai0", "Dev1/ai2")
        ]
        ni.task.timing.cfg_samp_clk_timing.assert_called_once_with(
            rate=1000.0,
            source="",
            active_edge=constants.Edge.RISING,
            sample_mode=constants.AcquisitionType.FINITE,
            samps_per_chan=40,
        )
        ni.task.control.assert_called_once_with(constants.TaskMode.TASK_COMMIT)
        ni.task.start.assert_not_called()
        ni.task.read.assert_not_called()
        ni.task.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()

        assert session.layout.shape == (2, 4, 2)
        assert session.layout.channel_indices == (1, 0)
        assert session.layout.dtype == np.dtype(np.int16)
        assert digitizer.bit_depth == 16
        assert digitizer.data_range == units.IntRange(-32768, 32767)
    finally:
        session.close()
        session.close()
        digitizer.close()

    ni.task.stop.assert_not_called()
    ni.task.close.assert_called_once_with()


def test_prepare_forwards_external_timing(ni: FakeNI) -> None:
    """Pass explicit sample-clock and whole-run trigger routes to the AI task."""
    settings = replace(
        _settings(),
        external_clock=ExternalSampleClock(
            source="/Dev1/PFI0",
            max_rate=units.SampleRate("2 kS/s"),
            active_edge=SignalEdge.FALLING,
        ),
        start_trigger=DigitalStartTrigger(
            source="/Dev1/PFI1",
            active_edge=SignalEdge.FALLING,
        ),
    )
    digitizer = _device()
    session = digitizer.create_capture(settings)

    try:
        session.prepare()

        ni.task.timing.cfg_samp_clk_timing.assert_called_once_with(
            rate=2000.0,
            source="/Dev1/PFI0",
            active_edge=constants.Edge.FALLING,
            sample_mode=constants.AcquisitionType.FINITE,
            samps_per_chan=40,
        )
        ni.task.triggers.start_trigger.cfg_dig_edge_start_trig.assert_called_once_with(
            trigger_source="/Dev1/PFI1",
            trigger_edge=constants.Edge.FALLING,
        )
    finally:
        session.close()
        digitizer.close()


@pytest.mark.parametrize(
    ("width", "resolution", "justification", "bounds"),
    [
        (16, 14, constants.DataJustification.LEFT, (-32768, 32764)),
        (16, 14, constants.DataJustification.RIGHT, (-8192, 8191)),
        (32, 24, constants.DataJustification.LEFT, (-2147483648, 2147483392)),
    ],
)
def test_native_resolution_storage_and_code_bounds(
    ni: FakeNI,
    width: int,
    resolution: int,
    justification: constants.DataJustification,
    bounds: tuple[int, int],
) -> None:
    """Distinguish ADC resolution, storage width, and native bit justification."""
    ni.channel.ai_raw_samp_size = width
    ni.channel.ai_resolution = float(resolution)
    ni.channel.ai_raw_samp_justification = justification

    digitizer = _device()
    session = digitizer.create_capture(_settings())

    try:
        session.prepare()

        expected_dtype = np.dtype(np.int16 if width == 16 else np.int32)
        assert session.layout.dtype == expected_dtype
        assert digitizer.bit_depth == resolution
        assert digitizer.data_range == units.IntRange(*bounds)

        session.close()
        assert digitizer.bit_depth == resolution
    finally:
        session.close()
        digitizer.close()


@pytest.mark.parametrize("stage", ["channel", "timing", "commit", "reader", "format"])
def test_prepare_failure_closes_task_and_releases_reservation(
    ni: FakeNI,
    stage: str,
) -> None:
    """Release partial AI preparation while preserving its primary failure.

    A failing close adds context to the original exception. A later session
    must still be able to reserve and prepare this logical digitizer.
    """
    failure = RuntimeError("preparation failed")
    operation = {
        "channel": ni.task.ai_channels.add_ai_voltage_chan,
        "timing": ni.task.timing.cfg_samp_clk_timing,
        "commit": ni.task.control,
        "reader": ni.reader_factory,
    }.get(stage)

    if operation is not None:
        operation.side_effect = failure
    else:
        ni.channel.ai_raw_samp_size = 8

    ni.task.close.side_effect = RuntimeError("cleanup failed")
    digitizer = _device()
    session = digitizer.create_capture(_settings())

    try:
        error_type = NotImplementedError if stage == "format" else RuntimeError
        with pytest.raises(error_type) as caught:
            session.prepare()

        if stage != "format":
            assert caught.value is failure
        assert any("cleanup failed" in note for note in caught.value.__notes__)
        assert session.state is CaptureState.CLOSED
        ni.task.close.assert_called_once_with()

        if operation is not None:
            operation.side_effect = None
        ni.channel.ai_raw_samp_size = 16
        ni.task.close.side_effect = None

        retry = digitizer.create_capture(_settings())
        try:
            retry.prepare()
            assert retry.state is CaptureState.PREPARED
        finally:
            retry.close()
    finally:
        session.close()
        digitizer.close()


def test_entry_point_loads_without_creating_task(ni: FakeNI) -> None:
    """Resolve the installed categorized driver without touching native DAQmx."""
    entries = importlib_metadata.entry_points(
        group="dirigo.devices.digitizers",
        name="nidaq",
    )

    assert len(entries) == 1
    assert next(iter(entries)).load() is NIDigitizer
    assert NIDigitizer.config_type is NIDigitizerConfig
    ni.factory.assert_not_called()


def _channel_samples(first: int, count: int) -> SampleArray:
    """Provide distinct NI channel rows with consecutive sample positions."""
    positions = np.arange(first, first + count, dtype=np.int32)
    return np.stack((100 + positions, 200 + positions))


@pytest.mark.parametrize("storage_bits", [16, 32])
@pytest.mark.parametrize("timestamps_enabled", [False, True])
def test_integer_readout_preserves_layout_and_finite_completion(
    ni: FakeNI,
    storage_bits: int,
    timestamps_enabled: bool,
) -> None:
    """Convert NI channel rows into two complete record/sample/channel batches.

    Exercise both integer reader methods and reuse the caller's array. When
    timestamps are enabled, the coerced 2 kS/s NI rate must determine record
    times even though the request named 1 kS/s.

    Finite exhaustion must not invoke another SDK read or release the task
    before explicit cleanup.
    """
    ni.channel.ai_raw_samp_size = storage_bits
    ni.channel.ai_resolution = float(storage_bits)
    ni.task.timing.samp_clk_rate = 2000.0
    settings = replace(
        _settings(),
        batch_count=2,
        timestamps_enabled=timestamps_enabled,
    )
    digitizer = _device()
    session = digitizer.create_capture(settings)
    first = 0

    def read(
        data: SampleArray,
        *,
        number_of_samples_per_channel: int,
        timeout: float,
    ) -> int:
        """Supply eight packed samples per channel on each SDK call."""
        nonlocal first
        assert data.shape == (2, 8)
        assert data.flags.c_contiguous
        assert number_of_samples_per_channel == 8
        assert timeout == constants.WAIT_INFINITELY

        data[:] = _channel_samples(first, 8)
        first += 8
        return 8

    selected = ni.reader.read_int16 if storage_bits == 16 else ni.reader.read_int32
    unused = ni.reader.read_int32 if storage_bits == 16 else ni.reader.read_int16
    selected.side_effect = read

    try:
        session.prepare()

        ni.reader_factory.assert_called_once_with(ni.task.in_stream)
        assert ni.task.in_stream.auto_start is False
        assert ni.reader.verify_array_shape is True
        ni.task.start.assert_not_called()

        session.start()
        ni.task.start.assert_called_once_with()
        samples = np.empty(session.layout.shape, dtype=session.layout.dtype)

        for index in range(2):
            info = session.read_into(samples)
            assert info is not None

            expected = _channel_samples(8 * index, 8).T.reshape(2, 4, 2)
            np.testing.assert_array_equal(samples, expected)

            times = info.timestamps
            if timestamps_enabled:
                assert times is not None
                np.testing.assert_allclose(
                    times,
                    [0.004 * index, 0.004 * index + 0.002],
                )
            else:
                assert times is None

        assert session.state is CaptureState.COMPLETED
        assert session.batches_acquired == 2
        assert session.read_into(samples) is None
        assert selected.call_count == 2
        unused.assert_not_called()
        ni.task.stop.assert_not_called()
        ni.task.close.assert_not_called()

        session.close()
        session.close()
        ni.task.stop.assert_called_once_with()
        ni.task.close.assert_called_once_with()
    finally:
        session.close()
        digitizer.close()


@pytest.mark.parametrize("error_code", [-200284, -200474])
@pytest.mark.parametrize("partial_count", [0, 3, 8])
def test_timeout_preserves_progress_for_a_different_destination(
    ni: FakeNI,
    error_code: int,
    partial_count: int,
) -> None:
    """Keep consumed samples private and continue the same batch after timeout.

    NI packs actual samples by channel into the receive buffer's prefix.
    A smaller retry must use contiguous storage with the remaining sample
    count. Overwriting the first caller array must not corrupt retained
    progress.

    If the error reports a whole batch, retrying needs no additional SDK read.
    """
    calls: list[tuple[int, float]] = []

    def read(
        data: SampleArray,
        *,
        number_of_samples_per_channel: int,
        timeout: float,
    ) -> int:
        """Return a packed timeout result followed by the remaining samples."""
        calls.append((number_of_samples_per_channel, timeout))
        assert data.flags.c_contiguous

        if len(calls) == 1:
            data.reshape(-1)[: 2 * partial_count] = _channel_samples(
                0, partial_count
            ).reshape(-1)
            raise errors.DaqReadError("read timed out", error_code, partial_count)

        assert data.shape == (2, 8 - partial_count)
        data[:] = _channel_samples(partial_count, 8 - partial_count)
        return 8 - partial_count

    ni.reader.read_int16.side_effect = read
    digitizer = _device()
    session = digitizer.create_capture(_settings())

    try:
        session.prepare()
        session.start()

        abandoned = np.full(session.layout.shape, -1234, dtype=session.layout.dtype)
        with pytest.raises(TimeoutError) as caught:
            session.read_into(abandoned, timeout=0.01)

        assert isinstance(caught.value.__cause__, errors.DaqReadError)
        assert session.state is CaptureState.STARTED
        assert session.batches_acquired == 0
        np.testing.assert_array_equal(abandoned, -1234)
        abandoned.fill(-9999)

        destination = np.empty(session.layout.shape, dtype=session.layout.dtype)
        info = session.read_into(destination, timeout=0)
        assert info is not None

        np.testing.assert_array_equal(
            destination,
            _channel_samples(0, 8).T.reshape(2, 4, 2),
        )
        np.testing.assert_array_equal(abandoned, -9999)
        assert session.batches_acquired == 1

        expected_calls = [(8, 0.01)]
        if partial_count < 8:
            expected_calls.append((8 - partial_count, 0.0))
        assert calls == expected_calls
        ni.task.close.assert_not_called()
    finally:
        session.close()
        digitizer.close()


@pytest.mark.parametrize("error_code", [-200279, -200278])
def test_fatal_read_errors_survive_native_cleanup(
    ni: FakeNI,
    error_code: int,
) -> None:
    """Propagate data loss or premature exhaustion with cleanup context.

    These errors must not become retryable timeouts or normal finite EOF.
    Failing native stop and close operations must leave the original read
    exception primary while the session releases its reservation.
    """
    failure = errors.DaqReadError("capture failed", error_code, 0)
    ni.reader.read_int16.side_effect = failure
    digitizer = _device()
    session = digitizer.create_capture(_settings())

    try:
        session.prepare()
        session.start()
        ni.task.stop.side_effect = RuntimeError("stop failed")
        ni.task.close.side_effect = RuntimeError("clear failed")
        samples = np.empty(session.layout.shape, dtype=session.layout.dtype)

        with pytest.raises(errors.DaqReadError) as caught:
            session.read_into(samples, timeout=0.1)

        assert caught.value is failure
        notes = " ".join(failure.__notes__)
        assert "stop failed" in notes
        assert "clear failed" in notes
        assert session.state is CaptureState.CLOSED
        assert session.batches_acquired == 0
        ni.task.stop.assert_called_once_with()
        ni.task.close.assert_called_once_with()
    finally:
        session.close()
        digitizer.close()


@pytest.mark.parametrize("count", [-1, 7, 9])
def test_invalid_or_short_successful_reads_are_fatal(ni: FakeNI, count: int) -> None:
    """Reject impossible counts and short success instead of silently losing data."""
    ni.reader.read_int16.return_value = count
    digitizer = _device()
    session = digitizer.create_capture(_settings())

    try:
        session.prepare()
        session.start()
        samples = np.empty(session.layout.shape, dtype=session.layout.dtype)

        with pytest.raises(RuntimeError, match="NI returned"):
            session.read_into(samples)

        assert session.state is CaptureState.CLOSED
        assert session.batches_acquired == 0
        ni.task.close.assert_called_once_with()
    finally:
        session.close()
        digitizer.close()


def test_failed_start_attempts_stop_and_task_cleanup(ni: FakeNI) -> None:
    """Treat a failing native start as potentially partial and clean up once."""
    failure = RuntimeError("start failed")
    ni.task.start.side_effect = failure
    digitizer = _device()
    session = digitizer.create_capture(_settings())

    try:
        session.prepare()

        with pytest.raises(RuntimeError) as caught:
            session.start()

        assert caught.value is failure
        assert session.state is CaptureState.CLOSED
        ni.task.stop.assert_called_once_with()
        ni.task.close.assert_called_once_with()
    finally:
        session.close()
        digitizer.close()


def test_external_clock_cannot_supply_derived_timestamps(ni: FakeNI) -> None:
    """Reject rate-derived timestamps when external edge times are unknown."""
    settings = replace(
        _settings(),
        timestamps_enabled=True,
        external_clock=ExternalSampleClock(
            source="/Dev1/PFI0",
            max_rate=units.SampleRate("1 kS/s"),
        ),
    )
    digitizer = _device()

    try:
        with pytest.raises(NotImplementedError, match="internal sample clock"):
            digitizer.create_capture(settings)
        ni.factory.assert_not_called()
    finally:
        digitizer.close()


def test_prepared_internal_timing_is_cached_and_retained(ni: FakeNI) -> None:
    """Expose realized timing after preparation and retain it after cleanup.

    Timestamps are disabled, so timing must be captured independently of
    timestamp generation. Changing SDK attributes after preparation proves
    that property access uses cached descriptions rather than querying the
    task. The properties remain usable after their task has been cleared.
    """
    ni.task.timing.samp_clk_rate = 1250.0
    digitizer = _device()
    session = digitizer.create_capture(_settings())
    assert isinstance(session, NICaptureSession)

    try:
        with pytest.raises(RuntimeError, match="before successful preparation"):
            _ = session.sample_rate
        with pytest.raises(RuntimeError, match="before successful preparation"):
            _ = session.sample_clock_terminal
        ni.factory.assert_not_called()

        session.prepare()
        assert session.sample_rate == units.SampleRate("1.25 kS/s")
        assert session.sample_clock_terminal == "/Dev1/ai/SampleClock"

        ni.task.timing.samp_clk_rate = float("nan")
        ni.task.timing.samp_clk_term = "/changed"

        session.stop()
        assert session.sample_rate == units.SampleRate("1.25 kS/s")
        assert session.sample_clock_terminal == "/Dev1/ai/SampleClock"

        session.close()
        assert session.sample_rate == units.SampleRate("1.25 kS/s")
        assert session.sample_clock_terminal == "/Dev1/ai/SampleClock"
        ni.task.start.assert_not_called()
        ni.task.close.assert_called_once_with()
    finally:
        session.close()
        digitizer.close()


def test_external_clock_rate_is_unknown_and_terminal_is_internal(ni: FakeNI) -> None:
    """Keep an external maximum rate distinct from known internal timing.

    DAQmx's samp_clk_term describes the task's internal clock terminal,
    which differs from its configured PFI source. Its samp_clk_rate cannot
    supply a measured external rate.
    """
    clock = ExternalSampleClock(
        source="/Dev1/PFI0",
        max_rate=units.SampleRate("2 kS/s"),
    )
    ni.task.timing.samp_clk_rate = object()
    digitizer = _device()
    session = digitizer.create_capture(replace(_settings(), external_clock=clock))

    try:
        session.prepare()

        assert session.sample_rate is None
        assert session.sample_clock_terminal == "/Dev1/ai/SampleClock"
        assert session.sample_clock_terminal != clock.source

        session.close()
        assert session.sample_rate is None
    finally:
        session.close()
        digitizer.close()


def test_later_preparation_failure_does_not_publish_timing(ni: FakeNI) -> None:
    """Publish no timing description when a later preparation step fails.

    Clock properties can be read successfully before reader construction
    fails. The session must close its task and keep both properties
    unavailable because preparation as a whole did not succeed.
    """
    failure = RuntimeError("reader construction failed")
    ni.reader_factory.side_effect = failure
    digitizer = _device()
    session = digitizer.create_capture(_settings())

    try:
        with pytest.raises(RuntimeError) as caught:
            session.prepare()

        assert caught.value is failure
        assert session.state is CaptureState.CLOSED

        with pytest.raises(RuntimeError, match="before successful preparation"):
            _ = session.sample_rate
        with pytest.raises(RuntimeError, match="before successful preparation"):
            _ = session.sample_clock_terminal

        ni.task.close.assert_called_once_with()
    finally:
        session.close()
        digitizer.close()


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("samp_clk_rate", 0.0),
        ("samp_clk_rate", float("nan")),
        ("samp_clk_term", ""),
    ],
)
def test_invalid_reported_timing_fails_preparation(
    ni: FakeNI,
    attribute: str,
    value: object,
) -> None:
    """Reject unusable timing and release the partially prepared task."""
    setattr(ni.task.timing, attribute, value)
    digitizer = _device()
    session = digitizer.create_capture(_settings())

    try:
        with pytest.raises(RuntimeError, match="NI reported an invalid"):
            session.prepare()

        assert session.state is CaptureState.CLOSED
        ni.task.close.assert_called_once_with()
    finally:
        session.close()
        digitizer.close()
