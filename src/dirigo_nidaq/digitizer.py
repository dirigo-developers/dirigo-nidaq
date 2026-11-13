"""Finite, unscaled NI voltage-input capture with retryable batch delivery."""

import math
import re
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from typing import cast

import nidaqmx
import numpy as np
from dirigo import units
from dirigo.hardware.capture import (
    CaptureInfo,
    CaptureLayout,
    CaptureSettings,
    SampleArray,
)
from dirigo.hardware.capture_session import DigitizerCaptureSession
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.metadata import DriverIdentity
from dirigo.hardware.timing import SignalEdge
from nidaqmx import constants, errors, stream_readers
from nidaqmx.task.channels import AIChannel
from nidaqmx.utils import unflatten_channel_string

_READ_TIMEOUT_CODES = {-200284, -200474}


@dataclass(frozen=True, slots=True, kw_only=True)
class NIDigitizerConfig:
    """Map exposed input indices to an ordered NI channel selection.

    physical_channels accepts an NI list or range, such as Dev1/ai0:3 or
    Dev1/ai2,Dev1/ai0. CaptureSettings.channel_indices indexes the expanded
    selection, so index zero means the first configured input.

    Inputs must be distinct voltage-AI channels on one device. Validation
    parses names without querying hardware or creating a task.
    """

    physical_channels: str = field(
        metadata={"description": "Ordered NI input-channel list or range."}
    )
    expanded_channels: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Expand and validate the connection mapping without hardware access."""
        if not isinstance(self.physical_channels, str):
            raise TypeError("physical_channels must be a string.")

        channels = unflatten_channel_string(self.physical_channels)
        if not channels:
            raise ValueError("At least one physical channel is required.")

        names: list[str] = []
        keys: set[tuple[str, int]] = set()
        devices: set[str] = set()

        for channel in channels:
            match = re.fullmatch(
                r"/?([^/,\s]+)/ai([0-9]+)", channel, flags=re.IGNORECASE
            )
            if match is None:
                raise ValueError("Each input must name an NI AI channel.")

            device, number = match[1], int(match[2])
            key = (device.casefold(), number)
            if key in keys:
                raise ValueError("Physical channels must be distinct.")

            keys.add(key)
            devices.add(key[0])
            names.append(f"{device}/ai{number}")

        if len(devices) != 1:
            raise ValueError("Physical channels must belong to one NI device.")

        object.__setattr__(self, "expanded_channels", tuple(names))


@dataclass(frozen=True, slots=True, kw_only=True)
class NICaptureSettings(CaptureSettings):
    """Add voltage range and wiring mode to a per-run capture request.

    The same requested input range and terminal mode apply to every selected
    channel. This first backend supports uncompressed bipolar integer data.
    DAQmx validates device-specific ranges, wiring modes, timing, and routes.
    """

    input_voltage_range: units.VoltageRange
    terminal_config: constants.TerminalConfiguration = (
        constants.TerminalConfiguration.DEFAULT
    )

    def __post_init__(self) -> None:
        """Validate generic geometry and NI voltage-input settings."""
        CaptureSettings.__post_init__(self)

        if not isinstance(self.input_voltage_range, units.VoltageRange):
            raise TypeError("input_voltage_range must be a VoltageRange.")

        low = float(self.input_voltage_range.min)
        high = float(self.input_voltage_range.max)
        if not math.isfinite(low) or not math.isfinite(high) or not low < 0 < high:
            raise ValueError("input_voltage_range must be finite and bipolar.")

        if not isinstance(self.terminal_config, constants.TerminalConfiguration):
            raise TypeError("terminal_config must be a TerminalConfiguration.")


@dataclass(frozen=True, slots=True)
class _SampleFormat:
    """Describe signed, uncompressed native codes shared by selected channels."""

    bit_depth: int
    storage_bits: int
    justification: constants.DataJustification

    @property
    def dtype(self) -> np.dtype[np.int16] | np.dtype[np.int32]:
        """Native integer storage selected for the later unscaled reader."""
        return np.dtype(np.int16) if self.storage_bits == 16 else np.dtype(np.int32)

    @property
    def data_range(self) -> units.IntRange:
        """Bounds accounting for ADC resolution and native bit justification."""
        shift = (
            self.storage_bits - self.bit_depth
            if self.justification is constants.DataJustification.LEFT
            else 0
        )
        half_range = 1 << (self.bit_depth - 1)
        return units.IntRange(-half_range << shift, (half_range - 1) << shift)


def _sample_format(channel: AIChannel) -> _SampleFormat:
    """Inspect a committed channel and reject unsupported native representations."""
    width = channel.ai_raw_samp_size
    resolution = float(channel.ai_resolution)
    justification = channel.ai_raw_samp_justification

    if (
        width not in (16, 32)
        or channel.ai_resolution_units is not constants.ResolutionType.BITS
        or not math.isfinite(resolution)
        or not resolution.is_integer()
        or not 1 < resolution <= width
        or justification
        not in (
            constants.DataJustification.LEFT,
            constants.DataJustification.RIGHT,
        )
        or channel.ai_raw_data_compression_type
        is not constants.RawDataCompressionType.NONE
        or not channel.ai_rng_low < 0 < channel.ai_rng_high
    ):
        raise NotImplementedError("Unsupported NI native integer sample format.")

    return _SampleFormat(int(resolution), width, justification)


@dataclass(frozen=True, slots=True)
class _PreparedTiming:
    """Cache NI timing descriptions independently of native task lifetime."""

    sample_rate: units.SampleRate | None
    sample_clock_terminal: str


class NIDigitizer(Digitizer):
    """Expose configured NI voltage inputs without probing during construction.

    bit_depth and data_range become available after a successful preparation
    and describe the last successfully prepared native format. They remain
    readable after session closure; accessing them earlier raises RuntimeError.

    Sessions own their AI tasks. This device never resets or exclusively owns
    the whole DAQ card, allowing separately owned AO and counter tasks.
    """

    config_type = NIDigitizerConfig

    def __init__(self, config: NIDigitizerConfig) -> None:
        """Store the input mapping without creating a task or querying a device."""
        if not isinstance(config, NIDigitizerConfig):
            raise TypeError("config must be an NIDigitizerConfig.")

        super().__init__()
        self._config = config
        self._sample_format: _SampleFormat | None = None

    @property
    def config(self) -> NIDigitizerConfig:
        """Immutable mapping from exposed inputs to NI physical channels."""
        return self._config

    @property
    def channel_count(self) -> int:
        """Number of configured inputs, independent of hardware availability."""
        return len(self.config.expanded_channels)

    @property
    def bit_depth(self) -> int:
        """ADC resolution established by the last successful preparation."""
        return self._require_format().bit_depth

    @property
    def data_range(self) -> units.IntRange:
        """Native code bounds established by the last successful preparation."""
        return self._require_format().data_range

    def _require_format(self) -> _SampleFormat:
        """Reject unknown capabilities without initiating a hardware probe."""
        if self._sample_format is None:
            raise RuntimeError("NI sample format requires successful preparation.")
        return self._sample_format

    @property
    def driver_identity(self) -> DriverIdentity:
        """Identify the implementation and installed plugin distribution."""
        try:
            version = importlib_metadata.version("dirigo-nidaq")
        except importlib_metadata.PackageNotFoundError:
            version = None

        return DriverIdentity(
            implementation=super().driver_identity.implementation,
            distribution="dirigo-nidaq",
            version=version,
        )

    def _create_capture(self, settings: CaptureSettings) -> NICaptureSession:
        """Validate request types without preparing or reserving inputs."""
        if not isinstance(settings, NICaptureSettings):
            raise TypeError("NI capture requires NICaptureSettings.")
        if settings.batch_count is None:
            raise NotImplementedError("NI capture requires finite batch_count.")
        if settings.timestamps_enabled and settings.external_clock is not None:
            raise NotImplementedError(
                "NI timestamps currently require an internal sample clock."
            )
        if settings.record_trigger is not None:
            raise NotImplementedError(
                "NI capture does not support record triggers yet."
            )

        return NICaptureSession(self, settings)

    def create_capture(self, settings: CaptureSettings) -> NICaptureSession:
        """Construct an unprepared NI session using the core factory checks.

        Require NICaptureSettings at runtime. Keep the base argument type so
        this override remains compatible with the generic digitizer API.
        """
        return cast(NICaptureSession, super().create_capture(settings))

    def _close(self) -> None:
        """Release no persistent resources; borrowing sessions own their tasks."""
        return

    def _snapshot_settings(self) -> dict[str, object]:
        """Describe the configured connection without probing the device."""
        return {"physical_channels": self.config.expanded_channels}


class NICaptureSession(DigitizerCaptureSession):
    """Own finite AI capture and stage complete batches independently of callers.

    Construct through NIDigitizer.create_capture(). Preparation establishes
    sample_rate and sample_clock_terminal without starting a clock.

    These cached descriptions remain readable after stop or close. They do not
    establish that a clock is currently running or routable to a consumer.

    Internal-clock timestamps are derived record-start times using DAQmx's
    prepared sample rate, relative to the first sample.
    """

    def __init__(self, digitizer: NIDigitizer, settings: NICaptureSettings) -> None:
        super().__init__(
            digitizer,
            batch_count=settings.batch_count,
            timestamps_enabled=settings.timestamps_enabled,
        )
        self._ni_digitizer = digitizer
        self._settings = settings
        self._task: nidaqmx.Task | None = None
        self._reader: stream_readers.AnalogUnscaledReader | None = None
        self._pending: SampleArray | None = None
        self._transfer: SampleArray | None = None
        self._pending_count = 0
        self._timing: _PreparedTiming | None = None

    def _prepare_capture(self) -> CaptureLayout:
        """Configure and commit AI resources without starting or reading samples."""
        settings = self._settings
        batch_count = settings.batch_count
        assert batch_count is not None

        task = nidaqmx.Task()
        self._task = task
        channels = [
            task.ai_channels.add_ai_voltage_chan(
                physical_channel=self._ni_digitizer.config.expanded_channels[index],
                terminal_config=settings.terminal_config,
                min_val=float(settings.input_voltage_range.min),
                max_val=float(settings.input_voltage_range.max),
            )
            for index in settings.channel_indices
        ]

        clock = settings.external_clock
        task.timing.cfg_samp_clk_timing(
            rate=float(settings.sample_rate if clock is None else clock.max_rate),
            source="" if clock is None else clock.source,
            active_edge=(
                constants.Edge.FALLING
                if clock is not None and clock.active_edge is SignalEdge.FALLING
                else constants.Edge.RISING
            ),
            sample_mode=constants.AcquisitionType.FINITE,
            samps_per_chan=(
                settings.record_length * settings.records_per_batch * batch_count
            ),
        )

        trigger = settings.start_trigger
        if trigger is not None:
            task.triggers.start_trigger.cfg_dig_edge_start_trig(
                trigger_source=trigger.source,
                trigger_edge=(
                    constants.Edge.RISING
                    if trigger.active_edge is SignalEdge.RISING
                    else constants.Edge.FALLING
                ),
            )

        task.in_stream.auto_start = False
        task.control(constants.TaskMode.TASK_COMMIT)

        formats = [_sample_format(channel) for channel in channels]
        sample_format = formats[0]
        if any(value != sample_format for value in formats[1:]):
            raise NotImplementedError("Selected NI channels need one sample format.")

        layout = CaptureLayout(
            shape=(
                settings.records_per_batch,
                settings.record_length,
                len(settings.channel_indices),
            ),
            channel_indices=settings.channel_indices,
            dtype=sample_format.dtype,
        )

        timing = self._read_prepared_timing(task)
        self._prepare_readout(task, layout)
        self._ni_digitizer._sample_format = sample_format
        self._timing = timing
        return layout

    def _prepare_readout(self, task: nidaqmx.Task, layout: CaptureLayout) -> None:
        """Allocate private batch storage and bind an unscaled stream reader."""
        self._reader = stream_readers.AnalogUnscaledReader(task.in_stream)
        self._reader.verify_array_shape = True

        count = layout.records_per_batch * layout.record_length
        self._pending = np.empty((layout.channel_count, count), dtype=layout.dtype)
        self._transfer = np.empty(layout.channel_count * count, dtype=layout.dtype)

    def _require_task(self) -> nidaqmx.Task:
        """Return the prepared task or reject an internal lifecycle error."""
        if self._task is None:
            raise RuntimeError("NI input task is not prepared.")
        return self._task

    def _start_capture(self) -> None:
        """Start or arm the committed task without waiting for incoming samples."""
        self._require_task().start()

    def _read_into(
        self,
        destination: SampleArray,
        *,
        timeout: float | None,
    ) -> CaptureInfo:
        """Finish one batch using at most one NI read within the waiting budget.

        Timed-out reads may consume samples. Retain those privately and allow
        a retry with entirely different caller storage. Only complete batches
        are copied into the caller's record/sample/channel layout.
        """
        reader, pending, transfer = self._reader, self._pending, self._transfer
        assert reader is not None
        assert pending is not None
        assert transfer is not None

        layout = self.layout
        total = layout.records_per_batch * layout.record_length
        remaining = total - self._pending_count

        if remaining:
            # Reshape a flat prefix so every NI receive array is C-contiguous,
            # including a smaller read completing a previously partial batch.
            receive = transfer[: layout.channel_count * remaining].reshape(
                layout.channel_count, remaining
            )
            read = (
                reader.read_int16
                if layout.dtype == np.dtype(np.int16)
                else reader.read_int32
            )
            read_failure: errors.DaqReadError | None = None

            try:
                count = read(
                    receive,
                    number_of_samples_per_channel=remaining,
                    timeout=constants.WAIT_INFINITELY if timeout is None else timeout,
                )
            except errors.DaqReadError as failure:
                if failure.error_code not in _READ_TIMEOUT_CODES:
                    raise
                count = failure.samps_per_chan_read
                read_failure = failure

            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or not 0 <= count <= remaining
            ):
                raise RuntimeError("NI returned an invalid sample count.")

            if read_failure is None and count != remaining:
                raise RuntimeError("NI returned a short read without a timeout.")

            if count:
                # GROUP_BY_CHANNEL packs actual samples into the prefix.
                # Partial results cannot use the originally requested row stride.
                chunk = transfer[: layout.channel_count * count].reshape(
                    layout.channel_count, count
                )
                end = self._pending_count + count
                pending[:, self._pending_count : end] = chunk
                self._pending_count = end

            if read_failure is not None:
                raise TimeoutError("NI capture read timed out.") from read_failure

        info = CaptureInfo()
        if self.timestamps_enabled:
            rate = self.sample_rate
            assert rate is not None

            first_record = self.batches_acquired * layout.records_per_batch
            with np.errstate(over="raise", invalid="raise"):
                times = (
                    first_record + np.arange(layout.records_per_batch, dtype=np.float64)
                ) * (layout.record_length / float(rate))
            info = CaptureInfo(times)

        np.copyto(destination.reshape(total, layout.channel_count), pending.T)
        self._pending_count = 0
        return info

    def _stop_capture(self) -> None:
        """Stop acquisition, including after a partially failed native start."""
        if self._task is not None:
            self._task.stop()

    def _close_capture(self) -> None:
        """Release private staging and attempt native task cleanup exactly once."""
        self._reader = None
        self._pending = None
        self._transfer = None
        self._pending_count = 0

        task, self._task = self._task, None
        if task is not None:
            task.close()

    @property
    def sample_rate(self) -> units.SampleRate | None:
        """Prepared internal sampling rate, or None when externally clocked.

        Available only after successful preparation and retained after close.
        For an external clock, the configured maximum is not a measured rate.
        """
        return self._require_timing().sample_rate

    @property
    def sample_clock_terminal(self) -> str:
        """Internal NI sample-clock terminal reported by the prepared task.

        This is distinct from the configured source terminal. Reading this
        cached description does not export a signal or start acquisition.
        It remains readable after close, without keeping the task alive.
        """
        return self._require_timing().sample_clock_terminal

    def _require_timing(self) -> _PreparedTiming:
        """Reject timing access before a successful preparation."""
        if self._timing is None:
            raise RuntimeError(
                "NI timing is unavailable before successful preparation."
            )
        return self._timing

    def _read_prepared_timing(self, task: nidaqmx.Task) -> _PreparedTiming:
        """Read committed NI timing without publishing partial preparation."""
        terminal = task.timing.samp_clk_term
        if (
            not isinstance(terminal, str)
            or not terminal
            or terminal != terminal.strip()
        ):
            raise RuntimeError("NI reported an invalid sample-clock terminal.")

        sample_rate: units.SampleRate | None = None
        if self._settings.external_clock is None:
            rate = float(task.timing.samp_clk_rate)
            if not math.isfinite(rate) or rate <= 0:
                raise RuntimeError("NI reported an invalid sample rate.")
            sample_rate = units.SampleRate(rate)

        return _PreparedTiming(sample_rate, terminal)
