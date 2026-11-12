"""Finite waveform output for NI galvo scanners."""

import re
from collections.abc import Mapping

import nidaqmx
import numpy as np
from dirigo.hardware.analog import angle_voltage_coefficients
from dirigo.hardware.output import GalvoOutputSession
from dirigo.hardware.scanner import GalvoScanner
from dirigo.hardware.timing import ClockEdge, ExternalSampleClock
from dirigo.hardware.waveform import AngularWaveform
from nidaqmx import constants, errors, stream_writers

from dirigo_nidaq.scanner import NIGalvoScanner

_WAIT_UNTIL_DONE_TIMEOUT = -200560

_NI_CLOCK_EDGES = {
    ClockEdge.RISING: constants.Edge.RISING,
    ClockEdge.FALLING: constants.Edge.FALLING,
}


class NIGalvoOutputSession(GalvoOutputSession):
    """Output a finite group of galvo waveforms through one NI task.

    Each scanner must name one channel as Dev1/ao0 or /Dev1/ao0. Channels
    must be distinct and belong to the same NI device. Mapping order defines
    channel order and the corresponding rows of voltage samples.

    Construction performs no NI operations. prepare() reserves the scanners
    and preloads samples. With external_clock=None, output uses the onboard
    sample clock. Otherwise samples advance on the selected terminal edges
    after start(). Waveform sample rates remain nominal; external_clock.max_rate
    supplies the maximum expected clock rate for NI timing configuration.
    Closing releases the task without issuing a static or parking command.
    Sample-rate coercion and device capabilities remain the NI driver's
    responsibility.
    """

    def __init__(
        self,
        waveforms: Mapping[GalvoScanner, AngularWaveform],
        *,
        external_clock: ExternalSampleClock | None = None,
    ) -> None:
        """Validate waveforms, clock selection, and channels without a task."""
        super().__init__(waveforms, external_clock=external_clock)

        axes: list[NIGalvoScanner] = []
        channels: set[tuple[str, int]] = set()
        device: str | None = None

        for axis in self._waveforms:
            if not isinstance(axis, NIGalvoScanner):
                raise TypeError("NI output requires NIGalvoScanner instances.")

            channel = axis.config.angle_control_channel
            match = re.fullmatch(
                r"/?([A-Za-z0-9_-]+)/ao([0-9]+)", channel, flags=re.IGNORECASE
            )
            if match is None:
                raise ValueError(
                    "Each scanner must name one NI AO channel, e.g. Dev1/ao0."
                )

            channel_device = match[1].casefold()
            key = (channel_device, int(match[2]))
            if device is not None and channel_device != device:
                raise ValueError("All channels must belong to the same NI device.")
            if key in channels:
                raise ValueError("Scanners must use distinct NI AO channels.")

            device = channel_device
            channels.add(key)
            axes.append(axis)

        self._axes = tuple(axes)
        self._task: nidaqmx.Task | None = None

    def _prepare_output(self) -> None:
        """Convert angles and preload a finite task without starting it."""
        samples = np.empty((len(self._axes), self._sample_count), dtype=np.float64)
        for row, axis in zip(samples, self._axes, strict=True):
            config = axis.config
            gain, offset = angle_voltage_coefficients(
                config.angle_range,
                config.command_voltage_range,
                inverted=config.inverted,
            )
            with np.errstate(over="raise", invalid="raise"):
                np.multiply(self._waveforms[axis].samples, gain, out=row)
                np.add(row, offset, out=row)

            # Optical limits were validated before reservation. Bound only
            # endpoint roundoff to the configured command-voltage interval.
            np.clip(
                row,
                float(config.command_voltage_range.min),
                float(config.command_voltage_range.max),
                out=row,
            )

        task = nidaqmx.Task()
        self._task = task
        for axis in self._axes:
            config = axis.config
            task.ao_channels.add_ao_voltage_chan(
                physical_channel=config.angle_control_channel,
                min_val=float(config.command_voltage_range.min),
                max_val=float(config.command_voltage_range.max),
            )

        clock = self.external_clock
        task.timing.cfg_samp_clk_timing(
            rate=float(self._sample_rate if clock is None else clock.max_rate),
            source="" if clock is None else clock.source,
            active_edge=_NI_CLOCK_EDGES[
                ClockEdge.RISING if clock is None else clock.active_edge
            ],
            sample_mode=constants.AcquisitionType.FINITE,
            samps_per_chan=self._sample_count,
        )

        writer = stream_writers.AnalogMultiChannelWriter(
            task.out_stream, auto_start=False
        )
        if writer.write_many_sample(samples) != self._sample_count:
            raise RuntimeError("NI task did not accept every sample per channel.")

    def _require_task(self) -> nidaqmx.Task:
        """Return the prepared task or reject an internal lifecycle error."""
        if self._task is None:
            raise RuntimeError("NI output task is not prepared.")
        return self._task

    def _start_output(self) -> None:
        """Start the preloaded task using the selected sample clock."""
        self._require_task().start()

    def _wait_output(self, timeout: float | None) -> bool:
        """Translate a bounded NI completion timeout into False."""
        ni_timeout = constants.WAIT_INFINITELY if timeout is None else timeout
        try:
            self._require_task().wait_until_done(timeout=ni_timeout)
        except errors.DaqError as failure:
            if timeout is not None and failure.error_code == _WAIT_UNTIL_DONE_TIMEOUT:
                return False
            raise
        return True

    def _stop_output(self) -> None:
        """Stop the task, including after a partially failed start."""
        if self._task is not None:
            self._task.stop()

    def _close_output(self) -> None:
        """Attempt task cleanup once, including after partial preparation."""
        task, self._task = self._task, None
        if task is not None:
            task.close()
