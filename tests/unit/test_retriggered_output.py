"""Test retriggered galvo output through the public NI SDK boundaries."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import partial
from typing import cast
from unittest.mock import Mock, PropertyMock, call

import nidaqmx
import numpy as np
import pytest
from dirigo import units
from dirigo.hardware.output import OutputState
from dirigo.hardware.timing import DigitalStartTrigger
from dirigo.hardware.waveform import AngularWaveform
from nidaqmx import constants, errors, stream_writers
from numpy.typing import NDArray

from dirigo_nidaq import retriggered_output as output_module
from dirigo_nidaq.retriggered_output import NIRetriggeredGalvoOutputSession
from dirigo_nidaq.scanner import NIGalvoScanner, NIGalvoScannerConfig


@dataclass
class FakeClock:
    """Advance monotonic time through requested sleeps without real waiting."""

    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        """Return the simulated deadline clock."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Record and advance by the requested delay."""
        self.sleeps.append(seconds)
        self.now += seconds


@dataclass(frozen=True)
class FakeNI:
    """Expose native factories, calls, and generated-sample observations."""

    factory: Mock
    task: Mock
    writer_factory: Mock
    writer: Mock
    generated: PropertyMock


@pytest.fixture(autouse=True)
def ni(monkeypatch: pytest.MonkeyPatch) -> FakeNI:
    """Replace public NI factories without loading native driver libraries."""
    task = Mock()
    task.is_task_done.return_value = False
    generated = PropertyMock(return_value=0)
    monkeypatch.setattr(
        type(task.out_stream),
        "total_samp_per_chan_generated",
        generated,
        raising=False,
    )
    writer = Mock()
    writer.write_many_sample.return_value = 3
    factory = Mock(return_value=task)
    writer_factory = Mock(return_value=writer)
    monkeypatch.setattr(nidaqmx, "Task", factory)
    monkeypatch.setattr(stream_writers, "AnalogMultiChannelWriter", writer_factory)
    return FakeNI(factory, task, writer_factory, writer, generated)


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Replace only this module's time dependency for deterministic waits."""
    fake = FakeClock()
    monkeypatch.setattr(output_module, "time", fake)
    return fake


@pytest.fixture
def axes() -> Iterator[tuple[NIGalvoScanner, NIGalvoScanner]]:
    """Supply two passive axes with different voltage mappings and inversion."""
    x = NIGalvoScanner(
        NIGalvoScannerConfig(
            angle_control_channel="Dev1/ao0",
            angle_range=units.AngleRange("-10 deg", "30 deg"),
            command_voltage_range=units.VoltageRange("-1 V", "7 V"),
        )
    )
    y = NIGalvoScanner(
        NIGalvoScannerConfig(
            angle_control_channel="Dev1/ao1",
            angle_range=units.AngleRange("-20 deg", "20 deg"),
            command_voltage_range=units.VoltageRange("-4 V", "4 V"),
            inverted=True,
        )
    )
    try:
        yield x, y
    finally:
        y.close()
        x.close()


def _waveform(*degrees: float) -> AngularWaveform:
    """Build a three-sample frame from readable optical angles."""
    return AngularWaveform(
        [float(units.Angle(f"{value} deg")) for value in degrees],
        sample_rate=units.SampleRate("1 kS/s"),
    )


@pytest.fixture
def session(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
) -> Iterator[NIRetriggeredGalvoOutputSession]:
    """Close a two-frame session before its borrowed axes are closed."""
    output = NIRetriggeredGalvoOutputSession(
        {axes[0]: _waveform(0, 1, 2)},
        start_trigger=DigitalStartTrigger(source="/Dev1/PFI1"),
        frame_count=2,
    )
    try:
        yield output
    finally:
        output.close()


def test_unused_session_is_passive(
    session: NIRetriggeredGalvoOutputSession,
    ni: FakeNI,
) -> None:
    """Construct, inspect cached progress, and close without allocating NI tasks."""
    assert session.frame_count == 2
    assert session.completed_frames == 0
    assert session.external_clock is None
    assert session.state is OutputState.NEW
    session.close()
    session.close()
    ni.factory.assert_not_called()
    ni.generated.assert_not_called()


def test_prepare_preloads_one_frame_with_retriggering_enabled(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
) -> None:
    """Reuse grouped conversion while preloading only one frame for both axes.

    Reverse the axis order and give each axis a different voltage mapping.
    Check retriggering and regeneration inside the write callback to prove
    they are enabled before preloading. The expected run has two frames,
    but the buffer and finite timing describe only one three-sample frame.
    """
    x, y = axes
    output = NIRetriggeredGalvoOutputSession(
        {y: _waveform(-20, 0, 20), x: _waveform(-10, 0, 30)},
        start_trigger=DigitalStartTrigger(source="/Dev1/PFI1"),
        frame_count=2,
    )

    def acknowledge(samples: NDArray[np.float64]) -> int:
        """Verify repetition is configured before acknowledging one frame."""
        assert ni.task.triggers.start_trigger.retriggerable is True
        assert ni.task.out_stream.regen_mode is (
            constants.RegenerationMode.ALLOW_REGENERATION
        )
        assert samples.shape == (2, 3)
        np.testing.assert_allclose(samples, [[4, 0, -4], [-1, 1, 7]], atol=1e-12)
        ni.task.control.assert_not_called()
        return 3

    ni.writer.write_many_sample.side_effect = acknowledge
    try:
        output.prepare()
        assert output.state is OutputState.PREPARED
        assert ni.task.ao_channels.add_ao_voltage_chan.call_args_list == [
            call(physical_channel="Dev1/ao1", min_val=-4.0, max_val=4.0),
            call(physical_channel="Dev1/ao0", min_val=-1.0, max_val=7.0),
        ]
        ni.task.timing.cfg_samp_clk_timing.assert_called_once_with(
            rate=1000.0,
            source="",
            active_edge=constants.Edge.RISING,
            sample_mode=constants.AcquisitionType.FINITE,
            samps_per_chan=3,
        )
        ni.task.triggers.start_trigger.cfg_dig_edge_start_trig.assert_called_once_with(
            trigger_source="/Dev1/PFI1", trigger_edge=constants.Edge.RISING
        )
        ni.writer_factory.assert_called_once_with(ni.task.out_stream, auto_start=False)
        ni.writer.write_many_sample.assert_called_once()
        ni.task.control.assert_called_once_with(constants.TaskMode.TASK_COMMIT)
        ni.task.start.assert_not_called()
        ni.task.write.assert_not_called()
        assert x.angle is None
        assert y.angle is None
    finally:
        output.close()
    ni.task.stop.assert_not_called()
    ni.task.close.assert_called_once_with()


def test_partial_frames_do_not_establish_completion(
    session: NIRetriggeredGalvoOutputSession,
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
) -> None:
    """Require all frames, retaining reservations through timeout and completion.

    Each frame contains three samples per channel. Progress at two, three,
    and five samples is insufficient for this two-frame run. Six samples
    complete it even though native is_task_done() remains False. Completion
    disarms once, while close remains responsible for releasing the axis.
    """
    session.prepare()
    session.start()
    for generated, complete_frames in [(2, 0), (3, 1), (5, 1)]:
        ni.generated.return_value = generated
        assert session.wait(timeout=0) is False
        assert session.completed_frames == complete_frames
        assert session.state is OutputState.STARTED
        ni.task.stop.assert_not_called()

    ni.generated.return_value = 6
    assert session.wait(timeout=0) is True
    assert session.state is OutputState.COMPLETED
    assert session.completed_frames == 2
    ni.task.stop.assert_called_once_with()
    ni.task.close.assert_not_called()
    with pytest.raises(RuntimeError, match="reserved"):
        axes[0].close()
    reads = ni.generated.call_count
    assert session.wait(timeout=0) is True
    assert ni.generated.call_count == reads
    session.stop()
    session.close()
    ni.task.stop.assert_called_once_with()
    ni.task.close.assert_called_once_with()
    ni.task.wait_until_done.assert_not_called()
    axes[0].close()


@pytest.mark.parametrize("timeout", [None, 0.02])
def test_wait_observes_progress_between_short_sleeps(
    session: NIRetriggeredGalvoOutputSession,
    ni: FakeNI,
    clock: FakeClock,
    timeout: float | None,
) -> None:
    """Poll actual sample progress for both bounded and unbounded waits."""
    session.prepare()
    session.start()
    ni.generated.side_effect = [0, 3, 6]
    assert session.wait(timeout=timeout) is True
    assert clock.sleeps == [0.005, 0.005]
    assert session.completed_frames == 2
    ni.task.start.assert_called_once_with()
    ni.task.stop.assert_called_once_with()


def test_timeout_uses_one_budget_and_is_retryable(
    session: NIRetriggeredGalvoOutputSession,
    ni: FakeNI,
    clock: FakeClock,
) -> None:
    """Exhaust one deadline without disarming, then complete in a later wait."""
    session.prepare()
    session.start()
    assert session.wait(timeout=0.012) is False
    np.testing.assert_allclose(clock.sleeps, [0.005, 0.005, 0.002], atol=1e-12)
    assert session.state is OutputState.STARTED
    ni.task.stop.assert_not_called()
    ni.task.close.assert_not_called()
    ni.generated.return_value = 6
    assert session.wait(timeout=0) is True


@pytest.mark.parametrize("frame_count", [1, 3])
def test_completion_target_scales_with_frame_count(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
    frame_count: int,
) -> None:
    """Use samples per frame times requested frames, including a single frame."""
    output = NIRetriggeredGalvoOutputSession(
        {axes[0]: _waveform(0, 1, 2)},
        start_trigger=DigitalStartTrigger(source="/Dev1/PFI1"),
        frame_count=frame_count,
    )
    try:
        output.prepare()
        output.start()
        ni.generated.return_value = 3 * frame_count - 1
        assert output.wait(timeout=0) is False
        assert output.completed_frames == frame_count - 1
        ni.generated.return_value = 3 * frame_count
        assert output.wait(timeout=0) is True
        assert output.completed_frames == frame_count
    finally:
        output.close()


@pytest.mark.parametrize("explicit_stop", [False, True])
def test_cancellation_disarms_once_without_static_output(
    session: NIRetriggeredGalvoOutputSession,
    ni: FakeNI,
    explicit_stop: bool,
) -> None:
    """Stop explicitly or during close without replaying or parking the scanner."""
    session.prepare()
    session.start()
    if explicit_stop:
        session.stop()
        session.stop()
        assert session.wait(timeout=0) is False
        ni.task.close.assert_not_called()
    session.close()
    session.close()
    ni.task.stop.assert_called_once_with()
    ni.task.close.assert_called_once_with()
    ni.task.write.assert_not_called()
    ni.writer.write_many_sample.assert_called_once()


@pytest.mark.parametrize("stage", ["retrigger", "regeneration", "commit"])
def test_preparation_failure_releases_task_and_axis(
    session: NIRetriggeredGalvoOutputSession,
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    """Release partial preparation when repetition configuration is unsupported."""
    failure = errors.DaqError("unsupported configuration", -200452)
    if stage == "commit":
        ni.task.control.side_effect = failure
    else:
        target, attribute = (
            (ni.task.triggers.start_trigger, "retriggerable")
            if stage == "retrigger"
            else (ni.task.out_stream, "regen_mode")
        )
        monkeypatch.setattr(
            type(target), attribute, PropertyMock(side_effect=failure), raising=False
        )
    with pytest.raises(errors.DaqError) as caught:
        session.prepare()
    assert caught.value is failure
    assert session.state is OutputState.CLOSED
    ni.task.stop.assert_not_called()
    ni.task.close.assert_called_once_with()
    axes[0].close()


@pytest.mark.parametrize("stage", ["start", "status", "count", "stop"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_execution_failure_preserves_primary_error(
    session: NIRetriggeredGalvoOutputSession,
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
    stage: str,
    cleanup_fails: bool,
) -> None:
    """Disarm and release on native errors, preserving a failed close as a note.

    A start failure may follow a partial arm. A stop failure occurs when the
    complete sample count has been observed. Both must attempt native stop
    only once, still attempt close, and release the borrowed scanner.
    """
    session.prepare()
    if stage != "start":
        session.start()
    failure = errors.DaqError(f"{stage} failed", -200452)
    if stage == "start":
        ni.task.start.side_effect = failure
    elif stage == "status":
        ni.task.is_task_done.side_effect = failure
    elif stage == "count":
        ni.generated.side_effect = failure
    else:
        ni.generated.return_value = 6
        ni.task.stop.side_effect = failure
    if cleanup_fails:
        ni.task.close.side_effect = RuntimeError("cleanup failed")
    operation = session.start if stage == "start" else partial(session.wait, timeout=0)
    with pytest.raises(errors.DaqError) as caught:
        operation()
    assert caught.value is failure
    assert session.state is OutputState.CLOSED
    ni.task.stop.assert_called_once_with()
    ni.task.close.assert_called_once_with()
    if cleanup_fails:
        assert any("cleanup failed" in note for note in failure.__notes__)
    axes[0].close()


@pytest.mark.parametrize(
    ("counts", "done", "message"),
    [
        ((3, 2), False, "moved backwards"),
        ((7,), False, "more than the expected"),
        ((0,), True, "before all frames"),
    ],
)
def test_invalid_progress_is_fatal(
    session: NIRetriggeredGalvoOutputSession,
    ni: FakeNI,
    counts: tuple[int, ...],
    done: bool,
    message: str,
) -> None:
    """Reject a regressing count, excess output, or premature native completion."""
    session.prepare()
    session.start()
    ni.generated.side_effect = list(counts)
    ni.task.is_task_done.return_value = done
    if len(counts) == 2:
        assert session.wait(timeout=0) is False
    with pytest.raises(RuntimeError, match=message):
        session.wait(timeout=0)
    assert session.state is OutputState.CLOSED
    ni.task.stop.assert_called_once_with()
    ni.task.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("frame_count", "error_type"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_constructor_rejects_invalid_frame_count(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
    frame_count: object,
    error_type: type[Exception],
) -> None:
    """Reject noninteger or nonpositive targets without creating a native task."""
    with pytest.raises(error_type, match="frame_count"):
        NIRetriggeredGalvoOutputSession(
            {axes[0]: _waveform(0, 1, 2)},
            start_trigger=DigitalStartTrigger(source="/Dev1/PFI1"),
            frame_count=cast(int, frame_count),
        )
    ni.factory.assert_not_called()


def test_constructor_requires_an_explicit_frame_trigger(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
) -> None:
    """Reject None rather than silently allowing an ungated repeated run."""
    with pytest.raises(TypeError, match="start_trigger"):
        NIRetriggeredGalvoOutputSession(
            {axes[0]: _waveform(0, 1, 2)},
            start_trigger=cast(DigitalStartTrigger, None),
            frame_count=1,
        )
    ni.factory.assert_not_called()


def test_unlimited_output_requires_explicit_stop(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
) -> None:
    """An unlimited run has no finite completion wait."""
    output = NIRetriggeredGalvoOutputSession(
        {axes[0]: _waveform(0, 1, 2)},
        start_trigger=DigitalStartTrigger(source="/Dev1/PFI1"),
        frame_count=None,
    )

    try:
        output.prepare()
        output.start()

        assert output.frame_count is None

        with pytest.raises(RuntimeError, match="no completion target"):
            output.wait(timeout=0)

        # Rejecting the wait did not cancel an otherwise usable run.
        assert output.state is OutputState.STARTED
        ni.task.stop.assert_not_called()

        output.stop()
        ni.task.stop.assert_called_once_with()
    finally:
        output.close()
