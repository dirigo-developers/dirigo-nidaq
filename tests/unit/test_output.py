"""Test finite NI waveform output through mocked SDK boundaries."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from unittest.mock import Mock, call

import nidaqmx
import numpy as np
import pytest
from dirigo import units
from dirigo.hardware.output import OutputState
from dirigo.hardware.scanner import GalvoScannerConfig
from dirigo.hardware.simulated import SimulatedGalvoScanner
from dirigo.hardware.timing import DigitalStartTrigger, ExternalSampleClock, SignalEdge
from dirigo.hardware.waveform import AngularWaveform
from nidaqmx import constants, errors, stream_writers

from dirigo_nidaq.output import NIGalvoOutputSession
from dirigo_nidaq.scanner import NIGalvoScanner, NIGalvoScannerConfig


@dataclass(frozen=True)
class FakeNI:
    """Expose SDK factories and their recording task and writer."""

    task_factory: Mock
    writer_factory: Mock
    task: Mock
    writer: Mock


@pytest.fixture(autouse=True)
def ni(monkeypatch: pytest.MonkeyPatch) -> FakeNI:
    """Replace task and writer creation without requiring NI driver libraries."""
    task = Mock()
    writer = Mock()
    writer.write_many_sample.return_value = 3
    task_factory = Mock(return_value=task)
    writer_factory = Mock(return_value=writer)

    monkeypatch.setattr(nidaqmx, "Task", task_factory)
    monkeypatch.setattr(stream_writers, "AnalogMultiChannelWriter", writer_factory)

    return FakeNI(task_factory, writer_factory, task, writer)


@pytest.fixture
def axes() -> Iterator[tuple[NIGalvoScanner, NIGalvoScanner]]:
    """Provide two axes with different full-scale mappings and inversion."""
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
    """Convert readable optical angles into uniformly sampled radians."""
    return AngularWaveform(
        [float(units.Angle(f"{value} deg")) for value in degrees],
        sample_rate=units.SampleRate("1 kS/s"),
    )


def test_construction_and_unused_close_do_not_create_task(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
) -> None:
    """Validate and close an unused session without touching the NI SDK."""
    session = NIGalvoOutputSession({axes[0]: _waveform(0, 1, 2)})

    assert session.state is OutputState.NEW
    session.close()
    session.close()

    assert session.state is OutputState.CLOSED
    ni.task_factory.assert_not_called()
    ni.writer_factory.assert_not_called()


def test_prepare_preserves_channel_order_and_converts_each_axis(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
) -> None:
    """Preload a finite task using per-axis mapping and binding order.

    Insert y before x to prove that channel registration and matrix rows follow
    the caller's order. Different ranges, an asymmetric optical range, and an
    inverted y axis exercise both voltage coefficients for each channel.
    """
    x, y = axes
    session = NIGalvoOutputSession({y: _waveform(-20, 0, 20), x: _waveform(-10, 0, 30)})
    try:
        session.prepare()

        ni.task.triggers.start_trigger.cfg_dig_edge_start_trig.assert_not_called()
        assert session.state is OutputState.PREPARED
        ni.task_factory.assert_called_once_with()
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
        ni.writer_factory.assert_called_once_with(ni.task.out_stream, auto_start=False)
        ni.writer.write_many_sample.assert_called_once()

        samples = ni.writer.write_many_sample.call_args.args[0]
        assert samples.dtype == np.float64
        assert samples.flags.c_contiguous
        np.testing.assert_allclose(samples, [[4, 0, -4], [-1, 1, 7]], atol=1e-12)

        ni.task.start.assert_not_called()
        ni.task.write.assert_not_called()
        ni.task.close.assert_not_called()
        assert x.angle is None
        assert y.angle is None
    finally:
        session.close()

    ni.task.stop.assert_not_called()
    ni.task.close.assert_called_once_with()


def test_single_axis_still_uses_a_two_dimensional_sample_matrix(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
) -> None:
    """Represent one channel as one matrix row for the multichannel writer."""
    session = NIGalvoOutputSession({axes[0]: _waveform(-10, 0, 30)})
    try:
        session.prepare()

        samples = ni.writer.write_many_sample.call_args.args[0]
        assert samples.shape == (1, 3)
        np.testing.assert_allclose(samples, [[-1, 1, 7]], atol=1e-12)
    finally:
        session.close()


def test_timeout_keeps_output_active_until_later_completion(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
) -> None:
    """Translate a completion timeout without stopping or releasing the axis.

    The first wait raises NI's completion-timeout error. A later unbounded wait
    succeeds. Completion retains the reservation until close, after which a
    fresh session can prepare with the same scanner.
    """
    axis = axes[0]
    session = NIGalvoOutputSession({axis: _waveform(0, 1, 2)})
    ni.task.wait_until_done.side_effect = [
        errors.DaqError("Not done yet.", -200560),
        None,
    ]
    try:
        session.prepare()
        session.start()

        assert not session.wait(0.0)
        assert session.state is OutputState.STARTED
        ni.task.stop.assert_not_called()
        ni.task.close.assert_not_called()

        assert session.wait()
        assert session.state is OutputState.COMPLETED
        with pytest.raises(RuntimeError, match="reserved"):
            axis.close()
    finally:
        session.close()

    assert ni.task.wait_until_done.call_args_list == [
        call(timeout=0.0),
        call(timeout=constants.WAIT_INFINITELY),
    ]
    ni.task.start.assert_called_once_with()
    ni.task.stop.assert_not_called()
    ni.task.close.assert_called_once_with()
    assert axis.angle is None

    following = NIGalvoOutputSession({axis: _waveform(0, 1, 2)})
    try:
        following.prepare()
    finally:
        following.close()


@pytest.mark.parametrize("explicit_stop", [False, True])
def test_stop_and_active_close_do_not_issue_static_commands(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
    explicit_stop: bool,
) -> None:
    """Stop unfinished output once without changing static-command history.

    Exercise both an explicit stop and closing a running session. Explicit
    stop retains the reservation until close, and repeated stop or close calls
    do not repeat native cleanup.
    """
    axis = axes[0]
    session = NIGalvoOutputSession({axis: _waveform(0, 1, 2)})
    try:
        session.prepare()
        session.start()

        if explicit_stop:
            session.stop()
            session.stop()
            assert session.state is OutputState.STOPPED
            assert not session.wait(0.0)
            with pytest.raises(RuntimeError, match="reserved"):
                axis.close()
    finally:
        session.close()

    session.close()
    ni.task.stop.assert_called_once_with()
    ni.task.close.assert_called_once_with()
    ni.task.wait_until_done.assert_not_called()
    ni.task.write.assert_not_called()
    ni.task_factory.assert_called_once_with()
    assert axis.angle is None
    axis.close()


@pytest.mark.parametrize(
    ("channel", "message"),
    [
        ("Dev2/ao1", "same NI device"),
        ("Dev1/ao0", "distinct"),
        ("/dev1/AO00", "distinct"),
        ("Dev1/ao0:1", "one NI AO channel"),
        ("Dev1/ao0,Dev1/ao1", "one NI AO channel"),
        ("Dev*/ao1", "one NI AO channel"),
    ],
)
def test_invalid_channel_groups_are_rejected_without_a_task(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
    channel: str,
    message: str,
) -> None:
    """Reject multiple devices, duplicate channels, and channel expressions.

    Separate scanner objects can name the same physical channel. Duplicate
    detection accounts for a leading slash, case, and leading numeric zeros.
    """
    x, y = axes
    other = NIGalvoScanner(replace(y.config, angle_control_channel=channel))
    try:
        with pytest.raises(ValueError, match=message):
            NIGalvoOutputSession({x: _waveform(0, 1, 2), other: _waveform(0, 1, 2)})
    finally:
        other.close()

    ni.task_factory.assert_not_called()


def test_non_ni_scanner_is_rejected_without_a_task(ni: FakeNI) -> None:
    """Reject an otherwise compatible galvo implemented by another backend."""
    axis = SimulatedGalvoScanner(
        GalvoScannerConfig(angle_range=units.AngleRange("-10 deg", "10 deg"))
    )
    try:
        with pytest.raises(TypeError, match="NIGalvoScanner"):
            NIGalvoOutputSession({axis: _waveform(0, 1, 2)})
    finally:
        axis.close()

    ni.task_factory.assert_not_called()


@pytest.mark.parametrize(
    "stage",
    ["channel", "timing", "trigger", "writer", "preload", "start", "wait", "stop"],
)
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_native_failure_closes_task_and_releases_scanner(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
    stage: str,
    cleanup_fails: bool,
) -> None:
    """Preserve a native failure through partial preparation or active output.

    Inject an NI error at each task operation. Once created, the task must be
    closed exactly once, and a potentially started task must also be stopped.
    A cleanup failure becomes a note; the scanner reservation is always freed.
    """
    operations = {
        "channel": ni.task.ao_channels.add_ao_voltage_chan,
        "timing": ni.task.timing.cfg_samp_clk_timing,
        "trigger": ni.task.triggers.start_trigger.cfg_dig_edge_start_trig,
        "writer": ni.writer_factory,
        "preload": ni.writer.write_many_sample,
        "start": ni.task.start,
        "wait": ni.task.wait_until_done,
        "stop": ni.task.stop,
    }
    failure = errors.DaqError("NI operation failed.", -200077)
    operations[stage].side_effect = failure
    if cleanup_fails:
        ni.task.close.side_effect = RuntimeError("Cleanup failed.")

    axis = axes[0]
    session = NIGalvoOutputSession(
        {axis: _waveform(0, 1, 2)},
        start_trigger=(
            DigitalStartTrigger(source="/Dev1/PFI1") if stage == "trigger" else None
        ),
    )
    action: Callable[[], object] = session.prepare
    if stage in ("start", "wait", "stop"):
        session.prepare()
        action = session.start
    if stage in ("wait", "stop"):
        session.start()
        action = session.wait if stage == "wait" else session.stop

    with pytest.raises(errors.DaqError) as caught:
        action()

    assert caught.value is failure
    assert session.state is OutputState.CLOSED
    ni.task.close.assert_called_once_with()
    if stage in ("start", "wait", "stop"):
        ni.task.stop.assert_called_once_with()
    else:
        ni.task.stop.assert_not_called()
    if cleanup_fails:
        assert any(
            "Cleanup failed" in note for note in getattr(failure, "__notes__", ())
        )

    session.close()
    ni.task.close.assert_called_once_with()
    axis.close()


def test_task_creation_failure_releases_scanner(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
) -> None:
    """Release the reservation when task creation fails before ownership begins."""
    failure = RuntimeError("Task creation failed.")
    ni.task_factory.side_effect = failure
    session = NIGalvoOutputSession({axes[0]: _waveform(0, 1, 2)})

    with pytest.raises(RuntimeError, match="Task creation failed") as caught:
        session.prepare()

    assert caught.value is failure
    assert session.state is OutputState.CLOSED
    ni.task.close.assert_not_called()
    axes[0].close()


def test_incomplete_preload_closes_task_and_releases_scanner(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
) -> None:
    """Reject a short per-channel write before allowing the session to start."""
    ni.writer.write_many_sample.return_value = 2
    session = NIGalvoOutputSession({axes[0]: _waveform(0, 1, 2)})

    with pytest.raises(RuntimeError, match="every sample per channel"):
        session.prepare()

    assert session.state is OutputState.CLOSED
    ni.task.start.assert_not_called()
    ni.task.close.assert_called_once_with()
    axes[0].close()


def test_close_failure_is_reported_once_and_releases_scanner(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
) -> None:
    """Report failed cleanup while permanently closing and freeing the axis."""
    session = NIGalvoOutputSession({axes[0]: _waveform(0, 1, 2)})
    session.prepare()
    failure = RuntimeError("Cleanup failed.")
    ni.task.close.side_effect = failure

    with pytest.raises(RuntimeError, match="Cleanup failed") as caught:
        session.close()

    assert caught.value is failure
    assert session.state is OutputState.CLOSED
    session.close()
    ni.task.close.assert_called_once_with()
    axes[0].close()


def test_require_task_rejects_unprepared_session(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
) -> None:
    """Reject internal task access before preparation without creating a task.

    Call the helper directly because public lifecycle methods reject an
    unprepared session before reaching this internal guard.
    """
    session = NIGalvoOutputSession({axes[0]: _waveform(0, 1, 2)})

    try:
        with pytest.raises(RuntimeError, match="NI output task is not prepared"):
            session._require_task()
    finally:
        session.close()

    ni.task_factory.assert_not_called()


@pytest.fixture
def external_clock() -> ExternalSampleClock:
    """Provide a core clock description with an NI terminal identifier."""
    return ExternalSampleClock(
        source="/Dev1/PFI0",
        max_rate=units.SampleRate("1.2 kS/s"),
    )


@pytest.mark.parametrize(
    ("edge", "ni_edge"),
    [
        (SignalEdge.RISING, constants.Edge.RISING),
        (SignalEdge.FALLING, constants.Edge.FALLING),
    ],
)
@pytest.mark.parametrize("max_rate", [1000.0, 1200.0])
def test_external_clock_is_forwarded_during_prepare(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    external_clock: ExternalSampleClock,
    ni: FakeNI,
    edge: SignalEdge,
    ni_edge: constants.Edge,
    max_rate: float,
) -> None:
    """Translate the core edge and forward the source and maximum rate to NI.

    The waveform remains nominally 1 kS/s. Task timing receives the maximum
    expected external rate, and preparation still does not start output.
    """
    clock = replace(
        external_clock,
        max_rate=units.SampleRate(max_rate),
        active_edge=edge,
    )
    waveform = _waveform(0, 1, 2)
    session = NIGalvoOutputSession({axes[0]: waveform}, external_clock=clock)
    try:
        assert session.external_clock is clock
        ni.task_factory.assert_not_called()
        session.prepare()

        ni.task.timing.cfg_samp_clk_timing.assert_called_once_with(
            rate=max_rate,
            source="/Dev1/PFI0",
            active_edge=ni_edge,
            sample_mode=constants.AcquisitionType.FINITE,
            samps_per_chan=3,
        )
        assert waveform.sample_rate == units.SampleRate("1 kS/s")
        ni.task.start.assert_not_called()
    finally:
        session.close()


def test_external_clock_timeout_keeps_session_started(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    external_clock: ExternalSampleClock,
    ni: FakeNI,
) -> None:
    """Keep externally clocked output active when NI reports incomplete output.

    Simulate an NI completion timeout, as can occur when clock pulses are
    absent or insufficient. A bounded wait must return False and retain the
    task; closing must then stop it and release the scanner reservation.
    """
    session = NIGalvoOutputSession(
        {axes[0]: _waveform(0, 1, 2)}, external_clock=external_clock
    )
    ni.task.wait_until_done.side_effect = errors.DaqError("Not done yet.", -200560)

    try:
        session.prepare()
        session.start()
        assert not session.wait(timeout=0.25)
        assert session.state is OutputState.STARTED
        ni.task.wait_until_done.assert_called_once_with(timeout=0.25)
        ni.task.close.assert_not_called()
    finally:
        session.close()

    ni.task.stop.assert_called_once_with()
    ni.task.close.assert_called_once_with()
    axes[0].close()


@pytest.mark.parametrize("use_external_clock", [False, True])
@pytest.mark.parametrize(
    ("edge", "ni_edge"),
    [
        (SignalEdge.RISING, constants.Edge.RISING),
        (SignalEdge.FALLING, constants.Edge.FALLING),
    ],
)
def test_start_trigger_is_configured_before_arming(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    external_clock: ExternalSampleClock,
    ni: FakeNI,
    edge: SignalEdge,
    ni_edge: constants.Edge,
    use_external_clock: bool,
) -> None:
    """Translate trigger edges independently of the selected sample clock.

    Preparation configures the digital trigger and preloads samples without
    starting. start() delegates to NI start without performing a completion
    wait. Closing this unfinished session must stop and release the task.
    """
    trigger = DigitalStartTrigger(source="/Dev1/PFI1", active_edge=edge)
    session = NIGalvoOutputSession(
        {axes[0]: _waveform(0, 1, 2)},
        external_clock=external_clock if use_external_clock else None,
        start_trigger=trigger,
    )
    try:
        assert session.start_trigger is trigger
        ni.task_factory.assert_not_called()
        session.prepare()

        ni.task.triggers.start_trigger.cfg_dig_edge_start_trig.assert_called_once_with(
            trigger_source="/Dev1/PFI1", trigger_edge=ni_edge
        )
        ni.writer_factory.assert_called_once_with(ni.task.out_stream, auto_start=False)
        ni.task.start.assert_not_called()

        session.start()
        assert session.state is OutputState.STARTED
        ni.task.start.assert_called_once_with()
        ni.task.wait_until_done.assert_not_called()
    finally:
        session.close()

    ni.task.stop.assert_called_once_with()
    ni.task.close.assert_called_once_with()
    axes[0].close()


@pytest.mark.parametrize("complete_later", [False, True])
def test_trigger_timeout_allows_completion_or_cancellation(
    axes: tuple[NIGalvoScanner, NIGalvoScanner],
    ni: FakeNI,
    complete_later: bool,
) -> None:
    """Retain an armed run after timeout, then complete it or cancel on close.

    Mock NI reporting incomplete output followed by completion. If the caller
    instead closes after the timeout, the adapter must stop the unfinished
    task. The scanner reservation is released in either case.
    """
    session = NIGalvoOutputSession(
        {axes[0]: _waveform(0, 1, 2)},
        start_trigger=DigitalStartTrigger(source="/Dev1/PFI1"),
    )
    ni.task.wait_until_done.side_effect = [
        errors.DaqError("Not done yet.", -200560),
        None,
    ]
    try:
        session.prepare()
        session.start()
        assert not session.wait(timeout=0.25)
        assert session.state is OutputState.STARTED
        ni.task.stop.assert_not_called()
        ni.task.close.assert_not_called()

        if complete_later:
            assert session.wait(timeout=0.25)
            assert session.state is OutputState.COMPLETED
    finally:
        session.close()

    if complete_later:
        ni.task.stop.assert_not_called()
    else:
        ni.task.stop.assert_called_once_with()
    ni.task.close.assert_called_once_with()
    axes[0].close()
