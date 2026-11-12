"""Test NI resonant scanner commands without requiring NI hardware."""

from dataclasses import replace
from importlib import metadata as importlib_metadata
from typing import cast

import nidaqmx
import pytest
from dirigo import units

from dirigo_nidaq.scanner import (
    NIResonantScanner,
    NIResonantScannerConfig,
)


class FakeTask:
    """Record NI calls and allow failures during task operations."""

    def __init__(self, *, failure_at: str | None = None) -> None:
        self.ao_channels: FakeTask = self
        self.failure_at = failure_at
        self.failure = RuntimeError("NI operation failed")
        self.channel: tuple[str, float, float] | None = None
        self.writes: list[tuple[float, bool]] = []
        self.close_calls = 0
        self.close_failure: RuntimeError | None = None
        self.samples_written = 1

    def add_ao_voltage_chan(
        self,
        physical_channel: str,
        *,
        min_val: float,
        max_val: float,
    ) -> None:
        """Record channel configuration and optionally reject it."""
        self.channel = (physical_channel, min_val, max_val)
        if self.failure_at == "configure":
            raise self.failure

    def write(self, data: float, *, auto_start: bool) -> int:
        """Record the write and return its configured acknowledgement."""
        self.writes.append((data, auto_start))
        if self.failure_at == "write":
            raise self.failure
        return self.samples_written

    def close(self) -> None:
        """Count cleanup attempts and optionally raise a cleanup error."""
        self.close_calls += 1
        if self.close_failure is not None:
            raise self.close_failure


class FakeTasks:
    """Supply fresh tasks, or an explicitly queued task."""

    def __init__(self) -> None:
        self.created: list[FakeTask] = []
        self.next_task: FakeTask | None = None

    def __call__(self) -> FakeTask:
        """Return the queued task or create a new recording task."""
        task = self.next_task if self.next_task is not None else FakeTask()
        self.next_task = None
        self.created.append(task)
        return task


@pytest.fixture(autouse=True)
def tasks(monkeypatch: pytest.MonkeyPatch) -> FakeTasks:
    """Replace NI task creation so every test runs without NI hardware."""
    factory = FakeTasks()
    monkeypatch.setattr(nidaqmx, "Task", factory)
    return factory


@pytest.fixture
def config() -> NIResonantScannerConfig:
    """Provide a scanner whose full optical amplitude corresponds to 5 V."""
    return NIResonantScannerConfig(
        amplitude_control_channel="Dev1/ao0",
        command_voltage_range=units.VoltageRange("0 V", "5 V"),
        nominal_frequency=units.Frequency("7.91 kHz"),
        max_amplitude=units.Angle("26 deg"),
        settling_time=units.Time("100 ms"),
        frequency_tolerance=0.003,
    )


def test_construction_does_not_create_task(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
) -> None:
    """Store configuration without creating a task or commanding amplitude.

    Closing an unused scanner also creates no task. NI channel configuration
    is deferred until the caller issues an amplitude command.
    """
    scanner = NIResonantScanner(config)

    try:
        assert scanner.config is config
        assert scanner.amplitude is None
        assert tasks.created == []
    finally:
        scanner.close()

    assert tasks.created == []


@pytest.mark.parametrize("voltage_min", [0.0, 1.0])
def test_commands_use_nominal_full_scale(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
    voltage_min: float,
) -> None:
    """Map zero, half, and full amplitude across the vendor voltage range.

    Each command creates and closes its own task. A nonzero lower voltage
    verifies that zero optical amplitude maps to the configured lower endpoint.
    """
    config = replace(
        config,
        command_voltage_range=units.VoltageRange(voltage_min, 5.0),
    )
    scanner = NIResonantScanner(config)

    try:
        for command_count, (degrees, fraction) in enumerate(
            [(0, 0.0), (13, 0.5), (26, 1.0)],
            start=1,
        ):
            amplitude = units.Angle(f"{degrees} deg")
            scanner.amplitude = amplitude

            assert len(tasks.created) == command_count
            task = tasks.created[-1]
            assert task.channel == ("Dev1/ao0", voltage_min, 5.0)
            assert len(task.writes) == 1

            voltage, auto_start = task.writes[0]
            expected = voltage_min + fraction * (5.0 - voltage_min)
            assert voltage == pytest.approx(expected)
            assert auto_start is True
            assert task.close_calls == 1
            assert scanner.amplitude == amplitude
    finally:
        scanner.close()

    assert len(tasks.created) == 3


def test_invalid_amplitude_does_not_create_task(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
) -> None:
    """Reject an out-of-range amplitude before creating an NI task."""
    scanner = NIResonantScanner(config)

    try:
        with pytest.raises(ValueError, match="must be between zero and max"):
            scanner.amplitude = units.Angle("27 deg")

        assert tasks.created == []
        assert scanner.amplitude is None
    finally:
        scanner.close()


@pytest.mark.parametrize("failure_at", ["configure", "write"])
@pytest.mark.parametrize("cleanup_fails", [True, False])
def test_primary_failure_survives_cleanup(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
    failure_at: str,
    cleanup_fails: bool,
) -> None:
    """Preserve a command failure and the previous successful amplitude.

    After one successful command, inject a channel-configuration or write
    failure into the next command. Cleanup must run once; if it also fails,
    its error is attached as a note without replacing the original exception.
    """
    scanner = NIResonantScanner(config)

    try:
        previous_amplitude = units.Angle("6 deg")
        scanner.amplitude = previous_amplitude

        task = FakeTask(failure_at=failure_at)
        if cleanup_fails:
            task.close_failure = RuntimeError("cleanup failed")
        tasks.next_task = task

        with pytest.raises(RuntimeError) as caught:
            scanner.amplitude = units.Angle("13 deg")

        assert caught.value is task.failure
        assert task.channel == ("Dev1/ao0", 0.0, 5.0)
        assert task.close_calls == 1
        assert len(tasks.created) == 2
        assert scanner.amplitude == previous_amplitude

        if cleanup_fails:
            assert any(
                "cleanup failed" in note
                for note in getattr(caught.value, "__notes__", ())
            )
    finally:
        scanner.close()


def test_cleanup_failure_propagates(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
) -> None:
    """Propagate cleanup failure even when the NI write succeeds.

    A successful write followed by failed task cleanup does not count as a
    successful scanner command, so the recorded amplitude remains unknown.
    """
    scanner = NIResonantScanner(config)

    try:
        task = FakeTask()
        task.close_failure = RuntimeError("cleanup failed")
        tasks.next_task = task

        with pytest.raises(RuntimeError) as caught:
            scanner.amplitude = units.Angle("13 deg")

        assert caught.value is task.close_failure
        assert task.writes == [(pytest.approx(2.5), True)]
        assert task.close_calls == 1
        assert len(tasks.created) == 1
        assert scanner.amplitude is None
    finally:
        scanner.close()


def test_incomplete_write_does_not_update_amplitude(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
) -> None:
    """Reject a zero-sample acknowledgement, close the task, and retain history."""
    scanner = NIResonantScanner(config)

    try:
        task = FakeTask()
        task.samples_written = 0
        tasks.next_task = task

        with pytest.raises(RuntimeError, match="exactly one sample"):
            scanner.amplitude = units.Angle("13 deg")

        assert scanner.amplitude is None
        assert task.close_calls == 1
        assert len(tasks.created) == 1
    finally:
        scanner.close()


def test_close_is_idempotent_and_does_not_command_idle(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
) -> None:
    """Close repeatedly without issuing another voltage command.

    Only the explicit amplitude command creates a task. Once closed, the
    scanner rejects further commands before creating any additional task.
    """
    scanner = NIResonantScanner(config)

    try:
        scanner.amplitude = units.Angle("13 deg")
        scanner.close()
        scanner.close()

        with pytest.raises(RuntimeError, match="closed"):
            scanner.amplitude = units.Angle("0 deg")

        assert len(tasks.created) == 1
        assert tasks.created[0].writes == [(pytest.approx(2.5), True)]
        assert tasks.created[0].close_calls == 1
    finally:
        scanner.close()


def test_snapshot_includes_ni_settings(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
) -> None:
    """Include NI configuration and installed driver metadata without a task."""
    scanner = NIResonantScanner(config)

    try:
        snapshot = scanner.snapshot()

        assert snapshot["settings"]["amplitude_control_channel"] == "Dev1/ao0"
        assert snapshot["settings"]["command_voltage_range"] == {
            "min": {"value": 0.0, "unit": "V"},
            "max": {"value": 5.0, "unit": "V"},
        }
        assert snapshot["driver"]["distribution"] == "dirigo-nidaq"
        assert snapshot["driver"]["version"] == importlib_metadata.version(
            "dirigo-nidaq"
        )
        assert tasks.created == []
    finally:
        scanner.close()

    assert tasks.created == []


def test_entry_point_loads_without_creating_task(
    tasks: FakeTasks,
) -> None:
    """Resolve the installed scanner registration without accessing hardware."""
    entry_points = importlib_metadata.entry_points(
        group="dirigo.devices.scanners",
        name="nidaq_resonant",
    )

    assert len(entry_points) == 1
    assert next(iter(entry_points)).load() is NIResonantScanner
    assert NIResonantScanner.config_type is NIResonantScannerConfig
    assert tasks.created == []


def test_constructor_rejects_incorrect_config_type(
    tasks: FakeTasks,
) -> None:
    """Reject an incompatible configuration before creating any NI task."""
    with pytest.raises(
        TypeError,
        match="config must be an NIResonantScannerConfig",
    ):
        NIResonantScanner(cast(NIResonantScannerConfig, object()))

    assert tasks.created == []


def test_driver_identity_allows_missing_distribution_metadata(
    config: NIResonantScannerConfig,
    monkeypatch: pytest.MonkeyPatch,
    tasks: FakeTasks,
) -> None:
    """Preserve driver identity when installed version metadata is unavailable.

    Force the distribution lookup to raise PackageNotFoundError. Identity
    retrieval retains the distribution name and returns None for its version,
    without creating an NI task.
    """

    def missing_version(distribution_name: str) -> str:
        """Simulate unavailable metadata for the requested distribution."""
        raise importlib_metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(importlib_metadata, "version", missing_version)
    scanner = NIResonantScanner(config)

    try:
        identity = scanner.driver_identity

        assert identity.distribution == "dirigo-nidaq"
        assert identity.version is None
        assert tasks.created == []
    finally:
        scanner.close()
