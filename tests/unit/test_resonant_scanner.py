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
        self.ao_channels = self
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
        self.channel = (physical_channel, min_val, max_val)
        if self.failure_at == "configure":
            raise self.failure

    def write(self, data: float, *, auto_start: bool) -> int:
        self.writes.append((data, auto_start))
        if self.failure_at == "write":
            raise self.failure
        return self.samples_written

    def close(self) -> None:
        self.close_calls += 1
        if self.close_failure is not None:
            raise self.close_failure


class FakeTasks:
    """Supply fresh tasks, or an explicitly queued task."""

    def __init__(self) -> None:
        self.created: list[FakeTask] = []
        self.next_task: FakeTask | None = None

    def __call__(self) -> FakeTask:
        task = self.next_task if self.next_task is not None else FakeTask()
        self.next_task = None
        self.created.append(task)
        return task


@pytest.fixture(autouse=True)
def tasks(monkeypatch: pytest.MonkeyPatch) -> FakeTasks:
    factory = FakeTasks()
    monkeypatch.setattr(nidaqmx, "Task", factory)
    return factory


@pytest.fixture
def config() -> NIResonantScannerConfig:
    return NIResonantScannerConfig(
        amplitude_control_channel="Dev1/ao0",
        command_voltage_range=units.VoltageRange("0 V", "5 V"),
        nominal_frequency=units.Frequency("7.91 kHz"),
        max_amplitude=units.Angle("26 deg"),
        settling_time=units.Time("100 ms"),
        frequency_tolerance=0.003,
    )


def test_construction_checks_channel_without_writing(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
) -> None:
    scanner = NIResonantScanner(config)

    assert scanner.config is config
    assert scanner.amplitude is None
    assert len(tasks.created) == 1

    task = tasks.created[0]
    assert task.channel == ("Dev1/ao0", 0.0, 5.0)
    assert task.writes == []
    assert task.close_calls == 1
    scanner.close()


@pytest.mark.parametrize("voltage_min", [0.0, 1.0])
def test_commands_use_nominal_full_scale(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
    voltage_min: float,
) -> None:
    config = replace(
        config,
        command_voltage_range=units.VoltageRange(voltage_min, 5.0),
    )
    scanner = NIResonantScanner(config)

    for degrees, fraction in [(0, 0.0), (13, 0.5), (26, 1.0)]:
        amplitude = units.Angle(f"{degrees} deg")
        scanner.amplitude = amplitude

        task = tasks.created[-1]
        assert task.channel == ("Dev1/ao0", voltage_min, 5.0)
        assert len(task.writes) == 1

        voltage, auto_start = task.writes[0]
        expected = voltage_min + fraction * (5 - voltage_min)
        assert voltage == pytest.approx(expected)
        assert auto_start is True
        assert task.close_calls == 1
        assert scanner.amplitude == amplitude

    assert len(tasks.created) == 4
    scanner.close()


def test_invalid_amplitude_does_not_create_task(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
) -> None:
    scanner = NIResonantScanner(config)

    with pytest.raises(ValueError, match="must be between zero and max"):
        scanner.amplitude = units.Angle("27 deg")

    assert len(tasks.created) == 1
    assert scanner.amplitude is None
    scanner.close()


@pytest.mark.parametrize("during_construction", [True, False])
@pytest.mark.parametrize("cleanup_fails", [True, False])
def test_primary_failure_survives_cleanup(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
    during_construction: bool,
    cleanup_fails: bool,
) -> None:
    scanner = None if during_construction else NIResonantScanner(config)
    if scanner is not None:
        scanner.amplitude = units.Angle("6 deg")

    task = FakeTask(failure_at="configure" if during_construction else "write")
    if cleanup_fails:
        task.close_failure = RuntimeError("cleanup failed")
    tasks.next_task = task

    with pytest.raises(RuntimeError) as caught:  # ruff: ignore[pytest-raises-with-multiple-statements]
        if during_construction:
            NIResonantScanner(config)
        else:
            assert scanner is not None
            scanner.amplitude = units.Angle("13 deg")

    assert caught.value is task.failure
    assert task.close_calls == 1

    if cleanup_fails:
        assert any("cleanup failed" in note for note in caught.value.__notes__)

    if scanner is not None:
        assert scanner.amplitude == units.Angle("6 deg")
        scanner.close()


@pytest.mark.parametrize("during_construction", [True, False])
def test_cleanup_failure_propagates(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
    during_construction: bool,
) -> None:
    scanner = None if during_construction else NIResonantScanner(config)
    task = FakeTask()
    task.close_failure = RuntimeError("cleanup failed")
    tasks.next_task = task

    with pytest.raises(RuntimeError) as caught:  # ruff: ignore[pytest-raises-with-multiple-statements]
        if during_construction:
            NIResonantScanner(config)
        else:
            assert scanner is not None
            scanner.amplitude = units.Angle("13 deg")

    assert caught.value is task.close_failure
    assert task.close_calls == 1

    if scanner is not None:
        assert scanner.amplitude is None
        scanner.close()


def test_incomplete_write_does_not_update_amplitude(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
) -> None:
    scanner = NIResonantScanner(config)
    task = FakeTask()
    task.samples_written = 0
    tasks.next_task = task

    with pytest.raises(RuntimeError, match="exactly one sample"):
        scanner.amplitude = units.Angle("13 deg")

    assert scanner.amplitude is None
    assert task.close_calls == 1
    scanner.close()


def test_close_is_idempotent_and_does_not_command_idle(
    config: NIResonantScannerConfig,
    tasks: FakeTasks,
) -> None:
    scanner = NIResonantScanner(config)
    scanner.amplitude = units.Angle("13 deg")

    scanner.close()
    scanner.close()

    with pytest.raises(RuntimeError, match="closed"):
        scanner.amplitude = units.Angle("0 deg")

    assert len(tasks.created) == 2
    assert tasks.created[0].writes == []
    assert tasks.created[1].writes == [(pytest.approx(2.5), True)]
    assert all(task.close_calls == 1 for task in tasks.created)


def test_snapshot_includes_ni_settings(config: NIResonantScannerConfig) -> None:

    scanner = NIResonantScanner(config)

    snapshot = scanner.snapshot()

    assert snapshot["settings"]["amplitude_control_channel"] == "Dev1/ao0"
    assert snapshot["settings"]["command_voltage_range"] == {
        "min": {"value": 0.0, "unit": "V"},
        "max": {"value": 5.0, "unit": "V"},
    }
    assert snapshot["driver"]["distribution"] == "dirigo-nidaq"
    assert snapshot["driver"]["version"] == importlib_metadata.version("dirigo-nidaq")
    scanner.close()


def test_entry_point_loads_without_creating_task(
    tasks: FakeTasks,
) -> None:
    entry_points = importlib_metadata.entry_points(
        group="dirigo.devices.scanners",
        name="nidaq_resonant",
    )

    assert len(entry_points) == 1
    assert next(iter(entry_points)).load() is NIResonantScanner
    assert NIResonantScanner.config_type is NIResonantScannerConfig
    assert tasks.created == []


def test_constructor_rejects_incorrect_config_type() -> None:
    """Reject an object that is not an NI resonant scanner configuration."""
    with pytest.raises(
        TypeError,
        match="config must be an NIResonantScannerConfig",
    ):
        NIResonantScanner(cast(NIResonantScannerConfig, object()))


def test_driver_identity_allows_missing_distribution_metadata(
    config: NIResonantScannerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve driver identity when installed version metadata is unavailable.

    Force the distribution lookup to raise PackageNotFoundError. Identity
    retrieval must still succeed, retaining the distribution name while
    representing the unavailable version as None.
    """

    def missing_version(distribution_name: str) -> str:
        """Simulate unavailable metadata for the requested distribution."""
        raise importlib_metadata.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(
        importlib_metadata,
        "version",
        missing_version,
    )
    scanner = NIResonantScanner(config)

    try:
        identity = scanner.driver_identity

        assert identity.distribution == "dirigo-nidaq"
        assert identity.version is None
    finally:
        scanner.close()
