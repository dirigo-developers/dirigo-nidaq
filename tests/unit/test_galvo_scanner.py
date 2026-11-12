"""Test NI galvo configuration, nominal mapping, and scalar task ownership."""

from dataclasses import replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import nidaqmx
import pytest
from dirigo import units
from dirigo.hardware.config import load_hardware

from dirigo_nidaq.scanner import NIGalvoScanner, NIGalvoScannerConfig


@pytest.fixture
def config() -> NIGalvoScannerConfig:
    """Provide asymmetric optical and voltage ranges with a nonzero intercept."""
    return NIGalvoScannerConfig(
        angle_range=units.AngleRange("-10 deg", "30 deg"),
        angle_control_channel="Dev1/ao1",
        command_voltage_range=units.VoltageRange("-1 V", "7 V"),
    )


@pytest.fixture
def task_factory(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace the NI constructor with a mock scalar-output task."""
    factory = Mock()
    factory.return_value.write.return_value = 1
    monkeypatch.setattr(nidaqmx, "Task", factory)
    return factory


def test_loads_flat_configuration_without_creating_tasks(
    tmp_path: Path,
    config: NIGalvoScannerConfig,
    task_factory: Mock,
) -> None:
    """Parse inherited and NI fields from one table without accessing the DAQ.

    The real driver receives the parsed configuration through the categorized
    loader. Both construction and closure must leave task creation untouched.
    """
    path = tmp_path / "system_config.toml"
    path.write_text(
        """
[devices.slow_scanner]
category = "scanners"
driver = "nidaq_galvo"

[devices.slow_scanner.config]
angle_range = {min = "-10 deg", max = "30 deg"}
angle_control_channel = "Dev1/ao1"
command_voltage_range = {min = "-1 V", max = "7 V"}
inverted = true
""",
        encoding="utf-8",
    )
    hardware = load_hardware(
        path, drivers={"scanners": {"nidaq_galvo": NIGalvoScanner}}
    )
    try:
        scanner = hardware.require("slow_scanner", NIGalvoScanner)
        assert scanner.config == replace(config, inverted=True)
        assert scanner.angle is None
    finally:
        hardware.close()
    task_factory.assert_not_called()


@pytest.mark.parametrize(
    ("changes", "error_type", "message"),
    [
        ({"angle_control_channel": None}, TypeError, "angle_control_channel"),
        ({"angle_control_channel": ""}, ValueError, "angle_control_channel"),
        ({"angle_control_channel": "  "}, ValueError, "angle_control_channel"),
        ({"angle_range": object()}, TypeError, "angle_range"),
        (
            {"angle_range": units.AngleRange(float("-inf"), 1.0)},
            ValueError,
            "finite",
        ),
        (
            {"command_voltage_range": units.AngleRange("-1 deg", "1 deg")},
            TypeError,
            "command_voltage_range",
        ),
        (
            {"command_voltage_range": units.VoltageRange(-1.0, float("inf"))},
            ValueError,
            "endpoints",
        ),
        ({"inverted": 1}, TypeError, "inverted"),
        ({"inverted": "false"}, TypeError, "inverted"),
    ],
)
def test_config_rejects_invalid_fields(
    config: NIGalvoScannerConfig,
    changes: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Validate inherited optical limits alongside NI connection parameters.

    Replacing fields on a valid model must rerun parent validation and reject
    blank channels, invalid range types or endpoints, and non-boolean inversion.
    """
    with pytest.raises(error_type, match=message):
        replace(config, **changes)


def test_constructor_is_pure_and_checks_config_type(
    config: NIGalvoScannerConfig,
    task_factory: Mock,
) -> None:
    """Store typed configuration without creating a task or choosing an angle."""
    scanner = NIGalvoScanner(config)
    assert scanner.config is config
    assert scanner.angle is None
    scanner.close()

    with pytest.raises(TypeError, match="config must be an NIGalvoScannerConfig"):
        NIGalvoScanner(cast(NIGalvoScannerConfig, object()))

    task_factory.assert_not_called()


@pytest.mark.parametrize("inverted", [False, True])
@pytest.mark.parametrize(
    ("degrees", "normal_voltage"),
    [(-10, -1.0), (0, 1.0), (10, 3.0), (30, 7.0)],
)
def test_commands_use_full_vendor_ranges(
    config: NIGalvoScannerConfig,
    task_factory: Mock,
    inverted: bool,
    degrees: int,
    normal_voltage: float,
) -> None:
    """Map optical endpoints and interior angles, including inversion.

    The asymmetric ranges make zero optical angle map to a nonzero voltage.
    Inversion swaps voltage endpoints while preserving the full-scale map.
    Each scalar command must configure, write, and close one task.
    """
    scanner = NIGalvoScanner(replace(config, inverted=inverted))
    value = units.Angle(f"{degrees} deg")

    scanner.angle = value

    expected = 6.0 - normal_voltage if inverted else normal_voltage
    task = task_factory.return_value
    task_factory.assert_called_once_with()
    task.ao_channels.add_ao_voltage_chan.assert_called_once_with(
        physical_channel="Dev1/ao1", min_val=-1.0, max_val=7.0
    )
    task.write.assert_called_once_with(pytest.approx(expected), auto_start=True)
    task.close.assert_called_once_with()
    assert scanner.angle == value
    scanner.close()


@pytest.mark.parametrize("degrees", [-11, 31])
def test_invalid_angles_do_not_create_tasks(
    config: NIGalvoScannerConfig,
    task_factory: Mock,
    degrees: int,
) -> None:
    """Enforce inherited optical limits before entering the NI command path."""
    scanner = NIGalvoScanner(config)
    with pytest.raises(ValueError, match="angle_range"):
        scanner.angle = units.Angle(f"{degrees} deg")
    assert scanner.angle is None
    task_factory.assert_not_called()
    scanner.close()


@pytest.mark.parametrize("stage", ["configure", "write"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_operation_failure_preserves_history_and_primary_error(
    config: NIGalvoScannerConfig,
    task_factory: Mock,
    stage: str,
    cleanup_fails: bool,
) -> None:
    """Clean up failed scalar operations without replacing their original error.

    Establish a successful static command before injecting a channel-setup or
    write failure. The attempted command must not replace successful history,
    even when closing its task also fails.
    """
    scanner = NIGalvoScanner(config)
    previous = units.Angle("0 deg")
    scanner.angle = previous
    task_factory.reset_mock()
    task = task_factory.return_value
    failure = RuntimeError("NI operation failed.")
    operation = (
        task.ao_channels.add_ao_voltage_chan if stage == "configure" else task.write
    )
    operation.side_effect = failure
    if cleanup_fails:
        task.close.side_effect = RuntimeError("Cleanup failed.")

    with pytest.raises(RuntimeError, match="NI operation failed") as caught:
        scanner.angle = units.Angle("10 deg")

    assert caught.value is failure
    assert scanner.angle == previous
    task_factory.assert_called_once_with()
    task.close.assert_called_once_with()
    if cleanup_fails:
        assert any("Cleanup failed" in note for note in failure.__notes__)
    scanner.close()


def test_task_creation_failure_propagates(
    config: NIGalvoScannerConfig,
    task_factory: Mock,
) -> None:
    """Preserve a creation error without trying to close a nonexistent task."""
    scanner = NIGalvoScanner(config)
    failure = RuntimeError("Task creation failed.")
    task_factory.side_effect = failure

    with pytest.raises(RuntimeError, match="Task creation failed") as caught:
        scanner.angle = units.Angle("0 deg")

    assert caught.value is failure
    assert scanner.angle is None
    task_factory.return_value.close.assert_not_called()
    scanner.close()


def test_cleanup_failure_after_write_preserves_previous_history(
    config: NIGalvoScannerConfig,
    task_factory: Mock,
) -> None:
    """Treat task cleanup as part of completing a scalar command.

    A successful write followed by failed cleanup propagates the cleanup error.
    The cached value remains the previous fully successful static command.
    """
    scanner = NIGalvoScanner(config)
    previous = units.Angle("0 deg")
    scanner.angle = previous
    task_factory.reset_mock()
    task = task_factory.return_value
    failure = RuntimeError("Cleanup failed.")
    task.close.side_effect = failure

    with pytest.raises(RuntimeError, match="Cleanup failed") as caught:
        scanner.angle = units.Angle("10 deg")

    assert caught.value is failure
    assert scanner.angle == previous
    task.write.assert_called_once_with(pytest.approx(3.0), auto_start=True)
    task.close.assert_called_once_with()
    scanner.close()


@pytest.mark.parametrize("samples_written", [0, 2])
def test_incomplete_write_does_not_update_history(
    config: NIGalvoScannerConfig,
    task_factory: Mock,
    samples_written: int,
) -> None:
    """Require the NI API to acknowledge exactly one scalar sample."""
    scanner = NIGalvoScanner(config)
    task = task_factory.return_value
    task.write.return_value = samples_written

    with pytest.raises(RuntimeError, match="exactly one sample"):
        scanner.angle = units.Angle("0 deg")

    assert scanner.angle is None
    task.close.assert_called_once_with()
    scanner.close()


def test_close_does_not_park_and_disables_commands(
    config: NIGalvoScannerConfig,
    task_factory: Mock,
) -> None:
    """Keep closure idempotent without creating or writing an idle task."""
    scanner = NIGalvoScanner(config)
    scanner.angle = units.Angle("0 deg")
    scanner.close()
    scanner.close()

    with pytest.raises(RuntimeError, match="closed"):
        scanner.angle = units.Angle("1 deg")

    task_factory.assert_called_once_with()
    task_factory.return_value.write.assert_called_once_with(
        pytest.approx(1.0), auto_start=True
    )
    task_factory.return_value.close.assert_called_once_with()


def test_snapshot_includes_ni_mapping(
    config: NIGalvoScannerConfig,
) -> None:
    """Record connection, inversion, and installed driver metadata without output."""
    scanner = NIGalvoScanner(config)
    try:
        snapshot = scanner.snapshot()
        settings = snapshot["settings"]
        assert settings["angle_control_channel"] == "Dev1/ao1"
        assert settings["command_voltage_range"] == {
            "min": {"value": -1.0, "unit": "V"},
            "max": {"value": 7.0, "unit": "V"},
        }
        assert settings["inverted"] is False
        assert settings["last_static_angle"] is None
        assert snapshot["driver"]["distribution"] == "dirigo-nidaq"
        assert snapshot["driver"]["version"] == importlib_metadata.version(
            "dirigo-nidaq"
        )
    finally:
        scanner.close()


def test_identity_allows_missing_distribution_metadata(
    config: NIGalvoScannerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the driver distribution identifiable when its version is unavailable."""

    def missing_version(name: str) -> str:
        """Simulate an unavailable installation metadata record."""
        raise importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib_metadata, "version", missing_version)
    scanner = NIGalvoScanner(config)
    try:
        assert scanner.driver_identity.distribution == "dirigo-nidaq"
        assert scanner.driver_identity.version is None
    finally:
        scanner.close()


def test_entry_point_loads_without_creating_a_task(task_factory: Mock) -> None:
    """Resolve the categorized galvo registration without constructing hardware."""
    entries = importlib_metadata.entry_points(
        group="dirigo.devices.scanners", name="nidaq_galvo"
    )
    assert len(entries) == 1
    assert next(iter(entries)).load() is NIGalvoScanner
    assert NIGalvoScanner.config_type is NIGalvoScannerConfig
    task_factory.assert_not_called()
