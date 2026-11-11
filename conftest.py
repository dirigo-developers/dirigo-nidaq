"""Keep physical hardware tests opt-in for the NI plugin."""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_HARDWARE_DIRECTORY = (Path(__file__).parent / "tests" / "hardware").resolve()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add explicit opt-in for tests that access physical equipment."""
    parser.addoption(
        "--hardware",
        action="store_true",
        default=False,
        help="Enable tests that access physical hardware.",
    )


def _is_hardware_path(path: Path) -> bool:
    """Recognize files and directories in this suite's physical-test directory."""
    return path.resolve().is_relative_to(_HARDWARE_DIRECTORY)


def pytest_ignore_collect(
    collection_path: Path,
    config: pytest.Config,
) -> bool | None:
    """Avoid importing physical test modules during ordinary test runs."""
    if not config.getoption("hardware") and _is_hardware_path(collection_path):
        return True
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Mark physical tests before marker filtering and gate their execution."""
    enabled = config.getoption("hardware")
    disabled = pytest.mark.skip(
        reason="Pass --hardware to run physical hardware tests."
    )

    for item in items:
        marked = item.get_closest_marker("hardware") is not None
        if not marked and not _is_hardware_path(item.path):
            continue

        if not marked:
            item.add_marker(pytest.mark.hardware)
        if not enabled:
            item.add_marker(disabled)
