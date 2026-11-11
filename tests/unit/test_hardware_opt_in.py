"""Test the physical-test gate using isolated pytest subprocesses."""

from pathlib import Path

import pytest
from _pytest.pytester import Pytester


@pytest.fixture
def suite(pytester: Pytester) -> Pytester:
    """Create an isolated suite with a copy of the actual gate."""
    conftest = Path(__file__).parents[2] / "conftest.py"
    pytester.makeconftest(conftest.read_text(encoding="utf-8"))
    pytester.makeini(
        """
        [pytest]
        addopts = --strict-markers
        markers =
            hardware: accesses physical equipment
        """
    )
    pytester.makepyfile(
        test_unit="""
        def test_unit():
            assert 1 + 1 == 2
        """
    )
    (pytester.path / "tests").mkdir()
    return pytester


def test_default_run_does_not_import_physical_tests(suite: Pytester) -> None:
    """Ignore the physical directory before test-module imports."""
    directory = suite.path / "tests" / "hardware"
    directory.mkdir()
    (directory / "test_device.py").write_text(
        'raise RuntimeError("Physical test module must not be imported.")\n',
        encoding="utf-8",
    )

    result = suite.runpytest_subprocess("-q")

    result.assert_outcomes(passed=1)


def test_hardware_option_enables_directory_tests(suite: Pytester) -> None:
    """Automatically mark physical-directory tests for explicit selection."""
    directory = suite.path / "tests" / "hardware"
    directory.mkdir()
    (directory / "test_device.py").write_text(
        "def test_device():\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )

    result = suite.runpytest_subprocess("--hardware", "-m", "hardware", "-q")

    result.assert_outcomes(passed=1, deselected=1)


def test_marked_tests_are_skipped_until_enabled_and_then_report_errors(
    suite: Pytester,
) -> None:
    """Skip hardware fixtures by default; preserve errors after explicit opt-in."""
    suite.makepyfile(
        test_marked="""
        import pytest

        @pytest.fixture
        def connection():
            raise RuntimeError("Configured hardware unavailable.")

        @pytest.mark.hardware
        def test_device(connection):
            assert connection is not None
        """
    )

    ordinary = suite.runpytest_subprocess("-q")
    ordinary.assert_outcomes(passed=1, skipped=1)

    enabled = suite.runpytest_subprocess("--hardware", "-m", "hardware", "-q")
    enabled.assert_outcomes(errors=1, deselected=1)
    enabled.stdout.fnmatch_lines(["*RuntimeError: Configured hardware unavailable.*"])
