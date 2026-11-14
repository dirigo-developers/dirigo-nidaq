"""Test record-trigger placement without NI tasks or an Alazar SDK."""

from dataclasses import replace

import pytest
from dirigo import units
from dirigo.hardware.capture import CaptureSettings
from dirigo.hardware.timing import DigitalRecordTrigger
from dirigo.hardware.waveform import AngularWaveform

from dirigo_nidaq.timing import RecordTriggerPlan


@pytest.fixture
def plan() -> RecordTriggerPlan:
    """Plan six records on a slower, independently sampled AO waveform."""
    return RecordTriggerPlan(
        capture_settings=CaptureSettings(
            sample_rate=units.SampleRate("100 MS/s"),
            record_length=4096,
            records_per_batch=2,
            batch_count=3,
            record_trigger=DigitalRecordTrigger(source="external"),
        ),
        waveform=AngularWaveform(
            [0.0] * 32,
            sample_rate=units.SampleRate("20 kS/s"),
        ),
        interval_samples=5,
        first_trigger_sample=3,
    )


def test_edges_follow_ao_grid_across_batch_boundaries(
    plan: RecordTriggerPlan,
) -> None:
    """Derive all six edges from the full capture, without resetting per batch.

    Two records per batch and three batches require six triggers. Their AO
    positions form one continuous schedule despite the digitizer using a
    much faster sample clock and thousands of samples per record.
    """
    assert plan.record_count == 6
    assert plan.sample_indices == range(3, 29, 5)
    assert tuple(plan.sample_indices) == (3, 8, 13, 18, 23, 28)


def test_digitizer_sample_geometry_does_not_retime_triggers(
    plan: RecordTriggerPlan,
) -> None:
    """Keep trigger placement independent of digitizer rate and record length."""
    changed = replace(
        plan,
        capture_settings=replace(
            plan.capture_settings,
            sample_rate=units.SampleRate("200 MS/s"),
            record_length=8192,
        ),
    )

    assert changed.sample_indices == plan.sample_indices
    assert changed.record_count == plan.record_count


@pytest.mark.parametrize(
    ("first_trigger_sample", "expected"),
    [
        (0, (0, 5, 10, 15, 20, 25)),
        (6, (6, 11, 16, 21, 26, 31)),
    ],
)
def test_first_and_last_waveform_samples_are_valid_edges(
    plan: RecordTriggerPlan,
    first_trigger_sample: int,
    expected: tuple[int, ...],
) -> None:
    """Accept sample zero and the last existing sample as trigger positions."""
    changed = replace(plan, first_trigger_sample=first_trigger_sample)

    assert tuple(changed.sample_indices) == expected


@pytest.mark.parametrize("first_trigger_sample", [7, 32])
def test_rejects_edges_at_or_beyond_waveform_end(
    plan: RecordTriggerPlan,
    first_trigger_sample: int,
) -> None:
    """Reject a final edge at index 32 and a schedule starting beyond the data."""
    with pytest.raises(ValueError, match="within the waveform"):
        replace(plan, first_trigger_sample=first_trigger_sample)


def test_single_record_requires_only_its_one_edge_to_fit(
    plan: RecordTriggerPlan,
) -> None:
    """Avoid requiring an unused inter-record interval after the final edge."""
    changed = replace(
        plan,
        capture_settings=replace(
            plan.capture_settings,
            records_per_batch=1,
            batch_count=1,
        ),
        interval_samples=1000,
        first_trigger_sample=31,
    )

    assert changed.record_count == 1
    assert tuple(changed.sample_indices) == (31,)


@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        ("interval_samples", True, TypeError, "interval_samples.*integer"),
        ("interval_samples", 5.0, TypeError, "interval_samples.*integer"),
        ("interval_samples", 0, ValueError, "interval_samples.*positive"),
        ("interval_samples", -1, ValueError, "interval_samples.*positive"),
        ("first_trigger_sample", False, TypeError, "first_trigger_sample.*integer"),
        ("first_trigger_sample", 3.0, TypeError, "first_trigger_sample.*integer"),
        ("first_trigger_sample", -1, ValueError, "first_trigger_sample.*nonnegative"),
        ("capture_settings", object(), TypeError, "capture_settings.*CaptureSettings"),
        ("waveform", object(), TypeError, "waveform.*AngularWaveform"),
    ],
)
def test_rejects_invalid_plan_fields(
    plan: RecordTriggerPlan,
    field: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject wrong model types, noninteger timing, and invalid integer bounds."""
    with pytest.raises(error_type, match=message):
        replace(plan, **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_count", None, "finite capture"),
        ("record_trigger", None, "record_trigger"),
    ],
)
def test_requires_finite_record_triggered_capture(
    plan: RecordTriggerPlan,
    field: str,
    value: object,
    message: str,
) -> None:
    """Reject unlimited delivery and captures that do not consume record triggers."""
    settings = replace(plan.capture_settings, **{field: value})

    with pytest.raises(ValueError, match=message):
        replace(plan, capture_settings=settings)
