from dataclasses import fields
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import numpy as np
import pytest

from lima_pm25.domain import GridDefinition
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    MeasurementDisposition,
    ReconstructionConfig,
    SequentialMapSnapshot,
    SequentialMapUpdater,
    SequentialUpdateConfig,
    SequentialUpdateError,
    load_change10_evaluation_profile,
    process_measurement_batches,
)
from lima_pm25.simulation import (
    MeasurementBatch,
    MeasurementQuality,
    MeasurementRecord,
    SensorType,
)


START = datetime(2026, 9, 28, tzinfo=UTC)


def test_change10_profile_supplies_the_evaluated_r23_policy(tmp_path) -> None:
    payload = json.loads(Path("config/estimation_evaluation.json").read_text())
    artifact = tmp_path / "calibration.json"
    artifact.write_text("{}")
    payload["calibration_artifact"] = str(artifact)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(payload))
    profile = load_change10_evaluation_profile(profile_path)

    config = SequentialUpdateConfig.from_change10_profile(profile)

    assert config.reconstruction_config == profile.reconstruction_config
    assert config.history_duration == profile.history_duration
    assert not config.reconstruction_config.include_clipped_observations
    assert config.profile_status == "provisional_pending_adviser"
    assert config.calibration_artifact_sha256
    assert config.calibration_artifact_source == str(artifact)


def test_first_mobile_batch_updates_prior_and_reduces_uncertainty() -> None:
    updater = _updater()

    result = updater.update(_batch(_record("MOBILE", START, SensorType.MOBILE)))

    assert result.applied
    assert result.snapshot is updater.current_snapshot
    assert result.snapshot is not None
    assert result.snapshot.sequence_number == 1
    diagnostic = result.snapshot.uncertainty_reduction
    assert diagnostic.has_mobile_measurement
    assert diagnostic.passes_nonincrease
    assert diagnostic.local[0].reduction_ug_m3_squared > 0
    assert diagnostic.integrated_reduction_ug2_m4 > 0
    assert diagnostic.local[0].variance_before_ug_m3_squared == pytest.approx(100)


def test_successive_batches_publish_increasing_coupled_snapshots() -> None:
    updater = _updater()

    first = updater.update(_batch(_record("MOBILE", START, SensorType.MOBILE)))
    second_time = START + timedelta(minutes=5)
    second = updater.update(
        _batch(_record("MOBILE", second_time, SensorType.MOBILE, x_m=150))
    )

    assert first.snapshot is not None and second.snapshot is not None
    assert (first.snapshot.sequence_number, second.snapshot.sequence_number) == (1, 2)
    assert first.snapshot.measurement_cutoff_utc < second.snapshot.measurement_cutoff_utc
    assert second.snapshot.reconstruction_timestamp_utc == second_time
    assert second.snapshot.mean_ug_m3.shape == second.snapshot.active_mask.shape
    assert (
        second.snapshot.standard_deviation_ug_m3.shape
        == second.snapshot.active_mask.shape
    )
    assert second.snapshot.observations.used_record_count == 2


def test_same_time_fixed_and_mobile_records_publish_one_snapshot() -> None:
    updater = _updater()
    batch = _batch(
        _record("FIXED", START, SensorType.FIXED, value=12),
        _record("MOBILE", START, SensorType.MOBILE, x_m=150, value=28),
    )

    result = updater.update(batch)

    assert result.snapshot is not None
    assert result.snapshot.sequence_number == 1
    assert result.snapshot.incorporated_measurement_keys == (
        ("FIXED", START),
        ("MOBILE", START),
    )
    assert len(updater.update_log) == 1


def test_same_time_multiple_mobile_records_publish_one_snapshot() -> None:
    updater = _updater()
    batch = _batch(
        _record("MOBILE-A", START, SensorType.MOBILE, value=12),
        _record("MOBILE-B", START, SensorType.MOBILE, x_m=150, value=28),
    )

    result = updater.update(batch)

    assert result.snapshot is not None
    assert result.snapshot.sequence_number == 1
    assert len(result.snapshot.uncertainty_reduction.local) == 2
    assert result.log_entry.eligible_mobile_measurement_keys == (
        ("MOBILE-A", START),
        ("MOBILE-B", START),
    )


def test_history_window_includes_boundary_and_expires_older_records() -> None:
    updater = _updater(history_duration=timedelta(hours=6))
    boundary = START + timedelta(hours=6)
    after_boundary = boundary + timedelta(seconds=1)

    updater.update(_batch(_record("A", START, SensorType.FIXED)))
    at_boundary = updater.update(
        _batch(_record("B", boundary, SensorType.FIXED, x_m=150))
    )
    expired = updater.update(
        _batch(_record("C", after_boundary, SensorType.MOBILE, value=25))
    )

    assert at_boundary.snapshot is not None and expired.snapshot is not None
    assert at_boundary.snapshot.observations.used_measurement_keys == (
        ("A", START),
        ("B", boundary),
    )
    assert expired.snapshot.observations.used_measurement_keys == (
        ("B", boundary),
        ("C", after_boundary),
    )
    assert ("A", START) in {
        processed.key for processed in updater.processed_measurements
    }


def test_clipped_only_batch_is_policy_noop() -> None:
    updater = _updater()
    clipped = _record(
        "MOBILE",
        START,
        SensorType.MOBILE,
        value=0,
        quality=MeasurementQuality.CLIPPED,
    )

    result = updater.update(_batch(clipped))

    assert not result.applied
    assert result.snapshot is None
    assert updater.current_snapshot is None
    assert updater.latest_processed_timestamp_utc == START
    assert result.log_entry.policy_excluded_measurement_keys == (("MOBILE", START),)
    assert updater.processed_measurements[0].disposition is (
        MeasurementDisposition.POLICY_EXCLUDED
    )


def test_mixed_eligible_and_clipped_batch_accounts_for_every_record() -> None:
    updater = _updater()
    batch = _batch(
        _record("ELIGIBLE", START, SensorType.MOBILE),
        _record(
            "EXCLUDED",
            START,
            SensorType.FIXED,
            value=0,
            quality=MeasurementQuality.CLIPPED,
        ),
    )

    result = updater.update(batch)

    assert result.snapshot is not None
    assert result.log_entry.incorporated_measurement_keys == (("ELIGIBLE", START),)
    assert result.log_entry.policy_excluded_measurement_keys == (("EXCLUDED", START),)
    assert {item.disposition for item in updater.processed_measurements} == {
        MeasurementDisposition.INCORPORATED,
        MeasurementDisposition.POLICY_EXCLUDED,
    }


def test_duplicate_measurement_key_is_rejected_without_mutation() -> None:
    updater = _updater()
    duplicate = _record("MOBILE", START, SensorType.MOBILE)
    batch = MeasurementBatch(0, START, (duplicate, duplicate))

    with pytest.raises(SequentialUpdateError, match="duplicate"):
        updater.update(batch)

    assert updater.current_snapshot is None
    assert updater.latest_processed_timestamp_utc is None
    assert updater.processed_measurements == ()
    assert updater.update_log == ()


def test_older_or_equal_timestamp_is_rejected_without_mutation() -> None:
    updater = _updater()
    current = START + timedelta(minutes=5)
    incorporated = _record("MOBILE", current, SensorType.MOBILE)
    updater.update(_batch(incorporated))
    state = _state_signature(updater)

    with pytest.raises(SequentialUpdateError, match="strictly later"):
        updater.update(_batch(incorporated))
    assert _state_signature(updater) == state

    for timestamp in (START, current):
        with pytest.raises(SequentialUpdateError, match="strictly later"):
            updater.update(
                _batch(_record("LATE", timestamp, SensorType.MOBILE))
            )
        assert _state_signature(updater) == state


def test_invalid_record_rolls_back_all_state() -> None:
    updater = _updater()
    updater.update(_batch(_record("MOBILE", START, SensorType.MOBILE)))
    previous_snapshot = updater.current_snapshot
    state = _state_signature(updater)
    invalid_time = START + timedelta(minutes=5)

    with pytest.raises(ValueError, match="different CRS"):
        updater.update(
            _batch(
                _record(
                    "INVALID",
                    invalid_time,
                    SensorType.MOBILE,
                    crs="EPSG:3857",
                )
            )
        )

    assert _state_signature(updater) == state
    assert updater.current_snapshot is previous_snapshot
    retry = updater.update(
        _batch(_record("INVALID", invalid_time, SensorType.MOBILE, x_m=150))
    )
    assert retry.snapshot is not None
    assert retry.snapshot.sequence_number == 2


def test_reconstructor_failure_rolls_back_all_state() -> None:
    config = _config()
    failing = _FailingReconstructor(config.reconstruction_config, fail_on_call=4)
    updater = SequentialMapUpdater(_grid(), config, reconstructor=failing)
    updater.update(_batch(_record("FIRST", START, SensorType.MOBILE)))
    previous_snapshot = updater.current_snapshot
    state = _state_signature(updater)
    second_batch = _batch(
        _record(
            "SECOND",
            START + timedelta(minutes=5),
            SensorType.MOBILE,
            x_m=150,
        )
    )

    with pytest.raises(RuntimeError, match="synthetic estimator failure"):
        updater.update(second_batch)

    assert _state_signature(updater) == state
    assert updater.current_snapshot is previous_snapshot
    retry = updater.update(second_batch)
    assert retry.snapshot is not None
    assert retry.snapshot.sequence_number == 2


def test_snapshot_is_immutable_agent_consumable_map_state_without_truth() -> None:
    updater = _updater()
    result = updater.update(_batch(_record("MOBILE", START, SensorType.MOBILE)))
    assert result.snapshot is not None
    snapshot = result.snapshot

    assert not snapshot.mean_ug_m3.flags.writeable
    assert not snapshot.standard_deviation_ug_m3.flags.writeable
    assert not snapshot.active_mask.flags.writeable
    snapshot_fields = {item.name for item in fields(SequentialMapSnapshot)}
    assert "hidden_truth" not in snapshot_fields
    assert "sampled_truth_ug_m3" not in snapshot_fields
    assert "actions" not in snapshot_fields
    assert "rewards" not in snapshot_fields


def test_sequential_replay_is_deterministic() -> None:
    batches = (
        _batch(_record("FIXED", START, SensorType.FIXED, value=12)),
        _batch(
            _record(
                "MOBILE",
                START + timedelta(minutes=5),
                SensorType.MOBILE,
                x_m=150,
                value=28,
            )
        ),
    )

    first = process_measurement_batches(_updater(), batches)
    second = process_measurement_batches(_updater(), batches)

    assert first.log == second.log
    assert len(first.snapshots) == len(second.snapshots)
    for left, right in zip(first.snapshots, second.snapshots, strict=True):
        assert np.array_equal(left.mean_ug_m3, right.mean_ug_m3)
        assert np.array_equal(
            left.standard_deviation_ug_m3,
            right.standard_deviation_ug_m3,
        )
        assert left.uncertainty_reduction == right.uncertainty_reduction


class _FailingReconstructor:
    def __init__(self, config: ReconstructionConfig, *, fail_on_call: int) -> None:
        self._delegate = GaussianProcessFieldReconstructor(config)
        self._fail_on_call = fail_on_call
        self._calls = 0

    def reconstruct(self, request):
        self._calls += 1
        if self._calls == self._fail_on_call:
            raise RuntimeError("synthetic estimator failure")
        return self._delegate.reconstruct(request)


def _updater(*, history_duration: timedelta = timedelta(hours=6)):
    return SequentialMapUpdater(
        _grid(),
        _config(history_duration=history_duration),
    )


def _config(*, history_duration: timedelta = timedelta(hours=6)):
    return SequentialUpdateConfig(
        reconstruction_config=ReconstructionConfig(
            prior_mean_ug_m3=20,
            prior_stddev_ug_m3=10,
            x_length_scale_m=250,
            y_length_scale_m=250,
            temporal_length_scale_s=3_600,
        ),
        history_duration=history_duration,
        variance_numerical_tolerance_ug_m3_squared=1e-9,
        profile_sha256="fixture-profile-sha256",
        profile_status="synthetic_test",
        profile_source="tests/tesis/r2_3/test_sequential.py",
        calibration_artifact_sha256="fixture-calibration-sha256",
        calibration_artifact_source="synthetic test calibration",
        prediction_chunk_size=4,
    )


def _grid() -> GridDefinition:
    return GridDefinition(
        crs="EPSG:32718",
        domain_bounds_m=np.array([0, 0, 200, 100]),
        grid_bounds_m=np.array([0, 0, 200, 100]),
        cell_size_m=100,
        x_centers_m=np.array([50, 150]),
        y_centers_m=np.array([50]),
        active_mask=np.ones((1, 2), dtype=bool),
    )


def _record(
    sensor_id: str,
    timestamp: datetime,
    sensor_type: SensorType,
    *,
    x_m: float = 50,
    value: float = 30,
    quality: MeasurementQuality = MeasurementQuality.ACCEPTED,
    crs: str = "EPSG:32718",
) -> MeasurementRecord:
    return MeasurementRecord(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        timestamp_utc=timestamp,
        simulation_time_s=(timestamp - START).total_seconds(),
        x_m=x_m,
        y_m=50,
        crs=crs,
        value_ug_m3=value,
        quality_status=quality,
        noise_stddev_ug_m3=0,
        position_source_id=sensor_id,
        provenance="synthetic sequential-update test",
    )


def _batch(*records: MeasurementRecord) -> MeasurementBatch:
    ordered = tuple(sorted(records, key=lambda record: record.sensor_id))
    return MeasurementBatch(
        simulation_time_s=ordered[0].simulation_time_s,
        timestamp_utc=ordered[0].timestamp_utc,
        records=ordered,
    )


def _state_signature(updater: SequentialMapUpdater):
    snapshot = updater.current_snapshot
    return (
        updater.latest_processed_timestamp_utc,
        snapshot.sequence_number if snapshot is not None else None,
        snapshot.measurement_cutoff_utc if snapshot is not None else None,
        updater.processed_measurements,
        updater.update_log,
    )
