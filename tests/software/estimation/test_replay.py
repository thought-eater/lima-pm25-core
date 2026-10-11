from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from lima_pm25.domain import GridDefinition
from lima_pm25.estimation import (
    ReconstructionConfig,
    compose_historical_replay_batches,
    reconstruct_measurement_series,
)
from lima_pm25.simulation import (
    MeasurementQuality,
    MeasurementRecord,
    SensorType,
)


def test_replay_batches_merge_every_available_fixed_and_accepted_mobile_record() -> (
    None
):
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    fixed = tuple(
        _record(station, SensorType.FIXED, timestamp, value=10 + index)
        for index, station in enumerate(("CAMPO", "SANTA", "SJL"))
    )
    mobile = (
        _record("MOBILE", SensorType.MOBILE, timestamp, value=20),
        _record(
            "DROPPED",
            SensorType.MOBILE,
            timestamp,
            value=None,
            quality=MeasurementQuality.DROPPED,
        ),
    )

    batches = compose_historical_replay_batches(
        historical_fixed_records=fixed,
        simulated_mobile_attempts=mobile,
    )

    assert len(batches) == 1
    assert [record.sensor_id for record in batches[0].records] == [
        "CAMPO",
        "MOBILE",
        "SANTA",
        "SJL",
    ]
    assert all(record.sensor_id != "CERES" for record in batches[0].records)
    assert all(record.sensor_id != "DROPPED" for record in batches[0].records)


def test_replay_refuses_a_simulated_fixed_record() -> None:
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)

    with pytest.raises(ValueError, match="only mobile"):
        compose_historical_replay_batches(
            historical_fixed_records=(),
            simulated_mobile_attempts=(
                _record("SIMULATED_FIXED", SensorType.FIXED, timestamp, value=10),
            ),
        )


def test_reconstruction_series_uses_only_history_available_at_each_timestamp() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    grid = _grid()
    records = (
        _record("A", SensorType.FIXED, start, value=10, x_m=50),
        _record(
            "B",
            SensorType.FIXED,
            start + timedelta(hours=1),
            value=30,
            x_m=150,
            simulation_time_s=3_600,
        ),
    )
    batches = compose_historical_replay_batches(
        historical_fixed_records=records,
        simulated_mobile_attempts=(),
    )

    posteriors = reconstruct_measurement_series(
        grid=grid,
        config=ReconstructionConfig(
            prior_mean_ug_m3=20,
            prior_stddev_ug_m3=10,
            x_length_scale_m=100,
            y_length_scale_m=100,
            temporal_length_scale_s=3_600,
        ),
        batches=batches,
        reconstruction_timestamps_utc=(start, start + timedelta(hours=1)),
        history_duration=timedelta(hours=2),
    )

    assert posteriors[0].observations.used_measurement_keys == (("A", start),)
    assert posteriors[1].observations.used_measurement_keys == (
        ("A", start),
        ("B", start + timedelta(hours=1)),
    )


def _record(
    sensor_id: str,
    sensor_type: SensorType,
    timestamp: datetime,
    *,
    value: float | None,
    quality: MeasurementQuality | None = None,
    x_m: float = 50,
    simulation_time_s: float = 0,
) -> MeasurementRecord:
    return MeasurementRecord(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        timestamp_utc=timestamp,
        simulation_time_s=simulation_time_s,
        x_m=x_m,
        y_m=50,
        crs="EPSG:32718",
        value_ug_m3=value,
        quality_status=(
            quality if quality is not None else MeasurementQuality.ACCEPTED
        ),
        noise_stddev_ug_m3=0,
        position_source_id=sensor_id,
        provenance=(
            "observed fixed fixture"
            if sensor_type is SensorType.FIXED
            else "simulated mobile fixture"
        ),
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
