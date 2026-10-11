import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from lima_pm25.domain import AmbientBaseline, GridDefinition
from lima_pm25.estimation import (
    EvaluationStatus,
    R21AcceptanceProfile,
    R22AcceptanceProfile,
    ReconstructionConfig,
    evaluate_retained_synthetic_run,
    verify_retained_evaluation_bundle,
    write_retained_evaluation_bundle,
)
from lima_pm25.simulation import (
    ConcentrationFrame,
    MeasurementBatch,
    MeasurementQuality,
    MeasurementRecord,
    PM25State,
    SensorType,
    SimulationRunResult,
)


def test_retained_evaluation_keeps_truth_out_of_reconstruction_inputs(
    tmp_path: Path,
) -> None:
    grid = _grid()
    start = datetime(2025, 1, 1, tzinfo=UTC)
    first_truth = np.array([[10.0, 20.0], [30.0, 40.0]])
    second_truth = np.array([[12.0, 22.0], [32.0, 42.0]])
    fixed = _record("FIXED", SensorType.FIXED, start, 10, 50, 50, 0)
    mobile = _record("MOBILE", SensorType.MOBILE, start, 40, 150, 150, 0)
    later_fixed = _record(
        "FIXED",
        SensorType.FIXED,
        start + timedelta(hours=1),
        12,
        50,
        50,
        3_600,
    )
    run = SimulationRunResult(
        run_id="retained-test",
        manifest_json="{}",
        frames=(
            ConcentrationFrame(0, start, first_truth),
            ConcentrationFrame(3_600, start + timedelta(hours=1), second_truth),
        ),
        measurement_attempts=(fixed, mobile, later_fixed),
        measurement_batches=(
            MeasurementBatch(0, start, (fixed, mobile)),
            MeasurementBatch(
                3_600,
                start + timedelta(hours=1),
                (later_fixed,),
            ),
        ),
        diagnostics=(),
        final_state=PM25State(
            grid,
            second_truth,
            3_600,
            AmbientBaseline("ambient", 10, "retained fixture"),
        ),
    )

    result = evaluate_retained_synthetic_run(
        run=run,
        reconstruction_config=ReconstructionConfig(
            prior_mean_ug_m3=20,
            prior_stddev_ug_m3=20,
            x_length_scale_m=150,
            y_length_scale_m=150,
            temporal_length_scale_s=3_600,
        ),
        history_duration=timedelta(hours=2),
        scale_ug_m3=30,
        r21_profile=R21AcceptanceProfile(minimum_sample_count=8),
        r22_profile=R22AcceptanceProfile(minimum_sample_count=8),
    )

    assert result.run_id == "retained-test"
    assert result.fixed_mobile_r21.status in {
        EvaluationStatus.PASS,
        EvaluationStatus.FAIL,
    }
    assert result.fixed_mobile_r22.status in {
        EvaluationStatus.PASS,
        EvaluationStatus.FAIL,
    }
    assert result.fixed_mobile_r21.sample_count == 8
    assert result.fixed_mobile_r22.sample_count == 8
    assert result.fixed_only_posteriors[0].observations.used_measurement_keys == (
        ("FIXED", start),
    )
    assert result.fixed_mobile_posteriors[0].observations.used_measurement_keys == (
        ("FIXED", start),
        ("MOBILE", start),
    )
    assert len(result.mobile_uncertainty_changes) == 1
    assert result.mobile_uncertainty_changes[0].status is EvaluationStatus.PASS

    destination = tmp_path / "retained-bundle"
    written = write_retained_evaluation_bundle(
        evaluation=result,
        run=run,
        reconstruction_config=ReconstructionConfig(
            prior_mean_ug_m3=20,
            prior_stddev_ug_m3=20,
            x_length_scale_m=150,
            y_length_scale_m=150,
            temporal_length_scale_s=3_600,
        ),
        history_duration=timedelta(hours=2),
        scale_ug_m3=30,
        destination=destination,
        representative_frame_indices=(0,),
    )
    assert written == destination
    assert (destination / "maps.npz").is_file()
    assert (destination / "figures" / "frame-0000.png").is_file()
    assert len(tuple((destination / "figures").glob("*.png"))) == 3
    metrics = json.loads((destination / "metrics.json").read_text())
    assert metrics["run_id"] == "retained-test"
    assert metrics["map_validation"]["cell_time_count"] == 8
    assert metrics["temporal_aging"]["history_cutoff_applied"] is False
    curve = metrics["temporal_aging"]["standard_deviations_ug_m3"]
    assert np.all(np.diff(curve) >= -1e-9)
    assert curve[-1] == pytest.approx(20, rel=0.001)
    assert verify_retained_evaluation_bundle(destination) == ()
    with pytest.raises(FileExistsError, match="already exists"):
        write_retained_evaluation_bundle(
            evaluation=result,
            run=run,
            reconstruction_config=ReconstructionConfig(
                prior_mean_ug_m3=20,
                prior_stddev_ug_m3=20,
                x_length_scale_m=150,
                y_length_scale_m=150,
                temporal_length_scale_s=3_600,
            ),
            history_duration=timedelta(hours=2),
            scale_ug_m3=30,
            destination=destination,
        )


def _record(
    sensor_id: str,
    sensor_type: SensorType,
    timestamp: datetime,
    value: float,
    x_m: float,
    y_m: float,
    simulation_time_s: float,
) -> MeasurementRecord:
    return MeasurementRecord(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        timestamp_utc=timestamp,
        simulation_time_s=simulation_time_s,
        x_m=x_m,
        y_m=y_m,
        crs="EPSG:32718",
        value_ug_m3=value,
        quality_status=MeasurementQuality.ACCEPTED,
        noise_stddev_ug_m3=0,
        position_source_id=sensor_id,
        provenance="public retained fixture",
    )


def _grid() -> GridDefinition:
    return GridDefinition(
        crs="EPSG:32718",
        domain_bounds_m=np.array([0, 0, 200, 200]),
        grid_bounds_m=np.array([0, 0, 200, 200]),
        cell_size_m=100,
        x_centers_m=np.array([50, 150]),
        y_centers_m=np.array([50, 150]),
        active_mask=np.ones((2, 2), dtype=bool),
    )
