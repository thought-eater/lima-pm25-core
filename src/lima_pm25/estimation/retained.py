from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from lima_pm25.simulation import MeasurementBatch, SensorType, SimulationRunResult

from .evaluation import (
    R21AcceptanceProfile,
    R21Evaluation,
    R22AcceptanceProfile,
    R22Evaluation,
    UncertaintyChangeEvaluation,
    evaluate_r21,
    evaluate_r22,
    evaluate_uncertainty_change,
)
from .reconstruction import PosteriorField, ReconstructionConfig
from .replay import reconstruct_measurement_series


@dataclass(frozen=True, slots=True)
class RetainedSyntheticEvaluation:
    run_id: str
    fixed_only_posteriors: tuple[PosteriorField, ...]
    fixed_mobile_posteriors: tuple[PosteriorField, ...]
    fixed_only_r21: R21Evaluation
    fixed_mobile_r21: R21Evaluation
    fixed_mobile_r22: R22Evaluation
    mobile_uncertainty_changes: tuple[UncertaintyChangeEvaluation, ...]


def evaluate_retained_synthetic_run(
    *,
    run: SimulationRunResult,
    reconstruction_config: ReconstructionConfig,
    history_duration: timedelta,
    scale_ug_m3: float,
    r21_profile: R21AcceptanceProfile,
    r22_profile: R22AcceptanceProfile,
    uncertainty_numerical_tolerance: float = 1e-9,
    prediction_chunk_size: int = 2_048,
) -> RetainedSyntheticEvaluation:
    grid = run.final_state.grid
    timestamps = tuple(frame.timestamp_utc for frame in run.frames)
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("Retained concentration frames must have unique timestamps")
    fixed_batches = _filtered_batches(run.measurement_batches, SensorType.FIXED)
    all_batches = tuple(run.measurement_batches)
    fixed_only = reconstruct_measurement_series(
        grid=grid,
        config=reconstruction_config,
        batches=fixed_batches,
        reconstruction_timestamps_utc=timestamps,
        history_duration=history_duration,
        prediction_chunk_size=prediction_chunk_size,
    )
    fixed_mobile = reconstruct_measurement_series(
        grid=grid,
        config=reconstruction_config,
        batches=all_batches,
        reconstruction_timestamps_utc=timestamps,
        history_duration=history_duration,
        prediction_chunk_size=prediction_chunk_size,
    )
    truth_maps = tuple(frame.concentration_ug_m3 for frame in run.frames)
    prior_maps = tuple(
        np.where(
            grid.active_mask,
            reconstruction_config.prior_mean_ug_m3,
            np.nan,
        )
        for _ in run.frames
    )
    fixed_means = tuple(posterior.mean_ug_m3 for posterior in fixed_only)
    mixed_means = tuple(posterior.mean_ug_m3 for posterior in fixed_mobile)
    mixed_stddev = tuple(
        posterior.standard_deviation_ug_m3 for posterior in fixed_mobile
    )
    fixed_r21 = evaluate_r21(
        truth_maps=truth_maps,
        predicted_maps=fixed_means,
        baseline_maps=prior_maps,
        active_mask=grid.active_mask,
        scale_ug_m3=scale_ug_m3,
        profile=r21_profile,
    )
    mixed_r21 = evaluate_r21(
        truth_maps=truth_maps,
        predicted_maps=mixed_means,
        baseline_maps=prior_maps,
        active_mask=grid.active_mask,
        scale_ug_m3=scale_ug_m3,
        profile=r21_profile,
    )
    mixed_r22 = evaluate_r22(
        truth_maps=truth_maps,
        mean_maps=mixed_means,
        standard_deviation_maps=mixed_stddev,
        active_mask=grid.active_mask,
        prior_standard_deviation_ug_m3=(
            reconstruction_config.prior_stddev_ug_m3
        ),
        profile=r22_profile,
    )
    batches_by_timestamp = {
        batch.timestamp_utc: batch for batch in run.measurement_batches
    }
    changes = []
    for timestamp, before, after in zip(
        timestamps,
        fixed_only,
        fixed_mobile,
        strict=True,
    ):
        batch = batches_by_timestamp.get(timestamp)
        if batch is None:
            continue
        mobile_cells = tuple(
            cell
            for record in batch.records
            if record.sensor_type is SensorType.MOBILE
            for cell in (grid.xy_to_cell(record.x_m, record.y_m),)
            if cell is not None and grid.active_mask[cell]
        )
        if not mobile_cells:
            continue
        changes.append(
            evaluate_uncertainty_change(
                standard_deviation_before=before.standard_deviation_ug_m3,
                standard_deviation_after=after.standard_deviation_ug_m3,
                active_mask=grid.active_mask,
                observed_cells=mobile_cells,
                cell_area_m2=grid.cell_area_m2,
                numerical_tolerance=uncertainty_numerical_tolerance,
            )
        )
    change_values = tuple(changes)
    return RetainedSyntheticEvaluation(
        run_id=run.run_id,
        fixed_only_posteriors=fixed_only,
        fixed_mobile_posteriors=fixed_mobile,
        fixed_only_r21=fixed_r21,
        fixed_mobile_r21=mixed_r21,
        fixed_mobile_r22=mixed_r22,
        mobile_uncertainty_changes=change_values,
    )


def _filtered_batches(
    batches: tuple[MeasurementBatch, ...],
    sensor_type: SensorType,
) -> tuple[MeasurementBatch, ...]:
    selected = []
    for batch in batches:
        records = tuple(
            record for record in batch.records if record.sensor_type is sensor_type
        )
        if records:
            selected.append(
                MeasurementBatch(
                    simulation_time_s=batch.simulation_time_s,
                    timestamp_utc=batch.timestamp_utc,
                    records=records,
                )
            )
    return tuple(selected)
