from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np

from lima_pm25.domain import GridDefinition

from .historical import HistoricalPM25Dataset
from .partition import HistoricalPartition, HistoricalWindow
from .reconstruction import (
    GaussianProcessFieldReconstructor,
    ReconstructionConfig,
    ReconstructionRequest,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class GPCalibrationCandidate:
    history_duration: timedelta
    spatial_length_scale_m: float
    temporal_length_scale_s: float

    def __post_init__(self) -> None:
        if self.history_duration <= timedelta(0):
            raise ValueError("history_duration must be positive")
        _require_positive_finite(
            "spatial_length_scale_m",
            self.spatial_length_scale_m,
        )
        _require_positive_finite(
            "temporal_length_scale_s",
            self.temporal_length_scale_s,
        )


@dataclass(frozen=True, slots=True)
class GPCandidateScore:
    candidate: GPCalibrationCandidate
    sample_count: int
    mae_ug_m3: float
    rmse_ug_m3: float
    interval_coverage: float
    mean_interval_width_ug_m3: float
    mean_prior_interval_width_ug_m3: float
    station_scores: dict[str, dict[str, float | int]]


@dataclass(frozen=True, slots=True)
class GPCalibrationResult:
    station_ids: tuple[str, ...]
    prior_mean_ug_m3: float
    prior_stddev_ug_m3: float
    prior_method: str
    observation_noise_stddev_ug_m3: float
    selected_candidate: GPCalibrationCandidate
    selected_config: ReconstructionConfig
    candidate_scores: tuple[GPCandidateScore, ...]
    target_count: int
    selection_metric: str
    tie_breakers: tuple[str, ...]
    station_priors: dict[str, tuple[float, float]]

    def evidence_descriptor(self) -> dict[str, object]:
        return {
            "validation_method": "leave_one_station_out",
            "station_ids": list(self.station_ids),
            "station_priors": {
                station: {"prior_mean_ug_m3": mean, "prior_stddev_ug_m3": stddev}
                for station, (mean, stddev) in self.station_priors.items()
            },
            "prior_mean_ug_m3": self.prior_mean_ug_m3,
            "prior_stddev_ug_m3": self.prior_stddev_ug_m3,
            "prior_method": self.prior_method,
            "observation_noise_stddev_ug_m3": (self.observation_noise_stddev_ug_m3),
            "selected_candidate": _candidate_descriptor(self.selected_candidate),
            "selected_config": {
                "prior_mean_ug_m3": self.selected_config.prior_mean_ug_m3,
                "prior_stddev_ug_m3": self.selected_config.prior_stddev_ug_m3,
                "x_length_scale_m": self.selected_config.x_length_scale_m,
                "y_length_scale_m": self.selected_config.y_length_scale_m,
                "temporal_length_scale_s": (
                    self.selected_config.temporal_length_scale_s
                ),
                "include_clipped_observations": (
                    self.selected_config.include_clipped_observations
                ),
            },
            "candidate_scores": [
                {
                    **_candidate_descriptor(score.candidate),
                    "sample_count": score.sample_count,
                    "mae_ug_m3": score.mae_ug_m3,
                    "rmse_ug_m3": score.rmse_ug_m3,
                    "interval_coverage": score.interval_coverage,
                    "mean_interval_width_ug_m3": score.mean_interval_width_ug_m3,
                    "mean_prior_interval_width_ug_m3": score.mean_prior_interval_width_ug_m3,
                    "station_scores": score.station_scores,
                }
                for score in self.candidate_scores
            ],
            "target_count": self.target_count,
            "selection_metric": self.selection_metric,
            "tie_breakers": list(self.tie_breakers),
        }


def initial_gp_calibration_candidates() -> tuple[GPCalibrationCandidate, ...]:
    return tuple(
        GPCalibrationCandidate(
            history_duration=timedelta(hours=history_hours),
            spatial_length_scale_m=spatial_m,
            temporal_length_scale_s=temporal_hours * 3_600,
        )
        for history_hours in (6, 12, 24)
        for spatial_m in (500, 1_000, 2_000, 4_000)
        for temporal_hours in (1, 3, 6, 12)
    )


def calibrate_gp_from_historical_stations(
    *,
    pm25: HistoricalPM25Dataset,
    grid: GridDefinition,
    calibration_windows: Sequence[HistoricalWindow],
    station_ids: Sequence[str],
    observation_noise_stddev_ug_m3: float,
    candidates: Sequence[GPCalibrationCandidate] | None = None,
    minimum_input_record_count: int = 1,
) -> GPCalibrationResult:
    windows = tuple(calibration_windows)
    if not windows or any(
        window.partition is not HistoricalPartition.CALIBRATION for window in windows
    ):
        raise ValueError("calibration_windows must contain calibration windows")
    selected_station_ids = tuple(station_ids)
    if not selected_station_ids or len(set(selected_station_ids)) != len(
        selected_station_ids
    ):
        raise ValueError("station_ids must be nonempty and unique")
    _require_nonnegative_finite(
        "observation_noise_stddev_ug_m3",
        observation_noise_stddev_ug_m3,
    )
    _require_positive_integer(
        "minimum_input_record_count",
        minimum_input_record_count,
    )
    candidate_values = tuple(candidates or initial_gp_calibration_candidates())
    if not candidate_values:
        raise ValueError("At least one GP calibration candidate is required")
    if len(set(candidate_values)) != len(candidate_values):
        raise ValueError("GP calibration candidates must be unique")

    window_measurements = tuple(
        pm25.observed_measurements(
            grid=grid,
            replay_start_utc=window.context_start_utc,
            end_utc=window.end_utc,
            noise_stddev_ug_m3=observation_noise_stddev_ug_m3,
            station_ids=selected_station_ids,
        )
        for window in windows
    )
    calibration_records = tuple(
        record
        for window, measurements in zip(windows, window_measurements, strict=True)
        for record in measurements
        if window.start_utc <= record.timestamp_utc < window.end_utc
        and record.value_ug_m3 is not None
    )
    calibration_values = np.asarray(
        [record.value_ug_m3 for record in calibration_records],
        dtype=np.float64,
    )
    if calibration_values.size < 2:
        raise ValueError("Historical GP calibration requires at least two targets")
    prior_mean = float(np.median(calibration_values))
    prior_stddev = float(np.std(calibration_values))
    if not math.isfinite(prior_stddev) or prior_stddev <= 0:
        raise ValueError(
            "Historical calibration cannot define positive prior uncertainty"
        )

    # One donor-only prior per held-out station, shared by all candidates. The
    # pooled prior above belongs only to the final operational configuration.
    station_priors = {}
    for station in sorted({record.sensor_id for record in calibration_records}):
        donor_values = np.asarray(
            [
                record.value_ug_m3
                for record in calibration_records
                if record.sensor_id != station
            ],
            dtype=np.float64,
        )
        if donor_values.size < 2:
            raise ValueError(
                f"Insufficient donor observations when excluding {station}"
            )
        donor_stddev = float(np.std(donor_values))
        if not math.isfinite(donor_stddev) or donor_stddev <= 0:
            raise ValueError(f"Nonpositive prior uncertainty when excluding {station}")
        station_priors[station] = (float(np.median(donor_values)), donor_stddev)

    scores = tuple(
        _score_candidate(
            candidate=candidate,
            windows=windows,
            window_measurements=window_measurements,
            grid=grid,
            station_priors=station_priors,
            minimum_input_record_count=minimum_input_record_count,
        )
        for candidate in candidate_values
    )
    if any(score.sample_count == 0 for score in scores):
        raise ValueError("A GP candidate produced no eligible calibration targets")
    selected_score = min(
        scores,
        key=lambda score: (
            score.rmse_ug_m3,
            score.mae_ug_m3,
            abs(score.interval_coverage - 0.95),
            score.mean_interval_width_ug_m3,
            score.candidate.history_duration,
            score.candidate.spatial_length_scale_m,
            score.candidate.temporal_length_scale_s,
        ),
    )
    selected_config = ReconstructionConfig(
        prior_mean_ug_m3=prior_mean,
        prior_stddev_ug_m3=prior_stddev,
        x_length_scale_m=selected_score.candidate.spatial_length_scale_m,
        y_length_scale_m=selected_score.candidate.spatial_length_scale_m,
        temporal_length_scale_s=(selected_score.candidate.temporal_length_scale_s),
    )
    return GPCalibrationResult(
        station_ids=selected_station_ids,
        prior_mean_ug_m3=prior_mean,
        prior_stddev_ug_m3=prior_stddev,
        prior_method="median_and_population_standard_deviation",
        observation_noise_stddev_ug_m3=observation_noise_stddev_ug_m3,
        selected_candidate=selected_score.candidate,
        selected_config=selected_config,
        candidate_scores=scores,
        target_count=selected_score.sample_count,
        selection_metric="aggregate_rotating_station_rmse",
        tie_breakers=(
            "mae",
            "absolute_95_percent_coverage_error",
            "mean_interval_width",
            "candidate_values",
        ),
        station_priors=station_priors,
    )


def write_gp_calibration_result(
    result: GPCalibrationResult,
    path: Path | str,
) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"GP calibration result already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary GP calibration result already exists: {temporary}"
        )
    serialized = json.dumps(result.evidence_descriptor(), indent=2, sort_keys=True)
    try:
        temporary.write_text(f"{serialized}\n", encoding="utf-8")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _score_candidate(
    *,
    candidate: GPCalibrationCandidate,
    windows: tuple[HistoricalWindow, ...],
    window_measurements: tuple[tuple, ...],
    grid: GridDefinition,
    station_priors: dict[str, tuple[float, float]],
    minimum_input_record_count: int,
) -> GPCandidateScore:
    reconstructors = {
        station: GaussianProcessFieldReconstructor(
            ReconstructionConfig(
                prior_mean_ug_m3=mean,
                prior_stddev_ug_m3=stddev,
                x_length_scale_m=candidate.spatial_length_scale_m,
                y_length_scale_m=candidate.spatial_length_scale_m,
                temporal_length_scale_s=candidate.temporal_length_scale_s,
            )
        )
        for station, (mean, stddev) in station_priors.items()
    }
    targets: list[float] = []
    means: list[float] = []
    standard_deviations: list[float] = []
    prior_standard_deviations: list[float] = []
    target_stations: list[str] = []
    for window, measurements in zip(windows, window_measurements, strict=True):
        for target in measurements:
            if not window.start_utc <= target.timestamp_utc < window.end_utc:
                continue
            history_start = target.timestamp_utc - candidate.history_duration
            inputs = tuple(
                record
                for record in measurements
                if history_start <= record.timestamp_utc <= target.timestamp_utc
                and record.sensor_id != target.sensor_id
            )
            if len(inputs) < minimum_input_record_count:
                continue
            posterior = reconstructors[target.sensor_id].reconstruct_points(
                ReconstructionRequest(
                    grid=grid,
                    reconstruction_timestamp_utc=target.timestamp_utc,
                    measurements=inputs,
                ),
                ((target.x_m, target.y_m),),
            )
            assert target.value_ug_m3 is not None
            targets.append(float(target.value_ug_m3))
            means.append(float(posterior.mean_ug_m3[0]))
            standard_deviations.append(float(posterior.standard_deviation_ug_m3[0]))
            prior_standard_deviations.append(station_priors[target.sensor_id][1])
            target_stations.append(target.sensor_id)
    if not targets:
        raise ValueError("A GP candidate produced no eligible calibration targets")
    target_array = np.asarray(targets)
    mean_array = np.asarray(means)
    stddev_array = np.asarray(standard_deviations)
    errors = mean_array - target_array
    z95 = NormalDist().inv_cdf(0.975)
    covered = (mean_array - z95 * stddev_array <= target_array) & (
        target_array <= mean_array + z95 * stddev_array
    )
    prior_widths = 2 * z95 * np.asarray(prior_standard_deviations)
    widths = 2 * z95 * stddev_array
    station_array = np.asarray(target_stations)
    station_scores = {}
    for station in station_priors:
        mask = station_array == station
        if not np.any(mask):
            raise ValueError(f"No eligible targets for held-out station {station}")
        station_scores[station] = {
            "sample_count": int(np.count_nonzero(mask)),
            "mae_ug_m3": float(np.mean(np.abs(errors[mask]))),
            "rmse_ug_m3": float(np.sqrt(np.mean(errors[mask] ** 2))),
            "interval_coverage": float(np.mean(covered[mask])),
            "mean_interval_width_ug_m3": float(np.mean(widths[mask])),
            "mean_prior_interval_width_ug_m3": float(np.mean(prior_widths[mask])),
        }
    return GPCandidateScore(
        candidate,
        target_array.size,
        float(np.mean(np.abs(errors))),
        float(np.sqrt(np.mean(errors**2))),
        float(np.mean(covered)),
        float(np.mean(widths)),
        float(np.mean(prior_widths)),
        station_scores,
    )


def _candidate_descriptor(candidate: GPCalibrationCandidate) -> dict[str, float]:
    return {
        "history_duration_s": candidate.history_duration.total_seconds(),
        "spatial_length_scale_m": candidate.spatial_length_scale_m,
        "temporal_length_scale_s": candidate.temporal_length_scale_s,
    }


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _require_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
