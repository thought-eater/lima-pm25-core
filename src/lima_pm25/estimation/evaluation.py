from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from statistics import NormalDist
from typing import Sequence

import numpy as np


class EvaluationStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True, kw_only=True)
class R21AcceptanceProfile:
    maximum_nmae: float = 0.15
    maximum_nrmse: float = 0.20
    minimum_rmse_skill: float = 0.10
    minimum_sample_count: int = 1

    def __post_init__(self) -> None:
        _require_nonnegative_finite("maximum_nmae", self.maximum_nmae)
        _require_nonnegative_finite("maximum_nrmse", self.maximum_nrmse)
        _require_finite("minimum_rmse_skill", self.minimum_rmse_skill)
        _require_positive_integer("minimum_sample_count", self.minimum_sample_count)


@dataclass(frozen=True, slots=True)
class R21Evaluation:
    status: EvaluationStatus
    reasons: tuple[str, ...]
    map_count: int
    sample_count: int
    scale_ug_m3: float
    percentile_method: str
    mae_ug_m3: float | None
    rmse_ug_m3: float | None
    nmae: float | None
    nrmse: float | None
    r2: float | None
    baseline_rmse_ug_m3: float | None
    rmse_skill: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class R22AcceptanceProfile:
    nominal_coverage: float = 0.95
    minimum_coverage: float = 0.90
    maximum_coverage: float = 0.98
    require_narrower_than_prior: bool = True
    minimum_sample_count: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("nominal_coverage", self.nominal_coverage),
            ("minimum_coverage", self.minimum_coverage),
            ("maximum_coverage", self.maximum_coverage),
        ):
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if self.minimum_coverage > self.maximum_coverage:
            raise ValueError("minimum_coverage cannot exceed maximum_coverage")
        if not isinstance(self.require_narrower_than_prior, bool):
            raise ValueError("require_narrower_than_prior must be a boolean")
        _require_positive_integer("minimum_sample_count", self.minimum_sample_count)


@dataclass(frozen=True, slots=True)
class R22Evaluation:
    status: EvaluationStatus
    reasons: tuple[str, ...]
    map_count: int
    sample_count: int
    nominal_coverage: float
    empirical_coverage: float | None
    mean_interval_width_ug_m3: float | None
    prior_interval_width_ug_m3: float
    width_ratio: float | None


@dataclass(frozen=True, slots=True)
class UncertaintyChangeEvaluation:
    status: EvaluationStatus
    reasons: tuple[str, ...]
    local_variance_before: tuple[float, ...]
    local_variance_after: tuple[float, ...]
    integrated_variance_before: float | None
    integrated_variance_after: float | None


def robust_p95_p05_scale(
    calibration_truth: Sequence[np.ndarray],
    active_mask: np.ndarray,
    *,
    percentile_method: str = "linear",
) -> float:
    values = _flatten_active(calibration_truth, active_mask)
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("Calibration truth must contain finite active-cell values")
    p05, p95 = np.percentile(values, (5, 95), method=percentile_method)
    scale = float(p95 - p05)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Calibration P95-P05 scale must be positive and finite")
    return scale


def evaluate_r21(
    *,
    truth_maps: Sequence[np.ndarray],
    predicted_maps: Sequence[np.ndarray],
    baseline_maps: Sequence[np.ndarray],
    active_mask: np.ndarray,
    scale_ug_m3: float,
    profile: R21AcceptanceProfile | None = None,
    percentile_method: str = "linear",
) -> R21Evaluation:
    profile = profile or R21AcceptanceProfile()
    map_count = len(truth_maps)
    structural_reasons = _map_series_reasons(
        truth_maps,
        predicted_maps,
        baseline_maps,
        active_mask,
    )
    if not math.isfinite(scale_ug_m3) or scale_ug_m3 <= 0:
        structural_reasons.append("normalization_scale_not_positive")
    if structural_reasons:
        return R21Evaluation(
            EvaluationStatus.INVALID,
            tuple(structural_reasons),
            map_count,
            0,
            scale_ug_m3,
            percentile_method,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    truth = _flatten_active(truth_maps, active_mask)
    predicted = _flatten_active(predicted_maps, active_mask)
    baseline = _flatten_active(baseline_maps, active_mask)
    if truth.size < profile.minimum_sample_count:
        return R21Evaluation(
            EvaluationStatus.INVALID,
            ("insufficient_support",),
            map_count,
            int(truth.size),
            scale_ug_m3,
            percentile_method,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    errors = predicted - truth
    baseline_errors = baseline - truth
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_errors**2)))
    centered_truth = truth - float(np.mean(truth))
    total_variation = float(np.sum(centered_truth**2))
    if total_variation <= 0 or baseline_rmse <= 0:
        reasons = []
        if total_variation <= 0:
            reasons.append("zero_target_variance")
        if baseline_rmse <= 0:
            reasons.append("zero_baseline_rmse")
        return R21Evaluation(
            EvaluationStatus.INVALID,
            tuple(reasons),
            map_count,
            int(truth.size),
            scale_ug_m3,
            percentile_method,
            mae,
            rmse,
            mae / scale_ug_m3,
            rmse / scale_ug_m3,
            None,
            baseline_rmse,
            None,
        )

    nmae = mae / scale_ug_m3
    nrmse = rmse / scale_ug_m3
    r2 = 1 - float(np.sum(errors**2)) / total_variation
    skill = 1 - rmse / baseline_rmse
    failed = []
    if nmae > profile.maximum_nmae:
        failed.append("nmae_above_maximum")
    if nrmse > profile.maximum_nrmse:
        failed.append("nrmse_above_maximum")
    if skill < profile.minimum_rmse_skill:
        failed.append("rmse_skill_below_minimum")
    return R21Evaluation(
        EvaluationStatus.FAIL if failed else EvaluationStatus.PASS,
        tuple(failed),
        map_count,
        int(truth.size),
        scale_ug_m3,
        percentile_method,
        mae,
        rmse,
        nmae,
        nrmse,
        r2,
        baseline_rmse,
        skill,
    )


def evaluate_r22(
    *,
    truth_maps: Sequence[np.ndarray],
    mean_maps: Sequence[np.ndarray],
    standard_deviation_maps: Sequence[np.ndarray],
    active_mask: np.ndarray,
    prior_standard_deviation_ug_m3: float,
    profile: R22AcceptanceProfile | None = None,
) -> R22Evaluation:
    profile = profile or R22AcceptanceProfile()
    map_count = len(truth_maps)
    structural_reasons = _map_series_reasons(
        truth_maps,
        mean_maps,
        standard_deviation_maps,
        active_mask,
    )
    if (
        not math.isfinite(prior_standard_deviation_ug_m3)
        or prior_standard_deviation_ug_m3 <= 0
    ):
        structural_reasons.append("prior_standard_deviation_not_positive")
    if not structural_reasons:
        stddev = _flatten_active(standard_deviation_maps, active_mask)
        if np.any(stddev < 0):
            structural_reasons.append("negative_standard_deviation")
    z = NormalDist().inv_cdf(0.5 + profile.nominal_coverage / 2)
    prior_width = 2 * z * prior_standard_deviation_ug_m3
    if structural_reasons:
        return R22Evaluation(
            EvaluationStatus.INVALID,
            tuple(structural_reasons),
            map_count,
            0,
            profile.nominal_coverage,
            None,
            None,
            prior_width,
            None,
        )

    truth = _flatten_active(truth_maps, active_mask)
    mean = _flatten_active(mean_maps, active_mask)
    stddev = _flatten_active(standard_deviation_maps, active_mask)
    if truth.size < profile.minimum_sample_count:
        return R22Evaluation(
            EvaluationStatus.INVALID,
            ("insufficient_support",),
            map_count,
            int(truth.size),
            profile.nominal_coverage,
            None,
            None,
            prior_width,
            None,
        )

    lower = mean - z * stddev
    upper = mean + z * stddev
    coverage = float(np.mean((lower <= truth) & (truth <= upper)))
    width = float(np.mean(upper - lower))
    width_ratio = width / prior_width
    failed = []
    if coverage < profile.minimum_coverage:
        failed.append("coverage_below_minimum")
    if coverage > profile.maximum_coverage:
        failed.append("coverage_above_maximum")
    if profile.require_narrower_than_prior and width_ratio >= 1:
        failed.append("posterior_not_narrower_than_prior")
    return R22Evaluation(
        EvaluationStatus.FAIL if failed else EvaluationStatus.PASS,
        tuple(failed),
        map_count,
        int(truth.size),
        profile.nominal_coverage,
        coverage,
        width,
        prior_width,
        width_ratio,
    )


def evaluate_uncertainty_change(
    *,
    standard_deviation_before: np.ndarray,
    standard_deviation_after: np.ndarray,
    active_mask: np.ndarray,
    observed_cells: Sequence[tuple[int, int]],
    cell_area_m2: float,
    numerical_tolerance: float,
) -> UncertaintyChangeEvaluation:
    mask = np.asarray(active_mask, dtype=np.bool_)
    before = np.asarray(standard_deviation_before, dtype=np.float64)
    after = np.asarray(standard_deviation_after, dtype=np.float64)
    reasons = []
    if before.shape != mask.shape or after.shape != mask.shape:
        reasons.append("shape_mismatch")
    elif (
        not np.all(np.isfinite(before[mask]))
        or not np.all(np.isfinite(after[mask]))
        or np.any(before[mask] < 0)
        or np.any(after[mask] < 0)
    ):
        reasons.append("invalid_active_uncertainty")
    if not math.isfinite(cell_area_m2) or cell_area_m2 <= 0:
        reasons.append("cell_area_not_positive")
    if not math.isfinite(numerical_tolerance) or numerical_tolerance < 0:
        reasons.append("invalid_numerical_tolerance")
    cells = tuple(observed_cells)
    if not cells:
        reasons.append("no_observed_cells")
    elif not reasons:
        for row, column in cells:
            if (
                not 0 <= row < mask.shape[0]
                or not 0 <= column < mask.shape[1]
                or not mask[row, column]
            ):
                reasons.append("invalid_observed_cell")
                break
    if reasons:
        return UncertaintyChangeEvaluation(
            EvaluationStatus.INVALID,
            tuple(reasons),
            (),
            (),
            None,
            None,
        )

    variance_before = before**2
    variance_after = after**2
    local_before = tuple(float(variance_before[cell]) for cell in cells)
    local_after = tuple(float(variance_after[cell]) for cell in cells)
    integrated_before = float(np.sum(variance_before[mask]) * cell_area_m2)
    integrated_after = float(np.sum(variance_after[mask]) * cell_area_m2)
    failed = []
    if any(
        after_value > before_value + numerical_tolerance
        for before_value, after_value in zip(local_before, local_after, strict=True)
    ):
        failed.append("local_variance_increased")
    if integrated_after > integrated_before + numerical_tolerance:
        failed.append("integrated_variance_increased")
    return UncertaintyChangeEvaluation(
        EvaluationStatus.FAIL if failed else EvaluationStatus.PASS,
        tuple(failed),
        local_before,
        local_after,
        integrated_before,
        integrated_after,
    )


def _map_series_reasons(
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
    third: Sequence[np.ndarray],
    active_mask: np.ndarray,
) -> list[str]:
    reasons: list[str] = []
    if not first:
        reasons.append("no_requested_maps")
        return reasons
    if len(first) != len(second) or len(first) != len(third):
        reasons.append("map_count_mismatch")
        return reasons
    mask = np.asarray(active_mask, dtype=np.bool_)
    if mask.ndim != 2 or not np.any(mask):
        reasons.append("invalid_active_mask")
        return reasons
    for series in (first, second, third):
        for value in series:
            array = np.asarray(value, dtype=np.float64)
            if array.shape != mask.shape:
                reasons.append("map_shape_mismatch")
                return reasons
            if not np.all(np.isfinite(array[mask])):
                reasons.append("nonfinite_active_value")
                return reasons
    return reasons


def _flatten_active(
    maps: Sequence[np.ndarray],
    active_mask: np.ndarray,
) -> np.ndarray:
    mask = np.asarray(active_mask, dtype=np.bool_)
    return np.concatenate(
        [np.asarray(value, dtype=np.float64)[mask] for value in maps]
    )


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_nonnegative_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
