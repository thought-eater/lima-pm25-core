import numpy as np
import pytest

from lima_pm25.estimation import (
    EvaluationStatus,
    R21AcceptanceProfile,
    R22AcceptanceProfile,
    evaluate_r21,
    evaluate_r22,
    evaluate_uncertainty_change,
    robust_p95_p05_scale,
)


def test_robust_scale_uses_only_active_calibration_truth() -> None:
    mask = np.array([[True, True], [False, True]])
    calibration = (
        np.array([[0.0, 10.0], [1_000_000.0, 20.0]]),
        np.array([[30.0, 40.0], [-1_000_000.0, 50.0]]),
    )

    scale = robust_p95_p05_scale(calibration, mask)

    expected = float(np.percentile([0, 10, 20, 30, 40, 50], 95) - np.percentile([0, 10, 20, 30, 40, 50], 5))
    assert scale == pytest.approx(expected)


def test_r21_profile_passes_and_records_normalized_and_baseline_metrics() -> None:
    mask = np.ones((2, 2), dtype=bool)
    truth = (np.array([[10.0, 20.0], [30.0, 40.0]]),)
    predicted = (np.array([[11.0, 19.0], [31.0, 39.0]]),)
    baseline = (np.full((2, 2), 20.0),)

    result = evaluate_r21(
        truth_maps=truth,
        predicted_maps=predicted,
        baseline_maps=baseline,
        active_mask=mask,
        scale_ug_m3=30,
        profile=R21AcceptanceProfile(
            maximum_nmae=0.1,
            maximum_nrmse=0.1,
            minimum_rmse_skill=0.5,
            minimum_sample_count=4,
        ),
    )

    assert result.status is EvaluationStatus.PASS
    assert result.map_count == 1
    assert result.sample_count == 4
    assert result.mae_ug_m3 == 1
    assert result.rmse_ug_m3 == 1
    assert result.nmae == pytest.approx(1 / 30)
    assert result.nrmse == pytest.approx(1 / 30)
    assert result.r2 == pytest.approx(0.992)
    assert result.rmse_skill is not None and result.rmse_skill > 0.9


def test_r21_profile_distinguishes_fail_from_invalid() -> None:
    mask = np.ones((1, 3), dtype=bool)
    truth = (np.array([[0.0, 10.0, 20.0]]),)
    bad = (np.array([[20.0, 20.0, 20.0]]),)
    baseline = (np.full((1, 3), 10.0),)

    failed = evaluate_r21(
        truth_maps=truth,
        predicted_maps=bad,
        baseline_maps=baseline,
        active_mask=mask,
        scale_ug_m3=20,
    )
    invalid = evaluate_r21(
        truth_maps=truth,
        predicted_maps=bad,
        baseline_maps=baseline,
        active_mask=mask,
        scale_ug_m3=0,
    )

    assert failed.status is EvaluationStatus.FAIL
    assert "nrmse_above_maximum" in failed.reasons
    assert invalid.status is EvaluationStatus.INVALID
    assert invalid.reasons == ("normalization_scale_not_positive",)


def test_r21_zero_target_variance_is_invalid() -> None:
    mask = np.ones((1, 2), dtype=bool)
    truth = (np.full((1, 2), 10.0),)

    result = evaluate_r21(
        truth_maps=truth,
        predicted_maps=(np.full((1, 2), 9.0),),
        baseline_maps=(np.full((1, 2), 8.0),),
        active_mask=mask,
        scale_ug_m3=5,
    )

    assert result.status is EvaluationStatus.INVALID
    assert "zero_target_variance" in result.reasons


def test_r22_profile_checks_coverage_and_interval_width() -> None:
    mask = np.ones((1, 10), dtype=bool)
    truth = (np.zeros((1, 10)),)
    mean = (np.zeros((1, 10)),)
    standard_deviation = (np.ones((1, 10)),)

    result = evaluate_r22(
        truth_maps=truth,
        mean_maps=mean,
        standard_deviation_maps=standard_deviation,
        active_mask=mask,
        prior_standard_deviation_ug_m3=2,
        profile=R22AcceptanceProfile(
            minimum_coverage=0.9,
            maximum_coverage=1.0 - 1e-12,
            minimum_sample_count=10,
        ),
    )

    assert result.status is EvaluationStatus.FAIL
    assert result.empirical_coverage == 1
    assert result.width_ratio == pytest.approx(0.5)
    assert result.reasons == ("coverage_above_maximum",)


def test_r22_profile_passes_with_calibrated_narrower_intervals() -> None:
    mask = np.ones((1, 10), dtype=bool)
    truth = np.zeros((1, 10))
    truth[0, 0] = 3

    result = evaluate_r22(
        truth_maps=(truth,),
        mean_maps=(np.zeros((1, 10)),),
        standard_deviation_maps=(np.ones((1, 10)),),
        active_mask=mask,
        prior_standard_deviation_ug_m3=2,
        profile=R22AcceptanceProfile(minimum_sample_count=10),
    )

    assert result.status is EvaluationStatus.PASS
    assert result.empirical_coverage == 0.9
    assert result.width_ratio == pytest.approx(0.5)


def test_uncertainty_change_checks_local_and_integrated_variance() -> None:
    mask = np.ones((2, 2), dtype=bool)
    before = np.full((2, 2), 2.0)
    after = np.array([[1.0, 1.5], [2.0, 2.0]])

    passed = evaluate_uncertainty_change(
        standard_deviation_before=before,
        standard_deviation_after=after,
        active_mask=mask,
        observed_cells=((0, 0),),
        cell_area_m2=100,
        numerical_tolerance=1e-12,
    )
    failed = evaluate_uncertainty_change(
        standard_deviation_before=after,
        standard_deviation_after=before,
        active_mask=mask,
        observed_cells=((0, 0),),
        cell_area_m2=100,
        numerical_tolerance=1e-12,
    )

    assert passed.status is EvaluationStatus.PASS
    assert passed.local_variance_before == (4.0,)
    assert passed.local_variance_after == (1.0,)
    assert passed.integrated_variance_after < passed.integrated_variance_before
    assert failed.status is EvaluationStatus.FAIL
    assert failed.reasons == (
        "local_variance_increased",
        "integrated_variance_increased",
    )
