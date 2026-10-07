from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lima_pm25.domain import DomainConfig, build_grid
from lima_pm25.estimation import (
    GPCalibrationCandidate,
    GPCandidateScore,
    EvaluationStatus,
    GaussianProcessFieldReconstructor,
    HistoricalPM25Dataset,
    HistoricalPM25Record,
    HistoricalPartition,
    HistoricalWindow,
    calibrate_gp_from_historical_stations,
    initial_gp_calibration_candidates,
    load_change10_calibration_artifact,
    load_change10_evaluation_profile,
    write_gp_calibration_result,
    write_change10_calibration_artifact,
)
from pyproj import Transformer
from shapely.geometry import box


def test_initial_candidate_profile_contains_the_declared_48_candidates() -> None:
    candidates = initial_gp_calibration_candidates()

    assert len(candidates) == 48
    assert {candidate.history_duration for candidate in candidates} == {
        timedelta(hours=6),
        timedelta(hours=12),
        timedelta(hours=24),
    }
    assert {candidate.spatial_length_scale_m for candidate in candidates} == {
        500,
        1_000,
        2_000,
        4_000,
    }


def test_rotating_station_calibration_selects_and_persists_one_candidate(
    tmp_path: Path,
) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    grid, coordinates = _grid_and_coordinates()
    dataset = _dataset(start, coordinates)
    window = HistoricalWindow(
        HistoricalPartition.CALIBRATION,
        start,
        start + timedelta(hours=2),
        start + timedelta(hours=8),
    )
    candidates = (
        GPCalibrationCandidate(
            history_duration=timedelta(hours=2),
            spatial_length_scale_m=1_000,
            temporal_length_scale_s=3_600,
        ),
        GPCalibrationCandidate(
            history_duration=timedelta(hours=4),
            spatial_length_scale_m=2_000,
            temporal_length_scale_s=7_200,
        ),
    )

    result = calibrate_gp_from_historical_stations(
        pm25=dataset,
        grid=grid,
        calibration_windows=(window,),
        station_ids=("A", "B", "C", "CERES"),
        observation_noise_stddev_ug_m3=1,
        candidates=candidates,
        minimum_input_record_count=2,
    )

    assert result.selected_candidate in candidates
    assert result.target_count == 18
    assert len(result.candidate_scores) == 2
    assert result.station_ids == ("A", "B", "C", "CERES")
    assert result.prior_mean_ug_m3 > 0
    assert result.prior_stddev_ug_m3 > 0
    assert result.selected_config.x_length_scale_m == (
        result.selected_candidate.spatial_length_scale_m
    )
    destination = tmp_path / "gp-calibration.json"
    write_gp_calibration_result(result, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["selection_metric"] == "aggregate_rotating_station_rmse"
    assert len(payload["candidate_scores"]) == 2
    assert payload["validation_method"] == "leave_one_station_out"
    assert set(payload["station_priors"]) == {"A", "B", "C"}
    for score in result.candidate_scores:
        assert set(score.station_scores) == {"A", "B", "C"}
        assert score.mean_prior_interval_width_ug_m3 == pytest.approx(sum(
            values["mean_prior_interval_width_ug_m3"] * values["sample_count"]
            for values in score.station_scores.values()
        ) / score.sample_count)
    with pytest.raises(FileExistsError, match="already exists"):
        write_gp_calibration_result(result, destination)


def test_held_out_values_cannot_change_their_predictions_or_priors(monkeypatch):
    start = datetime(2025, 1, 1, tzinfo=UTC)
    grid, coordinates = _grid_and_coordinates()
    original = _dataset(start, coordinates)
    window = HistoricalWindow(
        HistoricalPartition.CALIBRATION, start,
        start + timedelta(hours=2), start + timedelta(hours=8),
    )
    candidate = GPCalibrationCandidate(
        history_duration=timedelta(hours=2),
        spatial_length_scale_m=4_000, temporal_length_scale_s=3_600,
    )
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
    station_at = {transformer.transform(*point): station for station, point in coordinates.items()}
    captured = []
    reconstruct = GaussianProcessFieldReconstructor.reconstruct_points

    def inspect(self, request, positions):
        held_out = station_at[positions[0]]
        assert request.measurements
        assert all(record.sensor_id != held_out for record in request.measurements)
        assert all(
            request.reconstruction_timestamp_utc - candidate.history_duration
            <= record.timestamp_utc <= request.reconstruction_timestamp_utc
            for record in request.measurements
        )
        posterior = reconstruct(self, request, positions)
        if held_out == "A":
            captured.append((self.config.prior_mean_ug_m3, self.config.prior_stddev_ug_m3,
                             posterior.mean_ug_m3[0], posterior.standard_deviation_ug_m3[0]))
        return posterior

    monkeypatch.setattr(GaussianProcessFieldReconstructor, "reconstruct_points", inspect)
    results = []
    for dataset in (original, replace(original, records=tuple(
        replace(record, value_ug_m3=record.value_ug_m3 + 1_000)
        if record.station_id == "A" else record
        for record in original.records
    ))):
        results.append(calibrate_gp_from_historical_stations(
            pm25=dataset, grid=grid, calibration_windows=(window,),
            station_ids=("A", "B", "C", "CERES"),
            observation_noise_stddev_ug_m3=1, candidates=(candidate,),
        ))
    assert len(captured) == 12
    np.testing.assert_array_equal(captured[:6], captured[6:])
    donors = [r.value_ug_m3 for r in original.records
              if r.station_id != "A" and window.start_utc <= r.timestamp_utc < window.end_utc]
    assert results[0].station_priors["A"] == pytest.approx((np.median(donors), np.std(donors)))
    assert results[0].candidate_scores[0].station_scores["A"]["rmse_ug_m3"] != (
        results[1].candidate_scores[0].station_scores["A"]["rmse_ug_m3"]
    )


def test_historical_width_check_uses_the_held_out_priors():
    from lima_pm25.estimation.experiment import _historical_r22_status

    profile = load_change10_evaluation_profile("config/estimation_evaluation.json")
    score = GPCandidateScore(
        GPCalibrationCandidate(history_duration=timedelta(hours=6),
                               spatial_length_scale_m=2_000, temporal_length_scale_s=3_600),
        3, 1, 1, 0.95, 10, 8, {},
    )
    status, reasons = _historical_r22_status(score, profile)
    assert status is EvaluationStatus.FAIL
    assert reasons == ("historical_posterior_not_narrower_than_prior",)
    assert _historical_r22_status(
        replace(score, mean_prior_interval_width_ug_m3=12), profile,
    ) == (EvaluationStatus.PASS, ())


def test_synthetic_selection_uses_historical_selection_and_freezes_actual_result(tmp_path, monkeypatch):
    import lima_pm25.cli as cli
    import lima_pm25.estimation.experiment as experiment

    path = tmp_path / "profile.json"
    path.write_bytes(Path("config/estimation_evaluation.json").read_bytes())
    profile = load_change10_evaluation_profile(path)
    historical_config = replace(profile.reconstruction_config,
                                prior_mean_ug_m3=30, prior_stddev_ug_m3=15,
                                x_length_scale_m=4_000, y_length_scale_m=4_000,
                                temporal_length_scale_s=21_600)
    history = timedelta(hours=12)
    scores = tuple(GPCandidateScore(
        GPCalibrationCandidate(history_duration=history, spatial_length_scale_m=space,
                               temporal_length_scale_s=21_600),
        3, 1, 1, 0.80, 10, 20, {},
    ) for space in profile.synthetic_spatial_candidates_m)
    historical = SimpleNamespace(selected_config=historical_config,
                                 selected_candidate=scores[-1].candidate, candidate_scores=scores)
    run = SimpleNamespace(run_id="calibration-only", frames=(
        SimpleNamespace(concentration_ug_m3=np.array([[0.0, 50.0]])),
    ))
    monkeypatch.setattr(experiment, "create_synthetic_study_family_requests",
                        lambda *args, **kwargs: {"control": object()})
    monkeypatch.setattr(experiment, "_nominal_request", lambda *args, **kwargs: object())
    monkeypatch.setattr(experiment, "run_simulation", lambda request: run)
    seen = []

    def evaluate(**kwargs):
        config = kwargs["reconstruction_config"]
        seen.append((kwargs["history_duration"], config.temporal_length_scale_s, config.prior_mean_ug_m3))
        return SimpleNamespace(fixed_mobile_r21=SimpleNamespace(
            status=EvaluationStatus.PASS, rmse_ug_m3=abs(config.x_length_scale_m - 1_000) + 1,
        ))

    monkeypatch.setattr(experiment, "evaluate_retained_synthetic_run", evaluate)
    artifact = experiment.calibrate_change10_configuration(
        domain=SimpleNamespace(grid=SimpleNamespace(active_mask=np.ones((1, 2), dtype=bool))),
        station_catalog=object(), study_config=object(), profile=profile,
        pm25=SimpleNamespace(resource_sha256="a" * 64),
        weather=SimpleNamespace(resource_sha256="b" * 64, select_window=lambda *args: object()),
        partition=SimpleNamespace(calibration_windows=(SimpleNamespace(start_utc=None),)),
        historical_calibration=historical, historical_scale_ug_m3=40,
    )
    assert seen == [(history, 21_600, 30)] * 4
    assert artifact.synthetic_selected_spatial_m == 1_000
    assert artifact.historical_r22_status is EvaluationStatus.FAIL
    assert artifact.historical_r22_reasons == ("historical_coverage_below_minimum",)
    frozen = cli._freeze_calibrated_profile(profile, artifact)
    assert frozen.reconstruction_config == artifact.reconstruction_config
    assert frozen.history_duration == history
    assert frozen.routine_source_scale == artifact.routine_source_scale
    assert frozen.r21_profile == profile.r21_profile
    assert frozen.r22_profile == profile.r22_profile
    assert frozen.source_sha256 != profile.source_sha256
    artifact = replace(artifact, profile_sha256=frozen.source_sha256)
    destination = tmp_path / "calibration.json"
    write_change10_calibration_artifact(artifact, destination)
    loaded = load_change10_calibration_artifact(destination)
    assert loaded.descriptor() == artifact.descriptor()


def _grid_and_coordinates():
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
    lon_lat = {
        "A": (-77.0432, -12.0705),
        "B": (-76.99925, -11.9808),
        "C": (-76.9714, -12.043),
    }
    projected = {
        station: transformer.transform(longitude, latitude)
        for station, (longitude, latitude) in lon_lat.items()
    }
    x_values = [position[0] for position in projected.values()]
    y_values = [position[1] for position in projected.values()]
    boundary = box(
        min(x_values) - 1_000,
        min(y_values) - 1_000,
        max(x_values) + 1_000,
        max(y_values) + 1_000,
    )
    grid = build_grid(boundary, "EPSG:32718", 500)
    return grid, lon_lat


def _dataset(
    start: datetime,
    coordinates: dict[str, tuple[float, float]],
) -> HistoricalPM25Dataset:
    records = tuple(
        HistoricalPM25Record(
            station_id=station_id,
            timestamp_utc=start + timedelta(hours=hour),
            longitude_deg=longitude,
            latitude_deg=latitude,
            altitude_m=None,
            value_ug_m3=15 + station_index * 3 + hour,
            province="LIMA",
            district="TEST",
            source="senamhi_observed",
        )
        for hour in range(8)
        for station_index, (station_id, (longitude, latitude)) in enumerate(
            coordinates.items()
        )
    )
    return HistoricalPM25Dataset(
        records=records,
        resource_id="pm25.csv",
        resource_sha256="1" * 64,
        source_timezone="UTC",
    )
