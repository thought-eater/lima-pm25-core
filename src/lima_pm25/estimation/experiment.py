from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from lima_pm25.domain import (
    AmbientBaseline,
    FixedStationCatalog,
    RegionalBoundaryForcing,
    SpatialDomain,
    StudyEnvironmentConfig,
)
from lima_pm25.simulation import (
    HistoricalWeatherDataset,
    RegionalBoundarySchedule,
    RoadLineSource,
    SensorType,
    create_synthetic_study_family_requests,
    run_simulation,
    write_run_bundle,
)

from .evaluation import (
    EvaluationStatus,
    R21AcceptanceProfile,
    R22AcceptanceProfile,
)
from .evidence import write_retained_evaluation_bundle, export_replay_figures
from .historical import HistoricalPM25Dataset
from .partition import HistoricalPartitionManifest
from .calibration import GPCalibrationResult, GPCandidateScore
from .reconstruction import ReconstructionConfig
from .replay import compose_historical_replay_batches, reconstruct_measurement_series
from .retained import RetainedSyntheticEvaluation, evaluate_retained_synthetic_run


@dataclass(frozen=True, slots=True)
class Change10EvaluationProfile:
    status: str
    historical_observation_noise_stddev_ug_m3: float
    reconstruction_config: ReconstructionConfig
    history_duration: timedelta
    r21_profile: R21AcceptanceProfile
    r22_profile: R22AcceptanceProfile
    synthetic_calibration_member_id: str
    synthetic_spatial_candidates_m: tuple[float, ...]
    historical_station_best_spatial_m: float
    synthetic_selected_spatial_m: float
    calibration_artifact_path: Path
    routine_source_scale: float
    minimum_synthetic_scale_fraction: float
    retained_weather_window_index: int
    representative_frame_indices: tuple[int, ...]
    uncertainty_numerical_tolerance: float
    source_path: Path
    source_sha256: str
    verification: dict[str, object]


@dataclass(frozen=True, slots=True)
class Change10EvaluationResult:
    fingerprint: str
    destination: Path
    scale_ug_m3: float
    synthetic_scale_ug_m3: float
    routine_source_scale: float
    selected_config: ReconstructionConfig
    evaluation: RetainedSyntheticEvaluation
    historical_r22_status: EvaluationStatus
    historical_r22_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Change10CalibrationArtifact:
    profile_sha256: str
    pm25_resource_sha256: str
    weather_resource_sha256: str
    historical_scale_ug_m3: float
    historical_station_best_spatial_m: float
    synthetic_selected_spatial_m: float
    routine_source_scale: float
    synthetic_contrast_ug_m3: float
    historical_interval_coverage: float
    historical_mean_interval_width_ug_m3: float
    historical_mean_prior_interval_width_ug_m3: float
    historical_r22_status: EvaluationStatus
    historical_r22_reasons: tuple[str, ...]
    historical_validation_method: str
    reconstruction_config: ReconstructionConfig
    history_duration_s: float
    calibration_run_id: str
    spatial_candidate_rmse_ug_m3: tuple[tuple[float, float], ...]
    source_path: Path | None = None
    source_sha256: str | None = None

    def descriptor(self) -> dict[str, object]:
        return {
            "profile_sha256": self.profile_sha256,
            "pm25_resource_sha256": self.pm25_resource_sha256,
            "weather_resource_sha256": self.weather_resource_sha256,
            "historical_scale_ug_m3": self.historical_scale_ug_m3,
            "historical_station_best_spatial_m": (
                self.historical_station_best_spatial_m
            ),
            "synthetic_selected_spatial_m": self.synthetic_selected_spatial_m,
            "routine_source_scale": self.routine_source_scale,
            "synthetic_contrast_ug_m3": self.synthetic_contrast_ug_m3,
            "historical_interval_coverage": self.historical_interval_coverage,
            "historical_mean_interval_width_ug_m3": (
                self.historical_mean_interval_width_ug_m3
            ),
            "historical_mean_prior_interval_width_ug_m3": (
                self.historical_mean_prior_interval_width_ug_m3
            ),
            "historical_r22_status": self.historical_r22_status.value,
            "historical_r22_reasons": list(self.historical_r22_reasons),
            "historical_validation_method": self.historical_validation_method,
            "reconstruction_config": asdict(self.reconstruction_config),
            "history_duration_s": self.history_duration_s,
            "calibration_run_id": self.calibration_run_id,
            "spatial_candidate_rmse_ug_m3": {
                f"{scale:g}": rmse
                for scale, rmse in self.spatial_candidate_rmse_ug_m3
            },
        }


def load_change10_evaluation_profile(
    path: Path | str,
) -> Change10EvaluationProfile:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("Unsupported change-10 evaluation schema_version")
    if payload.get("evidence_schema_version") != 2:
        raise ValueError("Unsupported change-10 evidence_schema_version")
    gp = payload["gp"]
    r21 = payload["r21"]
    r22 = payload["r22"]
    history_duration_s = float(gp["history_duration_s"])
    tolerance = float(payload["uncertainty_numerical_tolerance"])
    noise = float(payload["historical_observation_noise_stddev_ug_m3"])
    if history_duration_s <= 0 or not math.isfinite(history_duration_s):
        raise ValueError("history_duration_s must be positive and finite")
    if noise < 0 or not math.isfinite(noise):
        raise ValueError("historical observation noise must be nonnegative")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("uncertainty tolerance must be nonnegative")
    scale_fraction = float(payload["minimum_synthetic_scale_fraction"])
    if not math.isfinite(scale_fraction) or not 0 < scale_fraction <= 1:
        raise ValueError("minimum_synthetic_scale_fraction must be in (0, 1]")
    indices = tuple(payload["representative_frame_indices"])
    if len(set(indices)) != len(indices):
        raise ValueError("representative frame indices must be unique")
    spatial_calibration = payload["spatial_calibration"]
    spatial_candidates = tuple(
        float(value) for value in spatial_calibration["candidates_m"]
    )
    if not spatial_candidates or any(
        not math.isfinite(value) or value <= 0 for value in spatial_candidates
    ):
        raise ValueError("synthetic spatial candidates must be positive")
    final_spatial = float(
        spatial_calibration["synthetic_fixed_mobile_selected_m"]
    )
    if final_spatial not in spatial_candidates:
        raise ValueError("final spatial scale must belong to calibration candidates")
    if float(gp["x_length_scale_m"]) != final_spatial or float(
        gp["y_length_scale_m"]
    ) != final_spatial:
        raise ValueError("final GP spatial scales must match calibration selection")
    routine_source_scale = float(payload["routine_source_scale"])
    if not math.isfinite(routine_source_scale) or routine_source_scale <= 0:
        raise ValueError("routine_source_scale must be positive and finite")
    retained_window_index = int(payload["retained_weather_window_index"])
    if retained_window_index < 0:
        raise ValueError("retained_weather_window_index must be nonnegative")
    return Change10EvaluationProfile(
        status=str(payload["status"]),
        historical_observation_noise_stddev_ug_m3=noise,
        reconstruction_config=ReconstructionConfig(
            prior_mean_ug_m3=float(gp["prior_mean_ug_m3"]),
            prior_stddev_ug_m3=float(gp["prior_stddev_ug_m3"]),
            x_length_scale_m=float(gp["x_length_scale_m"]),
            y_length_scale_m=float(gp["y_length_scale_m"]),
            temporal_length_scale_s=float(gp["temporal_length_scale_s"]),
        ),
        history_duration=timedelta(seconds=history_duration_s),
        r21_profile=R21AcceptanceProfile(**r21),
        r22_profile=R22AcceptanceProfile(**r22),
        synthetic_calibration_member_id=str(
            payload["synthetic_calibration_member_id"]
        ),
        synthetic_spatial_candidates_m=spatial_candidates,
        historical_station_best_spatial_m=float(
            spatial_calibration["historical_station_best_m"]
        ),
        synthetic_selected_spatial_m=final_spatial,
        calibration_artifact_path=Path(payload["calibration_artifact"]),
        routine_source_scale=routine_source_scale,
        minimum_synthetic_scale_fraction=scale_fraction,
        retained_weather_window_index=retained_window_index,
        representative_frame_indices=indices,
        uncertainty_numerical_tolerance=tolerance,
        source_path=source,
        source_sha256=sha256(source.read_bytes()).hexdigest(),
        verification=dict(payload["verification"]),
    )


def calibrate_change10_configuration(
    *,
    domain: SpatialDomain,
    station_catalog: FixedStationCatalog,
    study_config: StudyEnvironmentConfig,
    profile: Change10EvaluationProfile,
    pm25: HistoricalPM25Dataset,
    weather: HistoricalWeatherDataset,
    partition: HistoricalPartitionManifest,
    historical_calibration: GPCalibrationResult,
    historical_scale_ug_m3: float,
) -> Change10CalibrationArtifact:
    base_config = historical_calibration.selected_config
    history_duration = historical_calibration.selected_candidate.history_duration
    requests = create_synthetic_study_family_requests(
        domain,
        station_catalog=station_catalog,
        study_config=study_config,
    )
    try:
        template = requests[profile.synthetic_calibration_member_id]
    except KeyError as error:
        raise ValueError(f"Unknown configured synthetic member {error.args[0]!r}") from error
    duration = timedelta(hours=3)
    calibration_weather = weather.select_window(
        partition.calibration_windows[0].start_utc,
        duration,
    )
    required_synthetic_scale = (
        profile.minimum_synthetic_scale_fraction * historical_scale_ug_m3
    )
    routine_source_scale = 1.0
    calibration_run = None
    synthetic_scale = 0.0
    for _ in range(5):
        calibration_request = _nominal_request(
            template,
            calibration_weather,
            base_config.prior_mean_ug_m3,
            source_scale=routine_source_scale,
            seed_offset=0,
        )
        calibration_run = run_simulation(calibration_request)
        synthetic_scale = _synthetic_p99_p50_contrast(
            tuple(frame.concentration_ug_m3 for frame in calibration_run.frames),
            domain.grid.active_mask,
        )
        if synthetic_scale >= required_synthetic_scale:
            break
        routine_source_scale *= required_synthetic_scale / synthetic_scale
        if routine_source_scale > 10_000:
            break
    if synthetic_scale < required_synthetic_scale:
        raise ValueError("Synthetic calibration field remains insufficiently variable")
    assert calibration_run is not None
    calibration_evaluations = []
    for spatial_scale in profile.synthetic_spatial_candidates_m:
        candidate_config = replace(
            base_config,
            x_length_scale_m=spatial_scale,
            y_length_scale_m=spatial_scale,
        )
        candidate_evaluation = evaluate_retained_synthetic_run(
            run=calibration_run,
            reconstruction_config=candidate_config,
            history_duration=history_duration,
            scale_ug_m3=historical_scale_ug_m3,
            r21_profile=profile.r21_profile,
            r22_profile=profile.r22_profile,
            uncertainty_numerical_tolerance=(
                profile.uncertainty_numerical_tolerance
            ),
        )
        calibration_evaluations.append((candidate_config, candidate_evaluation))
    eligible_calibration_evaluations = tuple(
        item
        for item in calibration_evaluations
        if item[1].fixed_mobile_r21.status is EvaluationStatus.PASS
    )
    if not eligible_calibration_evaluations:
        raise ValueError(
            "No spatial GP candidate passes the frozen R2.1 calibration profile"
        )
    selected_config, _ = min(
        eligible_calibration_evaluations,
        key=lambda item: (
            item[1].fixed_mobile_r21.rmse_ug_m3,
            item[0].x_length_scale_m,
        ),
    )
    historical_score = next(
        score
        for score in historical_calibration.candidate_scores
        if score.candidate.history_duration == history_duration
        and score.candidate.spatial_length_scale_m
        == selected_config.x_length_scale_m
        and score.candidate.temporal_length_scale_s
        == selected_config.temporal_length_scale_s
    )
    historical_r22_status, historical_r22_reasons = _historical_r22_status(
        historical_score,
        profile,
    )
    return Change10CalibrationArtifact(
        profile_sha256=profile.source_sha256,
        pm25_resource_sha256=pm25.resource_sha256,
        weather_resource_sha256=weather.resource_sha256 or "",
        historical_scale_ug_m3=historical_scale_ug_m3,
        historical_station_best_spatial_m=(
            historical_calibration.selected_candidate.spatial_length_scale_m
        ),
        synthetic_selected_spatial_m=selected_config.x_length_scale_m,
        routine_source_scale=routine_source_scale,
        synthetic_contrast_ug_m3=synthetic_scale,
        historical_interval_coverage=historical_score.interval_coverage,
        historical_mean_interval_width_ug_m3=(
            historical_score.mean_interval_width_ug_m3
        ),
        historical_mean_prior_interval_width_ug_m3=(
            historical_score.mean_prior_interval_width_ug_m3
        ),
        historical_r22_status=historical_r22_status,
        historical_r22_reasons=historical_r22_reasons,
        historical_validation_method="leave_one_station_out",
        reconstruction_config=selected_config,
        history_duration_s=history_duration.total_seconds(),
        calibration_run_id=calibration_run.run_id,
        spatial_candidate_rmse_ug_m3=tuple(
            (
                candidate_config.x_length_scale_m,
                float(candidate_evaluation.fixed_mobile_r21.rmse_ug_m3),
            )
            for candidate_config, candidate_evaluation in calibration_evaluations
        ),
    )


def write_change10_calibration_artifact(
    artifact: Change10CalibrationArtifact,
    path: Path | str,
) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"Change-10 calibration artifact already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        f"{json.dumps(artifact.descriptor(), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return destination


def load_change10_calibration_artifact(
    path: Path | str,
) -> Change10CalibrationArtifact:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    candidate_scores = tuple(
        (float(scale), float(rmse))
        for scale, rmse in payload["spatial_candidate_rmse_ug_m3"].items()
    )
    return Change10CalibrationArtifact(
        profile_sha256=str(payload["profile_sha256"]),
        pm25_resource_sha256=str(payload["pm25_resource_sha256"]),
        weather_resource_sha256=str(payload["weather_resource_sha256"]),
        historical_scale_ug_m3=float(payload["historical_scale_ug_m3"]),
        historical_station_best_spatial_m=float(
            payload["historical_station_best_spatial_m"]
        ),
        synthetic_selected_spatial_m=float(
            payload["synthetic_selected_spatial_m"]
        ),
        routine_source_scale=float(payload["routine_source_scale"]),
        synthetic_contrast_ug_m3=float(payload["synthetic_contrast_ug_m3"]),
        historical_interval_coverage=float(
            payload["historical_interval_coverage"]
        ),
        historical_mean_interval_width_ug_m3=float(
            payload["historical_mean_interval_width_ug_m3"]
        ),
        historical_mean_prior_interval_width_ug_m3=float(
            payload["historical_mean_prior_interval_width_ug_m3"]
        ),
        historical_r22_status=EvaluationStatus(
            payload["historical_r22_status"]
        ),
        historical_r22_reasons=tuple(payload["historical_r22_reasons"]),
        historical_validation_method=str(payload["historical_validation_method"]),
        reconstruction_config=ReconstructionConfig(**payload["reconstruction_config"]),
        history_duration_s=float(payload["history_duration_s"]),
        calibration_run_id=str(payload["calibration_run_id"]),
        spatial_candidate_rmse_ug_m3=candidate_scores,
        source_path=source,
        source_sha256=sha256(source.read_bytes()).hexdigest(),
    )


def run_change10_retained_evaluation(
    *,
    domain: SpatialDomain,
    station_catalog: FixedStationCatalog,
    study_config: StudyEnvironmentConfig,
    profile: Change10EvaluationProfile,
    calibration: Change10CalibrationArtifact,
    weather: HistoricalWeatherDataset,
    partition: HistoricalPartitionManifest,
    output_parent: Path | str,
) -> Change10EvaluationResult:
    if calibration.profile_sha256 != profile.source_sha256:
        raise ValueError("Calibration artifact does not match evaluation profile")
    if calibration.weather_resource_sha256 != weather.resource_sha256:
        raise ValueError("Calibration artifact does not match weather resource")
    if calibration.synthetic_selected_spatial_m != profile.synthetic_selected_spatial_m:
        raise ValueError("Calibration artifact does not match final GP spatial scale")
    if calibration.reconstruction_config != profile.reconstruction_config or (
        calibration.history_duration_s != profile.history_duration.total_seconds()
    ):
        raise ValueError("Calibration artifact does not match frozen GP configuration")
    if calibration.routine_source_scale != profile.routine_source_scale:
        raise ValueError("Calibration artifact does not match frozen source scale")
    requests = create_synthetic_study_family_requests(
        domain,
        station_catalog=station_catalog,
        study_config=study_config,
    )
    template = requests[profile.synthetic_calibration_member_id]
    duration = timedelta(hours=3)
    retained_weather = weather.select_window(
        partition.replay_windows[profile.retained_weather_window_index].start_utc,
        duration,
    )
    retained_request = _nominal_request(
        template,
        retained_weather,
        profile.reconstruction_config.prior_mean_ug_m3,
        source_scale=calibration.routine_source_scale,
        seed_offset=1,
    )
    retained_run = run_simulation(retained_request)
    evaluation = evaluate_retained_synthetic_run(
        run=retained_run,
        reconstruction_config=profile.reconstruction_config,
        history_duration=profile.history_duration,
        scale_ug_m3=calibration.historical_scale_ug_m3,
        r21_profile=profile.r21_profile,
        r22_profile=profile.r22_profile,
        uncertainty_numerical_tolerance=profile.uncertainty_numerical_tolerance,
    )
    identity = {
        "profile_sha256": profile.source_sha256,
        "calibration_artifact_sha256": calibration.source_sha256,
        "profile_status": profile.status,
        "calibration_run_id": calibration.calibration_run_id,
        "retained_run_id": retained_run.run_id,
        "historical_scale_ug_m3": calibration.historical_scale_ug_m3,
        "synthetic_scale_ug_m3": calibration.synthetic_contrast_ug_m3,
        "routine_source_scale": calibration.routine_source_scale,
        "historical_r22_status": calibration.historical_r22_status.value,
        "historical_r22_reasons": list(calibration.historical_r22_reasons),
        "historical_validation_method": calibration.historical_validation_method,
        "selected_spatial_length_scale_m": profile.reconstruction_config.x_length_scale_m,
    }
    fingerprint = sha256(
        json.dumps(identity, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    destination = Path(output_parent) / fingerprint
    if destination.exists():
        raise FileExistsError(f"Change-10 evaluation already exists: {destination}")
    destination.mkdir(parents=True)
    try:
        write_run_bundle(retained_run, destination / "synthetic_retained")
        write_retained_evaluation_bundle(
            evaluation=evaluation,
            run=retained_run,
            reconstruction_config=profile.reconstruction_config,
            history_duration=profile.history_duration,
            scale_ug_m3=calibration.historical_scale_ug_m3,
            destination=destination / "r2_evaluation",
            representative_frame_indices=profile.representative_frame_indices,
        )
        (destination / "identity.json").write_text(
            f"{json.dumps({'fingerprint': fingerprint, **identity}, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    except BaseException:
        _remove_tree(destination)
        raise
    return Change10EvaluationResult(
        fingerprint,
        destination,
        calibration.historical_scale_ug_m3,
        calibration.synthetic_contrast_ug_m3,
        calibration.routine_source_scale,
        profile.reconstruction_config,
        evaluation,
        calibration.historical_r22_status,
        calibration.historical_r22_reasons,
    )


def _nominal_request(template, weather, ambient_value: float, *, source_scale: float, seed_offset: int):
    sources = tuple(
        RoadLineSource(
            source_id=source.source_id,
            line=source.line,
            emission_rate_ug_m_s=source.emission_rate_ug_m_s * source_scale,
            provenance=(
                f"output-constrained provisional nominal source; scale={source_scale:g}; "
                f"base={source.provenance}"
            ),
            starts_at_s=0.0,
            ends_at_s=weather.duration_s,
            role=source.role,
        )
        for source in template.sources
    )
    initial = template.model.initialize_uniform(
        AmbientBaseline(
            provider_id="provisional-nominal-ambient",
            concentration_ug_m3=ambient_value,
            provenance="historical calibration median; not observed field truth",
        )
    )
    return replace(
        template,
        initial_state=initial,
        weather=weather,
        config=replace(
            template.config,
            start_utc=weather.start_utc,
            duration_s=weather.duration_s,
            seed=template.config.seed + seed_offset,
        ),
        sources=sources,
        regional_boundary=RegionalBoundarySchedule.constant(
            RegionalBoundaryForcing(
                provider_id="provisional-nominal-boundary",
                concentration_ug_m3=ambient_value,
                provenance="historical calibration median boundary",
            )
        ),
        scenario_identity=None,
    )


def _historical_r22_status(
    score: GPCandidateScore,
    profile: Change10EvaluationProfile,
) -> tuple[EvaluationStatus, tuple[str, ...]]:
    reasons = []
    if score.interval_coverage < profile.r22_profile.minimum_coverage:
        reasons.append("historical_coverage_below_minimum")
    if score.interval_coverage > profile.r22_profile.maximum_coverage:
        reasons.append("historical_coverage_above_maximum")
    if score.mean_interval_width_ug_m3 >= score.mean_prior_interval_width_ug_m3:
        reasons.append("historical_posterior_not_narrower_than_prior")
    return (EvaluationStatus.FAIL, tuple(reasons)) if reasons else (EvaluationStatus.PASS, ())


def _synthetic_p99_p50_contrast(
    maps: tuple[np.ndarray, ...],
    active_mask: np.ndarray,
) -> float:
    values = np.concatenate([np.asarray(field)[active_mask] for field in maps])
    p50, p99 = np.percentile(values, (50, 99), method="linear")
    contrast = float(p99 - p50)
    if not math.isfinite(contrast) or contrast <= 0:
        raise ValueError("Synthetic P99-P50 contrast must be positive")
    return contrast


def write_provisional_historical_replay(
    *,
    domain: SpatialDomain,
    station_catalog: FixedStationCatalog,
    study_config: StudyEnvironmentConfig,
    profile: Change10EvaluationProfile,
    pm25: HistoricalPM25Dataset,
    weather: HistoricalWeatherDataset,
    partition: HistoricalPartitionManifest,
    destination: Path | str,
    reconstruction_config: ReconstructionConfig | None = None,
    routine_source_scale: float = 1.0,
) -> Path:
    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"Historical replay already exists: {output}")
    template = create_synthetic_study_family_requests(
        domain,
        station_catalog=station_catalog,
        study_config=study_config,
    )[profile.synthetic_calibration_member_id]
    window = partition.replay_windows[0]
    duration = min(timedelta(hours=3), window.duration)
    historical_weather = weather.select_window(window.start_utc, duration)
    replay_end = window.start_utc + duration
    observed_values = [
        float(record.value_ug_m3)
        for record in pm25.observed_records
        if window.context_start_utc <= record.timestamp_utc < replay_end
        and record.station_id in partition.station_ids
        and record.value_ug_m3 is not None
    ]
    if not observed_values:
        raise ValueError("Historical replay window has no observed PM2.5 values")
    ambient_value = float(np.median(observed_values))
    initial_state = template.model.initialize_uniform(
        AmbientBaseline(
            provider_id="historical-replay-ambient",
            concentration_ug_m3=ambient_value,
            provenance=(
                "median of eligible observed PM2.5 context; model input, not field truth"
            ),
        )
    )
    mobile_configs = tuple(
        config
        for config in template.sensor_configs
        if config.sensor_type is SensorType.MOBILE
    )
    request = replace(
        template,
        initial_state=initial_state,
        weather=historical_weather,
        config=replace(
            template.config,
            start_utc=window.start_utc,
            duration_s=duration.total_seconds(),
            concentration_output_interval_s=3_600,
            concentration_output_times_s=(),
        ),
        sensor_configs=mobile_configs,
        sources=tuple(
            RoadLineSource(
                source_id=source.source_id,
                line=source.line,
                emission_rate_ug_m_s=(
                    source.emission_rate_ug_m_s * routine_source_scale
                ),
                provenance=(
                    "frozen output-constrained replay source; "
                    f"scale={routine_source_scale:g}; base={source.provenance}"
                ),
                starts_at_s=0.0,
                ends_at_s=duration.total_seconds(),
                role=source.role,
            )
            for source in template.sources
        ),
        regional_boundary=RegionalBoundarySchedule.constant(
            RegionalBoundaryForcing(
                provider_id="historical-replay-boundary",
                concentration_ug_m3=ambient_value,
                provenance=(
                    "historical-context median boundary; constrained replay input"
                ),
            )
        ),
        scenario_identity=None,
    )
    run = run_simulation(request)
    fixed = pm25.observed_measurements(
        grid=domain.grid,
        replay_start_utc=window.start_utc,
        end_utc=replay_end,
        noise_stddev_ug_m3=(
            profile.historical_observation_noise_stddev_ug_m3
        ),
        station_ids=partition.station_ids,
    )
    batches = compose_historical_replay_batches(
        historical_fixed_records=fixed,
        simulated_mobile_attempts=run.measurement_attempts,
    )
    posteriors = reconstruct_measurement_series(
        grid=domain.grid,
        config=reconstruction_config or profile.reconstruction_config,
        batches=batches,
        reconstruction_timestamps_utc=tuple(
            frame.timestamp_utc for frame in run.frames
        ),
        history_duration=profile.history_duration,
    )
    output.mkdir(parents=True)
    try:
        write_run_bundle(run, output / "model_run")
        np.savez_compressed(
            output / "posterior_maps.npz",
            timestamps_utc=np.asarray(
                [posterior.reconstruction_timestamp_utc.isoformat() for posterior in posteriors]
            ),
            active_mask=domain.grid.active_mask,
            mean_ug_m3=np.stack([posterior.mean_ug_m3 for posterior in posteriors]),
            stddev_ug_m3=np.stack(
                [posterior.standard_deviation_ug_m3 for posterior in posteriors]
            ),
        )
        fixed_counts = {
            station_id: sum(record.sensor_id == station_id for record in fixed)
            for station_id in partition.station_ids
        }
        metadata = {
            "purpose": "provisional_historical_deployment_proxy",
            "window_start_utc": window.start_utc.isoformat(),
            "window_end_utc": replay_end.isoformat(),
            "historical_weather_resource": weather.resource_id,
            "fixed_observed_counts": fixed_counts,
            "mobile_measurement_kind": "model_sampled",
            "hidden_field_available_to_gp": False,
            "profile_status": profile.status,
            "limitations": [
                "The complete replay field is model-generated, not observed Lima truth.",
                "Routine source rates and unresolved physics remain provisional.",
                "Ceres has no historical PM2.5 readings in the supplied product.",
            ],
        }
        (output / "metadata.json").write_text(
            f"{json.dumps(metadata, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        export_replay_figures(
            output, grid=domain.grid,
            config=reconstruction_config or profile.reconstruction_config,
            measurements=tuple(r for batch in batches for r in batch.records),
            history_duration=profile.history_duration,
            sites=station_catalog.sites,
        )
    except BaseException:
        _remove_tree(output)
        raise
    return output


def finalize_change10_evidence(
    *,
    destination: Path | str,
    profile: Change10EvaluationProfile,
) -> Path:
    root = Path(destination)
    verification_path = root / "verification_report.json"
    checksums_path = root / "checksums.json"
    if verification_path.exists() or checksums_path.exists():
        raise FileExistsError("Change-10 evidence is already finalized")
    build_artifacts = {}
    for artifact in sorted(Path("dist").glob("lima_pm25_core-0.1.0*")):
        if artifact.is_file():
            build_artifacts[artifact.name] = sha256(artifact.read_bytes()).hexdigest()
    verification_payload = {
        "experiment_fingerprint": root.name,
        "profile_sha256": profile.source_sha256,
        **profile.verification,
        "build_artifacts_sha256": build_artifacts,
    }
    verification_path.write_text(
        f"{json.dumps(verification_payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    files = {
        artifact.relative_to(root).as_posix(): sha256(artifact.read_bytes()).hexdigest()
        for artifact in sorted(root.rglob("*"))
        if artifact.is_file() and artifact != checksums_path
    }
    checksums_path.write_text(
        f"{json.dumps({'files': files}, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return checksums_path


def verify_change10_evidence(path: Path | str) -> tuple[str, ...]:
    root = Path(path)
    checksums_path = root / "checksums.json"
    if not checksums_path.is_file():
        return ("missing:checksums.json",)
    payload = json.loads(checksums_path.read_text(encoding="utf-8"))
    expected = payload.get("files")
    if not isinstance(expected, dict):
        return ("invalid:checksums.json",)
    failures = []
    for relative, digest in sorted(expected.items()):
        artifact = root / relative
        if not artifact.is_file():
            failures.append(f"missing:{relative}")
        elif sha256(artifact.read_bytes()).hexdigest() != digest:
            failures.append(f"changed:{relative}")
    actual = {
        artifact.relative_to(root).as_posix()
        for artifact in root.rglob("*")
        if artifact.is_file() and artifact != checksums_path
    }
    failures.extend(
        f"unexpected:{relative}" for relative in sorted(actual.difference(expected))
    )
    return tuple(failures)


def _remove_tree(root: Path) -> None:
    for artifact in sorted(root.rglob("*"), reverse=True):
        if artifact.is_file():
            artifact.unlink()
        elif artifact.is_dir():
            artifact.rmdir()
    root.rmdir()
