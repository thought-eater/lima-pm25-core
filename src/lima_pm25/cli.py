import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from lima_pm25.domain import (
    SpatialDomain,
    create_lima_pm25_station_catalog,
    create_spatial_domain,
    load_study_environment_config,
    plot_spatial_domain,
)
from lima_pm25.estimation import (
    ParameterEvidenceClass,
    ScenarioParameterEvidence,
    build_historical_partition_manifest,
    build_lima_plausibility_profile,
    calibrate_change10_configuration,
    calibrate_gp_from_historical_stations,
    finalize_change10_evidence,
    load_change10_calibration_artifact,
    load_change10_evaluation_profile,
    load_historical_pm25_csv,
    run_change10_retained_evaluation,
    verify_change10_evidence,
    write_change10_calibration_artifact,
    write_gp_calibration_result,
    write_historical_partition_manifest,
    write_lima_plausibility_profile,
    write_provisional_historical_replay,
)
from lima_pm25.simulation import (
    EpisodeWeatherMode,
    animate_synthetic_demo,
    build_environment_manifest,
    compose_episode,
    create_synthetic_study_family_requests,
    load_discretization_protocol,
    load_episode_generation_config,
    load_historical_weather_csv,
    resolve_scenario_family,
    run_complete_synthetic_demo,
    run_simulation,
    run_study_discretization_experiment,
    verify_environment_bundle,
    write_discretization_evidence,
    write_environment_bundle,
    write_run_bundle,
)


def _print_domain_summary(domain: SpatialDomain) -> None:
    print(f"CRS proyectado: {domain.grid.crs}")
    print(f"Nodos: {domain.road_graph.number_of_nodes()}")
    print(f"Aristas: {domain.road_graph.number_of_edges()}")
    print(f"Grafo cargado desde: {domain.graph_source}")
    print(f"Grilla cargada desde: {domain.grid_source}")
    print(
        "Grilla: "
        f"{domain.grid.rows} filas x {domain.grid.columns} columnas "
        f"({domain.grid.cell_size_m:g} m)"
    )
    print(
        "Dimensión de la grilla: "
        f"{domain.grid.width_m:.2f} m x {domain.grid.height_m:.2f} m"
    )
    print(f"Celdas activas: {domain.grid.active_mask.sum()}")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("el valor debe ser mayor que cero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("el valor debe ser no negativo")
    return parsed


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ejecutar el simulador sintético de PM2.5 o examinar su dominio espacial."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--episode-config",
        type=Path,
        help="generar un episodio con fuentes temporales y meteorología particionada",
    )
    parser.add_argument("--episode-seed", type=int, default=0)
    parser.add_argument(
        "--episode-partition", choices=("training", "evaluation"), default="training"
    )
    parser.add_argument(
        "--episode-weather",
        choices=tuple(mode.value for mode in EpisodeWeatherMode),
        default=None,
    )
    parser.add_argument("--episode-output", type=Path, default=Path("cache/episodes"))
    mode.add_argument(
        "--animate",
        action="store_true",
        help="mostrar la simulación sintética como animación acelerada",
    )
    mode.add_argument(
        "--domain-only",
        action="store_true",
        help="construir y visualizar únicamente el dominio espacial estático",
    )
    mode.add_argument(
        "--study-evidence",
        action="store_true",
        help="generar el paquete configurado de evidencia del entorno de estudio R1.2",
    )
    mode.add_argument(
        "--study-sensitivity",
        action="store_true",
        help="ejecutar los experimentos espaciales y temporales predeclarados de R1.2",
    )
    mode.add_argument(
        "--historical-calibration",
        action="store_true",
        help="caracterizar los datos suministrados de Lima y calibrar candidatos del GP",
    )
    mode.add_argument(
        "--estimation-evaluation",
        action="store_true",
        help="ejecutar la evaluación retenida R2.1/R2.2 con el perfil fijado",
    )
    parser.add_argument(
        "--playback-speed",
        type=_positive_float,
        default=60.0,
        help="segundos simulados por segundo real en la animación (predeterminado: 60)",
    )
    parser.add_argument(
        "--frame-seconds",
        type=_positive_float,
        default=60.0,
        help="segundos simulados entre cuadros de animación (predeterminado: 60)",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="abrir la animación en t=0 y esperar al botón Reproducir",
    )
    parser.add_argument(
        "--study-config",
        type=Path,
        default=Path("config/study_environment.json"),
        help="configuración del estudio (predeterminada: config/study_environment.json)",
    )
    parser.add_argument(
        "--study-output",
        type=Path,
        default=Path("resultados/r1_2_entorno/paquetes"),
        help="directorio principal de los paquetes opcionales de evidencia del estudio",
    )
    parser.add_argument(
        "--discretization-evidence",
        type=Path,
        default=Path("resultados/r1_2_entorno/discretization_evidence.json"),
        help="informe de discretización que se incorpora si está disponible",
    )
    parser.add_argument(
        "--sensitivity-protocol",
        type=Path,
        default=Path("config/discretization_protocol.json"),
        help="protocolo de sensibilidad predeclarado",
    )
    parser.add_argument(
        "--sensitivity-output",
        type=Path,
        default=Path("resultados/r1_2_entorno/discretization_evidence.json"),
        help="destino nuevo del experimento; no se sobrescriben resultados",
    )
    parser.add_argument(
        "--pm25-data",
        type=Path,
        default=Path("data/pm25_lima_2025_2026_imputed.csv"),
        help="CSV histórico de PM2.5 con procedencia por dato",
    )
    parser.add_argument(
        "--weather-data",
        type=Path,
        default=Path("data/var_meteorologicas_lima_2025_2026_imputed.csv"),
        help="CSV meteorológico histórico con procedencia por dato",
    )
    parser.add_argument(
        "--weather-station",
        default="CAMPO_DE_MARTE",
        help="estación meteorológica utilizada como entrada espacialmente uniforme",
    )
    parser.add_argument(
        "--historical-output",
        type=Path,
        default=Path("resultados/r2_1_r2_2_estimacion/calibracion_historica"),
        help="directorio principal de los productos de calibración histórica",
    )
    parser.add_argument(
        "--historical-noise-stddev",
        type=_nonnegative_float,
        default=None,
        help="desviación estándar explícita del ruido de medición de PM2.5 en µg/m³",
    )
    parser.add_argument(
        "--estimation-profile",
        type=Path,
        default=Path("config/estimation_evaluation.json"),
        help="perfil fijado de evaluación R2.1/R2.2",
    )
    parser.add_argument(
        "--estimation-output",
        type=Path,
        default=Path("resultados/r2_1_r2_2_estimacion/evaluacion"),
        help="directorio principal de la evidencia de evaluación retenida",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _create_parser()
    args = parser.parse_args(argv)
    if args.estimation_evaluation:
        _run_change10_evaluation(
            args.study_config,
            args.estimation_profile,
            args.estimation_output,
            args.pm25_data,
            args.weather_data,
            args.weather_station,
        )
        return
    if args.historical_calibration:
        if args.historical_noise_stddev is None:
            parser.error(
                "--historical-calibration requiere "
                "--historical-noise-stddev; no se supone un valor científico predeterminado"
            )
        _run_historical_calibration(
            args.study_config,
            args.estimation_profile,
            args.pm25_data,
            args.weather_data,
            args.weather_station,
            args.historical_output,
            args.historical_noise_stddev,
        )
        return
    if args.study_sensitivity:
        _run_study_sensitivity(
            args.study_config,
            args.sensitivity_protocol,
            args.sensitivity_output,
        )
        return
    if args.study_evidence:
        _build_study_evidence(
            args.study_config,
            args.study_output,
            args.discretization_evidence,
        )
        return

    study_config = load_study_environment_config(args.study_config)
    domain = create_spatial_domain(study_config.domain)
    station_catalog = create_lima_pm25_station_catalog(domain)
    _print_domain_summary(domain)

    if args.episode_config:
        template = create_synthetic_study_family_requests(
            domain, station_catalog=station_catalog, study_config=study_config
        )["event"]
        weather = load_historical_weather_csv(
            args.weather_data, station_id=args.weather_station
        )
        request = compose_episode(
            template,
            weather,
            load_episode_generation_config(args.episode_config),
            seed=args.episode_seed,
            partition=args.episode_partition,
            weather_mode=args.episode_weather,
        )
        result = run_simulation(request)
        destination = args.episode_output / result.run_id
        write_run_bundle(result, destination)
        metadata = json.loads(request.episode_parameters_json)
        print(f"{metadata['weather_label']} — {metadata['partition']}")
        print(f"Inicio UTC: {request.config.start_utc.isoformat()}")
        print(f"Resultados: {destination}")
        return

    if args.domain_only:
        preview_path = study_config.domain.cache_directory / "domain_preview.png"
        plot_spatial_domain(
            domain,
            preview_path,
            station_catalog=station_catalog,
        )
        print(f"Diagnóstico espacial: {preview_path}")
        return

    if args.animate:
        final_state, budgets = animate_synthetic_demo(
            domain,
            playback_speed=args.playback_speed,
            frame_duration_s=args.frame_seconds,
            start_paused=args.start_paused,
            station_catalog=station_catalog,
        )
        results_path = Path("cache/simulations/pm25_demo")
    else:
        result, results_path = run_complete_synthetic_demo(
            domain,
            station_catalog=station_catalog,
            study_config=study_config,
        )
        final_state = result.final_state
        budgets = tuple(diagnostic.budget for diagnostic in result.diagnostics)

    total_residual = sum(budget.numerical_residual_ug for budget in budgets)
    print(f"Tiempo simulado: {final_state.simulation_time_s / 3600:g} h")
    print(f"PM2.5 máximo final: {final_state.concentration_ug_m3.max():.3f} µg/m³")
    print(f"Residual acumulado de masa: {total_residual:.6g} ug")
    print(f"Resultados: {results_path}")


def _build_study_evidence(
    config_path: Path,
    output_parent: Path,
    discretization_evidence_path: Path,
) -> None:
    study_config = load_study_environment_config(config_path)
    domain = create_spatial_domain(study_config.domain)
    station_catalog = create_lima_pm25_station_catalog(domain)
    _print_domain_summary(domain)
    requests = create_synthetic_study_family_requests(
        domain,
        station_catalog=station_catalog,
        study_config=study_config,
    )
    if len(study_config.scenario_families) != 1:
        raise ValueError(
            "The initial evidence command supports one configured study family"
        )
    resolved = resolve_scenario_family(
        study_config.scenario_families[0],
        requests,
    )
    discretization_evidence = (
        json.loads(discretization_evidence_path.read_text(encoding="utf-8"))
        if discretization_evidence_path.is_file()
        else None
    )
    manifest = build_environment_manifest(
        study_config,
        domain,
        station_catalog,
        (resolved,),
        discretization_evidence=discretization_evidence,
    )
    destination = output_parent / str(manifest["fingerprint"])
    if destination.exists():
        verification = verify_environment_bundle(destination)
        if not verification.valid:
            raise RuntimeError(
                "Existing study bundle failed verification: "
                + ", ".join(verification.failures)
            )
        existing = json.loads((destination / "environment_manifest.json").read_text())
        if existing["fingerprint"] != manifest["fingerprint"]:
            raise ValueError(
                "Existing environment differs from current inputs; remove the "
                "superseded output before regenerating it"
            )
    else:
        write_environment_bundle(
            study_config,
            domain,
            station_catalog,
            (resolved,),
            destination,
            discretization_evidence=discretization_evidence,
        )
    print(f"Evidencia R1.2: {destination}")
    if discretization_evidence is None:
        print("Estado de selección técnica: pendiente de evidencia experimental")
    else:
        selection = discretization_evidence["technical_selection"]
        print(f"Estado de selección técnica: {selection['status']}")


def _run_study_sensitivity(
    config_path: Path,
    protocol_path: Path,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Experiment output already exists: {output_path}")
    study_config = load_study_environment_config(config_path)
    protocol = load_discretization_protocol(protocol_path)
    domain = create_spatial_domain(study_config.domain)
    _print_domain_summary(domain)
    evidence = run_study_discretization_experiment(
        domain,
        study_config,
        protocol,
    )
    write_discretization_evidence(evidence, output_path)
    selection = evidence["technical_selection"]
    print(f"Evidencia de discretización: {output_path}")
    print(f"Tamaño de celda seleccionado: {selection['cell_size_m']:g} m")
    print(
        "Intervalo de salida seleccionado: "
        f"{selection['concentration_output_interval_s']:g} s"
    )


def _run_historical_calibration(
    study_config_path: Path,
    profile_path: Path,
    pm25_path: Path,
    weather_path: Path,
    weather_station_id: str,
    output_parent: Path,
    observation_noise_stddev_ug_m3: float,
) -> None:
    study_config = load_study_environment_config(study_config_path)
    profile = load_change10_evaluation_profile(profile_path)
    if profile.calibration_artifact_path.exists():
        raise FileExistsError(
            f"Calibration artifact already exists: {profile.calibration_artifact_path}"
        )
    if observation_noise_stddev_ug_m3 != (
        profile.historical_observation_noise_stddev_ug_m3
    ):
        raise ValueError(
            "Requested observation noise does not match frozen change-10 profile"
        )
    domain = create_spatial_domain(study_config.domain)
    station_catalog = create_lima_pm25_station_catalog(domain)
    pm25 = load_historical_pm25_csv(pm25_path)
    weather = load_historical_weather_csv(
        weather_path,
        station_id=weather_station_id,
    )
    station_ids = tuple(site.site_id for site in station_catalog.sites)
    partition = build_historical_partition_manifest(
        pm25=pm25,
        weather=weather,
        station_ids=station_ids,
    )
    plausibility = build_lima_plausibility_profile(
        pm25=pm25,
        weather=weather,
        parameters=(
            ScenarioParameterEvidence(
                "mixing_height_m",
                200.0,
                "m",
                ParameterEvidenceClass.SYNTHETIC,
                "altura efectiva constante adoptada como supuesto simplificador aceptado para el alcance de la tesis; no medida ni calibrada para Lima",
            ),
            ScenarioParameterEvidence(
                "horizontal_diffusivity_m2_s",
                25.0,
                "m2/s",
                ParameterEvidenceClass.SYNTHETIC,
                "difusividad horizontal constante adoptada como supuesto del modelo para el alcance de la tesis; no calibrada específicamente para Lima",
            ),
            ScenarioParameterEvidence(
                "dry_deposition_velocity_m_s",
                0.001,
                "m/s",
                ParameterEvidenceClass.LITERATURE_CONSTRAINED,
                "orden de magnitud genérico de PM2.5 sustentado en Zhang et al. 2014; no específico de Lima",
            ),
            ScenarioParameterEvidence(
                "wet_scavenging_per_mm",
                0.03,
                "1/mm",
                ParameterEvidenceClass.LITERATURE_CONSTRAINED,
                "orden de magnitud genérico del coeficiente de lavado de PM2.5 bajo nubes; precipitación escasa en Lima",
            ),
            ScenarioParameterEvidence(
                "routine_road_source_rates",
                110.0,
                "ug/(m s)",
                ParameterEvidenceClass.SYNTHETIC,
                "escala representativa de demostración; no es una tasa de emisión inferida para Lima",
            ),
        ),
    )
    calibration = calibrate_gp_from_historical_stations(
        pm25=pm25,
        grid=domain.grid,
        calibration_windows=partition.calibration_windows,
        station_ids=station_ids,
        observation_noise_stddev_ug_m3=observation_noise_stddev_ug_m3,
    )
    calibration_values = np.asarray(
        [
            float(record.value_ug_m3)
            for window in partition.calibration_windows
            for record in pm25.observed_records
            if window.start_utc <= record.timestamp_utc < window.end_utc
            and record.station_id in station_ids
            and record.value_ug_m3 is not None
        ]
    )
    p05, p95 = np.percentile(calibration_values, (5, 95), method="linear")
    historical_scale = float(p95 - p05)
    identity_payload = {
        "partition": partition.evidence_descriptor(),
        "noise_stddev_ug_m3": observation_noise_stddev_ug_m3,
        "validation_method": "leave_one_station_out",
        "gp_calibration_sha256": sha256(
            json.dumps(calibration.evidence_descriptor(), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest(),
    }
    fingerprint = sha256(
        json.dumps(
            identity_payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    destination = output_parent / fingerprint
    if destination.exists():
        raise FileExistsError(
            f"Historical calibration bundle already exists: {destination}"
        )
    destination.mkdir(parents=True)
    try:
        write_historical_partition_manifest(
            partition,
            destination / "partition.json",
        )
        write_lima_plausibility_profile(
            plausibility,
            destination / "plausibility.json",
        )
        write_gp_calibration_result(
            calibration,
            destination / "gp_calibration.json",
        )
        identity = {"fingerprint": fingerprint, **identity_payload}
        (destination / "identity.json").write_text(
            f"{json.dumps(identity, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    except BaseException:
        for artifact in destination.glob("*"):
            artifact.unlink()
        destination.rmdir()
        raise
    selected = calibration.selected_candidate
    selected_score = next(
        score for score in calibration.candidate_scores if score.candidate == selected
    )
    print(f"Calibración histórica: {destination}")
    print(f"Valores objetivo históricos: {calibration.target_count}")
    print(
        "GP seleccionado: "
        f"historial={selected.history_duration.total_seconds() / 3600:g} h, "
        f"espacio={selected.spatial_length_scale_m:g} m, "
        f"tiempo={selected.temporal_length_scale_s / 3600:g} h"
    )
    print(
        f"RMSE={selected_score.rmse_ug_m3:.3f} ug/m3, "
        f"MAE={selected_score.mae_ug_m3:.3f} ug/m3, "
        f"cobertura95={selected_score.interval_coverage:.3f}",
        flush=True,
    )
    # The historical report is already persisted if a later calibration fails.
    final_calibration = calibrate_change10_configuration(
        domain=domain,
        station_catalog=station_catalog,
        study_config=study_config,
        profile=profile,
        pm25=pm25,
        weather=weather,
        partition=partition,
        historical_calibration=calibration,
        historical_scale_ug_m3=historical_scale,
    )
    profile = _freeze_calibrated_profile(profile, final_calibration)
    final_calibration = replace(final_calibration, profile_sha256=profile.source_sha256)
    write_change10_calibration_artifact(
        final_calibration, profile.calibration_artifact_path
    )
    print(f"Estado del perfil: {profile.status}")
    print(
        f"Configuración GP final: {final_calibration.synthetic_selected_spatial_m:g} m"
    )
    print(f"Artefacto congelado: {profile.calibration_artifact_path}")
    print(f"R2.2 histórico: {final_calibration.historical_r22_status.value}")
    if final_calibration.historical_r22_reasons:
        print(
            "Criterios incumplidos: "
            + ", ".join(final_calibration.historical_r22_reasons)
        )


def _freeze_calibrated_profile(profile, calibration):
    """Persist the actual selection before any retained evaluation is opened."""
    payload = json.loads(profile.source_path.read_text(encoding="utf-8"))
    config = calibration.reconstruction_config
    payload["gp"].update(
        {
            "prior_mean_ug_m3": config.prior_mean_ug_m3,
            "prior_stddev_ug_m3": config.prior_stddev_ug_m3,
            "x_length_scale_m": config.x_length_scale_m,
            "y_length_scale_m": config.y_length_scale_m,
            "temporal_length_scale_s": config.temporal_length_scale_s,
            "history_duration_s": calibration.history_duration_s,
        }
    )
    payload["spatial_calibration"]["historical_station_best_m"] = (
        calibration.historical_station_best_spatial_m
    )
    payload["spatial_calibration"]["synthetic_fixed_mobile_selected_m"] = (
        calibration.synthetic_selected_spatial_m
    )
    payload["routine_source_scale"] = calibration.routine_source_scale
    payload["historical_validation_method"] = calibration.historical_validation_method
    temporary = profile.source_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(profile.source_path)
    return load_change10_evaluation_profile(profile.source_path)


def _run_change10_evaluation(
    study_config_path: Path,
    profile_path: Path,
    output_parent: Path,
    pm25_path: Path,
    weather_path: Path,
    weather_station_id: str,
) -> None:
    study_config = load_study_environment_config(study_config_path)
    domain = create_spatial_domain(study_config.domain)
    station_catalog = create_lima_pm25_station_catalog(domain)
    profile = load_change10_evaluation_profile(profile_path)
    calibration_artifact = load_change10_calibration_artifact(
        profile.calibration_artifact_path
    )
    pm25 = load_historical_pm25_csv(pm25_path)
    weather = load_historical_weather_csv(
        weather_path,
        station_id=weather_station_id,
    )
    station_ids = tuple(site.site_id for site in station_catalog.sites)
    partition = build_historical_partition_manifest(
        pm25=pm25,
        weather=weather,
        station_ids=station_ids,
    )
    if calibration_artifact.pm25_resource_sha256 != pm25.resource_sha256:
        raise ValueError("Calibration artifact does not match PM2.5 resource")
    result = run_change10_retained_evaluation(
        domain=domain,
        station_catalog=station_catalog,
        study_config=study_config,
        profile=profile,
        calibration=calibration_artifact,
        weather=weather,
        partition=partition,
        output_parent=output_parent,
    )
    replay_destination = write_provisional_historical_replay(
        domain=domain,
        station_catalog=station_catalog,
        study_config=study_config,
        profile=profile,
        pm25=pm25,
        weather=weather,
        partition=partition,
        destination=result.destination / "historical_replay",
        reconstruction_config=result.selected_config,
        routine_source_scale=calibration_artifact.routine_source_scale,
    )
    finalize_change10_evidence(destination=result.destination, profile=profile)
    verification_failures = verify_change10_evidence(result.destination)
    if verification_failures:
        raise RuntimeError(
            "Change-10 evidence verification failed: "
            + ", ".join(verification_failures)
        )
    print(f"Evaluación retenida R2.1/R2.2: {result.destination}")
    print(f"Escala histórica P95-P05: {result.scale_ug_m3:.3f} ug/m3")
    print(f"Escala sintética: {result.synthetic_scale_ug_m3:.3f} ug/m3")
    print(f"Escala de fuentes rutinarias: {result.routine_source_scale:.3f}")
    print(
        f"Escala espacial GP retenida: {result.selected_config.x_length_scale_m:.0f} m"
    )
    print(f"R2.1: {result.evaluation.fixed_mobile_r21.status.value}")
    print(f"R2.2 histórico: {result.historical_r22_status.value}")
    print(
        "Cobertura sintética diagnóstica: "
        f"{result.evaluation.fixed_mobile_r22.empirical_coverage:.3f}"
    )
    print(f"Reproducción histórica: {replay_destination}")
    print("Verificación de integridad del paquete: válida")
    print(f"Estado del perfil: {profile.status}")
