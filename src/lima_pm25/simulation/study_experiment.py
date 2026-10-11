from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from lima_pm25.domain import (
    SelectionStatus,
    SpatialDomain,
    StudyEnvironmentConfig,
    build_grid,
    create_lima_pm25_station_catalog,
)

from .demo import create_synthetic_study_family_requests
from .orchestration import SimulationRunResult, run_simulation
from .study_evidence import assess_grid_convergence


def load_discretization_protocol(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        protocol = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid discretization protocol JSON: {error}") from error
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("Unsupported discretization protocol")
    horizontal = _mapping("horizontal", protocol.get("horizontal"))
    temporal = _mapping("temporal", protocol.get("temporal"))
    _increasing_positive_values(
        "candidate_cell_sizes_m",
        horizontal.get("candidate_cell_sizes_m"),
        expected_count=3,
    )
    _increasing_nonnegative_values(
        "horizontal shared_times_s", horizontal.get("shared_times_s")
    )
    _fraction("probe_relative_tolerance", horizontal.get("probe_relative_tolerance"))
    _positive(
        "probe_absolute_floor_ug_m3", horizontal.get("probe_absolute_floor_ug_m3")
    )
    _fraction(
        "domain_mean_relative_tolerance",
        horizontal.get("domain_mean_relative_tolerance"),
    )
    _fraction(
        "event_excess_mass_relative_tolerance",
        horizontal.get("event_excess_mass_relative_tolerance"),
    )
    _fraction(
        "event_effect_significance_fraction_of_peak",
        horizontal.get("event_effect_significance_fraction_of_peak"),
    )
    _positive(
        "plume_centroid_tolerance_m", horizontal.get("plume_centroid_tolerance_m")
    )
    _positive(
        "mass_residual_relative_tolerance",
        horizontal.get("mass_residual_relative_tolerance"),
    )
    _increasing_positive_values(
        "candidate_output_intervals_s",
        temporal.get("candidate_output_intervals_s"),
        expected_count=3,
    )
    _increasing_nonnegative_values(
        "temporal shared_times_s", temporal.get("shared_times_s")
    )
    _fraction(
        "shared_field_l1_relative_tolerance",
        temporal.get("shared_field_l1_relative_tolerance"),
    )
    _fraction(
        "information_loss_relative_tolerance",
        temporal.get("information_loss_relative_tolerance"),
    )
    _fraction(
        "trajectory_l1_relative_tolerance",
        temporal.get("trajectory_l1_relative_tolerance"),
    )
    _fraction(
        "temporal event_effect_significance_fraction_of_peak",
        temporal.get("event_effect_significance_fraction_of_peak"),
    )
    _positive(
        "concentration_scale_floor_ug_m3",
        temporal.get("concentration_scale_floor_ug_m3"),
    )
    return protocol


def run_study_discretization_experiment(
    domain: SpatialDomain,
    config: StudyEnvironmentConfig,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    horizontal_protocol = _mapping("horizontal", protocol["horizontal"])
    temporal_protocol = _mapping("temporal", protocol["temporal"])
    protocol_fingerprint = _fingerprint(protocol)
    horizontal = _run_horizontal_experiment(
        domain,
        config,
        horizontal_protocol,
    )
    selected_cell_size_m = float(horizontal["selected_cell_size_m"])
    temporal = _run_temporal_experiment(
        domain,
        config,
        temporal_protocol,
        selected_cell_size_m,
    )
    return {
        "schema_version": 1,
        "study_configuration_fingerprint": config.fingerprint,
        "protocol_fingerprint": protocol_fingerprint,
        "protocol": protocol,
        "horizontal": horizontal,
        "temporal": temporal,
        "technical_selection": {
            "cell_size_m": selected_cell_size_m,
            "concentration_output_interval_s": temporal["selected_output_interval_s"],
            "status": "technical_selection_complete",
        },
    }


def write_discretization_evidence(
    evidence: Mapping[str, object],
    output_path: Path | str,
) -> Path:
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Discretization evidence already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _run_horizontal_experiment(
    base_domain: SpatialDomain,
    config: StudyEnvironmentConfig,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    sizes = tuple(float(value) for value in protocol["candidate_cell_sizes_m"])
    shared_times = tuple(float(value) for value in protocol["shared_times_s"])
    results = {}
    scalar_series: dict[str, dict[float, float]] = {}
    for cell_size_m in sizes:
        candidate_domain, candidate_config = _candidate_environment(
            base_domain,
            config,
            cell_size_m=cell_size_m,
            output_interval_s=min(shared_times),
            explicit_output_times_s=shared_times,
        )
        catalog = create_lima_pm25_station_catalog(candidate_domain)
        requests = create_synthetic_study_family_requests(
            candidate_domain,
            station_catalog=catalog,
            study_config=candidate_config,
        )
        started = perf_counter()
        runs = {
            member_id: run_simulation(request)
            for member_id, request in requests.items()
        }
        runtime_s = perf_counter() - started
        metrics = _horizontal_metrics(
            candidate_domain,
            catalog,
            requests,
            runs,
            shared_times,
        )
        results[cell_size_m] = {
            "shape": list(candidate_domain.grid.shape),
            "active_cells": int(np.count_nonzero(candidate_domain.grid.active_mask)),
            "runtime_s": runtime_s,
            "stored_field_bytes": sum(
                frame.concentration_ug_m3.nbytes
                for run in runs.values()
                for frame in run.frames
            ),
            "metrics": metrics,
        }
        _append_scalar_series(scalar_series, cell_size_m, metrics)

    pair_results = {
        "100_vs_50": _compare_horizontal_pair(results[100.0], results[50.0], protocol),
        "200_vs_100": _compare_horizontal_pair(
            results[200.0], results[100.0], protocol
        ),
    }
    if pair_results["100_vs_50"]["passes"]:
        selected = 200.0 if pair_results["200_vs_100"]["passes"] else 100.0
    else:
        selected = 50.0
    convergence = {
        name: assess_grid_convergence(values) for name, values in scalar_series.items()
    }
    return {
        "results_by_cell_size_m": {str(size): results[size] for size in sizes},
        "adjacent_comparisons": pair_results,
        "convergence": convergence,
        "selected_cell_size_m": selected,
        "selection_rule": protocol["selection_rule"],
        "status": "technical_selection_complete",
    }


def _run_temporal_experiment(
    base_domain: SpatialDomain,
    config: StudyEnvironmentConfig,
    protocol: Mapping[str, Any],
    selected_cell_size_m: float,
) -> dict[str, object]:
    intervals = tuple(
        float(value) for value in protocol["candidate_output_intervals_s"]
    )
    shared_times = tuple(float(value) for value in protocol["shared_times_s"])
    runs_by_interval = {}
    summaries = {}
    for interval_s in intervals:
        candidate_domain, candidate_config = _candidate_environment(
            base_domain,
            config,
            cell_size_m=selected_cell_size_m,
            output_interval_s=interval_s,
            explicit_output_times_s=shared_times,
        )
        catalog = create_lima_pm25_station_catalog(candidate_domain)
        requests = create_synthetic_study_family_requests(
            candidate_domain,
            station_catalog=catalog,
            study_config=candidate_config,
        )
        started = perf_counter()
        runs = {
            member_id: run_simulation(request)
            for member_id, request in requests.items()
        }
        runtime_s = perf_counter() - started
        runs_by_interval[interval_s] = runs
        summaries[interval_s] = {
            "runtime_s": runtime_s,
            "frame_count": sum(len(run.frames) for run in runs.values()),
            "stored_field_bytes": sum(
                frame.concentration_ug_m3.nbytes
                for run in runs.values()
                for frame in run.frames
            ),
        }

    fine_interval = intervals[0]
    comparisons = {}
    for interval_s in intervals[1:]:
        comparisons[f"{int(interval_s)}_vs_{int(fine_interval)}"] = (
            _compare_temporal_runs(
                runs_by_interval[interval_s],
                runs_by_interval[fine_interval],
                shared_times,
                protocol,
            )
        )
    selected = fine_interval
    for interval_s in intervals[1:]:
        comparison = comparisons[f"{int(interval_s)}_vs_{int(fine_interval)}"]
        if comparison["passes"]:
            selected = interval_s
    return {
        "results_by_output_interval_s": {
            str(interval): summaries[interval] for interval in intervals
        },
        "comparisons_to_300_s": comparisons,
        "selected_output_interval_s": selected,
        "selection_rule": protocol["selection_rule"],
        "status": "technical_selection_complete",
    }


def _candidate_environment(
    base_domain: SpatialDomain,
    config: StudyEnvironmentConfig,
    *,
    cell_size_m: float,
    output_interval_s: float,
    explicit_output_times_s: tuple[float, ...],
) -> tuple[SpatialDomain, StudyEnvironmentConfig]:
    domain_config = replace(config.domain, cell_size_m=cell_size_m)
    grid = build_grid(base_domain.boundary, base_domain.grid.crs, cell_size_m)
    domain = SpatialDomain(
        config=domain_config,
        road_graph=base_domain.road_graph,
        boundary=base_domain.boundary,
        grid=grid,
        graph_source=base_domain.graph_source,
        grid_source="sensitivity",
    )
    temporal = replace(
        config.temporal,
        concentration_output_interval_s=output_interval_s,
        concentration_output_times_s=explicit_output_times_s,
    )
    candidate_config = replace(
        config,
        domain=domain_config,
        temporal=temporal,
        selected_cell_size_status=SelectionStatus.PENDING_EVIDENCE,
    )
    return domain, candidate_config


def _horizontal_metrics(domain, catalog, requests, runs, times):
    metrics = {"members": {}, "event_effect": {}, "mass_balance": {}}
    for member_id, run in runs.items():
        by_time = _frames_by_time(run)
        member_metrics = {}
        for time_s in times:
            field = by_time[time_s].concentration_ug_m3
            active = field[domain.grid.active_mask]
            stations = {}
            for site in catalog.sites:
                cell = domain.grid.xy_to_cell(site.x_m, site.y_m)
                stations[site.site_id] = float(field[cell])
            member_metrics[str(time_s)] = {
                "domain_mean_ug_m3": float(np.mean(active)),
                "domain_maximum_ug_m3": float(np.max(active)),
                "total_mass_ug": float(
                    np.sum(active)
                    * domain.grid.cell_area_m2
                    * requests[member_id].model.physics.mixing_height_m
                ),
                "stations_ug_m3": stations,
            }
        metrics["members"][member_id] = member_metrics
        metrics["mass_balance"][member_id] = _maximum_relative_mass_residual(run)

    control_frames = _frames_by_time(runs["control"])
    event_frames = _frames_by_time(runs["event"])
    for time_s in times:
        difference = np.maximum(
            event_frames[time_s].concentration_ug_m3
            - control_frames[time_s].concentration_ug_m3,
            0,
        )
        active_difference = difference[domain.grid.active_mask]
        excess_mass = float(
            np.sum(active_difference)
            * domain.grid.cell_area_m2
            * requests["event"].model.physics.mixing_height_m
        )
        centroid = _field_centroid(domain, difference)
        ordering = bool(
            np.all(
                event_frames[time_s].concentration_ug_m3[domain.grid.active_mask]
                + 1e-12
                >= control_frames[time_s].concentration_ug_m3[domain.grid.active_mask]
            )
        )
        metrics["event_effect"][str(time_s)] = {
            "integrated_excess_mass_ug": excess_mass,
            "centroid_x_m": None if centroid is None else centroid[0],
            "centroid_y_m": None if centroid is None else centroid[1],
            "event_not_below_control": ordering,
        }
    return metrics


def _compare_horizontal_pair(coarse, fine, protocol):
    coarse_metrics = coarse["metrics"]
    fine_metrics = fine["metrics"]
    probe_differences = []
    mean_differences = []
    excess_differences = []
    centroid_distances = []
    ordering_stable = True
    floor = float(protocol["probe_absolute_floor_ug_m3"])
    peak_excess_mass = max(
        values["integrated_excess_mass_ug"]
        for values in fine_metrics["event_effect"].values()
    )
    significance_floor = (
        peak_excess_mass * protocol["event_effect_significance_fraction_of_peak"]
    )
    excluded_event_times = []
    for member_id, fine_times in fine_metrics["members"].items():
        for time_s, fine_values in fine_times.items():
            coarse_values = coarse_metrics["members"][member_id][time_s]
            mean_differences.append(
                _relative_difference(
                    coarse_values["domain_mean_ug_m3"],
                    fine_values["domain_mean_ug_m3"],
                    floor,
                )
            )
            for site_id, fine_value in fine_values["stations_ug_m3"].items():
                probe_differences.append(
                    _relative_difference(
                        coarse_values["stations_ug_m3"][site_id],
                        fine_value,
                        floor,
                    )
                )
    for time_s, fine_effect in fine_metrics["event_effect"].items():
        coarse_effect = coarse_metrics["event_effect"][time_s]
        if fine_effect["integrated_excess_mass_ug"] < significance_floor:
            excluded_event_times.append(time_s)
            ordering_stable = ordering_stable and bool(
                coarse_effect["event_not_below_control"]
                == fine_effect["event_not_below_control"]
                == True
            )
            continue
        excess_differences.append(
            _relative_difference(
                coarse_effect["integrated_excess_mass_ug"],
                fine_effect["integrated_excess_mass_ug"],
                np.finfo(np.float64).eps,
            )
        )
        if (
            fine_effect["centroid_x_m"] is not None
            and coarse_effect["centroid_x_m"] is not None
        ):
            centroid_distances.append(
                math.hypot(
                    coarse_effect["centroid_x_m"] - fine_effect["centroid_x_m"],
                    coarse_effect["centroid_y_m"] - fine_effect["centroid_y_m"],
                )
            )
        ordering_stable = ordering_stable and bool(
            coarse_effect["event_not_below_control"]
            == fine_effect["event_not_below_control"]
            == True
        )
    mass_residual = max(
        *coarse_metrics["mass_balance"].values(),
        *fine_metrics["mass_balance"].values(),
    )
    checks = {
        "probe_concentrations": max(probe_differences, default=0)
        <= protocol["probe_relative_tolerance"],
        "domain_mean": max(mean_differences, default=0)
        <= protocol["domain_mean_relative_tolerance"],
        "event_excess_mass": max(excess_differences, default=0)
        <= protocol["event_excess_mass_relative_tolerance"],
        "plume_centroid": max(centroid_distances, default=0)
        <= protocol["plume_centroid_tolerance_m"],
        "mass_balance": mass_residual <= protocol["mass_residual_relative_tolerance"],
        "control_event_ordering": ordering_stable,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "maximum_probe_relative_difference": max(probe_differences, default=0),
        "maximum_mean_relative_difference": max(mean_differences, default=0),
        "maximum_event_excess_mass_relative_difference": max(
            excess_differences,
            default=0,
        ),
        "maximum_centroid_distance_m": max(centroid_distances, default=0),
        "maximum_relative_mass_residual": mass_residual,
        "event_significance_floor_ug": significance_floor,
        "excluded_negligible_event_times_s": excluded_event_times,
    }


def _compare_temporal_runs(candidate_runs, fine_runs, shared_times, protocol):
    shared_differences = []
    for member_id, fine_run in fine_runs.items():
        candidate_by_time = _frames_by_time(candidate_runs[member_id])
        fine_by_time = _frames_by_time(fine_run)
        active = fine_run.final_state.grid.active_mask
        for time_s in shared_times:
            fine_field = fine_by_time[time_s].concentration_ug_m3[active]
            candidate_field = candidate_by_time[time_s].concentration_ug_m3[active]
            scale = max(float(np.sum(np.abs(fine_field))), np.finfo(np.float64).eps)
            shared_differences.append(
                float(np.sum(np.abs(candidate_field - fine_field))) / scale
            )
    fine_peaks = _temporal_peak_metrics(fine_runs)
    candidate_peaks = _temporal_peak_metrics(candidate_runs)
    floor = float(protocol["concentration_scale_floor_ug_m3"])
    losses = {
        name: max(0.0, fine_value - candidate_peaks[name]) / max(abs(fine_value), floor)
        for name, fine_value in fine_peaks.items()
    }
    trajectory = _temporal_trajectory_loss(candidate_runs, fine_runs, protocol)
    checks = {
        "shared_fields": max(shared_differences, default=0)
        <= protocol["shared_field_l1_relative_tolerance"],
        "information_loss": max(losses.values(), default=0)
        <= protocol["information_loss_relative_tolerance"],
        "trajectory_loss": max(
            trajectory["mean_total_field_l1_relative_difference"],
            trajectory["mean_meaningful_event_effect_l1_relative_difference"],
        )
        <= protocol["trajectory_l1_relative_tolerance"],
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "maximum_shared_field_l1_relative_difference": max(
            shared_differences,
            default=0,
        ),
        "information_loss_by_quantity": losses,
        "trajectory_loss": trajectory,
    }


def _temporal_trajectory_loss(candidate_runs, fine_runs, protocol):
    total_differences = []
    event_differences = []
    fine_control = _frames_by_time(fine_runs["control"])
    fine_event = _frames_by_time(fine_runs["event"])
    candidate_control = _frames_by_time(candidate_runs["control"])
    candidate_event = _frames_by_time(candidate_runs["event"])
    active = fine_runs["event"].final_state.grid.active_mask
    fine_times = sorted(set(fine_control).intersection(fine_event))
    peak_effect = max(
        float(
            np.sum(
                np.abs(
                    fine_event[time].concentration_ug_m3[active]
                    - fine_control[time].concentration_ug_m3[active]
                )
            )
        )
        for time in fine_times
    )
    effect_floor = peak_effect * protocol["event_effect_significance_fraction_of_peak"]
    excluded_times = []
    candidate_times = sorted(set(candidate_control).intersection(candidate_event))
    for time_s in fine_times:
        persisted_time = max(time for time in candidate_times if time <= time_s)
        fine_total = fine_event[time_s].concentration_ug_m3[active]
        persisted_total = candidate_event[persisted_time].concentration_ug_m3[active]
        total_scale = max(float(np.sum(np.abs(fine_total))), np.finfo(np.float64).eps)
        total_differences.append(
            float(np.sum(np.abs(persisted_total - fine_total))) / total_scale
        )
        fine_effect = (
            fine_event[time_s].concentration_ug_m3[active]
            - fine_control[time_s].concentration_ug_m3[active]
        )
        effect_scale = float(np.sum(np.abs(fine_effect)))
        if effect_scale < effect_floor:
            excluded_times.append(time_s)
            continue
        persisted_effect = (
            candidate_event[persisted_time].concentration_ug_m3[active]
            - candidate_control[persisted_time].concentration_ug_m3[active]
        )
        event_differences.append(
            float(np.sum(np.abs(persisted_effect - fine_effect))) / effect_scale
        )
    return {
        "mean_total_field_l1_relative_difference": float(np.mean(total_differences)),
        "maximum_total_field_l1_relative_difference": max(total_differences, default=0),
        "mean_meaningful_event_effect_l1_relative_difference": (
            float(np.mean(event_differences)) if event_differences else 0.0
        ),
        "maximum_meaningful_event_effect_l1_relative_difference": max(
            event_differences,
            default=0,
        ),
        "event_effect_significance_floor": effect_floor,
        "excluded_negligible_event_times_s": excluded_times,
    }


def _temporal_peak_metrics(runs):
    control_by_time = _frames_by_time(runs["control"])
    event_by_time = _frames_by_time(runs["event"])
    common_times = sorted(set(control_by_time).intersection(event_by_time))
    domain_peaks = []
    excess_mass = []
    grid = runs["event"].final_state.grid
    for time_s in common_times:
        event_field = event_by_time[time_s].concentration_ug_m3
        control_field = control_by_time[time_s].concentration_ug_m3
        domain_peaks.append(float(np.max(event_field[grid.active_mask])))
        excess_mass.append(
            float(
                np.sum(np.maximum(event_field - control_field, 0)[grid.active_mask])
                * grid.cell_area_m2
            )
        )
    return {
        "event_domain_peak_ug_m3": max(domain_peaks, default=0),
        "event_excess_area_integral_ug_m": max(excess_mass, default=0),
    }


def _frames_by_time(run: SimulationRunResult):
    return {frame.simulation_time_s: frame for frame in run.frames}


def _maximum_relative_mass_residual(run: SimulationRunResult) -> float:
    maximum = 0.0
    for diagnostic in run.diagnostics:
        budget = diagnostic.budget
        scale = max(
            abs(budget.initial_mass_ug)
            + abs(budget.emitted_mass_ug)
            + abs(budget.boundary_inflow_mass_ug),
            1.0,
        )
        maximum = max(maximum, abs(budget.numerical_residual_ug) / scale)
    return maximum


def _field_centroid(domain: SpatialDomain, field: np.ndarray):
    weights = field[domain.grid.active_mask]
    total = float(np.sum(weights))
    if total <= np.finfo(np.float64).eps:
        return None
    center_x, center_y = np.meshgrid(
        domain.grid.x_centers_m,
        domain.grid.y_centers_m,
    )
    return (
        float(np.sum(center_x[domain.grid.active_mask] * weights) / total),
        float(np.sum(center_y[domain.grid.active_mask] * weights) / total),
    )


def _append_scalar_series(series, cell_size_m, metrics):
    for member_id, times in metrics["members"].items():
        for time_s, values in times.items():
            for name in ("domain_mean_ug_m3", "total_mass_ug"):
                series.setdefault(f"{member_id}:{time_s}:{name}", {})[cell_size_m] = (
                    values[name]
                )
            for site_id, value in values["stations_ug_m3"].items():
                series.setdefault(
                    f"{member_id}:{time_s}:station:{site_id}",
                    {},
                )[cell_size_m] = value
    for time_s, values in metrics["event_effect"].items():
        series.setdefault(f"event_effect:{time_s}:excess_mass", {})[cell_size_m] = (
            values["integrated_excess_mass_ug"]
        )


def _relative_difference(candidate, reference, floor):
    return abs(float(candidate) - float(reference)) / max(abs(float(reference)), floor)


def _mapping(name, value):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")  # noqa: TRY004 -- protocol validation
    return value


def _increasing_positive_values(name, value, *, expected_count=None):
    values = _increasing_nonnegative_values(name, value)
    if any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive values")
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(f"{name} must contain {expected_count} values")
    return values


def _increasing_nonnegative_values(name, value):
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")  # noqa: TRY004 -- protocol validation
    values = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item < 0 for item in values):
        raise ValueError(f"{name} must contain nonnegative finite values")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{name} must be strictly increasing")
    return values


def _fraction(name, value):
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed < 1:
        raise ValueError(f"{name} must be in [0, 1)")


def _positive(name, value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _fingerprint(value):
    return sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
