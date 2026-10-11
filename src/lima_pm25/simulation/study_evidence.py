from __future__ import annotations

import json
import platform
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, TypedDict

import networkx as nx
import numpy as np
import osmnx as ox

from lima_pm25.domain import (
    FixedStationCatalog,
    GridDefinition,
    PM25SourceRole,
    ScenarioFactor,
    ScenarioFamilyDefinition,
    SpatialDomain,
    StudyEnvironmentConfig,
    plot_spatial_domain,
)

from .orchestration import (
    SimulationRunRequest,
    run_simulation,
    simulation_run_request_descriptor,
)


class _SensitivityResult(TypedDict):
    shape: list[int]
    active_cells: int
    runtime_s: float
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class ResolvedScenarioFamily:
    family_id: str
    manifest: dict[str, Any]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class EnvironmentBundleVerificationResult:
    valid: bool
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class HorizontalSensitivityProtocol:
    cell_sizes_m: tuple[float, ...]
    comparison_time_s: float
    probe_points_m: tuple[tuple[str, float, float], ...]
    relative_tolerance: float
    selection_quantities: tuple[str, ...]

    def __post_init__(self) -> None:
        sizes = tuple(float(value) for value in self.cell_sizes_m)
        if len(sizes) != 3 or tuple(sorted(sizes)) != sizes:
            raise ValueError("Sensitivity requires three increasing cell sizes")
        if any(not np.isfinite(value) or value <= 0 for value in sizes):
            raise ValueError("Sensitivity cell sizes must be positive and finite")
        coarse_ratio = sizes[2] / sizes[1]
        fine_ratio = sizes[1] / sizes[0]
        if not np.isclose(coarse_ratio, fine_ratio):
            raise ValueError("Sensitivity cell sizes require one refinement ratio")
        if not np.isfinite(self.comparison_time_s) or self.comparison_time_s < 0:
            raise ValueError("comparison_time_s must be nonnegative and finite")
        if not np.isfinite(self.relative_tolerance) or not (
            0 <= self.relative_tolerance < 1
        ):
            raise ValueError("relative_tolerance must be finite and in [0, 1)")
        probes = tuple(self.probe_points_m)
        names = [probe[0] for probe in probes]
        if len(names) != len(set(names)):
            raise ValueError("Sensitivity probe names must be unique")
        if any(
            not name.strip() or not np.isfinite(x_m) or not np.isfinite(y_m)
            for name, x_m, y_m in probes
        ):
            raise ValueError("Sensitivity probes require names and finite coordinates")
        quantities = tuple(self.selection_quantities)
        if not quantities:
            raise ValueError("selection_quantities cannot be empty")
        object.__setattr__(self, "cell_sizes_m", sizes)
        object.__setattr__(self, "probe_points_m", probes)
        object.__setattr__(self, "selection_quantities", quantities)


def run_horizontal_sensitivity(
    boundary,
    crs: str,
    protocol: HorizontalSensitivityProtocol,
    request_factory: Callable[[GridDefinition], SimulationRunRequest],
) -> dict[str, object]:
    from lima_pm25.domain import build_grid

    results: dict[float, _SensitivityResult] = {}
    for cell_size_m in protocol.cell_sizes_m:
        grid = build_grid(boundary, crs, cell_size_m)
        request = request_factory(grid)
        if request.model.grid is not grid or request.initial_state.grid is not grid:
            raise ValueError("Sensitivity request must use the supplied candidate grid")
        if protocol.comparison_time_s not in request.config.output_times_s:
            raise ValueError(
                "Sensitivity comparison time must be a requested concentration output"
            )
        started = perf_counter()
        run = run_simulation(request)
        runtime_s = perf_counter() - started
        frame = next(
            frame
            for frame in run.frames
            if np.isclose(frame.simulation_time_s, protocol.comparison_time_s)
        )
        active = frame.concentration_ug_m3[grid.active_mask]
        metrics: dict[str, float] = {
            "mean_ug_m3": float(np.mean(active)),
            "maximum_ug_m3": float(np.max(active)),
            "total_mass_ug": float(
                np.sum(active)
                * grid.cell_area_m2
                * request.model.physics.mixing_height_m
            ),
        }
        for name, x_m, y_m in protocol.probe_points_m:
            cell = grid.xy_to_cell(x_m, y_m)
            if cell is None or not grid.active_mask[cell]:
                raise ValueError(
                    f"Sensitivity probe {name!r} is outside the active grid"
                )
            metrics[f"probe:{name}:ug_m3"] = float(frame.concentration_ug_m3[cell])
        results[cell_size_m] = {
            "shape": list(grid.shape),
            "active_cells": int(np.count_nonzero(grid.active_mask)),
            "runtime_s": runtime_s,
            "metrics": metrics,
        }

    available = set(next(iter(results.values()))["metrics"])
    unknown = set(protocol.selection_quantities) - available
    if unknown:
        raise ValueError(f"Unknown sensitivity quantities: {sorted(unknown)}")
    convergence = {}
    acceptable = True
    fine_size, medium_size, _coarse_size = protocol.cell_sizes_m
    for quantity in protocol.selection_quantities:
        values = {
            size: results[size]["metrics"][quantity] for size in protocol.cell_sizes_m
        }
        convergence[quantity] = assess_grid_convergence(values)
        fine_value = values[fine_size]
        medium_value = values[medium_size]
        scale = max(abs(fine_value), np.finfo(np.float64).eps)
        relative_difference = abs(fine_value - medium_value) / scale
        convergence[quantity]["fine_medium_relative_difference"] = relative_difference
        convergence[quantity]["within_declared_tolerance"] = (
            relative_difference <= protocol.relative_tolerance
        )
        acceptable = acceptable and relative_difference <= protocol.relative_tolerance

    recommendation = medium_size if acceptable else fine_size
    return {
        "protocol": {
            "cell_sizes_m": list(protocol.cell_sizes_m),
            "comparison_time_s": protocol.comparison_time_s,
            "probe_points_m": [list(probe) for probe in protocol.probe_points_m],
            "relative_tolerance": protocol.relative_tolerance,
            "selection_quantities": list(protocol.selection_quantities),
        },
        "results_by_cell_size_m": {
            str(size): results[size] for size in protocol.cell_sizes_m
        },
        "convergence": convergence,
        "experimental_recommendation_cell_size_m": recommendation,
        "recommendation_status": "technical_recommendation",
        "claim_limit": (
            "Selection is based on declared quantities; non-convergent local or "
            "peak values require separate limitations."
        ),
    }


def assess_grid_convergence(
    values_by_cell_size: Mapping[float, float],
) -> dict[str, object]:
    if len(values_by_cell_size) != 3:
        return {"status": "not_applicable", "reason": "requires_three_values"}
    sizes = sorted(float(value) for value in values_by_cell_size)
    fine_size, medium_size, coarse_size = sizes
    fine = float(values_by_cell_size[fine_size])
    medium = float(values_by_cell_size[medium_size])
    coarse = float(values_by_cell_size[coarse_size])
    if not all(np.isfinite(value) for value in (fine, medium, coarse)):
        return {"status": "not_applicable", "reason": "nonfinite_value"}
    fine_ratio = medium_size / fine_size
    coarse_ratio = coarse_size / medium_size
    if not np.isclose(fine_ratio, coarse_ratio) or fine_ratio <= 1:
        return {"status": "not_applicable", "reason": "unequal_refinement_ratio"}
    coarse_difference = coarse - medium
    fine_difference = medium - fine
    if np.isclose(coarse_difference, 0) and np.isclose(fine_difference, 0):
        return {"status": "stable_identical", "refinement_ratio": fine_ratio}
    if coarse_difference * fine_difference <= 0:
        return {"status": "not_applicable", "reason": "oscillatory"}
    if abs(fine_difference) >= abs(coarse_difference):
        return {
            "status": "not_applicable",
            "reason": "differences_do_not_shrink",
        }
    apparent_order = float(
        np.log(abs(coarse_difference / fine_difference)) / np.log(fine_ratio)
    )
    denominator = fine_ratio**apparent_order - 1
    extrapolated = fine + (fine - medium) / denominator
    gci_fine_pct = (
        None
        if np.isclose(fine, 0)
        else 1.25 * abs((fine - medium) / fine) / denominator * 100
    )
    return {
        "status": "applicable",
        "refinement_ratio": fine_ratio,
        "apparent_order": apparent_order,
        "richardson_extrapolated_value": extrapolated,
        "gci_fine_pct": gci_fine_pct,
    }


def compare_output_schedule_sensitivity(
    request: SimulationRunRequest,
    additional_output_times_s: Sequence[float],
    *,
    absolute_tolerance_ug_m3: float,
) -> dict[str, object]:
    if not np.isfinite(absolute_tolerance_ug_m3) or absolute_tolerance_ug_m3 < 0:
        raise ValueError("absolute_tolerance_ug_m3 must be nonnegative and finite")
    combined_times = tuple(
        sorted(
            {
                *request.config.concentration_output_times_s,
                *(float(value) for value in additional_output_times_s),
            }
        )
    )
    dense_request = replace(
        request,
        config=replace(
            request.config,
            concentration_output_times_s=combined_times,
        ),
    )
    baseline = run_simulation(request)
    dense = run_simulation(dense_request)
    dense_by_time = {frame.simulation_time_s: frame for frame in dense.frames}
    comparisons = []
    within_tolerance = True
    for frame in baseline.frames:
        candidate = dense_by_time[frame.simulation_time_s]
        active = request.model.grid.active_mask
        maximum_difference = float(
            np.max(
                np.abs(
                    frame.concentration_ug_m3[active]
                    - candidate.concentration_ug_m3[active]
                )
            )
        )
        accepted = maximum_difference <= absolute_tolerance_ug_m3
        within_tolerance = within_tolerance and accepted
        comparisons.append(
            {
                "simulation_time_s": frame.simulation_time_s,
                "maximum_absolute_difference_ug_m3": maximum_difference,
                "within_tolerance": accepted,
            }
        )
    return {
        "baseline_output_times_s": [
            frame.simulation_time_s for frame in baseline.frames
        ],
        "additional_output_times_s": list(additional_output_times_s),
        "dense_output_times_s": [frame.simulation_time_s for frame in dense.frames],
        "absolute_tolerance_ug_m3": absolute_tolerance_ug_m3,
        "comparisons": comparisons,
        "within_tolerance": within_tolerance,
    }


def resolve_scenario_family(
    definition: ScenarioFamilyDefinition,
    requests: Mapping[str, SimulationRunRequest],
) -> ResolvedScenarioFamily:
    expected_ids = {member.member_id for member in definition.members}
    actual_ids = set(requests)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            "Scenario-family requests do not match member definitions: "
            f"missing={missing}, unexpected={unexpected}"
        )

    descriptors = {
        member_id: simulation_run_request_descriptor(requests[member_id])
        for member_id in sorted(actual_ids)
    }
    control_descriptor = descriptors[definition.control_member_id]
    members_payload = []
    for member in definition.members:
        if member.claims_calibrated_providers:
            raise ValueError(
                "Accepted calibrated-provider claims are unavailable until "
                "change 10 defines and verifies their evidence"
            )
        descriptor = descriptors[member.member_id]
        identity = descriptor.get("scenario_identity")
        if not isinstance(identity, dict):
            raise ValueError(  # noqa: TRY004 -- invalid family lineage is a value error
                f"Scenario member {member.member_id!r} lacks per-run family lineage"
            )
        expected_identity = {
            "family_id": definition.family_id,
            "family_purpose": definition.purpose.value,
            "member_id": member.member_id,
            "is_control": member.member_id == definition.control_member_id,
            "changed_factors": [factor.value for factor in member.changed_factors],
            "claims_calibrated_providers": member.claims_calibrated_providers,
            "calibration_evidence_id": member.calibration_evidence_id,
        }
        if identity != expected_identity:
            raise ValueError(
                f"Scenario member {member.member_id!r} run lineage does not match "
                "its family definition"
            )
        actual_changes = _changed_factors(control_descriptor, descriptor)
        declared_changes = set(member.changed_factors)
        if actual_changes != declared_changes:
            undeclared = sorted(
                factor.value for factor in actual_changes - declared_changes
            )
            unchanged = sorted(
                factor.value for factor in declared_changes - actual_changes
            )
            raise ValueError(
                f"Scenario member {member.member_id!r} factor mismatch: "
                f"undeclared={undeclared}, declared_but_unchanged={unchanged}"
            )
        members_payload.append(
            {
                "member_id": member.member_id,
                "is_control": member.member_id == definition.control_member_id,
                "changed_factors": [factor.value for factor in member.changed_factors],
                "claims_calibrated_providers": member.claims_calibrated_providers,
                "calibration_evidence_id": member.calibration_evidence_id,
                "request": descriptor,
            }
        )

    control_weather = _mapping("weather", control_descriptor["weather"])
    if (
        definition.purpose.value == "historical_replay"
        and control_weather.get("kind") != "historical"
    ):
        raise ValueError("Historical replay requires historical weather lineage")

    manifest = {
        "schema_version": 1,
        "family_id": definition.family_id,
        "purpose": definition.purpose.value,
        "hidden_field_origin": (
            "model_conditioned"
            if definition.purpose.value == "historical_replay"
            else "simulator_generated"
        ),
        "complete_field_is_observed_ground_truth": False,
        "constraint_evidence_ids": list(definition.constraint_evidence_ids),
        "control_member_id": definition.control_member_id,
        "members": members_payload,
    }
    fingerprint = _fingerprint(manifest)
    return ResolvedScenarioFamily(
        family_id=definition.family_id,
        manifest={**manifest, "fingerprint": fingerprint},
        fingerprint=fingerprint,
    )


def render_scenario_family_markdown(family: ResolvedScenarioFamily) -> str:
    purpose = family.manifest["purpose"]
    origin = family.manifest["hidden_field_origin"]
    lines = [
        f"# Scenario family: {family.family_id}",
        "",
        f"- Purpose: `{purpose}`",
        f"- Fingerprint: `{family.fingerprint}`",
        f"- Control member: `{family.manifest['control_member_id']}`",
        f"- Hidden-field origin: `{origin}`",
        "- Constraint evidence: `"
        + (", ".join(family.manifest["constraint_evidence_ids"]) or "none")
        + "`",
        "- Complete observed ground truth: `false`",
        "",
        "| Member | Control | Changed factors | Weather | Calibration evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    members = family.manifest["members"]
    if not isinstance(members, list):
        raise ValueError("Resolved family members must be an array")  # noqa: TRY004 -- manifest validation
    for member in members:
        request = _mapping("family member request", member["request"])
        weather = _mapping("weather", request["weather"])
        factors = ", ".join(member["changed_factors"]) or "none"
        calibration = member["calibration_evidence_id"] or "none"
        lines.append(
            f"| `{member['member_id']}` | `{str(member['is_control']).lower()}` "
            f"| {factors} | `{weather.get('kind')}` | `{calibration}` |"
        )
    for member in members:
        request = _mapping("family member request", member["request"])
        weather = _mapping("weather", request["weather"])
        lines.extend(
            (
                "",
                f"## Member: {member['member_id']}",
                "",
                f"- Weather provider: `{weather.get('provider_id')}`",
                f"- Weather provenance: `{weather.get('provenance')}`",
                f"- Weather resource: `{weather.get('resource_id')}`",
                f"- Weather station: `{weather.get('station_id')}`",
                f"- Covered UTC window: `{weather.get('start_utc')}` to `{weather.get('end_utc')}`",
                f"- Precipitation policy: `{weather.get('precipitation_source_policy')}`",
                f"- Synthetic construction method: `{weather.get('construction_method')}`",
                f"- Synthetic seed: `{weather.get('seed')}`",
                "",
                "Sources:",
                "",
            )
        )
        sources = _list_of_mappings("sources", request["sources"])
        if not sources:
            lines.append("- none")
        for source in sources:
            lines.append(
                f"- `{source.get('source_id')}`: role=`{source.get('role')}`, "
                f"provenance=`{source.get('provenance')}`"
            )
    lines.extend(
        (
            "",
            (
                "Generated fields remain model outputs. Sparse historical PM2.5 "
                "stations are not a complete observed truth surface."
            ),
        )
    )
    return "\n".join(lines)


def render_environment_markdown(manifest: Mapping[str, object]) -> str:
    spatial = _mapping("spatial", manifest["spatial"])
    vertical = _mapping(
        "vertical_representation",
        manifest["vertical_representation"],
    )
    temporal = _mapping("temporal_policy", manifest["temporal_policy"])
    families = _list_of_mappings("scenario_families", manifest["scenario_families"])
    lines = [
        f"# Study environment: {manifest['environment_id']}",
        "",
        f"- Manifest fingerprint: `{manifest['fingerprint']}`",
        f"- CRS: `{spatial['crs']}`",
        f"- Cell size: `{spatial['cell_size_m']} m`",
        f"- Grid shape: `{spatial['shape']}`",
        f"- Active cells: `{spatial['active_cells']}`",
        f"- Effective mixing height: `{vertical['mixing_height_m']} m`",
        f"- Vertical evidence status: `{vertical['evidence_status']}`",
        f"- Temporal model: `{temporal['model']}`",
        "",
        "## Scenario families",
        "",
    ]
    for family in families:
        lines.append(f"- `{family['family_id']}` — `{family['fingerprint']}`")
    lines.extend(
        (
            "",
            "## Interpretation limits",
            "",
            "- The grid is horizontal and vertically averaged; it has no vertical layers.",
            "- Synthetic hidden fields are not complete observed Lima ground truth.",
            "- Historical replay is a model-conditioned deployment proxy, not a field pilot.",
        )
    )
    return "\n".join(lines)


def render_discretization_markdown(evidence: Mapping[str, object]) -> str:
    selection = _mapping("technical_selection", evidence["technical_selection"])
    horizontal = _mapping("horizontal", evidence["horizontal"])
    temporal = _mapping("temporal", evidence["temporal"])
    lines = [
        "# Discretization evidence",
        "",
        f"- Protocol fingerprint: `{evidence['protocol_fingerprint']}`",
        f"- Selected horizontal cell size: `{selection['cell_size_m']} m`",
        (
            "- Selected concentration-output interval: "
            f"`{selection['concentration_output_interval_s']} s`"
        ),
        f"- Status: `{selection['status']}`",
        "",
        "## Horizontal adjacent comparisons",
        "",
        "| Comparison | Passes | Probe max | Mean max | Excess-mass max | Centroid max (m) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, comparison in horizontal["adjacent_comparisons"].items():
        lines.append(
            f"| `{name}` | `{str(comparison['passes']).lower()}` | "
            f"{comparison['maximum_probe_relative_difference']:.6g} | "
            f"{comparison['maximum_mean_relative_difference']:.6g} | "
            f"{comparison['maximum_event_excess_mass_relative_difference']:.6g} | "
            f"{comparison['maximum_centroid_distance_m']:.6g} |"
        )
    lines.extend(
        (
            "",
            "## Temporal comparisons to 300 s",
            "",
            "| Comparison | Passes | Shared-field max | Mean total trajectory loss | Mean event trajectory loss |",
            "| --- | --- | ---: | ---: | ---: |",
        )
    )
    for name, comparison in temporal["comparisons_to_300_s"].items():
        trajectory = comparison["trajectory_loss"]
        lines.append(
            f"| `{name}` | `{str(comparison['passes']).lower()}` | "
            f"{comparison['maximum_shared_field_l1_relative_difference']:.6g} | "
            f"{trajectory['mean_total_field_l1_relative_difference']:.6g} | "
            f"{trajectory['mean_meaningful_event_effect_l1_relative_difference']:.6g} |"
        )
    lines.extend(("", "## Claim limits", ""))
    for claim in _mapping("protocol", evidence["protocol"]).get("claim_limits", ()):
        lines.append(f"- {claim}")
    return "\n".join(lines)


def build_environment_manifest(
    config: StudyEnvironmentConfig,
    domain: SpatialDomain,
    station_catalog: FixedStationCatalog,
    scenario_families: Sequence[ResolvedScenarioFamily],
    *,
    road_graph_path: Path | str | None = None,
    grid_path: Path | str | None = None,
    lock_path: Path | str = Path("uv.lock"),
    source_root: Path | str = Path("src/lima_pm25"),
    discretization_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if config.domain.fingerprint != domain.config.fingerprint:
        raise ValueError("Study configuration and resolved domain do not match")
    families = tuple(scenario_families)
    configured_ids = {family.family_id for family in config.scenario_families}
    resolved_ids = {family.family_id for family in families}
    if configured_ids != resolved_ids:
        raise ValueError("Resolved scenario families do not match study configuration")
    _validate_family_configuration(config, families)

    graph_source = Path(
        road_graph_path or config.domain.cache_directory / "road_graph.graphml"
    )
    grid_source = Path(grid_path or config.domain.cache_directory / "grid.npz")
    if not graph_source.is_file():
        raise FileNotFoundError(f"Road graph artifact does not exist: {graph_source}")
    if not grid_source.is_file():
        raise FileNotFoundError(f"Grid artifact does not exist: {grid_source}")

    stations = _station_metadata(domain, station_catalog)
    try:
        package_version = version("lima-pm25-core")
    except PackageNotFoundError:
        package_version = "unknown"
    lock_file = Path(lock_path)
    source_directory = Path(source_root)
    resolved_discretization = _resolved_discretization_evidence(
        config,
        discretization_evidence,
    )
    base = {
        "schema_version": 1,
        "environment_id": config.environment_id,
        "study_configuration_fingerprint": config.fingerprint,
        "study_configuration": config.to_dict(),
        "station_catalog_id": config.station_catalog_id,
        "spatial": {
            "configuration_fingerprint": domain.config.fingerprint,
            "crs": domain.grid.crs,
            "domain_bounds_m": domain.grid.domain_bounds_m.tolist(),
            "grid_bounds_m": domain.grid.grid_bounds_m.tolist(),
            "shape": list(domain.grid.shape),
            "active_cells": int(np.count_nonzero(domain.grid.active_mask)),
            "cell_size_m": domain.grid.cell_size_m,
            "road_graph": {
                "bundle_path": "spatial/road_graph.graphml",
                "sha256": _file_sha256(graph_source),
                "nodes": domain.road_graph.number_of_nodes(),
                "edges": domain.road_graph.number_of_edges(),
            },
            "grid": {
                "bundle_path": "spatial/grid.npz",
                "sha256": _file_sha256(grid_source),
            },
        },
        "stations": stations,
        "vertical_representation": {
            **asdict(config.vertical),
            "evidence_status": config.vertical.evidence_status.value,
            "interpretation": "two-dimensional vertically averaged air column",
            "effective_cell_volume_expression": "cell_area_m2 * mixing_height_m",
            "resolves_vertical_layers": False,
        },
        "temporal_policy": {
            **asdict(config.temporal),
            "concentration_output_status": (
                config.temporal.concentration_output_status.value
            ),
            "mobile_sampling_status": config.temporal.mobile_sampling_status.value,
            "model": "event_driven_multirate",
        },
        "discretization_evidence": {
            "fingerprint": _fingerprint(resolved_discretization),
            "technical_selection": resolved_discretization.get("technical_selection"),
        },
        "scenario_families": [
            {
                "family_id": family.family_id,
                "fingerprint": family.fingerprint,
            }
            for family in sorted(families, key=lambda item: item.family_id)
        ],
        "software": {
            "package": "lima-pm25-core",
            "package_version": package_version,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "networkx_version": nx.__version__,
            "osmnx_version": ox.__version__,
            "uv_lock_sha256": (
                _file_sha256(lock_file) if lock_file.is_file() else None
            ),
            "source_tree_sha256": (
                _tree_sha256(source_directory) if source_directory.is_dir() else None
            ),
        },
    }
    return {**base, "fingerprint": _fingerprint(base)}


def write_environment_bundle(
    config: StudyEnvironmentConfig,
    domain: SpatialDomain,
    station_catalog: FixedStationCatalog,
    scenario_families: Sequence[ResolvedScenarioFamily],
    output_directory: Path | str,
    *,
    discretization_evidence: Mapping[str, object] | None = None,
    road_graph_path: Path | str | None = None,
    grid_path: Path | str | None = None,
) -> Path:
    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError(
            f"Environment bundle destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    graph_source = Path(
        road_graph_path or config.domain.cache_directory / "road_graph.graphml"
    )
    grid_source = Path(grid_path or config.domain.cache_directory / "grid.npz")
    try:
        resolved_discretization = _resolved_discretization_evidence(
            config,
            discretization_evidence,
        )
        manifest = build_environment_manifest(
            config,
            domain,
            station_catalog,
            scenario_families,
            road_graph_path=graph_source,
            grid_path=grid_source,
            discretization_evidence=discretization_evidence,
        )
        _write_json(temporary / "study_configuration.json", config.to_dict())
        _write_json(temporary / "environment_manifest.json", manifest)
        (temporary / "environment_documentation.md").write_text(
            render_environment_markdown(manifest) + "\n",
            encoding="utf-8",
        )
        _write_json(
            temporary / "scenario_families.json",
            {
                "families": [
                    family.manifest
                    for family in sorted(
                        scenario_families,
                        key=lambda item: item.family_id,
                    )
                ]
            },
        )
        (temporary / "scenario_documentation.md").write_text(
            "\n\n".join(
                render_scenario_family_markdown(family)
                for family in sorted(
                    scenario_families,
                    key=lambda item: item.family_id,
                )
            )
            + "\n",
            encoding="utf-8",
        )
        _write_json(
            temporary / "discretization_evidence.json",
            resolved_discretization,
        )
        if "technical_selection" in resolved_discretization:
            (temporary / "discretization_summary.md").write_text(
                render_discretization_markdown(resolved_discretization) + "\n",
                encoding="utf-8",
            )
        plot_spatial_domain(
            domain,
            temporary / "study_map.png",
            station_catalog=station_catalog,
        )
        _write_json(
            temporary / "map_metadata.json",
            _map_metadata(domain, station_catalog, str(manifest["fingerprint"])),
        )
        spatial_directory = temporary / "spatial"
        spatial_directory.mkdir()
        shutil.copy2(graph_source, spatial_directory / "road_graph.graphml")
        shutil.copy2(grid_source, spatial_directory / "grid.npz")
        files = {
            path.relative_to(temporary).as_posix(): _file_sha256(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        _write_json(
            temporary / "checksums.json",
            {"algorithm": "sha256", "files": files},
        )
        verification = verify_environment_bundle(temporary)
        if not verification.valid:
            raise RuntimeError(
                "New environment bundle failed self-verification: "
                + ", ".join(verification.failures)
            )
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def verify_environment_bundle(
    bundle_directory: Path | str,
) -> EnvironmentBundleVerificationResult:
    directory = Path(bundle_directory)
    checksum_path = directory / "checksums.json"
    if not checksum_path.is_file():
        return EnvironmentBundleVerificationResult(False, ("missing:checksums.json",))
    try:
        payload = json.loads(checksum_path.read_text(encoding="utf-8"))
        if payload.get("algorithm") != "sha256":
            return EnvironmentBundleVerificationResult(
                False,
                ("unsupported-checksum-algorithm",),
            )
        expected = payload["files"]
        if not isinstance(expected, dict):
            raise TypeError
    except json.JSONDecodeError, KeyError, TypeError:
        return EnvironmentBundleVerificationResult(
            False,
            ("invalid:checksums.json",),
        )
    failures = []
    for name, expected_digest in sorted(expected.items()):
        path = directory / name
        if not path.is_file():
            failures.append(f"missing:{name}")
        elif _file_sha256(path) != expected_digest:
            failures.append(f"changed:{name}")
    manifest_path = directory / "environment_manifest.json"
    families_path = directory / "scenario_families.json"
    configuration_path = directory / "study_configuration.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_fingerprint = manifest.pop("fingerprint")
        if _fingerprint(manifest) != stored_fingerprint:
            failures.append("invalid:environment-manifest-fingerprint")
        if json.loads(configuration_path.read_text(encoding="utf-8")) != manifest.get(
            "study_configuration"
        ):
            failures.append("mismatch:study-configuration")
        spatial = _mapping("spatial", manifest["spatial"])
        for artifact_name in ("road_graph", "grid"):
            artifact = _mapping(artifact_name, spatial[artifact_name])
            artifact_path = directory / str(artifact["bundle_path"])
            if (
                artifact_path.is_file()
                and _file_sha256(artifact_path) != artifact["sha256"]
            ):
                failures.append(f"mismatch:{artifact_name}-manifest-hash")
        family_payload = json.loads(families_path.read_text(encoding="utf-8"))
        family_references = {
            item["family_id"]: item["fingerprint"]
            for item in manifest["scenario_families"]
        }
        for family in family_payload["families"]:
            family_copy = dict(family)
            family_fingerprint = family_copy.pop("fingerprint")
            if _fingerprint(family_copy) != family_fingerprint:
                failures.append(
                    f"invalid:scenario-family-fingerprint:{family['family_id']}"
                )
            if family_references.get(family["family_id"]) != family_fingerprint:
                failures.append(
                    f"mismatch:scenario-family-reference:{family['family_id']}"
                )
    except json.JSONDecodeError, KeyError, OSError, TypeError, ValueError:
        failures.append("invalid:environment-bundle-metadata")
    return EnvironmentBundleVerificationResult(not failures, tuple(failures))


def _validate_family_configuration(
    config: StudyEnvironmentConfig,
    families: Sequence[ResolvedScenarioFamily],
) -> None:
    configured = {family.family_id: family for family in config.scenario_families}
    expected_mobility = {
        "road_tolerance_m": config.mobility_policy.road_tolerance_m,
        "maximum_speed_m_s": config.mobility_policy.maximum_speed_m_s,
    }
    for family in families:
        definition = configured[family.family_id]
        if family.manifest["purpose"] != definition.purpose.value:
            raise ValueError(
                f"Resolved family {family.family_id!r} has the wrong purpose"
            )
        members = family.manifest["members"]
        if not isinstance(members, list):
            raise ValueError("Resolved family members must be an array")  # noqa: TRY004 -- manifest validation
        for member in members:
            request = _mapping("family member request", member["request"])
            grid = _mapping("grid", request["grid"])
            if grid["cell_size_m"] != config.domain.cell_size_m:
                raise ValueError("Scenario grid does not match selected cell size")
            physics = _mapping("physics", request["physics"])
            if physics["mixing_height_m"] != config.vertical.mixing_height_m:
                raise ValueError(
                    "Scenario mixing height does not match study configuration"
                )
            run_config = _mapping("run_config", request["run_config"])
            if (
                run_config["concentration_output_interval_s"]
                != config.temporal.concentration_output_interval_s
                or tuple(run_config["concentration_output_times_s"])
                != config.temporal.concentration_output_times_s
            ):
                raise ValueError(
                    "Scenario output policy does not match study configuration"
                )
            mobile = _list_of_mappings(
                "mobile_replay_providers",
                request["mobile_replay_providers"],
            )
            for provider in mobile:
                if provider["mobility_policy"] != expected_mobility:
                    raise ValueError(
                        "Scenario mobility policy does not match study configuration"
                    )


def _resolved_discretization_evidence(
    config: StudyEnvironmentConfig,
    evidence: Mapping[str, object] | None,
) -> dict[str, object]:
    if evidence is None:
        return {
            "horizontal_status": "pending_evidence",
            "horizontal_candidates_m": list(config.horizontal_candidate_cell_sizes_m),
            "temporal_status": "pending_evidence",
            "vertical_status": config.vertical.evidence_status.value,
        }
    resolved = dict(evidence)
    selection = _mapping(
        "technical_selection",
        resolved.get("technical_selection"),
    )
    cell_size = selection.get("cell_size_m")
    output_interval = selection.get("concentration_output_interval_s")
    if cell_size is None or output_interval is None:
        raise ValueError(
            "Discretization evidence must include selected size and interval"
        )
    if float(cell_size) != config.domain.cell_size_m:
        raise ValueError(
            "Discretization evidence cell size does not match study configuration"
        )
    if float(output_interval) != config.temporal.concentration_output_interval_s:
        raise ValueError(
            "Discretization evidence output interval does not match study configuration"
        )
    if selection.get("status") != "technical_selection_complete":
        raise ValueError("Discretization evidence is not technically complete")
    if resolved.get("study_configuration_fingerprint") != config.fingerprint:
        raise ValueError(
            "Discretization evidence does not identify the selected configuration"
        )
    return resolved


def _station_metadata(
    domain: SpatialDomain,
    station_catalog: FixedStationCatalog,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for site in station_catalog.sites:
        cell = domain.grid.xy_to_cell(site.x_m, site.y_m)
        if cell is None or not domain.grid.active_mask[cell]:
            raise ValueError(f"Station {site.site_id!r} is not in an active cell")
        result.append(
            {
                "site_id": site.site_id,
                "longitude_deg": site.longitude_deg,
                "latitude_deg": site.latitude_deg,
                "altitude_m": site.altitude_m,
                "x_m": site.x_m,
                "y_m": site.y_m,
                "crs": site.crs,
                "row": cell[0],
                "column": cell[1],
                "provenance": asdict(site.provenance),
            }
        )
    return result


def _map_metadata(
    domain: SpatialDomain,
    station_catalog: FixedStationCatalog,
    environment_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "environment_manifest_fingerprint": environment_fingerprint,
        "crs": domain.grid.crs,
        "domain_bounds_m": domain.grid.domain_bounds_m.tolist(),
        "grid_bounds_m": domain.grid.grid_bounds_m.tolist(),
        "cell_size_m": domain.grid.cell_size_m,
        "shape": list(domain.grid.shape),
        "active_cells": int(np.count_nonzero(domain.grid.active_mask)),
        "stations": _station_metadata(domain, station_catalog),
        "openstreetmap_attribution": "© OpenStreetMap contributors",
        "openstreetmap_license": "ODbL",
        "openstreetmap_copyright_url": "https://www.openstreetmap.org/copyright",
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(directory: Path) -> str:
    digest = sha256()
    files = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix == ".py"
    )
    for path in files:
        relative = path.relative_to(directory).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _changed_factors(
    control: dict[str, object],
    candidate: dict[str, object],
) -> set[ScenarioFactor]:
    control_payloads = _factor_payloads(control)
    candidate_payloads = _factor_payloads(candidate)
    return {
        factor
        for factor in ScenarioFactor
        if _canonical_json(control_payloads[factor])
        != _canonical_json(candidate_payloads[factor])
    }


def _factor_payloads(
    descriptor: dict[str, object],
) -> dict[ScenarioFactor, object]:
    sources = _list_of_mappings("sources", descriptor["sources"])
    mobile = _list_of_mappings(
        "mobile_replay_providers",
        descriptor["mobile_replay_providers"],
    )
    run_config = _mapping("run_config", descriptor["run_config"])
    initial_state = _mapping("initial_state", descriptor["initial_state"])
    return {
        ScenarioFactor.SPATIAL_DOMAIN: descriptor["grid"],
        ScenarioFactor.PHYSICS: descriptor["physics"],
        ScenarioFactor.WEATHER: descriptor["weather"],
        ScenarioFactor.AMBIENT_BASELINE: initial_state["ambient_baseline"],
        ScenarioFactor.REGIONAL_BOUNDARY: descriptor["regional_boundary"],
        ScenarioFactor.ROUTINE_SOURCES: _sources_with_role(
            sources,
            PM25SourceRole.ROUTINE_ENVIRONMENTAL,
        ),
        ScenarioFactor.SYNTHETIC_VALIDATION_SOURCES: _sources_with_role(
            sources,
            PM25SourceRole.SYNTHETIC_VALIDATION,
        ),
        ScenarioFactor.EVENT_SOURCES: _sources_with_role(
            sources,
            PM25SourceRole.COUNTERFACTUAL_EVENT,
        ),
        ScenarioFactor.STATIONS: descriptor["fixed_station_sites"],
        ScenarioFactor.SENSORS: descriptor["sensors"],
        ScenarioFactor.MOBILITY: [
            {
                "provider_id": provider["provider_id"],
                "mobility_policy": provider["mobility_policy"],
            }
            for provider in mobile
        ],
        ScenarioFactor.TRAJECTORY: [
            {key: value for key, value in provider.items() if key != "mobility_policy"}
            for provider in mobile
        ],
        ScenarioFactor.TEMPORAL_CONFIGURATION: {
            key: value for key, value in run_config.items() if key != "seed"
        },
        ScenarioFactor.SEED: run_config["seed"],
    }


def _sources_with_role(
    sources: list[dict[str, object]],
    role: PM25SourceRole,
) -> list[dict[str, object]]:
    return [source for source in sources if source.get("role") == role.value]


def _mapping(name: str, value: object) -> dict[str, Any]:
    """Read a heterogeneous JSON descriptor after checking its container."""
    if not isinstance(value, dict):
        raise ValueError(f"Run descriptor {name!r} must be an object")  # noqa: TRY004 -- descriptor validation
    return value


def _list_of_mappings(
    name: str,
    value: object,
) -> list[dict[str, object]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"Run descriptor {name!r} must be an object array")
    return value


def _fingerprint(value: object) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)
