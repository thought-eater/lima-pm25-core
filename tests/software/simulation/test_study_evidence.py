from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path

import networkx as nx
import numpy as np
from shapely.geometry import box
import pytest

from lima_pm25.domain import (
    AmbientBaseline,
    AtmosphericForcing,
    DomainConfig,
    EffectiveVerticalRepresentation,
    EvidenceStatus,
    FixedStationCatalog,
    FixedStationSite,
    MobilityPolicy,
    PM25SourceRole,
    SENAMHI_PM25_PROVENANCE,
    ScenarioFactor,
    ScenarioFamilyDefinition,
    ScenarioFamilyPurpose,
    ScenarioMemberDefinition,
    SelectionStatus,
    SpatialDomain,
    StudyEnvironmentConfig,
    TemporalStudyPolicy,
    build_grid,
    save_grid,
)
from lima_pm25.simulation import (
    build_environment_manifest,
    GaussianKernelSource,
    HorizontalSensitivityProtocol,
    PM25PhysicsConfig,
    PM25Transport,
    ScheduledWeatherValue,
    SimulationRunConfig,
    SimulationRunRequest,
    SyntheticWeatherWindow,
    assess_grid_convergence,
    compare_output_schedule_sensitivity,
    resolve_scenario_family,
    render_scenario_family_markdown,
    render_environment_markdown,
    run_horizontal_sensitivity,
    verify_environment_bundle,
    write_environment_bundle,
)


START = datetime(2026, 9, 26, tzinfo=UTC)


def test_matched_control_and_event_family_resolves_one_declared_factor() -> None:
    definition = _family((ScenarioFactor.EVENT_SOURCES,))

    resolved = resolve_scenario_family(
        definition,
        _family_requests(definition, event_has_source=True),
    )

    assert resolved.family_id == definition.family_id
    assert len(resolved.fingerprint) == 64
    assert resolved.manifest["members"][1]["changed_factors"] == [
        "event_sources"
    ]
    assert resolved.manifest["complete_field_is_observed_ground_truth"] is False
    documentation = render_scenario_family_markdown(resolved)
    assert "Hidden-field origin: `simulator_generated`" in documentation
    assert "event_sources" in documentation
    assert "Weather provenance: `synthetic study-evidence test`" in documentation
    assert "role=`counterfactual_event`" in documentation
    assert "Complete observed ground truth: `false`" in documentation


def test_family_rejects_an_undeclared_factor_difference() -> None:
    definition = _family(())

    with pytest.raises(ValueError, match=r"undeclared=\['event_sources'\]"):
        resolve_scenario_family(
            definition,
            _family_requests(definition, event_has_source=True),
        )


def test_family_rejects_a_declared_factor_that_did_not_change() -> None:
    definition = _family((ScenarioFactor.EVENT_SOURCES,))

    with pytest.raises(ValueError, match="declared_but_unchanged"):
        resolve_scenario_family(
            definition,
            _family_requests(definition, event_has_source=False),
        )


def test_historical_replay_rejects_synthetic_weather() -> None:
    definition = replace(
        _family((ScenarioFactor.EVENT_SOURCES,)),
        purpose=ScenarioFamilyPurpose.HISTORICAL_REPLAY,
    )

    with pytest.raises(ValueError, match="requires historical weather"):
        resolve_scenario_family(
            definition,
            _family_requests(definition, event_has_source=True),
        )


def test_calibrated_provider_claim_waits_for_change_10_evidence_contract() -> None:
    definition = ScenarioFamilyDefinition(
        family_id="synthetic-event-family",
        purpose=ScenarioFamilyPurpose.STRESS_COUNTERFACTUAL,
        control_member_id="control",
        members=(
            ScenarioMemberDefinition(
                "control",
                claims_calibrated_providers=True,
                calibration_evidence_id="future-change-10-artifact",
            ),
            ScenarioMemberDefinition(
                "event",
                (ScenarioFactor.EVENT_SOURCES,),
            ),
        ),
    )

    with pytest.raises(ValueError, match="unavailable until change 10"):
        resolve_scenario_family(
            definition,
            _family_requests(definition, event_has_source=True),
        )


def test_grid_convergence_is_reported_only_for_consistent_refinement() -> None:
    convergent = assess_grid_convergence({50: 27, 100: 26, 200: 22})
    oscillatory = assess_grid_convergence({50: 24, 100: 29, 200: 22})
    growing = assess_grid_convergence({50: 31, 100: 26, 200: 22})

    assert convergent["status"] == "applicable"
    assert convergent["apparent_order"] == pytest.approx(2)
    assert oscillatory == {"status": "not_applicable", "reason": "oscillatory"}
    assert growing == {
        "status": "not_applicable",
        "reason": "differences_do_not_shrink",
    }


def test_horizontal_sensitivity_uses_common_coordinates_and_declared_rule() -> None:
    protocol = HorizontalSensitivityProtocol(
        cell_sizes_m=(50, 100, 200),
        comparison_time_s=120,
        probe_points_m=(("southwest", 25, 25),),
        relative_tolerance=1e-10,
        selection_quantities=("total_mass_ug",),
    )

    evidence = run_horizontal_sensitivity(
        box(0, 0, 200, 200),
        "EPSG:32718",
        protocol,
        lambda grid: _request(grid=grid),
    )

    assert evidence["protocol"]["cell_sizes_m"] == [50, 100, 200]
    assert evidence["experimental_recommendation_cell_size_m"] == 100
    assert evidence["recommendation_status"] == "technical_recommendation"
    assert set(evidence["results_by_cell_size_m"]) == {"50.0", "100.0", "200.0"}


def test_output_schedule_sensitivity_compares_shared_instants() -> None:
    evidence = compare_output_schedule_sensitivity(
        _request(),
        (35, 95),
        absolute_tolerance_ug_m3=1e-12,
    )

    assert evidence["baseline_output_times_s"] == [0, 60, 120]
    assert evidence["dense_output_times_s"] == [0, 35, 60, 95, 120]
    assert evidence["within_tolerance"]


def test_environment_bundle_is_portable_and_keeps_exact_station_position(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lima_pm25.simulation.study_evidence as evidence

    domain, config, catalog, graph_path, grid_path = _environment(tmp_path)
    family = resolve_scenario_family(
        config.scenario_families[0],
        _family_requests(config.scenario_families[0], event_has_source=True),
    )
    monkeypatch.setattr(
        evidence,
        "plot_spatial_domain",
        lambda _domain, path, *, station_catalog: path.write_bytes(b"map"),
    )

    bundle = write_environment_bundle(
        config,
        domain,
        catalog,
        (family,),
        tmp_path / "published" / "environment",
        road_graph_path=graph_path,
        grid_path=grid_path,
    )

    assert verify_environment_bundle(bundle).valid
    metadata = json.loads((bundle / "map_metadata.json").read_text(encoding="utf-8"))
    station = metadata["stations"][0]
    assert station["x_m"] == 25
    assert station["y_m"] == 75
    assert station["row"] == 0
    assert station["column"] == 0
    assert metadata["openstreetmap_license"] == "ODbL"
    assert (bundle / "spatial" / "road_graph.graphml").read_bytes() == b"graph"
    manifest = json.loads(
        (bundle / "environment_manifest.json").read_text(encoding="utf-8")
    )
    documentation = render_environment_markdown(manifest)
    assert manifest["fingerprint"] in documentation
    assert "vertically averaged" in documentation
    assert manifest["software"]["source_tree_sha256"] is not None
    assert (bundle / "scenario_documentation.md").is_file()

    graph_path.write_bytes(b"changed graph")
    changed_manifest = build_environment_manifest(
        config,
        domain,
        catalog,
        (family,),
        road_graph_path=graph_path,
        grid_path=grid_path,
    )
    assert changed_manifest["fingerprint"] != manifest["fingerprint"]

    graph_path.unlink()
    grid_path.unlink()
    assert verify_environment_bundle(bundle).valid

    (bundle / "map_metadata.json").write_text("{}", encoding="utf-8")
    changed = verify_environment_bundle(bundle)
    assert not changed.valid
    assert changed.failures == ("changed:map_metadata.json",)


def _family(
    event_factors: tuple[ScenarioFactor, ...],
) -> ScenarioFamilyDefinition:
    return ScenarioFamilyDefinition(
        family_id="synthetic-event-family",
        purpose=ScenarioFamilyPurpose.STRESS_COUNTERFACTUAL,
        control_member_id="control",
        members=(
            ScenarioMemberDefinition("control"),
            ScenarioMemberDefinition("event", event_factors),
        ),
    )


def _family_requests(
    definition: ScenarioFamilyDefinition,
    *,
    event_has_source: bool,
) -> dict[str, SimulationRunRequest]:
    return {
        "control": replace(
            _request(),
            scenario_identity=definition.run_identity("control"),
        ),
        "event": replace(
            _request(with_event=event_has_source),
            scenario_identity=definition.run_identity("event"),
        ),
    }


def _request(
    *,
    with_event: bool = False,
    grid=None,
) -> SimulationRunRequest:
    if grid is None:
        grid = build_grid(box(0, 0, 200, 200), "EPSG:32718", 100)
    model = PM25Transport(
        grid,
        PM25PhysicsConfig(
            mixing_height_m=100,
            diffusivity_m2_s=0,
            dry_deposition_velocity_m_s=0,
            wet_scavenging_per_mm=0,
        ),
    )
    initial = model.initialize_uniform(
        AmbientBaseline("synthetic-baseline", 5, "synthetic test baseline")
    )
    weather = SyntheticWeatherWindow(
        start_utc=START,
        duration_s=120,
        values=(
            ScheduledWeatherValue(
                0,
                AtmosphericForcing(
                    "synthetic-weather",
                    "synthetic study-evidence test",
                    0,
                    0,
                    0,
                ),
            ),
        ),
    )
    routine = GaussianKernelSource(
        source_id="routine",
        x_m=50,
        y_m=50,
        emission_rate_ug_s=1,
        sigma_m=50,
        ends_at_s=120,
        role=PM25SourceRole.ROUTINE_ENVIRONMENTAL,
        provenance="synthetic routine test source",
    )
    event = GaussianKernelSource(
        source_id="event",
        x_m=150,
        y_m=150,
        emission_rate_ug_s=1,
        sigma_m=50,
        starts_at_s=30,
        ends_at_s=60,
        role=PM25SourceRole.COUNTERFACTUAL_EVENT,
        provenance="synthetic counterfactual test source",
    )
    return SimulationRunRequest(
        model=model,
        initial_state=initial,
        weather=weather,
        config=SimulationRunConfig(
            start_utc=START,
            duration_s=120,
            concentration_output_interval_s=60,
            seed=1,
        ),
        sources=(routine, event) if with_event else (routine,),
    )


def _environment(
    tmp_path: Path,
) -> tuple[SpatialDomain, StudyEnvironmentConfig, FixedStationCatalog, Path, Path]:
    cache = tmp_path / "cache" / "domain"
    cache.mkdir(parents=True)
    domain_config = DomainConfig(
        left=-77.01,
        bottom=-12.01,
        right=-77,
        top=-12,
        cell_size_m=100,
        cache_directory=cache,
        osmnx_cache_directory=tmp_path / "cache" / "osmnx",
    )
    grid = build_grid(box(0, 0, 200, 200), "EPSG:32718", 100)
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:32718"
    graph.add_node(1, x=0.0, y=50.0)
    graph.add_node(2, x=200.0, y=50.0)
    graph.add_edge(1, 2, key=0, length=200.0)
    domain = SpatialDomain(
        config=domain_config,
        road_graph=graph,
        boundary=box(0, 0, 200, 200),
        grid=grid,
        graph_source="test",
        grid_source="test",
    )
    graph_path = cache / "road_graph.graphml"
    graph_path.write_bytes(b"graph")
    grid_path = cache / "grid.npz"
    save_grid(grid, grid_path, domain_config.fingerprint)
    family = _family((ScenarioFactor.EVENT_SOURCES,))
    config = StudyEnvironmentConfig(
        environment_id="test-environment",
        domain=domain_config,
        station_catalog_id="test-catalog",
        mobility_policy=MobilityPolicy(road_tolerance_m=5),
        vertical=EffectiveVerticalRepresentation(
            mixing_height_m=100,
            evidence_status=EvidenceStatus.SYNTHETIC_DEMO,
            provenance="synthetic test value",
        ),
        temporal=TemporalStudyPolicy(
            concentration_output_interval_s=60,
            mobile_sampling_interval_s=None,
        ),
        horizontal_candidate_cell_sizes_m=(50, 100, 200),
        selected_cell_size_status=(
            SelectionStatus.PROVISIONAL_IMPLEMENTATION_VALUE
        ),
        scenario_families=(family,),
    )
    catalog = FixedStationCatalog(
        (
            FixedStationSite(
                site_id="STATION",
                longitude_deg=-77,
                latitude_deg=-12,
                altitude_m=None,
                x_m=25,
                y_m=75,
                crs=grid.crs,
                provenance=SENAMHI_PM25_PROVENANCE,
            ),
        )
    )
    return domain, config, catalog, graph_path, grid_path
