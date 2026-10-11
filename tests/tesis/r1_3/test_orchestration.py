import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, box

from lima_pm25.domain import (
    SENAMHI_PM25_PROVENANCE,
    AmbientBaseline,
    AtmosphericForcing,
    DomainConfig,
    FixedStationCatalog,
    FixedStationSite,
    GridDefinition,
    MobilityPolicy,
    PM25SourceRole,
    RegionalBoundaryForcing,
    SpatialDomain,
)
from lima_pm25.simulation import (
    BoundaryMode,
    BundleVerificationResult,
    EpisodeClock,
    GaussianKernelSource,
    HistoricalWeatherDataset,
    MeasurementQuality,
    PM25PhysicsConfig,
    PM25Transport,
    PrecipitationSourcePolicy,
    RegionalBoundarySchedule,
    RoadLineSource,
    ScheduledBoundaryValue,
    ScheduledWeatherValue,
    SensorConfig,
    SensorType,
    SimulationRunConfig,
    SimulationRunRequest,
    SyntheticWeatherWindow,
    TimestampedTrajectoryProvider,
    TrajectoryCoverageError,
    TrajectoryPoint,
    WeatherRecord,
    WeatherValueProvenance,
    run_complete_synthetic_demo,
    run_simulation,
    verify_run_bundle,
    write_run_bundle,
)

START = datetime(2026, 9, 25, 5, tzinfo=UTC)


def test_episode_clock_maps_utc_and_elapsed_seconds_unambiguously() -> None:
    clock = EpisodeClock(START)

    assert clock.utc_at(90) == datetime(2026, 9, 25, 5, 1, 30, tzinfo=UTC)
    assert clock.elapsed_at(clock.utc_at(90)) == 90
    with pytest.raises(ValueError, match="nonnegative"):
        clock.utc_at(-1)
    with pytest.raises(ValueError, match="timezone-aware"):
        EpisodeClock(datetime(2026, 9, 25, 5))


def test_complete_run_orders_events_batches_and_source_intervals() -> None:
    request = _request()

    result = run_simulation(request)

    assert [frame.simulation_time_s for frame in result.frames] == [0, 70, 120]
    assert [batch.simulation_time_s for batch in result.measurement_batches] == [
        0,
        60,
        120,
    ]
    assert all(
        [record.sensor_id for record in batch.records] == ["FIXED", "MOBILE"]
        for batch in result.measurement_batches
    )
    assert [record.value_ug_m3 for record in result.measurement_batches[0].records] == [
        5,
        5,
    ]
    assert len(result.measurement_attempts) == 6
    assert all(
        attempt.quality_status is MeasurementQuality.ACCEPTED
        for attempt in result.measurement_attempts
    )
    mobile_attempts = tuple(
        attempt
        for attempt in result.measurement_attempts
        if attempt.sensor_type is SensorType.MOBILE
    )
    assert [attempt.x_m for attempt in mobile_attempts] == [25, 75, 125]
    assert all(
        attempt.position_source_id == "mobile-replay" for attempt in mobile_attempts
    )

    assert [diagnostic.end_time_s for diagnostic in result.diagnostics] == [
        40,
        50,
        60,
        70,
        100,
        120,
    ]
    assert (
        sum(
            diagnostic.budget.emitted_mass_ug
            for diagnostic in result.diagnostics
            if diagnostic.end_time_s <= 50
        )
        == 0
    )
    assert (
        sum(
            diagnostic.budget.emitted_mass_ug
            for diagnostic in result.diagnostics
            if diagnostic.start_time_s >= 100
        )
        == 0
    )
    assert sum(
        diagnostic.budget.emitted_mass_ug for diagnostic in result.diagnostics
    ) == pytest.approx(50_000)
    assert result.final_state.simulation_time_s == 120

    manifest = json.loads(result.manifest_json)
    assert manifest["weather"]["kind"] == "synthetic"
    assert manifest["weather"]["precipitation_source_policy"] is None
    assert manifest["run_id"] == result.run_id


def test_historical_lineage_and_explicit_ifs_policy_reach_diagnostics() -> None:
    ifs = WeatherValueProvenance("ecmwf_ifs")
    observed = WeatherValueProvenance("senamhi_observed")
    records = tuple(
        WeatherRecord(
            station_id="A",
            timestamp_utc=START + timedelta(seconds=second),
            temperature_c=None,
            relative_humidity_pct=None,
            eastward_wind_m_s=0,
            northward_wind_m_s=0,
            precipitation_mm_h=0.4,
            temperature_provenance=WeatherValueProvenance("missing_optional"),
            relative_humidity_provenance=WeatherValueProvenance("missing_optional"),
            precipitation_provenance=ifs,
            wind_direction_provenance=observed,
            wind_speed_provenance=observed,
        )
        for second in (0, 120)
    )
    dataset = HistoricalWeatherDataset(
        station_id="A",
        provider_id="weather:A",
        provenance="historical orchestration fixture",
        records=records,
        resource_id="weather.csv",
        resource_sha256="abc123",
        precipitation_source_policy=PrecipitationSourcePolicy.STATION_THEN_IFS,
    )
    request = replace(
        _request(),
        weather=dataset.select_window(START, timedelta(seconds=120)),
    )

    result = run_simulation(request)

    assert result.diagnostics[0].weather.kind == "historical"
    assert result.diagnostics[0].weather.temperature_c is None
    assert result.diagnostics[0].weather.precipitation_provenance == ifs
    assert (
        result.diagnostics[0].weather.precipitation_source_policy
        is PrecipitationSourcePolicy.STATION_THEN_IFS
    )
    manifest = json.loads(result.manifest_json)
    assert manifest["weather"]["resource_id"] == "weather.csv"
    assert manifest["weather"]["precipitation_source_policy"] == "station_then_ifs"


def test_regional_boundary_transition_splits_the_complete_engine() -> None:
    request = _request()
    boundary = RegionalBoundarySchedule(
        (
            ScheduledBoundaryValue(
                0,
                RegionalBoundaryForcing("regional", 5, "test boundary"),
            ),
            ScheduledBoundaryValue(
                30,
                RegionalBoundaryForcing("regional", 7, "test boundary"),
            ),
        )
    )
    request = replace(
        request,
        config=replace(request.config, boundary_mode=BoundaryMode.REGIONAL_INFLOW),
        regional_boundary=boundary,
    )

    result = run_simulation(request)

    assert result.diagnostics[0].end_time_s == 30
    assert result.diagnostics[0].regional_boundary is not None
    assert result.diagnostics[1].regional_boundary is not None
    assert result.diagnostics[0].regional_boundary.concentration_ug_m3 == 5
    assert result.diagnostics[1].regional_boundary.concentration_ug_m3 == 7


def test_empty_source_run_reports_zero_emitted_mass() -> None:
    result = run_simulation(replace(_request(), sources=()))

    assert all(
        diagnostic.budget.emitted_mass_ug == 0 for diagnostic in result.diagnostics
    )


def test_replay_coverage_is_rejected_before_transport_advances() -> None:
    request = _request(trajectory_times=(0, 60))

    with pytest.raises(TrajectoryCoverageError, match="120"):
        run_simulation(request)

    assert request.initial_state.simulation_time_s == 0
    assert request.initial_state.concentration_ug_m3.flags.writeable is False


def test_same_configuration_and_seed_replay_deterministically() -> None:
    first = run_simulation(_request(mobile_noise=2.0))
    second = run_simulation(_request(mobile_noise=2.0))

    assert first.run_id == second.run_id
    assert first.measurement_attempts == second.measurement_attempts
    assert first.measurement_batches == second.measurement_batches
    assert all(
        np.array_equal(left.concentration_ug_m3, right.concentration_ug_m3)
        for left, right in zip(first.frames, second.frames, strict=True)
    )


def test_explicit_irregular_output_times_merge_with_regular_interval() -> None:
    request = _request()
    request = replace(
        request,
        config=replace(
            request.config,
            concentration_output_interval_s=70,
            concentration_output_times_s=(35, 70, 119),
        ),
    )

    result = run_simulation(request)

    assert [frame.simulation_time_s for frame in result.frames] == [
        0,
        35,
        70,
        119,
        120,
    ]
    manifest = json.loads(result.manifest_json)
    assert manifest["run_config"]["concentration_output_times_s"] == [35, 70, 119]


def test_explicit_outputs_can_be_used_without_a_regular_interval() -> None:
    request = _request()
    request = replace(
        request,
        config=replace(
            request.config,
            concentration_output_interval_s=None,
            concentration_output_times_s=(0, 35, 119),
        ),
    )

    result = run_simulation(request)

    assert [frame.simulation_time_s for frame in result.frames] == [0, 35, 119, 120]


def test_explicit_output_times_are_ordered_and_within_episode() -> None:
    config = _request().config

    with pytest.raises(ValueError, match="strictly increasing"):
        replace(config, concentration_output_times_s=(30, 30))
    with pytest.raises(ValueError, match="cannot exceed duration_s"):
        replace(config, concentration_output_times_s=(121,))
    with pytest.raises(ValueError, match="require an interval or explicit times"):
        replace(
            config,
            concentration_output_interval_s=None,
            concentration_output_times_s=(),
        )


def test_bundle_is_complete_non_overwriting_and_tamper_evident(tmp_path) -> None:
    result = run_simulation(_request(mobile_noise=1.0))
    bundle = write_run_bundle(result, tmp_path / "run")

    verification = verify_run_bundle(bundle)
    assert verification == BundleVerificationResult(True, ())
    assert {path.name for path in bundle.iterdir()} == {
        "checksums.json",
        "concentration.npz",
        "diagnostics.jsonl",
        "manifest.json",
        "measurement_attempts.jsonl",
        "measurement_batches.jsonl",
    }
    with np.load(bundle / "concentration.npz", allow_pickle=False) as payload:
        assert payload["concentration_ug_m3"].shape == (3, 2, 2)
        assert payload["times_s"].tolist() == [0, 70, 120]
        assert payload["active_mask"].all()

    attempts_text = (bundle / "measurement_attempts.jsonl").read_text(encoding="utf-8")
    assert "sampled_truth" not in attempts_text
    assert "pre_bound" not in attempts_text
    assert '"position_source_id":"mobile-replay"' in attempts_text

    with pytest.raises(FileExistsError):
        write_run_bundle(result, bundle)

    (bundle / "measurement_attempts.jsonl").write_text(
        attempts_text + "{}\n",
        encoding="utf-8",
    )
    changed = verify_run_bundle(bundle)
    assert not changed.valid
    assert changed.failures == ("changed:measurement_attempts.jsonl",)


def test_documented_complete_example_records_field_and_mobile_positions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lima_pm25.simulation import demo

    domain = _domain()
    model = PM25Transport(
        domain.grid,
        PM25PhysicsConfig(
            mixing_height_m=100,
            diffusivity_m2_s=0,
            dry_deposition_velocity_m_s=0,
            wet_scavenging_per_mm=0,
        ),
    )
    initial = model.initialize_uniform(
        AmbientBaseline("demo-baseline", 5, "synthetic complete demo test")
    )
    line = LineString([(0, 50), (200, 50)])
    source = RoadLineSource(
        source_id="demo-road",
        line=line,
        emission_rate_ug_m_s=1,
        provenance="synthetic complete demo test",
        ends_at_s=10_800,
        role=PM25SourceRole.SYNTHETIC_VALIDATION,
    )
    monkeypatch.setattr(
        demo,
        "_create_synthetic_demo",
        lambda _domain: (model, initial, (source,)),
    )
    catalog = FixedStationCatalog(
        (
            FixedStationSite(
                site_id="STATION",
                longitude_deg=-77,
                latitude_deg=-12,
                altitude_m=None,
                x_m=50,
                y_m=50,
                crs=domain.grid.crs,
                provenance=SENAMHI_PM25_PROVENANCE,
            ),
        )
    )

    result, bundle = run_complete_synthetic_demo(
        domain,
        tmp_path / "complete",
        station_catalog=catalog,
    )

    assert [frame.simulation_time_s for frame in result.frames] == list(
        range(0, 10_801, 300)
    )
    mobile_attempts = tuple(
        attempt
        for attempt in result.measurement_attempts
        if attempt.sensor_id == "MOBILE_DEMO"
    )
    assert len(mobile_attempts) == 37
    assert [attempt.simulation_time_s for attempt in mobile_attempts] == list(
        range(0, 10_801, 300)
    )
    assert verify_run_bundle(bundle).valid

    repeated, repeated_bundle = run_complete_synthetic_demo(
        domain,
        tmp_path / "complete",
        station_catalog=catalog,
    )
    assert repeated.run_id == result.run_id
    assert repeated_bundle == bundle
    assert repeated.measurement_batches == result.measurement_batches
    assert all(
        np.array_equal(left.concentration_ug_m3, right.concentration_ug_m3)
        for left, right in zip(repeated.frames, result.frames, strict=True)
    )


def _request(
    *,
    trajectory_times: tuple[int, ...] = (0, 60, 120),
    mobile_noise: float = 0.0,
) -> SimulationRunRequest:
    domain = _domain()
    model = PM25Transport(
        domain.grid,
        PM25PhysicsConfig(
            mixing_height_m=100,
            diffusivity_m2_s=0,
            dry_deposition_velocity_m_s=0,
            wet_scavenging_per_mm=0,
        ),
    )
    initial = model.initialize_uniform(
        AmbientBaseline("baseline", 5, "synthetic orchestration fixture")
    )
    weather = SyntheticWeatherWindow(
        start_utc=START,
        duration_s=120,
        values=(
            ScheduledWeatherValue(
                0,
                AtmosphericForcing(
                    "synthetic-weather",
                    "synthetic orchestration fixture",
                    0,
                    0,
                    0,
                ),
            ),
            ScheduledWeatherValue(
                40,
                AtmosphericForcing(
                    "synthetic-weather",
                    "synthetic orchestration fixture",
                    0,
                    90,
                    0,
                ),
            ),
        ),
    )
    catalog = FixedStationCatalog(
        (
            FixedStationSite(
                site_id="STATION",
                longitude_deg=-77,
                latitude_deg=-12,
                altitude_m=None,
                x_m=50,
                y_m=50,
                crs=domain.grid.crs,
                provenance=SENAMHI_PM25_PROVENANCE,
            ),
        )
    )
    trajectory = TimestampedTrajectoryProvider(
        provider_id="mobile-replay",
        domain=domain,
        mobility_policy=MobilityPolicy(road_tolerance_m=1),
        points=tuple(
            TrajectoryPoint("MOBILE", time_s, 25 + time_s * 5 / 6, 50)
            for time_s in trajectory_times
        ),
    )
    sensors = (
        SensorConfig(
            sensor_id="FIXED",
            sensor_type=SensorType.FIXED,
            sampling_interval_s=60,
            first_due_time_s=0,
            station_site_id="STATION",
            provenance="fixed orchestration fixture",
        ),
        SensorConfig(
            sensor_id="MOBILE",
            sensor_type=SensorType.MOBILE,
            sampling_interval_s=60,
            first_due_time_s=0,
            mobile_position_provider_id=trajectory.provider_id,
            provenance="mobile orchestration fixture",
            noise_stddev_ug_m3=mobile_noise,
        ),
    )
    source = GaussianKernelSource(
        source_id="scheduled-source",
        x_m=150,
        y_m=150,
        emission_rate_ug_s=1_000,
        sigma_m=50,
        starts_at_s=50,
        ends_at_s=100,
        provenance="synthetic orchestration fixture",
    )
    return SimulationRunRequest(
        model=model,
        initial_state=initial,
        weather=weather,
        config=SimulationRunConfig(
            start_utc=START,
            duration_s=120,
            concentration_output_interval_s=70,
            seed=42,
        ),
        sensor_configs=sensors,
        station_catalog=catalog,
        mobile_position_providers=(trajectory,),
        sources=(source,),
    )


def _domain() -> SpatialDomain:
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:32718"
    graph.add_node(1, x=0.0, y=50.0)
    graph.add_node(2, x=200.0, y=50.0)
    graph.add_edge(
        1,
        2,
        key=0,
        length=200.0,
        geometry=LineString([(0, 50), (200, 50)]),
    )
    grid = GridDefinition(
        crs="EPSG:32718",
        domain_bounds_m=np.array([0, 0, 200, 200]),
        grid_bounds_m=np.array([0, 0, 200, 200]),
        cell_size_m=100,
        x_centers_m=np.array([50, 150]),
        y_centers_m=np.array([50, 150]),
        active_mask=np.ones((2, 2), dtype=bool),
    )
    return SpatialDomain(
        config=DomainConfig(
            left=-77.01,
            bottom=-12.01,
            right=-77,
            top=-12,
            road_network_buffer_m=50,
        ),
        road_graph=graph,
        boundary=box(0, 0, 200, 200),
        grid=grid,
        graph_source="test",
        grid_source="test",
    )
