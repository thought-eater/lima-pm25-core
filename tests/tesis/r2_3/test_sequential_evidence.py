from datetime import UTC, datetime, timedelta
import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
from shapely.geometry import LineString, box

from lima_pm25.domain import (
    AmbientBaseline,
    AtmosphericForcing,
    DomainConfig,
    FixedStationCatalog,
    FixedStationSite,
    GridDefinition,
    MobilityPolicy,
    SENAMHI_PM25_PROVENANCE,
    SpatialDomain,
)
from lima_pm25.estimation import (
    ReconstructionConfig,
    SequentialMapUpdater,
    SequentialUpdateConfig,
    process_measurement_batches,
    verify_sequential_update_bundle,
    write_r23_required_case_report,
    write_sequential_update_bundle,
)
from lima_pm25.simulation import (
    PM25PhysicsConfig,
    PM25Transport,
    ScheduledWeatherValue,
    SensorConfig,
    SensorType,
    SimulationRunConfig,
    SimulationRunRequest,
    SyntheticWeatherWindow,
    TimestampedTrajectoryProvider,
    TrajectoryPoint,
    create_lima_fixed_sensor_configs,
    run_simulation,
)


START = datetime(2026, 9, 28, tzinfo=UTC)


def test_r13_batches_drive_r23_and_persist_truth_isolated_evidence(
    tmp_path: Path,
) -> None:
    run = run_simulation(_simulation_request())
    config = _config()
    trace = process_measurement_batches(
        SequentialMapUpdater(run.final_state.grid, config),
        run.measurement_batches,
    )

    bundle = write_sequential_update_bundle(
        trace=trace,
        batches=run.measurement_batches,
        config=config,
        grid=run.final_state.grid,
        destination=tmp_path / "r23",
        upstream_manifest_json=run.manifest_json,
    )

    assert verify_sequential_update_bundle(bundle) == ()
    assert len(trace.snapshots) == len(run.measurement_batches) == 3
    assert all(
        result.log_entry.eligible_mobile_measurement_keys
        for result in trace.results
    )
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["hidden_truth_included"] is False
    assert manifest["upstream_run"]["run_id"] == run.run_id
    assert manifest["upstream_run"]["seed"] == 42
    public_inputs = (bundle / "public_batches.jsonl").read_text(encoding="utf-8")
    assert "sampled_truth" not in public_inputs
    assert "pre_bound" not in public_inputs
    with np.load(bundle / "snapshots.npz", allow_pickle=False) as payload:
        assert "truth_ug_m3" not in payload.files
        assert payload["mean_ug_m3"].shape == (3, 2, 2)
        assert payload["standard_deviation_ug_m3"].shape == (3, 2, 2)
    runtime = json.loads(
        (bundle / "verification_report.json").read_text(encoding="utf-8")
    )
    assert runtime["runtime_accepted"] is True
    assert runtime["eligible_mobile_incorporation_coverage"] == 1
    assert runtime["mobile_diagnostic_coverage"] == 1
    with (bundle / "update_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        snapshot = trace.snapshots[int(row["sequence"])-1]
        assert int(row["history_count"]) == snapshot.observations.used_record_count
        assert float(row["batch_integrated_variance_reduction"]) == snapshot.uncertainty_reduction.integrated_reduction_ug2_m4


def test_sequential_bundle_is_non_overwriting_and_tamper_evident(
    tmp_path: Path,
) -> None:
    run = run_simulation(_simulation_request())
    config = _config()
    trace = process_measurement_batches(
        SequentialMapUpdater(run.final_state.grid, config),
        run.measurement_batches,
    )
    bundle = write_sequential_update_bundle(
        trace=trace,
        batches=run.measurement_batches,
        config=config,
        grid=run.final_state.grid,
        destination=tmp_path / "r23",
        upstream_manifest_json=run.manifest_json,
    )

    try:
        write_sequential_update_bundle(
            trace=trace,
            batches=run.measurement_batches,
            config=config,
            grid=run.final_state.grid,
            destination=bundle,
            upstream_manifest_json=run.manifest_json,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("The bundle writer must not overwrite evidence")

    updates = bundle / "updates.jsonl"
    updates.write_text(updates.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    assert verify_sequential_update_bundle(bundle) == ("changed:updates.jsonl",)


def test_required_case_report_accepts_only_complete_passing_manifest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "required.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "functional-a",
                        "kind": "functional",
                        "classname": "tests.tesis.r2_3.test_sequential",
                        "name": "test_a",
                    },
                    {
                        "id": "integration-b",
                        "kind": "integration",
                        "classname": "tests.tesis.r2_3.test_sequential_evidence",
                        "name": "test_b",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite tests="2">
<testcase classname="tests.tesis.r2_3.test_sequential" name="test_a" />
<testcase classname="tests.tesis.r2_3.test_sequential_evidence" name="test_b" />
</testsuite></testsuites>
""",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"runtime_accepted": True}), encoding="utf-8")

    report_path = write_r23_required_case_report(
        required_case_manifest=manifest,
        junit_xml=junit,
        runtime_verification_report=runtime,
        destination=tmp_path / "report.json",
        configuration_sha256="configuration-sha256",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["accepted"] is True
    assert report["passed"] == report["required_case_count"] == 2
    assert report["failed"] == report["errored"] == 0
    assert report["skipped"] == report["missing"] == 0


def test_required_case_report_rejects_skipped_and_missing_cases(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "required.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "skipped",
                        "kind": "functional",
                        "classname": "tests.example",
                        "name": "test_skipped",
                    },
                    {
                        "id": "missing",
                        "kind": "integration",
                        "classname": "tests.example",
                        "name": "test_missing",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite tests="1">
<testcase classname="tests.example" name="test_skipped"><skipped /></testcase>
</testsuite></testsuites>
""",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"runtime_accepted": True}), encoding="utf-8")

    report_path = write_r23_required_case_report(
        required_case_manifest=manifest,
        junit_xml=junit,
        runtime_verification_report=runtime,
        destination=tmp_path / "report.json",
        configuration_sha256="configuration-sha256",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["accepted"] is False
    assert report["skipped"] == 1
    assert report["missing"] == 1


def _config() -> SequentialUpdateConfig:
    return SequentialUpdateConfig(
        reconstruction_config=ReconstructionConfig(
            prior_mean_ug_m3=20,
            prior_stddev_ug_m3=10,
            x_length_scale_m=250,
            y_length_scale_m=250,
            temporal_length_scale_s=3_600,
        ),
        history_duration=timedelta(hours=6),
        variance_numerical_tolerance_ug_m3_squared=1e-9,
        profile_sha256="fixture-profile-sha256",
        profile_status="synthetic_test",
        profile_source="tests/tesis/r2_3/test_sequential_evidence.py",
        calibration_artifact_sha256="fixture-calibration-sha256",
        calibration_artifact_source="synthetic test calibration",
        prediction_chunk_size=4,
    )


def _simulation_request() -> SimulationRunRequest:
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
        AmbientBaseline("baseline", 5, "synthetic sequential integration fixture")
    )
    weather = SyntheticWeatherWindow(
        start_utc=START,
        duration_s=120,
        values=(
            ScheduledWeatherValue(
                0,
                AtmosphericForcing(
                    "synthetic-weather",
                    "synthetic sequential integration fixture",
                    0,
                    0,
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
            TrajectoryPoint("MOBILE", time_s, 50 + time_s * 5 / 6, 50)
            for time_s in (0, 60, 120)
        ),
    )
    mobile = SensorConfig(
        sensor_id="MOBILE",
        sensor_type=SensorType.MOBILE,
        sampling_interval_s=60,
        first_due_time_s=0,
        mobile_position_provider_id=trajectory.provider_id,
        provenance="mobile sequential integration fixture",
    )
    return SimulationRunRequest(
        model=model,
        initial_state=initial,
        weather=weather,
        config=SimulationRunConfig(
            start_utc=START,
            duration_s=120,
            concentration_output_interval_s=60,
            seed=42,
        ),
        sensor_configs=(*create_lima_fixed_sensor_configs(catalog), mobile),
        station_catalog=catalog,
        mobile_position_providers=(trajectory,),
        sources=(),
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
