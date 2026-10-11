from __future__ import annotations

import csv
import json
import platform
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from enum import Enum
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from lima_pm25.domain import GridDefinition
from lima_pm25.simulation import MeasurementBatch, SensorType

from .reconstruction import (
    ReconstructionRequest,
    select_reconstruction_observations,
)
from .sequential import (
    SequentialUpdateConfig,
    SequentialUpdateTrace,
    grid_sha256,
)

SEQUENTIAL_EVIDENCE_SCHEMA_VERSION = 1
R23_REQUIRED_CASE_SCHEMA_VERSION = 1


def write_sequential_update_bundle(
    *,
    trace: SequentialUpdateTrace,
    batches: Sequence[MeasurementBatch],
    config: SequentialUpdateConfig,
    grid: GridDefinition,
    destination: Path | str,
    upstream_manifest_json: str | None = None,
) -> Path:
    ordered_batches = tuple(batches)
    if len(trace.results) != len(ordered_batches):
        raise ValueError("trace results must correspond one-to-one with input batches")
    for batch, result in zip(ordered_batches, trace.results, strict=True):
        expected_keys = {
            (record.sensor_id, record.timestamp_utc) for record in batch.records
        }
        accounted_keys = set(result.log_entry.incorporated_measurement_keys).union(
            result.log_entry.policy_excluded_measurement_keys
        )
        if result.log_entry.processed_timestamp_utc != batch.timestamp_utc or (
            accounted_keys != expected_keys
        ):
            raise ValueError("trace accounting does not match its public input batch")
    snapshots = trace.snapshots
    if not snapshots:
        raise ValueError("A sequential evidence bundle requires at least one snapshot")
    expected_grid_sha256 = grid_sha256(grid)
    if any(snapshot.grid_sha256 != expected_grid_sha256 for snapshot in snapshots):
        raise ValueError("Every snapshot must use the evidence grid")

    upstream = _upstream_identity(upstream_manifest_json)
    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"Sequential update bundle already exists: {output}")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary sequential update bundle already exists: {temporary}"
        )
    temporary.mkdir(parents=True)
    try:
        manifest = {
            "schema_version": SEQUENTIAL_EVIDENCE_SCHEMA_VERSION,
            "purpose": "r23_sequential_map_update_evidence",
            "hidden_truth_included": False,
            "grid": {
                "sha256": expected_grid_sha256,
                "crs": grid.crs,
                "shape": list(grid.shape),
                "cell_size_m": grid.cell_size_m,
            },
            "configuration": _jsonable(config),
            "upstream_run": upstream,
            "software": _software_identity(),
            "input_batch_count": len(ordered_batches),
            "snapshot_count": len(snapshots),
        }
        _write_json(temporary / "manifest.json", manifest)
        _write_json_lines(
            temporary / "public_batches.jsonl",
            (
                {
                    "simulation_time_s": batch.simulation_time_s,
                    "timestamp_utc": batch.timestamp_utc,
                    "records": batch.records,
                }
                for batch in ordered_batches
            ),
        )
        _write_json_lines(
            temporary / "updates.jsonl",
            (
                {
                    "log_entry": result.log_entry,
                    "uncertainty_reduction": (
                        result.snapshot.uncertainty_reduction
                        if result.snapshot is not None
                        else None
                    ),
                }
                for result in trace.results
            ),
        )
        np.savez_compressed(
            temporary / "snapshots.npz",
            sequence_numbers=np.asarray(
                [snapshot.sequence_number for snapshot in snapshots],
                dtype=np.int64,
            ),
            reconstruction_timestamps_utc=np.asarray(
                [
                    snapshot.reconstruction_timestamp_utc.isoformat()
                    for snapshot in snapshots
                ]
            ),
            measurement_cutoffs_utc=np.asarray(
                [snapshot.measurement_cutoff_utc.isoformat() for snapshot in snapshots]
            ),
            mean_ug_m3=np.stack([snapshot.mean_ug_m3 for snapshot in snapshots]),
            standard_deviation_ug_m3=np.stack(
                [snapshot.standard_deviation_ug_m3 for snapshot in snapshots]
            ),
            active_mask=grid.active_mask,
            x_centers_m=grid.x_centers_m,
            y_centers_m=grid.y_centers_m,
            crs=np.asarray(grid.crs),
            grid_sha256=np.asarray(expected_grid_sha256),
        )
        _write_json(
            temporary / "verification_report.json",
            _runtime_verification_payload(
                trace,
                ordered_batches,
                config,
                grid,
            ),
        )
        export_sequential_table(temporary)
        _write_checksums(temporary)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    except BaseException:
        _remove_tree(temporary)
        raise
    return output


def export_sequential_table(destination: Path | str) -> Path:
    """Tabulate existing updates; reductions refer to the complete batch."""
    root = Path(destination)
    manifest = json.loads((root / "manifest.json").read_text())
    duration = timedelta(seconds=manifest["configuration"]["history_duration"])
    batches = [
        json.loads(line)
        for line in (root / "public_batches.jsonl").read_text().splitlines()
    ]
    updates = [
        json.loads(line) for line in (root / "updates.jsonl").read_text().splitlines()
    ]
    records = {
        (r["sensor_id"], r["timestamp_utc"]): r for b in batches for r in b["records"]
    }
    incorporated = set()
    rows = []
    for update in updates:
        log = update["log_entry"]
        keys = [tuple(key) for key in log["incorporated_measurement_keys"]]
        incorporated.update(keys)
        t = datetime.fromisoformat(log["processed_timestamp_utc"])
        history_count = sum(
            t - duration <= datetime.fromisoformat(key[1]) <= t for key in incorporated
        )
        diagnostic = update["uncertainty_reduction"]
        if diagnostic is None:
            continue
        for local in diagnostic["local"]:
            rows.append(
                {
                    "sequence": log["sequence_number"],
                    "timestamp_utc": t.isoformat(),
                    "new_fixed_count": sum(
                        records[key]["sensor_type"] == "fixed" for key in keys
                    ),
                    "new_mobile_count": sum(
                        records[key]["sensor_type"] == "mobile" for key in keys
                    ),
                    "history_count": history_count,
                    "sensor_id": local["measurement_key"][0],
                    "local_variance_before": local["variance_before_ug_m3_squared"],
                    "local_variance_after": local["variance_after_ug_m3_squared"],
                    "local_variance_reduction": local["reduction_ug_m3_squared"],
                    "batch_integrated_variance_before": diagnostic[
                        "integrated_variance_before_ug2_m4"
                    ],
                    "batch_integrated_variance_after": diagnostic[
                        "integrated_variance_after_ug2_m4"
                    ],
                    "batch_integrated_variance_reduction": diagnostic[
                        "integrated_reduction_ug2_m4"
                    ],
                }
            )
    if not rows:
        raise ValueError("No applied updates to tabulate")
    path = root / "update_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_checksums(root)
    return path


def verify_sequential_update_bundle(path: Path | str) -> tuple[str, ...]:
    root = Path(path)
    checksums_path = root / "checksums.json"
    if not checksums_path.is_file():
        return ("missing:checksums.json",)
    try:
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
        expected = checksums["files"]
        if not isinstance(expected, dict):
            raise TypeError
    except json.JSONDecodeError, KeyError, TypeError:
        return ("invalid:checksums.json",)

    failures = []
    for relative, expected_sha256 in sorted(expected.items()):
        artifact = root / relative
        if not artifact.is_file():
            failures.append(f"missing:{relative}")
        elif _file_sha256(artifact) != expected_sha256:
            failures.append(f"changed:{relative}")
    actual = {
        artifact.relative_to(root).as_posix()
        for artifact in root.rglob("*")
        if artifact.is_file() and artifact != checksums_path
    }
    failures.extend(
        f"unexpected:{relative}" for relative in sorted(actual.difference(expected))
    )
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != SEQUENTIAL_EVIDENCE_SCHEMA_VERSION:
                failures.append("invalid:manifest-schema-version")
            if manifest.get("hidden_truth_included") is not False:
                failures.append("invalid:hidden-truth-declaration")
        except json.JSONDecodeError:
            failures.append("invalid:manifest.json")
    return tuple(dict.fromkeys(failures))


def write_r23_required_case_report(
    *,
    required_case_manifest: Path | str,
    junit_xml: Path | str,
    runtime_verification_report: Path | str,
    destination: Path | str,
    configuration_sha256: str,
) -> Path:
    manifest_path = Path(required_case_manifest)
    junit_path = Path(junit_xml)
    runtime_path = Path(runtime_verification_report)
    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"R2.3 verification report already exists: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != R23_REQUIRED_CASE_SCHEMA_VERSION:
        raise ValueError("Unsupported R2.3 required-case schema_version")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("R2.3 required-case manifest must contain cases")
    identities = [
        (case.get("classname"), case.get("name"))
        for case in cases
        if isinstance(case, dict)
    ]
    if len(identities) != len(cases) or len(set(identities)) != len(identities):
        raise ValueError("R2.3 required cases must have unique test identities")

    outcomes = _junit_outcomes(junit_path)
    case_results = []
    counts = {"passed": 0, "failed": 0, "errored": 0, "skipped": 0, "missing": 0}
    for case in cases:
        identity = (case["classname"], case["name"])
        outcome = _required_case_outcome(outcomes, identity)
        counts[outcome] += 1
        case_results.append({**case, "outcome": outcome})
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_accepted = runtime.get("runtime_accepted") is True
    accepted = (
        counts["passed"] == len(cases)
        and all(
            counts[name] == 0 for name in ("failed", "errored", "skipped", "missing")
        )
        and runtime_accepted
    )
    report = {
        "schema_version": R23_REQUIRED_CASE_SCHEMA_VERSION,
        "accepted": accepted,
        "required_case_count": len(cases),
        **counts,
        "configuration_sha256": configuration_sha256,
        "runtime_accepted": runtime_accepted,
        "runtime_verification": runtime,
        "software": _software_identity(),
        "test_artifacts": {
            "required_case_manifest": str(manifest_path),
            "required_case_manifest_sha256": _file_sha256(manifest_path),
            "junit_xml": str(junit_path),
            "junit_xml_sha256": _file_sha256(junit_path),
            "runtime_verification_report": str(runtime_path),
            "runtime_verification_report_sha256": _file_sha256(runtime_path),
        },
        "cases": case_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    return output


def _runtime_verification_payload(
    trace: SequentialUpdateTrace,
    batches: tuple[MeasurementBatch, ...],
    config: SequentialUpdateConfig,
    grid: GridDefinition,
) -> dict[str, object]:
    submitted_mobile_keys = []
    mobile_entries = []
    for batch, result in zip(batches, trace.results, strict=True):
        eligible, _ = select_reconstruction_observations(
            ReconstructionRequest(
                grid=grid,
                reconstruction_timestamp_utc=batch.timestamp_utc,
                measurements=batch.records,
            ),
            config.reconstruction_config,
        )
        mobile_keys = tuple(
            (record.sensor_id, record.timestamp_utc)
            for record in eligible
            if record.sensor_type is SensorType.MOBILE
        )
        submitted_mobile_keys.extend(mobile_keys)
        if mobile_keys:
            mobile_entries.append(result)
    incorporated_mobile_keys = tuple(
        key for entry in trace.log for key in entry.eligible_mobile_measurement_keys
    )
    diagnostic_entries = tuple(
        result
        for result in mobile_entries
        if result.snapshot is not None
        and result.snapshot.uncertainty_reduction.has_mobile_measurement
    )
    nonincrease_entries = tuple(
        result
        for result in diagnostic_entries
        if result.snapshot is not None
        and result.snapshot.uncertainty_reduction.passes_nonincrease
    )
    positive_entries = tuple(
        result
        for result in diagnostic_entries
        if result.snapshot is not None
        and any(
            local.sensor_type.value == "mobile" and local.reduction_ug_m3_squared > 0
            for local in result.snapshot.uncertainty_reduction.local
        )
        and result.snapshot.uncertainty_reduction.integrated_reduction_ug2_m4 > 0
    )
    submitted_mobile_set = set(submitted_mobile_keys)
    incorporated_mobile_set = set(incorporated_mobile_keys)
    unique_incorporation = len(incorporated_mobile_keys) == len(incorporated_mobile_set)
    incorporation_coverage = (
        len(submitted_mobile_set.intersection(incorporated_mobile_set))
        / len(submitted_mobile_set)
        if submitted_mobile_set
        else None
    )
    diagnostic_coverage = (
        len(diagnostic_entries) / len(mobile_entries) if mobile_entries else None
    )
    runtime_accepted = bool(
        submitted_mobile_set
        and incorporation_coverage == 1.0
        and incorporated_mobile_set == submitted_mobile_set
        and unique_incorporation
        and diagnostic_coverage == 1.0
        and len(nonincrease_entries) == len(mobile_entries)
        and positive_entries
    )
    return {
        "runtime_accepted": runtime_accepted,
        "eligible_mobile_measurement_count": len(submitted_mobile_set),
        "incorporated_mobile_measurement_count": len(incorporated_mobile_keys),
        "unique_incorporated_mobile_measurement_count": len(incorporated_mobile_set),
        "mobile_measurements_incorporated_exactly_once": unique_incorporation,
        "eligible_mobile_incorporation_coverage": incorporation_coverage,
        "applied_mobile_batch_count": len(mobile_entries),
        "mobile_diagnostic_count": len(diagnostic_entries),
        "mobile_diagnostic_coverage": diagnostic_coverage,
        "mobile_nonincrease_pass_count": len(nonincrease_entries),
        "positive_mobile_reduction_batch_count": len(positive_entries),
    }


def _junit_outcomes(path: Path) -> dict[tuple[str, str], str]:
    root = ET.parse(path).getroot()
    outcomes: dict[tuple[str, str], str] = {}
    for testcase in root.iter("testcase"):
        identity = (
            testcase.attrib.get("classname", ""),
            testcase.attrib.get("name", ""),
        )
        if identity in outcomes:
            raise ValueError(f"Duplicate JUnit test identity {identity!r}")
        if testcase.find("failure") is not None:
            outcome = "failed"
        elif testcase.find("error") is not None:
            outcome = "errored"
        elif testcase.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        outcomes[identity] = outcome
    return outcomes


def _required_case_outcome(
    outcomes: Mapping[tuple[str, str], str],
    identity: tuple[str, str],
) -> str:
    exact = outcomes.get(identity)
    if exact is not None:
        return exact
    expected_classname, expected_name = identity
    expected_module = expected_classname.rsplit(".", 1)[-1]
    candidates = [
        outcome
        for (classname, name), outcome in outcomes.items()
        if name == expected_name and classname.rsplit(".", 1)[-1] == expected_module
    ]
    return candidates[0] if len(candidates) == 1 else "missing"


def _upstream_identity(manifest_json: str | None) -> dict[str, object] | None:
    if manifest_json is None:
        return None
    payload = json.loads(manifest_json)
    run_config = payload.get("run_config")
    return {
        "run_id": payload.get("run_id"),
        "seed": run_config.get("seed") if isinstance(run_config, dict) else None,
        "manifest_sha256": sha256(manifest_json.encode("utf-8")).hexdigest(),
    }


def _software_identity() -> dict[str, str]:
    try:
        package_version = version("lima-pm25-core")
    except PackageNotFoundError:
        package_version = "unknown"
    try:
        sklearn_version = version("scikit-learn")
    except PackageNotFoundError:
        sklearn_version = "unknown"
    return {
        "package": "lima-pm25-core",
        "package_version": package_version,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn_version,
    }


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    total_seconds = getattr(value, "total_seconds", None)
    if callable(total_seconds):
        return total_seconds()
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        f"{json.dumps(_jsonable(payload), indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _write_json_lines(path: Path, values) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(
                json.dumps(_jsonable(value), separators=(",", ":"), sort_keys=True)
                + "\n"
            )


def _write_checksums(root: Path) -> None:
    _write_json(
        root / "checksums.json",
        {
            "algorithm": "sha256",
            "files": {
                artifact.relative_to(root).as_posix(): _file_sha256(artifact)
                for artifact in sorted(root.rglob("*"))
                if artifact.is_file() and artifact.name != "checksums.json"
            },
        },
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for artifact in sorted(root.rglob("*"), reverse=True):
        if artifact.is_file():
            artifact.unlink()
        elif artifact.is_dir():
            artifact.rmdir()
    root.rmdir()
