from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from .orchestration import SimulationRunResult, _jsonable


@dataclass(frozen=True, slots=True)
class BundleVerificationResult:
    valid: bool
    failures: tuple[str, ...]


def write_run_bundle(
    result: SimulationRunResult,
    output_directory: Path | str,
) -> Path:
    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError(f"Run bundle destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-",
            dir=destination.parent,
        )
    )
    try:
        (temporary / "manifest.json").write_text(
            result.manifest_json,
            encoding="utf-8",
        )
        _write_concentration(result, temporary / "concentration.npz")
        _write_json_lines(
            temporary / "measurement_attempts.jsonl",
            result.measurement_attempts,
        )
        _write_json_lines(
            temporary / "measurement_batches.jsonl",
            (
                {
                    "simulation_time_s": batch.simulation_time_s,
                    "timestamp_utc": batch.timestamp_utc,
                    "measurement_keys": [
                        {
                            "sensor_id": record.sensor_id,
                            "timestamp_utc": record.timestamp_utc,
                        }
                        for record in batch.records
                    ],
                }
                for batch in result.measurement_batches
            ),
        )
        _write_json_lines(
            temporary / "diagnostics.jsonl",
            result.diagnostics,
        )
        checksums = {
            path.name: _file_sha256(path)
            for path in sorted(temporary.iterdir(), key=lambda path: path.name)
            if path.is_file()
        }
        (temporary / "checksums.json").write_text(
            json.dumps(
                {
                    "algorithm": "sha256",
                    "files": checksums,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def verify_run_bundle(bundle_directory: Path | str) -> BundleVerificationResult:
    directory = Path(bundle_directory)
    checksum_path = directory / "checksums.json"
    if not checksum_path.is_file():
        return BundleVerificationResult(False, ("missing:checksums.json",))
    try:
        payload = json.loads(checksum_path.read_text(encoding="utf-8"))
        if payload.get("algorithm") != "sha256":
            return BundleVerificationResult(False, ("unsupported-checksum-algorithm",))
        expected = payload["files"]
        if not isinstance(expected, dict):
            raise TypeError
    except json.JSONDecodeError, KeyError, TypeError:
        return BundleVerificationResult(False, ("invalid:checksums.json",))

    failures: list[str] = []
    for name, expected_digest in sorted(expected.items()):
        path = directory / name
        if not path.is_file():
            failures.append(f"missing:{name}")
        elif _file_sha256(path) != expected_digest:
            failures.append(f"changed:{name}")
    return BundleVerificationResult(not failures, tuple(failures))


def _write_concentration(result: SimulationRunResult, path: Path) -> None:
    if not result.frames:
        raise ValueError("A completed run must contain concentration frames")
    grid = result.final_state.grid
    np.savez_compressed(
        path,
        times_s=np.asarray(
            [frame.simulation_time_s for frame in result.frames],
            dtype=np.float64,
        ),
        timestamps_utc=np.asarray(
            [frame.timestamp_utc.isoformat() for frame in result.frames],
        ),
        concentration_ug_m3=np.stack(
            [frame.concentration_ug_m3 for frame in result.frames]
        ),
        x_centers_m=grid.x_centers_m,
        y_centers_m=grid.y_centers_m,
        active_mask=grid.active_mask,
        crs=np.asarray(grid.crs),
    )


def _write_json_lines(path: Path, values) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(
                json.dumps(
                    _jsonable(value),
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
