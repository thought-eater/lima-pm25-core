from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from lima_pm25.simulation import SimulationRunResult

from .reconstruction import ReconstructionConfig
from .retained import RetainedSyntheticEvaluation
from .figures import (map_coverage, render_concentration_series, render_replay,
                      render_mobile_uncertainty, render_temporal_aging, used_observations)


def write_retained_evaluation_bundle(
    *,
    evaluation: RetainedSyntheticEvaluation,
    run: SimulationRunResult,
    reconstruction_config: ReconstructionConfig,
    history_duration: timedelta,
    scale_ug_m3: float,
    destination: Path | str,
    representative_frame_indices: Sequence[int] = (),
) -> Path:
    output = Path(destination)
    if output.exists():
        raise FileExistsError(f"Retained evaluation bundle already exists: {output}")
    if evaluation.run_id != run.run_id:
        raise ValueError("Evaluation and simulation run identities must match")
    if history_duration <= timedelta(0):
        raise ValueError("history_duration must be positive")
    if not np.isfinite(scale_ug_m3) or scale_ug_m3 <= 0:
        raise ValueError("scale_ug_m3 must be positive and finite")
    indices = tuple(representative_frame_indices)
    if len(set(indices)) != len(indices) or any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(run.frames)
        for index in indices
    ):
        raise ValueError("representative_frame_indices must be unique valid indices")

    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary retained evaluation bundle already exists: {temporary}"
        )
    temporary.mkdir(parents=True)
    try:
        _write_json(
            temporary / "configuration.json",
            {
                "run_id": run.run_id,
                "history_duration_s": history_duration.total_seconds(),
                "scale_ug_m3": scale_ug_m3,
                "reconstruction_config": asdict(reconstruction_config),
                "representative_frame_indices": list(indices),
            },
        )
        _write_json(temporary / "metrics.json", _metrics_payload(evaluation))
        _write_json(
            temporary / "execution_log.json",
            {
                "run_id": run.run_id,
                "frame_count": len(run.frames),
                "measurement_attempt_count": len(run.measurement_attempts),
                "measurement_batch_count": len(run.measurement_batches),
                "fixed_only_map_count": len(evaluation.fixed_only_posteriors),
                "fixed_mobile_map_count": len(
                    evaluation.fixed_mobile_posteriors
                ),
                "mobile_uncertainty_comparison_count": len(
                    evaluation.mobile_uncertainty_changes
                ),
            },
        )
        np.savez_compressed(
            temporary / "maps.npz",
            timestamps_utc=np.asarray(
                [frame.timestamp_utc.isoformat() for frame in run.frames]
            ),
            active_mask=run.final_state.grid.active_mask,
            retained_truth_ug_m3=np.stack(
                [frame.concentration_ug_m3 for frame in run.frames]
            ),
            fixed_only_mean_ug_m3=np.stack(
                [posterior.mean_ug_m3 for posterior in evaluation.fixed_only_posteriors]
            ),
            fixed_only_stddev_ug_m3=np.stack(
                [
                    posterior.standard_deviation_ug_m3
                    for posterior in evaluation.fixed_only_posteriors
                ]
            ),
            fixed_mobile_mean_ug_m3=np.stack(
                [
                    posterior.mean_ug_m3
                    for posterior in evaluation.fixed_mobile_posteriors
                ]
            ),
            fixed_mobile_stddev_ug_m3=np.stack(
                [
                    posterior.standard_deviation_ug_m3
                    for posterior in evaluation.fixed_mobile_posteriors
                ]
            ),
        )
        export_evaluation_figures(
            temporary, grid=run.final_state.grid,
            measurements=run.measurement_attempts,
        )
        _write_checksums(temporary)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    except BaseException:
        _remove_tree(temporary)
        raise
    return output


def verify_retained_evaluation_bundle(path: Path | str) -> tuple[str, ...]:
    root = Path(path)
    manifest_path = root / "checksums.json"
    if not manifest_path.is_file():
        return ("missing:checksums.json",)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        if artifact.is_file() and artifact != manifest_path
    }
    extra = actual.difference(expected)
    failures.extend(f"unexpected:{relative}" for relative in sorted(extra))
    return tuple(failures)


def _metrics_payload(evaluation: RetainedSyntheticEvaluation) -> dict[str, object]:
    return {
        "run_id": evaluation.run_id,
        "fixed_only_r21": _r21_payload(evaluation.fixed_only_r21),
        "fixed_mobile_r21": _r21_payload(evaluation.fixed_mobile_r21),
        "fixed_mobile_r22": _r22_payload(evaluation.fixed_mobile_r22),
        "mobile_uncertainty_changes": [
            {
                "status": change.status.value,
                "reasons": list(change.reasons),
                "local_variance_before": list(change.local_variance_before),
                "local_variance_after": list(change.local_variance_after),
                "integrated_variance_before": (
                    change.integrated_variance_before
                ),
                "integrated_variance_after": change.integrated_variance_after,
            }
            for change in evaluation.mobile_uncertainty_changes
        ],
    }


def _r21_payload(value) -> dict[str, object]:
    payload = asdict(value)
    payload["status"] = value.status.value
    payload["reasons"] = list(value.reasons)
    return payload


def _r22_payload(value) -> dict[str, object]:
    payload = asdict(value)
    payload["status"] = value.status.value
    payload["reasons"] = list(value.reasons)
    return payload


def export_evaluation_figures(destination, *, grid, measurements):
    """Refresh derived presentation outputs from stored maps; never refit them."""
    root = Path(destination)
    configuration = json.loads((root / "configuration.json").read_text())
    config = ReconstructionConfig(**configuration["reconstruction_config"])
    with np.load(root / "maps.npz", allow_pickle=False) as data:
        if not np.array_equal(data["active_mask"], grid.active_mask):
            raise ValueError("Presentation grid does not match stored maps")
        timestamps = tuple(datetime.fromisoformat(str(t)) for t in data["timestamps_utc"])
        truth = data["retained_truth_ug_m3"]
        mean = data["fixed_mobile_mean_ug_m3"]
        stddev = data["fixed_mobile_stddev_ug_m3"]
        fixed_stddev = data["fixed_only_stddev_ug_m3"]
    observations = used_observations(grid, config, measurements, timestamps,
                                    timedelta(seconds=configuration["history_duration_s"]))
    indices = tuple(configuration["representative_frame_indices"])
    metrics = json.loads((root / "metrics.json").read_text())
    coverage = map_coverage(mean, stddev, grid.active_mask)
    p50, p99 = np.percentile(truth[:, grid.active_mask], [50, 99])
    coverage.update({"required_map_count": len(timestamps), "p50_ug_m3": float(p50),
                     "p99_ug_m3": float(p99), "contrast_ug_m3": float(p99-p50),
                     "contrast_fraction_of_historical_scale": float((p99-p50)/configuration["scale_ug_m3"]),
                     "timestamps_utc": [t.isoformat() for t in timestamps]})
    metrics["map_validation"] = coverage
    if indices:
        figures = root / "figures"
        render_concentration_series(grid, timestamps, truth, mean, observations, indices, figures)
        # Middle declared frame, not a frame chosen by its improvement score.
        index = indices[len(indices)//2]
        metrics["illustrated_mobile_uncertainty"] = render_mobile_uncertainty(
            grid, timestamps, fixed_stddev, stddev, observations, index,
            figures / "mobile-uncertainty.png",
        )
        metrics["temporal_aging"] = render_temporal_aging(
            grid, config, timestamps[0], figures / "temporal-aging.png",
        )
    _write_json(root / "metrics.json", metrics)
    _write_checksums(root)


def export_replay_figures(destination, *, grid, config, measurements, history_duration, sites=()):
    """Export historical maps and counts from the actual eligible input records."""
    root = Path(destination)
    with np.load(root / "posterior_maps.npz", allow_pickle=False) as data:
        if not np.array_equal(data["active_mask"], grid.active_mask):
            raise ValueError("Replay grid does not match stored maps")
        timestamps = tuple(datetime.fromisoformat(str(t)) for t in data["timestamps_utc"])
        mean, stddev = data["mean_ug_m3"], data["stddev_ug_m3"]
    observations = used_observations(grid, config, measurements, timestamps, history_duration)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    render_replay(grid, timestamps, mean, stddev, observations, figures / "replay-maps.png", sites)
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["map_validation"] = map_coverage(mean, stddev, grid.active_mask)
    used = {(r.sensor_id, r.timestamp_utc): r for group in observations for r in group}
    start = datetime.fromisoformat(metadata["window_start_utc"])
    end = datetime.fromisoformat(metadata["window_end_utc"])
    metadata["used_measurements"] = {
        period: {
            kind: sum(r.sensor_type.value == kind and predicate(r.timestamp_utc)
                      for r in used.values())
            for kind in ("fixed", "mobile")
        }
        for period, predicate in (
            ("previous_context", lambda t: t < start),
            ("window_inclusive_end", lambda t: start <= t <= end),
        )
    }
    metadata["used_measurements_by_frame"] = [
        {"timestamp_utc": t.isoformat(),
         "fixed": sum(r.sensor_type.value == "fixed" for r in group),
         "mobile": sum(r.sensor_type.value == "mobile" for r in group)}
        for t, group in zip(timestamps, observations, strict=True)
    ]
    _write_json(root / "metadata.json", metadata)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _write_checksums(root: Path) -> None:
    files = {
        artifact.relative_to(root).as_posix(): sha256(artifact.read_bytes()).hexdigest()
        for artifact in sorted(root.rglob("*"))
        if artifact.is_file() and artifact.name != "checksums.json"
    }
    _write_json(root / "checksums.json", {"files": files})


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for artifact in sorted(root.rglob("*"), reverse=True):
        if artifact.is_file():
            artifact.unlink()
        elif artifact.is_dir():
            artifact.rmdir()
    root.rmdir()
