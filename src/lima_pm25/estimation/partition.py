from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

from lima_pm25.simulation import HistoricalWeatherDataset, WeatherCoverageError

from .historical import HistoricalPM25Dataset


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class HistoricalPartition(StrEnum):
    CALIBRATION = "calibration"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoricalPartitionConfig:
    calibration_fraction: float = 0.70
    calibration_window_count: int = 8
    replay_window_count: int = 4
    window_duration: timedelta = timedelta(hours=24)
    context_duration: timedelta = timedelta(hours=24)
    minimum_observed_station_count: int = 3

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.calibration_fraction)
            or not 0 < self.calibration_fraction < 1
        ):
            raise ValueError("calibration_fraction must be between 0 and 1")
        _require_positive_integer(
            "calibration_window_count",
            self.calibration_window_count,
        )
        _require_positive_integer("replay_window_count", self.replay_window_count)
        if self.window_duration <= timedelta(0):
            raise ValueError("window_duration must be positive")
        if self.context_duration < timedelta(0):
            raise ValueError("context_duration must be nonnegative")
        if self.window_duration % timedelta(hours=1) != timedelta(0):
            raise ValueError("window_duration must contain complete hours")
        if self.context_duration % timedelta(hours=1) != timedelta(0):
            raise ValueError("context_duration must contain complete hours")
        _require_positive_integer(
            "minimum_observed_station_count",
            self.minimum_observed_station_count,
        )


DEFAULT_HISTORICAL_PARTITION_CONFIG = HistoricalPartitionConfig()


@dataclass(frozen=True, slots=True)
class HistoricalWindow:
    partition: HistoricalPartition
    context_start_utc: datetime
    start_utc: datetime
    end_utc: datetime

    @property
    def duration(self) -> timedelta:
        return self.end_utc - self.start_utc

    @property
    def context_duration(self) -> timedelta:
        return self.start_utc - self.context_start_utc


@dataclass(frozen=True, slots=True)
class HistoricalPartitionManifest:
    pm25_resource_id: str
    pm25_resource_sha256: str
    weather_resource_id: str
    weather_resource_sha256: str
    weather_station_id: str
    station_ids: tuple[str, ...]
    station_observed_counts: tuple[tuple[str, int], ...]
    calibration_fraction: float
    calibration_start_utc: datetime
    calibration_end_utc: datetime
    replay_start_utc: datetime
    replay_end_utc: datetime
    minimum_observed_station_count: int
    eligible_hour_count: int
    selection_method: str
    pm25_observation_policy: str
    calibration_windows: tuple[HistoricalWindow, ...]
    replay_windows: tuple[HistoricalWindow, ...]

    def evidence_descriptor(self) -> dict[str, object]:
        return {
            "pm25_resource_id": self.pm25_resource_id,
            "pm25_resource_sha256": self.pm25_resource_sha256,
            "weather_resource_id": self.weather_resource_id,
            "weather_resource_sha256": self.weather_resource_sha256,
            "weather_station_id": self.weather_station_id,
            "station_ids": list(self.station_ids),
            "station_observed_counts": dict(self.station_observed_counts),
            "calibration_fraction": self.calibration_fraction,
            "calibration_start_utc": self.calibration_start_utc.isoformat(),
            "calibration_end_utc": self.calibration_end_utc.isoformat(),
            "replay_start_utc": self.replay_start_utc.isoformat(),
            "replay_end_utc": self.replay_end_utc.isoformat(),
            "minimum_observed_station_count": self.minimum_observed_station_count,
            "eligible_hour_count": self.eligible_hour_count,
            "selection_method": self.selection_method,
            "pm25_observation_policy": self.pm25_observation_policy,
            "calibration_windows": [
                _window_descriptor(window) for window in self.calibration_windows
            ],
            "replay_windows": [
                _window_descriptor(window) for window in self.replay_windows
            ],
        }


def build_historical_partition_manifest(
    *,
    pm25: HistoricalPM25Dataset,
    weather: HistoricalWeatherDataset,
    station_ids: Sequence[str],
    config: HistoricalPartitionConfig = DEFAULT_HISTORICAL_PARTITION_CONFIG,
) -> HistoricalPartitionManifest:
    selected_station_ids = tuple(station_ids)
    if not selected_station_ids:
        raise ValueError("station_ids cannot be empty")
    if len(set(selected_station_ids)) != len(selected_station_ids):
        raise ValueError("station_ids must be unique")
    if config.minimum_observed_station_count > len(selected_station_ids):
        raise ValueError(
            "minimum_observed_station_count cannot exceed selected station count"
        )
    if weather.resource_id is None or weather.resource_sha256 is None:
        raise ValueError(
            "Historical partition requires identified weather resource content"
        )

    station_counts = {station_id: 0 for station_id in selected_station_ids}
    observed_at: dict[datetime, set[str]] = {}
    selected = set(selected_station_ids)
    for record in pm25.observed_records:
        if record.station_id not in selected:
            continue
        station_counts[record.station_id] += 1
        observed_at.setdefault(record.timestamp_utc, set()).add(record.station_id)

    eligible_hours = tuple(
        timestamp
        for timestamp in sorted(observed_at)
        if len(observed_at[timestamp]) >= config.minimum_observed_station_count
        and _weather_covers(weather, timestamp)
    )
    if not eligible_hours:
        raise ValueError("No hourly coverage satisfies the historical partition policy")

    overall_start = eligible_hours[0]
    overall_end = eligible_hours[-1] + timedelta(hours=1)
    total_hours = int((overall_end - overall_start) / timedelta(hours=1))
    calibration_hours = math.floor(total_hours * config.calibration_fraction)
    if calibration_hours <= 0 or calibration_hours >= total_hours:
        raise ValueError("Chronological partition produces an empty partition")
    calibration_end = overall_start + timedelta(hours=calibration_hours)

    calibration_starts = _candidate_window_starts(
        eligible_hours,
        partition_start=overall_start,
        partition_end=calibration_end,
        config=config,
    )
    replay_starts = _candidate_window_starts(
        eligible_hours,
        partition_start=calibration_end,
        partition_end=overall_end,
        config=config,
    )
    selected_calibration = _select_spread_nonoverlapping(
        calibration_starts,
        config.calibration_window_count,
        config.window_duration,
    )
    selected_replay = _select_spread_nonoverlapping(
        replay_starts,
        config.replay_window_count,
        config.window_duration,
    )

    calibration_windows = tuple(
        HistoricalWindow(
            HistoricalPartition.CALIBRATION,
            start - config.context_duration,
            start,
            start + config.window_duration,
        )
        for start in selected_calibration
    )
    replay_windows = tuple(
        HistoricalWindow(
            HistoricalPartition.REPLAY,
            start - config.context_duration,
            start,
            start + config.window_duration,
        )
        for start in selected_replay
    )
    return HistoricalPartitionManifest(
        pm25_resource_id=pm25.resource_id,
        pm25_resource_sha256=pm25.resource_sha256,
        weather_resource_id=weather.resource_id,
        weather_resource_sha256=weather.resource_sha256,
        weather_station_id=weather.station_id,
        station_ids=selected_station_ids,
        station_observed_counts=tuple(station_counts.items()),
        calibration_fraction=config.calibration_fraction,
        calibration_start_utc=overall_start,
        calibration_end_utc=calibration_end,
        replay_start_utc=calibration_end,
        replay_end_utc=overall_end,
        minimum_observed_station_count=config.minimum_observed_station_count,
        eligible_hour_count=len(eligible_hours),
        selection_method="chronological_fraction_spread_nonoverlapping",
        pm25_observation_policy="present_senamhi_observed_only",
        calibration_windows=calibration_windows,
        replay_windows=replay_windows,
    )


def write_historical_partition_manifest(
    manifest: HistoricalPartitionManifest,
    path: Path | str,
) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"Historical partition manifest already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary historical partition manifest already exists: {temporary}"
        )
    serialized = json.dumps(
        manifest.evidence_descriptor(),
        indent=2,
        sort_keys=True,
    )
    try:
        temporary.write_text(f"{serialized}\n", encoding="utf-8")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _candidate_window_starts(
    eligible_hours: tuple[datetime, ...],
    *,
    partition_start: datetime,
    partition_end: datetime,
    config: HistoricalPartitionConfig,
) -> tuple[datetime, ...]:
    eligible = set(eligible_hours)
    scored_hours = int(config.window_duration / timedelta(hours=1))
    context_hours = int(config.context_duration / timedelta(hours=1))
    candidates = []
    for start in eligible_hours:
        context_start = start - config.context_duration
        end = start + config.window_duration
        if context_start < partition_start or end > partition_end:
            continue
        required = (
            context_start + timedelta(hours=offset)
            for offset in range(context_hours + scored_hours)
        )
        if all(timestamp in eligible for timestamp in required):
            candidates.append(start)
    return tuple(candidates)


def _select_spread_nonoverlapping(
    candidates: tuple[datetime, ...],
    count: int,
    duration: timedelta,
) -> tuple[datetime, ...]:
    available = list(candidates)
    selected: list[datetime] = []
    while len(selected) < count:
        nonoverlapping = [
            candidate
            for candidate in available
            if all(abs(candidate - existing) >= duration for existing in selected)
        ]
        if not nonoverlapping:
            raise ValueError(
                f"Historical coverage cannot supply {count} nonoverlapping "
                f"windows of {duration}"
            )
        if not selected:
            chosen = nonoverlapping[0]
        else:
            chosen = max(
                nonoverlapping,
                key=lambda candidate: (
                    min(abs(candidate - existing) for existing in selected),
                    -candidate.timestamp(),
                ),
            )
        selected.append(chosen)
        available.remove(chosen)
    return tuple(sorted(selected))


def _weather_covers(weather: HistoricalWeatherDataset, timestamp: datetime) -> bool:
    try:
        weather.record_at(timestamp)
    except WeatherCoverageError:
        return False
    return True


def _window_descriptor(window: HistoricalWindow) -> dict[str, str]:
    return {
        "partition": window.partition.value,
        "context_start_utc": window.context_start_utc.isoformat(),
        "start_utc": window.start_utc.isoformat(),
        "end_utc": window.end_utc.isoformat(),
    }
