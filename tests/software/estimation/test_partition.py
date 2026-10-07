from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import pytest

from lima_pm25.estimation import (
    HistoricalPM25Dataset,
    HistoricalPM25Record,
    HistoricalPartition,
    HistoricalPartitionConfig,
    build_historical_partition_manifest,
    write_historical_partition_manifest,
)
from lima_pm25.simulation import (
    HistoricalWeatherDataset,
    WeatherRecord,
    WeatherValueProvenance,
)


def test_partition_selects_disjoint_spread_calibration_and_replay_windows() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    hours = 20 * 24
    pm25 = _pm25_dataset(start, hours)
    weather = _weather_dataset(start, hours)
    config = HistoricalPartitionConfig(
        calibration_fraction=0.7,
        calibration_window_count=3,
        replay_window_count=2,
        window_duration=timedelta(hours=24),
        context_duration=timedelta(hours=12),
        minimum_observed_station_count=3,
    )

    manifest = build_historical_partition_manifest(
        pm25=pm25,
        weather=weather,
        station_ids=("A", "B", "C", "CERES"),
        config=config,
    )

    assert len(manifest.calibration_windows) == 3
    assert len(manifest.replay_windows) == 2
    assert manifest.station_observed_counts == (
        ("A", hours),
        ("B", hours),
        ("C", hours),
        ("CERES", 0),
    )
    assert all(
        window.partition is HistoricalPartition.CALIBRATION
        and window.end_utc <= manifest.calibration_end_utc
        for window in manifest.calibration_windows
    )
    assert all(
        window.partition is HistoricalPartition.REPLAY
        and window.context_start_utc >= manifest.replay_start_utc
        for window in manifest.replay_windows
    )
    assert all(
        window.duration == timedelta(hours=24)
        and window.context_duration == timedelta(hours=12)
        for window in (*manifest.calibration_windows, *manifest.replay_windows)
    )
    assert manifest.evidence_descriptor()["station_observed_counts"]["CERES"] == 0


def test_partition_refuses_insufficient_concurrent_station_coverage() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    pm25 = _pm25_dataset(start, 72, station_ids=("A", "B"))
    weather = _weather_dataset(start, 72)

    try:
        build_historical_partition_manifest(
            pm25=pm25,
            weather=weather,
            station_ids=("A", "B", "C"),
            config=HistoricalPartitionConfig(
                calibration_window_count=1,
                replay_window_count=1,
                window_duration=timedelta(hours=6),
                context_duration=timedelta(0),
                minimum_observed_station_count=3,
            ),
        )
    except ValueError as error:
        assert "No hourly coverage" in str(error)
    else:
        raise AssertionError("Expected insufficient station coverage to fail")


def test_partition_manifest_is_persisted_without_overwriting(tmp_path: Path) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    manifest = build_historical_partition_manifest(
        pm25=_pm25_dataset(start, 10 * 24),
        weather=_weather_dataset(start, 10 * 24),
        station_ids=("A", "B", "C", "CERES"),
        config=HistoricalPartitionConfig(
            calibration_window_count=1,
            replay_window_count=1,
            window_duration=timedelta(hours=24),
            context_duration=timedelta(0),
            minimum_observed_station_count=3,
        ),
    )
    destination = tmp_path / "partition.json"

    written = write_historical_partition_manifest(manifest, destination)

    assert written == destination
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["selection_method"] == (
        "chronological_fraction_spread_nonoverlapping"
    )
    assert payload["pm25_observation_policy"] == (
        "present_senamhi_observed_only"
    )
    with pytest.raises(FileExistsError, match="already exists"):
        write_historical_partition_manifest(manifest, destination)


def _pm25_dataset(
    start: datetime,
    hours: int,
    *,
    station_ids: tuple[str, ...] = ("A", "B", "C"),
) -> HistoricalPM25Dataset:
    records = tuple(
        HistoricalPM25Record(
            station_id=station_id,
            timestamp_utc=start + timedelta(hours=hour),
            longitude_deg=-77 + index * 0.01,
            latitude_deg=-12,
            altitude_m=None,
            value_ug_m3=10 + index + hour % 24,
            province="LIMA",
            district="TEST",
            source="senamhi_observed",
        )
        for hour in range(hours)
        for index, station_id in enumerate(station_ids)
    )
    return HistoricalPM25Dataset(
        records=records,
        resource_id="pm25.csv",
        resource_sha256="1" * 64,
        source_timezone="UTC",
    )


def _weather_dataset(start: datetime, hours: int) -> HistoricalWeatherDataset:
    provenance = WeatherValueProvenance("unit_test_fixture")
    records = tuple(
        WeatherRecord(
            station_id="WEATHER",
            timestamp_utc=start + timedelta(hours=hour),
            temperature_c=20,
            relative_humidity_pct=80,
            eastward_wind_m_s=1,
            northward_wind_m_s=0,
            precipitation_mm_h=0,
            temperature_provenance=provenance,
            relative_humidity_provenance=provenance,
            precipitation_provenance=provenance,
            wind_direction_provenance=provenance,
            wind_speed_provenance=provenance,
        )
        for hour in range(hours)
    )
    return HistoricalWeatherDataset(
        station_id="WEATHER",
        provider_id="weather:test",
        provenance="unit test",
        records=records,
        resource_id="weather.csv",
        resource_sha256="2" * 64,
    )
