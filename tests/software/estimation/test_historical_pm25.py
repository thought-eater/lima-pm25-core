import csv
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from lima_pm25.domain import GridDefinition
from lima_pm25.estimation import (
    HistoricalPM25Record,
    PM25CsvConfig,
    load_historical_pm25_csv,
)
from lima_pm25.simulation import SensorType

HEADERS = (
    "ESTACION",
    "FECHA",
    "HORA",
    "LONGITUD",
    "LATITUD",
    "ALTITUD",
    "PM2_5",
    "PROVINCIA",
    "DISTRITO",
    "PM2_5_SOURCE",
)


def test_loader_preserves_values_coordinates_missingness_and_source(
    tmp_path: Path,
) -> None:
    path = _write_pm25_csv(
        tmp_path,
        [
            _row("CAMPO_DE_MARTE", 0, value=22.2),
            _row(
                "CAMPO_DE_MARTE",
                10000,
                value=23.4,
                source="imputed_linear",
            ),
            _row(
                "CERES",
                0,
                longitude=-76.92706,
                latitude=-12.02869,
                altitude="",
                value="",
                source="missing_explicit",
            ),
        ],
    )

    dataset = load_historical_pm25_csv(path)

    assert dataset.records[0] == HistoricalPM25Record(
        station_id="CAMPO_DE_MARTE",
        timestamp_utc=datetime(2025, 1, 1, 5, tzinfo=UTC),
        longitude_deg=-77.0432,
        latitude_deg=-12.0705,
        altitude_m=117,
        value_ug_m3=22.2,
        province="LIMA",
        district="JESUS_MARIA",
        source="senamhi_observed",
    )
    assert dataset.records[1].value_ug_m3 == 23.4
    assert dataset.records[1].source == "imputed_linear"
    assert dataset.records[2].value_ug_m3 is None
    assert dataset.records[2].altitude_m is None
    assert dataset.observed_records == (dataset.records[0],)
    assert not hasattr(dataset.records[0], "instrument")

    descriptor = dataset.evidence_descriptor()
    assert descriptor["record_count"] == 3
    assert descriptor["observed_record_count"] == 1
    assert descriptor["missing_value_count"] == 1
    assert descriptor["station_ids"] == ["CAMPO_DE_MARTE", "CERES"]
    assert descriptor["source_counts"] == {
        "imputed_linear": 1,
        "missing_explicit": 1,
        "senamhi_observed": 1,
    }
    assert len(str(descriptor["resource_sha256"])) == 64


def test_loader_requires_pm25_source_column(tmp_path: Path) -> None:
    path = tmp_path / "missing-source.csv"
    headers = tuple(column for column in HEADERS if column != "PM2_5_SOURCE")
    row = _row("CAMPO_DE_MARTE", 0)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerow({column: row[column] for column in headers})

    with pytest.raises(ValueError, match="missing required columns: PM2_5_SOURCE"):
        load_historical_pm25_csv(path)


def test_loader_rejects_duplicate_station_timestamp(tmp_path: Path) -> None:
    row = _row("CAMPO_DE_MARTE", 0)
    path = _write_pm25_csv(tmp_path, [row, row])

    with pytest.raises(ValueError, match="Duplicate PM2.5 record"):
        load_historical_pm25_csv(path)


def test_custom_schema_timezone_and_missing_marker_are_explicit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "custom.csv"
    headers = tuple(f"custom_{header}" for header in HEADERS)
    values = _row("A", 0, value="NA")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerow({f"custom_{key}": value for key, value in values.items()})
    config = PM25CsvConfig(
        station_column="custom_ESTACION",
        date_column="custom_FECHA",
        time_column="custom_HORA",
        longitude_column="custom_LONGITUD",
        latitude_column="custom_LATITUD",
        altitude_column="custom_ALTITUD",
        value_column="custom_PM2_5",
        province_column="custom_PROVINCIA",
        district_column="custom_DISTRITO",
        source_column="custom_PM2_5_SOURCE",
        source_timezone="UTC",
        missing_values=frozenset({"", "NA"}),
    )

    dataset = load_historical_pm25_csv(path, config=config)

    assert dataset.records[0].timestamp_utc == datetime(2025, 1, 1, tzinfo=UTC)
    assert dataset.records[0].value_ug_m3 is None


def test_observed_measurements_exclude_imputed_pm25_and_project_source_coordinates(
    tmp_path: Path,
) -> None:
    path = _write_pm25_csv(
        tmp_path,
        [
            _row("CAMPO_DE_MARTE", 0, value=22.2),
            _row(
                "CAMPO_DE_MARTE",
                10000,
                value=23.4,
                source="imputed_linear",
            ),
            _row("CAMPO_DE_MARTE", 20000, value=24.5),
        ],
    )
    dataset = load_historical_pm25_csv(path)
    grid = _lima_grid()

    records = dataset.observed_measurements(
        grid=grid,
        replay_start_utc=datetime(2025, 1, 1, 5, tzinfo=UTC),
        end_utc=datetime(2025, 1, 1, 8, tzinfo=UTC),
        noise_stddev_ug_m3=2,
        station_ids=("CAMPO_DE_MARTE",),
    )

    assert len(records) == 2
    assert [record.value_ug_m3 for record in records] == [22.2, 24.5]
    assert [record.simulation_time_s for record in records] == [0, 7_200]
    assert all(record.sensor_type is SensorType.FIXED for record in records)
    assert all(record.position_source_id == "CAMPO_DE_MARTE" for record in records)
    assert all(record.noise_stddev_ug_m3 == 2 for record in records)
    assert all(
        "PM2_5_SOURCE=senamhi_observed" in record.provenance for record in records
    )
    assert all(grid.is_active_xy(record.x_m, record.y_m) for record in records)


def _write_pm25_csv(
    directory: Path,
    rows: list[dict[str, object]],
) -> Path:
    path = directory / "pm25.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(
    station: str,
    hour: int,
    *,
    longitude: object = -77.0432,
    latitude: object = -12.0705,
    altitude: object = 117,
    value: object = 20,
    source: str = "senamhi_observed",
) -> dict[str, object]:
    return {
        "ESTACION": station,
        "FECHA": 20250101,
        "HORA": hour,
        "LONGITUD": longitude,
        "LATITUD": latitude,
        "ALTITUD": altitude,
        "PM2_5": value,
        "PROVINCIA": "LIMA",
        "DISTRITO": "JESUS_MARIA",
        "PM2_5_SOURCE": source,
    }


def _lima_grid() -> GridDefinition:
    return GridDefinition(
        crs="EPSG:32718",
        domain_bounds_m=np.array([275_000, 8_660_000, 295_000, 8_680_000]),
        grid_bounds_m=np.array([275_000, 8_660_000, 295_000, 8_680_000]),
        cell_size_m=1_000,
        x_centers_m=275_500 + np.arange(20) * 1_000,
        y_centers_m=8_660_500 + np.arange(20) * 1_000,
        active_mask=np.ones((20, 20), dtype=bool),
    )
