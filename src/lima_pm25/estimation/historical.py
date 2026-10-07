from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import math
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from pyproj import CRS, Transformer

from lima_pm25.domain import GridDefinition
from lima_pm25.simulation import (
    MeasurementQuality,
    MeasurementRecord,
    SensorType,
)


@dataclass(frozen=True, slots=True)
class PM25CsvConfig:
    station_column: str = "ESTACION"
    date_column: str = "FECHA"
    time_column: str = "HORA"
    longitude_column: str = "LONGITUD"
    latitude_column: str = "LATITUD"
    altitude_column: str = "ALTITUD"
    value_column: str = "PM2_5"
    province_column: str = "PROVINCIA"
    district_column: str = "DISTRITO"
    source_column: str = "PM2_5_SOURCE"
    source_timezone: str = "America/Lima"
    missing_values: frozenset[str] = frozenset({"", "s/d", "nan", "null"})
    delimiter: str = ","
    encoding: str = "utf-8-sig"

    def __post_init__(self) -> None:
        if any(not column.strip() for column in self.required_columns):
            raise ValueError("PM2.5 source column names must be nonempty")
        if len(set(self.required_columns)) != len(self.required_columns):
            raise ValueError("PM2.5 source column names must be distinct")
        if len(self.delimiter) != 1:
            raise ValueError("delimiter must contain exactly one character")
        if not self.encoding.strip():
            raise ValueError("encoding must be nonempty")
        if not self.source_timezone.strip():
            raise ValueError("source_timezone must be nonempty")
        object.__setattr__(
            self,
            "missing_values",
            frozenset(value.strip().casefold() for value in self.missing_values),
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.station_column,
            self.date_column,
            self.time_column,
            self.longitude_column,
            self.latitude_column,
            self.altitude_column,
            self.value_column,
            self.province_column,
            self.district_column,
            self.source_column,
        )


DEFAULT_PM25_CSV_CONFIG = PM25CsvConfig()


@dataclass(frozen=True, slots=True)
class HistoricalPM25Record:
    station_id: str
    timestamp_utc: datetime
    longitude_deg: float
    latitude_deg: float
    altitude_m: float | None
    value_ug_m3: float | None
    province: str
    district: str
    source: str

    def __post_init__(self) -> None:
        _require_text("station_id", self.station_id)
        _require_utc("timestamp_utc", self.timestamp_utc)
        if not math.isfinite(self.longitude_deg) or not -180 <= self.longitude_deg <= 180:
            raise ValueError("longitude_deg must be finite and between -180 and 180")
        if not math.isfinite(self.latitude_deg) or not -90 <= self.latitude_deg <= 90:
            raise ValueError("latitude_deg must be finite and between -90 and 90")
        if self.altitude_m is not None and not math.isfinite(self.altitude_m):
            raise ValueError("altitude_m must be finite when supplied")
        if self.value_ug_m3 is not None and not math.isfinite(self.value_ug_m3):
            raise ValueError("value_ug_m3 must be finite when supplied")
        _require_text("province", self.province)
        _require_text("district", self.district)
        _require_text("source", self.source)

    @property
    def is_observed(self) -> bool:
        return self.value_ug_m3 is not None and self.source == "senamhi_observed"


@dataclass(frozen=True, slots=True)
class HistoricalPM25Dataset:
    records: tuple[HistoricalPM25Record, ...]
    resource_id: str
    resource_sha256: str
    source_timezone: str
    _observed_records: tuple[HistoricalPM25Record, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records:
            raise ValueError("Historical PM2.5 dataset has no records")
        _require_text("resource_id", self.resource_id)
        _require_sha256(self.resource_sha256)
        _require_text("source_timezone", self.source_timezone)

        keys: set[tuple[str, datetime]] = set()
        for record in records:
            key = (record.station_id, record.timestamp_utc)
            if key in keys:
                raise ValueError(
                    f"Duplicate PM2.5 record for station {record.station_id!r} at "
                    f"{record.timestamp_utc.isoformat()}"
                )
            keys.add(key)

        observed = tuple(
            sorted(
                (record for record in records if record.is_observed),
                key=lambda record: (record.timestamp_utc, record.station_id),
            )
        )
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "_observed_records", observed)

    @property
    def observed_records(self) -> tuple[HistoricalPM25Record, ...]:
        return self._observed_records

    @property
    def station_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(record.station_id for record in self.records))

    @property
    def coverage_start_utc(self) -> datetime:
        return min(record.timestamp_utc for record in self.records)

    @property
    def coverage_end_utc(self) -> datetime:
        return max(record.timestamp_utc for record in self.records)

    def evidence_descriptor(self) -> dict[str, object]:
        source_counts = Counter(record.source for record in self.records)
        return {
            "kind": "historical_pm25",
            "resource_id": self.resource_id,
            "resource_sha256": self.resource_sha256,
            "source_timezone": self.source_timezone,
            "record_count": len(self.records),
            "observed_record_count": len(self.observed_records),
            "missing_value_count": sum(
                record.value_ug_m3 is None for record in self.records
            ),
            "station_ids": list(self.station_ids),
            "coverage_start_utc": self.coverage_start_utc.isoformat(),
            "coverage_end_utc": self.coverage_end_utc.isoformat(),
            "source_counts": dict(sorted(source_counts.items())),
        }

    def observed_measurements(
        self,
        *,
        grid: GridDefinition,
        replay_start_utc: datetime,
        end_utc: datetime,
        noise_stddev_ug_m3: float,
        station_ids: Sequence[str] | None = None,
    ) -> tuple[MeasurementRecord, ...]:
        _require_utc("replay_start_utc", replay_start_utc)
        _require_utc("end_utc", end_utc)
        if end_utc <= replay_start_utc:
            raise ValueError("end_utc must be after replay_start_utc")
        if not math.isfinite(noise_stddev_ug_m3) or noise_stddev_ug_m3 < 0:
            raise ValueError("noise_stddev_ug_m3 must be finite and nonnegative")

        grid_crs = CRS.from_user_input(grid.crs)
        transformer = Transformer.from_crs("EPSG:4326", grid_crs, always_xy=True)
        selected_station_ids = None if station_ids is None else frozenset(station_ids)
        if selected_station_ids is not None and not selected_station_ids:
            raise ValueError("station_ids cannot be empty when supplied")
        measurements: list[MeasurementRecord] = []
        for record in self.observed_records:
            if (
                selected_station_ids is not None
                and record.station_id not in selected_station_ids
            ):
                continue
            if not replay_start_utc <= record.timestamp_utc < end_utc:
                continue
            assert record.value_ug_m3 is not None
            if record.value_ug_m3 < 0:
                raise ValueError(
                    f"Observed PM2.5 for station {record.station_id!r} at "
                    f"{record.timestamp_utc.isoformat()} is negative"
                )
            x_m, y_m = transformer.transform(
                record.longitude_deg,
                record.latitude_deg,
                errcheck=True,
            )
            x_m = float(x_m)
            y_m = float(y_m)
            if not grid.is_active_xy(x_m, y_m):
                raise ValueError(
                    f"Historical PM2.5 station {record.station_id!r} is not in "
                    "an active reconstruction cell"
                )
            measurements.append(
                MeasurementRecord(
                    sensor_id=record.station_id,
                    sensor_type=SensorType.FIXED,
                    timestamp_utc=record.timestamp_utc,
                    simulation_time_s=(
                        record.timestamp_utc - replay_start_utc
                    ).total_seconds(),
                    x_m=x_m,
                    y_m=y_m,
                    crs=grid.crs,
                    value_ug_m3=record.value_ug_m3,
                    quality_status=MeasurementQuality.ACCEPTED,
                    noise_stddev_ug_m3=noise_stddev_ug_m3,
                    position_source_id=record.station_id,
                    provenance=(
                        f"observed PM2.5 from {self.resource_id}; "
                        f"PM2_5_SOURCE={record.source}"
                    ),
                )
            )
        return tuple(measurements)


def load_historical_pm25_csv(
    path: Path | str,
    *,
    config: PM25CsvConfig = DEFAULT_PM25_CSV_CONFIG,
) -> HistoricalPM25Dataset:
    source_path = Path(path)
    timezone = _load_timezone(config.source_timezone)
    records: list[HistoricalPM25Record] = []

    with source_path.open("r", newline="", encoding=config.encoding) as stream:
        reader = csv.DictReader(stream, delimiter=config.delimiter, strict=True)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [
            column for column in config.required_columns if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                "PM2.5 CSV is missing required columns: "
                f"{', '.join(missing_columns)}"
            )

        for row in reader:
            line_number = reader.line_num
            records.append(
                HistoricalPM25Record(
                    station_id=_required_text(
                        row,
                        config.station_column,
                        line_number,
                    ),
                    timestamp_utc=_parse_timestamp(
                        row,
                        config,
                        timezone,
                        line_number,
                    ),
                    longitude_deg=_required_float(
                        row,
                        config.longitude_column,
                        line_number,
                    ),
                    latitude_deg=_required_float(
                        row,
                        config.latitude_column,
                        line_number,
                    ),
                    altitude_m=_optional_float(row, config.altitude_column, config),
                    value_ug_m3=_optional_float(row, config.value_column, config),
                    province=_required_text(
                        row,
                        config.province_column,
                        line_number,
                    ),
                    district=_required_text(
                        row,
                        config.district_column,
                        line_number,
                    ),
                    source=_required_text(
                        row,
                        config.source_column,
                        line_number,
                    ),
                )
            )

    return HistoricalPM25Dataset(
        records=tuple(records),
        resource_id=source_path.name,
        resource_sha256=sha256(source_path.read_bytes()).hexdigest(),
        source_timezone=config.source_timezone,
    )


def _parse_timestamp(
    row: dict[str, str | None],
    config: PM25CsvConfig,
    timezone: ZoneInfo,
    line_number: int,
) -> datetime:
    date_text = (row.get(config.date_column) or "").strip()
    time_text = (row.get(config.time_column) or "").strip()
    try:
        hour_code = int(time_text)
    except ValueError as error:
        raise ValueError(
            f"PM2.5 CSV line {line_number} has invalid HORA {time_text!r}"
        ) from error
    if hour_code not in range(0, 240_000, 10_000):
        raise ValueError(
            f"PM2.5 CSV line {line_number} has non-hourly HORA {time_text!r}"
        )
    try:
        local = datetime.strptime(
            f"{date_text}{hour_code // 10_000:02d}",
            "%Y%m%d%H",
        ).replace(tzinfo=timezone)
    except ValueError as error:
        raise ValueError(
            f"PM2.5 CSV line {line_number} has invalid FECHA/HORA "
            f"{date_text!r}/{time_text!r}"
        ) from error
    return local.astimezone(UTC)


def _required_text(
    row: dict[str, str | None],
    column: str,
    line_number: int,
) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        raise ValueError(
            f"PM2.5 CSV line {line_number} has empty required column {column!r}"
        )
    return value


def _required_float(
    row: dict[str, str | None],
    column: str,
    line_number: int,
) -> float:
    text = (row.get(column) or "").strip()
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(
            f"PM2.5 CSV line {line_number} has invalid numeric column "
            f"{column!r}: {text!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"PM2.5 CSV line {line_number} column {column!r} must be finite"
        )
    return value


def _optional_float(
    row: dict[str, str | None],
    column: str,
    config: PM25CsvConfig,
) -> float | None:
    text = (row.get(column) or "").strip()
    if text.casefold() in config.missing_values:
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(
            f"PM2.5 value {text!r} in column {column!r} is not numeric"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"PM2.5 value in column {column!r} must be finite")
    return value


def _load_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Unknown source_timezone {name!r}") from error


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _require_utc(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("resource_sha256 must be a lowercase SHA-256 hexadecimal digest")
