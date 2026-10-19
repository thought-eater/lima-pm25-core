from __future__ import annotations

import csv
import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from lima_pm25.domain import AtmosphericForcing


class MissingValuePolicy(StrEnum):
    ERROR = "error"
    GAP = "gap"


class PrecipitationSourcePolicy(StrEnum):
    STATION_ONLY = "station_only"
    STATION_THEN_IFS = "station_then_ifs"


class WeatherCoverageError(ValueError):
    """Raised when historical forcing is requested without valid coverage."""


@dataclass(frozen=True, slots=True)
class WeatherCsvConfig:
    station_column: str = "ESTACION"
    date_column: str = "FECHA"
    time_column: str = "HORA"
    temperature_column: str = "TEMP"
    relative_humidity_column: str = "HR"
    precipitation_column: str = "PP"
    wind_direction_column: str = "DIR_VIENTO"
    wind_speed_column: str = "VEL_VIENTO"
    ifs_precipitation_column: str = "PP_IFS"
    source_timezone: str = "America/Lima"
    missing_values: frozenset[str] = frozenset({"", "s/d", "nan", "null"})
    missing_value_policy: MissingValuePolicy | str = MissingValuePolicy.GAP
    precipitation_source_policy: PrecipitationSourcePolicy | str = (
        PrecipitationSourcePolicy.STATION_ONLY
    )
    maximum_gap: timedelta = timedelta(hours=1)
    delimiter: str = ","
    encoding: str = "utf-8-sig"

    def __post_init__(self) -> None:
        columns = self.required_columns
        if any(not name.strip() for name in columns):
            raise ValueError("Weather source column names must be nonempty")
        if len(set(columns)) != len(columns):
            raise ValueError("Weather source column names must be distinct")
        if len(self.delimiter) != 1:
            raise ValueError("delimiter must contain exactly one character")
        if not self.encoding.strip():
            raise ValueError("encoding must be nonempty")
        if self.maximum_gap <= timedelta(0):
            raise ValueError("maximum_gap must be positive")

        if not self.source_timezone.strip():
            raise ValueError("source_timezone must be nonempty")

        object.__setattr__(
            self,
            "missing_values",
            frozenset(value.strip().casefold() for value in self.missing_values),
        )
        object.__setattr__(
            self,
            "missing_value_policy",
            MissingValuePolicy(self.missing_value_policy),
        )
        object.__setattr__(
            self,
            "precipitation_source_policy",
            PrecipitationSourcePolicy(self.precipitation_source_policy),
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        value_columns = (
            self.station_column,
            self.date_column,
            self.time_column,
            self.temperature_column,
            self.relative_humidity_column,
            self.precipitation_column,
            self.wind_direction_column,
            self.wind_speed_column,
        )
        provenance_columns = tuple(
            f"{value_column}_SOURCE" for value_column in value_columns[3:]
        )
        if (
            self.precipitation_source_policy
            is PrecipitationSourcePolicy.STATION_THEN_IFS
        ):
            return (*value_columns, *provenance_columns, self.ifs_precipitation_column)
        return (*value_columns, *provenance_columns)


DEFAULT_WEATHER_CSV_CONFIG = WeatherCsvConfig()


@dataclass(frozen=True, slots=True)
class WeatherValueProvenance:
    source: str
    upstream_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source cannot be empty")
        if self.source.strip().casefold() == "unspecified":
            raise ValueError("source cannot be 'unspecified'")
        object.__setattr__(self, "upstream_sources", tuple(self.upstream_sources))


@dataclass(frozen=True, slots=True)
class WeatherDiagnostic:
    timestamp_utc: datetime
    provider_id: str
    provider_provenance: str
    kind: str
    resource_id: str | None
    resource_sha256: str | None
    station_id: str | None
    precipitation_source_policy: PrecipitationSourcePolicy | None
    temperature_c: float | None
    relative_humidity_pct: float | None
    temperature_provenance: WeatherValueProvenance
    relative_humidity_provenance: WeatherValueProvenance
    precipitation_provenance: WeatherValueProvenance
    wind_direction_provenance: WeatherValueProvenance
    wind_speed_provenance: WeatherValueProvenance

    def __post_init__(self) -> None:
        _require_aware_datetime("timestamp_utc", self.timestamp_utc)
        if self.timestamp_utc.utcoffset() != timedelta(0):
            raise ValueError("timestamp_utc must use UTC")
        _require_usable_text("provider_id", self.provider_id)
        _require_usable_text("provider_provenance", self.provider_provenance)
        if self.kind not in {"historical", "synthetic"}:
            raise ValueError("kind must be 'historical' or 'synthetic'")
        _require_optional_finite("temperature_c", self.temperature_c)
        _require_optional_finite(
            "relative_humidity_pct",
            self.relative_humidity_pct,
        )


@dataclass(frozen=True, slots=True)
class WeatherRecord:
    station_id: str
    timestamp_utc: datetime
    temperature_c: float | None
    relative_humidity_pct: float | None
    eastward_wind_m_s: float
    northward_wind_m_s: float
    precipitation_mm_h: float
    temperature_provenance: WeatherValueProvenance
    relative_humidity_provenance: WeatherValueProvenance
    precipitation_provenance: WeatherValueProvenance
    wind_direction_provenance: WeatherValueProvenance
    wind_speed_provenance: WeatherValueProvenance

    def __post_init__(self) -> None:
        _require_usable_text("station_id", self.station_id)
        _require_aware_datetime("timestamp_utc", self.timestamp_utc)
        if self.timestamp_utc.utcoffset() != timedelta(0):
            raise ValueError("timestamp_utc must use UTC")
        _require_optional_finite("temperature_c", self.temperature_c)
        _require_optional_finite(
            "relative_humidity_pct",
            self.relative_humidity_pct,
        )
        _require_finite("eastward_wind_m_s", self.eastward_wind_m_s)
        _require_finite("northward_wind_m_s", self.northward_wind_m_s)
        _require_nonnegative_finite(
            "precipitation_mm_h",
            self.precipitation_mm_h,
        )


@dataclass(frozen=True, slots=True)
class WeatherWindow:
    dataset: HistoricalWeatherDataset = field(repr=False, compare=False)
    start_utc: datetime
    end_utc: datetime
    records: tuple[WeatherRecord, ...]
    forcing_change_times_s: tuple[float, ...]
    selection_parameters_json: str = "{}"

    @property
    def duration_s(self) -> float:
        return (self.end_utc - self.start_utc).total_seconds()

    def utc_at_elapsed_s(self, elapsed_s: float) -> datetime:
        if not math.isfinite(elapsed_s) or not 0 <= elapsed_s <= self.duration_s:
            raise WeatherCoverageError(
                f"Elapsed time {elapsed_s!r} s is outside weather window "
                f"[0, {self.duration_s:g}] s"
            )
        return self.start_utc + timedelta(seconds=elapsed_s)

    def forcing_at_elapsed_s(self, elapsed_s: float) -> AtmosphericForcing:
        return self.dataset.forcing_at(self.utc_at_elapsed_s(elapsed_s))

    @property
    def provider_id(self) -> str:
        return self.dataset.provider_id

    @property
    def provenance(self) -> str:
        return self.dataset.provenance

    @property
    def transition_times_s(self) -> tuple[float, ...]:
        return self.forcing_change_times_s

    def diagnostic_at_elapsed_s(self, elapsed_s: float) -> WeatherDiagnostic:
        record = self.dataset.record_at(self.utc_at_elapsed_s(elapsed_s))
        return WeatherDiagnostic(
            timestamp_utc=record.timestamp_utc,
            provider_id=self.dataset.provider_id,
            provider_provenance=self.dataset.provenance,
            kind="historical",
            resource_id=self.dataset.resource_id,
            resource_sha256=self.dataset.resource_sha256,
            station_id=self.dataset.station_id,
            precipitation_source_policy=PrecipitationSourcePolicy(
                self.dataset.precipitation_source_policy
            ),
            temperature_c=record.temperature_c,
            relative_humidity_pct=record.relative_humidity_pct,
            temperature_provenance=record.temperature_provenance,
            relative_humidity_provenance=record.relative_humidity_provenance,
            precipitation_provenance=record.precipitation_provenance,
            wind_direction_provenance=record.wind_direction_provenance,
            wind_speed_provenance=record.wind_speed_provenance,
        )

    def evidence_descriptor(self) -> dict[str, object]:
        records_payload = "\n".join(repr(record) for record in self.records).encode(
            "utf-8"
        )
        return {
            "kind": "historical",
            "provider_id": self.dataset.provider_id,
            "provenance": self.dataset.provenance,
            "resource_id": self.dataset.resource_id,
            "resource_sha256": self.dataset.resource_sha256,
            "station_id": self.dataset.station_id,
            "start_utc": self.start_utc.isoformat(),
            "end_utc": self.end_utc.isoformat(),
            "precipitation_source_policy": (
                PrecipitationSourcePolicy(
                    self.dataset.precipitation_source_policy
                ).value
            ),
            "records_sha256": sha256(records_payload).hexdigest(),
            "selection": json.loads(self.selection_parameters_json),
            "transition_times_s": list(self.transition_times_s),
        }


@dataclass(frozen=True, slots=True)
class ScheduledWeatherValue:
    starts_at_s: float
    forcing: AtmosphericForcing

    def __post_init__(self) -> None:
        _require_nonnegative_finite("starts_at_s", self.starts_at_s)


@dataclass(frozen=True, slots=True)
class SyntheticWeatherWindow:
    start_utc: datetime
    duration_s: float
    values: tuple[ScheduledWeatherValue, ...]
    construction_method: str = "explicit_schedule"
    stochastic: bool = False
    seed: int | None = None
    interpolate_wind: bool = False
    historical_base: WeatherWindow | None = field(
        default=None, repr=False, compare=False
    )
    generation_parameters_json: str = "{}"

    def __post_init__(self) -> None:
        _require_aware_datetime("start_utc", self.start_utc)
        if self.start_utc.utcoffset() != timedelta(0):
            raise ValueError("start_utc must use UTC")
        _require_positive_finite("duration_s", self.duration_s)
        _require_usable_text("construction_method", self.construction_method)
        if self.stochastic and (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("Stochastic synthetic weather requires a nonnegative seed")
        values = tuple(self.values)
        if not values or values[0].starts_at_s != 0:
            raise ValueError("Synthetic weather must define its first value at t=0")
        for left, right in pairwise(values):
            if right.starts_at_s <= left.starts_at_s:
                raise ValueError("Synthetic weather start times must increase")
        if self.interpolate_wind:
            if values[-1].starts_at_s != self.duration_s:
                raise ValueError("Interpolated wind needs an endpoint at episode end")
        elif values[-1].starts_at_s >= self.duration_s:
            raise ValueError("Synthetic weather transitions must precede episode end")
        if self.historical_base is not None and (
            self.historical_base.start_utc != self.start_utc
            or self.historical_base.duration_s != self.duration_s
        ):
            raise ValueError(
                "Perturbations and historical base must cover the same window"
            )
        if not isinstance(json.loads(self.generation_parameters_json), dict):
            raise TypeError("Weather generation parameters must be a JSON object")
        provider_ids = {value.forcing.provider_id for value in values}
        provenances = {value.forcing.provenance for value in values}
        if len(provider_ids) != 1 or len(provenances) != 1:
            raise ValueError(
                "Synthetic weather values must retain one provider and provenance"
            )
        object.__setattr__(self, "values", values)

    @property
    def end_utc(self) -> datetime:
        return self.start_utc + timedelta(seconds=self.duration_s)

    @property
    def provider_id(self) -> str:
        return self.values[0].forcing.provider_id

    @property
    def provenance(self) -> str:
        return self.values[0].forcing.provenance

    @property
    def transition_times_s(self) -> tuple[float, ...]:
        return tuple(
            value.starts_at_s
            for value in self.values[1:]
            if value.starts_at_s < self.duration_s
        )

    def utc_at_elapsed_s(self, elapsed_s: float) -> datetime:
        if not math.isfinite(elapsed_s) or not 0 <= elapsed_s <= self.duration_s:
            raise WeatherCoverageError(
                f"Elapsed time {elapsed_s!r} s is outside synthetic weather window "
                f"[0, {self.duration_s:g}] s"
            )
        return self.start_utc + timedelta(seconds=elapsed_s)

    def forcing_at_elapsed_s(self, elapsed_s: float) -> AtmosphericForcing:
        self.utc_at_elapsed_s(elapsed_s)
        starts = tuple(value.starts_at_s for value in self.values)
        index = bisect_right(starts, elapsed_s) - 1
        left = self.values[index]
        if not self.interpolate_wind or index == len(self.values) - 1:
            return left.forcing
        right = self.values[index + 1]
        fraction = (elapsed_s - left.starts_at_s) / (
            right.starts_at_s - left.starts_at_s
        )
        u, v = tuple(
            _linear(a, b, fraction)
            for a, b in zip(
                left.forcing.wind_components_m_s,
                right.forcing.wind_components_m_s,
                strict=True,
            )
        )
        precipitation = (
            self.historical_base.forcing_at_elapsed_s(elapsed_s).precipitation_mm_h
            if self.historical_base is not None
            else left.forcing.precipitation_mm_h
        )
        return _forcing_from_components(
            u, v, precipitation, self.provider_id, self.provenance
        )

    def diagnostic_at_elapsed_s(self, elapsed_s: float) -> WeatherDiagnostic:
        forcing = self.forcing_at_elapsed_s(elapsed_s)
        provenance = WeatherValueProvenance("synthetic_generated")
        if self.historical_base is not None:
            base = self.historical_base.diagnostic_at_elapsed_s(elapsed_s)
            return replace(
                base,
                kind="synthetic",
                provider_id=self.provider_id,
                provider_provenance=self.provenance,
                wind_direction_provenance=WeatherValueProvenance(
                    "synthetic_perturbation",
                    (
                        base.wind_direction_provenance.source,
                        *base.wind_direction_provenance.upstream_sources,
                    ),
                ),
                wind_speed_provenance=WeatherValueProvenance(
                    "synthetic_perturbation",
                    (
                        base.wind_speed_provenance.source,
                        *base.wind_speed_provenance.upstream_sources,
                    ),
                ),
            )
        return WeatherDiagnostic(
            timestamp_utc=self.utc_at_elapsed_s(elapsed_s),
            provider_id=forcing.provider_id,
            provider_provenance=forcing.provenance,
            kind="synthetic",
            resource_id=None,
            resource_sha256=None,
            station_id=None,
            precipitation_source_policy=None,
            temperature_c=None,
            relative_humidity_pct=None,
            temperature_provenance=provenance,
            relative_humidity_provenance=provenance,
            precipitation_provenance=provenance,
            wind_direction_provenance=provenance,
            wind_speed_provenance=provenance,
        )

    def evidence_descriptor(self) -> dict[str, object]:
        return {
            "kind": "synthetic",
            "provider_id": self.provider_id,
            "provenance": self.provenance,
            "construction_method": self.construction_method,
            "stochastic": self.stochastic,
            "seed": self.seed,
            "interpolate_wind": self.interpolate_wind,
            "generation_parameters": json.loads(self.generation_parameters_json),
            "historical_base": self.historical_base.evidence_descriptor()
            if self.historical_base is not None
            else None,
            "start_utc": self.start_utc.isoformat(),
            "end_utc": self.end_utc.isoformat(),
            "precipitation_source_policy": None,
            "values": [
                {
                    "starts_at_s": value.starts_at_s,
                    "wind_speed_m_s": value.forcing.wind_speed_m_s,
                    "wind_from_direction_deg": (value.forcing.wind_from_direction_deg),
                    "precipitation_mm_h": value.forcing.precipitation_mm_h,
                }
                for value in self.values
            ],
            "transition_times_s": list(self.transition_times_s),
        }


@dataclass(frozen=True, slots=True)
class HistoricalWeatherDataset:
    station_id: str
    provider_id: str
    provenance: str
    records: tuple[WeatherRecord, ...]
    maximum_gap: timedelta = timedelta(hours=1)
    unavailable_timestamps_utc: tuple[datetime, ...] = ()
    resource_id: str | None = None
    resource_sha256: str | None = None
    precipitation_source_policy: PrecipitationSourcePolicy | str = (
        PrecipitationSourcePolicy.STATION_ONLY
    )
    _timestamps: tuple[datetime, ...] = field(init=False, repr=False, compare=False)
    _segments: tuple[tuple[datetime, datetime], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_usable_text("station_id", self.station_id)
        _require_usable_text("provider_id", self.provider_id)
        _require_usable_text("provenance", self.provenance)
        object.__setattr__(
            self,
            "precipitation_source_policy",
            PrecipitationSourcePolicy(self.precipitation_source_policy),
        )
        if self.maximum_gap <= timedelta(0):
            raise ValueError("maximum_gap must be positive")
        records = tuple(self.records)
        if not records:
            raise ValueError(
                f"Station {self.station_id!r} has no usable weather records"
            )

        timestamps = tuple(record.timestamp_utc for record in records)
        previous: datetime | None = None
        for record in records:
            if record.station_id != self.station_id:
                raise ValueError(
                    "All weather records must belong to the dataset station"
                )
            if previous is not None and record.timestamp_utc <= previous:
                raise ValueError(
                    "Weather record timestamps must be strictly increasing"
                )
            previous = record.timestamp_utc

        unavailable = tuple(self.unavailable_timestamps_utc)
        if any(current <= previous for previous, current in pairwise(unavailable)):
            raise ValueError("Unavailable timestamps must be strictly increasing")
        for timestamp in unavailable:
            _require_aware_datetime("unavailable timestamp", timestamp)
            if timestamp.utcoffset() != timedelta(0):
                raise ValueError("Unavailable timestamps must use UTC")

        segment_start = timestamps[0]
        segments: list[tuple[datetime, datetime]] = []
        for left, right in pairwise(timestamps):
            unavailable_index = bisect_right(unavailable, left)
            crosses_unavailable = (
                unavailable_index < len(unavailable)
                and unavailable[unavailable_index] < right
            )
            if right - left > self.maximum_gap or crosses_unavailable:
                segments.append((segment_start, left))
                segment_start = right
        segments.append((segment_start, timestamps[-1]))

        object.__setattr__(self, "records", records)
        object.__setattr__(self, "unavailable_timestamps_utc", unavailable)
        object.__setattr__(self, "_timestamps", timestamps)
        object.__setattr__(self, "_segments", tuple(segments))

    @property
    def coverage_start_utc(self) -> datetime:
        return self._timestamps[0]

    @property
    def coverage_end_utc(self) -> datetime:
        return self._timestamps[-1]

    @property
    def contiguous_segments(self) -> tuple[tuple[datetime, datetime], ...]:
        return self._segments

    def record_at(
        self,
        timestamp: datetime,
    ) -> WeatherRecord:
        timestamp_utc = _as_utc(timestamp)
        if timestamp_utc < self.coverage_start_utc:
            raise self._outside_coverage_error(timestamp_utc)
        if timestamp_utc > self.coverage_end_utc:
            raise self._outside_coverage_error(timestamp_utc)

        right_index = bisect_left(self._timestamps, timestamp_utc)
        if self._timestamps[right_index] == timestamp_utc:
            return self.records[right_index]

        left_index = right_index - 1
        left = self.records[left_index]
        right = self.records[right_index]
        interval = right.timestamp_utc - left.timestamp_utc
        unavailable_index = bisect_right(
            self.unavailable_timestamps_utc,
            left.timestamp_utc,
        )
        crosses_unavailable = (
            unavailable_index < len(self.unavailable_timestamps_utc)
            and self.unavailable_timestamps_utc[unavailable_index] < right.timestamp_utc
        )
        if interval > self.maximum_gap or crosses_unavailable:
            raise WeatherCoverageError(
                f"Weather lookup at {timestamp_utc.isoformat()} crosses uncovered "
                f"interval {left.timestamp_utc.isoformat()} to "
                f"{right.timestamp_utc.isoformat()}"
            )

        fraction = (timestamp_utc - left.timestamp_utc) / interval
        continuous_provenance = _interpolated_provenance(
            left.temperature_provenance,
            right.temperature_provenance,
        )
        humidity_provenance = _interpolated_provenance(
            left.relative_humidity_provenance,
            right.relative_humidity_provenance,
        )
        wind_direction_provenance = _interpolated_provenance(
            left.wind_direction_provenance,
            right.wind_direction_provenance,
        )
        wind_speed_provenance = _interpolated_provenance(
            left.wind_speed_provenance,
            right.wind_speed_provenance,
        )
        return WeatherRecord(
            station_id=self.station_id,
            timestamp_utc=timestamp_utc,
            temperature_c=_optional_linear(
                left.temperature_c,
                right.temperature_c,
                fraction,
            ),
            relative_humidity_pct=_optional_linear(
                left.relative_humidity_pct,
                right.relative_humidity_pct,
                fraction,
            ),
            eastward_wind_m_s=_linear(
                left.eastward_wind_m_s,
                right.eastward_wind_m_s,
                fraction,
            ),
            northward_wind_m_s=_linear(
                left.northward_wind_m_s,
                right.northward_wind_m_s,
                fraction,
            ),
            precipitation_mm_h=left.precipitation_mm_h,
            temperature_provenance=continuous_provenance,
            relative_humidity_provenance=humidity_provenance,
            precipitation_provenance=left.precipitation_provenance,
            wind_direction_provenance=wind_direction_provenance,
            wind_speed_provenance=wind_speed_provenance,
        )

    def forcing_at(
        self,
        timestamp: datetime,
    ) -> AtmosphericForcing:
        record = self.record_at(timestamp)
        speed = math.hypot(record.eastward_wind_m_s, record.northward_wind_m_s)
        direction = (
            0.0
            if speed == 0
            else math.degrees(
                math.atan2(
                    -record.eastward_wind_m_s,
                    -record.northward_wind_m_s,
                )
            )
            % 360
        )
        return AtmosphericForcing(
            provider_id=self.provider_id,
            provenance=(
                f"{self.provenance}; wind_direction="
                f"{record.wind_direction_provenance.source}; wind_speed="
                f"{record.wind_speed_provenance.source}; precipitation="
                f"{record.precipitation_provenance.source}"
            ),
            wind_speed_m_s=speed,
            wind_from_direction_deg=direction,
            precipitation_mm_h=record.precipitation_mm_h,
        )

    def select_window(
        self,
        start_utc: datetime,
        duration: timedelta,
    ) -> WeatherWindow:
        start = _as_utc(start_utc)
        if duration <= timedelta(0):
            raise ValueError("Weather window duration must be positive")
        end = start + duration
        segment = self._segment_containing(start)
        if segment is None or end > segment[1]:
            raise WeatherCoverageError(
                f"Weather window {start.isoformat()} to {end.isoformat()} is not "
                "fully covered by one contiguous segment"
            )

        start_record = self.record_at(start)
        end_record = self.record_at(end)
        first_internal = bisect_right(self._timestamps, start)
        after_internal = bisect_left(self._timestamps, end)
        internal_records = self.records[first_internal:after_internal]
        records = (start_record, *internal_records, end_record)
        change_times = tuple(
            (timestamp - start).total_seconds()
            for timestamp in self._timestamps[first_internal:after_internal]
        )
        return WeatherWindow(self, start, end, records, change_times)

    def within_period(
        self, start_utc: datetime, end_utc: datetime
    ) -> HistoricalWeatherDataset:
        """Restrict records before fitting/sampling; never borrow boundary records."""
        start, end = _as_utc(start_utc), _as_utc(end_utc)
        if end <= start:
            raise ValueError("Weather period must end after its start")
        return replace(
            self,
            records=tuple(
                record for record in self.records if start <= record.timestamp_utc < end
            ),
            unavailable_timestamps_utc=tuple(
                t for t in self.unavailable_timestamps_utc if start <= t < end
            ),
        )

    def sample_window(self, duration: timedelta, *, seed: int) -> WeatherWindow:
        """Uniformly choose a covered hourly start from this bounded dataset."""
        _require_seed(seed)
        if duration <= timedelta(0):
            raise ValueError("Weather window duration must be positive")
        candidates = [
            record.timestamp_utc
            for left, right in self.contiguous_segments
            for record in self.records
            if left <= record.timestamp_utc
            and record.timestamp_utc + duration <= right
            and record.timestamp_utc.minute == 0
            and record.timestamp_utc.second == 0
            and record.timestamp_utc.microsecond == 0
        ]
        if not candidates:
            raise WeatherCoverageError(
                "No complete window fits inside the selected period"
            )
        rng = np.random.default_rng(seed)
        window = self.select_window(
            candidates[int(rng.integers(len(candidates)))], duration
        )
        return replace(
            window,
            selection_parameters_json=json.dumps(
                {
                    "method": "uniform_covered_hourly_start",
                    "seed": seed,
                    "candidate_count": len(candidates),
                    "available_start_utc": self.coverage_start_utc.isoformat(),
                    "available_end_utc": self.coverage_end_utc.isoformat(),
                },
                sort_keys=True,
            ),
        )

    def _segment_containing(
        self,
        timestamp_utc: datetime,
    ) -> tuple[datetime, datetime] | None:
        segment_starts = tuple(segment[0] for segment in self._segments)
        index = bisect_right(segment_starts, timestamp_utc) - 1
        if index < 0:
            return None
        segment = self._segments[index]
        return segment if timestamp_utc <= segment[1] else None

    def _outside_coverage_error(self, timestamp_utc: datetime) -> WeatherCoverageError:
        return WeatherCoverageError(
            f"Weather lookup {timestamp_utc.isoformat()} is outside coverage "
            f"{self.coverage_start_utc.isoformat()} to "
            f"{self.coverage_end_utc.isoformat()}"
        )


@dataclass(frozen=True, slots=True)
class CorrelatedWindModel:
    """Lima-hour means plus a stationary, bivariate hourly AR(1) residual.

    Fitted variability is atmospheric variability, not instrument uncertainty.
    Precipitation is supplied separately, never inferred from wind residuals.
    """

    hourly_mean_uv: tuple[tuple[float, float], ...]
    residual_covariance: tuple[tuple[float, float], tuple[float, float]]
    persistence: float
    maximum_speed_m_s: float
    fitting_evidence_json: str

    def __post_init__(self) -> None:
        mean = np.asarray(self.hourly_mean_uv, dtype=float)
        covariance = np.asarray(self.residual_covariance, dtype=float)
        if mean.shape != (24, 2) or not np.all(np.isfinite(mean)):
            raise ValueError("Wind model requires 24 finite vector means")
        if (
            covariance.shape != (2, 2)
            or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.T)
            or np.min(np.linalg.eigvalsh(covariance)) < -1e-12
        ):
            raise ValueError("Wind covariance must be symmetric positive semidefinite")
        if not math.isfinite(self.persistence) or not 0 <= self.persistence < 1:
            raise ValueError("Wind persistence must be in [0, 1)")
        _require_nonnegative_finite("maximum_speed_m_s", self.maximum_speed_m_s)
        object.__setattr__(
            self, "hourly_mean_uv", tuple(tuple(float(v) for v in row) for row in mean)
        )
        object.__setattr__(
            self,
            "residual_covariance",
            tuple(tuple(float(v) for v in row) for row in covariance),
        )
        if not isinstance(json.loads(self.fitting_evidence_json), dict):
            raise TypeError("Wind fitting evidence must be a JSON object")

    @classmethod
    def fit(cls, training: HistoricalWeatherDataset) -> CorrelatedWindModel:
        """Fit only the already bounded training dataset, never evaluation data."""
        records = training.records
        if any(
            r.timestamp_utc.minute
            or r.timestamp_utc.second
            or r.timestamp_utc.microsecond
            for r in records
        ):
            raise ValueError("Wind fitting requires exact hourly records")
        local_hours = np.asarray(
            [
                r.timestamp_utc.astimezone(ZoneInfo("America/Lima")).hour
                for r in records
            ],
            dtype=np.int64,
        )
        counts = np.bincount(local_hours, minlength=24)
        if np.any(counts < 2):
            raise ValueError(
                "Wind fitting needs at least two records for every local hour"
            )
        vectors = np.asarray(
            [(r.eastward_wind_m_s, r.northward_wind_m_s) for r in records]
        )
        means = np.asarray(
            [vectors[local_hours == hour].mean(axis=0) for hour in range(24)]
        )
        residuals = vectors - means[local_hours]
        covariance = residuals.T @ residuals / len(residuals)
        adjacent = np.asarray(
            [
                right.timestamp_utc - left.timestamp_utc == timedelta(hours=1)
                and training.maximum_gap >= timedelta(hours=1)
                and not any(
                    left.timestamp_utc < t < right.timestamp_utc
                    for t in training.unavailable_timestamps_utc
                )
                for left, right in pairwise(records)
            ]
        )
        if not np.any(adjacent):
            raise ValueError("Wind persistence requires adjacent covered hours")
        previous, following = residuals[:-1][adjacent], residuals[1:][adjacent]
        denominator = float(np.sum(previous**2))
        raw_persistence = (
            float(np.sum(previous * following)) / denominator
            if denominator > 0
            else 0.0
        )
        persistence = float(np.clip(raw_persistence, 0.0, 0.99))
        provenance_counts: dict[str, int] = {}
        for record in records:
            for variable, provenance in (
                ("speed", record.wind_speed_provenance),
                ("direction", record.wind_direction_provenance),
            ):
                key = f"{variable}:{provenance.source}"
                provenance_counts[key] = provenance_counts.get(key, 0) + 1
        evidence = {
            "method": "local_hour_mean_bivariate_ar1",
            "timezone": "America/Lima",
            "resource_id": training.resource_id,
            "resource_sha256": training.resource_sha256,
            "station_id": training.station_id,
            "start_utc": records[0].timestamp_utc.isoformat(),
            "end_utc": records[-1].timestamp_utc.isoformat(),
            "record_count": len(records),
            "adjacent_pairs": int(np.sum(adjacent)),
            "records_sha256": sha256(
                "\n".join(repr(r) for r in records).encode()
            ).hexdigest(),
            "counts_by_local_hour": counts.tolist(),
            "source_counts": provenance_counts,
            "raw_persistence": raw_persistence,
            "persistence_bounds": [0.0, 0.99],
            "interpretation": "atmospheric variability; not measurement error or full Lima climate validation",
        }
        return cls(
            tuple(map(tuple, means.tolist())),
            tuple(map(tuple, covariance.tolist())),
            persistence,
            float(np.max(np.linalg.norm(vectors, axis=1))),
            json.dumps(evidence, sort_keys=True),
        )

    def evidence_descriptor(self) -> dict[str, object]:
        return {
            "hourly_mean_uv": self.hourly_mean_uv,
            "residual_covariance": self.residual_covariance,
            "persistence": self.persistence,
            "maximum_speed_m_s": self.maximum_speed_m_s,
            "fitting": json.loads(self.fitting_evidence_json),
        }

    def generate(
        self,
        *,
        seed: int,
        start_utc: datetime,
        duration_s: float,
        historical_base: WeatherWindow | None = None,
        variability_scale: float = 1.0,
        precipitation_mm_h: float = 0.0,
    ) -> SyntheticWeatherWindow:
        """Generate once on clock-hour knots; lookups never draw random numbers."""
        _require_seed(seed)
        start = _as_utc(start_utc)
        _require_positive_finite("duration_s", duration_s)
        _require_nonnegative_finite("variability_scale", variability_scale)
        _require_nonnegative_finite("precipitation_mm_h", precipitation_mm_h)
        if start.minute or start.second or start.microsecond:
            raise ValueError(
                "Generated wind episodes must start on an exact clock hour"
            )
        if historical_base is not None and (
            historical_base.start_utc != start
            or historical_base.duration_s != duration_s
        ):
            raise ValueError("Historical base and generated window must match")
        if historical_base is not None and precipitation_mm_h != 0:
            raise ValueError("Perturbed weather retains the historical precipitation")
        count = math.ceil(duration_s / 3600) + 1
        rng = np.random.default_rng(seed)
        innovations = rng.multivariate_normal(
            np.zeros(2),
            self.residual_covariance,
            size=count,
            check_valid="raise",
            method="eigh",
        )
        residual = innovations[0]
        residuals = [residual]
        for innovation in innovations[1:]:
            residual = (
                self.persistence * residual
                + math.sqrt(1 - self.persistence**2) * innovation
            )
            residuals.append(residual)
        times = sorted(
            {
                0.0,
                duration_s,
                *(float(t) for t in range(3600, math.ceil(duration_s), 3600)),
                *(
                    historical_base.transition_times_s
                    if historical_base is not None
                    else ()
                ),
            }
        )
        vectors = []
        limit = self.maximum_speed_m_s
        if historical_base is not None:
            # Do not constrain the historical base to the fitting-period maximum.
            limit = max(
                limit,
                *(
                    math.hypot(r.eastward_wind_m_s, r.northward_wind_m_s)
                    for r in historical_base.records
                ),
            )
        capped = 0
        for time_s in times:
            index = int(time_s // 3600)
            fraction = time_s / 3600 - index
            residual = residuals[index]
            if fraction:
                residual = residual + fraction * (residuals[index + 1] - residual)
            if historical_base is None:
                hour = (
                    (start + timedelta(seconds=index * 3600))
                    .astimezone(ZoneInfo("America/Lima"))
                    .hour
                )
                base = np.asarray(self.hourly_mean_uv[hour])
                if fraction:
                    base = base + fraction * (
                        np.asarray(self.hourly_mean_uv[(hour + 1) % 24]) - base
                    )
            else:
                base = np.asarray(
                    historical_base.forcing_at_elapsed_s(time_s).wind_components_m_s
                )
            vector = base + variability_scale * residual
            speed = float(np.linalg.norm(vector))
            if speed > limit and not (
                historical_base is not None and variability_scale == 0
            ):
                vector *= limit / speed
                capped += 1
            vectors.append(vector)
        kind = "historical_perturbed" if historical_base is not None else "synthetic"
        provider_id = f"{kind}-wind-{seed}"
        provenance = (
            "synthetic hourly correlated wind constrained by training meteorology"
        )
        values = tuple(
            ScheduledWeatherValue(
                time_s,
                _forcing_from_components(
                    float(vector[0]),
                    float(vector[1]),
                    historical_base.forcing_at_elapsed_s(time_s).precipitation_mm_h
                    if historical_base is not None
                    else precipitation_mm_h,
                    provider_id,
                    provenance,
                ),
            )
            for time_s, vector in zip(times, vectors, strict=True)
        )
        parameters = {
            "weather_mode": kind,
            "model": self.evidence_descriptor(),
            "variability_scale": variability_scale,
            "wind_knot_interval_s": 3600,
            "speed_limit_m_s": limit,
            "speed_limited_knots": capped,
            "precipitation_policy": "historical_unchanged"
            if historical_base is not None
            else "explicit_constant_synthetic",
        }
        return SyntheticWeatherWindow(
            start,
            duration_s,
            values,
            construction_method="local_hour_mean_bivariate_ar1"
            if historical_base is None
            else "historical_plus_bivariate_ar1",
            stochastic=True,
            seed=seed,
            interpolate_wind=True,
            historical_base=historical_base,
            generation_parameters_json=json.dumps(parameters, sort_keys=True),
        )


def _forcing_from_components(
    u: float, v: float, precipitation: float, provider_id: str, provenance: str
) -> AtmosphericForcing:
    speed = math.hypot(u, v)
    direction = math.degrees(math.atan2(-u, -v)) % 360 if speed else 0.0
    return AtmosphericForcing(provider_id, provenance, speed, direction, precipitation)


def _require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")


def load_historical_weather_csv(
    path: Path | str,
    *,
    station_id: str,
    config: WeatherCsvConfig = DEFAULT_WEATHER_CSV_CONFIG,
    provider_id: str | None = None,
    provenance: str | None = None,
) -> HistoricalWeatherDataset:
    source_path = Path(path)
    _require_usable_text("station_id", station_id)
    station_id = station_id.strip()
    source_timezone = _load_timezone(config.source_timezone)
    records: list[WeatherRecord] = []
    unavailable: list[datetime] = []
    station_rows = 0

    with source_path.open(
        "r",
        newline="",
        encoding=config.encoding,
    ) as stream:
        reader = csv.DictReader(stream, delimiter=config.delimiter, strict=True)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [
            column for column in config.required_columns if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"Weather CSV is missing required columns: {', '.join(missing_columns)}"
            )

        for row in reader:
            if (row.get(config.station_column) or "").strip() != station_id:
                continue
            station_rows += 1
            line_number = reader.line_num
            timestamp_utc = _parse_source_timestamp(
                row,
                config,
                source_timezone,
                line_number,
            )
            parsed = _parse_measurements(row, config)
            if parsed is None:
                if config.missing_value_policy is MissingValuePolicy.ERROR:
                    raise ValueError(
                        f"Weather CSV line {line_number} has a missing required value "
                        f"for station {station_id!r}"
                    )
                unavailable.append(timestamp_utc)
                continue
            (
                temperature,
                humidity,
                precipitation,
                direction,
                speed,
                temperature_provenance,
                humidity_provenance,
                precipitation_provenance,
                direction_provenance,
                speed_provenance,
            ) = parsed
            eastward, northward = _wind_components(speed, direction)
            records.append(
                WeatherRecord(
                    station_id=station_id,
                    timestamp_utc=timestamp_utc,
                    temperature_c=temperature,
                    relative_humidity_pct=humidity,
                    eastward_wind_m_s=eastward,
                    northward_wind_m_s=northward,
                    precipitation_mm_h=precipitation,
                    temperature_provenance=temperature_provenance,
                    relative_humidity_provenance=humidity_provenance,
                    precipitation_provenance=precipitation_provenance,
                    wind_direction_provenance=direction_provenance,
                    wind_speed_provenance=speed_provenance,
                )
            )

    if station_rows == 0:
        raise ValueError(f"Station {station_id!r} is not present in {source_path.name}")

    return HistoricalWeatherDataset(
        station_id=station_id,
        provider_id=provider_id or f"historical-weather:{station_id.casefold()}",
        provenance=provenance
        or f"supplied R1.1 meteorological product {source_path.name}",
        records=tuple(records),
        maximum_gap=config.maximum_gap,
        unavailable_timestamps_utc=tuple(unavailable),
        resource_id=source_path.name,
        resource_sha256=sha256(source_path.read_bytes()).hexdigest(),
        precipitation_source_policy=config.precipitation_source_policy,
    )


def _parse_source_timestamp(
    row: dict[str, str | None],
    config: WeatherCsvConfig,
    source_timezone: ZoneInfo,
    line_number: int,
) -> datetime:
    date_text = (row.get(config.date_column) or "").strip()
    time_text = (row.get(config.time_column) or "").strip()
    try:
        hour_code = int(time_text)
    except ValueError as error:
        raise ValueError(
            f"Weather CSV line {line_number} has invalid HORA {time_text!r}"
        ) from error
    if hour_code not in range(0, 240_000, 10_000):
        raise ValueError(
            f"Weather CSV line {line_number} has non-hourly HORA {time_text!r}"
        )
    try:
        local_timestamp = datetime.strptime(
            f"{date_text}{hour_code // 10_000:02d}",
            "%Y%m%d%H",
        ).replace(tzinfo=source_timezone)
    except ValueError as error:
        raise ValueError(
            f"Weather CSV line {line_number} has invalid FECHA/HORA "
            f"{date_text!r}/{time_text!r}"
        ) from error
    return local_timestamp.astimezone(UTC)


def _parse_measurements(
    row: dict[str, str | None],
    config: WeatherCsvConfig,
) -> (
    tuple[
        float | None,
        float | None,
        float,
        float,
        float,
        WeatherValueProvenance,
        WeatherValueProvenance,
        WeatherValueProvenance,
        WeatherValueProvenance,
        WeatherValueProvenance,
    ]
    | None
):
    temperature = _parse_optional_float(row, config.temperature_column, config)
    humidity = _parse_optional_float(row, config.relative_humidity_column, config)
    precipitation = _parse_optional_float(row, config.precipitation_column, config)
    direction = _parse_optional_float(row, config.wind_direction_column, config)
    speed = _parse_optional_float(row, config.wind_speed_column, config)
    temperature_provenance = _parse_value_provenance(
        row,
        config.temperature_column,
    )
    humidity_provenance = _parse_value_provenance(
        row,
        config.relative_humidity_column,
    )
    precipitation_provenance = _parse_value_provenance(
        row,
        config.precipitation_column,
    )
    direction_provenance = _parse_value_provenance(
        row,
        config.wind_direction_column,
    )
    speed_provenance = _parse_value_provenance(
        row,
        config.wind_speed_column,
    )

    if (
        (
            precipitation is None
            or not _source_is_usable(precipitation_provenance.source)
        )
        and config.precipitation_source_policy
        is PrecipitationSourcePolicy.STATION_THEN_IFS
    ):
        ifs_precipitation = _parse_optional_float(
            row,
            config.ifs_precipitation_column,
            config,
        )
        if ifs_precipitation is not None:
            precipitation = ifs_precipitation
            precipitation_provenance = WeatherValueProvenance("ecmwf_ifs")

    speed_is_usable = _source_is_usable(speed_provenance.source)
    if speed == 0 and speed_is_usable:
        direction = 0.0
    wind_available = (
        speed is not None
        and speed_is_usable
        and (
            speed == 0
            or (
                direction is not None and _source_is_usable(direction_provenance.source)
            )
        )
    )
    precipitation_available = precipitation is not None and (
        precipitation_provenance.source == "ecmwf_ifs"
        or _source_is_usable(precipitation_provenance.source)
    )
    if not wind_available or not precipitation_available:
        return None

    assert precipitation is not None
    assert direction is not None
    assert speed is not None
    return (
        temperature,
        humidity,
        precipitation,
        direction,
        speed,
        temperature_provenance,
        humidity_provenance,
        precipitation_provenance,
        direction_provenance,
        speed_provenance,
    )


def _parse_optional_float(
    row: dict[str, str | None],
    column: str,
    config: WeatherCsvConfig,
) -> float | None:
    text = (row.get(column) or "").strip()
    if text.casefold() in config.missing_values:
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(
            f"Weather value {text!r} in column {column!r} is not numeric"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"Weather value in column {column!r} must be finite")
    return value


def _parse_value_provenance(
    row: dict[str, str | None],
    value_column: str,
) -> WeatherValueProvenance:
    source_column = f"{value_column}_SOURCE"
    source = (row.get(source_column) or "").strip()
    return WeatherValueProvenance(source)


def _source_is_usable(source: str) -> bool:
    return source == "senamhi_observed" or source.startswith("imputed_")


def _wind_components(
    speed_m_s: float,
    from_direction_deg: float,
) -> tuple[float, float]:
    _require_nonnegative_finite("wind speed", speed_m_s)
    if speed_m_s == 0:
        return 0.0, 0.0
    radians = math.radians(from_direction_deg)
    return -speed_m_s * math.sin(radians), -speed_m_s * math.cos(radians)


def _as_utc(value: datetime) -> datetime:
    _require_aware_datetime("timestamp", value)
    return value.astimezone(UTC)


def _load_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Unknown source_timezone {name!r}") from error


def _linear(left: float, right: float, fraction: float) -> float:
    return left + fraction * (right - left)


def _optional_linear(
    left: float | None,
    right: float | None,
    fraction: float,
) -> float | None:
    if left is None or right is None:
        return None
    return _linear(left, right, fraction)


def _interpolated_provenance(
    left: WeatherValueProvenance,
    right: WeatherValueProvenance,
) -> WeatherValueProvenance:
    upstream_sources = tuple(dict.fromkeys((left.source, right.source)))
    return WeatherValueProvenance(
        "adapter_linear_interpolation",
        upstream_sources,
    )


def _require_usable_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _require_aware_datetime(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_optional_finite(name: str, value: float | None) -> None:
    if value is not None:
        _require_finite(name, value)


def _require_nonnegative_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_positive_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
