from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

import numpy as np
from pyproj import CRS

from lima_pm25.domain import FixedStationCatalog, GridDefinition

from .trajectory import MobilePositionProvider


class SensorType(StrEnum):
    FIXED = "fixed"
    MOBILE = "mobile"


class MeasurementQuality(StrEnum):
    ACCEPTED = "accepted"
    CLIPPED = "clipped"
    DROPPED = "dropped"
    OUTSIDE_ACTIVE_GRID = "outside_active_grid"


class SamplingSupportError(ValueError):
    """Raised when a projected position has no valid active grid cell."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SensorConfig:
    sensor_id: str
    sensor_type: SensorType
    sampling_interval_s: float
    first_due_time_s: float
    provenance: str
    station_site_id: str | None = None
    mobile_position_provider_id: str | None = None
    bias_ug_m3: float = 0.0
    noise_stddev_ug_m3: float = 0.0
    dropout_probability: float = 0.0
    maximum_value_ug_m3: float | None = None

    def __post_init__(self) -> None:
        _require_usable_text("sensor_id", self.sensor_id)
        object.__setattr__(self, "sensor_type", SensorType(self.sensor_type))
        _require_positive_finite("sampling_interval_s", self.sampling_interval_s)
        _require_nonnegative_finite("first_due_time_s", self.first_due_time_s)
        _require_usable_text("provenance", self.provenance)
        _require_finite("bias_ug_m3", self.bias_ug_m3)
        _require_nonnegative_finite("noise_stddev_ug_m3", self.noise_stddev_ug_m3)
        if not math.isfinite(self.dropout_probability) or not (
            0 <= self.dropout_probability <= 1
        ):
            raise ValueError("dropout_probability must be finite and between 0 and 1")
        if self.maximum_value_ug_m3 is not None:
            _require_nonnegative_finite(
                "maximum_value_ug_m3",
                self.maximum_value_ug_m3,
            )

        if self.sensor_type is SensorType.FIXED:
            _require_usable_text("station_site_id", self.station_site_id)
            if self.mobile_position_provider_id is not None:
                raise ValueError("A fixed sensor cannot reference a mobile provider")
        else:
            _require_usable_text(
                "mobile_position_provider_id",
                self.mobile_position_provider_id,
            )
            if self.station_site_id is not None:
                raise ValueError(
                    "A mobile sensor cannot reference a fixed station site"
                )


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    sensor_id: str
    sensor_type: SensorType
    timestamp_utc: datetime
    simulation_time_s: float
    x_m: float
    y_m: float
    crs: str
    value_ug_m3: float | None
    quality_status: MeasurementQuality
    noise_stddev_ug_m3: float
    position_source_id: str
    provenance: str

    def __post_init__(self) -> None:
        _require_usable_text("sensor_id", self.sensor_id)
        object.__setattr__(self, "sensor_type", SensorType(self.sensor_type))
        _require_utc_datetime("timestamp_utc", self.timestamp_utc)
        _require_nonnegative_finite("simulation_time_s", self.simulation_time_s)
        _require_finite("x_m", self.x_m)
        _require_finite("y_m", self.y_m)
        if not CRS.from_user_input(self.crs).is_projected:
            raise ValueError("Measurement CRS must be projected")
        quality = MeasurementQuality(self.quality_status)
        object.__setattr__(self, "quality_status", quality)
        _require_nonnegative_finite("noise_stddev_ug_m3", self.noise_stddev_ug_m3)
        _require_usable_text("position_source_id", self.position_source_id)
        _require_usable_text("provenance", self.provenance)
        if quality in {
            MeasurementQuality.DROPPED,
            MeasurementQuality.OUTSIDE_ACTIVE_GRID,
        }:
            if self.value_ug_m3 is not None:
                raise ValueError(
                    "A failed measurement attempt cannot contain a numeric value"
                )
        else:
            if self.value_ug_m3 is None:
                raise ValueError("An accepted measurement must contain a numeric value")
            _require_nonnegative_finite("value_ug_m3", self.value_ug_m3)

    @property
    def is_accepted(self) -> bool:
        return self.value_ug_m3 is not None


@dataclass(slots=True)
class SensorRuntimeState:
    sensor_id: str
    next_due_time_s: float
    attempts_completed: int = 0

    def __post_init__(self) -> None:
        _require_usable_text("sensor_id", self.sensor_id)
        _require_nonnegative_finite("next_due_time_s", self.next_due_time_s)
        if self.attempts_completed < 0:
            raise ValueError("attempts_completed must be nonnegative")


@dataclass(slots=True)
class MeasurementHistory:
    _records: list[MeasurementRecord] = field(default_factory=list, repr=False)
    _keys: set[tuple[str, datetime]] = field(default_factory=set, repr=False)

    @property
    def records(self) -> tuple[MeasurementRecord, ...]:
        return tuple(self._records)

    @property
    def accepted_records(self) -> tuple[MeasurementRecord, ...]:
        return tuple(record for record in self._records if record.is_accepted)

    def extend(self, records: Sequence[MeasurementRecord]) -> None:
        additions = tuple(records)
        new_keys: set[tuple[str, datetime]] = set()
        for record in additions:
            key = (record.sensor_id, record.timestamp_utc)
            if key in self._keys or key in new_keys:
                raise ValueError(
                    f"Duplicate measurement attempt for sensor {record.sensor_id!r} "
                    f"at {record.timestamp_utc.isoformat()}"
                )
            new_keys.add(key)

        self._records.extend(additions)
        self._records.sort(key=lambda record: (record.timestamp_utc, record.sensor_id))
        self._keys.update(new_keys)


class SensingService:
    def __init__(
        self,
        grid: GridDefinition,
        sensor_configs: Sequence[SensorConfig],
        *,
        station_catalog: FixedStationCatalog | None = None,
        mobile_position_providers: Sequence[MobilePositionProvider] = (),
        rng: np.random.Generator,
    ) -> None:
        self.grid = grid
        self._station_catalog = station_catalog
        self._rng = rng
        self.history = MeasurementHistory()

        configs: dict[str, SensorConfig] = {}
        for config in sensor_configs:
            if config.sensor_id in configs:
                raise ValueError(f"Duplicate sensor_id {config.sensor_id!r}")
            configs[config.sensor_id] = config
        self._configs = configs

        providers: dict[str, MobilePositionProvider] = {}
        for provider in mobile_position_providers:
            _require_usable_text("provider_id", provider.provider_id)
            if provider.provider_id in providers:
                raise ValueError(
                    f"Duplicate mobile provider_id {provider.provider_id!r}"
                )
            if CRS.from_user_input(provider.crs) != CRS.from_user_input(grid.crs):
                raise ValueError(
                    f"Mobile provider {provider.provider_id!r} uses a different CRS"
                )
            providers[provider.provider_id] = provider
        self._mobile_position_providers = providers

        self._validate_position_references()
        self._states = {
            config.sensor_id: SensorRuntimeState(
                config.sensor_id,
                config.first_due_time_s,
            )
            for config in configs.values()
        }

    @property
    def next_due_time_s(self) -> float | None:
        if not self._states:
            return None
        return min(state.next_due_time_s for state in self._states.values())

    def state_for(self, sensor_id: str) -> SensorRuntimeState:
        try:
            return self._states[sensor_id]
        except KeyError as error:
            raise KeyError(f"Unknown sensor_id {sensor_id!r}") from error

    def sample_due(
        self,
        through_time_s: float,
        *,
        timestamp_at: Callable[[float], datetime],
        concentration_at: Callable[[float], np.ndarray],
    ) -> tuple[MeasurementRecord, ...]:
        _require_nonnegative_finite("through_time_s", through_time_s)
        produced: list[MeasurementRecord] = []

        while True:
            due = [
                (state.next_due_time_s, sensor_id)
                for sensor_id, state in self._states.items()
                if state.next_due_time_s <= through_time_s
            ]
            if not due:
                break
            due_time_s, sensor_id = min(due)
            config = self._configs[sensor_id]
            state = self._states[sensor_id]
            record = self._sample_one(
                config,
                due_time_s,
                timestamp_at=timestamp_at,
                concentration_at=concentration_at,
            )
            produced.append(record)
            state.attempts_completed += 1
            state.next_due_time_s = (
                config.first_due_time_s
                + state.attempts_completed * config.sampling_interval_s
            )

        self.history.extend(produced)
        return tuple(produced)

    def _validate_position_references(self) -> None:
        for config in self._configs.values():
            if config.sensor_type is SensorType.FIXED:
                if self._station_catalog is None:
                    raise ValueError("A fixed sensor requires a station catalog")
                site = self._station_catalog.site(config.station_site_id or "")
                if CRS.from_user_input(site.crs) != CRS.from_user_input(self.grid.crs):
                    raise ValueError(
                        f"Fixed station {site.site_id!r} uses a different CRS"
                    )
                if not self.grid.is_active_xy(site.x_m, site.y_m):
                    raise ValueError(
                        f"Fixed station {site.site_id!r} is not in an active grid cell"
                    )
            elif (
                config.mobile_position_provider_id
                not in self._mobile_position_providers
            ):
                raise ValueError(
                    "Unknown mobile position provider "
                    f"{config.mobile_position_provider_id!r}"
                )

    def _sample_one(
        self,
        config: SensorConfig,
        simulation_time_s: float,
        *,
        timestamp_at: Callable[[float], datetime],
        concentration_at: Callable[[float], np.ndarray],
    ) -> MeasurementRecord:
        timestamp_utc = timestamp_at(simulation_time_s)
        _require_utc_datetime("timestamp_at result", timestamp_utc)
        x_m, y_m, position_source_id = self._position_for(
            config,
            simulation_time_s,
        )
        if config.sensor_type is SensorType.MOBILE and not self.grid.is_active_xy(
            x_m,
            y_m,
        ):
            return MeasurementRecord(
                sensor_id=config.sensor_id,
                sensor_type=config.sensor_type,
                timestamp_utc=timestamp_utc,
                simulation_time_s=simulation_time_s,
                x_m=x_m,
                y_m=y_m,
                crs=self.grid.crs,
                value_ug_m3=None,
                quality_status=MeasurementQuality.OUTSIDE_ACTIVE_GRID,
                noise_stddev_ug_m3=config.noise_stddev_ug_m3,
                position_source_id=position_source_id,
                provenance=config.provenance,
            )
        truth = sample_cell_concentration(
            self.grid,
            concentration_at(simulation_time_s),
            x_m,
            y_m,
        )

        noise = (
            float(self._rng.normal(0.0, config.noise_stddev_ug_m3))
            if config.noise_stddev_ug_m3 > 0
            else 0.0
        )
        pre_bound = truth + config.bias_ug_m3 + noise
        dropped = (
            self._rng.random() < config.dropout_probability
            if config.dropout_probability > 0
            else False
        )
        if dropped:
            value = None
            quality = MeasurementQuality.DROPPED
        else:
            value = max(0.0, pre_bound)
            if config.maximum_value_ug_m3 is not None:
                value = min(config.maximum_value_ug_m3, value)
            quality = (
                MeasurementQuality.CLIPPED
                if value != pre_bound
                else MeasurementQuality.ACCEPTED
            )

        record = MeasurementRecord(
            sensor_id=config.sensor_id,
            sensor_type=config.sensor_type,
            timestamp_utc=timestamp_utc,
            simulation_time_s=simulation_time_s,
            x_m=x_m,
            y_m=y_m,
            crs=self.grid.crs,
            value_ug_m3=value,
            quality_status=quality,
            noise_stddev_ug_m3=config.noise_stddev_ug_m3,
            position_source_id=position_source_id,
            provenance=config.provenance,
        )
        return record

    def _position_for(
        self,
        config: SensorConfig,
        simulation_time_s: float,
    ) -> tuple[float, float, str]:
        if config.sensor_type is SensorType.FIXED:
            if self._station_catalog is None:
                raise RuntimeError("Fixed station catalog is unavailable")
            site = self._station_catalog.site(config.station_site_id or "")
            return site.x_m, site.y_m, site.site_id

        provider_id = config.mobile_position_provider_id or ""
        provider = self._mobile_position_providers[provider_id]
        x_m, y_m = provider.position_at(config.sensor_id, simulation_time_s)
        _require_finite("mobile x_m", x_m)
        _require_finite("mobile y_m", y_m)
        return float(x_m), float(y_m), provider.provider_id


def create_lima_fixed_sensor_configs(
    station_catalog: FixedStationCatalog,
    *,
    provenance: str = "baseline synthetic sensing at SENAMHI PM2.5 sites",
) -> tuple[SensorConfig, ...]:
    return tuple(
        SensorConfig(
            sensor_id=site.site_id,
            sensor_type=SensorType.FIXED,
            sampling_interval_s=3600.0,
            first_due_time_s=0.0,
            station_site_id=site.site_id,
            provenance=provenance,
        )
        for site in station_catalog.sites
    )


def sample_cell_concentration(
    grid: GridDefinition,
    concentration_ug_m3: np.ndarray,
    x_m: float,
    y_m: float,
) -> float:
    concentration = np.asarray(concentration_ug_m3, dtype=np.float64)
    if concentration.shape != grid.shape:
        raise ValueError(
            f"Concentration shape {concentration.shape} does not match grid shape "
            f"{grid.shape}"
        )
    _require_finite("x_m", x_m)
    _require_finite("y_m", y_m)
    cell = grid.xy_to_cell(x_m, y_m)
    if cell is None:
        raise SamplingSupportError("Sensor position is outside the grid bounds")
    if not grid.active_mask[cell]:
        raise SamplingSupportError("Sensor position is inside an inactive grid cell")
    value = float(concentration[cell])
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "The containing active cell must have a finite nonnegative PM2.5 "
            "concentration"
        )
    return value


def _require_utc_datetime(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_nonnegative_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_positive_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_usable_text(name: str, value: str | None) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
