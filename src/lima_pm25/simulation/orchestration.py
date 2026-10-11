from __future__ import annotations

import json
import math
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timedelta
from enum import Enum
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from itertools import pairwise
from typing import Protocol

import numpy as np

from lima_pm25.domain import (
    AtmosphericForcing,
    FixedStationCatalog,
    PM25SourceProvider,
    RegionalBoundaryForcing,
    ScenarioRunIdentity,
)

from .sensing import (
    MeasurementRecord,
    SensingService,
    SensorConfig,
    SensorType,
)
from .trajectory import MobilePositionProvider
from .transport import BoundaryMode, MassBudget, PM25State, PM25Transport
from .weather import WeatherDiagnostic

ORCHESTRATION_SCHEMA_VERSION = 1
EVENT_ORDER_VERSION = "interval-then-output-sensors-batch-v1"
_TIME_TOLERANCE_S = 1e-9


class WeatherEpisode(Protocol):
    @property
    def start_utc(self) -> datetime: ...

    @property
    def end_utc(self) -> datetime: ...

    @property
    def duration_s(self) -> float: ...

    @property
    def provider_id(self) -> str: ...

    @property
    def provenance(self) -> str: ...

    @property
    def transition_times_s(self) -> tuple[float, ...]: ...

    def forcing_at_elapsed_s(self, elapsed_s: float) -> AtmosphericForcing: ...

    def diagnostic_at_elapsed_s(
        self,
        elapsed_s: float,
    ) -> WeatherDiagnostic: ...

    def evidence_descriptor(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class EpisodeClock:
    start_utc: datetime

    def __post_init__(self) -> None:
        _require_utc("start_utc", self.start_utc)

    def utc_at(self, elapsed_s: float) -> datetime:
        _require_nonnegative_finite("elapsed_s", elapsed_s)
        return self.start_utc + timedelta(seconds=elapsed_s)

    def elapsed_at(self, timestamp_utc: datetime) -> float:
        _require_utc("timestamp_utc", timestamp_utc)
        elapsed_s = (timestamp_utc - self.start_utc).total_seconds()
        _require_nonnegative_finite("elapsed timestamp", elapsed_s)
        return elapsed_s


@dataclass(frozen=True, slots=True)
class SimulationRunConfig:
    start_utc: datetime
    duration_s: float
    concentration_output_interval_s: float | None
    seed: int
    concentration_output_times_s: tuple[float, ...] = ()
    boundary_mode: BoundaryMode | str = BoundaryMode.CLOSED
    event_order_version: str = EVENT_ORDER_VERSION

    def __post_init__(self) -> None:
        _require_utc("start_utc", self.start_utc)
        _require_positive_finite("duration_s", self.duration_s)
        if self.concentration_output_interval_s is not None:
            _require_positive_finite(
                "concentration_output_interval_s",
                self.concentration_output_interval_s,
            )
        explicit_times = tuple(self.concentration_output_times_s)
        previous = -math.inf
        for value in explicit_times:
            _require_nonnegative_finite(
                "concentration_output_times_s value",
                value,
            )
            if value <= previous:
                raise ValueError(
                    "concentration_output_times_s must be strictly increasing"
                )
            if value > self.duration_s + _TIME_TOLERANCE_S:
                raise ValueError(
                    "concentration_output_times_s cannot exceed duration_s"
                )
            previous = value
        object.__setattr__(
            self,
            "concentration_output_times_s",
            explicit_times,
        )
        if self.concentration_output_interval_s is None and not explicit_times:
            raise ValueError("Simulation outputs require an interval or explicit times")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "boundary_mode", BoundaryMode(self.boundary_mode))
        if self.event_order_version != EVENT_ORDER_VERSION:
            raise ValueError(
                f"Unsupported event_order_version {self.event_order_version!r}"
            )

    @property
    def output_times_s(self) -> tuple[float, ...]:
        times: list[float] = []
        if self.concentration_output_interval_s is not None:
            interval_s = self.concentration_output_interval_s
            count = math.floor(self.duration_s / interval_s + _TIME_TOLERANCE_S)
            times.extend(
                index * interval_s
                for index in range(count + 1)
                if index * interval_s < self.duration_s - _TIME_TOLERANCE_S
            )
        times.extend(self.concentration_output_times_s)
        times.append(self.duration_s)
        return _deduplicated_times(times)


@dataclass(frozen=True, slots=True)
class ScheduledBoundaryValue:
    starts_at_s: float
    value: RegionalBoundaryForcing

    def __post_init__(self) -> None:
        _require_nonnegative_finite("starts_at_s", self.starts_at_s)


@dataclass(frozen=True, slots=True)
class RegionalBoundarySchedule:
    values: tuple[ScheduledBoundaryValue, ...]

    def __post_init__(self) -> None:
        values = tuple(self.values)
        if not values or values[0].starts_at_s != 0:
            raise ValueError("Regional boundary must define its first value at t=0")
        for left, right in pairwise(values):
            if right.starts_at_s <= left.starts_at_s:
                raise ValueError("Regional-boundary start times must increase")
        object.__setattr__(self, "values", values)

    @classmethod
    def constant(
        cls,
        value: RegionalBoundaryForcing,
    ) -> RegionalBoundarySchedule:
        return cls((ScheduledBoundaryValue(0.0, value),))

    @property
    def transition_times_s(self) -> tuple[float, ...]:
        return tuple(value.starts_at_s for value in self.values[1:])

    def boundary_at_elapsed_s(self, elapsed_s: float) -> RegionalBoundaryForcing:
        _require_nonnegative_finite("elapsed_s", elapsed_s)
        selected = self.values[0].value
        for scheduled in self.values[1:]:
            if scheduled.starts_at_s > elapsed_s:
                break
            selected = scheduled.value
        return selected

    def evidence_descriptor(self) -> dict[str, object]:
        return {
            "values": [
                {
                    "starts_at_s": item.starts_at_s,
                    "value": asdict(item.value),
                }
                for item in self.values
            ],
            "transition_times_s": list(self.transition_times_s),
        }


@dataclass(frozen=True, slots=True)
class SimulationRunRequest:
    model: PM25Transport
    initial_state: PM25State
    weather: WeatherEpisode
    config: SimulationRunConfig
    sensor_configs: tuple[SensorConfig, ...] = ()
    station_catalog: FixedStationCatalog | None = None
    mobile_position_providers: tuple[MobilePositionProvider, ...] = ()
    sources: tuple[PM25SourceProvider, ...] = ()
    regional_boundary: RegionalBoundarySchedule | None = None
    scenario_identity: ScenarioRunIdentity | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_configs", tuple(self.sensor_configs))
        object.__setattr__(
            self,
            "mobile_position_providers",
            tuple(self.mobile_position_providers),
        )
        object.__setattr__(self, "sources", tuple(self.sources))


@dataclass(frozen=True, slots=True)
class ConcentrationFrame:
    simulation_time_s: float
    timestamp_utc: datetime
    concentration_ug_m3: np.ndarray

    def __post_init__(self) -> None:
        _require_nonnegative_finite("simulation_time_s", self.simulation_time_s)
        _require_utc("timestamp_utc", self.timestamp_utc)
        concentration = np.asarray(self.concentration_ug_m3, dtype=np.float64).copy()
        concentration.setflags(write=False)
        object.__setattr__(self, "concentration_ug_m3", concentration)


@dataclass(frozen=True, slots=True)
class MeasurementBatch:
    simulation_time_s: float
    timestamp_utc: datetime
    records: tuple[MeasurementRecord, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_finite("simulation_time_s", self.simulation_time_s)
        _require_utc("timestamp_utc", self.timestamp_utc)
        records = tuple(self.records)
        if not records:
            raise ValueError("Measurement batches cannot be empty")
        if any(not record.is_accepted for record in records):
            raise ValueError("Measurement batches contain only numeric records")
        if tuple(sorted(records, key=lambda record: record.sensor_id)) != records:
            raise ValueError("Measurement batches must be ordered by sensor_id")
        if any(record.timestamp_utc != self.timestamp_utc for record in records):
            raise ValueError("Every batch record must share its timestamp")
        object.__setattr__(self, "records", records)


@dataclass(frozen=True, slots=True)
class IntervalDiagnostic:
    start_time_s: float
    end_time_s: float
    weather: WeatherDiagnostic
    forcing: AtmosphericForcing
    regional_boundary: RegionalBoundaryForcing | None
    budget: MassBudget
    events_at_end: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SimulationRunResult:
    run_id: str
    manifest_json: str
    frames: tuple[ConcentrationFrame, ...]
    measurement_attempts: tuple[MeasurementRecord, ...]
    measurement_batches: tuple[MeasurementBatch, ...]
    diagnostics: tuple[IntervalDiagnostic, ...]
    final_state: PM25State


def run_simulation(request: SimulationRunRequest) -> SimulationRunResult:
    clock = EpisodeClock(request.config.start_utc)
    sensing = SensingService(
        request.model.grid,
        request.sensor_configs,
        station_catalog=request.station_catalog,
        mobile_position_providers=request.mobile_position_providers,
        rng=np.random.default_rng(request.config.seed),
    )
    schedules = _validate_request(request)
    manifest_base = _request_manifest(request)
    run_id = sha256(_canonical_json(manifest_base).encode("utf-8")).hexdigest()
    manifest = {**manifest_base, "run_id": run_id}

    frames: list[ConcentrationFrame] = []
    batches: list[MeasurementBatch] = []
    diagnostics: list[IntervalDiagnostic] = []
    state = request.initial_state

    _record_frame_if_due(
        frames,
        state,
        clock,
        request.config.output_times_s,
    )
    initial_attempts = sensing.sample_due(
        0.0,
        timestamp_at=clock.utc_at,
        concentration_at=lambda _time_s: state.concentration_ug_m3,
    )
    _append_batch(batches, initial_attempts)

    current_time_s = 0.0
    for end_time_s in schedules["event_times"]:
        forcing = request.weather.forcing_at_elapsed_s(current_time_s)
        weather_diagnostic = request.weather.diagnostic_at_elapsed_s(current_time_s)
        regional_boundary = (
            request.regional_boundary.boundary_at_elapsed_s(current_time_s)
            if request.regional_boundary is not None
            else None
        )
        state, budget = request.model.step(
            state,
            duration_s=end_time_s - current_time_s,
            forcing=forcing,
            regional_boundary=regional_boundary,
            sources=request.sources,
            boundary_mode=request.config.boundary_mode,
        )
        events = _events_at(end_time_s, schedules)
        diagnostics.append(
            IntervalDiagnostic(
                start_time_s=current_time_s,
                end_time_s=end_time_s,
                weather=weather_diagnostic,
                forcing=forcing,
                regional_boundary=regional_boundary,
                budget=budget,
                events_at_end=events,
            )
        )
        _record_frame_if_due(
            frames,
            state,
            clock,
            request.config.output_times_s,
        )
        attempts = sensing.sample_due(
            end_time_s,
            timestamp_at=clock.utc_at,
            concentration_at=lambda _time_s, state=state: state.concentration_ug_m3,
        )
        _append_batch(batches, attempts)
        current_time_s = end_time_s

    return SimulationRunResult(
        run_id=run_id,
        manifest_json=json.dumps(
            _jsonable(manifest),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        frames=tuple(frames),
        measurement_attempts=sensing.history.records,
        measurement_batches=tuple(batches),
        diagnostics=tuple(diagnostics),
        final_state=state,
    )


def simulation_run_request_descriptor(
    request: SimulationRunRequest,
) -> dict[str, object]:
    """Return the canonical public descriptor used to identify one run request."""
    return json.loads(_canonical_json(_request_manifest(request)))


def _validate_request(
    request: SimulationRunRequest,
) -> dict[str, tuple[float, ...]]:
    config = request.config
    if request.initial_state.simulation_time_s != 0:
        raise ValueError("Complete episodes must start from simulation_time_s=0")
    if request.initial_state.grid is not request.model.grid:
        raise ValueError("Initial state and transport model must share one grid")
    if request.weather.start_utc != config.start_utc:
        raise ValueError("Weather window and run must share the same UTC start")
    if not math.isclose(
        request.weather.duration_s,
        config.duration_s,
        rel_tol=0,
        abs_tol=_TIME_TOLERANCE_S,
    ):
        raise ValueError("Weather window must cover the complete run duration")
    if request.weather.end_utc != config.start_utc + timedelta(
        seconds=config.duration_s
    ):
        raise ValueError("Weather window end does not match the run end")

    request.weather.forcing_at_elapsed_s(0.0)
    request.weather.forcing_at_elapsed_s(config.duration_s)
    request.weather.diagnostic_at_elapsed_s(0.0)
    request.weather.diagnostic_at_elapsed_s(config.duration_s)
    weather_times = _validated_transition_times(
        "weather",
        request.weather.transition_times_s,
        config.duration_s,
    )

    if config.boundary_mode is BoundaryMode.REGIONAL_INFLOW:
        if request.regional_boundary is None:
            raise ValueError("regional_boundary is required for regional-inflow runs")
    elif request.regional_boundary is not None:
        raise ValueError("Closed-boundary runs cannot configure regional inflow")
    boundary_times = _validated_transition_times(
        "regional boundary",
        (
            request.regional_boundary.transition_times_s
            if request.regional_boundary is not None
            else ()
        ),
        config.duration_s,
    )

    source_times: list[float] = []
    source_ids: set[str] = set()
    for source in request.sources:
        if source.source_id in source_ids:
            raise ValueError(f"Duplicate source_id {source.source_id!r}")
        source_ids.add(source.source_id)
        try:
            transitions = source.transition_times_s
            descriptor = source.evidence_descriptor()
        except AttributeError as error:
            raise ValueError(
                f"Source {source.source_id!r} must expose transitions and evidence"
            ) from error
        if not isinstance(descriptor, dict):
            # Provider validation consistently reports ValueError to callers.
            raise ValueError("Source evidence_descriptor must return a mapping")  # noqa: TRY004
        source_times.extend(
            _validated_transition_times(
                f"source {source.source_id}",
                transitions,
                config.duration_s,
            )
        )

    provider_by_id = {
        provider.provider_id: provider for provider in request.mobile_position_providers
    }
    if len(provider_by_id) != len(request.mobile_position_providers):
        raise ValueError("Mobile replay provider IDs must be unique")
    sensor_due_times: list[float] = []
    for sensor in request.sensor_configs:
        due_times = _sensor_due_times(sensor, config.duration_s)
        sensor_due_times.extend(due_times)
        if sensor.sensor_type is SensorType.MOBILE:
            provider_id = sensor.mobile_position_provider_id or ""
            provider = provider_by_id.get(provider_id)
            if provider is None:
                raise ValueError(f"Unknown mobile replay provider {provider_id!r}")
            validator = getattr(provider, "validate_due_times", None)
            if not callable(validator):
                raise ValueError(
                    f"Mobile provider {provider_id!r} is not a replay provider "
                    "with preflight due-time validation"
                )
            validator(sensor.sensor_id, due_times)
            descriptor = getattr(provider, "evidence_descriptor", None)
            if not callable(descriptor) or not isinstance(descriptor(), dict):
                raise ValueError(
                    f"Mobile provider {provider_id!r} must expose replay evidence"
                )

    output_times = config.output_times_s
    event_times = tuple(
        sorted(
            {
                *weather_times,
                *boundary_times,
                *source_times,
                *sensor_due_times,
                *output_times,
                config.duration_s,
            }
            - {0.0}
        )
    )
    return {
        "weather": weather_times,
        "boundary": boundary_times,
        "source": tuple(sorted(set(source_times))),
        "sensor": tuple(sorted(set(sensor_due_times))),
        "output": output_times,
        "event_times": event_times,
    }


def _sensor_due_times(sensor: SensorConfig, duration_s: float) -> tuple[float, ...]:
    if sensor.first_due_time_s > duration_s + _TIME_TOLERANCE_S:
        return ()
    count = math.floor(
        (duration_s - sensor.first_due_time_s) / sensor.sampling_interval_s
        + _TIME_TOLERANCE_S
    )
    return tuple(
        sensor.first_due_time_s + index * sensor.sampling_interval_s
        for index in range(count + 1)
    )


def _validated_transition_times(
    name: str,
    values: Sequence[float],
    duration_s: float,
) -> tuple[float, ...]:
    result = tuple(values)
    previous = -math.inf
    for value in result:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} transition times must be nonnegative and finite")
        if value <= previous:
            raise ValueError(f"{name} transition times must be strictly increasing")
        previous = value
    return tuple(value for value in result if 0 < value < duration_s)


def _events_at(
    elapsed_s: float,
    schedules: dict[str, tuple[float, ...]],
) -> tuple[str, ...]:
    events = [
        name
        for name in ("weather", "boundary", "source", "output", "sensor")
        if _time_is_due(elapsed_s, schedules[name])
    ]
    if _time_is_due(elapsed_s, (schedules["event_times"][-1],)):
        events.append("episode_end")
    return tuple(events)


def _record_frame_if_due(
    frames: list[ConcentrationFrame],
    state: PM25State,
    clock: EpisodeClock,
    output_times_s: Sequence[float],
) -> None:
    if not _time_is_due(state.simulation_time_s, output_times_s):
        return
    frames.append(
        ConcentrationFrame(
            simulation_time_s=state.simulation_time_s,
            timestamp_utc=clock.utc_at(state.simulation_time_s),
            concentration_ug_m3=state.concentration_ug_m3,
        )
    )


def _append_batch(
    batches: list[MeasurementBatch],
    attempts: Sequence[MeasurementRecord],
) -> None:
    accepted = tuple(record for record in attempts if record.is_accepted)
    if not accepted:
        return
    by_timestamp: dict[datetime, list[MeasurementRecord]] = {}
    for record in accepted:
        by_timestamp.setdefault(record.timestamp_utc, []).append(record)
    for timestamp, records in sorted(by_timestamp.items()):
        ordered = tuple(sorted(records, key=lambda record: record.sensor_id))
        batches.append(
            MeasurementBatch(
                simulation_time_s=ordered[0].simulation_time_s,
                timestamp_utc=timestamp,
                records=ordered,
            )
        )


def _request_manifest(request: SimulationRunRequest) -> dict[str, object]:
    try:
        package_version = version("lima-pm25-core")
    except PackageNotFoundError:
        package_version = "unknown"
    grid = request.model.grid
    grid_digest = sha256()
    grid_digest.update(grid.crs.encode("utf-8"))
    for array in (
        grid.domain_bounds_m,
        grid.grid_bounds_m,
        grid.x_centers_m,
        grid.y_centers_m,
        grid.active_mask,
    ):
        grid_digest.update(np.ascontiguousarray(array).tobytes())
    initial_digest = sha256(
        np.ascontiguousarray(
            request.initial_state.concentration_ug_m3,
        ).tobytes()
    ).hexdigest()
    mobile_evidence = []
    for provider in request.mobile_position_providers:
        descriptor = getattr(provider, "evidence_descriptor", None)
        if callable(descriptor):
            mobile_evidence.append(descriptor())

    return {
        "schema_version": ORCHESTRATION_SCHEMA_VERSION,
        "event_order_version": request.config.event_order_version,
        "software": {
            "package": "lima-pm25-core",
            "package_version": package_version,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "scenario_identity": request.scenario_identity,
        "run_config": {
            "start_utc": request.config.start_utc,
            "duration_s": request.config.duration_s,
            "concentration_output_interval_s": (
                request.config.concentration_output_interval_s
            ),
            "concentration_output_times_s": (
                request.config.concentration_output_times_s
            ),
            "output_times_s": request.config.output_times_s,
            "seed": request.config.seed,
            "boundary_mode": request.config.boundary_mode,
        },
        "grid": {
            "crs": grid.crs,
            "shape": grid.shape,
            "cell_size_m": grid.cell_size_m,
            "domain_bounds_m": grid.domain_bounds_m,
            "grid_bounds_m": grid.grid_bounds_m,
            "grid_sha256": grid_digest.hexdigest(),
        },
        "physics": request.model.physics,
        "initial_state": {
            "concentration_sha256": initial_digest,
            "ambient_baseline": request.initial_state.ambient_baseline,
        },
        "weather": request.weather.evidence_descriptor(),
        "regional_boundary": (
            request.regional_boundary.evidence_descriptor()
            if request.regional_boundary is not None
            else None
        ),
        "sources": [source.evidence_descriptor() for source in request.sources],
        "sensors": list(request.sensor_configs),
        "fixed_station_sites": (
            list(request.station_catalog.sites)
            if request.station_catalog is not None
            else []
        ),
        "mobile_replay_providers": mobile_evidence,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        separators=(",", ":"),
        sort_keys=True,
    )


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
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _time_is_due(elapsed_s: float, times_s: Sequence[float]) -> bool:
    return any(
        math.isclose(elapsed_s, time_s, rel_tol=0, abs_tol=_TIME_TOLERANCE_S)
        for time_s in times_s
    )


def _deduplicated_times(values: Sequence[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in sorted(values):
        if not result or not math.isclose(
            value,
            result[-1],
            rel_tol=0,
            abs_tol=_TIME_TOLERANCE_S,
        ):
            result.append(float(value))
    return tuple(result)


def _require_utc(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")


def _require_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
