from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from lima_pm25.domain import GridDefinition
from lima_pm25.simulation import (
    MeasurementBatch,
    MeasurementRecord,
    SensorType,
)

from .reconstruction import (
    DEFAULT_PREDICTION_CHUNK_SIZE,
    GaussianProcessFieldReconstructor,
    PosteriorField,
    ReconstructionConfig,
    ReconstructionDiagnostics,
    ReconstructionObservationMetadata,
    ReconstructionRequest,
    select_reconstruction_observations,
)

if TYPE_CHECKING:
    from .experiment import Change10EvaluationProfile


MeasurementKey = tuple[str, datetime]


class SequentialUpdateError(ValueError):
    """Raised when a measurement batch violates the sequential contract."""


class MeasurementDisposition(StrEnum):
    INCORPORATED = "incorporated"
    POLICY_EXCLUDED = "policy_excluded"


@dataclass(frozen=True, slots=True, kw_only=True)
class SequentialUpdateConfig:
    reconstruction_config: ReconstructionConfig
    history_duration: timedelta
    variance_numerical_tolerance_ug_m3_squared: float
    profile_sha256: str
    profile_status: str
    profile_source: str
    calibration_artifact_sha256: str
    calibration_artifact_source: str
    prediction_chunk_size: int = DEFAULT_PREDICTION_CHUNK_SIZE

    def __post_init__(self) -> None:
        if not isinstance(self.reconstruction_config, ReconstructionConfig):
            raise TypeError("reconstruction_config must be a ReconstructionConfig")
        if not isinstance(self.history_duration, timedelta) or (
            self.history_duration <= timedelta(0)
        ):
            raise ValueError("history_duration must be a positive timedelta")
        tolerance = self.variance_numerical_tolerance_ug_m3_squared
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                "variance_numerical_tolerance_ug_m3_squared must be finite and "
                "nonnegative"
            )
        for name, value in (
            ("profile_sha256", self.profile_sha256),
            ("profile_status", self.profile_status),
            ("profile_source", self.profile_source),
            ("calibration_artifact_sha256", self.calibration_artifact_sha256),
            ("calibration_artifact_source", self.calibration_artifact_source),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        if (
            isinstance(self.prediction_chunk_size, bool)
            or not isinstance(self.prediction_chunk_size, int)
            or self.prediction_chunk_size <= 0
        ):
            raise ValueError("prediction_chunk_size must be a positive integer")

    @classmethod
    def from_change10_profile(
        cls,
        profile: Change10EvaluationProfile,
    ) -> SequentialUpdateConfig:
        calibration_artifact = profile.calibration_artifact_path
        return cls(
            reconstruction_config=profile.reconstruction_config,
            history_duration=profile.history_duration,
            variance_numerical_tolerance_ug_m3_squared=(
                profile.uncertainty_numerical_tolerance
            ),
            profile_sha256=profile.source_sha256,
            profile_status=profile.status,
            profile_source=str(profile.source_path),
            calibration_artifact_sha256=sha256(
                calibration_artifact.read_bytes()
            ).hexdigest(),
            calibration_artifact_source=str(calibration_artifact),
        )


@dataclass(frozen=True, slots=True)
class ProcessedMeasurement:
    sensor_id: str
    timestamp_utc: datetime
    sensor_type: SensorType
    disposition: MeasurementDisposition

    @property
    def key(self) -> MeasurementKey:
        return self.sensor_id, self.timestamp_utc


@dataclass(frozen=True, slots=True)
class LocalVarianceReduction:
    measurement_key: MeasurementKey
    sensor_type: SensorType
    row: int
    column: int
    variance_before_ug_m3_squared: float
    variance_after_ug_m3_squared: float
    reduction_ug_m3_squared: float


@dataclass(frozen=True, slots=True)
class UncertaintyReductionDiagnostic:
    prediction_timestamp_utc: datetime
    local: tuple[LocalVarianceReduction, ...]
    integrated_variance_before_ug2_m4: float
    integrated_variance_after_ug2_m4: float
    integrated_reduction_ug2_m4: float
    local_tolerance_ug_m3_squared: float
    integrated_tolerance_ug2_m4: float
    local_nonincrease: bool
    integrated_nonincrease: bool

    def __post_init__(self) -> None:
        _require_utc("prediction_timestamp_utc", self.prediction_timestamp_utc)
        if not self.local:
            raise ValueError("uncertainty diagnostics require local observations")
        numeric_values = (
            self.integrated_variance_before_ug2_m4,
            self.integrated_variance_after_ug2_m4,
            self.integrated_reduction_ug2_m4,
            self.local_tolerance_ug_m3_squared,
            self.integrated_tolerance_ug2_m4,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("uncertainty diagnostics must be finite")
        if self.local_tolerance_ug_m3_squared < 0 or (
            self.integrated_tolerance_ug2_m4 < 0
        ):
            raise ValueError("uncertainty tolerances must be nonnegative")

    @property
    def passes_nonincrease(self) -> bool:
        return self.local_nonincrease and self.integrated_nonincrease

    @property
    def has_mobile_measurement(self) -> bool:
        return any(item.sensor_type is SensorType.MOBILE for item in self.local)


@dataclass(frozen=True, slots=True)
class SequentialMapSnapshot:
    sequence_number: int
    reconstruction_timestamp_utc: datetime
    measurement_cutoff_utc: datetime
    grid: GridDefinition
    grid_sha256: str
    mean_ug_m3: np.ndarray
    standard_deviation_ug_m3: np.ndarray
    observations: ReconstructionObservationMetadata
    estimator_diagnostics: ReconstructionDiagnostics
    uncertainty_reduction: UncertaintyReductionDiagnostic
    incorporated_measurement_keys: tuple[MeasurementKey, ...]

    def __post_init__(self) -> None:
        if self.sequence_number <= 0:
            raise ValueError("sequence_number must be positive")
        _require_utc(
            "reconstruction_timestamp_utc",
            self.reconstruction_timestamp_utc,
        )
        _require_utc("measurement_cutoff_utc", self.measurement_cutoff_utc)
        if self.reconstruction_timestamp_utc != self.measurement_cutoff_utc:
            raise ValueError(
                "sequential reconstruction timestamp and cutoff must match"
            )
        if not isinstance(self.grid, GridDefinition):
            raise TypeError("grid must be a GridDefinition")
        if self.grid_sha256 != grid_sha256(self.grid):
            raise ValueError("grid_sha256 must identify the snapshot grid")
        mean = _immutable_array(self.mean_ug_m3, np.float64)
        standard_deviation = _immutable_array(
            self.standard_deviation_ug_m3,
            np.float64,
        )
        if mean.shape != self.grid.shape or standard_deviation.shape != self.grid.shape:
            raise ValueError("snapshot arrays must match the grid shape")
        mask = self.grid.active_mask
        if not np.all(np.isfinite(mean[mask])) or not np.all(
            np.isfinite(standard_deviation[mask])
        ):
            raise ValueError("active snapshot values must be finite")
        if np.any(standard_deviation[mask] < 0):
            raise ValueError("active posterior standard deviation must be nonnegative")
        if np.any(~np.isnan(mean[~mask])) or np.any(
            ~np.isnan(standard_deviation[~mask])
        ):
            raise ValueError("inactive snapshot cells must contain NaN")
        if (
            self.uncertainty_reduction.prediction_timestamp_utc
            != self.reconstruction_timestamp_utc
        ):
            raise ValueError("uncertainty diagnostic timestamp must match snapshot")
        if not self.incorporated_measurement_keys:
            raise ValueError("an applied snapshot requires incorporated measurements")
        object.__setattr__(self, "mean_ug_m3", mean)
        object.__setattr__(
            self,
            "standard_deviation_ug_m3",
            standard_deviation,
        )

    @property
    def active_mask(self) -> np.ndarray:
        return self.grid.active_mask


@dataclass(frozen=True, slots=True)
class SequentialUpdateLogEntry:
    processed_timestamp_utc: datetime
    applied: bool
    sequence_number: int | None
    incorporated_measurement_keys: tuple[MeasurementKey, ...]
    policy_excluded_measurement_keys: tuple[MeasurementKey, ...]
    eligible_mobile_measurement_keys: tuple[MeasurementKey, ...]
    diagnostic_passes_nonincrease: bool | None


@dataclass(frozen=True, slots=True)
class SequentialUpdateResult:
    applied: bool
    snapshot: SequentialMapSnapshot | None
    log_entry: SequentialUpdateLogEntry


@dataclass(frozen=True, slots=True)
class SequentialUpdateTrace:
    results: tuple[SequentialUpdateResult, ...]

    @property
    def snapshots(self) -> tuple[SequentialMapSnapshot, ...]:
        return tuple(
            result.snapshot for result in self.results if result.snapshot is not None
        )

    @property
    def log(self) -> tuple[SequentialUpdateLogEntry, ...]:
        return tuple(result.log_entry for result in self.results)


class _Reconstructor(Protocol):
    def reconstruct(self, request: ReconstructionRequest) -> PosteriorField: ...


@dataclass(frozen=True, slots=True)
class _UpdaterState:
    latest_processed_timestamp_utc: datetime | None = None
    sequence_number: int = 0
    current_snapshot: SequentialMapSnapshot | None = None
    active_incorporated_records: tuple[MeasurementRecord, ...] = ()
    processed_measurements: tuple[ProcessedMeasurement, ...] = ()
    processed_keys: frozenset[MeasurementKey] = frozenset()
    update_log: tuple[SequentialUpdateLogEntry, ...] = ()


class SequentialMapUpdater:
    def __init__(
        self,
        grid: GridDefinition,
        config: SequentialUpdateConfig,
        *,
        reconstructor: _Reconstructor | None = None,
    ) -> None:
        if not isinstance(grid, GridDefinition):
            raise TypeError("grid must be a GridDefinition")
        if not isinstance(config, SequentialUpdateConfig):
            raise TypeError("config must be a SequentialUpdateConfig")
        self.grid = grid
        self.config = config
        self._grid_sha256 = grid_sha256(grid)
        self._reconstructor = reconstructor or GaussianProcessFieldReconstructor(
            config.reconstruction_config,
            prediction_chunk_size=config.prediction_chunk_size,
        )
        self._state = _UpdaterState()

    @property
    def current_snapshot(self) -> SequentialMapSnapshot | None:
        return self._state.current_snapshot

    @property
    def latest_processed_timestamp_utc(self) -> datetime | None:
        return self._state.latest_processed_timestamp_utc

    @property
    def processed_measurements(self) -> tuple[ProcessedMeasurement, ...]:
        return self._state.processed_measurements

    @property
    def update_log(self) -> tuple[SequentialUpdateLogEntry, ...]:
        return self._state.update_log

    def update(self, batch: MeasurementBatch) -> SequentialUpdateResult:
        if not isinstance(batch, MeasurementBatch):
            raise TypeError("batch must be a MeasurementBatch")
        self._validate_batch_order_and_keys(batch)

        batch_request = ReconstructionRequest(
            grid=self.grid,
            reconstruction_timestamp_utc=batch.timestamp_utc,
            measurements=batch.records,
        )
        eligible, _ = select_reconstruction_observations(
            batch_request,
            self.config.reconstruction_config,
        )
        eligible_keys = frozenset(_measurement_key(record) for record in eligible)
        excluded = tuple(
            record
            for record in batch.records
            if _measurement_key(record) not in eligible_keys
        )

        if not eligible:
            return self._commit_policy_noop(batch, excluded)

        history_start = batch.timestamp_utc - self.config.history_duration
        previous_records = tuple(
            record
            for record in self._state.active_incorporated_records
            if history_start <= record.timestamp_utc <= batch.timestamp_utc
        )
        updated_records = (*previous_records, *eligible)
        before = self._reconstructor.reconstruct(
            ReconstructionRequest(
                grid=self.grid,
                reconstruction_timestamp_utc=batch.timestamp_utc,
                measurements=previous_records,
            )
        )
        after = self._reconstructor.reconstruct(
            ReconstructionRequest(
                grid=self.grid,
                reconstruction_timestamp_utc=batch.timestamp_utc,
                measurements=updated_records,
            )
        )
        _validate_posterior(before, self.grid, batch.timestamp_utc, previous_records)
        _validate_posterior(after, self.grid, batch.timestamp_utc, updated_records)
        diagnostic = _uncertainty_reduction(
            grid=self.grid,
            before=before,
            after=after,
            new_records=eligible,
            local_tolerance=(self.config.variance_numerical_tolerance_ug_m3_squared),
        )

        sequence_number = self._state.sequence_number + 1
        incorporated_keys = tuple(_measurement_key(record) for record in eligible)
        excluded_keys = tuple(_measurement_key(record) for record in excluded)
        snapshot = SequentialMapSnapshot(
            sequence_number=sequence_number,
            reconstruction_timestamp_utc=after.reconstruction_timestamp_utc,
            measurement_cutoff_utc=batch.timestamp_utc,
            grid=self.grid,
            grid_sha256=self._grid_sha256,
            mean_ug_m3=after.mean_ug_m3,
            standard_deviation_ug_m3=after.standard_deviation_ug_m3,
            observations=after.observations,
            estimator_diagnostics=after.diagnostics,
            uncertainty_reduction=diagnostic,
            incorporated_measurement_keys=incorporated_keys,
        )
        log_entry = SequentialUpdateLogEntry(
            processed_timestamp_utc=batch.timestamp_utc,
            applied=True,
            sequence_number=sequence_number,
            incorporated_measurement_keys=incorporated_keys,
            policy_excluded_measurement_keys=excluded_keys,
            eligible_mobile_measurement_keys=tuple(
                _measurement_key(record)
                for record in eligible
                if record.sensor_type is SensorType.MOBILE
            ),
            diagnostic_passes_nonincrease=diagnostic.passes_nonincrease,
        )
        dispositions = _processed_measurements(eligible, excluded)
        candidate_state = _UpdaterState(
            latest_processed_timestamp_utc=batch.timestamp_utc,
            sequence_number=sequence_number,
            current_snapshot=snapshot,
            active_incorporated_records=tuple(updated_records),
            processed_measurements=(
                *self._state.processed_measurements,
                *dispositions,
            ),
            processed_keys=self._state.processed_keys.union(
                _measurement_key(record) for record in batch.records
            ),
            update_log=(*self._state.update_log, log_entry),
        )
        self._state = candidate_state
        return SequentialUpdateResult(True, snapshot, log_entry)

    def _validate_batch_order_and_keys(self, batch: MeasurementBatch) -> None:
        latest = self._state.latest_processed_timestamp_utc
        if latest is not None and batch.timestamp_utc <= latest:
            raise SequentialUpdateError(
                "Batch timestamp must be strictly later than the latest processed "
                "timestamp"
            )
        keys = tuple(_measurement_key(record) for record in batch.records)
        if len(set(keys)) != len(keys):
            raise SequentialUpdateError("Batch contains duplicate measurement keys")
        duplicates = self._state.processed_keys.intersection(keys)
        if duplicates:
            raise SequentialUpdateError(
                "Batch contains an already processed measurement key"
            )

    def _commit_policy_noop(
        self,
        batch: MeasurementBatch,
        excluded: tuple[MeasurementRecord, ...],
    ) -> SequentialUpdateResult:
        excluded_keys = tuple(_measurement_key(record) for record in excluded)
        log_entry = SequentialUpdateLogEntry(
            processed_timestamp_utc=batch.timestamp_utc,
            applied=False,
            sequence_number=None,
            incorporated_measurement_keys=(),
            policy_excluded_measurement_keys=excluded_keys,
            eligible_mobile_measurement_keys=(),
            diagnostic_passes_nonincrease=None,
        )
        dispositions = tuple(
            ProcessedMeasurement(
                record.sensor_id,
                record.timestamp_utc,
                record.sensor_type,
                MeasurementDisposition.POLICY_EXCLUDED,
            )
            for record in excluded
        )
        self._state = _UpdaterState(
            latest_processed_timestamp_utc=batch.timestamp_utc,
            sequence_number=self._state.sequence_number,
            current_snapshot=self._state.current_snapshot,
            active_incorporated_records=(self._state.active_incorporated_records),
            processed_measurements=(
                *self._state.processed_measurements,
                *dispositions,
            ),
            processed_keys=self._state.processed_keys.union(excluded_keys),
            update_log=(*self._state.update_log, log_entry),
        )
        return SequentialUpdateResult(False, None, log_entry)


def process_measurement_batches(
    updater: SequentialMapUpdater,
    batches: Sequence[MeasurementBatch],
) -> SequentialUpdateTrace:
    if not isinstance(updater, SequentialMapUpdater):
        raise TypeError("updater must be a SequentialMapUpdater")
    return SequentialUpdateTrace(tuple(updater.update(batch) for batch in batches))


def grid_sha256(grid: GridDefinition) -> str:
    digest = sha256()
    digest.update(grid.crs.encode("utf-8"))
    digest.update(np.float64(grid.cell_size_m).tobytes())
    for array in (
        grid.domain_bounds_m,
        grid.grid_bounds_m,
        grid.x_centers_m,
        grid.y_centers_m,
        grid.active_mask,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _processed_measurements(
    incorporated: Sequence[MeasurementRecord],
    excluded: Sequence[MeasurementRecord],
) -> tuple[ProcessedMeasurement, ...]:
    dispositions = [
        ProcessedMeasurement(
            record.sensor_id,
            record.timestamp_utc,
            record.sensor_type,
            MeasurementDisposition.INCORPORATED,
        )
        for record in incorporated
    ]
    dispositions.extend(
        ProcessedMeasurement(
            record.sensor_id,
            record.timestamp_utc,
            record.sensor_type,
            MeasurementDisposition.POLICY_EXCLUDED,
        )
        for record in excluded
    )
    return tuple(sorted(dispositions, key=lambda item: item.sensor_id))


def _validate_posterior(
    posterior: PosteriorField,
    grid: GridDefinition,
    timestamp: datetime,
    expected_records: Sequence[MeasurementRecord],
) -> None:
    if not isinstance(posterior, PosteriorField):
        raise TypeError("reconstructor must return PosteriorField")
    if posterior.reconstruction_timestamp_utc != timestamp:
        raise ValueError("posterior timestamp does not match the incoming batch")
    if posterior.mean_ug_m3.shape != grid.shape or (
        posterior.standard_deviation_ug_m3.shape != grid.shape
    ):
        raise ValueError("posterior arrays do not match the configured grid")
    if not np.array_equal(posterior.active_mask, grid.active_mask):
        raise ValueError("posterior active mask does not match the configured grid")
    expected_keys = tuple(
        _measurement_key(record)
        for record in sorted(
            expected_records,
            key=lambda record: (record.timestamp_utc, record.sensor_id),
        )
    )
    if posterior.observations.used_measurement_keys != expected_keys:
        raise ValueError("posterior observation keys do not match requested history")


def _uncertainty_reduction(
    *,
    grid: GridDefinition,
    before: PosteriorField,
    after: PosteriorField,
    new_records: Sequence[MeasurementRecord],
    local_tolerance: float,
) -> UncertaintyReductionDiagnostic:
    variance_before = before.standard_deviation_ug_m3**2
    variance_after = after.standard_deviation_ug_m3**2
    local = []
    for record in new_records:
        cell = grid.xy_to_cell(record.x_m, record.y_m)
        if cell is None or not grid.active_mask[cell]:
            raise ValueError("An incorporated record lacks active grid support")
        before_value = float(variance_before[cell])
        after_value = float(variance_after[cell])
        local.append(
            LocalVarianceReduction(
                measurement_key=_measurement_key(record),
                sensor_type=record.sensor_type,
                row=cell[0],
                column=cell[1],
                variance_before_ug_m3_squared=before_value,
                variance_after_ug_m3_squared=after_value,
                reduction_ug_m3_squared=before_value - after_value,
            )
        )
    active_area_m2 = float(np.count_nonzero(grid.active_mask) * grid.cell_area_m2)
    integrated_tolerance = local_tolerance * active_area_m2
    integrated_before = float(
        np.sum(variance_before[grid.active_mask]) * grid.cell_area_m2
    )
    integrated_after = float(
        np.sum(variance_after[grid.active_mask]) * grid.cell_area_m2
    )
    return UncertaintyReductionDiagnostic(
        prediction_timestamp_utc=after.reconstruction_timestamp_utc,
        local=tuple(local),
        integrated_variance_before_ug2_m4=integrated_before,
        integrated_variance_after_ug2_m4=integrated_after,
        integrated_reduction_ug2_m4=integrated_before - integrated_after,
        local_tolerance_ug_m3_squared=local_tolerance,
        integrated_tolerance_ug2_m4=integrated_tolerance,
        local_nonincrease=all(
            item.variance_after_ug_m3_squared
            <= item.variance_before_ug_m3_squared + local_tolerance
            for item in local
        ),
        integrated_nonincrease=(
            integrated_after <= integrated_before + integrated_tolerance
        ),
    )


def _measurement_key(record: MeasurementRecord) -> MeasurementKey:
    return record.sensor_id, record.timestamp_utc


def _immutable_array(value: ArrayLike, dtype: DTypeLike) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _require_utc(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")
