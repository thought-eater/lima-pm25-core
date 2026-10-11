from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from pyproj import CRS
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern

from lima_pm25.domain import GridDefinition
from lima_pm25.simulation import MeasurementQuality, MeasurementRecord

DEFAULT_PREDICTION_CHUNK_SIZE = 2_048
RELATIVE_NUMERICAL_JITTER = 1e-10


@dataclass(frozen=True, slots=True, kw_only=True)
class ReconstructionConfig:
    prior_mean_ug_m3: float
    prior_stddev_ug_m3: float
    x_length_scale_m: float
    y_length_scale_m: float
    temporal_length_scale_s: float
    include_clipped_observations: bool = False

    def __post_init__(self) -> None:
        _require_nonnegative_finite("prior_mean_ug_m3", self.prior_mean_ug_m3)
        _require_positive_finite("prior_stddev_ug_m3", self.prior_stddev_ug_m3)
        _require_positive_finite("x_length_scale_m", self.x_length_scale_m)
        _require_positive_finite("y_length_scale_m", self.y_length_scale_m)
        _require_positive_finite(
            "temporal_length_scale_s",
            self.temporal_length_scale_s,
        )
        if not isinstance(self.include_clipped_observations, bool):
            raise ValueError("include_clipped_observations must be a boolean")  # noqa: TRY004 -- preserve configuration validation contract


@dataclass(frozen=True, slots=True, kw_only=True)
class ReconstructionRequest:
    grid: GridDefinition
    reconstruction_timestamp_utc: datetime
    measurements: Sequence[MeasurementRecord]

    def __post_init__(self) -> None:
        if not isinstance(self.grid, GridDefinition):
            raise TypeError("grid must be a GridDefinition")
        _require_metric_crs(self.grid.crs)
        _require_utc_datetime(
            "reconstruction_timestamp_utc",
            self.reconstruction_timestamp_utc,
        )
        measurements = tuple(self.measurements)
        if not all(isinstance(record, MeasurementRecord) for record in measurements):
            raise TypeError("measurements must contain only MeasurementRecord values")
        object.__setattr__(self, "measurements", measurements)


@dataclass(frozen=True, slots=True)
class ReconstructionObservationMetadata:
    provided_record_count: int
    used_record_count: int
    dropped_record_count: int
    future_record_count: int
    excluded_clipped_record_count: int
    used_measurement_keys: tuple[tuple[str, datetime], ...]


@dataclass(frozen=True, slots=True)
class ReconstructionDiagnostics:
    posterior_fitted: bool
    kernel: str
    numerical_jitter_ug_m3_squared: float
    log_marginal_likelihood: float | None


@dataclass(frozen=True, slots=True)
class PosteriorField:
    mean_ug_m3: np.ndarray
    standard_deviation_ug_m3: np.ndarray
    active_mask: np.ndarray
    reconstruction_timestamp_utc: datetime
    observations: ReconstructionObservationMetadata
    diagnostics: ReconstructionDiagnostics

    def __post_init__(self) -> None:
        mean = _immutable_array(self.mean_ug_m3, np.float64)
        standard_deviation = _immutable_array(
            self.standard_deviation_ug_m3,
            np.float64,
        )
        active_mask = _immutable_array(self.active_mask, np.bool_)
        object.__setattr__(self, "mean_ug_m3", mean)
        object.__setattr__(
            self,
            "standard_deviation_ug_m3",
            standard_deviation,
        )
        object.__setattr__(self, "active_mask", active_mask)

        if (
            mean.shape != active_mask.shape
            or standard_deviation.shape != active_mask.shape
        ):
            raise ValueError(
                "Posterior arrays and active_mask must have the same shape"
            )
        if not np.all(np.isfinite(mean[active_mask])):
            raise ValueError("Active posterior means must be finite")
        if not np.all(np.isfinite(standard_deviation[active_mask])):
            raise ValueError("Active posterior standard deviations must be finite")
        if np.any(standard_deviation[active_mask] < 0):
            raise ValueError("Posterior standard deviations must be nonnegative")
        if np.any(~np.isnan(mean[~active_mask])) or np.any(
            ~np.isnan(standard_deviation[~active_mask])
        ):
            raise ValueError(
                "Inactive posterior cells must be explicitly masked with NaN"
            )
        _require_utc_datetime(
            "reconstruction_timestamp_utc",
            self.reconstruction_timestamp_utc,
        )


@dataclass(frozen=True, slots=True)
class PointPosterior:
    positions_m: np.ndarray
    mean_ug_m3: np.ndarray
    standard_deviation_ug_m3: np.ndarray
    reconstruction_timestamp_utc: datetime
    observations: ReconstructionObservationMetadata
    diagnostics: ReconstructionDiagnostics

    def __post_init__(self) -> None:
        positions = _immutable_array(self.positions_m, np.float64)
        mean = _immutable_array(self.mean_ug_m3, np.float64)
        standard_deviation = _immutable_array(
            self.standard_deviation_ug_m3,
            np.float64,
        )
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "mean_ug_m3", mean)
        object.__setattr__(
            self,
            "standard_deviation_ug_m3",
            standard_deviation,
        )
        if positions.ndim != 2 or positions.shape[1:] != (2,):
            raise ValueError("positions_m must have shape (n, 2)")
        if mean.shape != (positions.shape[0],):
            raise ValueError("Point posterior mean must match positions_m")
        if standard_deviation.shape != mean.shape:
            raise ValueError("Point posterior uncertainty must match positions_m")
        if not np.all(np.isfinite(positions)):
            raise ValueError("Point posterior positions must be finite")
        if not np.all(np.isfinite(mean)):
            raise ValueError("Point posterior means must be finite")
        if not np.all(np.isfinite(standard_deviation)) or np.any(
            standard_deviation < 0
        ):
            raise ValueError(
                "Point posterior standard deviations must be finite and nonnegative"
            )
        _require_utc_datetime(
            "reconstruction_timestamp_utc",
            self.reconstruction_timestamp_utc,
        )


class GaussianProcessFieldReconstructor:
    def __init__(
        self,
        config: ReconstructionConfig,
        *,
        prediction_chunk_size: int = DEFAULT_PREDICTION_CHUNK_SIZE,
    ) -> None:
        if (
            isinstance(prediction_chunk_size, bool)
            or not isinstance(prediction_chunk_size, int)
            or prediction_chunk_size <= 0
        ):
            raise ValueError("prediction_chunk_size must be a positive integer")
        self.config = config
        self.prediction_chunk_size = prediction_chunk_size

    def reconstruct(self, request: ReconstructionRequest) -> PosteriorField:
        records, observation_metadata = select_reconstruction_observations(
            request,
            self.config,
        )
        kernel = _build_kernel(self.config)
        numerical_jitter = _numerical_jitter_ug_m3_squared(self.config)
        if not records:
            mean, standard_deviation = _prior_arrays(request.grid, self.config)
            return PosteriorField(
                mean_ug_m3=mean,
                standard_deviation_ug_m3=standard_deviation,
                active_mask=request.grid.active_mask,
                reconstruction_timestamp_utc=request.reconstruction_timestamp_utc,
                observations=observation_metadata,
                diagnostics=ReconstructionDiagnostics(
                    posterior_fitted=False,
                    kernel=str(kernel),
                    numerical_jitter_ug_m3_squared=numerical_jitter,
                    log_marginal_likelihood=None,
                ),
            )

        regressor = _fit_regressor(
            request,
            records,
            self.config,
            numerical_jitter,
        )

        mean, standard_deviation = _predict_active_cells(
            request,
            regressor,
            self.config,
            self.prediction_chunk_size,
        )

        return PosteriorField(
            mean_ug_m3=mean,
            standard_deviation_ug_m3=standard_deviation,
            active_mask=request.grid.active_mask,
            reconstruction_timestamp_utc=request.reconstruction_timestamp_utc,
            observations=observation_metadata,
            diagnostics=ReconstructionDiagnostics(
                posterior_fitted=True,
                kernel=str(regressor.kernel_),
                numerical_jitter_ug_m3_squared=numerical_jitter,
                log_marginal_likelihood=float(regressor.log_marginal_likelihood_value_),
            ),
        )

    def reconstruct_points(
        self,
        request: ReconstructionRequest,
        positions_m: Sequence[tuple[float, float]],
    ) -> PointPosterior:
        positions = np.asarray(tuple(positions_m), dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1:] != (2,) or not positions.size:
            raise ValueError("positions_m must contain one or more (x_m, y_m) pairs")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions_m must contain only finite coordinates")
        for x_m, y_m in positions:
            if not request.grid.is_active_xy(float(x_m), float(y_m)):
                raise ValueError(
                    "Every point prediction position must be in an active grid cell"
                )

        records, observation_metadata = select_reconstruction_observations(
            request,
            self.config,
        )
        kernel = _build_kernel(self.config)
        numerical_jitter = _numerical_jitter_ug_m3_squared(self.config)
        if not records:
            mean = np.full(positions.shape[0], self.config.prior_mean_ug_m3)
            standard_deviation = np.full(
                positions.shape[0],
                self.config.prior_stddev_ug_m3,
            )
            return PointPosterior(
                positions,
                mean,
                standard_deviation,
                request.reconstruction_timestamp_utc,
                observation_metadata,
                ReconstructionDiagnostics(
                    posterior_fitted=False,
                    kernel=str(kernel),
                    numerical_jitter_ug_m3_squared=numerical_jitter,
                    log_marginal_likelihood=None,
                ),
            )

        regressor = _fit_regressor(
            request,
            records,
            self.config,
            numerical_jitter,
        )
        x_origin, y_origin, _, _ = request.grid.grid_bounds_m
        features = np.column_stack(
            (
                positions[:, 0] - x_origin,
                positions[:, 1] - y_origin,
                np.zeros(positions.shape[0], dtype=np.float64),
            )
        )
        residual_mean, standard_deviation = regressor.predict(
            features,
            return_std=True,
        )
        return PointPosterior(
            positions,
            residual_mean + self.config.prior_mean_ug_m3,
            standard_deviation,
            request.reconstruction_timestamp_utc,
            observation_metadata,
            ReconstructionDiagnostics(
                posterior_fitted=True,
                kernel=str(regressor.kernel_),
                numerical_jitter_ug_m3_squared=numerical_jitter,
                log_marginal_likelihood=float(regressor.log_marginal_likelihood_value_),
            ),
        )


def select_reconstruction_observations(
    request: ReconstructionRequest,
    config: ReconstructionConfig,
) -> tuple[tuple[MeasurementRecord, ...], ReconstructionObservationMetadata]:
    """Return the deterministic observation set admitted by reconstruction policy."""
    used: list[MeasurementRecord] = []
    dropped_count = 0
    future_count = 0
    excluded_clipped_count = 0
    grid_crs = CRS.from_user_input(request.grid.crs)
    provided_keys: set[tuple[str, datetime]] = set()

    for record in request.measurements:
        key = (record.sensor_id, record.timestamp_utc)
        if key in provided_keys:
            raise ValueError(
                f"Duplicate measurement for sensor {record.sensor_id!r} at "
                f"{record.timestamp_utc.isoformat()}"
            )
        provided_keys.add(key)
        if not record.is_accepted:
            dropped_count += 1
            continue
        if record.timestamp_utc > request.reconstruction_timestamp_utc:
            future_count += 1
            continue
        if (
            record.quality_status is MeasurementQuality.CLIPPED
            and not config.include_clipped_observations
        ):
            excluded_clipped_count += 1
            continue
        if CRS.from_user_input(record.crs) != grid_crs:
            raise ValueError(
                f"Accepted measurement from sensor {record.sensor_id!r} uses a "
                "different CRS than the reconstruction grid"
            )
        if not request.grid.is_active_xy(record.x_m, record.y_m):
            raise ValueError(
                f"Accepted measurement from sensor {record.sensor_id!r} is not "
                "inside an active reconstruction cell"
            )
        used.append(record)

    used.sort(key=lambda record: (record.timestamp_utc, record.sensor_id))
    records = tuple(used)
    metadata = ReconstructionObservationMetadata(
        provided_record_count=len(request.measurements),
        used_record_count=len(records),
        dropped_record_count=dropped_count,
        future_record_count=future_count,
        excluded_clipped_record_count=excluded_clipped_count,
        used_measurement_keys=tuple(
            (record.sensor_id, record.timestamp_utc) for record in records
        ),
    )
    return records, metadata


def _build_kernel(config: ReconstructionConfig) -> Kernel:
    return ConstantKernel(
        config.prior_stddev_ug_m3**2,
        constant_value_bounds="fixed",
    ) * Matern(
        length_scale=np.array(
            [
                config.x_length_scale_m,
                config.y_length_scale_m,
                config.temporal_length_scale_s,
            ],
            dtype=np.float64,
        ),
        length_scale_bounds="fixed",
        nu=1.5,
    )


def _fit_regressor(
    request: ReconstructionRequest,
    records: tuple[MeasurementRecord, ...],
    config: ReconstructionConfig,
    numerical_jitter_ug_m3_squared: float,
) -> GaussianProcessRegressor:
    features, targets, alpha = _training_arrays(
        request,
        records,
        config,
        numerical_jitter_ug_m3_squared,
    )
    regressor = GaussianProcessRegressor(
        kernel=_build_kernel(config),
        alpha=alpha,
        optimizer=None,
        normalize_y=False,
        copy_X_train=True,
    )
    regressor.fit(features, targets)
    return regressor


def _training_arrays(
    request: ReconstructionRequest,
    records: tuple[MeasurementRecord, ...],
    config: ReconstructionConfig,
    numerical_jitter_ug_m3_squared: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_origin, y_origin, _, _ = request.grid.grid_bounds_m
    features = np.asarray(
        [
            (
                record.x_m - x_origin,
                record.y_m - y_origin,
                (
                    record.timestamp_utc - request.reconstruction_timestamp_utc
                ).total_seconds(),
            )
            for record in records
        ],
        dtype=np.float64,
    )
    targets = (
        np.asarray(
            [record.value_ug_m3 for record in records],
            dtype=np.float64,
        )
        - config.prior_mean_ug_m3
    )
    alpha = np.asarray(
        [
            record.noise_stddev_ug_m3**2 + numerical_jitter_ug_m3_squared
            for record in records
        ],
        dtype=np.float64,
    )
    return features, targets, alpha


def _predict_active_cells(
    request: ReconstructionRequest,
    regressor: GaussianProcessRegressor,
    config: ReconstructionConfig,
    prediction_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = request.grid
    rows, columns = np.nonzero(grid.active_mask)
    x_origin, y_origin, _, _ = grid.grid_bounds_m
    active_features = np.column_stack(
        (
            grid.x_centers_m[columns] - x_origin,
            grid.y_centers_m[rows] - y_origin,
            np.zeros(rows.size, dtype=np.float64),
        )
    )
    active_mean = np.empty(rows.size, dtype=np.float64)
    active_standard_deviation = np.empty(rows.size, dtype=np.float64)

    for start in range(0, rows.size, prediction_chunk_size):
        stop = min(start + prediction_chunk_size, rows.size)
        residual_mean, standard_deviation = regressor.predict(
            active_features[start:stop],
            return_std=True,
        )
        active_mean[start:stop] = residual_mean + config.prior_mean_ug_m3
        active_standard_deviation[start:stop] = standard_deviation

    mean = np.full(grid.shape, np.nan, dtype=np.float64)
    standard_deviation = np.full(grid.shape, np.nan, dtype=np.float64)
    mean[rows, columns] = active_mean
    standard_deviation[rows, columns] = active_standard_deviation
    return mean, standard_deviation


def _numerical_jitter_ug_m3_squared(config: ReconstructionConfig) -> float:
    return RELATIVE_NUMERICAL_JITTER * config.prior_stddev_ug_m3**2


def _prior_arrays(
    grid: GridDefinition,
    config: ReconstructionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.full(grid.shape, np.nan, dtype=np.float64)
    standard_deviation = np.full(grid.shape, np.nan, dtype=np.float64)
    mean[grid.active_mask] = config.prior_mean_ug_m3
    standard_deviation[grid.active_mask] = config.prior_stddev_ug_m3
    return mean, standard_deviation


def _require_utc_datetime(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")


def _require_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _require_metric_crs(crs_value: str) -> None:
    crs = CRS.from_user_input(crs_value)
    if not crs.is_projected or len(crs.axis_info) < 2:
        raise ValueError("The reconstruction CRS must be projected in meters")
    if any(
        not math.isclose(axis.unit_conversion_factor or 0.0, 1.0)
        for axis in crs.axis_info[:2]
    ):
        raise ValueError("The reconstruction CRS axes must use meters")


def _immutable_array(value: ArrayLike, dtype: DTypeLike) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
