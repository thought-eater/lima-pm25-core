"""Grillas y mediciones sintéticas para pruebas de reconstrucción."""

from datetime import datetime

import numpy as np

from lima_pm25.domain import GridDefinition
from lima_pm25.estimation import GaussianProcessFieldReconstructor, ReconstructionConfig
from lima_pm25.simulation import MeasurementQuality, MeasurementRecord, SensorType


def _reconstruction_config(**overrides: object) -> ReconstructionConfig:
    values: dict[str, object] = {
        "prior_mean_ug_m3": 20,
        "prior_stddev_ug_m3": 15,
        "x_length_scale_m": 300,
        "y_length_scale_m": 300,
        "temporal_length_scale_s": 3600,
    }
    values.update(overrides)
    return ReconstructionConfig(**values)  # type: ignore[arg-type]


def _reconstructor(
    *,
    prediction_chunk_size: int = 4,
    **overrides: object,
) -> GaussianProcessFieldReconstructor:
    return GaussianProcessFieldReconstructor(
        _reconstruction_config(**overrides),
        prediction_chunk_size=prediction_chunk_size,
    )


def _grid(
    *,
    rows: int = 2,
    columns: int = 2,
    active_mask: np.ndarray | None = None,
) -> GridDefinition:
    mask = (
        np.ones((rows, columns), dtype=bool)
        if active_mask is None
        else active_mask
    )
    return GridDefinition(
        crs="EPSG:32718",
        domain_bounds_m=np.array([0, 0, columns * 100, rows * 100]),
        grid_bounds_m=np.array([0, 0, columns * 100, rows * 100]),
        cell_size_m=100,
        x_centers_m=50 + np.arange(columns) * 100,
        y_centers_m=50 + np.arange(rows) * 100,
        active_mask=mask,
    )


def _record(
    sensor_id: str,
    timestamp: datetime,
    *,
    x_m: float = 50,
    y_m: float = 50,
    value: float | None = 20,
    noise_stddev: float = 0,
    crs: str = "EPSG:32718",
    quality: MeasurementQuality | None = None,
) -> MeasurementRecord:
    return MeasurementRecord(
        sensor_id=sensor_id,
        sensor_type=SensorType.FIXED,
        timestamp_utc=timestamp,
        simulation_time_s=0,
        x_m=x_m,
        y_m=y_m,
        crs=crs,
        value_ug_m3=value,
        quality_status=(
            quality
            if quality is not None
            else (
                MeasurementQuality.ACCEPTED
                if value is not None
                else MeasurementQuality.DROPPED
            )
        ),
        noise_stddev_ug_m3=noise_stddev,
        position_source_id=sensor_id,
        provenance="synthetic reconstruction test",
    )
