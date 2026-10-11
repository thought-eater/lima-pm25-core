"""Entradas sintéticas compartidas por pruebas de contratos y ensayos numéricos."""

from dataclasses import replace

import numpy as np
from shapely.geometry import box

from lima_pm25.domain import (
    AmbientBaseline,
    AtmosphericForcing,
    GridDefinition,
    RegionalBoundaryForcing,
    build_grid,
)
from lima_pm25.simulation.transport import (
    GaussianKernelSource,
    PM25PhysicsConfig,
    PM25Transport,
)


def _square_grid(side_cells: int = 41, cell_size_m: float = 100) -> GridDefinition:
    side_m = side_cells * cell_size_m
    return build_grid(box(0, 0, side_m, side_m), "EPSG:32718", cell_size_m)


def _physics(**changes: float) -> PM25PhysicsConfig:
    values = {
        "mixing_height_m": 100.0,
        "diffusivity_m2_s": 0.0,
        "dry_deposition_velocity_m_s": 0.0,
        "wet_scavenging_per_mm": 0.0,
    }
    values.update(changes)
    return PM25PhysicsConfig(**values)


def _forcing(**changes: float) -> AtmosphericForcing:
    values = {
        "provider_id": "test-meteorology",
        "provenance": "test fixture",
        "wind_speed_m_s": 0.0,
        "wind_from_direction_deg": 0.0,
        "precipitation_mm_h": 0.0,
    }
    values.update(changes)
    return AtmosphericForcing(**values)


def _source(**changes: object) -> GaussianKernelSource:
    source = GaussianKernelSource(
        source_id="source",
        x_m=2_050.0,
        y_m=2_050.0,
        emission_rate_ug_s=1_000.0,
        sigma_m=200.0,
        starts_at_s=0.0,
        ends_at_s=None,
        provenance="synthetic test source",
    )
    return replace(source, **changes)


def _baseline(concentration_ug_m3: float) -> AmbientBaseline:
    return AmbientBaseline(
        provider_id="test-ambient",
        concentration_ug_m3=concentration_ug_m3,
        provenance="test fixture",
    )


def _boundary(concentration_ug_m3: float) -> RegionalBoundaryForcing:
    return RegionalBoundaryForcing(
        provider_id="test-regional-boundary",
        concentration_ug_m3=concentration_ug_m3,
        provenance="test fixture",
    )


def _cell_volume(model: PM25Transport) -> float:
    return model.grid.cell_area_m2 * model.physics.mixing_height_m


def _centroid(grid: GridDefinition, concentration: np.ndarray) -> tuple[float, float]:
    center_x, center_y = np.meshgrid(grid.x_centers_m, grid.y_centers_m)
    total = concentration.sum()
    return (
        float(np.sum(concentration * center_x) / total),
        float(np.sum(concentration * center_y) / total),
    )


def _x_variance(
    grid: GridDefinition,
    concentration: np.ndarray,
    center_x_m: float,
) -> float:
    squared_distance = (grid.x_centers_m[np.newaxis, :] - center_x_m) ** 2
    return float(np.sum(concentration * squared_distance) / concentration.sum())
