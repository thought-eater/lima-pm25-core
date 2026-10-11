from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from itertools import pairwise

import numpy as np
from shapely.geometry import LineString

from lima_pm25.domain import (
    AmbientBaseline,
    AtmosphericForcing,
    GridDefinition,
    PM25SourceContribution,
    PM25SourceProvider,
    PM25SourceRole,
    RegionalBoundaryForcing,
    SourceDescriptor,
)


class BoundaryMode(StrEnum):
    CLOSED = "closed"
    REGIONAL_INFLOW = "regional_inflow"


@dataclass(frozen=True, slots=True)
class PM25PhysicsConfig:
    mixing_height_m: float
    diffusivity_m2_s: float
    dry_deposition_velocity_m_s: float
    wet_scavenging_per_mm: float
    cfl_safety_factor: float = 0.8

    def __post_init__(self) -> None:
        _require_positive_finite("mixing_height_m", self.mixing_height_m)
        _require_nonnegative_finite("diffusivity_m2_s", self.diffusivity_m2_s)
        _require_nonnegative_finite(
            "dry_deposition_velocity_m_s",
            self.dry_deposition_velocity_m_s,
        )
        _require_nonnegative_finite(
            "wet_scavenging_per_mm",
            self.wet_scavenging_per_mm,
        )
        if not math.isfinite(self.cfl_safety_factor) or not (
            0 < self.cfl_safety_factor < 1
        ):
            raise ValueError("cfl_safety_factor must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class SourceMassDiagnostic:
    source_id: str
    emitted_mass_ug: float


@dataclass(frozen=True, slots=True)
class GaussianKernelSource:
    """Distribute *emitted mass* around a center; transport disperses it later.

    The Gaussian kernel is a spatial emission footprint, not an atmospheric
    Gaussian plume model. Location, width and rate need scenario evidence before
    this source can represent real emissions.
    """

    source_id: str
    x_m: float
    y_m: float
    emission_rate_ug_s: float
    sigma_m: float
    starts_at_s: float = 0.0
    ends_at_s: float | None = None
    role: PM25SourceRole = PM25SourceRole.SYNTHETIC_VALIDATION
    provenance: str = "synthetic Gaussian kernel"

    def __post_init__(self) -> None:
        _require_usable_text("source_id", self.source_id)
        object.__setattr__(self, "role", PM25SourceRole(self.role))
        _require_usable_text("provenance", self.provenance)
        if not math.isfinite(self.x_m) or not math.isfinite(self.y_m):
            raise ValueError("Source coordinates must be finite")
        _require_nonnegative_finite("emission_rate_ug_s", self.emission_rate_ug_s)
        _require_positive_finite("sigma_m", self.sigma_m)
        _require_nonnegative_finite("starts_at_s", self.starts_at_s)
        if self.ends_at_s is not None and (
            not math.isfinite(self.ends_at_s) or self.ends_at_s <= self.starts_at_s
        ):
            raise ValueError("ends_at_s must be finite and greater than starts_at_s")

    def emitted_mass_ug(self, start_s: float, duration_s: float) -> float:
        interval_end = start_s + duration_s
        active_start = max(start_s, self.starts_at_s)
        active_end = min(
            interval_end,
            self.ends_at_s if self.ends_at_s is not None else interval_end,
        )
        active_duration = max(0.0, active_end - active_start)
        return self.emission_rate_ug_s * active_duration

    @property
    def transition_times_s(self) -> tuple[float, ...]:
        times = [self.starts_at_s]
        if self.ends_at_s is not None:
            times.append(self.ends_at_s)
        return tuple(times)

    def evidence_descriptor(self) -> dict[str, object]:
        return {
            "type": "gaussian_kernel",
            "source_id": self.source_id,
            "role": self.role.value,
            "provenance": self.provenance,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "emission_rate_ug_s": self.emission_rate_ug_s,
            "sigma_m": self.sigma_m,
            "starts_at_s": self.starts_at_s,
            "ends_at_s": self.ends_at_s,
        }

    def spatial_weights(self, grid: GridDefinition) -> np.ndarray:
        if not grid.is_active_xy(self.x_m, self.y_m):
            raise ValueError(f"Source {self.source_id!r} is outside the active domain")

        center_x, center_y = np.meshgrid(grid.x_centers_m, grid.y_centers_m)
        squared_distance = (center_x - self.x_m) ** 2 + (center_y - self.y_m) ** 2
        closest_squared_distance = float(np.min(squared_distance[grid.active_mask]))
        exponent = (
            -0.5 * (squared_distance - closest_squared_distance) / self.sigma_m**2
        )
        weights = np.zeros(grid.shape, dtype=np.float64)
        weights[grid.active_mask] = np.exp(exponent[grid.active_mask])
        weight_sum = float(np.sum(weights, dtype=np.float64))
        if not math.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError(f"Source {self.source_id!r} has invalid Gaussian weights")
        weights /= weight_sum
        return weights

    def contribution_for_interval(
        self,
        grid: GridDefinition,
        *,
        start_s: float,
        duration_s: float,
    ) -> PM25SourceContribution:
        weights = self.spatial_weights(grid)
        emitted_mass = self.emitted_mass_ug(start_s, duration_s)
        return PM25SourceContribution(weights * emitted_mass, emitted_mass)


@dataclass(slots=True)
class RoadLineSource:
    """Road emission source with a mutable cache of grid-dependent weights."""

    source_id: str
    line: LineString
    emission_rate_ug_m_s: float
    provenance: str
    starts_at_s: float = 0.0
    ends_at_s: float | None = None
    role: PM25SourceRole = PM25SourceRole.ROUTINE_ENVIRONMENTAL
    _weight_cache: dict[
        int,
        tuple[GridDefinition, LineString, np.ndarray, float],
    ] = field(default_factory=dict, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        _require_usable_text("source_id", self.source_id)
        object.__setattr__(self, "role", PM25SourceRole(self.role))
        _require_usable_text("provenance", self.provenance)
        if not isinstance(self.line, LineString) or self.line.is_empty:
            raise ValueError("line must be a nonempty LineString")
        coordinates = np.asarray(self.line.coords, dtype=np.float64)
        if not np.all(np.isfinite(coordinates)) or self.line.length <= 0:
            raise ValueError("line must contain finite coordinates and positive length")
        _require_nonnegative_finite(
            "emission_rate_ug_m_s",
            self.emission_rate_ug_m_s,
        )
        _require_nonnegative_finite("starts_at_s", self.starts_at_s)
        if self.ends_at_s is not None and (
            not math.isfinite(self.ends_at_s) or self.ends_at_s <= self.starts_at_s
        ):
            raise ValueError("ends_at_s must be finite and greater than starts_at_s")

    def emitted_mass_ug(
        self,
        start_s: float,
        duration_s: float,
        active_length_m: float,
    ) -> float:
        interval_end = start_s + duration_s
        active_start = max(start_s, self.starts_at_s)
        active_end = min(
            interval_end,
            self.ends_at_s if self.ends_at_s is not None else interval_end,
        )
        active_duration = max(0.0, active_end - active_start)
        return self.emission_rate_ug_m_s * active_length_m * active_duration

    @property
    def transition_times_s(self) -> tuple[float, ...]:
        times = [self.starts_at_s]
        if self.ends_at_s is not None:
            times.append(self.ends_at_s)
        return tuple(times)

    def evidence_descriptor(self) -> dict[str, object]:
        return {
            "type": "road_line",
            "source_id": self.source_id,
            "role": self.role.value,
            "provenance": self.provenance,
            "line_wkt": self.line.wkt,
            "emission_rate_ug_m_s": self.emission_rate_ug_m_s,
            "starts_at_s": self.starts_at_s,
            "ends_at_s": self.ends_at_s,
        }

    def spatial_weights(self, grid: GridDefinition) -> tuple[np.ndarray, float]:
        cache_key = id(grid)
        cached = self._weight_cache.get(cache_key)
        if cached is not None and cached[0] is grid and cached[1] is self.line:
            return cached[2].copy(), cached[3]

        weights, active_length = _line_weights_on_grid(self.line, grid)
        weight_sum = float(np.sum(weights, dtype=np.float64))
        if not math.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError(f"Source {self.source_id!r} is outside the active domain")
        if not math.isfinite(active_length) or active_length <= 0:
            raise ValueError(f"Source {self.source_id!r} has invalid active length")
        weights /= weight_sum
        cached_weights = _immutable_float_array(weights)
        self._weight_cache[cache_key] = (
            grid,
            self.line,
            cached_weights,
            active_length,
        )
        return weights, active_length

    def contribution_for_interval(
        self,
        grid: GridDefinition,
        *,
        start_s: float,
        duration_s: float,
    ) -> PM25SourceContribution:
        weights, active_length = self.spatial_weights(grid)
        emitted_mass = self.emitted_mass_ug(start_s, duration_s, active_length)
        return PM25SourceContribution(weights * emitted_mass, emitted_mass)


@dataclass(frozen=True, slots=True)
class PM25State:
    grid: GridDefinition = field(repr=False, compare=False)
    concentration_ug_m3: np.ndarray
    simulation_time_s: float
    ambient_baseline: AmbientBaseline | None = None

    def __post_init__(self) -> None:
        concentration = np.asarray(self.concentration_ug_m3, dtype=np.float64)
        if concentration.shape != self.grid.shape:
            raise ValueError(
                f"Concentration shape {concentration.shape} does not match "
                f"grid shape {self.grid.shape}"
            )
        active_values = concentration[self.grid.active_mask]
        if not np.all(np.isfinite(active_values)):
            raise ValueError("Active PM2.5 concentrations must be finite")
        if np.any(active_values < 0):
            raise ValueError("Active PM2.5 concentrations must be nonnegative")
        if np.any(concentration[~self.grid.active_mask] != 0):
            raise ValueError("Inactive PM2.5 cells must be zero")
        object.__setattr__(
            self,
            "concentration_ug_m3",
            _immutable_float_array(concentration),
        )
        if not math.isfinite(self.simulation_time_s) or self.simulation_time_s < 0:
            raise ValueError("simulation_time_s must be nonnegative and finite")


@dataclass(frozen=True, slots=True)
class MassBudget:
    initial_mass_ug: float
    emitted_mass_ug: float
    boundary_inflow_mass_ug: float
    boundary_outflow_mass_ug: float
    dry_deposited_mass_ug: float
    wet_deposited_mass_ug: float
    final_mass_ug: float
    numerical_residual_ug: float
    substeps: int
    substep_duration_s: float
    source_emissions: tuple[SourceMassDiagnostic, ...] = ()

    @property
    def expected_final_mass_ug(self) -> float:
        return (
            self.initial_mass_ug
            + self.emitted_mass_ug
            + self.boundary_inflow_mass_ug
            - self.boundary_outflow_mass_ug
            - self.dry_deposited_mass_ug
            - self.wet_deposited_mass_ug
        )

    @property
    def relative_residual(self) -> float:
        scale = max(abs(self.final_mass_ug), abs(self.expected_final_mass_ug), 1.0)
        return self.numerical_residual_ug / scale

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["expected_final_mass_ug"] = self.expected_final_mass_ug
        result["relative_residual"] = self.relative_residual
        return result


@dataclass(frozen=True, slots=True)
class SimulationFrame:
    state: PM25State
    forcing: AtmosphericForcing
    regional_boundary: RegionalBoundaryForcing | None
    sources: tuple[SourceDescriptor, ...]
    budget: MassBudget | None


class PM25Transport:
    def __init__(self, grid: GridDefinition, physics: PM25PhysicsConfig) -> None:
        self.grid = grid
        self.physics = physics
        self._active_mask = grid.active_mask
        self._cell_volume_m3 = grid.cell_area_m2 * physics.mixing_height_m
        self._face_area_m2 = grid.cell_size_m * physics.mixing_height_m

    def initialize_uniform(
        self,
        baseline: AmbientBaseline,
        *,
        simulation_time_s: float = 0.0,
    ) -> PM25State:
        concentration = np.zeros(self.grid.shape, dtype=np.float64)
        concentration[self._active_mask] = baseline.concentration_ug_m3
        return PM25State(
            self.grid,
            concentration,
            simulation_time_s,
            ambient_baseline=baseline,
        )

    def initialize_from_array(
        self,
        concentration_ug_m3: np.ndarray,
        *,
        simulation_time_s: float = 0.0,
        ambient_baseline: AmbientBaseline | None = None,
    ) -> PM25State:
        return PM25State(
            self.grid,
            concentration_ug_m3,
            simulation_time_s,
            ambient_baseline=ambient_baseline,
        )

    def total_mass_ug(self, concentration_ug_m3: np.ndarray) -> float:
        return float(
            np.sum(concentration_ug_m3[self._active_mask], dtype=np.float64)
            * self._cell_volume_m3
        )

    def maximum_stable_timestep_s(self, forcing: AtmosphericForcing) -> float:
        eastward, northward = forcing.wind_components_m_s
        spacing = self.grid.cell_size_m
        denominator = (
            abs(eastward) / spacing
            + abs(northward) / spacing
            + 4 * self.physics.diffusivity_m2_s / spacing**2
        )
        if denominator == 0:
            return math.inf
        return self.physics.cfl_safety_factor / denominator

    def step(
        self,
        state: PM25State,
        *,
        duration_s: float,
        forcing: AtmosphericForcing,
        regional_boundary: RegionalBoundaryForcing | None = None,
        sources: Sequence[PM25SourceProvider] = (),
        boundary_mode: BoundaryMode | str = BoundaryMode.CLOSED,
    ) -> tuple[PM25State, MassBudget]:
        self._validate_state(state)
        _require_positive_finite("duration_s", duration_s)
        mode = BoundaryMode(boundary_mode)
        if mode is BoundaryMode.REGIONAL_INFLOW and regional_boundary is None:
            raise ValueError(
                "regional_boundary is required for regional-inflow boundaries"
            )

        prepared_sources = _prepare_source_providers(sources)
        maximum_timestep = self.maximum_stable_timestep_s(forcing)
        substeps = (
            1
            if math.isinf(maximum_timestep)
            else max(1, math.ceil(duration_s / maximum_timestep))
        )
        substep_duration = duration_s / substeps

        concentration = state.concentration_ug_m3.copy()
        initial_mass = self.total_mass_ug(concentration)
        emitted_by_source = np.zeros(len(prepared_sources), dtype=np.float64)
        boundary_inflow = 0.0
        boundary_outflow = 0.0
        dry_deposited = 0.0
        wet_deposited = 0.0
        current_time = state.simulation_time_s

        for _ in range(substeps):
            emitted_by_source += self._inject_sources(
                concentration,
                current_time,
                substep_duration,
                prepared_sources,
            )
            concentration, inflow, outflow = self._advect(
                concentration,
                substep_duration,
                forcing,
                mode,
                regional_boundary,
            )
            boundary_inflow += inflow
            boundary_outflow += outflow
            concentration = self._diffuse(concentration, substep_duration)

            concentration, removed_dry = self._apply_exponential_sink(
                concentration,
                self.physics.dry_deposition_velocity_m_s / self.physics.mixing_height_m,
                substep_duration,
            )
            dry_deposited += removed_dry

            wet_rate_s = (
                self.physics.wet_scavenging_per_mm * forcing.precipitation_mm_h / 3600
            )
            concentration, removed_wet = self._apply_exponential_sink(
                concentration,
                wet_rate_s,
                substep_duration,
            )
            wet_deposited += removed_wet

            self._enforce_numerical_invariants(concentration)
            current_time += substep_duration

        emitted_mass = float(np.sum(emitted_by_source, dtype=np.float64))
        final_mass = self.total_mass_ug(concentration)
        expected_final_mass = (
            initial_mass
            + emitted_mass
            + boundary_inflow
            - boundary_outflow
            - dry_deposited
            - wet_deposited
        )
        residual = final_mass - expected_final_mass
        budget = MassBudget(
            initial_mass_ug=initial_mass,
            emitted_mass_ug=emitted_mass,
            boundary_inflow_mass_ug=boundary_inflow,
            boundary_outflow_mass_ug=boundary_outflow,
            dry_deposited_mass_ug=dry_deposited,
            wet_deposited_mass_ug=wet_deposited,
            final_mass_ug=final_mass,
            numerical_residual_ug=residual,
            substeps=substeps,
            substep_duration_s=substep_duration,
            source_emissions=tuple(
                SourceMassDiagnostic(
                    source_id=descriptor.source_id,
                    emitted_mass_ug=float(source_mass),
                )
                for (_, descriptor), source_mass in zip(
                    prepared_sources,
                    emitted_by_source,
                    strict=True,
                )
            ),
        )
        final_time = state.simulation_time_s + duration_s
        return PM25State(
            self.grid,
            concentration,
            final_time,
            ambient_baseline=state.ambient_baseline,
        ), budget

    def _validate_state(self, state: PM25State) -> None:
        if not _grids_match(state.grid, self.grid):
            raise ValueError("PM2.5 state grid does not match the transport grid")

    def _inject_sources(
        self,
        concentration: np.ndarray,
        current_time_s: float,
        duration_s: float,
        prepared_sources: tuple[
            tuple[PM25SourceProvider, SourceDescriptor],
            ...,
        ],
    ) -> np.ndarray:
        validated_contributions = []
        emitted_mass = np.zeros(len(prepared_sources), dtype=np.float64)
        for index, (source, descriptor) in enumerate(prepared_sources):
            try:
                contribution = source.contribution_for_interval(
                    self.grid,
                    start_s=current_time_s,
                    duration_s=duration_s,
                )
            except (AttributeError, TypeError) as error:
                raise ValueError(
                    f"Source {descriptor.source_id!r} cannot provide an interval contribution"
                ) from error
            mass_by_cell = self._validate_source_contribution(
                contribution,
                descriptor.source_id,
            )
            if _describe_source(source) != descriptor:
                raise ValueError(
                    f"Source {descriptor.source_id!r} metadata changed during the step"
                )
            validated_contributions.append(mass_by_cell)
            emitted_mass[index] = contribution.integrated_mass_ug

        for mass_by_cell in validated_contributions:
            concentration += mass_by_cell / self._cell_volume_m3
        return emitted_mass

    def _validate_source_contribution(
        self,
        contribution: PM25SourceContribution,
        source_id: str,
    ) -> np.ndarray:
        if not isinstance(contribution, PM25SourceContribution):
            raise ValueError(  # noqa: TRY004 -- preserve source validation contract
                f"Source {source_id!r} must return PM25SourceContribution"
            )
        if contribution.unit != "ug":
            raise ValueError(f"Source {source_id!r} contribution unit must be 'ug'")

        mass_by_cell = contribution.mass_by_cell_ug
        if mass_by_cell.shape != self.grid.shape:
            raise ValueError(
                f"Source {source_id!r} contribution shape {mass_by_cell.shape} "
                f"does not match grid shape {self.grid.shape}"
            )
        if not np.all(np.isfinite(mass_by_cell)):
            raise ValueError(f"Source {source_id!r} contribution values must be finite")
        if np.any(mass_by_cell < 0):
            raise ValueError(
                f"Source {source_id!r} contribution values must be nonnegative"
            )
        if np.any(mass_by_cell[~self._active_mask] != 0):
            raise ValueError(
                f"Source {source_id!r} contribution must be zero on inactive cells"
            )
        if not math.isfinite(contribution.integrated_mass_ug) or (
            contribution.integrated_mass_ug < 0
        ):
            raise ValueError(
                f"Source {source_id!r} integrated mass must be nonnegative and finite"
            )

        cell_mass = float(np.sum(mass_by_cell, dtype=np.float64))
        tolerance = (
            64
            * np.finfo(np.float64).eps
            * max(
                abs(contribution.integrated_mass_ug),
                1.0,
            )
        )
        if not math.isclose(
            cell_mass,
            contribution.integrated_mass_ug,
            rel_tol=0,
            abs_tol=tolerance,
        ):
            raise ValueError(
                f"Source {source_id!r} cell mass {cell_mass:g} ug does not match "
                f"declared mass {contribution.integrated_mass_ug:g} ug"
            )
        return _immutable_float_array(mass_by_cell)

    def _advect(
        self,
        concentration: np.ndarray,
        duration_s: float,
        forcing: AtmosphericForcing,
        boundary_mode: BoundaryMode,
        regional_boundary: RegionalBoundaryForcing | None,
    ) -> tuple[np.ndarray, float, float]:
        eastward, northward = forcing.wind_components_m_s
        active = self._active_mask
        rows, columns = self.grid.shape

        outside_value = (
            regional_boundary.concentration_ug_m3
            if boundary_mode is BoundaryMode.REGIONAL_INFLOW
            and regional_boundary is not None
            else 0.0
        )
        face_concentration = np.where(active, concentration, outside_value)

        padded_x = np.pad(
            face_concentration,
            ((0, 0), (1, 1)),
            constant_values=outside_value,
        )
        padded_active_x = np.pad(active, ((0, 0), (1, 1)), constant_values=False)
        left_concentration = padded_x[:, :-1]
        right_concentration = padded_x[:, 1:]
        left_active = padded_active_x[:, :-1]
        right_active = padded_active_x[:, 1:]
        x_internal = left_active & right_active
        x_boundary = left_active ^ right_active
        x_allowed = (
            x_internal | x_boundary
            if boundary_mode is BoundaryMode.REGIONAL_INFLOW
            else x_internal
        )
        x_upwind = left_concentration if eastward >= 0 else right_concentration
        x_flux = eastward * self._face_area_m2 * x_upwind * x_allowed

        padded_y = np.pad(
            face_concentration,
            ((1, 1), (0, 0)),
            constant_values=outside_value,
        )
        padded_active_y = np.pad(active, ((1, 1), (0, 0)), constant_values=False)
        bottom_concentration = padded_y[:-1, :]
        top_concentration = padded_y[1:, :]
        bottom_active = padded_active_y[:-1, :]
        top_active = padded_active_y[1:, :]
        y_internal = bottom_active & top_active
        y_boundary = bottom_active ^ top_active
        y_allowed = (
            y_internal | y_boundary
            if boundary_mode is BoundaryMode.REGIONAL_INFLOW
            else y_internal
        )
        y_upwind = bottom_concentration if northward >= 0 else top_concentration
        y_flux = northward * self._face_area_m2 * y_upwind * y_allowed

        net_mass_rate = (
            x_flux[:, :columns]
            - x_flux[:, 1 : columns + 1]
            + y_flux[:rows, :]
            - y_flux[1 : rows + 1, :]
        )
        updated = concentration + duration_s * net_mass_rate / self._cell_volume_m3
        updated[~active] = 0.0

        x_outward_flux = np.where(left_active, x_flux, -x_flux)[x_boundary]
        y_outward_flux = np.where(bottom_active, y_flux, -y_flux)[y_boundary]
        outward_flux = np.concatenate((x_outward_flux, y_outward_flux))
        inflow_mass = float(
            np.sum(np.maximum(-outward_flux, 0), dtype=np.float64) * duration_s
        )
        outflow_mass = float(
            np.sum(np.maximum(outward_flux, 0), dtype=np.float64) * duration_s
        )
        return updated, inflow_mass, outflow_mass

    def _diffuse(self, concentration: np.ndarray, duration_s: float) -> np.ndarray:
        diffusivity = self.physics.diffusivity_m2_s
        if diffusivity == 0:
            return concentration

        active = self._active_mask
        rows, columns = self.grid.shape
        coefficient = -diffusivity * self._face_area_m2 / self.grid.cell_size_m

        x_flux = np.zeros((rows, columns + 1), dtype=np.float64)
        x_internal = active[:, :-1] & active[:, 1:]
        x_flux[:, 1:columns] = np.where(
            x_internal,
            coefficient * (concentration[:, 1:] - concentration[:, :-1]),
            0.0,
        )

        y_flux = np.zeros((rows + 1, columns), dtype=np.float64)
        y_internal = active[:-1, :] & active[1:, :]
        y_flux[1:rows, :] = np.where(
            y_internal,
            coefficient * (concentration[1:, :] - concentration[:-1, :]),
            0.0,
        )

        net_mass_rate = (
            x_flux[:, :columns]
            - x_flux[:, 1 : columns + 1]
            + y_flux[:rows, :]
            - y_flux[1 : rows + 1, :]
        )
        updated = concentration + duration_s * net_mass_rate / self._cell_volume_m3
        updated[~active] = 0.0
        return updated

    def _apply_exponential_sink(
        self,
        concentration: np.ndarray,
        removal_rate_s: float,
        duration_s: float,
    ) -> tuple[np.ndarray, float]:
        if removal_rate_s == 0:
            return concentration, 0.0
        mass_before = self.total_mass_ug(concentration)
        updated = concentration * math.exp(-removal_rate_s * duration_s)
        updated[~self._active_mask] = 0.0
        removed_mass = mass_before - self.total_mass_ug(updated)
        return updated, removed_mass

    def _enforce_numerical_invariants(self, concentration: np.ndarray) -> None:
        active_values = concentration[self._active_mask]
        scale = max(float(np.max(active_values, initial=0.0)), 1.0)
        tolerance = 1e-12 * scale
        minimum = float(np.min(active_values, initial=0.0))
        if minimum < -tolerance:
            raise RuntimeError(
                f"PM2.5 transport produced negative concentration {minimum:g} ug/m3"
            )
        concentration[self._active_mask] = np.maximum(active_values, 0.0)
        concentration[~self._active_mask] = 0.0


def iter_simulation(
    model: PM25Transport,
    initial_state: PM25State,
    *,
    duration_s: float,
    frame_duration_s: float,
    forcing_at: Callable[[float], AtmosphericForcing],
    regional_boundary_at: Callable[[float], RegionalBoundaryForcing] | None = None,
    sources: Sequence[PM25SourceProvider] = (),
    boundary_mode: BoundaryMode | str = BoundaryMode.CLOSED,
    forcing_change_times_s: Sequence[float] = (),
    regional_boundary_change_times_s: Sequence[float] = (),
    source_change_times_s: Sequence[float] = (),
) -> Iterator[SimulationFrame]:
    _require_positive_finite("duration_s", duration_s)
    _require_positive_finite("frame_duration_s", frame_duration_s)
    end_time = initial_state.simulation_time_s + duration_s
    change_times = tuple(
        time
        for time in sorted(
            {
                *forcing_change_times_s,
                *regional_boundary_change_times_s,
                *source_change_times_s,
            }
        )
        if initial_state.simulation_time_s < time < end_time
    )
    mode = BoundaryMode(boundary_mode)
    if mode is BoundaryMode.REGIONAL_INFLOW and regional_boundary_at is None:
        raise ValueError(
            "regional_boundary_at is required for regional-inflow boundaries"
        )
    source_descriptors = tuple(
        descriptor for _, descriptor in _prepare_source_providers(sources)
    )
    state = initial_state
    initial_boundary = (
        regional_boundary_at(state.simulation_time_s)
        if regional_boundary_at is not None
        else None
    )
    yield SimulationFrame(
        state,
        forcing_at(state.simulation_time_s),
        initial_boundary,
        source_descriptors,
        None,
    )

    while end_time - state.simulation_time_s > 1e-9:
        remaining = end_time - state.simulation_time_s
        next_change = next(
            (time for time in change_times if time > state.simulation_time_s + 1e-9),
            end_time,
        )
        step_duration = min(
            frame_duration_s,
            remaining,
            next_change - state.simulation_time_s,
        )
        forcing = forcing_at(state.simulation_time_s)
        regional_boundary = (
            regional_boundary_at(state.simulation_time_s)
            if regional_boundary_at is not None
            else None
        )
        state, budget = model.step(
            state,
            duration_s=step_duration,
            forcing=forcing,
            regional_boundary=regional_boundary,
            sources=sources,
            boundary_mode=mode,
        )
        yield SimulationFrame(
            state,
            forcing,
            regional_boundary,
            source_descriptors,
            budget,
        )


def _prepare_source_providers(
    sources: Sequence[PM25SourceProvider],
) -> tuple[tuple[PM25SourceProvider, SourceDescriptor], ...]:
    prepared = []
    source_ids = set()
    for source in sources:
        descriptor = _describe_source(source)
        if descriptor.source_id in source_ids:
            raise ValueError(f"Duplicate source_id {descriptor.source_id!r}")
        source_ids.add(descriptor.source_id)
        prepared.append((source, descriptor))
    return tuple(prepared)


def _describe_source(source: PM25SourceProvider) -> SourceDescriptor:
    try:
        source_id = source.source_id
        role = PM25SourceRole(source.role)
        provenance = source.provenance
        contribution_method = source.contribution_for_interval
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "Every source provider must expose a usable source_id, role, "
            "provenance, and contribution_for_interval method"
        ) from error
    _require_usable_text("source_id", source_id)
    _require_usable_text("provenance", provenance)
    if not callable(contribution_method):
        raise ValueError(  # noqa: TRY004 -- preserve source validation contract
            f"Source {source_id!r} contribution_for_interval must be callable"
        )
    return SourceDescriptor(source_id, role, provenance)


def _grids_match(first: GridDefinition, second: GridDefinition) -> bool:
    return (
        first.crs == second.crs
        and first.cell_size_m == second.cell_size_m
        and np.array_equal(first.domain_bounds_m, second.domain_bounds_m)
        and np.array_equal(first.grid_bounds_m, second.grid_bounds_m)
        and np.array_equal(first.x_centers_m, second.x_centers_m)
        and np.array_equal(first.y_centers_m, second.y_centers_m)
        and np.array_equal(first.active_mask, second.active_mask)
    )


def _line_weights_on_grid(
    line: LineString,
    grid: GridDefinition,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(grid.shape, dtype=np.float64)
    active_length = 0.0
    coordinates = np.asarray(line.coords, dtype=np.float64)[:, :2]
    x_min, y_min, _, _ = grid.grid_bounds_m

    for start, end in pairwise(coordinates):
        delta_x, delta_y = end - start
        segment_length = float(np.hypot(delta_x, delta_y))
        if segment_length == 0:
            continue

        breakpoints = [0.0, 1.0]
        if delta_x != 0:
            breakpoints.extend(
                crossing
                for column in range(grid.columns + 1)
                if 0
                < (crossing := (x_min + column * grid.cell_size_m - start[0]) / delta_x)
                < 1
            )
        if delta_y != 0:
            breakpoints.extend(
                crossing
                for row in range(grid.rows + 1)
                if 0
                < (crossing := (y_min + row * grid.cell_size_m - start[1]) / delta_y)
                < 1
            )

        ordered_points = []
        for value in sorted(breakpoints):
            if not ordered_points or not math.isclose(
                value,
                ordered_points[-1],
                rel_tol=0,
                abs_tol=1e-12,
            ):
                ordered_points.append(value)

        for interval_start, interval_end in pairwise(ordered_points):
            midpoint_fraction = (interval_start + interval_end) / 2
            midpoint_x = start[0] + midpoint_fraction * delta_x
            midpoint_y = start[1] + midpoint_fraction * delta_y
            cells = _active_cells_touching_xy(grid, midpoint_x, midpoint_y)
            if not cells:
                continue
            interval_length = segment_length * (interval_end - interval_start)
            active_length += interval_length
            shared_length = interval_length / len(cells)
            for row, column in cells:
                weights[row, column] += shared_length

    return weights, active_length


def _active_cells_touching_xy(
    grid: GridDefinition,
    x_m: float,
    y_m: float,
) -> tuple[tuple[int, int], ...]:
    x_min, y_min, _, _ = grid.grid_bounds_m
    column_position = (x_m - x_min) / grid.cell_size_m
    row_position = (y_m - y_min) / grid.cell_size_m
    columns = _touching_indices(column_position, grid.columns)
    rows = _touching_indices(row_position, grid.rows)
    return tuple(
        (row, column)
        for row in rows
        for column in columns
        if grid.active_mask[row, column]
    )


def _touching_indices(position: float, count: int) -> tuple[int, ...]:
    nearest_boundary = round(position)
    if math.isclose(position, nearest_boundary, rel_tol=0, abs_tol=1e-10):
        candidates = (nearest_boundary - 1, nearest_boundary)
    else:
        candidates = (math.floor(position),)
    return tuple(index for index in candidates if 0 <= index < count)


def _immutable_float_array(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _require_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")


def _require_usable_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
