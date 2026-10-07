from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Protocol

import numpy as np

from .spatial import GridDefinition


class PM25SourceRole(StrEnum):
    ROUTINE_ENVIRONMENTAL = "routine_environmental"
    SYNTHETIC_VALIDATION = "synthetic_validation"
    COUNTERFACTUAL_EVENT = "counterfactual_event"


@dataclass(frozen=True, slots=True)
class AtmosphericForcing:
    provider_id: str
    provenance: str
    wind_speed_m_s: float
    wind_from_direction_deg: float
    precipitation_mm_h: float

    def __post_init__(self) -> None:
        _require_usable_text("provider_id", self.provider_id)
        _require_usable_text("provenance", self.provenance)
        _require_nonnegative_finite("wind_speed_m_s", self.wind_speed_m_s)
        if not math.isfinite(self.wind_from_direction_deg):
            raise ValueError("wind_from_direction_deg must be finite")
        _require_nonnegative_finite("precipitation_mm_h", self.precipitation_mm_h)
        object.__setattr__(
            self,
            "wind_from_direction_deg",
            self.wind_from_direction_deg % 360,
        )

    @property
    def wind_components_m_s(self) -> tuple[float, float]:
        radians = math.radians(self.wind_from_direction_deg)
        eastward = -self.wind_speed_m_s * math.sin(radians)
        northward = -self.wind_speed_m_s * math.cos(radians)
        return eastward, northward


@dataclass(frozen=True, slots=True)
class AmbientBaseline:
    provider_id: str
    concentration_ug_m3: float
    provenance: str

    def __post_init__(self) -> None:
        _require_usable_text("provider_id", self.provider_id)
        _require_nonnegative_finite(
            "concentration_ug_m3",
            self.concentration_ug_m3,
        )
        _require_usable_text("provenance", self.provenance)


@dataclass(frozen=True, slots=True)
class RegionalBoundaryForcing:
    provider_id: str
    concentration_ug_m3: float
    provenance: str

    def __post_init__(self) -> None:
        _require_usable_text("provider_id", self.provider_id)
        _require_nonnegative_finite(
            "concentration_ug_m3",
            self.concentration_ug_m3,
        )
        _require_usable_text("provenance", self.provenance)


@dataclass(frozen=True, slots=True)
class PM25SourceContribution:
    mass_by_cell_ug: np.ndarray
    integrated_mass_ug: float
    unit: str = "ug"

    def __post_init__(self) -> None:
        array = np.asarray(self.mass_by_cell_ug, dtype=np.float64)
        mass = np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)
        object.__setattr__(self, "mass_by_cell_ug", mass)


class PM25SourceProvider(Protocol):
    source_id: str
    role: PM25SourceRole
    provenance: str

    @property
    def transition_times_s(self) -> tuple[float, ...]: ...

    def evidence_descriptor(self) -> dict[str, object]: ...

    def contribution_for_interval(
        self,
        grid: GridDefinition,
        *,
        start_s: float,
        duration_s: float,
    ) -> PM25SourceContribution: ...


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    source_id: str
    role: PM25SourceRole
    provenance: str


def _require_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")


def _require_usable_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
