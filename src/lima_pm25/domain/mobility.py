from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class MobilityPolicy:
    road_tolerance_m: float
    maximum_speed_m_s: float | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_finite("road_tolerance_m", self.road_tolerance_m)
        if self.maximum_speed_m_s is not None and (
            not math.isfinite(self.maximum_speed_m_s) or self.maximum_speed_m_s <= 0
        ):
            raise ValueError("maximum_speed_m_s must be positive and finite")


def _require_nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")
