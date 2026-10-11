from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

import numpy as np
import osmnx as ox
from pyproj import CRS
from shapely.geometry import Point

from lima_pm25.domain import MobilityPolicy, SpatialDomain


class TrajectoryValidationError(ValueError):
    """Raised when supplied trajectory data violates its declared contract."""


class TrajectoryCoverageError(LookupError):
    """Raised when no explicit position exists at a requested due time."""


class MobilePositionProvider(Protocol):
    provider_id: str
    crs: str

    def position_at(
        self,
        sensor_id: str,
        simulation_time_s: float,
    ) -> tuple[float, float]: ...


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    sensor_id: str
    simulation_time_s: float
    x_m: float
    y_m: float

    def __post_init__(self) -> None:
        _require_text("sensor_id", self.sensor_id)
        _require_nonnegative_finite("simulation_time_s", self.simulation_time_s)
        _require_finite("x_m", self.x_m)
        _require_finite("y_m", self.y_m)


class TimestampedTrajectoryProvider:
    def __init__(
        self,
        *,
        provider_id: str,
        domain: SpatialDomain,
        mobility_policy: MobilityPolicy,
        points: Sequence[TrajectoryPoint],
        resource_id: str | None = None,
        resource_sha256: str | None = None,
    ) -> None:
        _require_text("provider_id", provider_id)
        if resource_id is not None:
            _require_text("resource_id", resource_id)
        if resource_sha256 is not None:
            _require_text("resource_sha256", resource_sha256)
        trajectory = tuple(points)
        if not trajectory:
            raise TrajectoryValidationError("Trajectory points cannot be empty")

        sensor_id = trajectory[0].sensor_id
        previous_time = -math.inf
        for index, point in enumerate(trajectory):
            if point.sensor_id != sensor_id:
                raise TrajectoryValidationError(
                    f"Trajectory record {index} has mixed sensor identity "
                    f"{point.sensor_id!r}; expected {sensor_id!r}"
                )
            if point.simulation_time_s <= previous_time:
                raise TrajectoryValidationError(
                    f"Trajectory record {index} timestamp must be strictly increasing"
                )
            previous_time = point.simulation_time_s

        _validate_policy_compliance(
            trajectory,
            domain=domain,
            policy=mobility_policy,
        )

        self.provider_id = provider_id
        self.crs = CRS.from_user_input(domain.grid.crs).to_string()
        self.sensor_id = sensor_id
        self.mobility_policy = mobility_policy
        self.points = trajectory
        self.resource_id = resource_id
        self.resource_sha256 = resource_sha256
        self._points_by_time = {point.simulation_time_s: point for point in trajectory}

    @property
    def coverage_start_s(self) -> float:
        return self.points[0].simulation_time_s

    @property
    def coverage_end_s(self) -> float:
        return self.points[-1].simulation_time_s

    def validate_due_times(
        self,
        sensor_id: str,
        due_times_s: Sequence[float],
    ) -> None:
        self._require_sensor(sensor_id)
        missing: list[float] = []
        for due_time_s in due_times_s:
            _require_nonnegative_finite("due_time_s", due_time_s)
            if due_time_s not in self._points_by_time:
                missing.append(due_time_s)
        if missing:
            formatted = ", ".join(f"{time_s:g}" for time_s in missing)
            raise TrajectoryCoverageError(
                f"Trajectory has no explicit position at due time(s): {formatted} s"
            )

    def position_at(
        self,
        sensor_id: str,
        simulation_time_s: float,
    ) -> tuple[float, float]:
        self._require_sensor(sensor_id)
        _require_nonnegative_finite("simulation_time_s", simulation_time_s)
        try:
            point = self._points_by_time[simulation_time_s]
        except KeyError as error:
            raise TrajectoryCoverageError(
                "Trajectory has no explicit position for sensor "
                f"{sensor_id!r} at {simulation_time_s:g} s"
            ) from error
        return point.x_m, point.y_m

    def evidence_descriptor(self) -> dict[str, object]:
        return {
            "type": "timestamped_replay",
            "provider_id": self.provider_id,
            "sensor_id": self.sensor_id,
            "crs": self.crs,
            "resource_id": self.resource_id,
            "resource_sha256": self.resource_sha256,
            "mobility_policy": {
                "road_tolerance_m": self.mobility_policy.road_tolerance_m,
                "maximum_speed_m_s": self.mobility_policy.maximum_speed_m_s,
            },
            "points": [
                {
                    "simulation_time_s": point.simulation_time_s,
                    "x_m": point.x_m,
                    "y_m": point.y_m,
                }
                for point in self.points
            ],
        }

    def _require_sensor(self, sensor_id: str) -> None:
        if sensor_id != self.sensor_id:
            raise TrajectoryCoverageError(
                f"Trajectory belongs to sensor {self.sensor_id!r}, not {sensor_id!r}"
            )


def _validate_policy_compliance(
    points: tuple[TrajectoryPoint, ...],
    *,
    domain: SpatialDomain,
    policy: MobilityPolicy,
) -> None:
    permitted_area = domain.boundary.buffer(domain.config.road_network_buffer_m)
    for index, point in enumerate(points):
        if not permitted_area.covers(Point(point.x_m, point.y_m)):
            raise TrajectoryValidationError(
                f"Trajectory record {index} exceeds road_network_buffer_m"
            )

    x = np.asarray([point.x_m for point in points], dtype=np.float64)
    y = np.asarray([point.y_m for point in points], dtype=np.float64)
    _, distances = ox.distance.nearest_edges(
        domain.road_graph,
        x,
        y,
        return_dist=True,
    )
    for index, distance in enumerate(np.atleast_1d(distances)):
        if float(distance) > policy.road_tolerance_m:
            raise TrajectoryValidationError(
                f"Trajectory record {index} is {float(distance):g} m from road "
                f"support, exceeding road_tolerance_m={policy.road_tolerance_m:g}"
            )

    if policy.maximum_speed_m_s is None:
        return
    for index, (first, second) in enumerate(pairwise(points)):
        duration_s = second.simulation_time_s - first.simulation_time_s
        distance_m = math.hypot(second.x_m - first.x_m, second.y_m - first.y_m)
        speed_m_s = distance_m / duration_s
        if speed_m_s > policy.maximum_speed_m_s:
            raise TrajectoryValidationError(
                f"Trajectory interval {index}-{index + 1} has speed "
                f"{speed_m_s:g} m/s, exceeding maximum_speed_m_s="
                f"{policy.maximum_speed_m_s:g}"
            )


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_nonnegative_finite(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
