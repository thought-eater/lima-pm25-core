import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, box

from lima_pm25.domain import (
    DomainConfig,
    GridDefinition,
    MobilityPolicy,
    SpatialDomain,
)
from lima_pm25.simulation import (
    TimestampedTrajectoryProvider,
    TrajectoryCoverageError,
    TrajectoryPoint,
    TrajectoryValidationError,
)


def test_exact_due_positions_are_returned_without_interpolation() -> None:
    provider = _provider(
        (
            TrajectoryPoint("MOBILE", 0, 10, 50),
            TrajectoryPoint("MOBILE", 60, 50, 50),
            TrajectoryPoint("MOBILE", 120, 120, 50),
        )
    )

    provider.validate_due_times("MOBILE", (0, 60, 120))
    assert provider.position_at("MOBILE", 120) == (120, 50)
    assert provider.position_at("MOBILE", 0) == (10, 50)
    assert provider.position_at("MOBILE", 120) == (120, 50)

    with pytest.raises(TrajectoryCoverageError, match="no explicit position"):
        provider.position_at("MOBILE", 30)


def test_due_time_coverage_and_sensor_identity_are_explicit() -> None:
    provider = _provider((TrajectoryPoint("MOBILE", 0, 10, 50),))

    with pytest.raises(TrajectoryCoverageError, match="60"):
        provider.validate_due_times("MOBILE", (0, 60))
    with pytest.raises(TrajectoryCoverageError, match="belongs to sensor"):
        provider.position_at("OTHER", 0)


def test_trajectory_rejects_mixed_sensor_identity_and_nonincreasing_time() -> None:
    with pytest.raises(TrajectoryValidationError, match="mixed sensor identity"):
        _provider(
            (
                TrajectoryPoint("MOBILE", 0, 10, 50),
                TrajectoryPoint("OTHER", 60, 20, 50),
            )
        )

    with pytest.raises(TrajectoryValidationError, match="strictly increasing"):
        _provider(
            (
                TrajectoryPoint("MOBILE", 60, 10, 50),
                TrajectoryPoint("MOBILE", 60, 20, 50),
            )
        )


def test_trajectory_rejects_offroad_and_excessively_external_positions() -> None:
    with pytest.raises(TrajectoryValidationError, match="road_tolerance_m"):
        _provider((TrajectoryPoint("MOBILE", 0, 20, 60),), road_tolerance_m=5)

    with pytest.raises(TrajectoryValidationError, match="road_network_buffer_m"):
        _provider((TrajectoryPoint("MOBILE", 0, 151, 50),))


def test_domain_buffer_controls_the_permitted_road_excursion() -> None:
    point = (TrajectoryPoint("MOBILE", 0, 120, 50),)
    assert _provider(point, road_network_buffer_m=50).position_at("MOBILE", 0) == (
        120,
        50,
    )
    with pytest.raises(TrajectoryValidationError, match="road_network_buffer_m"):
        _provider(point, road_network_buffer_m=0)


def test_optional_maximum_speed_is_applied_only_when_configured() -> None:
    points = (
        TrajectoryPoint("MOBILE", 0, 10, 50),
        TrajectoryPoint("MOBILE", 1, 30, 50),
    )

    _provider(points, maximum_speed_m_s=None)
    with pytest.raises(TrajectoryValidationError, match="maximum_speed_m_s"):
        _provider(points, maximum_speed_m_s=10)


def test_repeated_providers_replay_the_same_positions() -> None:
    points = (
        TrajectoryPoint("MOBILE", 0, 10, 50),
        TrajectoryPoint("MOBILE", 60, 50, 50),
    )
    first = _provider(points)
    second = _provider(points)

    assert first.provider_id == second.provider_id == "trajectory-provider"
    first_replay = tuple(first.position_at("MOBILE", time_s) for time_s in (60, 0))
    second_replay = tuple(second.position_at("MOBILE", time_s) for time_s in (60, 0))
    assert first_replay == second_replay


def test_optional_input_resource_identity_is_kept_in_replay_evidence() -> None:
    domain = _domain()
    provider = TimestampedTrajectoryProvider(
        provider_id="trajectory-provider",
        domain=domain,
        mobility_policy=MobilityPolicy(road_tolerance_m=2),
        points=(TrajectoryPoint("MOBILE", 0, 10, 50),),
        resource_id="mobile-route.csv",
        resource_sha256="abc123",
    )

    evidence = provider.evidence_descriptor()
    assert evidence["resource_id"] == "mobile-route.csv"
    assert evidence["resource_sha256"] == "abc123"


def _provider(
    points: tuple[TrajectoryPoint, ...],
    *,
    road_tolerance_m: float = 2,
    maximum_speed_m_s: float | None = None,
    road_network_buffer_m: float = 50,
) -> TimestampedTrajectoryProvider:
    domain = _domain(road_network_buffer_m=road_network_buffer_m)
    policy = MobilityPolicy(
        road_tolerance_m=road_tolerance_m,
        maximum_speed_m_s=maximum_speed_m_s,
    )
    return TimestampedTrajectoryProvider(
        provider_id="trajectory-provider",
        domain=domain,
        mobility_policy=policy,
        points=points,
    )


def _domain(*, road_network_buffer_m: float = 50) -> SpatialDomain:
    config = DomainConfig(
        left=-77.01,
        bottom=-12.01,
        right=-77,
        top=-12,
        road_network_buffer_m=road_network_buffer_m,
    )
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:32718"
    graph.add_node(1, x=-50.0, y=50.0)
    graph.add_node(2, x=150.0, y=50.0)
    graph.add_edge(
        1,
        2,
        key=0,
        length=200.0,
        geometry=LineString([(-50, 50), (150, 50)]),
    )
    grid = GridDefinition(
        crs="EPSG:32718",
        domain_bounds_m=np.array([0, 0, 100, 100]),
        grid_bounds_m=np.array([0, 0, 100, 100]),
        cell_size_m=100,
        x_centers_m=np.array([50]),
        y_centers_m=np.array([50]),
        active_mask=np.ones((1, 1), dtype=bool),
    )
    return SpatialDomain(
        config=config,
        road_graph=graph,
        boundary=box(0, 0, 100, 100),
        grid=grid,
        graph_source="test",
        grid_source="test",
    )
