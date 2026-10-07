from dataclasses import FrozenInstanceError, fields

import pytest

from lima_pm25.domain import DomainConfig, MobilityPolicy


def test_road_policy_requires_tolerance_is_immutable_and_has_no_speed_default() -> None:
    policy = MobilityPolicy(road_tolerance_m=8)

    assert {entry.name for entry in fields(policy)} == {
        "road_tolerance_m",
        "maximum_speed_m_s",
    }
    assert DomainConfig().network_type == "drive"
    assert DomainConfig().road_network_buffer_m == 50
    assert policy.maximum_speed_m_s is None
    with pytest.raises(TypeError, match="road_tolerance_m"):
        MobilityPolicy()
    with pytest.raises(FrozenInstanceError):
        policy.road_tolerance_m = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    ("values", "message"),
    (
        ({"road_tolerance_m": -1}, "road_tolerance_m"),
        ({"road_tolerance_m": float("nan")}, "road_tolerance_m"),
        ({"road_tolerance_m": 8, "maximum_speed_m_s": 0}, "maximum_speed_m_s"),
        (
            {"road_tolerance_m": 8, "maximum_speed_m_s": float("inf")},
            "maximum_speed_m_s",
        ),
    ),
)
def test_mobility_policy_rejects_invalid_fields(
    values: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MobilityPolicy(**values)
