"""Mass references for daily activity and finite-area transient emissions."""

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
from shapely.geometry import LineString, box

from lima_pm25.simulation import (
    AreaSource,
    EmissionProfile,
    PM25Transport,
    RoadLineSource,
)
from tests.transport_cases import _forcing, _physics, _source, _square_grid


@pytest.mark.parametrize(
    ("interpolation", "expected"), [("constant", 48), ("linear", 56)]
)
def test_profile_integrates_across_knots_and_active_limits(interpolation, expected):
    # Constant: 2*4 + 4*4 = 24 s. Linear: (2+4)/2*4 + 4*4 = 28 s.
    source = _source(
        emission_rate_ug_s=2,
        starts_at_s=2,
        ends_at_s=10,
        profile=EmissionProfile(((0, 0), (2, 2), (6, 4), (10, 4)), interpolation),
    )
    mass = source.emitted_mass_ug(0, 12)
    assert mass == pytest.approx(expected)
    assert sum(
        source.emitted_mass_ug(t, 0.5) for t in np.arange(0, 12, 0.5)
    ) == pytest.approx(mass)
    assert source.emitted_mass_ug(12, 5) == 0


@pytest.mark.parametrize("interpolation", ["constant", "linear"])
def test_daily_profile_uses_lima_time_and_repeats_across_midnight(interpolation):
    start = datetime(2025, 1, 2, 3, tzinfo=UTC)  # 22:00 in Lima
    profile = EmissionProfile.daily(
        ((0, 0.5), (6, 0.5), (8, 2), (10, 2), (12, 1), (20, 1), (22, 0.5), (24, 0.5)),
        start_utc=start,
        duration_s=48 * 3600,
        interpolation=interpolation,
    )
    assert profile.value_at(0) == 0.5
    assert profile.value_at(10 * 3600) == 2  # local 08:00
    assert profile.value_at(34 * 3600) == 2
    assert profile.integral(0, 24 * 3600) == pytest.approx(
        profile.integral(24 * 3600, 48 * 3600)
    )


def test_transient_area_emits_only_active_intersection_and_conserves_mass():
    grid = _square_grid(side_cells=2, cell_size_m=10)
    mask = grid.active_mask.copy()
    mask[1, 0] = False
    grid = replace(grid, active_mask=mask)
    # 10 m wide intersection in each row; one row is inactive: 100 m².
    source = AreaSource(
        "area",
        box(-10, 0, 10, 20),
        3,
        "test stress",
        2,
        6,
        EmissionProfile(((0, 0), (2, 0), (4, 2), (6, 0)), "linear"),
    )
    contribution = source.contribution_for_interval(grid, start_s=0, duration_s=10)
    # Triangle: 4 s * multiplier 2 / 2 = 4 s; 100 m² * 3 * 4.
    assert contribution.integrated_mass_ug == pytest.approx(1200)
    assert contribution.mass_by_cell_ug[0, 0] == pytest.approx(1200)
    assert contribution.mass_by_cell_ug[1, 0] == 0
    model = PM25Transport(grid, _physics())
    state = model.initialize_from_array(np.zeros(grid.shape))
    final, budget = model.step(
        state, duration_s=10, forcing=_forcing(), sources=(source,)
    )
    assert budget.emitted_mass_ug == pytest.approx(1200)
    assert budget.final_mass_ug == pytest.approx(1200)
    assert budget.numerical_residual_ug == pytest.approx(0, abs=1e-10)
    assert np.all(final.concentration_ug_m3 >= 0)


def test_road_peak_and_additional_event_sum_without_sensor_discretization():
    grid = _square_grid(side_cells=4, cell_size_m=10)
    routine = RoadLineSource(
        "road",
        LineString([(5, 5), (35, 5)]),
        2,
        "fixture",
        profile=EmissionProfile(((0, 1), (3.5, 3), (8.25, 1))),
    )
    event = _source(
        source_id="event",
        x_m=15,
        y_m=15,
        sigma_m=4,
        emission_rate_ug_s=10,
        starts_at_s=4.25,
        ends_at_s=7.75,
    )
    model = PM25Transport(grid, _physics())
    state = model.initialize_from_array(np.zeros(grid.shape))
    _, budget = model.step(
        state, duration_s=10, forcing=_forcing(), sources=(routine, event)
    )
    expected_road = 30 * 2 * (3.5 + 3 * 4.75 + 1.75)
    assert budget.emitted_mass_ug == pytest.approx(expected_road + 35)
    assert {
        s.source_id: s.emitted_mass_ug for s in budget.source_emissions
    } == pytest.approx({"road": expected_road, "event": 35})


def test_area_mass_is_independent_of_horizontal_resolution():
    source = AreaSource("area", box(7, 11, 81, 83), 2, "fixture stress", 0, 10)
    masses = [
        source.contribution_for_interval(
            _square_grid(side_cells=int(100 / size), cell_size_m=size),
            start_s=0,
            duration_s=10,
        ).integrated_mass_ug
        for size in (10, 20, 50)
    ]
    assert masses == pytest.approx([74 * 72 * 20] * 3)
