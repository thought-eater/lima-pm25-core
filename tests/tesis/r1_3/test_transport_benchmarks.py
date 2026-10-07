"""Ensayos de transporte frente a referencias físicas y numéricas controladas."""

import math

import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from lima_pm25.domain import PM25SourceRole, build_grid
from lima_pm25.simulation.transport import BoundaryMode, PM25Transport, RoadLineSource, iter_simulation
from tests.transport_cases import (
    _baseline, _boundary, _cell_volume, _centroid, _forcing, _physics,
    _source, _square_grid, _x_variance,
)


def test_diffusion_conserves_mass_and_matches_gaussian_variance_growth() -> None:
    grid = _square_grid(side_cells=61)
    model = PM25Transport(grid, _physics(diffusivity_m2_s=20))
    source = _source(x_m=3_050, y_m=3_050, sigma_m=300)
    concentration = source.spatial_weights(grid) * 1e9 / _cell_volume(model)
    state = model.initialize_from_array(concentration)
    initial_variance = _x_variance(grid, concentration, source.x_m)

    result, budget = model.step(
        state,
        duration_s=600,
        forcing=_forcing(),
        boundary_mode=BoundaryMode.CLOSED,
    )
    final_variance = _x_variance(grid, result.concentration_ug_m3, source.x_m)

    assert budget.final_mass_ug == pytest.approx(budget.initial_mass_ug, rel=1e-12)
    assert budget.numerical_residual_ug == pytest.approx(0, abs=1e-5)
    assert final_variance - initial_variance == pytest.approx(2 * 20 * 600, rel=0.08)


@pytest.mark.parametrize(
    ("wind_from_direction_deg", "expected_x_m", "expected_y_m"),
    [
        pytest.param(0, 0, -300, id="south"),
        pytest.param(90, -300, 0, id="west"),
        pytest.param(180, 0, 300, id="north"),
        pytest.param(270, 300, 0, id="east"),
    ],
)
def test_advection_moves_gaussian_centroid_with_wind(
    wind_from_direction_deg: float,
    expected_x_m: float,
    expected_y_m: float,
) -> None:
    grid = _square_grid(side_cells=61)
    model = PM25Transport(grid, _physics())
    source = _source(x_m=3_050, y_m=3_050, sigma_m=250)
    concentration = source.spatial_weights(grid) * 1e9 / _cell_volume(model)
    initial_centroid = _centroid(grid, concentration)

    result, budget = model.step(
        model.initialize_from_array(concentration),
        duration_s=300,
        forcing=_forcing(
            wind_speed_m_s=1,
            wind_from_direction_deg=wind_from_direction_deg,
        ),
        boundary_mode=BoundaryMode.CLOSED,
    )
    final_centroid = _centroid(grid, result.concentration_ug_m3)

    assert final_centroid[0] - initial_centroid[0] == pytest.approx(
        expected_x_m,
        abs=1,
    )
    assert final_centroid[1] - initial_centroid[1] == pytest.approx(
        expected_y_m,
        abs=1,
    )
    assert budget.final_mass_ug == pytest.approx(budget.initial_mass_ug, rel=1e-12)


def test_open_boundary_records_outflow() -> None:
    grid = _square_grid(side_cells=21)
    model = PM25Transport(grid, _physics())
    source = _source(x_m=1_850, y_m=1_050, sigma_m=100)
    concentration = source.spatial_weights(grid) * 1e9 / _cell_volume(model)

    _, budget = model.step(
        model.initialize_from_array(concentration),
        duration_s=300,
        forcing=_forcing(wind_speed_m_s=2, wind_from_direction_deg=270),
        regional_boundary=_boundary(0),
        boundary_mode=BoundaryMode.REGIONAL_INFLOW,
    )

    assert budget.boundary_outflow_mass_ug > 0
    assert budget.boundary_inflow_mass_ug == 0
    assert abs(budget.relative_residual) < 1e-12


def test_irregular_boundary_admits_only_expected_regional_inflow() -> None:
    grid = build_grid(
        Polygon([(0, 0), (500, 0), (0, 500)]),
        "EPSG:32718",
        100,
    )
    mixing_height_m = 100
    wind_speed_m_s = 1
    boundary_concentration_ug_m3 = 10
    duration_s = 10
    model = PM25Transport(grid, _physics(mixing_height_m=mixing_height_m))

    result, budget = model.step(
        model.initialize_uniform(_baseline(0)),
        duration_s=duration_s,
        forcing=_forcing(
            wind_speed_m_s=wind_speed_m_s,
            wind_from_direction_deg=270,
        ),
        regional_boundary=_boundary(boundary_concentration_ug_m3),
        boundary_mode=BoundaryMode.REGIONAL_INFLOW,
    )

    active_rows = np.count_nonzero(np.any(grid.active_mask, axis=1))
    expected_inflow_ug = (
        active_rows
        * wind_speed_m_s
        * grid.cell_size_m
        * mixing_height_m
        * boundary_concentration_ug_m3
        * duration_s
    )
    assert np.any(~grid.active_mask)
    assert np.all(result.concentration_ug_m3[~grid.active_mask] == 0)
    assert budget.boundary_inflow_mass_ug == pytest.approx(expected_inflow_ug)
    assert budget.final_mass_ug == pytest.approx(expected_inflow_ug)
    assert budget.emitted_mass_ug == 0
    assert budget.boundary_outflow_mass_ug == 0
    assert budget.dry_deposited_mass_ug == 0
    assert budget.wet_deposited_mass_ug == 0
    assert abs(budget.relative_residual) < 1e-10


def test_dry_deposition_matches_exponential_solution() -> None:
    mixing_height = 200
    deposition_velocity = 0.002
    duration = 3_600
    model = PM25Transport(
        _square_grid(),
        _physics(
            mixing_height_m=mixing_height,
            dry_deposition_velocity_m_s=deposition_velocity,
        ),
    )
    initial = model.initialize_uniform(_baseline(10))
    initial_mass = model.total_mass_ug(initial.concentration_ug_m3)

    result, budget = model.step(
        initial,
        duration_s=duration,
        forcing=_forcing(),
        boundary_mode=BoundaryMode.CLOSED,
    )
    expected_mass = initial_mass * math.exp(
        -deposition_velocity / mixing_height * duration
    )

    assert budget.final_mass_ug == pytest.approx(expected_mass, rel=1e-12)
    assert budget.dry_deposited_mass_ug == pytest.approx(
        initial_mass - expected_mass,
        rel=1e-12,
    )
    assert np.all(result.concentration_ug_m3[model.grid.active_mask] >= 0)


def test_wet_deposition_depends_on_precipitation() -> None:
    coefficient = 0.04
    precipitation = 3
    duration = 3_600
    model = PM25Transport(
        _square_grid(),
        _physics(wet_scavenging_per_mm=coefficient),
    )
    initial = model.initialize_uniform(_baseline(10))
    initial_mass = model.total_mass_ug(initial.concentration_ug_m3)

    dry_result, dry_budget = model.step(
        initial,
        duration_s=duration,
        forcing=_forcing(precipitation_mm_h=0),
        boundary_mode=BoundaryMode.CLOSED,
    )
    wet_result, wet_budget = model.step(
        initial,
        duration_s=duration,
        forcing=_forcing(precipitation_mm_h=precipitation),
        boundary_mode=BoundaryMode.CLOSED,
    )
    expected_wet_mass = initial_mass * math.exp(-coefficient * precipitation)

    assert dry_budget.wet_deposited_mass_ug == 0
    assert np.array_equal(dry_result.concentration_ug_m3, initial.concentration_ug_m3)
    assert wet_budget.final_mass_ug == pytest.approx(expected_wet_mass, rel=1e-12)
    assert wet_budget.wet_deposited_mass_ug == pytest.approx(
        initial_mass - expected_wet_mass,
        rel=1e-12,
    )
    assert np.all(wet_result.concentration_ug_m3[model.grid.active_mask] >= 0)


def test_automatic_substeps_keep_full_budget_balanced_and_nonnegative() -> None:
    grid = build_grid(
        Polygon([(0, 0), (4_100, 0), (0, 4_100)]),
        "EPSG:32718",
        100,
    )
    model = PM25Transport(
        grid,
        _physics(
            diffusivity_m2_s=50,
            dry_deposition_velocity_m_s=0.001,
            wet_scavenging_per_mm=0.02,
        ),
    )
    source = _source(
        x_m=1_050,
        y_m=1_050,
        emission_rate_ug_s=50_000,
    )

    result, budget = model.step(
        model.initialize_uniform(_baseline(8)),
        duration_s=3_600,
        forcing=_forcing(
            wind_speed_m_s=8,
            wind_from_direction_deg=315,
            precipitation_mm_h=2,
        ),
        regional_boundary=_boundary(8),
        sources=(source,),
        boundary_mode=BoundaryMode.REGIONAL_INFLOW,
    )

    assert budget.substeps > 1
    assert budget.substeps * budget.substep_duration_s == pytest.approx(3_600)
    assert result.simulation_time_s == 3_600
    assert np.any(~model.grid.active_mask)
    assert np.all(result.concentration_ug_m3[model.grid.active_mask] >= 0)
    assert np.all(result.concentration_ug_m3[~model.grid.active_mask] == 0)
    assert budget.emitted_mass_ug > 0
    assert budget.boundary_inflow_mass_ug > 0
    assert budget.boundary_outflow_mass_ug > 0
    assert budget.dry_deposited_mass_ug > 0
    assert budget.wet_deposited_mass_ug > 0
    assert abs(budget.relative_residual) < 1e-12
    assert budget.source_emissions[0].emitted_mass_ug == pytest.approx(
        budget.emitted_mass_ug
    )
    assert budget.final_mass_ug == pytest.approx(
        budget.expected_final_mass_ug + budget.numerical_residual_ug
    )


def test_counterfactual_line_event_changes_only_source_inputs() -> None:
    model = PM25Transport(_square_grid(side_cells=5), _physics())
    baseline = _baseline(8)
    boundary = _boundary(8)
    forcing = _forcing(wind_speed_m_s=1, wind_from_direction_deg=270)
    initial = model.initialize_uniform(baseline)
    event = RoadLineSource(
        source_id="rush-hour-event",
        line=LineString([(150, 250), (350, 250)]),
        emission_rate_ug_m_s=2,
        provenance="synthetic rush-hour stress test",
        role=PM25SourceRole.COUNTERFACTUAL_EVENT,
    )

    arguments = {
        "duration_s": 60,
        "frame_duration_s": 60,
        "forcing_at": lambda _time_s: forcing,
        "regional_boundary_at": lambda _time_s: boundary,
        "boundary_mode": BoundaryMode.REGIONAL_INFLOW,
    }
    baseline_frames = tuple(iter_simulation(model, initial, **arguments))
    event_frames = tuple(
        iter_simulation(model, initial, sources=(event,), **arguments)
    )
    baseline_state = baseline_frames[-1].state
    event_state = event_frames[-1].state
    baseline_budget = baseline_frames[-1].budget
    event_budget = event_frames[-1].budget

    assert baseline_budget is not None
    assert event_budget is not None

    assert baseline_state.ambient_baseline is baseline
    assert event_state.ambient_baseline is baseline
    assert [frame.forcing for frame in baseline_frames] == [
        frame.forcing for frame in event_frames
    ]
    assert [frame.regional_boundary for frame in baseline_frames] == [
        frame.regional_boundary for frame in event_frames
    ]
    assert baseline_budget.source_emissions == ()
    assert len(event_budget.source_emissions) == 1
    assert event_budget.source_emissions[0].source_id == "rush-hour-event"
    assert event_frames[-1].sources[0].role is PM25SourceRole.COUNTERFACTUAL_EVENT
    assert event_budget.initial_mass_ug == baseline_budget.initial_mass_ug
    assert event_budget.boundary_inflow_mass_ug == (
        baseline_budget.boundary_inflow_mass_ug
    )
    assert event_budget.boundary_outflow_mass_ug == (
        baseline_budget.boundary_outflow_mass_ug
    )
    assert event_budget.emitted_mass_ug > baseline_budget.emitted_mass_ug
    assert not np.array_equal(
        event_state.concentration_ug_m3,
        baseline_state.concentration_ug_m3,
    )


def test_timestep_refinement_produces_convergent_fields() -> None:
    grid = _square_grid(side_cells=41)
    coarse = PM25Transport(
        grid,
        _physics(diffusivity_m2_s=20, cfl_safety_factor=0.8),
    )
    refined = PM25Transport(
        grid,
        _physics(diffusivity_m2_s=20, cfl_safety_factor=0.4),
    )
    fine = PM25Transport(
        grid,
        _physics(diffusivity_m2_s=20, cfl_safety_factor=0.2),
    )
    source = _source(sigma_m=250)
    concentration = source.spatial_weights(grid) * 1e9 / _cell_volume(coarse)
    forcing = _forcing(wind_speed_m_s=1, wind_from_direction_deg=270)

    coarse_state, _ = coarse.step(
        coarse.initialize_from_array(concentration),
        duration_s=600,
        forcing=forcing,
        boundary_mode=BoundaryMode.CLOSED,
    )
    refined_state, _ = refined.step(
        refined.initialize_from_array(concentration),
        duration_s=600,
        forcing=forcing,
        boundary_mode=BoundaryMode.CLOSED,
    )
    fine_state, _ = fine.step(
        fine.initialize_from_array(concentration),
        duration_s=600,
        forcing=forcing,
        boundary_mode=BoundaryMode.CLOSED,
    )
    coarse_difference = np.sum(
        np.abs(
            coarse_state.concentration_ug_m3
            - refined_state.concentration_ug_m3
        )
    )
    refined_difference = np.sum(
        np.abs(
            refined_state.concentration_ug_m3
            - fine_state.concentration_ug_m3
        )
    )
    reference = np.sum(np.abs(fine_state.concentration_ug_m3))

    assert refined_difference < coarse_difference
    assert refined_difference / reference < 0.03
