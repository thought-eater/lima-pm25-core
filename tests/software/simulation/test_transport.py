import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon, box

from lima_pm25.domain import (
    AtmosphericForcing,
    DomainConfig,
    GridDefinition,
    PM25SourceContribution,
    PM25SourceRole,
    RegionalBoundaryForcing,
    SpatialDomain,
    build_grid,
)
from lima_pm25.simulation import demo as demo_module
from lima_pm25.simulation.demo import (
    _build_pm25_summary_figure,
    _create_synthetic_demo,
    _demo_playback_interval_ms,
    _road_corridor_by_name,
    _synthetic_demo_forcing,
    run_synthetic_demo,
)
from lima_pm25.simulation.transport import (
    BoundaryMode,
    PM25State,
    PM25Transport,
    RoadLineSource,
    SimulationFrame,
    iter_simulation,
)
from tests.transport_cases import (
    _baseline,
    _boundary,
    _cell_volume,
    _forcing,
    _physics,
    _source,
    _square_grid,
)


def test_physics_config_requires_explicit_valid_parameters() -> None:
    with pytest.raises(ValueError, match="mixing_height_m"):
        _physics(mixing_height_m=0)
    with pytest.raises(ValueError, match="diffusivity_m2_s"):
        _physics(diffusivity_m2_s=-1)
    with pytest.raises(ValueError, match="dry_deposition_velocity"):
        _physics(dry_deposition_velocity_m_s=-1)
    with pytest.raises(ValueError, match="wet_scavenging"):
        _physics(wet_scavenging_per_mm=-1)


def test_pm25_state_validates_grid_values_and_inactive_cells_on_creation() -> None:
    grid = build_grid(
        Polygon([(0, 0), (300, 0), (0, 300)]),
        "EPSG:32718",
        100,
    )
    valid = np.zeros(grid.shape)
    valid[grid.active_mask] = 1

    with pytest.raises(ValueError, match="shape"):
        PM25State(grid, np.zeros((1, 1)), simulation_time_s=0)

    invalid = valid.copy()
    invalid[grid.active_mask] = np.nan
    with pytest.raises(ValueError, match="finite"):
        PM25State(grid, invalid, simulation_time_s=0)

    invalid = valid.copy()
    invalid[grid.active_mask] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        PM25State(grid, invalid, simulation_time_s=0)

    invalid = valid.copy()
    invalid[~grid.active_mask] = 1
    with pytest.raises(ValueError, match="Inactive"):
        PM25State(grid, invalid, simulation_time_s=0)


def test_pm25_state_owns_an_immutable_concentration_copy() -> None:
    grid = _square_grid(side_cells=2)
    concentration = np.ones(grid.shape)
    state = PM25State(grid, concentration, simulation_time_s=0)

    concentration[:] = 2

    assert np.all(state.concentration_ug_m3 == 1)
    with pytest.raises(ValueError, match="WRITEABLE"):
        state.concentration_ug_m3.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        state.concentration_ug_m3[0, 0] = 2
    with pytest.raises(ValueError, match="WRITEABLE"):
        grid.active_mask.setflags(write=True)


def test_transport_rejects_state_from_a_different_grid() -> None:
    model = PM25Transport(_square_grid(side_cells=2), _physics())
    other_grid = _square_grid(side_cells=2, cell_size_m=50)
    state = PM25State(other_grid, np.zeros(other_grid.shape), simulation_time_s=0)

    with pytest.raises(ValueError, match="state grid"):
        model.step(state, duration_s=60, forcing=_forcing())


def test_source_contribution_owns_an_immutable_mass_copy() -> None:
    mass = np.array([[10.0]])
    contribution = PM25SourceContribution(mass, 10)

    mass[0, 0] = 20

    assert contribution.mass_by_cell_ug[0, 0] == 10
    with pytest.raises(ValueError, match="WRITEABLE"):
        contribution.mass_by_cell_ug.setflags(write=True)


@pytest.mark.parametrize(
    ("direction", "expected_east", "expected_north"),
    [
        (0, 0, -2),
        (90, -2, 0),
        (180, 0, 2),
        (270, 2, 0),
    ],
)
def test_meteorological_wind_conversion(
    direction: float,
    expected_east: float,
    expected_north: float,
) -> None:
    forcing = _forcing(wind_speed_m_s=2, wind_from_direction_deg=direction)

    east, north = forcing.wind_components_m_s

    assert east == pytest.approx(expected_east, abs=1e-12)
    assert north == pytest.approx(expected_north, abs=1e-12)


def test_gaussian_source_weights_are_normalized() -> None:
    model = PM25Transport(_square_grid(), _physics())
    source = _source()

    weights = source.spatial_weights(model.grid)

    assert weights.shape == model.grid.shape
    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1)
    assert weights[20, 20] == weights.max()


def test_source_only_injects_exact_mass_and_respects_schedule() -> None:
    model = PM25Transport(_square_grid(), _physics())
    source = _source(
        emission_rate_ug_s=1_000,
        starts_at_s=50,
        ends_at_s=150,
    )
    state = model.initialize_uniform(_baseline(0))

    state, budget = model.step(
        state,
        duration_s=200,
        forcing=_forcing(),
        sources=(source,),
        boundary_mode=BoundaryMode.CLOSED,
    )

    assert budget.emitted_mass_ug == pytest.approx(100_000)
    assert budget.source_emissions[0].source_id == "source"
    assert budget.source_emissions[0].emitted_mass_ug == pytest.approx(100_000)
    assert budget.final_mass_ug == pytest.approx(100_000, rel=1e-12)
    assert model.total_mass_ug(state.concentration_ug_m3) == pytest.approx(100_000)
    assert budget.numerical_residual_ug == pytest.approx(0, abs=1e-7)


def test_multiple_source_emissions_are_additive() -> None:
    model = PM25Transport(_square_grid(), _physics())
    sources = (
        _source(source_id="a", emission_rate_ug_s=1_000),
        RoadLineSource(
            source_id="b",
            line=LineString([(50, 250), (450, 250)]),
            emission_rate_ug_m_s=5,
            provenance="synthetic road test",
        ),
    )

    _, budget = model.step(
        model.initialize_uniform(_baseline(0)),
        duration_s=60,
        forcing=_forcing(),
        sources=sources,
        boundary_mode=BoundaryMode.CLOSED,
    )

    assert budget.emitted_mass_ug == pytest.approx(180_000)
    assert budget.final_mass_ug == pytest.approx(180_000, rel=1e-12)
    assert [entry.source_id for entry in budget.source_emissions] == ["a", "b"]
    assert sum(entry.emitted_mass_ug for entry in budget.source_emissions) == (
        pytest.approx(budget.emitted_mass_ug)
    )


def test_road_line_source_distributes_exact_scheduled_mass_by_length() -> None:
    grid = _square_grid(side_cells=5)
    source = RoadLineSource(
        source_id="rush-hour-road",
        line=LineString([(50, 250), (450, 250)]),
        emission_rate_ug_m_s=2,
        provenance="synthetic traffic scenario",
        starts_at_s=30,
        ends_at_s=90,
    )

    weights, active_length = source.spatial_weights(grid)
    contribution = source.contribution_for_interval(
        grid,
        start_s=0,
        duration_s=120,
    )

    assert active_length == pytest.approx(400)
    assert np.count_nonzero(weights) == 5
    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1)
    assert contribution.integrated_mass_ug == pytest.approx(48_000)
    assert contribution.mass_by_cell_ug.sum() == pytest.approx(48_000)


def test_road_line_cache_recomputes_if_its_mutable_geometry_changes() -> None:
    grid = _square_grid(side_cells=2)
    source = RoadLineSource(
        source_id="road",
        line=LineString([(0, 50), (100, 50)]),
        emission_rate_ug_m_s=1,
        provenance="synthetic cache test",
    )

    first_weights, first_length = source.spatial_weights(grid)
    source.line = LineString([(0, 150), (200, 150)])
    second_weights, second_length = source.spatial_weights(grid)

    assert first_length == pytest.approx(100)
    assert first_weights[0, 0] == pytest.approx(1)
    assert second_length == pytest.approx(200)
    assert second_weights[1] == pytest.approx([0.5, 0.5])
    assert second_weights[0].sum() == 0
    assert source.spatial_weights(grid)[0] == pytest.approx(second_weights)


def test_road_line_source_outside_active_domain_is_rejected() -> None:
    grid = _square_grid(side_cells=5)
    source = RoadLineSource(
        source_id="outside-road",
        line=LineString([(-200, -100), (-100, -100)]),
        emission_rate_ug_m_s=2,
        provenance="synthetic traffic scenario",
    )

    with pytest.raises(ValueError, match="outside the active domain"):
        source.contribution_for_interval(grid, start_s=0, duration_s=60)


def test_road_line_source_clips_emitting_length_to_active_grid_cells() -> None:
    grid = _square_grid(side_cells=3)
    source = RoadLineSource(
        source_id="partly-outside-road",
        line=LineString([(-100, 150), (250, 150)]),
        emission_rate_ug_m_s=2,
        provenance="synthetic traffic scenario",
    )

    contribution = source.contribution_for_interval(
        grid,
        start_s=0,
        duration_s=10,
    )

    assert contribution.integrated_mass_ug == pytest.approx(5_000)
    assert contribution.mass_by_cell_ug.sum() == pytest.approx(5_000)
    assert np.all(contribution.mass_by_cell_ug[~grid.active_mask] == 0)


def test_road_line_on_shared_faces_splits_weight_deterministically() -> None:
    grid = _square_grid(side_cells=2)
    source = RoadLineSource(
        source_id="shared-face-road",
        line=LineString([(0, 100), (200, 100)]),
        emission_rate_ug_m_s=1,
        provenance="synthetic traffic scenario",
    )

    weights, active_length = source.spatial_weights(grid)

    assert active_length == pytest.approx(200)
    assert weights == pytest.approx(np.full((2, 2), 0.25))
    assert weights.sum() == pytest.approx(1)


def test_road_line_weights_shared_and_single_cell_segments_by_local_length() -> None:
    grid = _square_grid(side_cells=2)
    diagonal_length = math.sqrt(50**2 + 50**2)
    source = RoadLineSource(
        source_id="mixed-overlap-road",
        line=LineString([(0, 100), (100, 100), (150, 150)]),
        emission_rate_ug_m_s=1,
        provenance="synthetic traffic scenario",
    )

    weights, active_length = source.spatial_weights(grid)

    assert active_length == pytest.approx(100 + diagonal_length)
    assert weights[0, 0] == pytest.approx(50 / active_length)
    assert weights[1, 0] == pytest.approx(50 / active_length)
    assert weights[1, 1] == pytest.approx(diagonal_length / active_length)
    assert weights[0, 1] == 0


def test_source_outside_active_domain_is_rejected() -> None:
    model = PM25Transport(_square_grid(), _physics())
    source = _source(x_m=-1)

    with pytest.raises(ValueError, match="outside the active domain"):
        model.step(
            model.initialize_uniform(_baseline(0)),
            duration_s=60,
            forcing=_forcing(),
            sources=(source,),
        )


def test_source_provider_metadata_and_unique_identity_are_required() -> None:
    model = PM25Transport(_square_grid(side_cells=2), _physics())
    mass = np.zeros(model.grid.shape)
    contribution = PM25SourceContribution(mass, 0)
    invalid_sources = (
        _StaticSource(" ", contribution),
        _StaticSource("source", contribution, provenance=""),
        _StaticSource("source", contribution, role="unknown"),  # ty: ignore[invalid-argument-type] -- invalid role
    )

    for source in invalid_sources:
        with pytest.raises(ValueError, match="source provider|source_id|provenance"):
            model.step(
                model.initialize_uniform(_baseline(0)),
                duration_s=60,
                forcing=_forcing(),
                sources=(source,),
            )

    duplicate_sources = (
        _StaticSource("duplicate", contribution),
        _StaticSource("duplicate", contribution),
    )
    with pytest.raises(ValueError, match="Duplicate source_id"):
        model.step(
            model.initialize_uniform(_baseline(0)),
            duration_s=60,
            forcing=_forcing(),
            sources=duplicate_sources,
        )


def test_malformed_source_contributions_are_rejected_before_injection() -> None:
    grid = build_grid(
        Polygon([(0, 0), (300, 0), (0, 300)]),
        "EPSG:32718",
        100,
    )
    model = PM25Transport(grid, _physics())
    active_cell = tuple(np.argwhere(grid.active_mask)[0])
    inactive_cell = tuple(np.argwhere(~grid.active_mask)[0])
    valid = np.zeros(grid.shape)
    valid[active_cell] = 10

    wrong_shape = PM25SourceContribution(np.zeros((1, 1)), 0)
    wrong_unit = PM25SourceContribution(valid, 10, unit="kg")
    nonfinite = valid.copy()
    nonfinite[active_cell] = np.nan
    negative = valid.copy()
    negative[active_cell] = -1
    inactive = valid.copy()
    inactive[inactive_cell] = 1
    cases = (
        (object(), "must return"),
        (wrong_shape, "shape"),
        (wrong_unit, "unit"),
        (PM25SourceContribution(nonfinite, 10), "finite"),
        (PM25SourceContribution(negative, -1), "nonnegative"),
        (PM25SourceContribution(inactive, 11), "inactive"),
        (PM25SourceContribution(valid, math.nan), "integrated mass"),
        (PM25SourceContribution(valid, 11), "does not match"),
    )

    for contribution, message in cases:
        with pytest.raises(ValueError, match=message):
            model.step(
                model.initialize_uniform(_baseline(0)),
                duration_s=60,
                forcing=_forcing(),
                sources=(_StaticSource("malformed", contribution),),  # ty: ignore[invalid-argument-type] -- includes non-contribution objects
            )


def test_structural_source_provider_injects_without_geometry_attributes() -> None:
    model = PM25Transport(_square_grid(side_cells=2), _physics())
    mass = np.zeros(model.grid.shape)
    mass[0, 0] = 25
    source = _StaticSource(
        "structural-source",
        PM25SourceContribution(mass, 25),
    )

    _, budget = model.step(
        model.initialize_uniform(_baseline(0)),
        duration_s=60,
        forcing=_forcing(),
        sources=(source,),
    )

    assert not hasattr(source, "x_m")
    assert budget.emitted_mass_ug == pytest.approx(25)
    assert budget.source_emissions[0].source_id == "structural-source"


def test_source_metadata_must_remain_stable_during_contribution() -> None:
    model = PM25Transport(_square_grid(side_cells=2), _physics())
    mass = np.zeros(model.grid.shape)

    class MutatingSource(_StaticSource):
        def contribution_for_interval(
            self,
            grid: GridDefinition,
            *,
            start_s: float,
            duration_s: float,
        ) -> PM25SourceContribution:
            self.source_id = "changed"
            return self.contribution

    source = MutatingSource("original", PM25SourceContribution(mass, 0))

    with pytest.raises(ValueError, match="metadata changed"):
        model.step(
            model.initialize_uniform(_baseline(0)),
            duration_s=60,
            forcing=_forcing(),
            sources=(source,),
        )


def test_empty_source_collection_never_creates_an_implicit_source() -> None:
    model = PM25Transport(_square_grid(side_cells=2), _physics())

    _, budget = model.step(
        model.initialize_uniform(_baseline(3)),
        duration_s=60,
        forcing=_forcing(),
        sources=(),
    )

    assert budget.emitted_mass_ug == 0
    assert budget.source_emissions == ()


def test_no_processes_leave_state_unchanged() -> None:
    model = PM25Transport(_square_grid(), _physics())
    concentration = np.zeros(model.grid.shape)
    concentration[model.grid.active_mask] = np.arange(
        np.count_nonzero(model.grid.active_mask)
    )
    state = model.initialize_from_array(concentration)

    result, budget = model.step(
        state,
        duration_s=3_600,
        forcing=_forcing(),
        boundary_mode=BoundaryMode.CLOSED,
    )

    assert np.array_equal(result.concentration_ug_m3, concentration)
    assert budget.numerical_residual_ug == pytest.approx(0)


def test_diffusion_exchanges_equal_mass_across_a_shared_face() -> None:
    grid = build_grid(box(0, 0, 200, 100), "EPSG:32718", 100)
    model = PM25Transport(grid, _physics(diffusivity_m2_s=10))
    concentration = np.array([[10.0, 0.0]])
    initial_cell_mass = concentration * _cell_volume(model)

    result, budget = model.step(
        model.initialize_from_array(concentration),
        duration_s=10,
        forcing=_forcing(),
    )
    final_cell_mass = result.concentration_ug_m3 * _cell_volume(model)

    assert initial_cell_mass[0, 0] - final_cell_mass[0, 0] == pytest.approx(
        final_cell_mass[0, 1] - initial_cell_mass[0, 1]
    )
    assert budget.final_mass_ug == pytest.approx(budget.initial_mass_ug)


def test_uniform_background_remains_uniform_with_open_boundaries() -> None:
    model = PM25Transport(_square_grid(), _physics(diffusivity_m2_s=15))
    initial = model.initialize_uniform(_baseline(12))

    result, budget = model.step(
        initial,
        duration_s=1_800,
        forcing=_forcing(wind_speed_m_s=3, wind_from_direction_deg=225),
        regional_boundary=_boundary(12),
        boundary_mode=BoundaryMode.REGIONAL_INFLOW,
    )

    assert result.concentration_ug_m3[model.grid.active_mask] == pytest.approx(12)
    assert budget.boundary_inflow_mass_ug > 0
    assert budget.boundary_outflow_mass_ug == pytest.approx(
        budget.boundary_inflow_mass_ug,
        rel=1e-12,
    )
    assert budget.numerical_residual_ug == pytest.approx(0, abs=1e-4)


def test_regional_boundary_is_independent_of_meteorological_forcing() -> None:
    model = PM25Transport(_square_grid(side_cells=5), _physics())
    forcing = _forcing(wind_speed_m_s=1, wind_from_direction_deg=270)
    initial = model.initialize_uniform(_baseline(12))

    assert not hasattr(forcing, "background_pm25_ug_m3")
    with pytest.raises(ValueError, match="regional_boundary"):
        model.step(
            initial,
            duration_s=60,
            forcing=forcing,
            boundary_mode=BoundaryMode.REGIONAL_INFLOW,
        )

    matched, matched_budget = model.step(
        initial,
        duration_s=60,
        forcing=forcing,
        regional_boundary=_boundary(12),
        boundary_mode=BoundaryMode.REGIONAL_INFLOW,
    )
    clean, clean_budget = model.step(
        initial,
        duration_s=60,
        forcing=forcing,
        regional_boundary=_boundary(0),
        boundary_mode=BoundaryMode.REGIONAL_INFLOW,
    )

    assert matched.concentration_ug_m3[model.grid.active_mask] == pytest.approx(12)
    assert matched_budget.boundary_inflow_mass_ug > 0
    assert clean_budget.boundary_inflow_mass_ug == 0
    assert not np.array_equal(
        matched.concentration_ug_m3,
        clean.concentration_ug_m3,
    )


def test_inactive_cells_stay_zero_and_repeated_steps_are_deterministic() -> None:
    grid = build_grid(
        Polygon([(0, 0), (2_000, 0), (0, 2_000)]),
        "EPSG:32718",
        100,
    )
    model = PM25Transport(grid, _physics(diffusivity_m2_s=20))
    source = _source(x_m=50, y_m=50, emission_rate_ug_s=10_000)
    forcing = _forcing(wind_speed_m_s=2, wind_from_direction_deg=225)
    initial = model.initialize_uniform(_baseline(5))

    first_state, first_budget = model.step(
        initial,
        duration_s=600,
        forcing=forcing,
        sources=(source,),
    )
    second_state, second_budget = model.step(
        initial,
        duration_s=600,
        forcing=forcing,
        sources=(source,),
    )

    assert np.any(~grid.active_mask)
    assert np.all(first_state.concentration_ug_m3[~grid.active_mask] == 0)
    assert np.array_equal(
        first_state.concentration_ug_m3,
        second_state.concentration_ug_m3,
    )
    assert first_state.simulation_time_s == second_state.simulation_time_s
    assert first_state.simulation_time_s == 600
    assert first_budget == second_budget
    assert first_budget.final_mass_ug == pytest.approx(
        first_budget.initial_mass_ug + first_budget.emitted_mass_ug,
        rel=1e-12,
    )
    assert abs(first_budget.relative_residual) < 1e-12


def test_incremental_runner_splits_at_forcing_changes_and_is_deterministic() -> None:
    model = PM25Transport(_square_grid(side_cells=5), _physics())
    initial = model.initialize_uniform(_baseline(5))

    def forcing_at(time_s: float) -> AtmosphericForcing:
        return _forcing(precipitation_mm_h=float(time_s >= 1_000))

    def run() -> tuple[SimulationFrame, ...]:
        return tuple(
            iter_simulation(
                model,
                initial,
                duration_s=2_000,
                frame_duration_s=700,
                forcing_at=forcing_at,
                forcing_change_times_s=(1_000,),
                boundary_mode=BoundaryMode.CLOSED,
            )
        )

    first = run()
    second = run()

    assert [frame.state.simulation_time_s for frame in first] == [
        0,
        700,
        1_000,
        1_700,
        2_000,
    ]
    assert first[2].forcing.precipitation_mm_h == 0
    assert first[3].forcing.precipitation_mm_h == 1
    assert all(
        np.array_equal(
            first_frame.state.concentration_ug_m3,
            second_frame.state.concentration_ug_m3,
        )
        for first_frame, second_frame in zip(first, second, strict=True)
    )


def test_simulation_frames_retain_independent_input_provenance() -> None:
    model = PM25Transport(_square_grid(side_cells=5), _physics())
    baseline = _baseline(5)
    boundary = _boundary(5)
    source = RoadLineSource(
        source_id="road",
        line=LineString([(50, 250), (450, 250)]),
        emission_rate_ug_m_s=1,
        provenance="synthetic road test",
    )

    frames = tuple(
        iter_simulation(
            model,
            model.initialize_uniform(baseline),
            duration_s=60,
            frame_duration_s=60,
            forcing_at=lambda _time_s: _forcing(),
            regional_boundary_at=lambda _time_s: boundary,
            sources=(source,),
            boundary_mode=BoundaryMode.REGIONAL_INFLOW,
        )
    )

    assert frames[0].state.ambient_baseline is baseline
    assert frames[1].forcing.provider_id == "test-meteorology"
    assert frames[1].forcing.provenance == "test fixture"
    assert frames[1].regional_boundary is boundary
    assert frames[1].sources[0].source_id == "road"
    assert frames[1].sources[0].role is PM25SourceRole.ROUTINE_ENVIRONMENTAL
    assert frames[1].sources[0].provenance == "synthetic road test"


def test_regional_boundary_changes_split_frames_independently_of_weather() -> None:
    model = PM25Transport(_square_grid(side_cells=5), _physics())
    initial = model.initialize_uniform(_baseline(5))

    def boundary_at(time_s: float) -> RegionalBoundaryForcing:
        return _boundary(5 + float(time_s >= 1_000))

    frames = tuple(
        iter_simulation(
            model,
            initial,
            duration_s=2_000,
            frame_duration_s=700,
            forcing_at=lambda _time_s: _forcing(),
            regional_boundary_at=boundary_at,
            boundary_mode=BoundaryMode.REGIONAL_INFLOW,
            regional_boundary_change_times_s=(1_000,),
        )
    )

    assert [frame.state.simulation_time_s for frame in frames] == [
        0,
        700,
        1_000,
        1_700,
        2_000,
    ]
    assert frames[2].regional_boundary is not None
    assert frames[3].regional_boundary is not None
    assert frames[2].regional_boundary.concentration_ug_m3 == 5
    assert frames[3].regional_boundary.concentration_ug_m3 == 6
    assert all(frame.forcing == frames[0].forcing for frame in frames)


def test_source_transitions_split_frames_before_transport_and_sinks() -> None:
    model = PM25Transport(
        _square_grid(side_cells=5),
        _physics(dry_deposition_velocity_m_s=1),
    )
    source = _source(
        x_m=250,
        y_m=250,
        emission_rate_ug_s=1_000,
        starts_at_s=50,
    )

    frames = tuple(
        iter_simulation(
            model,
            model.initialize_uniform(_baseline(0)),
            duration_s=100,
            frame_duration_s=100,
            forcing_at=lambda _time_s: _forcing(),
            sources=(source,),
            source_change_times_s=(50,),
        )
    )

    assert [frame.state.simulation_time_s for frame in frames] == [0, 50, 100]
    assert frames[1].budget is not None
    assert frames[2].budget is not None
    assert frames[1].budget.emitted_mass_ug == 0
    assert frames[2].budget.emitted_mass_ug == pytest.approx(50_000)
    assert frames[2].budget.final_mass_ug == pytest.approx(50_000 * math.exp(-0.5))


def test_summary_colorbar_has_dedicated_non_overlapping_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _square_grid(side_cells=10)
    graph = nx.MultiDiGraph()
    domain = SpatialDomain(
        config=DomainConfig(),
        road_graph=graph,
        boundary=box(0, 0, 1_000, 1_000),
        grid=grid,
        graph_source="test",
        grid_source="test",
    )
    sources = _demo_road_sources()
    frames = tuple(
        SimulationFrame(
            PM25State(
                grid,
                np.full(grid.shape, concentration),
                simulation_time_s=time_s,
            ),
            _synthetic_demo_forcing(time_s),
            _boundary(8),
            (),
            None,
        )
        for time_s, concentration in (
            (0, 8),
            (3_600, 10),
            (5_400, 12),
            (10_800, 9),
        )
    )

    monkeypatch.setattr(
        demo_module.ox,
        "graph_to_gdfs",
        lambda _graph: (None, _DemoEdges()),
    )
    figure, axes, colorbar_axis = _build_pm25_summary_figure(
        domain,
        frames,
        sources,
    )
    figure.canvas.draw()

    assert colorbar_axis.get_position().x0 > max(
        axis.get_position().x1 for axis in axes
    )
    plt.close(figure)


def test_demo_playback_interval_respects_simulated_speed() -> None:
    grid = _square_grid(side_cells=2)
    frames = tuple(
        SimulationFrame(
            PM25State(
                grid,
                np.full(grid.shape, 8.0),
                simulation_time_s=time_s,
            ),
            _forcing(),
            None,
            (),
            None,
        )
        for time_s in (0.0, 30.0, 90.0)
    )

    assert _demo_playback_interval_ms(frames, 0, playback_speed=60.0) == 500
    assert _demo_playback_interval_ms(frames, 1, playback_speed=60.0) == 1_000
    assert _demo_playback_interval_ms(frames, 2, playback_speed=60.0) == 1


def test_synthetic_demo_uses_four_road_sources_and_three_weather_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _square_grid(side_cells=10)
    domain = SpatialDomain(
        config=DomainConfig(),
        road_graph=nx.MultiDiGraph(),
        boundary=box(0, 0, 1_000, 1_000),
        grid=grid,
        graph_source="test",
        grid_source="test",
    )
    monkeypatch.setattr(
        demo_module.ox,
        "graph_to_gdfs",
        lambda _graph: (None, _DemoEdges()),
    )

    _, _, sources = _create_synthetic_demo(domain)
    weather = tuple(_synthetic_demo_forcing(time_s) for time_s in (0, 3_600, 7_200))

    assert [source.source_id for source in sources] == [
        "tupac-amaru",
        "javier-prado-este",
        "circunvalacion",
        "alfonso-ugarte-event",
    ]
    assert [source.role for source in sources] == [
        PM25SourceRole.ROUTINE_ENVIRONMENTAL,
        PM25SourceRole.ROUTINE_ENVIRONMENTAL,
        PM25SourceRole.ROUTINE_ENVIRONMENTAL,
        PM25SourceRole.COUNTERFACTUAL_EVENT,
    ]
    assert sources[-1].starts_at_s == 3_600
    assert sources[-1].ends_at_s == 7_200
    assert all(source.line.length >= 800 for source in sources)
    assert [forcing.wind_from_direction_deg for forcing in weather] == [
        270,
        180,
        90,
    ]
    assert [forcing.precipitation_mm_h for forcing in weather] == [0, 0, 2]
    assert np.asarray(
        [forcing.wind_components_m_s for forcing in weather]
    ) == pytest.approx(
        np.array([(1.5, 0), (0, 1.2), (-1, 0)]),
        abs=1e-12,
    )


def test_demo_road_selection_rejects_an_unknown_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _square_grid(side_cells=10)
    domain = SpatialDomain(
        config=DomainConfig(),
        road_graph=nx.MultiDiGraph(),
        boundary=box(0, 0, 1_000, 1_000),
        grid=grid,
        graph_source="test",
        grid_source="test",
    )
    monkeypatch.setattr(
        demo_module.ox,
        "graph_to_gdfs",
        lambda _graph: (None, _DemoEdges()),
    )

    with pytest.raises(ValueError, match="No OSM road geometry"):
        _road_corridor_by_name(domain, "unknown-road")


def test_synthetic_demo_persists_reproducible_input_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    grid = _square_grid(side_cells=10)
    domain = SpatialDomain(
        config=DomainConfig(),
        road_graph=nx.MultiDiGraph(),
        boundary=box(0, 0, 1_000, 1_000),
        grid=grid,
        graph_source="test",
        grid_source="test",
    )

    monkeypatch.setattr(
        demo_module.ox,
        "graph_to_gdfs",
        lambda _graph: (None, _DemoEdges()),
    )

    _, budgets = run_synthetic_demo(domain, tmp_path)
    scenario = json.loads((tmp_path / "scenario.json").read_text())
    serialized_budgets = json.loads((tmp_path / "mass_budget.json").read_text())
    with np.load(tmp_path / "concentration.npz", allow_pickle=False) as data:
        persisted_source_ids = data["source_ids"].tolist()

    assert scenario["ambient_baseline"]["provider_id"] == ("synthetic-ambient-baseline")
    assert scenario["regional_boundary"]["provider_id"] == (
        "synthetic-regional-boundary"
    )
    assert scenario["physics"]["mixing_height_m"] == 200
    assert scenario["boundary_mode"] == "regional_inflow"
    assert scenario["grid"]["domain_config_fingerprint"] == domain.config.fingerprint
    assert scenario["grid"]["shape"] == [10, 10]
    assert len(scenario["sources"]) == 4
    assert all(
        source["source_type"] == "RoadLineSource" for source in scenario["sources"]
    )
    assert scenario["sources"][-1]["role"] == "counterfactual_event"
    assert scenario["sources"][-1]["starts_at_s"] == 3_600
    assert scenario["sources"][-1]["ends_at_s"] == 7_200
    assert len(scenario["sources"][0]["line_xy_m"]) >= 2
    assert len(scenario["input_history"]) == len(budgets)
    precipitation_values = {
        entry["meteorology"]["precipitation_mm_h"]
        for entry in scenario["input_history"]
    }
    assert precipitation_values == {0.0, 2.0}
    assert {
        entry["meteorology"]["wind_from_direction_deg"]
        for entry in scenario["input_history"]
    } == {90.0, 180.0, 270.0}
    assert (
        "alfonso-ugarte-event"
        not in (scenario["input_history"][0]["active_source_ids"])
    )
    event_index = next(
        index
        for index, entry in enumerate(scenario["input_history"])
        if entry["interval_start_s"] == 3_600
    )
    assert (
        "alfonso-ugarte-event"
        in (scenario["input_history"][event_index]["active_source_ids"])
    )
    assert [
        entry["source_id"] for entry in serialized_budgets[0]["source_emissions"]
    ] == [
        "tupac-amaru",
        "javier-prado-este",
        "circunvalacion",
        "alfonso-ugarte-event",
    ]
    assert persisted_source_ids == [
        "tupac-amaru",
        "javier-prado-este",
        "circunvalacion",
        "alfonso-ugarte-event",
    ]
    assert serialized_budgets[0]["source_emissions"][-1]["emitted_mass_ug"] == 0
    assert (
        serialized_budgets[event_index]["source_emissions"][-1]["emitted_mass_ug"] > 0
    )
    assert serialized_budgets[-1]["wet_deposited_mass_ug"] > 0
    assert (tmp_path / "pm25_demo.png").is_file()


def _demo_road_sources() -> tuple[RoadLineSource, ...]:
    return tuple(
        RoadLineSource(
            source_id=source_id,
            line=line,
            emission_rate_ug_m_s=rate,
            provenance="synthetic demo test",
            starts_at_s=starts_at_s,
            ends_at_s=ends_at_s,
            role=role,
        )
        for source_id, line, rate, starts_at_s, ends_at_s, role in (
            (
                "tupac-amaru",
                LineString([(100, 150), (900, 150)]),
                120,
                0,
                10_800,
                PM25SourceRole.ROUTINE_ENVIRONMENTAL,
            ),
            (
                "javier-prado-este",
                LineString([(100, 350), (900, 350)]),
                100,
                0,
                10_800,
                PM25SourceRole.ROUTINE_ENVIRONMENTAL,
            ),
            (
                "circunvalacion",
                LineString([(100, 550), (900, 550)]),
                110,
                0,
                10_800,
                PM25SourceRole.ROUTINE_ENVIRONMENTAL,
            ),
            (
                "alfonso-ugarte-event",
                LineString([(100, 750), (900, 750)]),
                220,
                3_600,
                7_200,
                PM25SourceRole.COUNTERFACTUAL_EVENT,
            ),
        )
    )


class _DemoEdges:
    def __init__(self) -> None:
        sources = _demo_road_sources()
        self.geometry = tuple(source.line for source in sources)
        self._names = (
            "Avenida Tupac Amaru",
            "Avenida Javier Prado Este",
            "Avenida Circunvalacion",
            "Avenida Alfonso Ugarte",
        )

    def __contains__(self, key: object) -> bool:
        return key == "name"

    def __getitem__(self, key: str):
        if key != "name":
            raise KeyError(key)
        return self._names

    def plot(self, *, ax, **_kwargs):
        return ax


class _StaticSource:
    def __init__(
        self,
        source_id: str,
        contribution: PM25SourceContribution,
        *,
        role: PM25SourceRole = PM25SourceRole.ROUTINE_ENVIRONMENTAL,
        provenance: str = "test fixture",
    ) -> None:
        self.source_id = source_id
        self.role = role
        self.provenance = provenance
        self.contribution = contribution

    @property
    def transition_times_s(self) -> tuple[float, ...]:
        return ()

    def evidence_descriptor(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "role": self.role.value,
            "provenance": self.provenance,
        }

    def contribution_for_interval(
        self,
        grid: GridDefinition,
        *,
        start_s: float,
        duration_s: float,
    ) -> PM25SourceContribution:
        return self.contribution
