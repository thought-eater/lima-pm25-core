"""Synthetic PM2.5 mechanics demo; its input values are not calibrated for Lima."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import osmnx as ox
from shapely import line_merge, unary_union
from shapely.geometry import LineString, MultiLineString

from lima_pm25.domain import (
    AmbientBaseline,
    AtmosphericForcing,
    FixedStationCatalog,
    MobilityPolicy,
    PM25SourceRole,
    RegionalBoundaryForcing,
    SpatialDomain,
    StudyEnvironmentConfig,
    plot_fixed_station_catalog,
)

from .evidence import verify_run_bundle, write_run_bundle
from .orchestration import (
    RegionalBoundarySchedule,
    SimulationRunConfig,
    SimulationRunRequest,
    SimulationRunResult,
    run_simulation,
)
from .sensing import SensorConfig, SensorType, create_lima_fixed_sensor_configs
from .trajectory import TimestampedTrajectoryProvider, TrajectoryPoint
from .transport import (
    BoundaryMode,
    MassBudget,
    PM25PhysicsConfig,
    PM25State,
    PM25Transport,
    RoadLineSource,
    SimulationFrame,
    iter_simulation,
)
from .weather import ScheduledWeatherValue, SyntheticWeatherWindow

_SYNTHETIC_DEMO_DURATION_S = 10_800.0
_SYNTHETIC_DEMO_PHASE_TIMES_S = (3_600.0, 7_200.0)
_SYNTHETIC_DEMO_START_UTC = datetime(2026, 1, 1, 5, tzinfo=UTC)
_SYNTHETIC_DEMO_ROADS = (
    (
        "tupac-amaru",
        "Avenida Tupac Amaru",
        "Amaru",
        120.0,
        PM25SourceRole.ROUTINE_ENVIRONMENTAL,
    ),
    (
        "javier-prado-este",
        "Avenida Javier Prado Este",
        "Javier Prado Este",
        100.0,
        PM25SourceRole.ROUTINE_ENVIRONMENTAL,
    ),
    (
        "circunvalacion",
        "Avenida Circunvalacion",
        "Circunvalaci",
        110.0,
        PM25SourceRole.ROUTINE_ENVIRONMENTAL,
    ),
    (
        "alfonso-ugarte-event",
        "Avenida Alfonso Ugarte",
        "Alfonso Ugarte",
        220.0,
        PM25SourceRole.COUNTERFACTUAL_EVENT,
    ),
)


def run_synthetic_demo(
    domain: SpatialDomain,
    output_directory: Path = Path("cache/simulations/pm25_demo"),
    *,
    station_catalog: FixedStationCatalog | None = None,
) -> tuple[PM25State, tuple[MassBudget, ...]]:
    model, initial_state, sources = _create_synthetic_demo(domain)
    frames = tuple(
        iter_simulation(
            model,
            initial_state,
            duration_s=_SYNTHETIC_DEMO_DURATION_S,
            frame_duration_s=300,
            forcing_at=_synthetic_demo_forcing,
            regional_boundary_at=_synthetic_demo_regional_boundary,
            sources=sources,
            boundary_mode=BoundaryMode.REGIONAL_INFLOW,
            forcing_change_times_s=_SYNTHETIC_DEMO_PHASE_TIMES_S,
            source_change_times_s=_SYNTHETIC_DEMO_PHASE_TIMES_S,
        )
    )
    _save_demo_results(
        domain,
        model,
        frames,
        sources,
        output_directory,
        station_catalog=station_catalog,
    )
    budgets = tuple(frame.budget for frame in frames if frame.budget is not None)
    return frames[-1].state, budgets


def run_complete_synthetic_demo(
    domain: SpatialDomain,
    output_directory: Path = Path("resultados/r1_3_simulacion"),
    *,
    station_catalog: FixedStationCatalog,
    study_config: StudyEnvironmentConfig | None = None,
) -> tuple[SimulationRunResult, Path]:
    if study_config is not None:
        request = create_synthetic_study_family_requests(
            domain,
            station_catalog=station_catalog,
            study_config=study_config,
        )["event"]
        result = run_simulation(request)
        bundle_directory = output_directory / result.run_id
        if bundle_directory.exists():
            verification = verify_run_bundle(bundle_directory)
            if not verification.valid:
                raise RuntimeError(str(verification.failures))
        else:
            write_run_bundle(result, bundle_directory)
        return result, bundle_directory
    model, initial_state, sources = _create_synthetic_demo(domain)
    mobile_provider = _demo_mobile_replay(domain, sources[0].line)
    mobile_sensor = SensorConfig(
        sensor_id="MOBILE_DEMO",
        sensor_type=SensorType.MOBILE,
        sampling_interval_s=300.0,
        first_due_time_s=0.0,
        mobile_position_provider_id=mobile_provider.provider_id,
        provenance="synthetic mobile sensing demonstration",
        noise_stddev_ug_m3=1.0,
    )
    weather = SyntheticWeatherWindow(
        start_utc=_SYNTHETIC_DEMO_START_UTC,
        duration_s=_SYNTHETIC_DEMO_DURATION_S,
        values=tuple(
            ScheduledWeatherValue(time_s, _synthetic_demo_forcing(time_s))
            for time_s in (0.0, *_SYNTHETIC_DEMO_PHASE_TIMES_S)
        ),
    )
    request = SimulationRunRequest(
        model=model,
        initial_state=initial_state,
        weather=weather,
        config=SimulationRunConfig(
            start_utc=_SYNTHETIC_DEMO_START_UTC,
            duration_s=_SYNTHETIC_DEMO_DURATION_S,
            concentration_output_interval_s=300.0,
            seed=20260925,
            boundary_mode=BoundaryMode.REGIONAL_INFLOW,
        ),
        sensor_configs=(
            *create_lima_fixed_sensor_configs(station_catalog),
            mobile_sensor,
        ),
        station_catalog=station_catalog,
        mobile_position_providers=(mobile_provider,),
        sources=sources,
        regional_boundary=RegionalBoundarySchedule.constant(
            _synthetic_demo_regional_boundary(0.0)
        ),
    )
    result = run_simulation(request)
    bundle_directory = output_directory / result.run_id
    if bundle_directory.exists():
        verification = verify_run_bundle(bundle_directory)
        if not verification.valid:
            raise RuntimeError(
                "Existing complete-demo bundle failed verification: "
                + ", ".join(verification.failures)
            )
    else:
        write_run_bundle(result, bundle_directory)
    return result, bundle_directory


def create_synthetic_study_family_requests(
    domain: SpatialDomain,
    *,
    station_catalog: FixedStationCatalog,
    study_config: StudyEnvironmentConfig,
) -> dict[str, SimulationRunRequest]:
    if domain.config.fingerprint != study_config.domain.fingerprint:
        raise ValueError("Study configuration and domain do not match")
    model, initial_state, sources = _create_synthetic_demo(domain)
    if model.physics.mixing_height_m != study_config.vertical.mixing_height_m:
        raise ValueError(
            "Synthetic study family mixing height must match study configuration"
        )
    if study_config.temporal.mobile_sampling_interval_s is None:
        raise ValueError("Synthetic study family requires a mobile sampling interval")
    mobile_provider = _demo_mobile_replay(
        domain,
        sources[0].line,
        mobility_policy=study_config.mobility_policy,
        sampling_interval_s=study_config.temporal.mobile_sampling_interval_s,
    )
    mobile_sensor = SensorConfig(
        sensor_id="MOBILE_DEMO",
        sensor_type=SensorType.MOBILE,
        sampling_interval_s=study_config.temporal.mobile_sampling_interval_s,
        first_due_time_s=0.0,
        mobile_position_provider_id=mobile_provider.provider_id,
        provenance="synthetic mobile sensing demonstration",
        noise_stddev_ug_m3=1.0,
    )
    weather = SyntheticWeatherWindow(
        start_utc=_SYNTHETIC_DEMO_START_UTC,
        duration_s=_SYNTHETIC_DEMO_DURATION_S,
        values=tuple(
            ScheduledWeatherValue(time_s, _synthetic_demo_forcing(time_s))
            for time_s in (0.0, *_SYNTHETIC_DEMO_PHASE_TIMES_S)
        ),
    )
    run_config = SimulationRunConfig(
        start_utc=_SYNTHETIC_DEMO_START_UTC,
        duration_s=_SYNTHETIC_DEMO_DURATION_S,
        concentration_output_interval_s=(
            study_config.temporal.concentration_output_interval_s
        ),
        concentration_output_times_s=(
            study_config.temporal.concentration_output_times_s
        ),
        seed=20260925,
        boundary_mode=BoundaryMode.REGIONAL_INFLOW,
    )
    shared = SimulationRunRequest(
        model=model,
        initial_state=initial_state,
        weather=weather,
        config=run_config,
        sensor_configs=(
            *create_lima_fixed_sensor_configs(station_catalog),
            mobile_sensor,
        ),
        station_catalog=station_catalog,
        mobile_position_providers=(mobile_provider,),
        regional_boundary=RegionalBoundarySchedule.constant(
            _synthetic_demo_regional_boundary(0.0)
        ),
    )
    routine_sources = tuple(
        source
        for source in sources
        if source.role is not PM25SourceRole.COUNTERFACTUAL_EVENT
    )
    return {
        "control": replace(
            shared,
            sources=routine_sources,
            scenario_identity=study_config.scenario_families[0].run_identity("control"),
        ),
        "event": replace(
            shared,
            sources=sources,
            scenario_identity=study_config.scenario_families[0].run_identity("event"),
        ),
    }


def animate_synthetic_demo(
    domain: SpatialDomain,
    *,
    playback_speed: float = 60.0,
    frame_duration_s: float = 60.0,
    start_paused: bool = False,
    station_catalog: FixedStationCatalog | None = None,
    output_directory: Path = Path("cache/simulations/pm25_demo"),
) -> tuple[PM25State, tuple[MassBudget, ...]]:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    if not math.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError("playback_speed must be positive and finite")
    model, initial_state, sources = _create_synthetic_demo(domain)
    frames = tuple(
        iter_simulation(
            model,
            initial_state,
            duration_s=_SYNTHETIC_DEMO_DURATION_S,
            frame_duration_s=frame_duration_s,
            forcing_at=_synthetic_demo_forcing,
            regional_boundary_at=_synthetic_demo_regional_boundary,
            sources=sources,
            boundary_mode=BoundaryMode.REGIONAL_INFLOW,
            forcing_change_times_s=_SYNTHETIC_DEMO_PHASE_TIMES_S,
            source_change_times_s=_SYNTHETIC_DEMO_PHASE_TIMES_S,
        )
    )
    first_frame = frames[0]
    minimum_scale, maximum_scale = _demo_concentration_limits(domain, frames)

    figure = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid_spec = figure.add_gridspec(1, 3, width_ratios=(1, 0.04, 0.32))
    axis = figure.add_subplot(grid_spec[0, 0])
    colorbar_axis = figure.add_subplot(grid_spec[0, 1])
    status_axis = figure.add_subplot(grid_spec[0, 2])
    _, edges = ox.graph_to_gdfs(domain.road_graph)
    x_min, y_min, x_max, y_max = domain.grid.grid_bounds_m
    extent = (float(x_min), float(x_max), float(y_min), float(y_max))
    field = _masked_demo_field(domain, first_frame.state)
    image = axis.imshow(
        field,
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=minimum_scale,
        vmax=maximum_scale,
        interpolation="nearest",
    )
    # GeoPandas accepts scalar colors; the accessor stub rejects this valid call.
    edges.plot(ax=axis, color="white", linewidth=0.18, alpha=0.25)  # ty: ignore[invalid-argument-type]
    if station_catalog is not None:
        plot_fixed_station_catalog(axis, station_catalog)
    source_artists = tuple(
        axis.plot(
            np.asarray(source.line.coords)[:, 0],
            np.asarray(source.line.coords)[:, 1],
            linewidth=2.2,
            solid_capstyle="round",
            zorder=4,
        )[0]
        for source in sources
    )
    _update_demo_source_artists(
        source_artists,
        sources,
        first_frame.state.simulation_time_s,
    )

    x_min, y_min, x_max, y_max = domain.grid.grid_bounds_m
    arrow_length = 0.09 * min(x_max - x_min, y_max - y_min)
    arrow_x = x_min + 0.1 * (x_max - x_min)
    arrow_y = y_min + 0.88 * (y_max - y_min)
    arrow_dx, arrow_dy = _scaled_wind_arrow(first_frame.forcing, arrow_length)
    wind_arrow = axis.quiver(
        [arrow_x],
        [arrow_y],
        [arrow_dx],
        [arrow_dy],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#f4f4f4",
        edgecolor="#202020",
        linewidth=0.6,
        width=0.008,
        zorder=5,
    )
    axis.text(
        arrow_x,
        arrow_y - 0.035 * (y_max - y_min),
        "destino del viento",
        color="white",
        fontsize=8,
        ha="center",
        va="top",
        zorder=5,
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
    )
    axis.set_title(_demo_frame_title(first_frame))
    axis.set_aspect("equal")
    axis.set_axis_off()
    figure.colorbar(image, cax=colorbar_axis, label="PM2.5 (ug/m3)")

    status_axis.set_axis_off()
    status_text = status_axis.text(
        0,
        1,
        _demo_status_text(first_frame, sources, ()),
        transform=status_axis.transAxes,
        va="top",
        family="monospace",
        fontsize=10,
        linespacing=1.5,
    )
    status_axis.text(
        0,
        0.16,
        "verde: estacion fija de PM2.5\n"
        "cian: fuente rutinaria\n"
        "magenta: evento contrafactual\n"
        "gris: fuente inactiva",
        transform=status_axis.transAxes,
        va="bottom",
        fontsize=9,
        color="#404040",
    )

    play_axis = status_axis.inset_axes((0.0, 0.04, 0.47, 0.065))
    restart_axis = status_axis.inset_axes((0.53, 0.04, 0.47, 0.065))
    play_button = Button(
        play_axis,
        "Reproducir" if start_paused else "Pausar",
    )
    restart_button = Button(restart_axis, "Reiniciar")

    current_frame_index = 0
    playing = not start_paused

    def render_frame(frame_index: int) -> None:
        frame = frames[frame_index]
        field = _masked_demo_field(domain, frame.state)
        image.set_data(field)
        source_time_s = _frame_interval_start_s(frame)
        _update_demo_source_artists(source_artists, sources, source_time_s)
        arrow_dx, arrow_dy = _scaled_wind_arrow(frame.forcing, arrow_length)
        wind_arrow.set_UVC(arrow_dx, arrow_dy)
        axis.set_title(_demo_frame_title(frame))
        completed_budgets = tuple(
            candidate.budget
            for candidate in frames[1 : frame_index + 1]
            if candidate.budget is not None
        )
        status_text.set_text(_demo_status_text(frame, sources, completed_budgets))
        figure.canvas.draw_idle()

    timer = figure.canvas.new_timer(
        interval=_demo_playback_interval_ms(frames, 0, playback_speed)
    )

    def stop_playback(*, finished: bool = False) -> None:
        nonlocal playing
        timer.stop()
        playing = False
        play_button.label.set_text("Repetir" if finished else "Reproducir")
        figure.canvas.draw_idle()

    def advance_frame() -> None:
        nonlocal current_frame_index
        if current_frame_index >= len(frames) - 1:
            stop_playback(finished=True)
            return
        current_frame_index += 1
        render_frame(current_frame_index)
        if current_frame_index >= len(frames) - 1:
            stop_playback(finished=True)
        else:
            timer.interval = _demo_playback_interval_ms(
                frames,
                current_frame_index,
                playback_speed,
            )

    def toggle_playback(_event) -> None:
        nonlocal current_frame_index, playing
        if playing:
            stop_playback()
            return
        if current_frame_index >= len(frames) - 1:
            current_frame_index = 0
            render_frame(current_frame_index)
        playing = True
        play_button.label.set_text("Pausar")
        timer.interval = _demo_playback_interval_ms(
            frames,
            current_frame_index,
            playback_speed,
        )
        timer.start()
        figure.canvas.draw_idle()

    def restart_playback(_event) -> None:
        nonlocal current_frame_index
        stop_playback()
        current_frame_index = 0
        render_frame(current_frame_index)

    timer.add_callback(advance_frame)
    play_button.on_clicked(toggle_playback)
    restart_button.on_clicked(restart_playback)
    if playing:
        timer.start()
    plt.show()

    _save_demo_results(
        domain,
        model,
        frames,
        sources,
        output_directory,
        station_catalog=station_catalog,
    )
    budgets = tuple(frame.budget for frame in frames if frame.budget is not None)
    return frames[-1].state, budgets


def _create_synthetic_demo(
    domain: SpatialDomain,
) -> tuple[PM25Transport, PM25State, tuple[RoadLineSource, ...]]:
    physics = PM25PhysicsConfig(
        mixing_height_m=200.0,
        diffusivity_m2_s=25.0,
        dry_deposition_velocity_m_s=0.001,
        wet_scavenging_per_mm=0.03,
    )
    model = PM25Transport(domain.grid, physics)
    sources = []
    _, road_edges = ox.graph_to_gdfs(domain.road_graph)
    for source_id, road_name, road_fragment, rate, role in _SYNTHETIC_DEMO_ROADS:
        event = role is PM25SourceRole.COUNTERFACTUAL_EVENT
        source = RoadLineSource(
            source_id=source_id,
            line=_road_corridor_by_name(domain, road_fragment, road_edges),
            emission_rate_ug_m_s=rate,
            provenance=f"synthetic emissions on OSM geometry for {road_name}",
            starts_at_s=3_600.0 if event else 0.0,
            ends_at_s=7_200.0 if event else _SYNTHETIC_DEMO_DURATION_S,
            role=role,
        )
        source.spatial_weights(domain.grid)
        sources.append(source)
    state = model.initialize_uniform(
        AmbientBaseline(
            provider_id="synthetic-ambient-baseline",
            concentration_ug_m3=8.0,
            provenance="synthetic PM2.5 mechanics demonstration",
        )
    )
    return model, state, tuple(sources)


def _demo_mobile_replay(
    domain: SpatialDomain,
    line: LineString,
    *,
    mobility_policy: MobilityPolicy | None = None,
    sampling_interval_s: float = 300.0,
) -> TimestampedTrajectoryProvider:
    if mobility_policy is None:
        mobility_policy = MobilityPolicy(road_tolerance_m=5.0)
    due_times = tuple(
        float(time_s)
        for time_s in np.arange(
            0,
            _SYNTHETIC_DEMO_DURATION_S + 1e-9,
            sampling_interval_s,
        )
    )
    points = []
    for index, time_s in enumerate(due_times):
        cycle_position = index % 20
        fraction = (
            cycle_position / 10 if cycle_position <= 10 else (20 - cycle_position) / 10
        )
        point = line.interpolate(fraction, normalized=True)
        points.append(
            TrajectoryPoint(
                "MOBILE_DEMO",
                time_s,
                float(point.x),
                float(point.y),
            )
        )
    return TimestampedTrajectoryProvider(
        provider_id="synthetic-mobile-replay",
        domain=domain,
        mobility_policy=mobility_policy,
        points=tuple(points),
    )


def _synthetic_demo_forcing(simulation_time_s: float) -> AtmosphericForcing:
    if simulation_time_s < 3_600:
        wind_speed_m_s = 1.5
        wind_from_direction_deg = 270.0
        precipitation_mm_h = 0.0
    elif simulation_time_s < 7_200:
        wind_speed_m_s = 1.2
        wind_from_direction_deg = 180.0
        precipitation_mm_h = 0.0
    else:
        wind_speed_m_s = 1.0
        wind_from_direction_deg = 90.0
        precipitation_mm_h = 2.0
    return AtmosphericForcing(
        provider_id="synthetic-meteorology",
        provenance="synthetic PM2.5 mechanics demonstration",
        wind_speed_m_s=wind_speed_m_s,
        wind_from_direction_deg=wind_from_direction_deg,
        precipitation_mm_h=precipitation_mm_h,
    )


def _synthetic_demo_regional_boundary(
    _simulation_time_s: float,
) -> RegionalBoundaryForcing:
    return RegionalBoundaryForcing(
        provider_id="synthetic-regional-boundary",
        concentration_ug_m3=8.0,
        provenance="synthetic PM2.5 mechanics demonstration",
    )


def _road_corridor_by_name(
    domain: SpatialDomain,
    name_fragment: str,
    edges=None,
) -> LineString:
    if edges is None:
        _, edges = ox.graph_to_gdfs(domain.road_graph)
    if "name" not in edges:
        raise ValueError("The road graph has no edge names for demo source selection")

    matching_lines = []
    for name, geometry in zip(edges["name"], edges.geometry, strict=True):
        if name_fragment.casefold() not in str(name).casefold():
            continue
        if isinstance(geometry, LineString):
            matching_lines.append(geometry)
        elif isinstance(geometry, MultiLineString):
            matching_lines.extend(geometry.geoms)

    if not matching_lines:
        raise ValueError(
            f"No OSM road geometry matches demo road fragment {name_fragment!r}"
        )

    merged = line_merge(unary_union(matching_lines))
    if isinstance(merged, LineString):
        candidates = (merged,)
    elif isinstance(merged, MultiLineString):
        candidates = tuple(merged.geoms)
    else:
        candidates = ()
    if not candidates:
        raise ValueError(
            f"OSM road geometry for {name_fragment!r} cannot form a line source"
        )

    corridor = max(candidates, key=lambda line: line.length)
    if corridor.length < 5 * domain.grid.cell_size_m:
        raise ValueError(
            f"OSM road geometry for {name_fragment!r} is too short for the demo"
        )
    return corridor


def _serialize_demo_source(source: RoadLineSource) -> dict[str, object]:
    return {
        "source_type": type(source).__name__,
        "source_id": source.source_id,
        "role": source.role,
        "provenance": source.provenance,
        "emission_rate_ug_m_s": source.emission_rate_ug_m_s,
        "starts_at_s": source.starts_at_s,
        "ends_at_s": source.ends_at_s,
        "line_length_m": source.line.length,
        "line_xy_m": np.asarray(source.line.coords).tolist(),
    }


def _source_is_active(source: RoadLineSource, simulation_time_s: float) -> bool:
    return source.starts_at_s <= simulation_time_s and (
        source.ends_at_s is None or simulation_time_s < source.ends_at_s
    )


def _frame_interval_start_s(frame: SimulationFrame) -> float:
    if frame.budget is None:
        return frame.state.simulation_time_s
    return (
        frame.state.simulation_time_s
        - frame.budget.substeps * frame.budget.substep_duration_s
    )


def _masked_demo_field(domain: SpatialDomain, state: PM25State) -> np.ma.MaskedArray:
    return np.ma.masked_where(
        ~domain.grid.active_mask,
        state.concentration_ug_m3,
    )


def _demo_concentration_limits(
    domain: SpatialDomain,
    frames: Sequence[SimulationFrame],
) -> tuple[float, float]:
    active_values = [
        frame.state.concentration_ug_m3[domain.grid.active_mask] for frame in frames
    ]
    minimum = min(float(np.min(values)) for values in active_values)
    maximum = max(float(np.max(values)) for values in active_values)
    if math.isclose(minimum, maximum):
        maximum = minimum + 1.0
    return minimum, maximum


def _source_style(
    source: RoadLineSource, simulation_time_s: float
) -> dict[str, object]:
    if not _source_is_active(source, simulation_time_s):
        return {"color": "#77838f", "alpha": 0.45, "linestyle": ":"}
    if source.role is PM25SourceRole.COUNTERFACTUAL_EVENT:
        return {"color": "#ff3d9a", "alpha": 1.0, "linestyle": "-"}
    return {"color": "#00e5ff", "alpha": 0.95, "linestyle": "-"}


def _update_demo_source_artists(
    artists,
    sources: Sequence[RoadLineSource],
    simulation_time_s: float,
) -> None:
    for artist, source in zip(artists, sources, strict=True):
        style = _source_style(source, simulation_time_s)
        artist.set_color(style["color"])
        artist.set_alpha(style["alpha"])
        artist.set_linestyle(style["linestyle"])


def _plot_demo_sources(
    axis,
    sources: Sequence[RoadLineSource],
    simulation_time_s: float,
) -> None:
    for source in sources:
        coordinates = np.asarray(source.line.coords)
        axis.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            linewidth=2.0,
            solid_capstyle="round",
            zorder=4,
            **_source_style(source, simulation_time_s),
        )


def _scaled_wind_arrow(
    forcing: AtmosphericForcing,
    arrow_length_m: float,
) -> tuple[float, float]:
    eastward, northward = forcing.wind_components_m_s
    speed = math.hypot(eastward, northward)
    if speed == 0:
        return 0.0, 0.0
    return (
        eastward / speed * arrow_length_m,
        northward / speed * arrow_length_m,
    )


def _demo_phase_label(simulation_time_s: float) -> str:
    if simulation_time_s < 3_600:
        return "Fase 1: transporte hacia el este"
    if simulation_time_s < 7_200:
        return "Fase 2: evento y transporte hacia el norte"
    return "Fase 3: transporte hacia el oeste con lluvia"


def _demo_frame_title(frame: SimulationFrame) -> str:
    interval_start_s = _frame_interval_start_s(frame)
    maximum = float(np.max(frame.state.concentration_ug_m3))
    return (
        f"{_demo_phase_label(interval_start_s)}\n"
        f"t={frame.state.simulation_time_s / 3600:.2f} h | "
        f"viento desde {frame.forcing.wind_from_direction_deg:g} grados a "
        f"{frame.forcing.wind_speed_m_s:g} m/s | "
        f"lluvia={frame.forcing.precipitation_mm_h:g} mm/h | "
        f"max={maximum:.2f} ug/m3"
    )


def _demo_status_text(
    frame: SimulationFrame,
    sources: Sequence[RoadLineSource],
    budgets: Sequence[MassBudget],
) -> str:
    interval_start_s = _frame_interval_start_s(frame)
    active_sources = [
        source.source_id
        for source in sources
        if _source_is_active(source, interval_start_s)
    ]
    final_mass = f"{budgets[-1].final_mass_ug:10.3e}" if budgets else "       n/a"
    emitted = sum(budget.emitted_mass_ug for budget in budgets)
    outflow = sum(budget.boundary_outflow_mass_ug for budget in budgets)
    dry = sum(budget.dry_deposited_mass_ug for budget in budgets)
    wet = sum(budget.wet_deposited_mass_ug for budget in budgets)
    residual = sum(budget.numerical_residual_ug for budget in budgets)
    source_lines = "\n".join(f"  {source_id}" for source_id in active_sources)
    return (
        "FUENTES ACTIVAS\n"
        f"{source_lines or '  ninguna'}\n\n"
        "DIAGNOSTICO DE MASA (ug)\n"
        f"  campo      {final_mass}\n"
        f"  emitida    {emitted:10.3e}\n"
        f"  salida     {outflow:10.3e}\n"
        f"  dep. seca  {dry:10.3e}\n"
        f"  dep. hum.  {wet:10.3e}\n"
        f"  residual   {residual:10.3e}"
    )


def _demo_playback_interval_ms(
    frames: Sequence[SimulationFrame],
    frame_index: int,
    playback_speed: float,
) -> int:
    if frame_index >= len(frames) - 1:
        return 1
    current_time_s = frames[frame_index].state.simulation_time_s
    next_time_s = frames[frame_index + 1].state.simulation_time_s
    return max(1, round((next_time_s - current_time_s) / playback_speed * 1_000))


def _select_demo_summary_frames(
    frames: Sequence[SimulationFrame],
) -> tuple[SimulationFrame, SimulationFrame, SimulationFrame, SimulationFrame]:
    if not frames:
        raise ValueError("At least one simulation frame is required")
    target_times_s = (0.0, 3_600.0, 5_400.0, _SYNTHETIC_DEMO_DURATION_S)
    selected = tuple(
        min(
            frames,
            key=lambda frame: abs(frame.state.simulation_time_s - target_time_s),
        )
        for target_time_s in target_times_s
    )
    return selected[0], selected[1], selected[2], selected[3]


def _save_demo_results(
    domain: SpatialDomain,
    model: PM25Transport,
    frames: Sequence[SimulationFrame],
    sources: Sequence[RoadLineSource],
    output_directory: Path,
    *,
    station_catalog: FixedStationCatalog | None = None,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    states = tuple(frame.state for frame in frames)
    budgets = tuple(frame.budget for frame in frames if frame.budget is not None)

    np.savez_compressed(
        output_directory / "concentration.npz",
        times_s=np.array([snapshot.simulation_time_s for snapshot in states]),
        concentration_ug_m3=np.stack(
            [snapshot.concentration_ug_m3 for snapshot in states]
        ),
        source_ids=np.asarray([source.source_id for source in sources]),
    )
    (output_directory / "mass_budget.json").write_text(
        json.dumps([budget.to_dict() for budget in budgets], indent=2) + "\n",
        encoding="utf-8",
    )
    initial_baseline = states[0].ambient_baseline
    first_boundary = frames[0].regional_boundary
    (output_directory / "scenario.json").write_text(
        json.dumps(
            {
                "physics": asdict(model.physics),
                "boundary_mode": BoundaryMode.REGIONAL_INFLOW,
                "grid": {
                    "domain_config_fingerprint": domain.config.fingerprint,
                    "crs": domain.grid.crs,
                    "cell_size_m": domain.grid.cell_size_m,
                    "shape": domain.grid.shape,
                    "grid_bounds_m": domain.grid.grid_bounds_m.tolist(),
                    "active_cells": int(np.count_nonzero(domain.grid.active_mask)),
                },
                "ambient_baseline": (
                    asdict(initial_baseline) if initial_baseline is not None else None
                ),
                "regional_boundary": (
                    asdict(first_boundary) if first_boundary is not None else None
                ),
                "sources": [_serialize_demo_source(source) for source in sources],
                "input_history": [
                    {
                        "interval_start_s": _frame_interval_start_s(frame),
                        "interval_end_s": frame.state.simulation_time_s,
                        "meteorology": asdict(frame.forcing),
                        "regional_boundary": (
                            asdict(frame.regional_boundary)
                            if frame.regional_boundary is not None
                            else None
                        ),
                        "active_source_ids": [
                            source.source_id
                            for source in sources
                            if _source_is_active(
                                source,
                                _frame_interval_start_s(frame),
                            )
                        ],
                    }
                    for frame in frames
                    if frame.budget is not None
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot_pm25_demo(
        domain,
        frames,
        sources,
        output_directory / "pm25_demo.png",
        station_catalog=station_catalog,
    )


def plot_pm25_demo(
    domain: SpatialDomain,
    frames: Sequence[SimulationFrame],
    sources: Sequence[RoadLineSource],
    output_path: Path,
    *,
    station_catalog: FixedStationCatalog | None = None,
) -> None:
    import matplotlib.pyplot as plt

    figure, _, _ = _build_pm25_summary_figure(
        domain,
        frames,
        sources,
        station_catalog=station_catalog,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _build_pm25_summary_figure(
    domain: SpatialDomain,
    frames: Sequence[SimulationFrame],
    sources: Sequence[RoadLineSource],
    *,
    station_catalog: FixedStationCatalog | None = None,
):
    import matplotlib.pyplot as plt

    _, edges = ox.graph_to_gdfs(domain.road_graph)
    figure = plt.figure(figsize=(15, 11), constrained_layout=True)
    grid_spec = figure.add_gridspec(
        2,
        3,
        width_ratios=(1, 1, 0.04),
    )
    axes = (
        figure.add_subplot(grid_spec[0, 0]),
        figure.add_subplot(grid_spec[0, 1]),
        figure.add_subplot(grid_spec[1, 0]),
        figure.add_subplot(grid_spec[1, 1]),
    )
    colorbar_axis = figure.add_subplot(grid_spec[:, 2])
    x_min, y_min, x_max, y_max = domain.grid.grid_bounds_m
    extent = (float(x_min), float(x_max), float(y_min), float(y_max))
    selected_frames = _select_demo_summary_frames(frames)
    minimum_scale, maximum_scale = _demo_concentration_limits(domain, frames)

    image = None
    for axis, frame in zip(axes, selected_frames, strict=True):
        field = _masked_demo_field(domain, frame.state)
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=minimum_scale,
            vmax=maximum_scale,
            interpolation="nearest",
        )
        # Keep a scalar color so multi-part road geometries share the same color.
        edges.plot(ax=axis, color="white", linewidth=0.12, alpha=0.22)  # ty: ignore[invalid-argument-type]
        if station_catalog is not None:
            plot_fixed_station_catalog(axis, station_catalog)
        _plot_demo_sources(
            axis,
            sources,
            _frame_interval_start_s(frame),
        )
        x_min, y_min, x_max, y_max = domain.grid.grid_bounds_m
        arrow_length = 0.08 * min(x_max - x_min, y_max - y_min)
        arrow_dx, arrow_dy = _scaled_wind_arrow(frame.forcing, arrow_length)
        axis.quiver(
            [x_min + 0.1 * (x_max - x_min)],
            [y_min + 0.88 * (y_max - y_min)],
            [arrow_dx],
            [arrow_dy],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="#f4f4f4",
            edgecolor="#202020",
            linewidth=0.5,
            width=0.008,
            zorder=5,
        )
        axis.set_title(_demo_frame_title(frame), fontsize=10)
        axis.set_aspect("equal")
        axis.set_axis_off()

    if image is not None:
        figure.colorbar(image, cax=colorbar_axis, label="PM2.5 (ug/m3)")
    residual = sum(
        abs(frame.budget.numerical_residual_ug)
        for frame in frames
        if frame.budget is not None
    )
    figure.suptitle(
        "Demostracion sintetica de la mecanica de PM2.5 "
        f"(residual absoluto acumulado: {residual:.3g} ug)\n"
        "verde: estacion | cian: rutina | magenta: evento | gris: inactiva | "
        "flecha blanca: destino del viento"
    )
    return figure, axes, colorbar_axis
