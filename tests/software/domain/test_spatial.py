from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from pyproj import Geod
from shapely.geometry import LineString, box

from lima_pm25.domain import spatial as domain_module
from lima_pm25.domain import (
    DomainConfig,
    LIMA_PM25_STATION_DEFINITIONS,
    build_grid,
    create_spatial_domain,
    load_grid,
    save_grid,
)


def test_domain_config_validates_bounds_and_cell_size() -> None:
    with pytest.raises(ValueError, match="Longitude bounds"):
        DomainConfig(left=1, right=0)
    with pytest.raises(ValueError, match="Latitude bounds"):
        DomainConfig(bottom=1, top=0)
    with pytest.raises(ValueError, match="cell_size_m"):
        DomainConfig(cell_size_m=0)
    with pytest.raises(ValueError, match="road_network_buffer_m"):
        DomainConfig(road_network_buffer_m=-1)


def test_default_domain_gives_campo_de_marte_at_least_its_western_margin_to_south() -> None:
    config = DomainConfig()
    station = next(
        site for site in LIMA_PM25_STATION_DEFINITIONS if site.site_id == "CAMPO_DE_MARTE"
    )
    station_longitude = station.longitude_deg
    station_latitude = station.latitude_deg
    geod = Geod(ellps="WGS84")

    _, _, west_distance_m = geod.inv(
        station_longitude,
        station_latitude,
        config.left,
        station_latitude,
    )
    _, _, south_distance_m = geod.inv(
        station_longitude,
        station_latitude,
        station_longitude,
        config.bottom,
    )

    assert south_distance_m >= west_distance_m


def test_road_bbox_adds_the_configured_buffer_without_expanding_the_grid_bbox() -> None:
    config = DomainConfig(road_network_buffer_m=50)
    road_left, road_bottom, road_right, road_top = config.road_bbox
    geod = Geod(ellps="WGS84")
    middle_longitude = (config.left + config.right) / 2
    middle_latitude = (config.bottom + config.top) / 2

    distances = (
        geod.inv(config.left, middle_latitude, road_left, middle_latitude)[2],
        geod.inv(middle_longitude, config.bottom, middle_longitude, road_bottom)[2],
        geod.inv(config.right, middle_latitude, road_right, middle_latitude)[2],
        geod.inv(middle_longitude, config.top, middle_longitude, road_top)[2],
    )

    assert all(distance == pytest.approx(50, abs=0.05) for distance in distances)
    assert config.bbox == (config.left, config.bottom, config.right, config.top)


def test_configuration_fingerprint_tracks_spatial_inputs_not_cache_location(
    tmp_path: Path,
) -> None:
    config = DomainConfig()
    assert replace(config, cache_directory=tmp_path).fingerprint == config.fingerprint
    assert replace(config, osmnx_cache_directory=tmp_path).fingerprint == config.fingerprint
    assert replace(config, left=config.left + 0.001).fingerprint != config.fingerprint
    assert replace(config, cell_size_m=50).fingerprint != config.fingerprint
    assert replace(config, network_type="walk").fingerprint != config.fingerprint
    assert replace(config, road_network_buffer_m=75).fingerprint != config.fingerprint


def test_build_grid_uses_complete_cells_and_southwest_origin() -> None:
    grid = build_grid(box(0, 0, 250, 180), "EPSG:32718", 100)

    assert grid.shape == (2, 3)
    assert grid.cell_area_m2 == 10_000
    assert grid.grid_bounds_m.tolist() == [0, 0, 300, 200]
    assert grid.x_centers_m.tolist() == [50, 150, 250]
    assert grid.y_centers_m.tolist() == [50, 150]
    assert grid.active_mask.all()
    assert grid.cell_center(0, 0) == (50, 50)
    assert grid.cell_center(1, 0) == (50, 150)


def test_grid_coordinate_conversion_round_trip() -> None:
    grid = build_grid(box(1000, 2000, 1250, 2180), "EPSG:32718", 100)

    for row in range(grid.rows):
        for column in range(grid.columns):
            assert grid.xy_to_cell(*grid.cell_center(row, column)) == (row, column)

    assert grid.xy_to_cell(999, 2000) is None
    assert grid.xy_to_cell(1300, 2000) is None
    assert not grid.contains_xy(1000, 2200)
    with pytest.raises(IndexError):
        grid.cell_center(grid.rows, 0)


def test_grid_npz_round_trip(tmp_path: Path) -> None:
    original = build_grid(box(0, 0, 250, 180), "EPSG:32718", 100)
    path = tmp_path / "grid.npz"

    save_grid(original, path, "fingerprint")
    restored = load_grid(path, "fingerprint", "EPSG:32718")

    assert restored.crs == original.crs
    assert restored.shape == original.shape
    assert np.array_equal(restored.domain_bounds_m, original.domain_bounds_m)
    assert np.array_equal(restored.grid_bounds_m, original.grid_bounds_m)
    assert np.array_equal(restored.x_centers_m, original.x_centers_m)
    assert np.array_equal(restored.y_centers_m, original.y_centers_m)
    assert np.array_equal(restored.active_mask, original.active_mask)


def test_grid_npz_rejects_wrong_fingerprint(tmp_path: Path) -> None:
    grid = build_grid(box(0, 0, 100, 100), "EPSG:32718", 100)
    path = tmp_path / "grid.npz"
    save_grid(grid, path, "expected")

    with pytest.raises(ValueError, match="fingerprint"):
        load_grid(path, "different", "EPSG:32718")


def test_spatial_domain_reuses_processed_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DomainConfig(
        cache_directory=tmp_path / "cache" / "domain",
        osmnx_cache_directory=tmp_path / "cache" / "osmnx",
    )
    graph = _projected_test_graph()
    downloads = 0

    def download(_: DomainConfig) -> nx.MultiDiGraph:
        nonlocal downloads
        downloads += 1
        return graph.copy()

    monkeypatch.setattr(domain_module, "_download_and_project_graph", download)
    monkeypatch.setattr(
        domain_module,
        "_project_boundary",
        lambda _config, _crs: box(0, 0, 200, 200),
    )
    monkeypatch.setattr(domain_module, "_validate_projected_graph", lambda _graph: None)

    first = create_spatial_domain(config)
    second = create_spatial_domain(config)

    assert downloads == 1
    assert first.graph_source == "overpass"
    assert first.grid_source == "generated"
    assert second.graph_source == "cache"
    assert second.grid_source == "cache"
    assert (config.cache_directory / "road_graph.graphml").is_file()
    assert (config.cache_directory / "grid.npz").is_file()
    assert (config.cache_directory / "metadata.json").is_file()


def test_spatial_domain_rebuilds_corrupt_graph_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DomainConfig(
        cache_directory=tmp_path / "cache" / "domain",
        osmnx_cache_directory=tmp_path / "cache" / "osmnx",
    )
    graph = _projected_test_graph()
    downloads = 0

    def download(_: DomainConfig) -> nx.MultiDiGraph:
        nonlocal downloads
        downloads += 1
        return graph.copy()

    monkeypatch.setattr(domain_module, "_download_and_project_graph", download)
    monkeypatch.setattr(
        domain_module,
        "_project_boundary",
        lambda _config, _crs: box(0, 0, 200, 200),
    )
    monkeypatch.setattr(domain_module, "_validate_projected_graph", lambda _graph: None)

    create_spatial_domain(config)
    (config.cache_directory / "road_graph.graphml").write_text(
        "not graphml",
        encoding="ascii",
    )
    rebuilt = create_spatial_domain(config)

    assert downloads == 2
    assert rebuilt.graph_source == "overpass"


def test_cell_size_change_reuses_graph_and_rebuilds_only_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DomainConfig(
        cell_size_m=100,
        cache_directory=tmp_path / "cache" / "domain",
        osmnx_cache_directory=tmp_path / "cache" / "osmnx",
    )
    graph = _projected_test_graph()
    downloads = 0

    def download(_: DomainConfig) -> nx.MultiDiGraph:
        nonlocal downloads
        downloads += 1
        return graph.copy()

    monkeypatch.setattr(domain_module, "_download_and_project_graph", download)
    monkeypatch.setattr(
        domain_module,
        "_project_boundary",
        lambda _config, _crs: box(0, 0, 200, 200),
    )
    monkeypatch.setattr(domain_module, "_validate_projected_graph", lambda _graph: None)

    first = create_spatial_domain(config)
    second = create_spatial_domain(replace(config, cell_size_m=200))

    assert downloads == 1
    assert first.graph_source == "overpass"
    assert second.graph_source == "cache"
    assert first.grid.cell_size_m == 100
    assert second.grid.cell_size_m == 200
    assert second.grid_source == "generated"


def _projected_test_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:32718"
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=100.0, y=0.0)
    graph.add_edge(
        1,
        2,
        key=0,
        osmid=1,
        length=100.0,
        geometry=LineString([(0, 0), (100, 0)]),
    )
    return graph
