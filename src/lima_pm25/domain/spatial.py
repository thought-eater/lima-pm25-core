from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import CRS, Geod
from pyproj import __version__ as pyproj_version
from shapely import __version__ as shapely_version
from shapely import intersects_xy
from shapely.errors import ShapelyError
from shapely.geometry import LineString, Polygon, box

DOMAIN_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class DomainConfig:
    left: float = -77.05108161742221
    bottom: float = -12.079
    right: float = -76.90976879099601
    top: float = -11.974250416076465
    network_type: str = "drive"
    cell_size_m: float = 100.0
    road_network_buffer_m: float = 50.0
    cache_directory: Path = Path("cache/domain")
    osmnx_cache_directory: Path = Path("cache/osmnx")

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_directory", Path(self.cache_directory))
        object.__setattr__(
            self,
            "osmnx_cache_directory",
            Path(self.osmnx_cache_directory),
        )
        if not -180 <= self.left < self.right <= 180:
            raise ValueError(
                "Longitude bounds must satisfy -180 <= left < right <= 180"
            )
        if not -90 <= self.bottom < self.top <= 90:
            raise ValueError("Latitude bounds must satisfy -90 <= bottom < top <= 90")
        if not self.network_type:
            raise ValueError("network_type cannot be empty")
        if not math.isfinite(self.cell_size_m) or self.cell_size_m <= 0:
            raise ValueError("cell_size_m must be a positive finite number")
        if (
            not math.isfinite(self.road_network_buffer_m)
            or self.road_network_buffer_m < 0
        ):
            raise ValueError("road_network_buffer_m must be nonnegative and finite")

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.left, self.bottom, self.right, self.top

    @property
    def road_bbox(self) -> tuple[float, float, float, float]:
        if self.road_network_buffer_m == 0:
            return self.bbox

        geod = Geod(ellps="WGS84")
        middle_longitude = (self.left + self.right) / 2
        middle_latitude = (self.bottom + self.top) / 2
        buffered_left, _, _ = geod.fwd(
            self.left,
            middle_latitude,
            270,
            self.road_network_buffer_m,
        )
        buffered_right, _, _ = geod.fwd(
            self.right,
            middle_latitude,
            90,
            self.road_network_buffer_m,
        )
        _, buffered_bottom, _ = geod.fwd(
            middle_longitude,
            self.bottom,
            180,
            self.road_network_buffer_m,
        )
        _, buffered_top, _ = geod.fwd(
            middle_longitude,
            self.top,
            0,
            self.road_network_buffer_m,
        )
        return buffered_left, buffered_bottom, buffered_right, buffered_top

    @property
    def fingerprint(self) -> str:
        """Identify spatial configuration for cache reuse, not OSM graph content."""
        payload = {
            "bbox": self.bbox,
            "cell_size_m": self.cell_size_m,
            "network_type": self.network_type,
            "road_network_buffer_m": self.road_network_buffer_m,
            "schema_version": DOMAIN_SCHEMA_VERSION,
        }
        serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return sha256(serialized.encode("utf-8")).hexdigest()[:16]


DEFAULT_DOMAIN_CONFIG = DomainConfig()


@dataclass(frozen=True, slots=True)
class GridDefinition:
    crs: str
    domain_bounds_m: np.ndarray
    grid_bounds_m: np.ndarray
    cell_size_m: float
    x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    active_mask: np.ndarray

    def __post_init__(self) -> None:
        domain_bounds = _immutable_array(self.domain_bounds_m, np.float64)
        grid_bounds = _immutable_array(self.grid_bounds_m, np.float64)
        x_centers = _immutable_array(self.x_centers_m, np.float64)
        y_centers = _immutable_array(self.y_centers_m, np.float64)
        active_mask = _immutable_array(self.active_mask, np.bool_)

        object.__setattr__(self, "domain_bounds_m", domain_bounds)
        object.__setattr__(self, "grid_bounds_m", grid_bounds)
        object.__setattr__(self, "x_centers_m", x_centers)
        object.__setattr__(self, "y_centers_m", y_centers)
        object.__setattr__(self, "active_mask", active_mask)

        if domain_bounds.shape != (4,) or grid_bounds.shape != (4,):
            raise ValueError("Grid bounds must have shape (4,)")
        if not np.all(np.isfinite(domain_bounds)) or not np.all(
            np.isfinite(grid_bounds)
        ):
            raise ValueError("Grid bounds must be finite")
        if not math.isfinite(self.cell_size_m) or self.cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive and finite")
        if x_centers.ndim != 1 or y_centers.ndim != 1:
            raise ValueError("Grid center coordinates must be one-dimensional")
        if not x_centers.size or not y_centers.size:
            raise ValueError("Grid center coordinates cannot be empty")
        if active_mask.shape != (y_centers.size, x_centers.size):
            raise ValueError("active_mask shape must match the grid dimensions")
        if not np.all(np.isfinite(x_centers)) or not np.all(np.isfinite(y_centers)):
            raise ValueError("Grid center coordinates must be finite")
        if x_centers.size > 1 and not np.allclose(np.diff(x_centers), self.cell_size_m):
            raise ValueError("x center spacing does not match cell_size_m")
        if y_centers.size > 1 and not np.allclose(np.diff(y_centers), self.cell_size_m):
            raise ValueError("y center spacing does not match cell_size_m")

        CRS.from_user_input(self.crs)

    @property
    def shape(self) -> tuple[int, int]:
        return self.active_mask.shape

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def columns(self) -> int:
        return self.shape[1]

    @property
    def cell_area_m2(self) -> float:
        return self.cell_size_m**2

    @property
    def width_m(self) -> float:
        return self.columns * self.cell_size_m

    @property
    def height_m(self) -> float:
        return self.rows * self.cell_size_m

    def contains_xy(self, x: float, y: float) -> bool:
        x_min, y_min, x_max, y_max = self.grid_bounds_m
        return bool(x_min <= x < x_max and y_min <= y < y_max)

    def xy_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        if not self.contains_xy(x, y):
            return None

        x_min, y_min, _, _ = self.grid_bounds_m
        column = math.floor((x - x_min) / self.cell_size_m)
        row = math.floor((y - y_min) / self.cell_size_m)
        return row, column

    def cell_center(self, row: int, column: int) -> tuple[float, float]:
        if not 0 <= row < self.rows or not 0 <= column < self.columns:
            raise IndexError(
                f"Cell ({row}, {column}) is outside grid shape {self.shape}"
            )
        return float(self.x_centers_m[column]), float(self.y_centers_m[row])

    def is_active_xy(self, x: float, y: float) -> bool:
        cell = self.xy_to_cell(x, y)
        return cell is not None and bool(self.active_mask[cell])


@dataclass(frozen=True, slots=True)
class SpatialDomain:
    config: DomainConfig
    road_graph: nx.MultiDiGraph
    boundary: Polygon
    grid: GridDefinition
    graph_source: str
    grid_source: str


def create_spatial_domain(
    config: DomainConfig = DEFAULT_DOMAIN_CONFIG,
    *,
    force_refresh: bool = False,
) -> SpatialDomain:
    config.cache_directory.mkdir(parents=True, exist_ok=True)
    config.osmnx_cache_directory.mkdir(parents=True, exist_ok=True)

    ox.settings.use_cache = True
    ox.settings.cache_folder = config.osmnx_cache_directory

    graph_path = config.cache_directory / "road_graph.graphml"
    grid_path = config.cache_directory / "grid.npz"
    metadata_path = config.cache_directory / "metadata.json"
    cached_metadata = _read_metadata(metadata_path)
    cache_is_compatible = _metadata_is_compatible(cached_metadata, config)
    graph_cache_is_compatible = _graph_metadata_is_compatible(
        cached_metadata,
        config,
    )

    if not force_refresh and graph_cache_is_compatible and graph_path.exists():
        try:
            graph = ox.io.load_graphml(graph_path)
            _validate_projected_graph(graph)
            graph_source = "cache"
        except (
            AttributeError,
            KeyError,
            OSError,
            ShapelyError,
            SyntaxError,
            TypeError,
            ValueError,
            nx.NetworkXException,
        ):
            # GraphML readers expose backend-specific parse errors; the cache is disposable.
            graph = _download_and_project_graph(config)
            ox.io.save_graphml(graph, graph_path)
            graph_source = "overpass"
    else:
        graph = _download_and_project_graph(config)
        ox.io.save_graphml(graph, graph_path)
        graph_source = "overpass"

    boundary = _project_boundary(config, graph.graph["crs"])
    crs_string = CRS.from_user_input(graph.graph["crs"]).to_string()

    if not force_refresh and cache_is_compatible and grid_path.exists():
        try:
            grid = load_grid(grid_path, config.fingerprint, crs_string)
            grid_source = "cache"
        except KeyError, OSError, TypeError, ValueError:
            grid = build_grid(boundary, crs_string, config.cell_size_m)
            save_grid(grid, grid_path, config.fingerprint)
            grid_source = "generated"
    else:
        grid = build_grid(boundary, crs_string, config.cell_size_m)
        save_grid(grid, grid_path, config.fingerprint)
        grid_source = "generated"

    metadata_is_current = (
        graph_source == "cache"
        and grid_source == "cache"
        and _metadata_is_complete(cached_metadata)
    )
    if not metadata_is_current:
        _write_json(
            metadata_path,
            _build_metadata(
                config,
                graph,
                grid,
                cached_metadata if cache_is_compatible else None,
                preserve_graph_timestamp=graph_source == "cache",
                preserve_grid_timestamp=grid_source == "cache",
            ),
        )

    return SpatialDomain(
        config=config,
        road_graph=graph,
        boundary=boundary,
        grid=grid,
        graph_source=graph_source,
        grid_source=grid_source,
    )


def build_grid(
    boundary: Polygon,
    crs: str | CRS,
    cell_size_m: float,
) -> GridDefinition:
    if boundary.is_empty or not boundary.is_valid:
        raise ValueError("The projected domain boundary must be valid and non-empty")
    if not math.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive and finite")

    crs_object = CRS.from_user_input(crs)
    _validate_metric_crs(crs_object)

    domain_bounds = np.asarray(boundary.bounds, dtype=np.float64)
    x_min, y_min, x_max, y_max = domain_bounds
    columns = math.ceil((x_max - x_min) / cell_size_m)
    rows = math.ceil((y_max - y_min) / cell_size_m)
    grid_x_max = x_min + columns * cell_size_m
    grid_y_max = y_min + rows * cell_size_m
    grid_bounds = np.array([x_min, y_min, grid_x_max, grid_y_max], dtype=np.float64)

    x_centers = x_min + (np.arange(columns, dtype=np.float64) + 0.5) * cell_size_m
    y_centers = y_min + (np.arange(rows, dtype=np.float64) + 0.5) * cell_size_m
    center_x, center_y = np.meshgrid(x_centers, y_centers)
    active_mask = intersects_xy(boundary, center_x, center_y)

    return GridDefinition(
        crs=crs_object.to_string(),
        domain_bounds_m=domain_bounds,
        grid_bounds_m=grid_bounds,
        cell_size_m=cell_size_m,
        x_centers_m=x_centers,
        y_centers_m=y_centers,
        active_mask=active_mask,
    )


def save_grid(grid: GridDefinition, path: Path, fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            schema_version=np.array(DOMAIN_SCHEMA_VERSION, dtype=np.int64),
            config_fingerprint=np.array(fingerprint),
            crs=np.array(grid.crs),
            domain_bounds_m=grid.domain_bounds_m,
            grid_bounds_m=grid.grid_bounds_m,
            cell_size_m=np.array(grid.cell_size_m, dtype=np.float64),
            shape=np.asarray(grid.shape, dtype=np.int64),
            x_centers_m=grid.x_centers_m,
            y_centers_m=grid.y_centers_m,
            active_mask=grid.active_mask,
        )
    temporary_path.replace(path)


def load_grid(path: Path, fingerprint: str, expected_crs: str | CRS) -> GridDefinition:
    required_keys = {
        "schema_version",
        "config_fingerprint",
        "crs",
        "domain_bounds_m",
        "grid_bounds_m",
        "cell_size_m",
        "shape",
        "x_centers_m",
        "y_centers_m",
        "active_mask",
    }
    with np.load(path, allow_pickle=False) as data:
        if not required_keys.issubset(data.files):
            missing = required_keys.difference(data.files)
            raise KeyError(f"grid.npz is missing keys: {sorted(missing)}")
        if int(data["schema_version"].item()) != DOMAIN_SCHEMA_VERSION:
            raise ValueError("grid.npz schema version is incompatible")
        if str(data["config_fingerprint"].item()) != fingerprint:
            raise ValueError("grid.npz fingerprint does not match DomainConfig")

        crs = str(data["crs"].item())
        if CRS.from_user_input(crs) != CRS.from_user_input(expected_crs):
            raise ValueError("Grid and graph use different coordinate systems")

        grid = GridDefinition(
            crs=crs,
            domain_bounds_m=data["domain_bounds_m"].copy(),
            grid_bounds_m=data["grid_bounds_m"].copy(),
            cell_size_m=float(data["cell_size_m"].item()),
            x_centers_m=data["x_centers_m"].copy(),
            y_centers_m=data["y_centers_m"].copy(),
            active_mask=data["active_mask"].copy(),
        )
        stored_shape = tuple(int(value) for value in data["shape"])

    if stored_shape != grid.shape:
        raise ValueError("Stored grid shape does not match active_mask")
    return grid


def plot_spatial_domain(
    domain: SpatialDomain,
    output_path: Path,
    *,
    station_catalog=None,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges = ox.graph_to_gdfs(domain.road_graph, nodes=False)
    figure, axis = plt.subplots(figsize=(12, 9))
    x_min, y_min, x_max, y_max = domain.grid.grid_bounds_m
    active_overlay = np.where(domain.grid.active_mask, 1.0, np.nan)
    axis.imshow(
        active_overlay,
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        cmap="Blues",
        alpha=0.08,
        interpolation="nearest",
        zorder=1,
    )
    edges.plot(
        kind="geo",
        ax=axis,
        color=(85 / 255, 85 / 255, 85 / 255),
        linewidth=0.35,
        zorder=2,
    )

    boundary_x, boundary_y = domain.boundary.exterior.xy
    axis.plot(boundary_x, boundary_y, color="#c62828", linewidth=1.5, zorder=4)

    major_step = max(1, round(500 / domain.grid.cell_size_m))
    x_lines = (
        x_min
        + np.arange(0, domain.grid.columns + 1, major_step) * domain.grid.cell_size_m
    )
    y_lines = (
        y_min + np.arange(0, domain.grid.rows + 1, major_step) * domain.grid.cell_size_m
    )
    axis.vlines(x_lines, y_min, y_max, color="#1976d2", alpha=0.18, linewidth=0.4)
    axis.hlines(y_lines, x_min, x_max, color="#1976d2", alpha=0.18, linewidth=0.4)
    axis.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        color="#1976d2",
        linewidth=1,
        zorder=3,
    )

    if station_catalog is not None:
        from .stations import plot_fixed_station_catalog

        plot_fixed_station_catalog(axis, station_catalog)

    axis.set_title(
        f"Red vial, estaciones y grilla de {domain.grid.cell_size_m:g} m "
        f"({domain.grid.crs})"
    )
    axis.set_xlabel("Este (m)")
    axis.set_ylabel("Norte (m)")
    axis.set_aspect("equal")
    axis.text(
        0.995,
        0.005,
        "© OpenStreetMap contributors · ODbL",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#303030",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        zorder=10,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _download_and_project_graph(config: DomainConfig) -> nx.MultiDiGraph:
    graph = ox.graph_from_bbox(config.road_bbox, network_type=config.network_type)
    projected_graph = ox.project_graph(graph)
    _add_missing_edge_geometries(projected_graph)
    _validate_projected_graph(projected_graph)
    return projected_graph


def _add_missing_edge_geometries(graph: nx.MultiDiGraph) -> None:
    for u, v, _, edge_data in graph.edges(keys=True, data=True):
        if edge_data.get("geometry") is None:
            source = graph.nodes[u]
            target = graph.nodes[v]
            edge_data["geometry"] = LineString(
                [(source["x"], source["y"]), (target["x"], target["y"])]
            )


def _validate_projected_graph(graph: nx.MultiDiGraph) -> None:
    if not graph.nodes or not graph.edges:
        raise ValueError("The road graph cannot be empty")
    if "crs" not in graph.graph:
        raise ValueError("The road graph does not define a CRS")

    crs = CRS.from_user_input(graph.graph["crs"])
    _validate_metric_crs(crs)
    ox.convert.validate_graph(graph)

    for _, node_data in graph.nodes(data=True):
        if "x" not in node_data or "y" not in node_data:
            raise ValueError("Every road graph node must define x and y")
        if not math.isfinite(float(node_data["x"])) or not math.isfinite(
            float(node_data["y"])
        ):
            raise ValueError("Road graph node coordinates must be finite")

    for _, _, _, edge_data in graph.edges(keys=True, data=True):
        try:
            length = float(edge_data["length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Every road graph edge must have a numeric length"
            ) from error
        if not math.isfinite(length) or length <= 0:
            raise ValueError("Every road graph edge length must be positive")


def _validate_metric_crs(crs: CRS) -> None:
    if not crs.is_projected:
        raise ValueError("The simulation CRS must be projected")
    if not crs.axis_info:
        raise ValueError("The simulation CRS does not report axis units")
    for axis in crs.axis_info[:2]:
        if not math.isclose(axis.unit_conversion_factor or 0, 1.0):
            raise ValueError("The simulation CRS axes must use meters")


def _project_boundary(config: DomainConfig, graph_crs: Any) -> Polygon:
    boundary_wgs84 = box(config.left, config.bottom, config.right, config.top)
    projected, _ = ox.projection.project_geometry(
        boundary_wgs84,
        crs="EPSG:4326",
        to_crs=graph_crs,
    )
    if not isinstance(projected, Polygon):
        raise TypeError("The projected bounding box must be a Polygon")
    return projected


def _read_metadata(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError, OSError, UnicodeError:
        return None
    return content if isinstance(content, dict) else None


def _metadata_is_compatible(
    metadata: dict[str, Any] | None,
    config: DomainConfig,
) -> bool:
    return bool(
        metadata
        and metadata.get("schema_version") == DOMAIN_SCHEMA_VERSION
        and metadata.get("config_fingerprint") == config.fingerprint
    )


def _graph_metadata_is_compatible(
    metadata: dict[str, Any] | None,
    config: DomainConfig,
) -> bool:
    if not metadata or metadata.get("schema_version") != DOMAIN_SCHEMA_VERSION:
        return False
    bbox = metadata.get("bbox_wgs84")
    return bool(
        isinstance(bbox, dict)
        and bbox.get("left") == config.left
        and bbox.get("bottom") == config.bottom
        and bbox.get("right") == config.right
        and bbox.get("top") == config.top
        and metadata.get("network_type") == config.network_type
        and metadata.get("road_network_buffer_m") == config.road_network_buffer_m
    )


def _metadata_is_complete(metadata: dict[str, Any] | None) -> bool:
    if metadata is None:
        return False
    software = metadata.get("software")
    return isinstance(software, dict) and {
        "python",
        "networkx",
        "numpy",
        "osmnx",
        "pyproj",
        "shapely",
    }.issubset(software)


def _build_metadata(
    config: DomainConfig,
    graph: nx.MultiDiGraph,
    grid: GridDefinition,
    previous: dict[str, Any] | None,
    *,
    preserve_graph_timestamp: bool,
    preserve_grid_timestamp: bool,
) -> dict[str, Any]:
    generated_at = datetime.now(UTC).isoformat()
    previous_graph = previous.get("graph", {}) if previous else {}
    previous_grid = previous.get("grid", {}) if previous else {}

    return {
        "schema_version": DOMAIN_SCHEMA_VERSION,
        "config_fingerprint": config.fingerprint,
        "bbox_wgs84": {
            "left": config.left,
            "bottom": config.bottom,
            "right": config.right,
            "top": config.top,
        },
        "network_type": config.network_type,
        "road_network_buffer_m": config.road_network_buffer_m,
        "road_bbox_wgs84": {
            "left": config.road_bbox[0],
            "bottom": config.road_bbox[1],
            "right": config.road_bbox[2],
            "top": config.road_bbox[3],
        },
        "cell_size_m": config.cell_size_m,
        "projected_crs": grid.crs,
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "generated_at_utc": (
                previous_graph.get("generated_at_utc", generated_at)
                if preserve_graph_timestamp
                else generated_at
            ),
        },
        "grid": {
            "rows": grid.rows,
            "columns": grid.columns,
            "active_cells": int(np.count_nonzero(grid.active_mask)),
            "domain_bounds_m": grid.domain_bounds_m.tolist(),
            "grid_bounds_m": grid.grid_bounds_m.tolist(),
            "generated_at_utc": (
                previous_grid.get("generated_at_utc", generated_at)
                if preserve_grid_timestamp
                else generated_at
            ),
        },
        "software": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "numpy": np.__version__,
            "osmnx": ox.__version__,
            "pyproj": pyproj_version,
            "shapely": shapely_version,
        },
    }


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _immutable_array(value: object, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
