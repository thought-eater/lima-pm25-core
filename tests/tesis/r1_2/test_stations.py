"""Ubicación del catálogo PM2.5 y condiciones de validez de sus estaciones."""

from dataclasses import FrozenInstanceError

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform

from lima_pm25.domain import (
    DomainConfig,
    FixedStationCatalog,
    FixedStationDefinition,
    LIMA_PM25_STATION_DEFINITIONS,
    SENAMHI_PM25_PROVENANCE,
    SpatialDomain,
    build_grid,
    create_lima_pm25_station_catalog,
    plot_fixed_station_catalog,
    project_fixed_station_site,
)


def test_initial_lima_pm25_catalog_projects_real_station_coordinates() -> None:
    domain = _lima_test_domain()

    catalog = create_lima_pm25_station_catalog(domain)

    assert [site.site_id for site in catalog.sites] == [
        "CAMPO_DE_MARTE",
        "CERES",
        "SANTA_ANITA",
        "SAN_JUAN_DE_LURIGANCHO",
    ]
    transformer = Transformer.from_crs("EPSG:4326", domain.grid.crs, always_xy=True)
    for definition, site in zip(
        LIMA_PM25_STATION_DEFINITIONS,
        catalog.sites,
        strict=True,
    ):
        expected_x, expected_y = transformer.transform(
            definition.longitude_deg,
            definition.latitude_deg,
        )
        assert site.x_m == pytest.approx(expected_x)
        assert site.y_m == pytest.approx(expected_y)
        assert domain.grid.is_active_xy(site.x_m, site.y_m)
        assert site.provenance == SENAMHI_PM25_PROVENANCE
        assert site.provenance.resource_name == "pm25_lima_2025_2026_imputed.csv"


def test_fixed_station_site_is_immutable_and_has_no_sensor_behavior() -> None:
    site = create_lima_pm25_station_catalog(_lima_test_domain()).sites[0]

    with pytest.raises(FrozenInstanceError):
        site.x_m = 0  # type: ignore[misc]

    assert not {
        "sampling_interval_s",
        "bias_ug_m3",
        "noise_stddev_ug_m3",
        "dropout_probability",
        "latest_measurement",
    }.intersection(site.__slots__)


def test_station_projection_rejects_a_site_outside_the_study_domain() -> None:
    definition = FixedStationDefinition(
        site_id="OUTSIDE",
        longitude_deg=-77.2,
        latitude_deg=-12.0,
        altitude_m=None,
        provenance=SENAMHI_PM25_PROVENANCE,
    )

    with pytest.raises(ValueError, match="outside the study domain"):
        project_fixed_station_site(definition, _lima_test_domain())


def test_station_catalog_rejects_duplicate_site_identity() -> None:
    site = create_lima_pm25_station_catalog(_lima_test_domain()).sites[0]

    with pytest.raises(ValueError, match="Duplicate station site_id"):
        FixedStationCatalog((site, site))


def test_fixed_station_plot_uses_projected_positions_and_readable_labels() -> None:
    catalog = create_lima_pm25_station_catalog(_lima_test_domain())
    figure, axis = plt.subplots()

    collection, annotations = plot_fixed_station_catalog(axis, catalog)

    assert np.allclose(
        collection.get_offsets(),
        [(site.x_m, site.y_m) for site in catalog.sites],
    )
    assert [annotation.get_text() for annotation in annotations] == [
        "Campo de Marte",
        "Ceres",
        "Santa Anita",
        "San Juan de Lurigancho",
    ]
    plt.close(figure)


def _lima_test_domain() -> SpatialDomain:
    config = DomainConfig()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
    boundary = transform(
        transformer.transform,
        Polygon(
            [
                (config.left, config.bottom),
                (config.right, config.bottom),
                (config.right, config.top),
                (config.left, config.top),
                (config.left, config.bottom),
            ]
        ),
    )
    grid = build_grid(boundary, "EPSG:32718", config.cell_size_m)
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = grid.crs
    return SpatialDomain(
        config=config,
        road_graph=graph,
        boundary=boundary,
        grid=grid,
        graph_source="test",
        grid_source="test",
    )
