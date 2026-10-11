from __future__ import annotations

import math
from dataclasses import dataclass, field

from pyproj import CRS, Transformer
from shapely.geometry import Point

from .spatial import SpatialDomain


def _require_usable_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class StationProvenance:
    contract_id: str
    contract_version: str
    resource_name: str

    def __post_init__(self) -> None:
        _require_usable_text("contract_id", self.contract_id)
        _require_usable_text("contract_version", self.contract_version)
        _require_usable_text("resource_name", self.resource_name)


SENAMHI_PM25_PROVENANCE = StationProvenance(
    contract_id="senamhi-hourly-imputed",
    contract_version="1.0.0",
    resource_name="pm25_lima_2025_2026_imputed.csv",
)


@dataclass(frozen=True, slots=True)
class FixedStationDefinition:
    site_id: str
    longitude_deg: float
    latitude_deg: float
    altitude_m: float | None
    provenance: StationProvenance

    def __post_init__(self) -> None:
        _require_usable_text("site_id", self.site_id)
        if (
            not math.isfinite(self.longitude_deg)
            or not -180 <= self.longitude_deg <= 180
        ):
            raise ValueError("longitude_deg must be finite and between -180 and 180")
        if not math.isfinite(self.latitude_deg) or not -90 <= self.latitude_deg <= 90:
            raise ValueError("latitude_deg must be finite and between -90 and 90")
        if self.altitude_m is not None and not math.isfinite(self.altitude_m):
            raise ValueError("altitude_m must be finite when provided")


LIMA_PM25_STATION_DEFINITIONS = (
    FixedStationDefinition(
        site_id="CAMPO_DE_MARTE",
        longitude_deg=-77.0432,
        latitude_deg=-12.0705,
        altitude_m=117.0,
        provenance=SENAMHI_PM25_PROVENANCE,
    ),
    FixedStationDefinition(
        site_id="CERES",
        longitude_deg=-76.92706,
        latitude_deg=-12.02869,
        altitude_m=None,
        provenance=SENAMHI_PM25_PROVENANCE,
    ),
    FixedStationDefinition(
        site_id="SANTA_ANITA",
        longitude_deg=-76.9714,
        latitude_deg=-12.043,
        altitude_m=253.0,
        provenance=SENAMHI_PM25_PROVENANCE,
    ),
    FixedStationDefinition(
        site_id="SAN_JUAN_DE_LURIGANCHO",
        longitude_deg=-76.99925,
        latitude_deg=-11.9808,
        altitude_m=240.0,
        provenance=SENAMHI_PM25_PROVENANCE,
    ),
)

_STATION_DISPLAY_NAMES = {
    "CAMPO_DE_MARTE": "Campo de Marte",
    "CERES": "Ceres",
    "SANTA_ANITA": "Santa Anita",
    "SAN_JUAN_DE_LURIGANCHO": "San Juan de Lurigancho",
}


@dataclass(frozen=True, slots=True)
class FixedStationSite:
    site_id: str
    longitude_deg: float
    latitude_deg: float
    altitude_m: float | None
    x_m: float
    y_m: float
    crs: str
    provenance: StationProvenance

    def __post_init__(self) -> None:
        _require_usable_text("site_id", self.site_id)
        if (
            not math.isfinite(self.longitude_deg)
            or not -180 <= self.longitude_deg <= 180
        ):
            raise ValueError("longitude_deg must be finite and between -180 and 180")
        if not math.isfinite(self.latitude_deg) or not -90 <= self.latitude_deg <= 90:
            raise ValueError("latitude_deg must be finite and between -90 and 90")
        if self.altitude_m is not None and not math.isfinite(self.altitude_m):
            raise ValueError("altitude_m must be finite when provided")
        if not math.isfinite(self.x_m) or not math.isfinite(self.y_m):
            raise ValueError("Projected station coordinates must be finite")
        projected_crs = CRS.from_user_input(self.crs)
        if not projected_crs.is_projected:
            raise ValueError("Station CRS must be projected")


@dataclass(frozen=True, slots=True)
class FixedStationCatalog:
    sites: tuple[FixedStationSite, ...]
    _by_id: dict[str, FixedStationSite] = field(
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        sites = tuple(self.sites)
        by_id: dict[str, FixedStationSite] = {}
        for site in sites:
            if site.site_id in by_id:
                raise ValueError(f"Duplicate station site_id {site.site_id!r}")
            by_id[site.site_id] = site
        object.__setattr__(self, "sites", sites)
        object.__setattr__(self, "_by_id", by_id)

    def site(self, site_id: str) -> FixedStationSite:
        try:
            return self._by_id[site_id]
        except KeyError as error:
            raise KeyError(f"Unknown fixed station site_id {site_id!r}") from error


def project_fixed_station_site(
    definition: FixedStationDefinition,
    domain: SpatialDomain,
) -> FixedStationSite:
    transformer = Transformer.from_crs(
        "EPSG:4326",
        domain.grid.crs,
        always_xy=True,
    )
    x_m, y_m = transformer.transform(
        definition.longitude_deg,
        definition.latitude_deg,
        errcheck=True,
    )
    x_m = float(x_m)
    y_m = float(y_m)
    if not math.isfinite(x_m) or not math.isfinite(y_m):
        raise ValueError(f"Station {definition.site_id!r} projection is not finite")
    if not domain.boundary.covers(Point(x_m, y_m)):
        raise ValueError(f"Station {definition.site_id!r} is outside the study domain")
    if not domain.grid.is_active_xy(x_m, y_m):
        raise ValueError(
            f"Station {definition.site_id!r} is not in an active grid cell"
        )

    return FixedStationSite(
        site_id=definition.site_id,
        longitude_deg=definition.longitude_deg,
        latitude_deg=definition.latitude_deg,
        altitude_m=definition.altitude_m,
        x_m=x_m,
        y_m=y_m,
        crs=domain.grid.crs,
        provenance=definition.provenance,
    )


def create_lima_pm25_station_catalog(domain: SpatialDomain) -> FixedStationCatalog:
    return FixedStationCatalog(
        tuple(
            project_fixed_station_site(definition, domain)
            for definition in LIMA_PM25_STATION_DEFINITIONS
        )
    )


def plot_fixed_station_catalog(axis, catalog: FixedStationCatalog):
    """Plot fixed PM2.5 sites with presentation-safe labels."""
    collection = axis.scatter(
        [site.x_m for site in catalog.sites],
        [site.y_m for site in catalog.sites],
        marker="^",
        s=72,
        facecolor="#76ff03",
        edgecolor="#111111",
        linewidth=0.9,
        zorder=8,
        label="Estacion fija de PM2.5",
    )
    annotations = tuple(
        axis.annotate(
            _STATION_DISPLAY_NAMES.get(site.site_id, site.site_id),
            (site.x_m, site.y_m),
            xytext=(5, 5),
            textcoords="offset points",
            color="white",
            fontsize=7,
            fontweight="bold",
            zorder=9,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "#111111",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )
        for site in catalog.sites
    )
    return collection, annotations
