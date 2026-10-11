import csv
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from lima_pm25.domain import (
    AmbientBaseline,
    AtmosphericForcing,
    RegionalBoundaryForcing,
    build_grid,
)
from lima_pm25.simulation import (
    BoundaryMode,
    HistoricalWeatherDataset,
    MissingValuePolicy,
    PM25PhysicsConfig,
    PM25Transport,
    PrecipitationSourcePolicy,
    ScheduledWeatherValue,
    SyntheticWeatherWindow,
    WeatherCoverageError,
    WeatherCsvConfig,
    WeatherRecord,
    WeatherValueProvenance,
    iter_simulation,
    load_historical_weather_csv,
)

HEADERS = (
    "ESTACION",
    "FECHA",
    "HORA",
    "LONGITUD",
    "LATITUD",
    "ALTITUD",
    "TEMP",
    "HR",
    "PP",
    "DIR_VIENTO",
    "VEL_VIENTO",
    "RED",
    "DISTRITO",
    "TEMP_SOURCE",
    "HR_SOURCE",
    "PP_SOURCE",
    "DIR_VIENTO_SOURCE",
    "VEL_VIENTO_SOURCE",
    "PP_IFS",
)


def test_stochastic_synthetic_weather_requires_and_records_seed() -> None:
    forcing = AtmosphericForcing(
        "synthetic-weather",
        "synthetic weather test",
        1,
        90,
        0,
    )
    with pytest.raises(ValueError, match="requires a nonnegative seed"):
        SyntheticWeatherWindow(
            start_utc=datetime(2026, 1, 1, tzinfo=UTC),
            duration_s=60,
            values=(ScheduledWeatherValue(0, forcing),),
            construction_method="seeded-generator",
            stochastic=True,
        )

    weather = SyntheticWeatherWindow(
        start_utc=datetime(2026, 1, 1, tzinfo=UTC),
        duration_s=60,
        values=(ScheduledWeatherValue(0, forcing),),
        construction_method="seeded-generator",
        stochastic=True,
        seed=42,
    )

    descriptor = weather.evidence_descriptor()
    assert descriptor["construction_method"] == "seeded-generator"
    assert descriptor["stochastic"] is True
    assert descriptor["seed"] == 42


def test_loader_selects_one_station_and_normalizes_lima_time_to_utc(
    tmp_path: Path,
) -> None:
    path = _write_weather_csv(
        tmp_path,
        [
            _row("OTHER", 20250101, 0),
            _row("CAMPO_DE_MARTE", 20250101, 0, temperature=20, humidity=80),
            _row("CAMPO_DE_MARTE", 20250101, 10000, temperature=22, humidity=76),
        ],
    )

    dataset = load_historical_weather_csv(path, station_id="CAMPO_DE_MARTE")

    assert len(dataset.records) == 2
    assert dataset.coverage_start_utc == datetime(2025, 1, 1, 5, tzinfo=UTC)
    assert dataset.coverage_end_utc == datetime(2025, 1, 1, 6, tzinfo=UTC)
    assert dataset.records[0].temperature_c == 20
    assert dataset.records[0].relative_humidity_pct == 80
    assert dataset.provider_id == "historical-weather:campo_de_marte"


def test_loader_requires_source_columns(tmp_path: Path) -> None:
    path = tmp_path / "missing-source.csv"
    headers = tuple(column for column in HEADERS if column != "PP_SOURCE")
    row = _row("A", 20250101, 0)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerow({column: row[column] for column in headers})

    with pytest.raises(ValueError, match="missing required columns: PP_SOURCE"):
        load_historical_weather_csv(path, station_id="A")


def test_weather_record_requires_explicit_provenance_for_every_variable() -> None:
    with pytest.raises(TypeError, match="provenance"):
        WeatherRecord(  # ty: ignore[missing-argument] -- provenance is required
            station_id="A",
            timestamp_utc=datetime(2025, 1, 1, tzinfo=UTC),
            temperature_c=20,
            relative_humidity_pct=80,
            eastward_wind_m_s=0,
            northward_wind_m_s=0,
            precipitation_mm_h=0,
        )
    with pytest.raises(ValueError, match="unspecified"):
        WeatherValueProvenance("unspecified")


def test_loader_supports_configured_columns_timezone_and_missing_marker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "custom.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "site",
                "day",
                "hour",
                "temp",
                "rh",
                "rain",
                "direction",
                "speed",
                "temp_SOURCE",
                "rh_SOURCE",
                "rain_SOURCE",
                "direction_SOURCE",
                "speed_SOURCE",
            )
        )
        writer.writerow(
            (
                "A",
                "20250101",
                "0",
                "20",
                "80",
                "0",
                "NA",
                "0",
                "senamhi_observed",
                "senamhi_observed",
                "senamhi_observed",
                "not_applicable",
                "senamhi_observed",
            )
        )
        writer.writerow(
            (
                "A",
                "20250101",
                "10000",
                "21",
                "79",
                "0",
                "180",
                "1",
                "senamhi_observed",
                "senamhi_observed",
                "senamhi_observed",
                "senamhi_observed",
                "senamhi_observed",
            )
        )
    config = WeatherCsvConfig(
        station_column="site",
        date_column="day",
        time_column="hour",
        temperature_column="temp",
        relative_humidity_column="rh",
        precipitation_column="rain",
        wind_direction_column="direction",
        wind_speed_column="speed",
        source_timezone="UTC",
        missing_values=frozenset({"", "NA"}),
    )

    dataset = load_historical_weather_csv(path, station_id="A", config=config)

    assert dataset.records[0].timestamp_utc == datetime(2025, 1, 1, tzinfo=UTC)
    assert dataset.records[0].eastward_wind_m_s == 0
    assert dataset.records[0].northward_wind_m_s == 0
    assert dataset.records[0].wind_direction_provenance.source == "not_applicable"


def test_loader_preserves_upstream_outliers(
    tmp_path: Path,
) -> None:
    path = _write_weather_csv(
        tmp_path,
        [
            _row(
                "A",
                20250101,
                0,
                temperature=200,
                temperature_source="senamhi_operational_outlier",
            )
        ],
    )

    dataset = load_historical_weather_csv(path, station_id="A")

    assert dataset.records[0].temperature_c == 200
    assert dataset.records[0].temperature_provenance.source == (
        "senamhi_operational_outlier"
    )

    required_outlier = _write_weather_csv(
        tmp_path,
        [
            _row(
                "A",
                20250101,
                0,
                precipitation=80,
                precipitation_source="senamhi_operational_outlier",
            )
        ],
        name="required-outlier.csv",
    )
    with pytest.raises(ValueError, match="no usable weather records"):
        load_historical_weather_csv(required_outlier, station_id="A")

    unknown_timezone = replace(
        WeatherCsvConfig(),
        source_timezone="Unknown/Lima",
    )
    with pytest.raises(ValueError, match="Unknown source_timezone"):
        load_historical_weather_csv(
            path,
            station_id="A",
            config=unknown_timezone,
        )


def test_loader_rejects_an_unreadable_hour_code(tmp_path: Path) -> None:
    invalid_hour = _write_weather_csv(
        tmp_path,
        [_row("A", 20250101, 13000)],
        name="invalid-hour.csv",
    )

    with pytest.raises(ValueError, match="non-hourly HORA"):
        load_historical_weather_csv(invalid_hour, station_id="A")


def test_vector_interpolation_does_not_cross_to_southerly_wind(
    tmp_path: Path,
) -> None:
    path = _write_weather_csv(
        tmp_path,
        [
            _row("A", 20250101, 0, direction=350, speed=2),
            _row("A", 20250101, 10000, direction=10, speed=2),
        ],
    )
    dataset = load_historical_weather_csv(path, station_id="A")

    forcing = dataset.forcing_at(datetime(2025, 1, 1, 5, 30, tzinfo=UTC))

    assert forcing.wind_from_direction_deg == pytest.approx(0, abs=1e-12)
    assert forcing.wind_speed_m_s == pytest.approx(2 * math.cos(math.radians(10)))


def test_continuous_values_interpolate_and_precipitation_is_piecewise_constant(
    tmp_path: Path,
) -> None:
    path = _write_weather_csv(
        tmp_path,
        [
            _row(
                "A",
                20250101,
                0,
                temperature=20,
                humidity=80,
                precipitation=1,
            ),
            _row(
                "A",
                20250101,
                10000,
                temperature=24,
                humidity=70,
                precipitation=3,
            ),
        ],
    )
    dataset = load_historical_weather_csv(path, station_id="A")

    midpoint = dataset.record_at(datetime(2025, 1, 1, 5, 30, tzinfo=UTC))
    next_hour = dataset.record_at(datetime(2025, 1, 1, 6, tzinfo=UTC))

    assert midpoint.temperature_c == 22
    assert midpoint.relative_humidity_pct == 75
    assert midpoint.precipitation_mm_h == 1
    assert next_hour.precipitation_mm_h == 3
    assert midpoint.temperature_provenance.source == "adapter_linear_interpolation"
    assert midpoint.temperature_provenance.upstream_sources == ("senamhi_observed",)
    assert midpoint.precipitation_provenance.source == "senamhi_observed"


def test_optional_diagnostics_do_not_remove_transport_coverage(tmp_path: Path) -> None:
    path = _write_weather_csv(
        tmp_path,
        [
            _row("A", 20250101, 0, humidity=""),
            _row("A", 20250101, 10000, humidity=""),
        ],
    )

    dataset = load_historical_weather_csv(path, station_id="A")

    assert len(dataset.records) == 2
    assert dataset.records[0].relative_humidity_pct is None
    assert dataset.unavailable_timestamps_utc == ()


def test_missing_values_remain_uncovered_gaps(tmp_path: Path) -> None:
    path = _write_weather_csv(
        tmp_path,
        [
            _row("A", 20250101, 0),
            _row("A", 20250101, 10000, precipitation=""),
            _row("A", 20250101, 20000),
        ],
    )
    dataset = load_historical_weather_csv(
        path,
        station_id="A",
        config=WeatherCsvConfig(maximum_gap=timedelta(hours=3)),
    )
    missing_utc = datetime(2025, 1, 1, 6, tzinfo=UTC)

    assert dataset.unavailable_timestamps_utc == (missing_utc,)
    assert dataset.contiguous_segments == (
        (
            datetime(2025, 1, 1, 5, tzinfo=UTC),
            datetime(2025, 1, 1, 5, tzinfo=UTC),
        ),
        (
            datetime(2025, 1, 1, 7, tzinfo=UTC),
            datetime(2025, 1, 1, 7, tzinfo=UTC),
        ),
    )
    with pytest.raises(WeatherCoverageError, match="uncovered interval"):
        dataset.record_at(missing_utc)
    with pytest.raises(WeatherCoverageError, match="not fully covered"):
        dataset.select_window(
            datetime(2025, 1, 1, 5, tzinfo=UTC),
            timedelta(hours=2),
        )


def test_missing_values_can_be_rejected_explicitly(tmp_path: Path) -> None:
    path = _write_weather_csv(
        tmp_path,
        [_row("A", 20250101, 0, precipitation="")],
    )
    config = WeatherCsvConfig(missing_value_policy=MissingValuePolicy.ERROR)

    with pytest.raises(ValueError, match="missing required value"):
        load_historical_weather_csv(path, station_id="A", config=config)


def test_ifs_precipitation_requires_an_explicit_fallback_policy(
    tmp_path: Path,
) -> None:
    path = _write_weather_csv(
        tmp_path,
        [
            _row(
                "A",
                20250101,
                0,
                precipitation="",
                ifs_precipitation=0.4,
            )
        ],
    )

    with pytest.raises(ValueError, match="no usable weather records"):
        load_historical_weather_csv(path, station_id="A")

    dataset = load_historical_weather_csv(
        path,
        station_id="A",
        config=WeatherCsvConfig(
            precipitation_source_policy=PrecipitationSourcePolicy.STATION_THEN_IFS
        ),
    )

    assert dataset.records[0].precipitation_mm_h == 0.4
    assert dataset.records[0].precipitation_provenance == WeatherValueProvenance(
        "ecmwf_ifs"
    )
    assert (
        dataset.precipitation_source_policy
        is PrecipitationSourcePolicy.STATION_THEN_IFS
    )
    assert dataset.resource_id == path.name
    assert dataset.resource_sha256 is not None


def test_lookup_outside_coverage_always_fails() -> None:
    dataset = _dataset(hours=3)
    before = datetime(2024, 12, 31, 23, tzinfo=UTC)
    after = datetime(2025, 1, 1, 3, tzinfo=UTC)

    with pytest.raises(WeatherCoverageError, match="outside coverage"):
        dataset.record_at(before)
    with pytest.raises(WeatherCoverageError, match="outside coverage"):
        dataset.forcing_at(after)


def test_explicit_window_selection_is_reproducible() -> None:
    dataset = _dataset(hours=8)
    start = dataset.coverage_start_utc + timedelta(hours=2)

    first = dataset.select_window(start, timedelta(hours=3))
    second = dataset.select_window(start, timedelta(hours=3))

    assert first.start_utc == second.start_utc
    assert first.end_utc == second.end_utc
    assert first.records == second.records
    assert first.duration_s == 10_800


def test_weather_window_adapts_elapsed_seconds_without_mixing_pm25_inputs() -> None:
    dataset = _dataset(hours=3)
    window = dataset.select_window(dataset.coverage_start_utc, timedelta(hours=2))
    grid = build_grid(box(0, 0, 200, 200), "EPSG:32718", 100)
    model = PM25Transport(
        grid,
        PM25PhysicsConfig(
            mixing_height_m=100,
            diffusivity_m2_s=0,
            dry_deposition_velocity_m_s=0,
            wet_scavenging_per_mm=0,
        ),
    )
    initial = model.initialize_uniform(
        AmbientBaseline("ambient", 5, "independent test baseline")
    )
    weather_instants: list[datetime] = []
    boundary_instants: list[datetime] = []

    def forcing_at(elapsed_s: float):
        weather_instants.append(window.utc_at_elapsed_s(elapsed_s))
        return window.forcing_at_elapsed_s(elapsed_s)

    def boundary_at(elapsed_s: float) -> RegionalBoundaryForcing:
        boundary_instants.append(window.utc_at_elapsed_s(elapsed_s))
        return RegionalBoundaryForcing("boundary", 5, "independent test boundary")

    frames = tuple(
        iter_simulation(
            model,
            initial,
            duration_s=window.duration_s,
            frame_duration_s=window.duration_s,
            forcing_at=forcing_at,
            regional_boundary_at=boundary_at,
            boundary_mode=BoundaryMode.REGIONAL_INFLOW,
            forcing_change_times_s=window.forcing_change_times_s,
        )
    )

    assert [frame.state.simulation_time_s for frame in frames] == [
        0,
        3_600,
        7_200,
    ]
    assert weather_instants == boundary_instants
    assert all(frame.forcing.provider_id == "weather:A" for frame in frames)
    assert all(
        frame.regional_boundary is not None
        and frame.regional_boundary.provider_id == "boundary"
        for frame in frames
    )
    assert not hasattr(frames[0].forcing, "concentration_ug_m3")
    assert np.all(frames[-1].state.concentration_ug_m3[grid.active_mask] >= 0)


def _write_weather_csv(
    directory: Path,
    rows: list[dict[str, object]],
    *,
    name: str = "weather.csv",
) -> Path:
    path = directory / name
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(
    station: str,
    date: int,
    hour: int,
    *,
    temperature: object = 20,
    humidity: object = 80,
    precipitation: object = 0,
    direction: object = 270,
    speed: object = 1,
    temperature_source: str = "senamhi_observed",
    humidity_source: str = "senamhi_observed",
    precipitation_source: str = "senamhi_observed",
    direction_source: str = "senamhi_observed",
    speed_source: str = "senamhi_observed",
    ifs_precipitation: object = "",
) -> dict[str, object]:
    if temperature == "":
        temperature_source = "missing_explicit"
    if humidity == "":
        humidity_source = "missing_explicit"
    if precipitation == "":
        precipitation_source = "missing_explicit"
    if direction == "":
        direction_source = "missing_explicit"
    if speed == "":
        speed_source = "missing_explicit"
    return {
        "ESTACION": station,
        "FECHA": date,
        "HORA": hour,
        "LONGITUD": -77,
        "LATITUD": -12,
        "ALTITUD": 100,
        "TEMP": temperature,
        "HR": humidity,
        "PP": precipitation,
        "DIR_VIENTO": direction,
        "VEL_VIENTO": speed,
        "RED": "EMA",
        "DISTRITO": "LIMA",
        "TEMP_SOURCE": temperature_source,
        "HR_SOURCE": humidity_source,
        "PP_SOURCE": precipitation_source,
        "DIR_VIENTO_SOURCE": direction_source,
        "VEL_VIENTO_SOURCE": speed_source,
        "PP_IFS": ifs_precipitation,
    }


def _dataset(*, hours: int) -> HistoricalWeatherDataset:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    fixture_provenance = WeatherValueProvenance("unit_test_fixture")
    records = tuple(
        WeatherRecord(
            station_id="A",
            timestamp_utc=start + timedelta(hours=index),
            temperature_c=20 + index,
            relative_humidity_pct=80 - index,
            eastward_wind_m_s=0,
            northward_wind_m_s=0,
            precipitation_mm_h=float(index % 2),
            temperature_provenance=fixture_provenance,
            relative_humidity_provenance=fixture_provenance,
            precipitation_provenance=fixture_provenance,
            wind_direction_provenance=fixture_provenance,
            wind_speed_provenance=fixture_provenance,
        )
        for index in range(hours)
    )
    return HistoricalWeatherDataset(
        station_id="A",
        provider_id="weather:A",
        provenance="test fixture",
        records=records,
    )
