from dataclasses import fields
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from pyproj import CRS

from lima_pm25.domain import (
    SENAMHI_PM25_PROVENANCE,
    FixedStationCatalog,
    FixedStationSite,
    GridDefinition,
)
from lima_pm25.simulation import (
    MeasurementHistory,
    MeasurementQuality,
    MeasurementRecord,
    SamplingSupportError,
    SensingService,
    SensorConfig,
    SensorType,
    create_lima_fixed_sensor_configs,
    sample_cell_concentration,
)


def test_cell_sampling_returns_the_containing_finite_volume_value() -> None:
    grid = _grid()
    concentration = np.array([[10.0, 20.0], [30.0, 40.0]])

    assert sample_cell_concentration(grid, concentration, 50, 50) == 10
    assert sample_cell_concentration(grid, concentration, 99.999, 99.999) == 10
    assert sample_cell_concentration(grid, concentration, 100, 100) == 40


def test_cell_sampling_rejects_an_inactive_cell_without_using_neighbors() -> None:
    grid = _grid(active_mask=np.array([[True, False], [True, True]]))
    concentration = np.array([[10.0, 0.0], [30.0, 40.0]])

    with pytest.raises(SamplingSupportError, match="inactive grid cell"):
        sample_cell_concentration(grid, concentration, 150, 50)


def test_cell_sampling_uses_the_half_open_cell_face_convention() -> None:
    grid = _grid()
    concentration = np.array([[10.0, 20.0], [30.0, 40.0]])

    assert sample_cell_concentration(grid, concentration, 100 - 1e-9, 50) == 10
    assert sample_cell_concentration(grid, concentration, 100, 50) == 20


def test_cell_sampling_rejects_positions_outside_the_grid() -> None:
    grid = _grid()

    with pytest.raises(SamplingSupportError, match="outside the grid"):
        sample_cell_concentration(grid, np.zeros((2, 2)), -1, 50)


def test_baseline_fixed_sensors_are_hourly_from_zero_with_errors_disabled() -> None:
    catalog = _station_catalog()

    configs = create_lima_fixed_sensor_configs(catalog)

    assert len(configs) == 1
    config = configs[0]
    assert config.sensor_type is SensorType.FIXED
    assert config.sampling_interval_s == 3600
    assert config.first_due_time_s == 0
    assert config.bias_ug_m3 == 0
    assert config.noise_stddev_ug_m3 == 0
    assert config.dropout_probability == 0
    assert config.maximum_value_ug_m3 is None
    assert "minimum_value_ug_m3" not in config.__slots__


def test_fixed_sensor_samples_every_crossed_due_time_exactly_once() -> None:
    grid = _grid()
    catalog = _station_catalog()
    service = SensingService(
        grid,
        create_lima_fixed_sensor_configs(catalog),
        station_catalog=catalog,
        rng=np.random.default_rng(1),
    )
    start = datetime(2026, 9, 14, 15, tzinfo=UTC)

    records = service.sample_due(
        7200,
        timestamp_at=lambda elapsed: start + timedelta(seconds=elapsed),
        concentration_at=lambda elapsed: np.full(grid.shape, 10 + elapsed / 3600),
    )

    assert [record.simulation_time_s for record in records] == [0, 3600, 7200]
    assert [record.value_ug_m3 for record in records] == [10, 11, 12]
    assert all(
        record.quality_status is MeasurementQuality.ACCEPTED for record in records
    )
    assert all(
        record.provenance == "baseline synthetic sensing at SENAMHI PM2.5 sites"
        for record in records
    )
    assert all(record.position_source_id == "STATION" for record in records)
    assert (
        service.sample_due(
            7200,
            timestamp_at=lambda elapsed: start + timedelta(seconds=elapsed),
            concentration_at=lambda _: np.full(grid.shape, 99.0),
        )
        == ()
    )
    assert service.state_for("STATION").next_due_time_s == 10_800


def test_mobile_sensor_uses_explicit_first_due_and_provider_positions() -> None:
    grid = _grid()
    provider = _MobileProvider(grid.crs)
    config = SensorConfig(
        sensor_id="MOBILE",
        sensor_type=SensorType.MOBILE,
        sampling_interval_s=300,
        first_due_time_s=120,
        mobile_position_provider_id=provider.provider_id,
        provenance="synthetic mobile test",
    )
    service = SensingService(
        grid,
        (config,),
        mobile_position_providers=(provider,),
        rng=np.random.default_rng(2),
    )
    start = datetime(2026, 9, 14, 15, tzinfo=UTC)

    records = service.sample_due(
        720,
        timestamp_at=lambda elapsed: start + timedelta(seconds=elapsed),
        concentration_at=lambda _: np.array([[10.0, 20.0], [30.0, 40.0]]),
    )

    assert [record.simulation_time_s for record in records] == [120, 420, 720]
    assert [record.x_m for record in records] == [50, 100, 150]
    assert [record.value_ug_m3 for record in records] == [10, 20, 20]
    assert [record.position_source_id for record in records] == [
        "trajectory",
        "trajectory",
        "trajectory",
    ]
    assert all(record.position_source_id == provider.provider_id for record in records)
    assert config.__slots__.count("mobile_position_provider_id") == 1
    assert "x_m" not in config.__slots__
    assert "y_m" not in config.__slots__


def test_mobile_sensor_records_failed_attempt_outside_the_active_grid() -> None:
    grid = _grid()
    provider = _RecordedMobileProvider(
        grid.crs,
        {
            0: (50, 50),
            60: (250, 50),
            120: (150, 50),
        },
    )
    config = SensorConfig(
        sensor_id="MOBILE",
        sensor_type=SensorType.MOBILE,
        sampling_interval_s=60,
        first_due_time_s=0,
        mobile_position_provider_id=provider.provider_id,
        provenance="outside-grid mobile test",
    )
    service = SensingService(
        grid,
        (config,),
        mobile_position_providers=(provider,),
        rng=np.random.default_rng(2),
    )
    start = datetime(2026, 9, 14, 15, tzinfo=UTC)

    records = service.sample_due(
        120,
        timestamp_at=lambda elapsed: start + timedelta(seconds=elapsed),
        concentration_at=lambda _: np.array([[10.0, 20.0], [30.0, 40.0]]),
    )

    assert [record.simulation_time_s for record in records] == [0, 60, 120]
    assert [record.value_ug_m3 for record in records] == [10, None, 20]
    assert records[1].x_m == 250
    assert records[1].y_m == 50
    assert records[1].quality_status is MeasurementQuality.OUTSIDE_ACTIVE_GRID
    assert records[1].position_source_id == provider.provider_id
    assert not records[1].is_accepted
    assert provider.requested_times == [0, 60, 120]
    assert service.state_for("MOBILE").attempts_completed == 3
    assert service.state_for("MOBILE").next_due_time_s == 180
    assert service.history.records == records
    assert service.history.accepted_records == (records[0], records[2])


def test_bias_saturation_and_dropout_are_optional_and_explicit() -> None:
    grid = _grid()
    catalog = _station_catalog()
    accepted = SensorConfig(
        sensor_id="ACCEPTED",
        sensor_type=SensorType.FIXED,
        sampling_interval_s=60,
        first_due_time_s=0,
        station_site_id="STATION",
        provenance="configured error test",
        bias_ug_m3=-20,
    )
    dropped = SensorConfig(
        sensor_id="DROPPED",
        sensor_type=SensorType.FIXED,
        sampling_interval_s=60,
        first_due_time_s=0,
        station_site_id="STATION",
        provenance="configured dropout test",
        dropout_probability=1,
    )
    saturated = SensorConfig(
        sensor_id="SATURATED",
        sensor_type=SensorType.FIXED,
        sampling_interval_s=60,
        first_due_time_s=0,
        station_site_id="STATION",
        provenance="configured saturation test",
        maximum_value_ug_m3=5,
    )
    service = SensingService(
        grid,
        (accepted, dropped, saturated),
        station_catalog=catalog,
        rng=np.random.default_rng(3),
    )

    records = service.sample_due(
        0,
        timestamp_at=lambda _: datetime(2026, 9, 14, tzinfo=UTC),
        concentration_at=lambda _: np.full(grid.shape, 10.0),
    )

    assert records[0].sensor_id == "ACCEPTED"
    assert records[0].value_ug_m3 == 0
    assert records[0].quality_status is MeasurementQuality.CLIPPED
    assert records[1].sensor_id == "DROPPED"
    assert records[1].value_ug_m3 is None
    assert records[1].quality_status is MeasurementQuality.DROPPED
    assert records[2].sensor_id == "SATURATED"
    assert records[2].value_ug_m3 == 5
    assert records[2].quality_status is MeasurementQuality.CLIPPED
    assert service.history.accepted_records == (records[0], records[2])
    assert service.state_for("DROPPED").next_due_time_s == 60
    assert "latest_measurement" not in service.state_for("DROPPED").__slots__


def test_seeded_noise_and_dropout_sequence_is_reproducible() -> None:
    config = SensorConfig(
        sensor_id="STOCHASTIC",
        sensor_type=SensorType.FIXED,
        sampling_interval_s=1,
        first_due_time_s=0,
        station_site_id="STATION",
        provenance="seeded test",
        bias_ug_m3=1,
        noise_stddev_ug_m3=2,
        dropout_probability=0.4,
    )

    first = _sample_sequence(config, seed=44)
    second = _sample_sequence(config, seed=44)

    assert first == second
    assert any(record.quality_status is MeasurementQuality.DROPPED for record in first)
    assert any(record.is_accepted for record in first)


def test_measurement_history_orders_ties_and_rejects_duplicates() -> None:
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)
    history = MeasurementHistory()
    records = (
        _record("sensor-b", timestamp),
        _record("sensor-a", timestamp),
        _record("sensor-c", timestamp + timedelta(hours=1)),
    )

    history.extend(records)

    assert [record.sensor_id for record in history.records] == [
        "sensor-a",
        "sensor-b",
        "sensor-c",
    ]
    with pytest.raises(ValueError, match="Duplicate measurement attempt"):
        history.extend((_record("sensor-a", timestamp),))


def test_public_measurement_record_does_not_expose_hidden_truth() -> None:
    public_fields = {entry.name for entry in fields(MeasurementRecord)}

    assert "mobility_policy_id" not in public_fields
    assert "sampled_truth_ug_m3" not in public_fields
    assert "hidden_field" not in public_fields
    assert "pre_bound_value_ug_m3" not in public_fields
    from lima_pm25 import simulation

    assert not hasattr(simulation, "_MeasurementDiagnostic")


def test_sensor_configuration_validates_position_ownership_and_unique_ids() -> None:
    with pytest.raises(ValueError, match="cannot reference a mobile provider"):
        SensorConfig(
            sensor_id="INVALID",
            sensor_type=SensorType.FIXED,
            sampling_interval_s=60,
            first_due_time_s=0,
            station_site_id="STATION",
            mobile_position_provider_id="trajectory",
            provenance="invalid test",
        )

    config = SensorConfig(
        sensor_id="DUPLICATE",
        sensor_type=SensorType.FIXED,
        sampling_interval_s=60,
        first_due_time_s=0,
        station_site_id="STATION",
        provenance="duplicate test",
    )
    with pytest.raises(ValueError, match="Duplicate sensor_id"):
        SensingService(
            _grid(),
            (config, config),
            station_catalog=_station_catalog(),
            rng=np.random.default_rng(1),
        )


def _sample_sequence(config: SensorConfig, seed: int) -> tuple[MeasurementRecord, ...]:
    grid = _grid()
    service = SensingService(
        grid,
        (config,),
        station_catalog=_station_catalog(),
        rng=np.random.default_rng(seed),
    )
    return service.sample_due(
        20,
        timestamp_at=lambda elapsed: (
            datetime(2026, 9, 14, tzinfo=UTC) + timedelta(seconds=elapsed)
        ),
        concentration_at=lambda _: np.full(grid.shape, 10.0),
    )


def _grid(*, active_mask: np.ndarray | None = None) -> GridDefinition:
    mask = np.ones((2, 2), dtype=bool) if active_mask is None else active_mask
    return GridDefinition(
        crs="EPSG:32718",
        domain_bounds_m=np.array([0, 0, 200, 200]),
        grid_bounds_m=np.array([0, 0, 200, 200]),
        cell_size_m=100,
        x_centers_m=np.array([50, 150]),
        y_centers_m=np.array([50, 150]),
        active_mask=mask,
    )


def _station_catalog() -> FixedStationCatalog:
    return FixedStationCatalog(
        (
            FixedStationSite(
                site_id="STATION",
                longitude_deg=-77,
                latitude_deg=-12,
                altitude_m=None,
                x_m=50,
                y_m=50,
                crs="EPSG:32718",
                provenance=SENAMHI_PM25_PROVENANCE,
            ),
        )
    )


class _MobileProvider:
    provider_id = "trajectory"

    def __init__(self, crs: str) -> None:
        self.crs = CRS.from_user_input(crs).to_string()

    def position_at(
        self,
        sensor_id: str,
        simulation_time_s: float,
    ) -> tuple[float, float]:
        return 30 + simulation_time_s / 6, 50


class _RecordedMobileProvider:
    provider_id = "recorded-provider"

    def __init__(
        self,
        crs: str,
        positions: dict[float, tuple[float, float]],
    ) -> None:
        self.crs = CRS.from_user_input(crs).to_string()
        self.positions = positions
        self.requested_times: list[float] = []

    def position_at(
        self,
        sensor_id: str,
        simulation_time_s: float,
    ) -> tuple[float, float]:
        self.requested_times.append(simulation_time_s)
        return self.positions[simulation_time_s]


def _record(sensor_id: str, timestamp: datetime) -> MeasurementRecord:
    return MeasurementRecord(
        sensor_id=sensor_id,
        sensor_type=SensorType.FIXED,
        timestamp_utc=timestamp,
        simulation_time_s=0,
        x_m=50,
        y_m=50,
        crs="EPSG:32718",
        value_ug_m3=10,
        quality_status=MeasurementQuality.ACCEPTED,
        noise_stddev_ug_m3=0,
        position_source_id="STATION",
        provenance="history test",
    )
