from dataclasses import fields
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from lima_pm25.domain import GridDefinition
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionConfig,
    ReconstructionRequest,
)
from lima_pm25.simulation import (
    MeasurementQuality,
)
from tests.reconstruction_cases import (
    _grid,
    _reconstruction_config,
    _reconstructor,
    _record,
)


def test_empty_history_returns_the_configured_masked_prior() -> None:
    grid = _grid(active_mask=np.array([[True, False], [True, True]]))
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)

    posterior = _reconstructor().reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=timestamp,
            measurements=(),
        )
    )

    assert np.all(posterior.mean_ug_m3[grid.active_mask] == 20)
    assert np.all(posterior.standard_deviation_ug_m3[grid.active_mask] == 15)
    assert np.isnan(posterior.mean_ug_m3[0, 1])
    assert np.isnan(posterior.standard_deviation_ug_m3[0, 1])
    assert np.array_equal(posterior.active_mask, grid.active_mask)
    assert not posterior.diagnostics.posterior_fitted
    assert posterior.diagnostics.log_marginal_likelihood is None
    assert posterior.diagnostics.numerical_jitter_ug_m3_squared == pytest.approx(
        15**2 * 1e-10
    )
    assert not posterior.mean_ug_m3.flags.writeable


def test_preparation_uses_the_supplied_observation_set_in_chronological_order() -> None:
    grid = _grid()
    current = datetime(2026, 9, 14, 12, tzinfo=UTC)
    records = tuple(
        _record(
            f"sensor-{index:02d}",
            current - timedelta(hours=24 - index),
            x_m=50 if index % 2 == 0 else 150,
            y_m=50,
            value=10 + index,
        )
        for index in reversed(range(25))
    )

    posterior = _reconstructor().reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=current,
            measurements=records,
        )
    )

    assert posterior.observations.provided_record_count == 25
    assert posterior.observations.used_record_count == 25
    assert [
        timestamp for _, timestamp in posterior.observations.used_measurement_keys
    ] == [current - timedelta(hours=24 - index) for index in range(25)]

    selected_subset = records[-5:]
    subset_posterior = _reconstructor(temporal_length_scale_s=1e6).reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=current,
            measurements=selected_subset,
        )
    )
    complete_posterior = _reconstructor(temporal_length_scale_s=1e6).reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=current,
            measurements=records,
        )
    )
    assert subset_posterior.observations.provided_record_count == 5
    assert subset_posterior.observations.used_record_count == 5
    assert not np.allclose(
        complete_posterior.mean_ug_m3,
        subset_posterior.mean_ug_m3,
    )


def test_dropped_and_future_records_are_not_available_to_the_fit() -> None:
    grid = _grid()
    current = datetime(2026, 9, 14, 12, tzinfo=UTC)
    records = (
        _record("used", current, value=12),
        _record("dropped", current, value=None),
        _record("future", current + timedelta(minutes=1), value=99),
    )

    posterior = _reconstructor().reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=current,
            measurements=records,
        )
    )

    assert posterior.observations.used_record_count == 1
    assert posterior.observations.dropped_record_count == 1
    assert posterior.observations.future_record_count == 1
    assert posterior.observations.used_measurement_keys == (("used", current),)


def test_numeric_clipped_measurement_is_excluded_by_default() -> None:
    grid = _grid()
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)
    clipped = _record(
        "clipped",
        timestamp,
        value=0,
        quality=MeasurementQuality.CLIPPED,
    )

    posterior = _reconstructor().reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=timestamp,
            measurements=(clipped,),
        )
    )

    assert posterior.observations.used_record_count == 0
    assert posterior.observations.dropped_record_count == 0
    assert posterior.observations.excluded_clipped_record_count == 1
    assert not posterior.diagnostics.posterior_fitted


def test_numeric_clipped_measurement_can_be_included_explicitly() -> None:
    grid = _grid()
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)
    clipped = _record(
        "clipped",
        timestamp,
        value=0,
        quality=MeasurementQuality.CLIPPED,
    )

    posterior = _reconstructor(include_clipped_observations=True).reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=timestamp,
            measurements=(clipped,),
        )
    )

    assert posterior.observations.used_record_count == 1
    assert posterior.observations.dropped_record_count == 0
    assert posterior.observations.excluded_clipped_record_count == 0
    assert posterior.diagnostics.posterior_fitted


@pytest.mark.parametrize(
    ("sensor_id", "crs", "x_m", "message"),
    [
        (
            "wrong-crs",
            "EPSG:3857",
            50,
            "different CRS",
        ),
        (
            "outside",
            "EPSG:32718",
            500,
            "active reconstruction cell",
        ),
    ],
)
def test_incompatible_accepted_measurements_fail_explicitly(
    sensor_id: str,
    crs: str,
    x_m: float,
    message: str,
) -> None:
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)
    record = _record(sensor_id, timestamp, crs=crs, x_m=x_m)
    with pytest.raises(ValueError, match=message):
        _reconstructor().reconstruct(
            ReconstructionRequest(
                grid=_grid(),
                reconstruction_timestamp_utc=timestamp,
                measurements=(record,),
            )
        )


def test_duplicate_measurement_keys_fail_instead_of_counting_evidence_twice() -> None:
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)
    record = _record("duplicate", timestamp, value=30)

    with pytest.raises(ValueError, match="Duplicate measurement"):
        _reconstructor().reconstruct(
            ReconstructionRequest(
                grid=_grid(),
                reconstruction_timestamp_utc=timestamp,
                measurements=(record, record),
            )
        )


def test_reconstruction_rejects_a_projected_crs_that_does_not_use_meters() -> None:
    grid = GridDefinition(
        crs="EPSG:2277",
        domain_bounds_m=np.array([0, 0, 200, 200]),
        grid_bounds_m=np.array([0, 0, 200, 200]),
        cell_size_m=100,
        x_centers_m=np.array([50, 150]),
        y_centers_m=np.array([50, 150]),
        active_mask=np.ones((2, 2), dtype=bool),
    )

    with pytest.raises(ValueError, match="axes must use meters"):
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=datetime(2026, 9, 14, tzinfo=UTC),
            measurements=(),
        )


def test_declared_sensor_noise_prevents_complete_confidence() -> None:
    grid = _grid()
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)
    exact = _record("exact", timestamp, value=35, noise_stddev=0)
    noisy = _record("noisy", timestamp, value=35, noise_stddev=5)

    exact_posterior = _reconstructor().reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=timestamp,
            measurements=(exact,),
        )
    )
    noisy_posterior = _reconstructor().reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=timestamp,
            measurements=(noisy,),
        )
    )

    assert (
        noisy_posterior.standard_deviation_ug_m3[0, 0]
        > exact_posterior.standard_deviation_ug_m3[0, 0]
    )
    assert noisy.noise_stddev_ug_m3 == 5


def test_relative_jitter_stabilizes_colocated_zero_noise_observations() -> None:
    grid = _grid()
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)
    colocated = (
        _record("first", timestamp, value=35, noise_stddev=0),
        _record("second", timestamp, value=35, noise_stddev=0),
    )

    posterior = _reconstructor().reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=timestamp,
            measurements=colocated,
        )
    )

    assert np.all(np.isfinite(posterior.mean_ug_m3))
    assert np.all(np.isfinite(posterior.standard_deviation_ug_m3))
    assert all(record.noise_stddev_ug_m3 == 0 for record in colocated)
    assert posterior.diagnostics.numerical_jitter_ug_m3_squared == pytest.approx(
        15**2 * 1e-10
    )


def test_fitted_posterior_preserves_the_inactive_mask() -> None:
    mask = np.array([[True, False], [True, True]])
    grid = _grid(active_mask=mask)
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)

    posterior = _reconstructor().reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=timestamp,
            measurements=(_record("sensor", timestamp, value=30),),
        )
    )

    assert posterior.diagnostics.posterior_fitted
    assert np.all(np.isfinite(posterior.mean_ug_m3[mask]))
    assert np.all(np.isfinite(posterior.standard_deviation_ug_m3[mask]))
    assert np.isnan(posterior.mean_ug_m3[0, 1])
    assert np.isnan(posterior.standard_deviation_ug_m3[0, 1])


def test_point_reconstruction_matches_grid_prediction_at_cell_centers() -> None:
    grid = _grid(rows=3, columns=3)
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)
    records = (
        _record("a", timestamp, x_m=50, y_m=50, value=10),
        _record("b", timestamp, x_m=250, y_m=250, value=40),
    )
    request = ReconstructionRequest(
        grid=grid,
        reconstruction_timestamp_utc=timestamp,
        measurements=records,
    )
    reconstructor = _reconstructor()

    field = reconstructor.reconstruct(request)
    points = reconstructor.reconstruct_points(
        request,
        ((50, 50), (150, 150), (250, 250)),
    )

    assert np.allclose(
        points.mean_ug_m3,
        (field.mean_ug_m3[0, 0], field.mean_ug_m3[1, 1], field.mean_ug_m3[2, 2]),
    )
    assert np.allclose(
        points.standard_deviation_ug_m3,
        (
            field.standard_deviation_ug_m3[0, 0],
            field.standard_deviation_ug_m3[1, 1],
            field.standard_deviation_ug_m3[2, 2],
        ),
    )
    assert not points.positions_m.flags.writeable


def test_empty_history_point_reconstruction_returns_prior() -> None:
    grid = _grid()
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)

    points = _reconstructor().reconstruct_points(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=timestamp,
            measurements=(),
        ),
        ((50, 50),),
    )

    assert points.mean_ug_m3.tolist() == [20]
    assert points.standard_deviation_ug_m3.tolist() == [15]
    assert not points.diagnostics.posterior_fitted


def test_sparse_reconstruction_improves_on_the_prior_mean() -> None:
    grid = _grid(rows=5, columns=5)
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)
    center_x, center_y = np.meshgrid(grid.x_centers_m, grid.y_centers_m)
    truth = 10 + 0.02 * center_x + 0.01 * center_y
    observed_cells = (
        (0, 0),
        (0, 2),
        (0, 4),
        (2, 0),
        (2, 2),
        (2, 4),
        (4, 0),
        (4, 2),
        (4, 4),
    )
    records = tuple(
        _record(
            f"sensor-{row}-{column}",
            timestamp,
            x_m=float(grid.x_centers_m[column]),
            y_m=float(grid.y_centers_m[row]),
            value=float(truth[row, column]),
        )
        for row, column in observed_cells
    )

    posterior = _reconstructor(
        prior_mean_ug_m3=10,
        prior_stddev_ug_m3=20,
        x_length_scale_m=250,
        y_length_scale_m=250,
    ).reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=timestamp,
            measurements=records,
        )
    )

    posterior_rmse = float(np.sqrt(np.mean((posterior.mean_ug_m3 - truth) ** 2)))
    prior_rmse = float(np.sqrt(np.mean((10 - truth) ** 2)))
    assert posterior_rmse < prior_rmse
    assert np.all(np.isfinite(posterior.mean_ug_m3))
    assert np.all(posterior.standard_deviation_ug_m3 >= 0)


def test_reconstruction_is_deterministic_and_chunk_independent() -> None:
    grid = _grid(rows=3, columns=3)
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)
    records = (
        _record("a", timestamp, x_m=50, y_m=50, value=10),
        _record("b", timestamp, x_m=250, y_m=250, value=40),
    )
    request = ReconstructionRequest(
        grid=grid,
        reconstruction_timestamp_utc=timestamp,
        measurements=records,
    )

    first = _reconstructor(prediction_chunk_size=1).reconstruct(request)
    repeated = _reconstructor(prediction_chunk_size=1).reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=timestamp,
            measurements=tuple(reversed(records)),
        )
    )
    differently_chunked = _reconstructor(prediction_chunk_size=8).reconstruct(request)

    assert np.array_equal(first.mean_ug_m3, repeated.mean_ug_m3)
    assert np.array_equal(
        first.standard_deviation_ug_m3,
        repeated.standard_deviation_ug_m3,
    )
    assert np.allclose(
        first.mean_ug_m3,
        differently_chunked.mean_ug_m3,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.allclose(
        first.standard_deviation_ug_m3,
        differently_chunked.standard_deviation_ug_m3,
        rtol=1e-10,
        atol=1e-9,
    )
    assert first.diagnostics.kernel == repeated.diagnostics.kernel


def test_reconstruction_contract_cannot_receive_hidden_truth() -> None:
    request_fields = {entry.name for entry in fields(ReconstructionRequest)}

    assert request_fields == {
        "grid",
        "reconstruction_timestamp_utc",
        "measurements",
    }
    assert "hidden_field" not in request_fields
    assert "sampled_truth_ug_m3" not in request_fields


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"prior_mean_ug_m3": -1}, "prior_mean_ug_m3"),
        ({"prior_stddev_ug_m3": 0}, "prior_stddev_ug_m3"),
        ({"x_length_scale_m": 0}, "x_length_scale_m"),
        ({"temporal_length_scale_s": float("inf")}, "temporal_length_scale_s"),
        ({"include_clipped_observations": "yes"}, "include_clipped_observations"),
    ],
)
def test_reconstruction_configuration_is_explicit_and_validated(
    override: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "prior_mean_ug_m3": 20,
        "prior_stddev_ug_m3": 15,
        "x_length_scale_m": 300,
        "y_length_scale_m": 300,
        "temporal_length_scale_s": 3600,
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        ReconstructionConfig(**values)  # ty: ignore[invalid-argument-type] -- deliberately invalid configuration


@pytest.mark.parametrize("invalid_chunk_size", (0, -1, True, 1.5))
def test_prediction_chunk_size_is_a_validated_execution_parameter(
    invalid_chunk_size: object,
) -> None:
    with pytest.raises(ValueError, match="prediction_chunk_size"):
        GaussianProcessFieldReconstructor(
            _reconstruction_config(),
            prediction_chunk_size=invalid_chunk_size,  # ty: ignore[invalid-argument-type] -- deliberately invalid size
        )
