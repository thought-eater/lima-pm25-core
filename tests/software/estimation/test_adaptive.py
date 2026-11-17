from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from shapely.geometry import box

from lima_pm25.domain import build_grid
from lima_pm25.estimation import (
    AdaptiveMapUpdater,
    GaussianProcessFieldReconstructor,
    GPFitConfig,
    GPParameters,
    ReconstructionConfig,
    ReconstructionRequest,
    fit_adaptive_posterior,
)
from lima_pm25.simulation import MeasurementBatch, MeasurementRecord, SensorType


def inputs():
    grid = build_grid(box(0, 0, 800, 800), "EPSG:32718", 200)
    base = ReconstructionConfig(
        prior_mean_ug_m3=10,
        prior_stddev_ug_m3=5,
        x_length_scale_m=300,
        y_length_scale_m=300,
        temporal_length_scale_s=3600,
    )
    origin = datetime(2025, 1, 1, tzinfo=UTC)
    records = tuple(
        MeasurementRecord(
            sensor_id=f"S{i}",
            sensor_type=SensorType.MOBILE,
            timestamp_utc=origin + timedelta(seconds=t),
            simulation_time_s=t,
            x_m=x,
            y_m=y,
            crs=grid.crs,
            value_ug_m3=10 + np.sin(i + t / 600),
            quality_status="accepted",
            noise_stddev_ug_m3=1.0,
            position_source_id="test",
            provenance="Lecturas artificiales para comprobar ajuste",
        )
        for t in (0, 300, 600, 900)
        for i, (x, y) in enumerate(((100.0, 100.0), (500.0, 100.0), (300.0, 500.0)))
    )
    request = ReconstructionRequest(
        grid=grid,
        reconstruction_timestamp_utc=origin + timedelta(seconds=900),
        measurements=records,
    )
    return grid, base, request


def test_fit_ties_space_respects_bounds_and_does_not_reduce_likelihood():
    grid, base, request = inputs()
    settings = GPFitConfig(max_iterations=5)
    result = fit_adaptive_posterior(request, base, settings, GPParameters.initial(base))
    assert result.fit.attempted and result.fit.iterations <= 5
    assert result.fit.log_likelihood_after >= result.fit.log_likelihood_before - 1e-8
    assert 200 <= result.parameters.spatial_scale_m <= 4000
    assert 300 <= result.parameters.temporal_scale_s <= 21600
    assert 1.25 <= result.parameters.amplitude_stddev_ug_m3 <= 10
    reference = GaussianProcessFieldReconstructor(
        base, kernel=result.parameters.kernel()
    ).reconstruct(request)
    np.testing.assert_allclose(
        result.posterior.mean_ug_m3[grid.active_mask],
        reference.mean_ug_m3[grid.active_mask],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.posterior.standard_deviation_ug_m3[grid.active_mask],
        reference.standard_deviation_ug_m3[grid.active_mask],
        atol=1e-10,
    )


def test_future_reading_cannot_change_fit_or_posterior():
    grid, base, request = inputs()
    settings = GPFitConfig(max_iterations=3)
    initial = GPParameters.initial(base)
    first = fit_adaptive_posterior(request, base, settings, initial)
    future = replace(
        request.measurements[-1],
        timestamp_utc=request.reconstruction_timestamp_utc + timedelta(seconds=300),
        simulation_time_s=1200,
        value_ug_m3=1000,
    )
    second = fit_adaptive_posterior(
        replace(request, measurements=(*request.measurements, future)),
        base,
        settings,
        initial,
    )
    assert first.parameters == second.parameters
    np.testing.assert_array_equal(
        first.posterior.mean_ug_m3[grid.active_mask],
        second.posterior.mean_ug_m3[grid.active_mask],
    )
    assert second.posterior.observations.future_record_count == 1


def test_sparse_data_keeps_initial_parameters_and_predict_does_not_optimize():
    grid, base, request = inputs()
    updater = AdaptiveMapUpdater(grid, base, timedelta(hours=6), GPFitConfig())
    group = tuple(request.measurements[:3])
    before, after = updater.update(MeasurementBatch(0.0, group[0].timestamp_utc, group))
    assert before is not None
    assert not after.fit.attempted
    initial = updater.parameters
    old = updater.predict(group[0].timestamp_utc)
    later = updater.predict(group[0].timestamp_utc + timedelta(seconds=300))
    assert updater.parameters == initial and updater.timestamp == group[0].timestamp_utc
    assert np.all(
        later.standard_deviation_ug_m3[grid.active_mask]
        >= old.standard_deviation_ug_m3[grid.active_mask] - 1e-10
    )
    with pytest.raises(ValueError, match="cronológicamente"):
        updater.update(MeasurementBatch(0.0, group[0].timestamp_utc, group))


def test_reference_and_full_fit_start_from_identical_parameters(monkeypatch):
    import lima_pm25.estimation.adaptive as module

    grid, base, request = inputs()
    updater = AdaptiveMapUpdater(
        grid, base, timedelta(hours=6), GPFitConfig(max_iterations=2)
    )
    calls = []
    original = module.fit_adaptive_posterior

    def observed(req, config, settings, initial, **kwargs):
        calls.append((initial, req.measurements))
        return original(req, config, settings, initial, **kwargs)

    monkeypatch.setattr(module, "fit_adaptive_posterior", observed)
    for t in (0, 300, 600, 900):
        group = tuple(r for r in request.measurements if r.simulation_time_s == t)
        group = (replace(group[0], sensor_type=SensorType.FIXED), *group[1:])
        calls.clear()
        updater.update(MeasurementBatch(t, group[0].timestamp_utc, group))
        assert len(calls) == 2 and calls[0][0] == calls[1][0]
        assert len(calls[1][1]) - len(calls[0][1]) == 2
        assert sum(r.sensor_type is SensorType.MOBILE for r in calls[0][1]) == 2 * (
            t // 300
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spatial_bounds_m": [0, 4000]},
        {"temporal_bounds_s": [900, 300]},
        {"max_iterations": 0},
        {"min_measurements": True},
        {"min_time_span_s": -1},
    ],
)
def test_invalid_fit_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GPFitConfig(**kwargs)


def test_numerical_optimizer_failure_preserves_initial_posterior(monkeypatch):
    import lima_pm25.estimation.adaptive as module

    grid, base, request = inputs()
    initial = GPParameters.initial(base)

    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("fallo numérico artificial")

    monkeypatch.setattr(module, "minimize", fail)
    result = fit_adaptive_posterior(request, base, GPFitConfig(), initial)
    reference = GaussianProcessFieldReconstructor(base).reconstruct(request)
    assert result.parameters == initial
    assert result.fit.attempted and not result.fit.converged and not result.fit.improved
    np.testing.assert_array_equal(
        result.posterior.mean_ug_m3[grid.active_mask],
        reference.mean_ug_m3[grid.active_mask],
    )
