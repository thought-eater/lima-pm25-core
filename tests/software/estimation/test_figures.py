import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
from matplotlib.figure import Figure
from shapely.geometry import box

from lima_pm25.domain import build_grid
from lima_pm25.estimation.evidence import export_replay_figures
from lima_pm25.estimation.figures import (
    map_coverage,
    render_concentration_series,
    used_observations,
)
from lima_pm25.estimation.reconstruction import ReconstructionConfig
from lima_pm25.simulation import MeasurementQuality, MeasurementRecord, SensorType


def inputs():
    grid = build_grid(box(1000, 2000, 1200, 2200), "EPSG:32718", 100)
    config = ReconstructionConfig(
        prior_mean_ug_m3=20,
        prior_stddev_ug_m3=10,
        x_length_scale_m=1000,
        y_length_scale_m=1000,
        temporal_length_scale_s=3600,
    )
    t = datetime(2026, 1, 1, tzinfo=UTC)
    record = MeasurementRecord(
        sensor_id="fixed",
        sensor_type=SensorType.FIXED,
        timestamp_utc=t,
        simulation_time_s=0,
        x_m=1025,
        y_m=2025,
        crs=grid.crs,
        value_ug_m3=20,
        quality_status=MeasurementQuality.ACCEPTED,
        noise_stddev_ug_m3=0,
        position_source_id="fixed",
        provenance="test",
    )
    return grid, config, t, record


def test_overlays_use_only_eligible_history():
    grid, config, t, r = inputs()
    old = replace(r, sensor_id="old", timestamp_utc=t - timedelta(hours=7))
    future = replace(r, sensor_id="future", timestamp_utc=t + timedelta(seconds=1))
    clipped = replace(r, sensor_id="clipped", quality_status=MeasurementQuality.CLIPPED)
    groups = used_observations(
        grid, config, [r, old, future, clipped], [t], timedelta(hours=6)
    )
    assert groups == [(r,)]


def test_figure_error_mask_coordinates_and_shared_limits(tmp_path, monkeypatch):
    grid, _config, t, r = inputs()
    grid = replace(grid, active_mask=np.array([[True, True], [True, False]]))
    truth = np.array([[[1.0, 2.0], [3.0, 9999.0]], [[5.0, 6.0], [7.0, 9999.0]]])
    mean = truth + 2
    seen = []

    def inspect(fig, path, **kwargs):
        images = [ax.images[0] for ax in fig.axes if ax.images]
        seen.append(images[0].get_clim())
        np.testing.assert_allclose(images[0].get_extent(), (1.0, 1.2, 2.0, 2.2))
        assert images[0].get_clim() == images[1].get_clim() == (1.0, 9.0)
        assert images[2].get_array().mask[1, 1]
        assert np.all(images[2].get_array().compressed() == 2)

    monkeypatch.setattr(Figure, "savefig", inspect)
    render_concentration_series(
        grid, [t, t + timedelta(hours=1)], truth, mean, [(r,), (r,)], (0, 1), tmp_path
    )
    assert len(seen) == 2
    stats = map_coverage(mean, np.ones_like(mean), grid.active_mask)
    assert stats["cell_time_count"] == 6


def test_replay_counts_unique_actual_inputs_and_preserves_maps(tmp_path):
    grid, config, t, r = inputs()
    mobile = replace(r, sensor_id="mobile", sensor_type=SensorType.MOBILE)
    mean = np.ones((2, 2, 2)) * 20
    maps = tmp_path / "posterior_maps.npz"
    np.savez(
        maps,
        timestamps_utc=[t.isoformat(), (t + timedelta(hours=1)).isoformat()],
        active_mask=grid.active_mask,
        mean_ug_m3=mean,
        stddev_ug_m3=np.ones_like(mean),
    )
    before = maps.read_bytes()
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "window_start_utc": t.isoformat(),
                "window_end_utc": (t + timedelta(hours=1)).isoformat(),
                "fixed_observed_counts": {"CERES": 0},
            }
        )
    )
    export_replay_figures(
        tmp_path,
        grid=grid,
        config=config,
        measurements=[r, mobile],
        history_duration=timedelta(hours=6),
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["used_measurements"]["window_inclusive_end"] == {
        "fixed": 1,
        "mobile": 1,
    }
    assert metadata["fixed_observed_counts"]["CERES"] == 0
    assert metadata["map_validation"]["finite_concentration_fraction"] == 1
    assert maps.read_bytes() == before
    assert (tmp_path / "figures" / "replay-maps.png").is_file()
