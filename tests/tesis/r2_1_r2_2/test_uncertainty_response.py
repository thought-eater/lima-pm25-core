"""Respuesta espacial y temporal del GP en casos controlados de R2.2."""

from datetime import UTC, datetime, timedelta

import pytest

from lima_pm25.estimation import ReconstructionRequest
from tests.reconstruction_cases import _grid, _reconstructor, _record


def test_zero_noise_observation_has_near_zero_uncertainty_then_ages() -> None:
    grid = _grid()
    observed_at = datetime(2026, 9, 14, tzinfo=UTC)
    observation = _record("sensor", observed_at, value=35)
    reconstructor = _reconstructor(temporal_length_scale_s=3600)

    current = reconstructor.reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=observed_at,
            measurements=(observation,),
        )
    )
    one_scale_later = reconstructor.reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=observed_at + timedelta(hours=1),
            measurements=(observation,),
        )
    )
    three_scales_later = reconstructor.reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=observed_at + timedelta(hours=3),
            measurements=(observation,),
        )
    )

    assert current.standard_deviation_ug_m3[0, 0] < 1e-3
    assert current.mean_ug_m3[0, 0] == pytest.approx(35, abs=1e-8)
    assert (
        current.standard_deviation_ug_m3[0, 0]
        < one_scale_later.standard_deviation_ug_m3[0, 0]
        < three_scales_later.standard_deviation_ug_m3[0, 0]
        < 15
    )
    assert "WhiteKernel" not in current.diagnostics.kernel


def test_new_precise_observation_reduces_local_spatial_uncertainty() -> None:
    grid = _grid(rows=3, columns=3)
    timestamp = datetime(2026, 9, 14, tzinfo=UTC)
    first = _record("first", timestamp, x_m=50, y_m=50, value=20)
    additional = _record("additional", timestamp, x_m=250, y_m=250, value=40)
    reconstructor = _reconstructor(
        x_length_scale_m=120,
        y_length_scale_m=120,
    )

    before = reconstructor.reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=timestamp,
            measurements=(first,),
        )
    )
    after = reconstructor.reconstruct(
        ReconstructionRequest(
            grid=grid,
            reconstruction_timestamp_utc=timestamp,
            measurements=(first, additional),
        )
    )

    assert after.standard_deviation_ug_m3[2, 2] < 1e-3
    assert after.standard_deviation_ug_m3[2, 2] < before.standard_deviation_ug_m3[2, 2]
