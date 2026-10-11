"""Correspondencia de la selección conservada con el perfil operacional."""

import json
from pathlib import Path

from lima_pm25.estimation import load_change10_evaluation_profile


def test_retained_spatial_selection_matches_operational_profile() -> None:
    profile = load_change10_evaluation_profile("config/estimation_evaluation.json")
    calibration = json.loads(
        Path(profile.calibration_artifact_path).read_text(encoding="utf-8")
    )

    assert profile.historical_station_best_spatial_m == 4_000
    assert calibration["historical_station_best_spatial_m"] == 4_000
    assert profile.synthetic_selected_spatial_m == 2_000
    assert calibration["synthetic_selected_spatial_m"] == 2_000
    assert profile.reconstruction_config.x_length_scale_m == 2_000
    assert profile.reconstruction_config.y_length_scale_m == 2_000
    assert (
        min(
            calibration["spatial_candidate_rmse_ug_m3"],
            key=calibration["spatial_candidate_rmse_ug_m3"].get,
        )
        == "2000"
    )
    assert calibration["profile_sha256"] == profile.source_sha256
