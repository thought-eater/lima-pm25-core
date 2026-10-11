import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lima_pm25.estimation import (
    HistoricalPM25Dataset,
    HistoricalPM25Record,
    ParameterEvidenceClass,
    ScenarioParameterEvidence,
    build_lima_plausibility_profile,
    check_nominal_lima_conditions,
    write_lima_plausibility_profile,
)
from lima_pm25.simulation import (
    HistoricalWeatherDataset,
    WeatherRecord,
    WeatherValueProvenance,
)


def test_profile_uses_only_observed_pm25_and_retains_evidence_classes(
    tmp_path: Path,
) -> None:
    pm25 = _pm25_dataset()
    weather = _weather_dataset()
    parameters = (
        ScenarioParameterEvidence(
            "mixing_height_m",
            200,
            "m",
            ParameterEvidenceClass.SYNTHETIC,
            "provisional development value",
        ),
    )

    profile = build_lima_plausibility_profile(
        pm25=pm25,
        weather=weather,
        parameters=parameters,
    )

    assert profile.pm25_ug_m3.count == 10
    assert profile.pm25_ug_m3.maximum == 19
    assert profile.pm25_observed_source_counts == (("senamhi_observed", 10),)
    assert profile.precipitation_mm_h.maximum == 0.9
    assert profile.parameters == parameters
    destination = tmp_path / "plausibility.json"
    write_lima_plausibility_profile(profile, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["nominal_envelopes"]["pm25_ug_m3"] == "p01_to_p99"
    assert payload["nominal_envelopes"]["precipitation_mm_h"] == ("observed_min_to_max")
    assert payload["parameters"][0]["evidence_class"] == "synthetic"
    with pytest.raises(FileExistsError, match="already exists"):
        write_lima_plausibility_profile(profile, destination)


def test_nominal_check_rejects_out_of_envelope_rain_and_pm25() -> None:
    profile = build_lima_plausibility_profile(
        pm25=_pm25_dataset(),
        weather=_weather_dataset(),
    )

    accepted = check_nominal_lima_conditions(
        profile,
        pm25_ug_m3=(12, 15),
        wind_speed_m_s=(1.1, 1.5),
        precipitation_mm_h=(0.2, 0.5),
    )
    rejected = check_nominal_lima_conditions(
        profile,
        pm25_ug_m3=(1_000,),
        precipitation_mm_h=(100,),
    )

    assert accepted.nominal
    assert accepted.violations == ()
    assert not rejected.nominal
    assert rejected.violations == (
        "pm25_ug_m3:above_p99",
        "precipitation_mm_h:above_observed_maximum",
    )


def _pm25_dataset() -> HistoricalPM25Dataset:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    observed = tuple(
        HistoricalPM25Record(
            station_id="A",
            timestamp_utc=start + timedelta(hours=hour),
            longitude_deg=-77,
            latitude_deg=-12,
            altitude_m=None,
            value_ug_m3=10 + hour,
            province="LIMA",
            district="TEST",
            source="senamhi_observed",
        )
        for hour in range(10)
    )
    imputed = HistoricalPM25Record(
        station_id="A",
        timestamp_utc=start + timedelta(hours=10),
        longitude_deg=-77,
        latitude_deg=-12,
        altitude_m=None,
        value_ug_m3=10_000,
        province="LIMA",
        district="TEST",
        source="imputed_linear",
    )
    return HistoricalPM25Dataset(
        records=(*observed, imputed),
        resource_id="pm25.csv",
        resource_sha256="1" * 64,
        source_timezone="UTC",
    )


def _weather_dataset() -> HistoricalWeatherDataset:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    provenance = WeatherValueProvenance("senamhi_observed")
    records = tuple(
        WeatherRecord(
            station_id="WEATHER",
            timestamp_utc=start + timedelta(hours=hour),
            temperature_c=18 + hour,
            relative_humidity_pct=90 - hour,
            eastward_wind_m_s=1 + hour / 10,
            northward_wind_m_s=0,
            precipitation_mm_h=hour / 10,
            temperature_provenance=provenance,
            relative_humidity_provenance=provenance,
            precipitation_provenance=provenance,
            wind_direction_provenance=provenance,
            wind_speed_provenance=provenance,
        )
        for hour in range(10)
    )
    return HistoricalWeatherDataset(
        station_id="WEATHER",
        provider_id="weather:test",
        provenance="unit test",
        records=records,
        resource_id="weather.csv",
        resource_sha256="2" * 64,
    )
