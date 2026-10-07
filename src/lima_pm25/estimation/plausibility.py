from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from lima_pm25.simulation import HistoricalWeatherDataset

from .historical import HistoricalPM25Dataset


class ParameterEvidenceClass(StrEnum):
    OBSERVED = "observed"
    HISTORICALLY_CONSTRAINED = "historically_constrained"
    LITERATURE_CONSTRAINED = "literature_constrained"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    unit: str
    count: int
    minimum: float
    p01: float
    p05: float
    median: float
    p95: float
    p99: float
    maximum: float
    percentile_method: str = "linear"

    @property
    def nominal_lower(self) -> float:
        return self.p01

    @property
    def nominal_upper(self) -> float:
        return self.p99


@dataclass(frozen=True, slots=True)
class ScenarioParameterEvidence:
    name: str
    value: float
    unit: str
    evidence_class: ParameterEvidenceClass | str
    evidence: str

    def __post_init__(self) -> None:
        _require_text("name", self.name)
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")
        _require_text("unit", self.unit)
        object.__setattr__(
            self,
            "evidence_class",
            ParameterEvidenceClass(self.evidence_class),
        )
        _require_text("evidence", self.evidence)


@dataclass(frozen=True, slots=True)
class LimaPlausibilityProfile:
    pm25_resource_id: str
    pm25_resource_sha256: str
    weather_resource_id: str
    weather_resource_sha256: str
    weather_station_id: str
    pm25_observed_source_counts: tuple[tuple[str, int], ...]
    weather_source_counts: tuple[tuple[str, int], ...]
    pm25_ug_m3: DistributionSummary
    wind_speed_m_s: DistributionSummary
    precipitation_mm_h: DistributionSummary
    temperature_c: DistributionSummary | None
    relative_humidity_pct: DistributionSummary | None
    parameters: tuple[ScenarioParameterEvidence, ...] = ()

    def evidence_descriptor(self) -> dict[str, object]:
        return {
            "pm25_resource_id": self.pm25_resource_id,
            "pm25_resource_sha256": self.pm25_resource_sha256,
            "weather_resource_id": self.weather_resource_id,
            "weather_resource_sha256": self.weather_resource_sha256,
            "weather_station_id": self.weather_station_id,
            "pm25_observed_source_counts": dict(
                self.pm25_observed_source_counts
            ),
            "weather_source_counts": dict(self.weather_source_counts),
            "distributions": {
                "pm25_ug_m3": _summary_descriptor(self.pm25_ug_m3),
                "wind_speed_m_s": _summary_descriptor(self.wind_speed_m_s),
                "precipitation_mm_h": _summary_descriptor(
                    self.precipitation_mm_h
                ),
                "temperature_c": _optional_summary_descriptor(
                    self.temperature_c
                ),
                "relative_humidity_pct": _optional_summary_descriptor(
                    self.relative_humidity_pct
                ),
            },
            "nominal_envelopes": {
                "pm25_ug_m3": "p01_to_p99",
                "wind_speed_m_s": "p01_to_p99",
                "precipitation_mm_h": "observed_min_to_max",
                "temperature_c": "p01_to_p99",
                "relative_humidity_pct": "p01_to_p99",
            },
            "parameters": [
                {
                    "name": parameter.name,
                    "value": parameter.value,
                    "unit": parameter.unit,
                    "evidence_class": parameter.evidence_class.value,
                    "evidence": parameter.evidence,
                }
                for parameter in self.parameters
            ],
        }


@dataclass(frozen=True, slots=True)
class PlausibilityCheck:
    nominal: bool
    violations: tuple[str, ...]
    checked_counts: tuple[tuple[str, int], ...]


def build_lima_plausibility_profile(
    *,
    pm25: HistoricalPM25Dataset,
    weather: HistoricalWeatherDataset,
    parameters: Sequence[ScenarioParameterEvidence] = (),
) -> LimaPlausibilityProfile:
    if weather.resource_id is None or weather.resource_sha256 is None:
        raise ValueError("Plausibility profile requires identified weather content")
    pm25_values = [
        float(record.value_ug_m3)
        for record in pm25.observed_records
        if record.value_ug_m3 is not None
    ]
    if not pm25_values:
        raise ValueError("Plausibility profile requires observed PM2.5 values")
    wind_speeds = [
        math.hypot(record.eastward_wind_m_s, record.northward_wind_m_s)
        for record in weather.records
    ]
    precipitation = [record.precipitation_mm_h for record in weather.records]
    temperatures = [
        record.temperature_c
        for record in weather.records
        if record.temperature_c is not None
    ]
    humidity = [
        record.relative_humidity_pct
        for record in weather.records
        if record.relative_humidity_pct is not None
    ]
    pm25_sources = Counter(record.source for record in pm25.observed_records)
    weather_sources = Counter(
        provenance.source
        for record in weather.records
        for provenance in (
            record.temperature_provenance,
            record.relative_humidity_provenance,
            record.precipitation_provenance,
            record.wind_direction_provenance,
            record.wind_speed_provenance,
        )
    )
    parameter_values = tuple(parameters)
    names = [parameter.name for parameter in parameter_values]
    if len(set(names)) != len(names):
        raise ValueError("Scenario parameter evidence names must be unique")
    return LimaPlausibilityProfile(
        pm25_resource_id=pm25.resource_id,
        pm25_resource_sha256=pm25.resource_sha256,
        weather_resource_id=weather.resource_id,
        weather_resource_sha256=weather.resource_sha256,
        weather_station_id=weather.station_id,
        pm25_observed_source_counts=tuple(sorted(pm25_sources.items())),
        weather_source_counts=tuple(sorted(weather_sources.items())),
        pm25_ug_m3=_summarize(pm25_values, "ug/m3"),
        wind_speed_m_s=_summarize(wind_speeds, "m/s"),
        precipitation_mm_h=_summarize(precipitation, "mm/h"),
        temperature_c=(
            _summarize(temperatures, "degC") if temperatures else None
        ),
        relative_humidity_pct=(
            _summarize(humidity, "%") if humidity else None
        ),
        parameters=parameter_values,
    )


def check_nominal_lima_conditions(
    profile: LimaPlausibilityProfile,
    *,
    pm25_ug_m3: Sequence[float] = (),
    wind_speed_m_s: Sequence[float] = (),
    precipitation_mm_h: Sequence[float] = (),
    temperature_c: Sequence[float] = (),
    relative_humidity_pct: Sequence[float] = (),
) -> PlausibilityCheck:
    values: Mapping[str, tuple[Sequence[float], DistributionSummary | None]] = {
        "pm25_ug_m3": (pm25_ug_m3, profile.pm25_ug_m3),
        "wind_speed_m_s": (wind_speed_m_s, profile.wind_speed_m_s),
        "precipitation_mm_h": (
            precipitation_mm_h,
            profile.precipitation_mm_h,
        ),
        "temperature_c": (temperature_c, profile.temperature_c),
        "relative_humidity_pct": (
            relative_humidity_pct,
            profile.relative_humidity_pct,
        ),
    }
    violations: list[str] = []
    checked_counts: list[tuple[str, int]] = []
    for name, (provided, summary) in values.items():
        provided_values = np.asarray(tuple(provided), dtype=np.float64)
        checked_counts.append((name, int(provided_values.size)))
        if not provided_values.size:
            continue
        if summary is None:
            violations.append(f"{name}:historical_summary_unavailable")
            continue
        if not np.all(np.isfinite(provided_values)):
            violations.append(f"{name}:nonfinite")
            continue
        lower, upper, lower_label, upper_label = _nominal_bounds(name, summary)
        if np.any(provided_values < lower):
            violations.append(f"{name}:below_{lower_label}")
        if np.any(provided_values > upper):
            violations.append(f"{name}:above_{upper_label}")
    return PlausibilityCheck(
        nominal=not violations,
        violations=tuple(violations),
        checked_counts=tuple(checked_counts),
    )


def write_lima_plausibility_profile(
    profile: LimaPlausibilityProfile,
    path: Path | str,
) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"Lima plausibility profile already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"Temporary plausibility profile already exists: {temporary}"
        )
    serialized = json.dumps(profile.evidence_descriptor(), indent=2, sort_keys=True)
    try:
        temporary.write_text(f"{serialized}\n", encoding="utf-8")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _summarize(values: Sequence[float], unit: str) -> DistributionSummary:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"Distribution for {unit} requires finite values")
    percentiles = np.percentile(array, (1, 5, 50, 95, 99), method="linear")
    return DistributionSummary(
        unit=unit,
        count=int(array.size),
        minimum=float(np.min(array)),
        p01=float(percentiles[0]),
        p05=float(percentiles[1]),
        median=float(percentiles[2]),
        p95=float(percentiles[3]),
        p99=float(percentiles[4]),
        maximum=float(np.max(array)),
    )


def _summary_descriptor(summary: DistributionSummary) -> dict[str, object]:
    return {
        "unit": summary.unit,
        "count": summary.count,
        "minimum": summary.minimum,
        "p01": summary.p01,
        "p05": summary.p05,
        "median": summary.median,
        "p95": summary.p95,
        "p99": summary.p99,
        "maximum": summary.maximum,
        "percentile_method": summary.percentile_method,
    }


def _optional_summary_descriptor(
    summary: DistributionSummary | None,
) -> dict[str, object] | None:
    return None if summary is None else _summary_descriptor(summary)


def _nominal_bounds(
    name: str,
    summary: DistributionSummary,
) -> tuple[float, float, str, str]:
    if name == "precipitation_mm_h":
        return summary.minimum, summary.maximum, "observed_minimum", "observed_maximum"
    return summary.nominal_lower, summary.nominal_upper, "p01", "p99"


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
