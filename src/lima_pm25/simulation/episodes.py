"""Compose existing simulation requests into reproducible, partitioned episodes."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from lima_pm25.domain import ScenarioFamilyPurpose, ScenarioRunIdentity

from .orchestration import SimulationRunRequest
from .transport import AreaSource, EmissionProfile, GaussianKernelSource, RoadLineSource
from .weather import CorrelatedWindModel, HistoricalWeatherDataset, _require_seed


class EpisodeWeatherMode(StrEnum):
    HISTORICAL = "historical"
    HISTORICAL_PERTURBED = "historical_perturbed"
    SYNTHETIC = "synthetic"

    @property
    def label(self) -> str:
        return {
            self.HISTORICAL: "Meteorología histórica",
            self.HISTORICAL_PERTURBED: "Meteorología histórica con perturbaciones sintéticas",
            self.SYNTHETIC: "Meteorología sintética",
        }[self]


@dataclass(frozen=True, slots=True)
class SourceVariation:
    source_id: str
    provenance: str
    daily_hours: tuple[tuple[float, float], ...] = ()
    interpolation: str = "linear"
    intensity_range: tuple[float, float] = (1.0, 1.0)
    time_shift_s_range: tuple[float, float] = (0.0, 0.0)
    duration_scale_range: tuple[float, float] = (1.0, 1.0)
    activation_probability: float = 1.0

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.provenance.strip():
            raise ValueError("Source variation requires identity and provenance")
        for name in ("intensity_range", "time_shift_s_range", "duration_scale_range"):
            bounds = tuple(getattr(self, name))
            if (
                len(bounds) != 2
                or not all(math.isfinite(v) for v in bounds)
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"{name} must contain ordered finite bounds")
            object.__setattr__(self, name, bounds)
        if self.intensity_range[0] < 0 or self.duration_scale_range[0] <= 0:
            raise ValueError(
                "Intensity must be nonnegative and duration scale positive"
            )
        if (
            not math.isfinite(self.activation_probability)
            or not 0 <= self.activation_probability <= 1
        ):
            raise ValueError("activation_probability must be in [0, 1]")
        object.__setattr__(
            self, "daily_hours", tuple(tuple(point) for point in self.daily_hours)
        )
        EmissionProfile(
            tuple((hour * 3600, value) for hour, value in self.daily_hours)
            if self.daily_hours
            else ((0, 1),),
            self.interpolation,
        )
        if self.daily_hours and (
            self.daily_hours[-1][0] != 24
            or self.daily_hours[0][1] != self.daily_hours[-1][1]
        ):
            raise ValueError("Daily emission cycle must close at hour 24")
        if self.daily_hours and self.duration_scale_range != (1.0, 1.0):
            raise ValueError(
                "Daily cycles use phase shifts, not event duration scaling"
            )


@dataclass(frozen=True, slots=True)
class EpisodeGenerationConfig:
    training_start_utc: datetime
    training_end_utc: datetime
    evaluation_start_utc: datetime
    evaluation_end_utc: datetime
    weather_weights: tuple[float, float, float]
    source_variations: tuple[SourceVariation, ...]
    provenance: str
    stress: bool
    constraint_evidence_ids: tuple[str, ...] = ()
    perturbation_scale: float = 0.25
    synthetic_variability_scale: float = 1.0
    synthetic_precipitation_mm_h: float = 0.0
    area_sources: tuple[AreaSource, ...] = ()

    def __post_init__(self) -> None:
        dates = (
            self.training_start_utc,
            self.training_end_utc,
            self.evaluation_start_utc,
            self.evaluation_end_utc,
        )
        if any(value.utcoffset() != timedelta(0) for value in dates):
            raise ValueError("Episode partition limits must be UTC")
        if not dates[0] < dates[1] <= dates[2] < dates[3]:
            raise ValueError(
                "Training and evaluation periods must be ordered and disjoint"
            )
        weights = tuple(self.weather_weights)
        if (
            len(weights) != 3
            or any(not math.isfinite(v) or v < 0 for v in weights)
            or not math.isfinite(sum(weights))
            or sum(weights) <= 0
        ):
            raise ValueError(
                "Three finite nonnegative weather weights with positive sum are required"
            )
        object.__setattr__(self, "weather_weights", weights)
        object.__setattr__(self, "source_variations", tuple(self.source_variations))
        object.__setattr__(self, "area_sources", tuple(self.area_sources))
        object.__setattr__(
            self, "constraint_evidence_ids", tuple(self.constraint_evidence_ids)
        )
        if not self.provenance.strip():
            raise ValueError("Episode configuration needs provenance")
        if not self.stress and not self.constraint_evidence_ids:
            raise ValueError("Nominal episodes require Lima constraint evidence")
        if any(not value.strip() for value in self.constraint_evidence_ids):
            raise ValueError("Constraint evidence identifiers cannot be empty")
        if self.area_sources and not self.stress:
            raise ValueError("Area events in this generator belong to stress episodes")
        ids = [variation.source_id for variation in self.source_variations]
        if len(ids) != len(set(ids)):
            raise ValueError("Source variation IDs must be unique")
        for name in (
            "perturbation_scale",
            "synthetic_variability_scale",
            "synthetic_precipitation_mm_h",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def evidence_descriptor(self) -> dict[str, object]:
        return {
            "training_start_utc": self.training_start_utc.isoformat(),
            "training_end_utc": self.training_end_utc.isoformat(),
            "evaluation_start_utc": self.evaluation_start_utc.isoformat(),
            "evaluation_end_utc": self.evaluation_end_utc.isoformat(),
            "weather_weights": dict(
                zip(EpisodeWeatherMode, self.weather_weights, strict=True)
            ),
            "source_variations": [asdict(v) for v in self.source_variations],
            "provenance": self.provenance,
            "stress": self.stress,
            "constraint_evidence_ids": self.constraint_evidence_ids,
            "perturbation_scale": self.perturbation_scale,
            "synthetic_variability_scale": self.synthetic_variability_scale,
            "synthetic_precipitation_mm_h": self.synthetic_precipitation_mm_h,
            "area_sources": [s.evidence_descriptor() for s in self.area_sources],
        }


def load_episode_generation_config(path: Path | str) -> EpisodeGenerationConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in (
        "training_start_utc",
        "training_end_utc",
        "evaluation_start_utc",
        "evaluation_end_utc",
    ):
        payload[key] = datetime.fromisoformat(payload[key])
    payload["weather_weights"] = tuple(
        payload["weather_weights"][mode.value] for mode in EpisodeWeatherMode
    )
    payload["source_variations"] = tuple(
        SourceVariation(**item) for item in payload["source_variations"]
    )
    areas = []
    for item in payload.pop("area_sources", []):
        item["polygon"] = Polygon(item.pop("polygon_coordinates_m"))
        item["profile"] = EmissionProfile(**item.pop("profile", {}))
        areas.append(AreaSource(**item))
    return EpisodeGenerationConfig(**payload, area_sources=tuple(areas))


def compose_episode(
    template: SimulationRunRequest,
    weather: HistoricalWeatherDataset,
    config: EpisodeGenerationConfig,
    *,
    seed: int,
    partition: str = "training",
    weather_mode: EpisodeWeatherMode | str | None = None,
) -> SimulationRunRequest:
    """Reuse a request's domain, physics, sensors and elapsed-time trajectory.

    Training/evaluation use separate seed namespaces. Counterfactual callers can
    replace only the returned sources to retain the exact weather realization.
    """
    _require_seed(seed)
    if partition not in {"training", "evaluation"}:
        raise ValueError("partition must be training or evaluation")
    namespace = 0 if partition == "training" else 1
    streams = np.random.SeedSequence([namespace, seed]).spawn(4)
    mode_rng, window_seed, wind_seed, sensor_seed = (
        np.random.default_rng(streams[0]),
        int(streams[1].generate_state(1, dtype=np.uint64)[0]),
        int(streams[2].generate_state(1, dtype=np.uint64)[0]),
        int(streams[3].generate_state(1, dtype=np.uint64)[0]),
    )
    modes = tuple(EpisodeWeatherMode)
    mode = (
        EpisodeWeatherMode(weather_mode)
        if weather_mode is not None
        else modes[
            int(
                mode_rng.choice(
                    3,
                    p=np.asarray(config.weather_weights) / sum(config.weather_weights),
                )
            )
        ]
    )
    period_start, period_end = (
        (config.training_start_utc, config.training_end_utc)
        if partition == "training"
        else (config.evaluation_start_utc, config.evaluation_end_utc)
    )
    selected = weather.within_period(period_start, period_end).sample_window(
        timedelta(seconds=template.config.duration_s), seed=window_seed
    )
    generated_weather = selected
    if mode is not EpisodeWeatherMode.HISTORICAL:
        model = CorrelatedWindModel.fit(
            weather.within_period(config.training_start_utc, config.training_end_utc)
        )
        generated_weather = model.generate(
            seed=wind_seed,
            start_utc=selected.start_utc,
            duration_s=selected.duration_s,
            historical_base=selected
            if mode is EpisodeWeatherMode.HISTORICAL_PERTURBED
            else None,
            variability_scale=config.perturbation_scale
            if mode is EpisodeWeatherMode.HISTORICAL_PERTURBED
            else config.synthetic_variability_scale,
            precipitation_mm_h=config.synthetic_precipitation_mm_h
            if mode is EpisodeWeatherMode.SYNTHETIC
            else 0.0,
        )
    variations = {item.source_id: item for item in config.source_variations}
    all_sources = (*template.sources, *config.area_sources)
    if not config.stress and any(
        isinstance(source, AreaSource) for source in all_sources
    ):
        raise ValueError("Area events in this generator belong to stress episodes")
    source_ids = [source.source_id for source in all_sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Episode sources must have unique identities")
    if unknown := variations.keys() - set(source_ids):
        raise ValueError(f"Unknown source variations: {sorted(unknown)}")
    sources = []
    for source in all_sources:
        variation = variations.get(source.source_id)
        if variation is None:
            sources.append(source)
            continue
        if not isinstance(source, (RoadLineSource, GaussianKernelSource, AreaSource)):
            raise TypeError(
                f"Temporal variation is not supported for {source.source_id!r}"
            )
        # Stable source-specific streams: adding an event does not redraw other sources.
        identity = int.from_bytes(
            sha256(source.source_id.encode()).digest()[:8], "little"
        )
        rng = np.random.default_rng([identity, 4, namespace, seed])
        if rng.random() >= variation.activation_probability:
            continue
        intensity = float(rng.uniform(*variation.intensity_range))
        shift = float(rng.uniform(*variation.time_shift_s_range))
        scale = float(rng.uniform(*variation.duration_scale_range))
        start, end = source.starts_at_s, source.ends_at_s
        if variation.daily_hours:
            profile = EmissionProfile.daily(
                variation.daily_hours,
                start_utc=selected.start_utc - timedelta(seconds=shift),
                duration_s=selected.duration_s,
                interpolation=variation.interpolation,
            )
        else:
            if end is None and (shift != 0 or scale != 1):
                raise ValueError("Varying event timing requires a finite source end")
            start = source.starts_at_s + shift
            end = (
                start + (source.ends_at_s - source.starts_at_s) * scale
                if source.ends_at_s is not None
                else None
            )
            if start < 0 or (end is not None and end > selected.duration_s):
                raise ValueError(
                    "Sampled event does not fit the episode; adjust its bounds"
                )
            points = {
                0.0: source.profile.value_at(source.starts_at_s),
                start: source.profile.value_at(source.starts_at_s),
            }
            points.update(
                {
                    start + (t - source.starts_at_s) * scale: value
                    for t, value in source.profile.points
                    if t > source.starts_at_s
                    and (source.ends_at_s is None or t <= source.ends_at_s)
                }
            )
            if (
                end is not None
                and source.ends_at_s is not None
                and source.profile.interpolation == "linear"
            ):
                points[end] = source.profile.value_at(source.ends_at_s)
            profile = EmissionProfile(
                tuple(sorted(points.items())), source.profile.interpolation
            )
        profile = EmissionProfile(
            tuple((t, factor * intensity) for t, factor in profile.points),
            profile.interpolation,
        )
        sources.append(
            replace(
                source,
                profile=profile,
                starts_at_s=start,
                ends_at_s=end,
                provenance=f"{source.provenance}; {variation.provenance}",
            )
        )
    purpose = (
        ScenarioFamilyPurpose.STRESS_COUNTERFACTUAL
        if config.stress
        else (
            ScenarioFamilyPurpose.SYNTHETIC_TRAINING
            if partition == "training"
            else ScenarioFamilyPurpose.HELD_OUT_SYNTHETIC_EVALUATION
        )
    )
    metadata = {
        "seed": seed,
        "partition": partition,
        "weather_mode": mode.value,
        "weather_label": mode.label,
        "configuration": config.evidence_descriptor(),
        "selected_window": selected.evidence_descriptor(),
        "random_streams": "partitioned mode/window/wind/sensor plus stable source-ID streams",
    }
    return replace(
        template,
        weather=generated_weather,
        sources=tuple(sources),
        config=replace(template.config, start_utc=selected.start_utc, seed=sensor_seed),
        scenario_identity=ScenarioRunIdentity(
            "generated-episodes", purpose, f"{partition}-{mode.value}-{seed}", True
        ),
        episode_parameters_json=json.dumps(metadata, sort_keys=True),
    )
