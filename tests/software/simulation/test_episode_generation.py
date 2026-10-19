import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from lima_pm25.simulation import (
    CorrelatedWindModel,
    EmissionProfile,
    EpisodeGenerationConfig,
    EpisodeWeatherMode,
    HistoricalWeatherDataset,
    PM25Transport,
    SimulationRunConfig,
    SimulationRunRequest,
    SourceVariation,
    WeatherCoverageError,
    WeatherRecord,
    WeatherValueProvenance,
    compose_episode,
    load_episode_generation_config,
    run_simulation,
    verify_run_bundle,
    write_run_bundle,
)
from tests.transport_cases import _physics, _source, _square_grid


@pytest.fixture
def dataset():
    provenance = WeatherValueProvenance("senamhi_observed")
    start = datetime(2025, 1, 1, 5, tzinfo=UTC)
    rng = np.random.default_rng(122)
    residual = np.zeros(2)
    records = []
    for index in range(24 * 16):
        residual = 0.8 * residual + rng.normal(0, 0.1, size=2)
        records.append(
            WeatherRecord(
                "TEST",
                start + timedelta(hours=index),
                20,
                80,
                float(1 + 0.2 * np.sin(index * np.pi / 12) + residual[0]),
                float(0.5 + residual[1]),
                float(index % 7 == 0),
                provenance,
                provenance,
                provenance,
                provenance,
                provenance,
            )
        )
    return HistoricalWeatherDataset(
        "TEST",
        "fixture",
        "synthetic test fixture",
        tuple(records),
        resource_id="fixture.csv",
        resource_sha256="fixture",
    )


@pytest.fixture
def config(dataset):
    start = dataset.coverage_start_utc
    return EpisodeGenerationConfig(
        start,
        start + timedelta(days=8),
        start + timedelta(days=8),
        start + timedelta(days=16),
        (1, 1, 1),
        (
            SourceVariation(
                "road",
                "synthetic test",
                daily_hours=((0, 0.5), (8, 2), (18, 1.5), (24, 0.5)),
            ),
        ),
        "synthetic test",
        True,
    )


@pytest.fixture
def template(dataset):
    grid = _square_grid(side_cells=3)
    model = PM25Transport(grid, _physics())
    return SimulationRunRequest(
        model,
        model.initialize_from_array(np.zeros(grid.shape)),
        dataset.select_window(dataset.coverage_start_utc, timedelta(hours=3)),
        SimulationRunConfig(dataset.coverage_start_utc, 10800, 3600, 1),
        sources=(_source(source_id="road", x_m=150, y_m=150, sigma_m=40),),
    )


@pytest.mark.parametrize("mode", list(EpisodeWeatherMode))
def test_reproducible_composition_and_retained_partition(
    dataset, config, template, mode
):
    first = compose_episode(
        template, dataset, config, seed=12, partition="evaluation", weather_mode=mode
    )
    second = compose_episode(
        template, dataset, config, seed=12, partition="evaluation", weather_mode=mode
    )
    assert first.weather.evidence_descriptor() == second.weather.evidence_descriptor()
    assert [s.evidence_descriptor() for s in first.sources] == [
        s.evidence_descriptor() for s in second.sources
    ]
    assert first.config.start_utc >= config.evaluation_start_utc
    assert first.weather.end_utc < config.evaluation_end_utc
    for time_s in (0, 100, 3599, 3600, 9000, 10800):
        assert first.weather.forcing_at_elapsed_s(
            time_s
        ) == second.weather.forcing_at_elapsed_s(time_s)


def test_evaluation_values_cannot_change_fitted_model(dataset, config, template):
    altered = replace(
        dataset,
        records=tuple(
            replace(r, eastward_wind_m_s=99)
            if r.timestamp_utc >= config.evaluation_start_utc
            else r
            for r in dataset.records
        ),
    )
    first = compose_episode(template, dataset, config, seed=2, weather_mode="synthetic")
    second = compose_episode(
        template, altered, config, seed=2, weather_mode="synthetic"
    )
    assert first.weather.evidence_descriptor() == second.weather.evidence_descriptor()


def test_windows_do_not_cross_gaps_or_borrow_partition_endpoint(dataset):
    start = dataset.coverage_start_utc
    bounded = dataset.within_period(start, start + timedelta(hours=4))
    with pytest.raises(WeatherCoverageError):
        bounded.sample_window(timedelta(hours=4), seed=1)
    broken = replace(
        bounded,
        records=tuple(r for i, r in enumerate(bounded.records) if i != 1),
        unavailable_timestamps_utc=(start + timedelta(hours=1),),
    )
    with pytest.raises(WeatherCoverageError):
        broken.sample_window(timedelta(hours=2), seed=1)


def test_zero_perturbation_preserves_weather_and_rain_provenance(dataset):
    model = CorrelatedWindModel.fit(dataset)
    base = dataset.select_window(dataset.coverage_start_utc, timedelta(hours=3))
    result = model.generate(
        seed=42,
        start_utc=base.start_utc,
        duration_s=base.duration_s,
        historical_base=base,
        variability_scale=0,
    )
    before = base.evidence_descriptor()
    for time_s in (0, 123, 3599, 3600, 7199, 10800):
        expected, actual = (
            base.forcing_at_elapsed_s(time_s),
            result.forcing_at_elapsed_s(time_s),
        )
        assert actual.wind_components_m_s == pytest.approx(expected.wind_components_m_s)
        assert actual.precipitation_mm_h == expected.precipitation_mm_h
        diagnostic = result.diagnostic_at_elapsed_s(time_s)
        assert diagnostic.kind == "synthetic"
        assert (
            diagnostic.precipitation_provenance
            == base.diagnostic_at_elapsed_s(time_s).precipitation_provenance
        )
        assert diagnostic.wind_speed_provenance.source == "synthetic_perturbation"
    assert base.evidence_descriptor() == before


def test_correlated_weather_has_declared_covariance_and_persistence():
    # Long uncapped, zero-mean reference isolates the stochastic process.
    model = CorrelatedWindModel(
        ((0.0, 0.0),) * 24, ((1.0, 0.6), (0.6, 2.0)), 0.8, 100, "{}"
    )
    result = model.generate(
        seed=71, start_utc=datetime(2025, 1, 1, tzinfo=UTC), duration_s=10000 * 3600
    )
    vectors = np.asarray([item.forcing.wind_components_m_s for item in result.values])
    assert vectors.mean(axis=0) == pytest.approx([0, 0], abs=0.1)
    assert np.cov(vectors.T) == pytest.approx(
        np.asarray(model.residual_covariance), abs=0.15
    )
    for component in (0, 1):
        assert np.corrcoef(vectors[:-1, component], vectors[1:, component])[
            0, 1
        ] == pytest.approx(0.8, abs=0.03)


def test_environment_does_not_depend_on_sensor_seed_or_output_requests(
    dataset, config, template
):
    first = compose_episode(
        template, dataset, config, seed=9, weather_mode="historical_perturbed"
    )
    alternate = replace(
        template,
        config=replace(template.config, seed=999, concentration_output_interval_s=517),
    )
    second = compose_episode(
        alternate, dataset, config, seed=9, weather_mode="historical_perturbed"
    )
    for t in reversed((0, 17, 1000, 3599, 6000, 10800)):
        assert first.weather.forcing_at_elapsed_s(
            t
        ) == second.weather.forcing_at_elapsed_s(t)
    assert first.weather.evidence_descriptor() == second.weather.evidence_descriptor()


def test_adding_source_does_not_redraw_existing_sources_or_weather(
    dataset, config, template
):
    base = compose_episode(template, dataset, config, seed=17)
    extra = _source(
        source_id="extra",
        x_m=150,
        y_m=150,
        sigma_m=40,
        starts_at_s=3600,
        ends_at_s=7200,
    )
    config2 = replace(
        config,
        source_variations=(
            *config.source_variations,
            SourceVariation("extra", "fixture", intensity_range=(0.2, 1.2)),
        ),
    )
    other = compose_episode(
        replace(template, sources=(*template.sources, extra)), dataset, config2, seed=17
    )
    assert base.weather.evidence_descriptor() == other.weather.evidence_descriptor()
    assert (
        base.sources[0].evidence_descriptor() == other.sources[0].evidence_descriptor()
    )


def test_generated_episode_uses_existing_bundle_exporter(
    dataset, config, template, tmp_path
):
    request = compose_episode(
        template, dataset, config, seed=7, weather_mode="historical_perturbed"
    )
    result = run_simulation(request)
    write_run_bundle(result, tmp_path / result.run_id)
    assert verify_run_bundle(tmp_path / result.run_id).valid
    manifest = json.loads(result.manifest_json)
    assert (
        manifest["episode_generation"]["weather_label"]
        == "Meteorología histórica con perturbaciones sintéticas"
    )
    assert manifest["sources"][0]["profile"]["points"]
    assert all(abs(d.budget.relative_residual) < 1e-12 for d in result.diagnostics)


def test_example_configuration_loads_and_closes_daily_cycles():
    config = load_episode_generation_config("config/episode_generation.json")
    assert config.stress
    assert len(config.area_sources) == 1


@pytest.mark.parametrize(
    "points", [(), ((1, 1),), ((0, -1),), ((0, 1), (0, 2)), ((0, 1), (float("nan"), 1))]
)
def test_invalid_profiles_rejected(points):
    with pytest.raises(ValueError):
        EmissionProfile(points)


def test_overlapping_partitions_rejected(config):
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, evaluation_start_utc=config.training_start_utc)


def test_event_scaling_preserves_ramp_cut_by_active_end(dataset, config, template):
    source = _source(
        source_id="event",
        x_m=150,
        y_m=150,
        sigma_m=40,
        starts_at_s=2,
        ends_at_s=4,
        emission_rate_ug_s=1,
        profile=EmissionProfile(((0, 0), (6, 6)), "linear"),
    )
    variation = SourceVariation(
        "event", "fixture", time_shift_s_range=(2, 2), duration_scale_range=(2, 2)
    )
    generated = compose_episode(
        replace(template, sources=(source,)),
        dataset,
        replace(config, source_variations=(variation,)),
        seed=0,
        weather_mode="historical",
    )
    result = generated.sources[0].contribution_for_interval(
        template.model.grid, start_s=0, duration_s=10
    )
    # The original ramp emits integral_2^4 t dt = 6; doubling duration gives 12.
    assert result.integrated_mass_ug == pytest.approx(12)
