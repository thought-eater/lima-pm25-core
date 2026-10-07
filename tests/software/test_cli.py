from types import SimpleNamespace

import numpy as np
import pytest

import lima_pm25.cli as cli


@pytest.fixture
def isolated_cli(monkeypatch: pytest.MonkeyPatch):
    domain = object()
    station_catalog = object()
    monkeypatch.setattr(cli, "create_spatial_domain", lambda _config: domain)
    monkeypatch.setattr(
        cli,
        "create_lima_pm25_station_catalog",
        lambda _domain: station_catalog,
    )
    monkeypatch.setattr(cli, "_print_domain_summary", lambda _domain: None)
    return SimpleNamespace(domain=domain, station_catalog=station_catalog)


def test_no_arguments_run_headless_simulation(
    isolated_cli,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        cli,
        "run_complete_synthetic_demo",
        lambda domain, *, station_catalog, study_config: (
            calls.append((domain, station_catalog)) or _complete_result()
        ),
    )

    cli.main([])

    assert calls == [(isolated_cli.domain, isolated_cli.station_catalog)]


def test_complete_example_receives_selected_study_configuration(
    isolated_cli, monkeypatch, tmp_path,
) -> None:
    import json
    from pathlib import Path
    payload = json.loads(Path("config/study_environment.json").read_text())
    payload["temporal"]["concentration_output_interval_s"] = 437.0
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps(payload))
    received = []
    def run(domain, *, station_catalog, study_config):
        received.append(study_config.temporal.concentration_output_interval_s)
        return _complete_result()
    monkeypatch.setattr(cli, "run_complete_synthetic_demo", run)
    cli.main(["--study-config", str(config_path)])
    assert received == [437.0]


def test_domain_summary_reports_projected_operational_grid_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    graph = SimpleNamespace(
        number_of_nodes=lambda: 3,
        number_of_edges=lambda: 2,
    )
    grid = SimpleNamespace(
        crs="EPSG:32718",
        rows=4,
        columns=5,
        cell_size_m=100,
        width_m=500,
        height_m=400,
        active_mask=np.ones((4, 5), dtype=bool),
    )
    domain = SimpleNamespace(
        grid=grid,
        road_graph=graph,
        graph_source="cache",
        grid_source="cache",
    )

    cli._print_domain_summary(domain)

    output = capsys.readouterr().out
    assert "EPSG:32718" in output
    assert "4 filas x 5 columnas (100 m)" in output
    assert "500.00 m x 400.00 m" in output
    assert "Celdas activas: 20" in output
    assert "Distancia" not in output


def test_animate_uses_requested_playback_settings(
    isolated_cli,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def animate(
        domain,
        *,
        playback_speed,
        frame_duration_s,
        start_paused,
        station_catalog,
    ):
        calls.append(
            (
                domain,
                playback_speed,
                frame_duration_s,
                start_paused,
                station_catalog,
            )
        )
        return _result()

    monkeypatch.setattr(cli, "animate_synthetic_demo", animate)

    cli.main(
        [
            "--animate",
            "--playback-speed",
            "120",
            "--frame-seconds",
            "30",
            "--start-paused",
        ]
    )

    assert calls == [
        (
            isolated_cli.domain,
            120,
            30,
            True,
            isolated_cli.station_catalog,
        )
    ]


def test_animation_keeps_automatic_playback_by_default(
    isolated_cli,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_modes = []

    def animate(_domain, **settings):
        start_modes.append(settings["start_paused"])
        return _result()

    monkeypatch.setattr(cli, "animate_synthetic_demo", animate)

    cli.main(["--animate"])

    assert start_modes == [False]


def test_domain_only_does_not_run_simulation(
    isolated_cli,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        cli,
        "plot_spatial_domain",
        lambda domain, path, *, station_catalog: calls.append(
            (domain, path, station_catalog)
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_complete_synthetic_demo",
        lambda _domain, **_kwargs: pytest.fail("simulation must not run"),
    )

    cli.main(["--domain-only"])

    assert calls[0][0] is isolated_cli.domain
    assert calls[0][1].name == "domain_preview.png"
    assert calls[0][2] is isolated_cli.station_catalog


def test_study_evidence_uses_selected_configuration_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        cli,
        "_build_study_evidence",
        lambda config, output, discretization: calls.append(
            (config, output, discretization)
        ),
    )
    monkeypatch.setattr(
        cli,
        "create_spatial_domain",
        lambda *_args, **_kwargs: pytest.fail(
            "study mode delegates domain creation to the evidence builder"
        ),
    )

    cli.main(
        [
            "--study-evidence",
            "--study-config",
            "custom-study.json",
            "--study-output",
            "evidence",
        ]
    )

    assert calls == [
        (
            cli.Path("custom-study.json"),
            cli.Path("evidence"),
            cli.Path("resultados/r1_2_entorno/discretization_evidence.json"),
        )
    ]


def test_study_sensitivity_uses_predeclared_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        cli,
        "_run_study_sensitivity",
        lambda config, protocol, output: calls.append((config, protocol, output)),
    )

    cli.main(
        [
            "--study-sensitivity",
            "--study-config",
            "study.json",
            "--sensitivity-protocol",
            "protocol.json",
            "--sensitivity-output",
            "result.json",
        ]
    )

    assert calls == [
        (
            cli.Path("study.json"),
            cli.Path("protocol.json"),
            cli.Path("result.json"),
        )
    ]


def test_sensitivity_command_writes_only_experimental_summary(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "create_spatial_domain", lambda config: object())
    monkeypatch.setattr(cli, "_print_domain_summary", lambda domain: None)
    def experiment(domain, config, protocol):
        calls.append(config)
        return {"technical_selection": {"cell_size_m": 200.0,
                "concentration_output_interval_s": 300.0}}
    monkeypatch.setattr(cli, "run_study_discretization_experiment", experiment)
    output = tmp_path / "result.json"
    cli._run_study_sensitivity(cli.Path("config/study_environment.json"),
        cli.Path("config/discretization_protocol.json"), output)
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == [output]
    with pytest.raises(FileExistsError):
        cli._run_study_sensitivity(cli.Path("config/study_environment.json"),
            cli.Path("config/discretization_protocol.json"), output)
    assert len(calls) == 1


def test_historical_calibration_requires_noise_and_delegates_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        cli,
        "_run_historical_calibration",
        lambda study, profile, pm25, weather, station, output, noise: calls.append(
            (study, profile, pm25, weather, station, output, noise)
        ),
    )

    with pytest.raises(SystemExit):
        cli.main(["--historical-calibration"])

    cli.main(
        [
            "--historical-calibration",
            "--historical-noise-stddev",
            "0",
            "--pm25-data",
            "pm.csv",
            "--weather-data",
            "weather.csv",
            "--weather-station",
            "WEATHER",
            "--historical-output",
            "historical-output",
            "--study-config",
            "study.json",
        ]
    )

    assert calls == [
        (
            cli.Path("study.json"),
            cli.Path("config/estimation_evaluation.json"),
            cli.Path("pm.csv"),
            cli.Path("weather.csv"),
            "WEATHER",
            cli.Path("historical-output"),
            0,
        )
    ]


def _result():
    state = SimpleNamespace(
        simulation_time_s=10_800,
        concentration_ug_m3=np.array([[8.0]]),
    )
    budget = SimpleNamespace(numerical_residual_ug=0.0)
    return state, (budget,)


def _complete_result():
    state, budgets = _result()
    result = SimpleNamespace(
        final_state=state,
        diagnostics=tuple(SimpleNamespace(budget=budget) for budget in budgets),
    )
    return result, "resultados/r1_3_simulacion/test-run"
