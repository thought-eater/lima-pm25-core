"""Protocolo declarado y lectura del informe de discretización conservado.

Estas comprobaciones no vuelven a ejecutar las simulaciones de sensibilidad.
"""

import json
from pathlib import Path

from lima_pm25.simulation import load_discretization_protocol, render_discretization_markdown


def test_predeclared_discretization_protocol_has_fixed_candidates_and_rules() -> None:
    protocol = load_discretization_protocol("config/discretization_protocol.json")

    assert protocol["horizontal"]["candidate_cell_sizes_m"] == [50, 100, 200]
    assert protocol["horizontal"]["probe_relative_tolerance"] == 0.10
    assert protocol["horizontal"]["domain_mean_relative_tolerance"] == 0.05
    assert protocol["horizontal"]["event_effect_significance_fraction_of_peak"] == 0.01
    assert protocol["temporal"]["candidate_output_intervals_s"] == [
        300,
        900,
        3600,
    ]
    assert protocol["temporal"]["shared_field_l1_relative_tolerance"] == 0.03
    assert protocol["temporal"]["trajectory_l1_relative_tolerance"] == 0.05


def test_completed_discretization_evidence_has_human_summary() -> None:
    evidence = json.loads(
        Path("resultados/r1_2_entorno/discretization_evidence.json").read_text(
            encoding="utf-8"
        )
    )

    summary = render_discretization_markdown(evidence)

    assert "Selected horizontal cell size: `200.0 m`" in summary
    assert "Selected concentration-output interval: `300.0 s`" in summary
    assert "`100_vs_50`" in summary
    assert "`900_vs_300`" in summary
