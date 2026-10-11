import json
from pathlib import Path

import pytest

from lima_pm25.domain import (
    EvidenceStatus,
    ScenarioFactor,
    ScenarioFamilyDefinition,
    ScenarioFamilyPurpose,
    ScenarioMemberDefinition,
    SelectionStatus,
    load_study_environment_config,
)


def test_versioned_study_configuration_loads_with_explicit_statuses() -> None:
    config = load_study_environment_config("config/study_environment.json")

    assert config.environment_id == "lima-r1.2-study"
    assert config.domain.cell_size_m == 200
    assert config.horizontal_candidate_cell_sizes_m == (50, 100, 200)
    assert (
        config.selected_cell_size_status is SelectionStatus.TECHNICAL_SELECTION_COMPLETE
    )
    assert config.vertical.evidence_status is EvidenceStatus.SYNTHETIC_DEMO
    assert config.temporal.mobile_sampling_interval_s == 300
    assert (
        config.temporal.concentration_output_status
        is SelectionStatus.TECHNICAL_SELECTION_COMPLETE
    )
    family = config.scenario_families[0]
    assert family.purpose is ScenarioFamilyPurpose.STRESS_COUNTERFACTUAL
    assert family.members[1].changed_factors == (ScenarioFactor.EVENT_SOURCES,)
    assert len(config.fingerprint) == 64


def test_configuration_round_trip_retains_fingerprint(tmp_path: Path) -> None:
    original = load_study_environment_config("config/study_environment.json")
    path = tmp_path / "study.json"
    path.write_text(
        json.dumps(original.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    restored = load_study_environment_config(path)

    assert restored == original
    assert restored.fingerprint == original.fingerprint


def test_configuration_requires_selected_cell_size_among_candidates(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        Path("config/study_environment.json").read_text(encoding="utf-8")
    )
    payload["domain"]["cell_size_m"] = 75
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be one horizontal candidate"):
        load_study_environment_config(path)


def test_control_member_cannot_declare_changed_factors(tmp_path: Path) -> None:
    payload = json.loads(
        Path("config/study_environment.json").read_text(encoding="utf-8")
    )
    payload["scenario_families"][0]["members"][0]["changed_factors"] = ["weather"]
    path = tmp_path / "invalid-family.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="control member"):
        load_study_environment_config(path)


def test_temporal_policy_requires_interval_or_explicit_times(tmp_path: Path) -> None:
    payload = json.loads(
        Path("config/study_environment.json").read_text(encoding="utf-8")
    )
    payload["temporal"]["concentration_output_interval_s"] = None
    payload["temporal"]["concentration_output_times_s"] = []
    path = tmp_path / "invalid-temporal.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="requires an output interval"):
        load_study_environment_config(path)


def test_nominal_synthetic_family_requires_constraint_evidence() -> None:
    with pytest.raises(ValueError, match="require Lima constraint evidence"):
        ScenarioFamilyDefinition(
            family_id="training",
            purpose=ScenarioFamilyPurpose.SYNTHETIC_TRAINING,
            control_member_id="control",
            members=(ScenarioMemberDefinition("control"),),
        )

    family = ScenarioFamilyDefinition(
        family_id="training",
        purpose=ScenarioFamilyPurpose.SYNTHETIC_TRAINING,
        control_member_id="control",
        members=(ScenarioMemberDefinition("control"),),
        constraint_evidence_ids=("lima-literature-profile-v1",),
    )
    assert family.constraint_evidence_ids == ("lima-literature-profile-v1",)
