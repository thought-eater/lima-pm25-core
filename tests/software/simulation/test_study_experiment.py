import json
from pathlib import Path

import pytest

from lima_pm25.simulation import (
    load_discretization_protocol,
    write_discretization_evidence,
)


def test_protocol_rejects_non_increasing_candidates(tmp_path: Path) -> None:
    protocol = json.loads(
        Path("config/discretization_protocol.json").read_text(encoding="utf-8")
    )
    protocol["horizontal"]["candidate_cell_sizes_m"] = [50, 50, 200]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="strictly increasing"):
        load_discretization_protocol(path)


def test_discretization_evidence_is_non_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    evidence = {"technical_selection": {"cell_size_m": 100}}

    write_discretization_evidence(evidence, path)

    assert json.loads(path.read_text(encoding="utf-8")) == evidence
    with pytest.raises(FileExistsError):
        write_discretization_evidence(evidence, path)

