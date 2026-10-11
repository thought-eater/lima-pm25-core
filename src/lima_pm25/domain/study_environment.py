from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

from .mobility import MobilityPolicy
from .spatial import DomainConfig

STUDY_ENVIRONMENT_SCHEMA_VERSION = 1


class SelectionStatus(StrEnum):
    CONFIRMED = "confirmed"
    PENDING_EVIDENCE = "pending_evidence"
    PENDING_ADVISER = "pending_adviser"
    PROVISIONAL_IMPLEMENTATION_VALUE = "provisional_implementation_value"
    TECHNICAL_SELECTION_COMPLETE = "technical_selection_complete"


class EvidenceStatus(StrEnum):
    SYNTHETIC_DEMO = "synthetic_demo"
    EVIDENCE_CONSTRAINED = "evidence_constrained"
    CALIBRATED = "calibrated"
    ADVISER_APPROVED = "adviser_approved"


class ScenarioFamilyPurpose(StrEnum):
    SYNTHETIC_TRAINING = "synthetic_training"
    HELD_OUT_SYNTHETIC_EVALUATION = "held_out_synthetic_evaluation"
    HISTORICAL_REPLAY = "historical_replay"
    STRESS_COUNTERFACTUAL = "stress_counterfactual"


class ScenarioFactor(StrEnum):
    SPATIAL_DOMAIN = "spatial_domain"
    PHYSICS = "physics"
    WEATHER = "weather"
    AMBIENT_BASELINE = "ambient_baseline"
    REGIONAL_BOUNDARY = "regional_boundary"
    ROUTINE_SOURCES = "routine_sources"
    SYNTHETIC_VALIDATION_SOURCES = "synthetic_validation_sources"
    EVENT_SOURCES = "event_sources"
    STATIONS = "stations"
    SENSORS = "sensors"
    MOBILITY = "mobility"
    TRAJECTORY = "trajectory"
    TEMPORAL_CONFIGURATION = "temporal_configuration"
    SEED = "seed"


@dataclass(frozen=True, slots=True)
class ScenarioMemberDefinition:
    member_id: str
    changed_factors: tuple[ScenarioFactor, ...] = ()
    claims_calibrated_providers: bool = False
    calibration_evidence_id: str | None = None

    def __post_init__(self) -> None:
        _require_text("member_id", self.member_id)
        factors = tuple(ScenarioFactor(value) for value in self.changed_factors)
        if len(factors) != len(set(factors)):
            raise ValueError("changed_factors must be unique")
        if self.claims_calibrated_providers:
            _require_text("calibration_evidence_id", self.calibration_evidence_id)
        elif self.calibration_evidence_id is not None:
            raise ValueError(
                "calibration_evidence_id requires claims_calibrated_providers"
            )
        object.__setattr__(self, "changed_factors", factors)


@dataclass(frozen=True, slots=True)
class ScenarioFamilyDefinition:
    family_id: str
    purpose: ScenarioFamilyPurpose
    control_member_id: str
    members: tuple[ScenarioMemberDefinition, ...]
    constraint_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text("family_id", self.family_id)
        _require_text("control_member_id", self.control_member_id)
        object.__setattr__(self, "purpose", ScenarioFamilyPurpose(self.purpose))
        members = tuple(self.members)
        if not members:
            raise ValueError("Scenario family must contain at least one member")
        by_id = {member.member_id: member for member in members}
        if len(by_id) != len(members):
            raise ValueError("Scenario family member IDs must be unique")
        try:
            control = by_id[self.control_member_id]
        except KeyError as error:
            raise ValueError(
                "control_member_id must identify one family member"
            ) from error
        if control.changed_factors:
            raise ValueError("The control member cannot declare changed factors")
        evidence = tuple(self.constraint_evidence_ids)
        for evidence_id in evidence:
            _require_text("constraint_evidence_id", evidence_id)
        if len(evidence) != len(set(evidence)):
            raise ValueError("constraint_evidence_ids must be unique")
        if (
            self.purpose
            in {
                ScenarioFamilyPurpose.SYNTHETIC_TRAINING,
                ScenarioFamilyPurpose.HELD_OUT_SYNTHETIC_EVALUATION,
            }
            and not evidence
        ):
            raise ValueError(
                "Nominal synthetic families require Lima constraint evidence"
            )
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "constraint_evidence_ids", evidence)

    def member(self, member_id: str) -> ScenarioMemberDefinition:
        for member in self.members:
            if member.member_id == member_id:
                return member
        raise KeyError(f"Unknown scenario member {member_id!r}")

    def run_identity(self, member_id: str) -> ScenarioRunIdentity:
        member = self.member(member_id)
        return ScenarioRunIdentity(
            family_id=self.family_id,
            family_purpose=self.purpose,
            member_id=member.member_id,
            is_control=member.member_id == self.control_member_id,
            changed_factors=member.changed_factors,
            claims_calibrated_providers=member.claims_calibrated_providers,
            calibration_evidence_id=member.calibration_evidence_id,
        )


@dataclass(frozen=True, slots=True)
class ScenarioRunIdentity:
    family_id: str
    family_purpose: ScenarioFamilyPurpose
    member_id: str
    is_control: bool
    changed_factors: tuple[ScenarioFactor, ...] = ()
    claims_calibrated_providers: bool = False
    calibration_evidence_id: str | None = None

    def __post_init__(self) -> None:
        _require_text("family_id", self.family_id)
        _require_text("member_id", self.member_id)
        object.__setattr__(
            self,
            "family_purpose",
            ScenarioFamilyPurpose(self.family_purpose),
        )
        factors = tuple(ScenarioFactor(value) for value in self.changed_factors)
        if len(factors) != len(set(factors)):
            raise ValueError("changed_factors must be unique")
        if self.is_control and factors:
            raise ValueError("A control run cannot declare changed factors")
        if self.claims_calibrated_providers:
            _require_text("calibration_evidence_id", self.calibration_evidence_id)
        elif self.calibration_evidence_id is not None:
            raise ValueError(
                "calibration_evidence_id requires claims_calibrated_providers"
            )
        object.__setattr__(self, "changed_factors", factors)


@dataclass(frozen=True, slots=True, kw_only=True)
class EffectiveVerticalRepresentation:
    mixing_height_m: float
    evidence_status: EvidenceStatus
    provenance: str
    time_varying: bool = False

    def __post_init__(self) -> None:
        _require_positive_finite("mixing_height_m", self.mixing_height_m)
        object.__setattr__(
            self,
            "evidence_status",
            EvidenceStatus(self.evidence_status),
        )
        _require_text("provenance", self.provenance)


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalStudyPolicy:
    concentration_output_interval_s: float | None
    concentration_output_times_s: tuple[float, ...] = ()
    concentration_output_status: SelectionStatus = (
        SelectionStatus.PROVISIONAL_IMPLEMENTATION_VALUE
    )
    mobile_sampling_interval_s: float | None = None
    mobile_sampling_status: SelectionStatus = (
        SelectionStatus.PROVISIONAL_IMPLEMENTATION_VALUE
    )

    def __post_init__(self) -> None:
        if self.concentration_output_interval_s is not None:
            _require_positive_finite(
                "concentration_output_interval_s",
                self.concentration_output_interval_s,
            )
        times = _strictly_increasing_nonnegative_times(
            "concentration_output_times_s",
            self.concentration_output_times_s,
        )
        object.__setattr__(self, "concentration_output_times_s", times)
        if self.concentration_output_interval_s is None and not times:
            raise ValueError(
                "Temporal policy requires an output interval or explicit times"
            )
        object.__setattr__(
            self,
            "concentration_output_status",
            SelectionStatus(self.concentration_output_status),
        )
        if self.mobile_sampling_interval_s is not None:
            _require_positive_finite(
                "mobile_sampling_interval_s",
                self.mobile_sampling_interval_s,
            )
        object.__setattr__(
            self,
            "mobile_sampling_status",
            SelectionStatus(self.mobile_sampling_status),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StudyEnvironmentConfig:
    environment_id: str
    domain: DomainConfig
    station_catalog_id: str
    mobility_policy: MobilityPolicy
    vertical: EffectiveVerticalRepresentation
    temporal: TemporalStudyPolicy
    horizontal_candidate_cell_sizes_m: tuple[float, ...]
    selected_cell_size_status: SelectionStatus
    scenario_families: tuple[ScenarioFamilyDefinition, ...]
    schema_version: int = STUDY_ENVIRONMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text("environment_id", self.environment_id)
        _require_text("station_catalog_id", self.station_catalog_id)
        if self.schema_version != STUDY_ENVIRONMENT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported study-environment schema {self.schema_version!r}"
            )
        candidates = tuple(
            float(value) for value in self.horizontal_candidate_cell_sizes_m
        )
        if not candidates:
            raise ValueError("horizontal_candidate_cell_sizes_m cannot be empty")
        for value in candidates:
            _require_positive_finite("horizontal candidate cell size", value)
        if len(candidates) != len(set(candidates)):
            raise ValueError("Horizontal candidate cell sizes must be unique")
        if tuple(sorted(candidates)) != candidates:
            raise ValueError("Horizontal candidate cell sizes must be increasing")
        if self.domain.cell_size_m not in candidates:
            raise ValueError(
                "The selected domain cell size must be one horizontal candidate"
            )
        families = tuple(self.scenario_families)
        family_ids = {family.family_id for family in families}
        if not families:
            raise ValueError("At least one scenario family is required")
        if len(family_ids) != len(families):
            raise ValueError("Scenario family IDs must be unique")
        object.__setattr__(
            self,
            "horizontal_candidate_cell_sizes_m",
            candidates,
        )
        object.__setattr__(
            self,
            "selected_cell_size_status",
            SelectionStatus(self.selected_cell_size_status),
        )
        object.__setattr__(self, "scenario_families", families)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "environment_id": self.environment_id,
            "domain": {
                "left": self.domain.left,
                "bottom": self.domain.bottom,
                "right": self.domain.right,
                "top": self.domain.top,
                "network_type": self.domain.network_type,
                "cell_size_m": self.domain.cell_size_m,
                "road_network_buffer_m": self.domain.road_network_buffer_m,
                "cache_directory": str(self.domain.cache_directory),
                "osmnx_cache_directory": str(self.domain.osmnx_cache_directory),
            },
            "station_catalog_id": self.station_catalog_id,
            "mobility_policy": {
                "road_tolerance_m": self.mobility_policy.road_tolerance_m,
                "maximum_speed_m_s": self.mobility_policy.maximum_speed_m_s,
            },
            "vertical": {
                "mixing_height_m": self.vertical.mixing_height_m,
                "evidence_status": self.vertical.evidence_status.value,
                "provenance": self.vertical.provenance,
                "time_varying": self.vertical.time_varying,
            },
            "temporal": {
                "concentration_output_interval_s": (
                    self.temporal.concentration_output_interval_s
                ),
                "concentration_output_times_s": list(
                    self.temporal.concentration_output_times_s
                ),
                "concentration_output_status": (
                    self.temporal.concentration_output_status.value
                ),
                "mobile_sampling_interval_s": (
                    self.temporal.mobile_sampling_interval_s
                ),
                "mobile_sampling_status": (self.temporal.mobile_sampling_status.value),
            },
            "horizontal_candidate_cell_sizes_m": list(
                self.horizontal_candidate_cell_sizes_m
            ),
            "selected_cell_size_status": self.selected_cell_size_status.value,
            "scenario_families": [
                {
                    "family_id": family.family_id,
                    "purpose": family.purpose.value,
                    "control_member_id": family.control_member_id,
                    "constraint_evidence_ids": list(family.constraint_evidence_ids),
                    "members": [
                        {
                            "member_id": member.member_id,
                            "changed_factors": [
                                factor.value for factor in member.changed_factors
                            ],
                            "claims_calibrated_providers": (
                                member.claims_calibrated_providers
                            ),
                            "calibration_evidence_id": (member.calibration_evidence_id),
                        }
                        for member in family.members
                    ],
                }
                for family in self.scenario_families
            ],
        }


def load_study_environment_config(path: Path | str) -> StudyEnvironmentConfig:
    source_path = Path(path)
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid study-environment JSON: {error}") from error
    if not isinstance(payload, dict):
        # Malformed configuration is reported as ValueError by this loader.
        raise ValueError("Study-environment configuration must be a JSON object")  # noqa: TRY004
    required = {
        "schema_version",
        "environment_id",
        "domain",
        "station_catalog_id",
        "mobility_policy",
        "vertical",
        "temporal",
        "horizontal_candidate_cell_sizes_m",
        "selected_cell_size_status",
        "scenario_families",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            "Study-environment configuration is missing keys: "
            + ", ".join(sorted(missing))
        )
    domain_payload = _require_mapping("domain", payload["domain"])
    mobility_payload = _require_mapping(
        "mobility_policy",
        payload["mobility_policy"],
    )
    vertical_payload = _require_mapping("vertical", payload["vertical"])
    temporal_payload = _require_mapping("temporal", payload["temporal"])
    families_payload = payload["scenario_families"]
    if not isinstance(families_payload, list):
        raise ValueError("scenario_families must be a JSON array")  # noqa: TRY004 -- configuration validation

    families = []
    for family_payload_value in families_payload:
        family_payload = _require_mapping(
            "scenario family",
            family_payload_value,
        )
        member_values = family_payload.get("members")
        if not isinstance(member_values, list):
            raise ValueError("Scenario-family members must be a JSON array")  # noqa: TRY004 -- configuration validation
        members = tuple(
            ScenarioMemberDefinition(
                member_id=_required_value(member, "member_id"),
                changed_factors=tuple(member.get("changed_factors", ())),
                claims_calibrated_providers=bool(
                    member.get("claims_calibrated_providers", False)
                ),
                calibration_evidence_id=member.get("calibration_evidence_id"),
            )
            for member_value in member_values
            for member in [_require_mapping("scenario member", member_value)]
        )
        families.append(
            ScenarioFamilyDefinition(
                family_id=_required_value(family_payload, "family_id"),
                purpose=_required_value(family_payload, "purpose"),
                control_member_id=_required_value(
                    family_payload,
                    "control_member_id",
                ),
                members=members,
                constraint_evidence_ids=tuple(
                    family_payload.get("constraint_evidence_ids", ())
                ),
            )
        )

    return StudyEnvironmentConfig(
        schema_version=int(payload["schema_version"]),
        environment_id=str(payload["environment_id"]),
        domain=DomainConfig(
            left=float(_required_value(domain_payload, "left")),
            bottom=float(_required_value(domain_payload, "bottom")),
            right=float(_required_value(domain_payload, "right")),
            top=float(_required_value(domain_payload, "top")),
            network_type=str(_required_value(domain_payload, "network_type")),
            cell_size_m=float(_required_value(domain_payload, "cell_size_m")),
            road_network_buffer_m=float(
                _required_value(domain_payload, "road_network_buffer_m")
            ),
            cache_directory=Path(domain_payload.get("cache_directory", "cache/domain")),
            osmnx_cache_directory=Path(
                domain_payload.get("osmnx_cache_directory", "cache/osmnx")
            ),
        ),
        station_catalog_id=str(payload["station_catalog_id"]),
        mobility_policy=MobilityPolicy(
            road_tolerance_m=float(
                _required_value(mobility_payload, "road_tolerance_m")
            ),
            maximum_speed_m_s=(
                None
                if mobility_payload.get("maximum_speed_m_s") is None
                else float(mobility_payload["maximum_speed_m_s"])
            ),
        ),
        vertical=EffectiveVerticalRepresentation(
            mixing_height_m=float(_required_value(vertical_payload, "mixing_height_m")),
            evidence_status=_required_value(
                vertical_payload,
                "evidence_status",
            ),
            provenance=str(_required_value(vertical_payload, "provenance")),
            time_varying=bool(vertical_payload.get("time_varying", False)),
        ),
        temporal=TemporalStudyPolicy(
            concentration_output_interval_s=(
                None
                if temporal_payload.get("concentration_output_interval_s") is None
                else float(temporal_payload["concentration_output_interval_s"])
            ),
            concentration_output_times_s=tuple(
                temporal_payload.get("concentration_output_times_s", ())
            ),
            concentration_output_status=temporal_payload.get(
                "concentration_output_status",
                SelectionStatus.PROVISIONAL_IMPLEMENTATION_VALUE,
            ),
            mobile_sampling_interval_s=(
                None
                if temporal_payload.get("mobile_sampling_interval_s") is None
                else float(temporal_payload["mobile_sampling_interval_s"])
            ),
            mobile_sampling_status=temporal_payload.get(
                "mobile_sampling_status",
                SelectionStatus.PROVISIONAL_IMPLEMENTATION_VALUE,
            ),
        ),
        horizontal_candidate_cell_sizes_m=tuple(
            payload["horizontal_candidate_cell_sizes_m"]
        ),
        selected_cell_size_status=payload["selected_cell_size_status"],
        scenario_families=tuple(families),
    )


def _required_value(mapping: dict[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise ValueError(f"Missing required configuration key {key!r}") from error


def _require_mapping(name: str, value: object) -> dict[str, Any]:
    """Narrow the JSON container; individual values are validated by the models."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")  # noqa: TRY004 -- configuration validation
    return value


def _strictly_increasing_nonnegative_times(
    name: str,
    values: tuple[float, ...],
) -> tuple[float, ...]:
    times = tuple(float(value) for value in values)
    previous = -math.inf
    for value in times:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must contain nonnegative finite values")
        if value <= previous:
            raise ValueError(f"{name} must be strictly increasing")
        previous = value
    return times


def _require_text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _require_positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
