"""
BeanHealth CLR — Knowledge Base Schema
=======================================

Pydantic models for every KB file. Each clinical value is an *entry*:

    id             unique across the whole KB
    value          the value itself (a number, or a structured object)
    source         where the value came from
    review_status  draft | needs_clinician_review | approved
    notes          free text — caveats, conflicts, open questions

An entry may only be `approved` with a non-null value and a named reviewer.
A null value marks a threshold nobody has set yet; the engine treats any rule
that needs it as unavailable rather than guessing.

Models forbid unknown fields, so a misspelt key fails loudly instead of being
silently ignored.
"""

from __future__ import annotations

from typing import Any, Dict, Generic, List, Literal, Optional, Tuple, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

ReviewStatus = Literal["draft", "needs_clinician_review", "approved"]
Eye = Literal["R", "L"]
Side = Literal["right", "left"]
Action = Literal["elevation", "depression", "adduction", "abduction", "intorsion", "extorsion"]
# Gaze positions, named from the PATIENT's point of view ("right" = the
# patient looks to their own right).
Gaze = Literal[
    "primary", "right", "left", "up", "down",
    "up_right", "up_left", "down_right", "down_left",
]
MuscleCode = Literal["MR", "LR", "SR", "IR", "SO", "IO"]
EYE_MUSCLE_PATTERN = r"^[RL]-(MR|LR|SR|IR|SO|IO)$"
ID_PATTERN = r"^[A-Za-z0-9_]+$"

T = TypeVar("T")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Alternative(Strict):
    """A competing value from another source, kept until a clinician decides."""
    value: Any
    unit: Optional[str] = None
    source: str = Field(min_length=1)
    notes: str = ""


class Entry(Strict, Generic[T]):
    id: str = Field(pattern=ID_PATTERN)
    value: T
    source: str = Field(min_length=1)
    review_status: ReviewStatus
    notes: str = ""
    alternatives: List[Alternative] = Field(default_factory=list)
    reviewed_by: Optional[str] = None
    reviewed_on: Optional[str] = None

    @model_validator(mode="after")
    def _approval_is_complete(self) -> "Entry[T]":
        if self.review_status == "approved":
            if self.value is None:
                raise ValueError(f"{self.id}: an approved entry needs a value")
            if not self.reviewed_by or not self.reviewed_on:
                raise ValueError(f"{self.id}: an approved entry needs reviewed_by and reviewed_on")
        return self


class Scalar(Entry[Optional[float]]):
    """A single number with a unit. A null value means 'not set yet'."""
    unit: str = Field(min_length=1)
    plausible_range: Optional[Tuple[float, float]] = None

    @model_validator(mode="after")
    def _range_is_ordered(self) -> "Scalar":
        if self.plausible_range and self.plausible_range[0] > self.plausible_range[1]:
            raise ValueError(f"{self.id}: plausible_range must be (low, high)")
        return self


# ─────────────────────────────────────────────────────────────
# manifest.yaml
# ─────────────────────────────────────────────────────────────

class Manifest(Strict):
    kb_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    released: str
    files: List[str] = Field(min_length=1)
    content_sha256: str


# ─────────────────────────────────────────────────────────────
# conventions.yaml
# ─────────────────────────────────────────────────────────────

class ReflexReading(Strict):
    """Reflex direction on the cornea → the way the eye is turned."""
    temporal: Literal["eso"]
    nasal: Literal["exo"]
    inferior: Literal["hyper"]
    superior: Literal["hypo"]


class SignConvention(Strict):
    horizontal_positive: Literal["eso"]
    vertical_positive: Literal["right_hyper"]


class ConventionsFile(Strict):
    laterality: Entry[str]
    sign_convention: Entry[SignConvention]
    reflex_reading: Entry[ReflexReading]
    prism_dioptre_conversion: Entry[str]


# ─────────────────────────────────────────────────────────────
# measurement.yaml / thresholds.yaml
# ─────────────────────────────────────────────────────────────

class ScalarsFile(Strict):
    entries: List[Scalar]


# ─────────────────────────────────────────────────────────────
# muscles.yaml
# ─────────────────────────────────────────────────────────────

class DiagnosticPosition(Strict):
    """The gaze where the muscle is the main mover, relative to its own eye."""
    vertical: Optional[Literal["up", "down"]] = None
    horizontal: Literal["in", "out"]


class Actions(Strict):
    primary: List[Action] = Field(min_length=1)
    secondary: List[Action] = Field(default_factory=list)
    tertiary: List[Action] = Field(default_factory=list)


class Muscle(Strict):
    name: str
    actions: Actions
    nerve: str
    diagnostic_position: DiagnosticPosition


class MuscleEntry(Entry[Muscle]):
    id: MuscleCode  # type: ignore[assignment]


class YokePair(Strict):
    gaze: Gaze
    muscles: Tuple[str, str]

    @model_validator(mode="after")
    def _eye_muscle_ids(self) -> "YokePair":
        import re
        for m in self.muscles:
            if not re.match(EYE_MUSCLE_PATTERN, m):
                raise ValueError(f"yoke muscle {m!r} is not an eye-muscle id like 'R-SR'")
        return self


class MusclesFile(Strict):
    muscles: List[MuscleEntry]
    plane_angles: List[Scalar]
    yoke_pairs: List[Entry[YokePair]]


# ─────────────────────────────────────────────────────────────
# parks_3_step.yaml
# ─────────────────────────────────────────────────────────────

class ParksCandidates(Strict):
    hyper_eye: Side
    candidates: List[str] = Field(min_length=1)


class ParksRow(Strict):
    hyper_eye: Side
    worse_on_gaze: Side
    worse_on_tilt: Side
    muscle: str = Field(pattern=EYE_MUSCLE_PATTERN)


class ParksInconclusive(Strict):
    reason: str


class ParksFile(Strict):
    step1: List[Entry[ParksCandidates]]
    rows: List[Entry[ParksRow]]
    inconclusive: Entry[ParksInconclusive]


# ─────────────────────────────────────────────────────────────
# conditions.yaml
# ─────────────────────────────────────────────────────────────

class Feature(Strict):
    """A measurable finding the rules engine computes; conditions cite these."""
    description: str
    params: Dict[str, List[str]] = Field(default_factory=dict)
    uses_thresholds: List[str] = Field(default_factory=list)
    measured_from: str


class FeatureRef(Strict):
    feature: str
    params: Dict[str, str] = Field(default_factory=dict)


class Condition(Strict):
    label: str
    kind: Literal["pattern", "red_flag"]
    urgency: Optional[Literal["URGENT"]] = None
    required: List[FeatureRef] = Field(default_factory=list)
    # Each group must have at least one of its features present.
    any_of: List[List[FeatureRef]] = Field(default_factory=list)
    supporting: List[FeatureRef] = Field(default_factory=list)
    report_wording: str

    @model_validator(mode="after")
    def _has_criteria(self) -> "Condition":
        if not self.required and not self.any_of:
            raise ValueError(f"condition {self.label!r} has no required features")
        if self.kind == "pattern" and self.urgency is not None:
            raise ValueError(f"condition {self.label!r}: only red flags carry an urgency")
        return self


class ConditionsFile(Strict):
    features: List[Entry[Feature]]
    conditions: List[Entry[Condition]]


# ─────────────────────────────────────────────────────────────
# screening.yaml — values the existing single-gaze screen uses
# ─────────────────────────────────────────────────────────────

class Icd10Code(Strict):
    code: str
    condition: str


class ScreeningFile(Strict):
    severity_thresholds: List[Scalar]
    icd10: List[Entry[Icd10Code]]
