"""
BeanHealth CLR — Examination Measurements
==========================================

The measurements the rules engine reads: a per-gaze deviation grid, head-tilt
captures and per-eye lid / pupil / duction measures. Milestone 6 fills these
from real frames; until then the synthetic generator (exam.sim) does.

Conventions (shared with the KB's conventions.yaml):
  • "R" / "L" and "right" / "left" are always the PATIENT's eye and side.
  • horizontal_pd  + = eso, − = exo
  • vertical_pd    + = right hyper, − = left hyper
  • Units are prism dioptres (PD); degrees are derived (PD = 100·tan θ).

Every number is a `Measured`: value, error and confidence. `error` is the
standard error of the fused value (one sigma, same unit as the value); the
engine compares differences against k × combined error.

Known limitations (brief §11), carried into every report: near fixation only
(no 6 m distance); no phoria / intermittent detection in Tier 1; torsion not
measured; thresholds unvalidated at near; paralytic vs restrictive not
separable in Tier 1; fixation compliance in children not guaranteed.
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from exam.kb.schema import Eye, Gaze

Confidence = Literal["HIGH", "MODERATE", "LOW"]
CONFIDENCE_ORDER: Dict[str, int] = {"LOW": 0, "MODERATE": 1, "HIGH": 2}

Protocol = Literal["nine_gaze", "five_gaze"]
PROTOCOL_POSITIONS: Dict[str, List[str]] = {
    "nine_gaze": ["primary", "right", "left", "up", "down",
                  "up_right", "up_left", "down_right", "down_left"],
    # Kids mode (brief §4.1).
    "five_gaze": ["primary", "right", "left", "up", "down"],
}

DuctionDirection = Literal["abduction", "adduction", "elevation", "depression"]


def pd_to_degrees(pd: float) -> float:
    return math.degrees(math.atan(pd / 100.0))


def degrees_to_pd(degrees: float) -> float:
    return 100.0 * math.tan(math.radians(degrees))


def lowest_confidence(levels: List[str]) -> Confidence:
    if not levels:
        return "LOW"
    return min(levels, key=lambda c: CONFIDENCE_ORDER[c])  # type: ignore[return-value]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Measured(Strict):
    value: float
    error: float = Field(gt=0, description="standard error, same unit as value")
    confidence: Confidence = "HIGH"


class GazeCell(Strict):
    status: Literal["OK", "INSUFFICIENT_DATA"] = "OK"
    horizontal_pd: Optional[Measured] = None
    vertical_pd: Optional[Measured] = None
    valid_frames: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _ok_has_values(self) -> "GazeCell":
        if self.status == "OK" and (self.horizontal_pd is None or self.vertical_pd is None):
            raise ValueError("an OK gaze cell needs horizontal_pd and vertical_pd")
        return self


class GazeGrid(Strict):
    """One cell per protocol position; a position without enough valid frames is INSUFFICIENT_DATA."""
    protocol: Protocol
    cells: Dict[Gaze, GazeCell]

    @model_validator(mode="after")
    def _cells_match_protocol(self) -> "GazeGrid":
        want = set(PROTOCOL_POSITIONS[self.protocol])
        if set(self.cells) != want:
            raise ValueError(
                f"{self.protocol} grid needs exactly {sorted(want)}, got {sorted(self.cells)}"
            )
        return self


class TiltMeasurements(Strict):
    """Vertical deviation (PD, + = right hyper) on head tilt toward the patient's right / left."""
    right: Optional[Measured] = None
    left: Optional[Measured] = None


class EyeMeasures(Strict):
    # Excursion in each direction as a fraction of what the head pose implies.
    ductions: Dict[DuctionDirection, Measured] = Field(default_factory=dict)
    fissure_height_mm: Optional[Measured] = None          # primary gaze
    fissure_ratio_adduction: Optional[Measured] = None    # adducting gaze ÷ primary
    pupil_diameter_mm: Optional[Measured] = None
    scleral_show_mm: Optional[Measured] = None            # upper lid to limbus


class ExamMeasurements(Strict):
    grid: GazeGrid
    tilt: Optional[TiltMeasurements] = None
    eyes: Dict[Eye, EyeMeasures] = Field(default_factory=dict)
