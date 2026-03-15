"""
BeanHealth CLR Tool — Module 6: Clinical Classification
========================================================

Responsibility:
    Given the displacement direction, severity, and asymmetry score (from
    Modules 4 & 5), produce:

      1. A clinical condition name  (e.g. "Esotropia")
      2. An ICD-10 code             (e.g. "H50.01")
      3. A triage urgency tier      (URGENT / ROUTINE / MONITOR / NORMAL)
      4. A referral recommendation  (e.g. "Refer to ophthalmology within 1 week")
      5. A referral timeframe       (e.g. "1 week")
      6. A plain-English narrative  (for parents / non-clinical users)

Classification logic:
    Direction of CLR displacement → condition name + ICD-10 code
    Severity tier                 → urgency tier + referral text

    Special case: if severity is NORMAL, direction is ignored → Orthophoria.

ICD-10 codes used:
    H50.01  Esotropia   (nasal displacement — eye turns inward)
    H50.11  Exotropia   (temporal displacement — eye turns outward)
    H50.21  Hypertropia (superior displacement — eye turns upward)
    H50.22  Hypotropia  (inferior displacement — eye turns downward)
    H50.40  Orthophoria (no significant deviation)

Urgency mapping:
    SEVERE   → URGENT   → "Refer to ophthalmology within 1 week"
    MODERATE → ROUTINE  → "Refer to ophthalmology within 4 weeks"
    MILD     → MONITOR  → "Monitor and re-screen in 3 months"
    NORMAL   → NORMAL   → "No referral required"

Pipeline position:  SIXTH — depends on Modules 4 & 5.
Failure behaviour:  No failure modes — pure lookup, always returns a result.

Author: BeanHealth
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from utils.constants import (
    ICD10,
    SEVERITY_MILD,
    SEVERITY_MODERATE,
    SEVERITY_NORMAL,
    SEVERITY_SEVERE,
    DIRECTION_NASAL,
    DIRECTION_TEMPORAL,
    DIRECTION_SUPERIOR,
    DIRECTION_INFERIOR,
    URGENCY_URGENT,
    URGENCY_ROUTINE,
    URGENCY_MONITOR,
    URGENCY_NORMAL,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """
    Clinical classification result for the triage report.
    """
    condition_name:          str   # e.g. "Esotropia"
    icd10_code:              str   # e.g. "H50.01"
    urgency_tier:            str   # URGENT / ROUTINE / MONITOR / NORMAL
    referral_recommendation: str   # Full text recommendation
    timeframe:               str   # e.g. "1 week" / "N/A"
    narrative:               str   # Plain-English for parents
    flags:                   List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Classification tables
# ─────────────────────────────────────────────────────────────

# Direction → (condition_name, icd10_code)
_DIRECTION_TO_CONDITION = {
    DIRECTION_NASAL:    ("Esotropia",   ICD10["esotropia"]),
    DIRECTION_TEMPORAL: ("Exotropia",   ICD10["exotropia"]),
    DIRECTION_SUPERIOR: ("Hypertropia", ICD10["hypertropia"]),
    DIRECTION_INFERIOR: ("Hypotropia",  ICD10["hypotropia"]),
}

# Severity → (urgency_tier, referral_recommendation, timeframe)
_SEVERITY_TO_URGENCY = {
    SEVERITY_SEVERE: (
        URGENCY_URGENT,
        "Refer to ophthalmology within 1 week",
        "1 week",
    ),
    SEVERITY_MODERATE: (
        URGENCY_ROUTINE,
        "Refer to ophthalmology within 4 weeks",
        "4 weeks",
    ),
    SEVERITY_MILD: (
        URGENCY_MONITOR,
        "Monitor and re-screen in 3 months",
        "3 months",
    ),
    SEVERITY_NORMAL: (
        URGENCY_NORMAL,
        "No referral required",
        "N/A",
    ),
}

# Narrative templates keyed by urgency tier
_NARRATIVE_TEMPLATES = {
    URGENCY_URGENT: (
        "This screening detected a significant asymmetry in the corneal light reflex, "
        "suggesting a possible large-angle strabismus ({condition}). "
        "This requires prompt assessment by an ophthalmologist. "
        "Please seek a referral within 1 week."
    ),
    URGENCY_ROUTINE: (
        "This screening detected a moderate asymmetry in the corneal light reflex, "
        "which may indicate {condition}. "
        "An ophthalmology assessment is recommended within 4 weeks for a full evaluation."
    ),
    URGENCY_MONITOR: (
        "This screening detected a mild asymmetry in the corneal light reflex. "
        "This may be within the normal range, but monitoring is recommended. "
        "Please re-screen in 3 months or sooner if you notice any change in eye alignment."
    ),
    URGENCY_NORMAL: (
        "No significant asymmetry in the corneal light reflex was detected. "
        "The eye alignment appears normal for this screening. "
        "Continue routine eye health monitoring as advised by your healthcare provider."
    ),
}


# ─────────────────────────────────────────────────────────────
# Core classification — individually unit-testable
# ─────────────────────────────────────────────────────────────

def classify(
    direction: str,
    severity:  str,
) -> dict:
    """
    Map direction + severity to clinical condition and urgency.

    Args:
        direction: "nasal" | "temporal" | "superior" | "inferior"
                   (direction of CLR displacement of the dominant eye)
        severity:  "NORMAL" | "MILD" | "MODERATE" | "SEVERE"

    Returns:
        dict with keys:
            condition_name, icd10_code, urgency_tier,
            referral_recommendation, timeframe, narrative
    """
    # If NORMAL severity, direction is irrelevant → Orthophoria
    if severity == SEVERITY_NORMAL:
        condition_name = "Orthophoria"
        icd10_code     = ICD10["orthophoria"]
    else:
        condition_name, icd10_code = _DIRECTION_TO_CONDITION.get(
            direction,
            ("Strabismus, unspecified", "H50.9"),   # fallback for unknown direction
        )

    urgency_tier, referral_recommendation, timeframe = _SEVERITY_TO_URGENCY[severity]

    # Build narrative — substitute condition name where needed
    narrative = _NARRATIVE_TEMPLATES[urgency_tier].format(condition=condition_name)

    logger.debug(
        f"[M6] direction={direction}, severity={severity} → "
        f"{condition_name} ({icd10_code}), urgency={urgency_tier}"
    )

    return {
        "condition_name":          condition_name,
        "icd10_code":              icd10_code,
        "urgency_tier":            urgency_tier,
        "referral_recommendation": referral_recommendation,
        "timeframe":               timeframe,
        "narrative":               narrative,
    }


# ─────────────────────────────────────────────────────────────
# Public API — full Module 6 classification
# ─────────────────────────────────────────────────────────────

def classify_strabismus(
    dominant_direction: str,
    severity:           str,
    asymmetry_score:    float,
    upstream_flags:     Optional[List[str]] = None,
) -> ClassificationResult:
    """
    Full Module 6 classification.

    Args:
        dominant_direction: direction of CLR displacement of the dominant eye
        severity:           severity tier from Module 5
        asymmetry_score:    asymmetry score from Module 5 (for flag logic)
        upstream_flags:     flags carried forward from Modules 1–5

    Returns:
        ClassificationResult
    """
    flags: List[str] = list(upstream_flags or [])

    result_dict = classify(dominant_direction, severity)

    # ── Direction reliability check ───────────────────────────────────────
    # Suppress specific condition label (Esotropia/Exotropia etc.) when the
    # direction measurement is unreliable. This prevents opposite conditions
    # appearing on consecutive scans of the same patient.
    #
    # Triggers when EITHER:
    #   (a) A pupil_disagreement flag is present AND severity is MILD.
    #       At small angles (<15°) a shifted pupil centre can flip the
    #       displacement direction quadrant entirely.
    #   (b) asymmetry_score < 0.05 — both eyes nearly equally displaced;
    #       the "dominant direction" is statistical noise in this case.
    #
    # Safety guarantee: urgency_tier is NEVER changed by this check.
    # A MONITOR result stays MONITOR. Only the direction-derived label is
    # replaced with the safe neutral fallback (H50.9).
    has_pupil_disagreement = any("pupil_disagreement" in f for f in flags)
    low_asymmetry = asymmetry_score < 0.05

    direction_unreliable = (
        (has_pupil_disagreement and severity == SEVERITY_MILD) or
        low_asymmetry
    )

    if direction_unreliable and severity != SEVERITY_NORMAL:
        result_dict["condition_name"] = "Strabismus, Unspecified"
        result_dict["icd10_code"]     = "H50.9"
        flags.append("direction_unreliable")
        logger.info(
            "[M6] Direction suppressed → Strabismus, Unspecified (H50.9). "
            f"pupil_disagreement={has_pupil_disagreement}, "
            f"low_asymmetry={low_asymmetry} (score={asymmetry_score:.3f}), "
            f"severity={severity}"
        )

    # ── Flag borderline cases ─────────────────────────────────────────────
    if severity == SEVERITY_NORMAL and asymmetry_score > 0.05:
        flags.append("borderline_asymmetry")
        logger.info("[M6] Borderline asymmetry flagged — NORMAL but asymmetry_score > 0.05")

    result = ClassificationResult(
        condition_name=result_dict["condition_name"],
        icd10_code=result_dict["icd10_code"],
        urgency_tier=result_dict["urgency_tier"],
        referral_recommendation=result_dict["referral_recommendation"],
        timeframe=result_dict["timeframe"],
        narrative=result_dict["narrative"],
        flags=flags,
    )

    logger.info(
        f"[M6] {result.condition_name} ({result.icd10_code}) | "
        f"urgency={result.urgency_tier} | timeframe={result.timeframe}"
    )

    return result
