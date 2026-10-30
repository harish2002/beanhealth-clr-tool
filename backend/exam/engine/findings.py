"""
BeanHealth CLR — Engine Output Models
======================================

What the rules engine returns. Every finding carries its rule ID, status,
confidence, the evidence it used (value ± error, threshold, margin), the KB
entries behind it with their review status, and the KB and engine versions.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import Field

from exam.kb.schema import Eye
from exam.models import Confidence, Strict

Status = Literal["REPORTED", "INCONCLUSIVE"]
FindingKind = Literal["measurement", "pattern", "red_flag"]


class Reason(Strict):
    """Why a finding is INCONCLUSIVE. `repeatable`: a new capture could resolve it."""
    code: str
    detail: str
    repeatable: bool


class Evidence(Strict):
    quantity: str
    value: float
    error: float
    unit: str
    confidence: Confidence
    threshold: Optional[float] = None
    threshold_id: Optional[str] = None
    # (value − threshold) / error: how many errors clear of the threshold.
    margin_sigma: Optional[float] = None


class KBRef(Strict):
    id: str
    value: Any
    review_status: str


class FeatureOutcome(Strict):
    feature: str
    params: dict = Field(default_factory=dict)
    role: Literal["required", "any_of", "supporting"]
    state: Literal["PRESENT", "ABSENT", "INCONCLUSIVE"]
    detail: str


class Finding(Strict):
    rule_id: str
    label: str
    kind: FindingKind
    affected_eye: Optional[Eye] = None
    status: Status
    # True when a REPORTED finding describes something abnormal — the only
    # findings that can lead to a referral.
    abnormal: bool
    urgency: Optional[Literal["URGENT"]] = None
    confidence: Confidence
    # Smallest margin (in errors) among the comparisons that decided it; a
    # REPORTED finding's margin exceeds k.
    margin_over_threshold: Optional[float] = None
    wording: str
    reasons: List[Reason] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    features: List[FeatureOutcome] = Field(default_factory=list)
    muscle: Optional[str] = None              # Parks 3-step result, e.g. "R-SO"
    kb_entries: List[KBRef] = Field(default_factory=list)
    kb_version: str
    kb_sha256: str
    engine_version: str


class NotEvaluated(Strict):
    """
    A condition nothing could be said about: no feature present, and every
    gap is one a repeat test cannot fill (a KB threshold not set yet, a
    measurement not available). Listed once in the report rather than as an
    INCONCLUSIVE finding in every exam.
    """
    rule_id: str
    affected_eye: Optional[Eye] = None
    reasons: List[Reason]


class Referral(Strict):
    decision: Literal["REFER", "REPEAT_TEST", "NO_REFERRAL"]
    urgent: bool = False
    # An URGENT red flag was found, but only at LOW confidence: not a
    # referral, but the report must put it at the top.
    urgent_unconfirmed: bool = False
    because: List[str] = Field(default_factory=list)
    advice: str


class Quality(Strict):
    protocol: str
    insufficient_positions: List[str] = Field(default_factory=list)
    tilt_captured: bool


class EngineResult(Strict):
    findings: List[Finding]
    not_evaluated: List[NotEvaluated] = Field(default_factory=list)
    referral: Referral
    quality: Quality
    positioning: str
    limitations: List[str]
    kb_version: str
    kb_sha256: str
    engine_version: str
