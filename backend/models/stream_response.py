"""
BeanHealth CLR Tool — Stream API Response Schema
=================================================

Pydantic v2 models for the POST /analyse-stream endpoint.

The stream endpoint accepts N frames (captured over 10 seconds),
runs the full pipeline on each, aggregates the results, and returns
a statistically robust average with a confidence score.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from models.response import (
    ClinicalResult,
    IntermediateImages,
    PatientInfo,
    TechnicalDetail,
)


class StreamClinicalResult(ClinicalResult):
    """ClinicalResult extended with multi-frame statistical fields."""

    deviation_std_deg: float = Field(
        description="Standard deviation of deviation across accepted frames. "
                    "Lower = more consistent = higher confidence."
    )


class StreamSuccessResponse(BaseModel):
    """
    Returned when the streaming pipeline produced enough accepted frames.
    """

    status: Literal["SUCCESS"]
    patient: PatientInfo

    # ── Multi-frame statistics ──────────────────────────────
    frames_total:    int   = Field(description="Total frames captured.")
    frames_accepted: int   = Field(description="Frames used in the average (passed quality + outlier checks).")
    frames_rejected: int   = Field(description="Frames discarded (blinks, low confidence, outliers).")

    per_frame_readings: List[Optional[float]] = Field(
        description="Deviation reading for each frame in order. None = rejected frame."
    )

    deviation_avg_deg: float = Field(description="Mean deviation across accepted frames (degrees).")
    deviation_std_deg: float = Field(description="Standard deviation of deviation (degrees). Low = stable.")
    deviation_min_deg: float = Field(description="Minimum accepted deviation (degrees).")
    deviation_max_deg: float = Field(description="Maximum accepted deviation (degrees).")
    asymmetry_avg:     float = Field(description="Mean asymmetry score across accepted frames.")

    aggregate_confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Confidence based on std dev: HIGH < 1°, MEDIUM < 3°, LOW >= 3°."
    )

    # ── Clinical result (based on averaged deviation) ───────
    result:    StreamClinicalResult
    technical: TechnicalDetail

    # ── Images from the best-quality frame ──────────────────
    intermediate_images: Optional[IntermediateImages] = None
    annotated_image_b64: Optional[str]               = None

    timestamp: str


class StreamInconclusiveResponse(BaseModel):
    """
    Returned when fewer than MIN_ACCEPTED_FRAMES passed quality checks.
    """

    status:       Literal["INCONCLUSIVE"]
    reason:       str
    reason_human: str
    patient:      Optional[PatientInfo] = None

    frames_total:    int
    frames_accepted: int
    frames_rejected: int
    per_frame_readings: List[Optional[float]]

    flags:     List[str] = Field(default_factory=list)
    timestamp: str


class StreamErrorResponse(BaseModel):
    """Unexpected crash during stream processing."""

    status:    Literal["ERROR"]
    message:   str
    patient:   Optional[PatientInfo] = None
    timestamp: str


StreamAnalyseResponse = Union[
    StreamSuccessResponse,
    StreamInconclusiveResponse,
    StreamErrorResponse,
]
