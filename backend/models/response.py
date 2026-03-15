"""
BeanHealth CLR Tool — API Response Schema
==========================================

Pydantic v2 models for the /analyse endpoint response.

Three possible top-level shapes:

  SuccessResponse      status="SUCCESS"
      Full triage report with result, technical detail, annotated image.

  InconclusiveResponse status="INCONCLUSIVE"
      Pipeline halted (no flash, no face, etc.) — reason + flags only.
      No triage result is included.

  ErrorResponse        status="ERROR"
      Unexpected internal crash — generic message only.
      No triage result, no raw traceback.

The API always returns HTTP 200 for SUCCESS and INCONCLUSIVE.
HTTP 422 is returned by FastAPI automatically for malformed requests.
HTTP 500 is reserved for truly catastrophic server failures.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Nested sub-models
# ─────────────────────────────────────────────────────────────

class PatientInfo(BaseModel):
    name: str
    age:  int


class ClinicalResult(BaseModel):
    """The triage decision output — present only in SUCCESS responses."""

    urgency_tier: Literal["URGENT", "ROUTINE", "MONITOR", "NORMAL"] = Field(
        description="Triage tier — drives referral urgency."
    )
    condition_name: str = Field(
        description="Clinical condition name, e.g. 'Esotropia'."
    )
    icd10_code: str = Field(
        description="ICD-10 code, e.g. 'H50.01'."
    )
    deviation_degrees: float = Field(
        description="Hirschberg deviation angle in degrees."
    )
    asymmetry_score: float = Field(
        description="Normalised asymmetry score (0 = symmetric, higher = more asymmetric)."
    )
    severity: Literal["NORMAL", "MILD", "MODERATE", "SEVERE"] = Field(
        description="Severity tier derived from deviation angle."
    )
    referral_recommendation: str = Field(
        description="Plain-text referral instruction."
    )
    timeframe: str = Field(
        description="Recommended timeframe for action, e.g. '4 weeks' or 'N/A'."
    )
    narrative: str = Field(
        description="Plain-English explanation for non-clinical users (parents/GPs)."
    )


class TechnicalDetail(BaseModel):
    """Raw CV measurements — for clinical review and debugging."""

    left_pupil:               List[float] = Field(description="(x, y) left pupil centre in crop pixels.")
    right_pupil:              List[float] = Field(description="(x, y) right pupil centre in crop pixels.")
    left_clr:                 List[float] = Field(description="(x, y) left CLR position in crop pixels.")
    right_clr:                List[float] = Field(description="(x, y) right CLR position in crop pixels.")
    left_displacement_norm:   float       = Field(description="Left CLR displacement / iris radius.")
    right_displacement_norm:  float       = Field(description="Right CLR displacement / iris radius.")
    left_direction:           str         = Field(description="Anatomical direction of left CLR displacement.")
    right_direction:          str         = Field(description="Anatomical direction of right CLR displacement.")
    deviation_mm:             float       = Field(description="Dominant-eye displacement in mm.")
    dominant_eye:             str         = Field(description="Which eye has the larger displacement.")
    confidence:               Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Overall pupil-localisation confidence (worst-case of both eyes)."
    )
    flags:                    List[str]   = Field(
        default_factory=list,
        description="Non-fatal warnings accumulated from all pipeline modules."
    )


class IntermediateImages(BaseModel):
    """Six-step pipeline visualisation — one image per processing stage."""

    module1_crops:  str = Field(description="Step 1: Raw eye crops extracted by MediaPipe face mesh.")
    module2_clahe:  str = Field(description="Step 2: Grayscale + CLAHE contrast enhancement — makes CLR visible.")
    module3_pupil:  str = Field(description="Step 3: Pupil centre localised (blue dot + iris ring).")
    module4_clr:    str = Field(description="Step 4: Corneal light reflex detected (amber dot).")
    module5_vector: str = Field(description="Step 5: Displacement vector drawn from pupil to CLR, with measurement.")
    module6_result: str = Field(description="Step 6: Final annotated image with deviation angle and clinical summary.")


# ─────────────────────────────────────────────────────────────
# Top-level response models
# ─────────────────────────────────────────────────────────────

class SuccessResponse(BaseModel):
    """
    Returned when all 7 pipeline modules completed successfully.

    HTTP 200.
    """

    status:   Literal["SUCCESS"]
    patient:  PatientInfo
    result:   ClinicalResult
    technical: TechnicalDetail
    intermediate_images: Optional[IntermediateImages] = Field(
        default=None, 
        description="Zoomed-in progress images from pipeline steps."
    )
    annotated_image_b64: str = Field(
        description="Base64-encoded JPEG of the original image with landmarks overlaid."
    )
    timestamp: str = Field(
        description="ISO 8601 UTC timestamp of when the analysis was performed."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "SUCCESS",
                    "patient": {"name": "Emma Wilson", "age": 5},
                    "result": {
                        "urgency_tier": "ROUTINE",
                        "condition_name": "Esotropia",
                        "icd10_code": "H50.01",
                        "deviation_degrees": 13.4,
                        "asymmetry_score": 0.3321,
                        "severity": "MODERATE",
                        "referral_recommendation": "Refer to ophthalmology within 4 weeks",
                        "timeframe": "4 weeks",
                        "narrative": "Moderate asymmetry detected...",
                    },
                    "technical": {
                        "left_pupil": [125.0, 60.0],
                        "right_pupil": [125.0, 60.0],
                        "left_clr": [135.0, 60.0],
                        "right_clr": [115.0, 60.0],
                        "left_displacement_norm": 0.333,
                        "right_displacement_norm": 0.333,
                        "left_direction": "nasal",
                        "right_direction": "nasal",
                        "deviation_mm": 1.914,
                        "dominant_eye": "equal",
                        "confidence": "HIGH",
                        "flags": [],
                    },
                    "intermediate_images": {
                        "module1_crops": "/9j/4AAQ...",
                        "module2_pupil": "/9j/4AAQ...",
                        "module3_clr": "/9j/4AAQ...",
                        "module4_vector": "/9j/4AAQ..."
                    },
                    "annotated_image_b64": "/9j/4AAQ...",
                    "timestamp": "2026-03-14T10:00:00+00:00",
                }
            ]
        }
    }


class InconclusiveResponse(BaseModel):
    """
    Returned when the pipeline was halted by a known, recoverable condition
    (e.g. no flash detected, no face in frame, eyes closed).

    HTTP 200 — the request was valid, but no triage result can be given.
    """

    status:        Literal["INCONCLUSIVE"]
    reason:        str = Field(description="Machine-readable reason code, e.g. 'no_flash'.")
    reason_human:  str = Field(description="Plain-English explanation for display in the UI.")
    flags:         List[str] = Field(default_factory=list)
    patient:       Optional[PatientInfo] = None
    timestamp:     str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "INCONCLUSIVE",
                    "reason": "no_flash",
                    "reason_human": "No torch/flash detected. Enable the torch and retry.",
                    "flags": ["no_flash"],
                    "patient": {"name": "Emma Wilson", "age": 5},
                    "timestamp": "2026-03-14T10:00:00+00:00",
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """
    Returned when an unexpected internal error crashed the pipeline.

    HTTP 500. Raw traceback is never exposed — it is logged server-side only.
    """

    status:    Literal["ERROR"]
    message:   str = Field(description="Generic user-safe error message.")
    patient:   Optional[PatientInfo] = None
    timestamp: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ERROR",
                    "message": "An unexpected error occurred. Please retry.",
                    "timestamp": "2026-03-14T10:00:00+00:00",
                }
            ]
        }
    }


# ─────────────────────────────────────────────────────────────
# Union type for OpenAPI schema
# ─────────────────────────────────────────────────────────────

AnalyseResponse = Union[SuccessResponse, InconclusiveResponse, ErrorResponse]
