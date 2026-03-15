"""
BeanHealth CLR Tool — FastAPI Entry Point
==========================================

Routes:
    POST /analyse   — Accept eye photo + patient info, run full CLR pipeline,
                      return SUCCESS / INCONCLUSIVE / ERROR report.
    GET  /health    — Liveness check.
    GET  /          — API root (redirects to /docs).

Error handling:
    DetectionError / CLRError  → HTTP 200, status=INCONCLUSIVE
    PipelineError              → HTTP 500, status=ERROR
    Unexpected exception       → HTTP 500, status=ERROR
    Pydantic validation error  → HTTP 422 (FastAPI built-in)

All raw tracebacks are logged server-side and never sent to the client.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from models.response import (
    AnalyseResponse,
    ErrorResponse,
    InconclusiveResponse,
    SuccessResponse,
)
from pipeline.module1_detection    import detect_and_crop_eyes
from pipeline.module2_pupil        import localise_pupils
from pipeline.module3_clr          import detect_clr
from pipeline.module4_displacement import compute_displacement
from pipeline.module5_asymmetry    import compute_asymmetry_and_angle
from pipeline.module6_classify     import classify_strabismus
from pipeline.module7_report       import generate_report
from utils.exceptions import CLRPipelineError, DetectionError, CLRError

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="BeanHealth CLR Tool API",
    description=(
        "Corneal Light Reflex (CLR) asymmetry analysis for strabismus triage screening. "
        "Upload a torch-lit eye photo and get a triage tier (URGENT / ROUTINE / MONITOR / NORMAL)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # open for hackathon; tighten post-launch
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ─────────────────────────────────────────────────────────────
# Image loading helper
# ─────────────────────────────────────────────────────────────

_MAX_IMAGE_PIXELS = 4000 * 3000   # ~12 MP cap (performance guard)


async def _load_image(upload: UploadFile) -> np.ndarray:
    """
    Read the uploaded file and return an RGB numpy array.

    Raises:
        HTTPException 422 — if the file is not a valid image.
    """
    raw_bytes = await upload.read()

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is not a valid image. Please upload a JPEG or PNG.",
        )

    if pil_img.width * pil_img.height > _MAX_IMAGE_PIXELS:
        # Downscale to fit within cap — preserves aspect ratio
        pil_img.thumbnail((3000, 2000), Image.LANCZOS)
        logger.info(f"[API] Image downscaled to {pil_img.size} (exceeded pixel cap)")

    return np.array(pil_img, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health() -> dict:
    """Liveness check — returns 200 OK if the server is running."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/", tags=["Meta"])
def root() -> dict:
    """API root — visit /docs for the interactive OpenAPI UI."""
    return {"message": "BeanHealth CLR API — visit /docs for OpenAPI UI"}


@app.post(
    "/analyse",
    tags=["Analysis"],
    summary="Analyse a corneal light reflex photo",
    response_description="SUCCESS, INCONCLUSIVE, or ERROR report",
)
async def analyse(
    image:        UploadFile = File(..., description="JPEG or PNG eye photo with torch enabled"),
    patient_name: str        = Form(..., min_length=1, max_length=100, description="Patient's full name"),
    patient_age:  int        = Form(..., ge=1, le=120, description="Patient's age in years"),
) -> JSONResponse:
    """
    Run the full 7-module CLR analysis pipeline on the uploaded image.

    **Request:** `multipart/form-data` with:
    - `image`        — torch-lit eye photo (JPEG / PNG)
    - `patient_name` — patient's name (string, required)
    - `patient_age`  — patient's age in years (integer 1–120, required)

    **Response 200 — SUCCESS:**
    Full triage report with urgency tier, ICD-10 code, referral recommendation,
    and base64-encoded annotated image.

    **Response 200 — INCONCLUSIVE:**
    Pipeline was halted (e.g. no flash, no face detected).
    Includes machine-readable reason and plain-English explanation.
    No triage result is included.

    **Response 422:** Malformed request (missing fields, wrong types).

    **Response 500:** Unexpected internal error (traceback logged server-side only).
    """
    patient_name = patient_name.strip()

    # ── Load image ───────────────────────────────────────────
    img_rgb = await _load_image(image)
    logger.info(
        f"[API] /analyse — patient='{patient_name}' age={patient_age} "
        f"image={img_rgb.shape[1]}×{img_rgb.shape[0]}px"
    )

    # ── Run pipeline ─────────────────────────────────────────
    try:
        # Module 1 — Eye detection & crop
        detection = detect_and_crop_eyes(img_rgb)

        # Module 2 — Pupil centre localisation
        pupil_result = localise_pupils(
            left_crop=detection.left_crop,
            right_crop=detection.right_crop,
            left_iris_landmarks_orig=detection.left_iris_landmarks,
            right_iris_landmarks_orig=detection.right_iris_landmarks,
            left_crop_box=detection.left_crop_box,
            right_crop_box=detection.right_crop_box,
        )

        # Module 3 — CLR bright spot detection
        clr_result = detect_clr(
            left_crop=detection.left_crop,
            right_crop=detection.right_crop,
            left_iris_radius=pupil_result.left_iris_radius,
            right_iris_radius=pupil_result.right_iris_radius,
            left_pupil=pupil_result.left_pupil,
            right_pupil=pupil_result.right_pupil,
        )

        # Module 4 — Displacement measurement
        upstream_flags = pupil_result.flags + clr_result.flags
        displacement = compute_displacement(
            left_pupil=pupil_result.left_pupil,
            right_pupil=pupil_result.right_pupil,
            left_clr=clr_result.left_clr,
            right_clr=clr_result.right_clr,
            left_iris_radius=pupil_result.left_iris_radius,
            right_iris_radius=pupil_result.right_iris_radius,
            upstream_flags=upstream_flags,
        )

        # Module 5 — Asymmetry score + Hirschberg angle
        asymmetry = compute_asymmetry_and_angle(
            left_displacement_norm=displacement.left_displacement_norm,
            right_displacement_norm=displacement.right_displacement_norm,
            upstream_flags=displacement.flags,
        )

        # Module 6 — Clinical classification
        # Use the dominant eye's direction for the condition label
        dominant_eye = asymmetry.dominant_eye
        dominant_dir = (
            displacement.left_direction if dominant_eye != "right"
            else displacement.right_direction
        )
        classification = classify_strabismus(
            dominant_direction=dominant_dir,
            severity=asymmetry.severity,
            asymmetry_score=asymmetry.asymmetry_score,
            upstream_flags=asymmetry.flags,
        )

        # Module 7 — Report + annotated image
        report = generate_report(
            patient_name=patient_name,
            patient_age=patient_age,
            original_img=img_rgb,
            detection=detection,
            pupil_result=pupil_result,
            clr_result=clr_result,
            displacement=displacement,
            asymmetry=asymmetry,
            classification=classification,
        )

        logger.info(
            f"[API] SUCCESS — urgency={report['result']['urgency_tier']} "
            f"condition={report['result']['condition_name']} "
            f"angle={report['result']['deviation_degrees']}°"
        )
        return JSONResponse(content=report, status_code=200)

    # ── Known pipeline halts → INCONCLUSIVE (HTTP 200) ───────
    except (DetectionError, CLRError, CLRPipelineError) as e:
        report = generate_report(
            patient_name=patient_name,
            patient_age=patient_age,
            original_img=None,
            error=e,
        )
        logger.warning(f"[API] INCONCLUSIVE — {e.code}: {e.human_message}")
        return JSONResponse(content=report, status_code=200)

    # ── Unexpected crashes → ERROR (HTTP 500) ─────────────────
    except Exception as e:
        report = generate_report(
            patient_name=patient_name,
            patient_age=patient_age,
            original_img=None,
            error=e,
        )
        logger.exception(f"[API] Unexpected pipeline crash: {e}")
        return JSONResponse(content=report, status_code=500)
