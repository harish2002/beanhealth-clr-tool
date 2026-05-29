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
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from datetime import datetime, timezone

from models.response import (
    AnalyseResponse,
    ErrorResponse,
    InconclusiveResponse,
    SuccessResponse,
)
from models.stream_response import StreamAnalyseResponse
from pipeline.module_aggregate  import aggregate_frame_results
from pipeline.module_calibrate  import (
    CalibrationMeasurement,
    SessionCalibration,
    calibrate_from_measurements,
)
from pipeline.module1_detection    import detect_and_crop_eyes, detect_eyes_only_fallback
from pipeline.module2_pupil        import localise_pupils
from pipeline.module3_clr          import CLRResult, detect_clr
from pipeline.module4_displacement import compute_displacement
from pipeline.module5_asymmetry    import compute_asymmetry_and_angle
from pipeline.module6_classify     import classify_strabismus
from pipeline.module7_report       import generate_report
from utils.device_fingerprint      import parse_device_model
from utils.exceptions              import CLRPipelineError, DetectionError, CLRError

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

        # Module 5 — Bilateral asymmetry vector + Hirschberg angle
        asymmetry = compute_asymmetry_and_angle(
            left_displacement_norm=displacement.left_displacement_norm,
            right_displacement_norm=displacement.right_displacement_norm,
            upstream_flags=displacement.flags,
            # Pass raw vectors so Module 5 can compute the kappa-cancelling BAV
            left_dx=displacement.left_dx,
            left_dy=displacement.left_dy,
            right_dx=displacement.right_dx,
            right_dy=displacement.right_dy,
            left_iris_radius=pupil_result.left_iris_radius,
            right_iris_radius=pupil_result.right_iris_radius,
        )

        # Module 6 — Clinical classification
        # BAV direction: if |bav_nasal| > |bav_vertical|, primary axis is nasal/temporal
        # otherwise it's vertical — use this to pick the condition label
        dominant_eye = asymmetry.dominant_eye
        if abs(asymmetry.bav_nasal) >= abs(asymmetry.bav_vertical):
            # Horizontal strabismus — use dominant eye's nasal/temporal direction
            dominant_dir = (
                displacement.left_direction if dominant_eye != "right"
                else displacement.right_direction
            )
        else:
            # Vertical strabismus — use dominant eye's superior/inferior direction
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


def _pad_for_detection(img_rgb: np.ndarray, pad_ratio: float = 0.6) -> np.ndarray:
    """
    Add a neutral border around an image so MediaPipe FaceMesh can detect a
    face that fills (or nearly fills) the frame.

    MediaPipe's face detector expects the face to occupy only part of the
    frame — it routinely fails ("no_face") on tight head-and-shoulders or
    eyes-only crops where the face touches the image edges.  Many published
    strabismus reference photos are exactly that kind of close-up.  Adding
    margin around the crop gives the detector the surrounding context it
    needs without altering the eye geometry we measure.

    Args:
        img_rgb:    RGB uint8 image.
        pad_ratio:  Border thickness as a fraction of each dimension.

    Returns:
        A new, larger RGB image with the original centred inside a border.
    """
    h, w = img_rgb.shape[:2]
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    # Edge-replicate keeps skin-tone context around the face rather than a
    # hard black frame, which the detector tolerates better.
    return cv2.copyMakeBorder(
        img_rgb, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
    )


# ─────────────────────────────────────────────────────────────
# POST /analyse-stream  — multi-frame averaged analysis
# ─────────────────────────────────────────────────────────────

async def _run_single_frame_pipeline(
    img_rgb:      np.ndarray,
    patient_name: str,
    patient_age:  int,
    calibration:  Optional[SessionCalibration] = None,
    allow_eye_only_fallback: bool = False,
) -> tuple[dict, Optional[CLRResult]]:
    """
    Run the full 7-module pipeline on one frame.

    Args:
        img_rgb:      RGB numpy array of the captured frame.
        patient_name: Patient name (passed to report).
        patient_age:  Patient age  (passed to report).
        calibration:  Optional session-level calibration.  When present the
                      Module 3 thresholds are replaced with the calibrated values.

    Returns:
        (report_dict, clr_result)
        clr_result is None if the pipeline failed before Module 3 completed
        (used by the caller to collect calibration measurements).
    """
    clr_result_out: Optional[CLRResult] = None
    try:
        try:
            detection = detect_and_crop_eyes(img_rgb)
        except DetectionError as de:
            # In test mode, fall back to Hough-based eyes-only detection when
            # FaceMesh can't find a face (e.g. tightly-cropped research photos).
            if allow_eye_only_fallback and de.code in {
                "no_face", "eyes_not_visible", "not_frontal", "crop_too_small",
            }:
                logger.info(
                    f"[API] FaceMesh failed ({de.code}); trying eyes-only fallback"
                )
                detection = detect_eyes_only_fallback(img_rgb)
            else:
                raise
        pupil_result = localise_pupils(
            left_crop=detection.left_crop,
            right_crop=detection.right_crop,
            left_iris_landmarks_orig=detection.left_iris_landmarks,
            right_iris_landmarks_orig=detection.right_iris_landmarks,
            left_crop_box=detection.left_crop_box,
            right_crop_box=detection.right_crop_box,
        )

        # ── Module 3 — apply session-calibrated thresholds if available ──
        clr_kwargs: dict = dict(
            left_crop=detection.left_crop,
            right_crop=detection.right_crop,
            left_iris_radius=pupil_result.left_iris_radius,
            right_iris_radius=pupil_result.right_iris_radius,
            left_pupil=pupil_result.left_pupil,
            right_pupil=pupil_result.right_pupil,
        )
        if calibration and calibration.calibrated:
            clr_kwargs["min_area_ratio"]      = calibration.session_min_area_ratio
            clr_kwargs["max_area_ratio"]      = calibration.session_max_area_ratio
            clr_kwargs["min_peak_brightness"] = calibration.session_min_peak_brightness

        clr_result = detect_clr(**clr_kwargs)
        clr_result_out = clr_result   # expose for calibration collection

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
        asymmetry    = compute_asymmetry_and_angle(
            left_displacement_norm=displacement.left_displacement_norm,
            right_displacement_norm=displacement.right_displacement_norm,
            upstream_flags=displacement.flags,
            left_dx=displacement.left_dx,
            left_dy=displacement.left_dy,
            right_dx=displacement.right_dx,
            right_dy=displacement.right_dy,
            left_iris_radius=pupil_result.left_iris_radius,
            right_iris_radius=pupil_result.right_iris_radius,
        )
        dominant_dir = (
            displacement.left_direction if asymmetry.dominant_eye != "right"
            else displacement.right_direction
        )
        classification = classify_strabismus(
            dominant_direction=dominant_dir,
            severity=asymmetry.severity,
            asymmetry_score=asymmetry.asymmetry_score,
            upstream_flags=asymmetry.flags,
        )
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
        return report, clr_result_out
    except Exception as e:
        return (
            generate_report(
                patient_name=patient_name,
                patient_age=patient_age,
                original_img=None,
                error=e,
            ),
            clr_result_out,  # may be None if Module 3 never completed
        )


@app.post(
    "/analyse-stream",
    tags=["Analysis"],
    summary="Multi-frame averaged CLR analysis (streaming mode)",
    response_description="Aggregated SUCCESS / INCONCLUSIVE / ERROR report",
)
async def analyse_stream(
    images:       List[UploadFile] = File(..., description="List of JPEG frames captured over ~10 seconds"),
    patient_name: str              = Form(..., min_length=1, max_length=100),
    patient_age:  int              = Form(..., ge=1, le=120),
    user_agent:   str              = Form("", description="navigator.userAgent from the browser (used for session-level calibration)"),
) -> JSONResponse:
    """
    Accept N frames captured during a streaming session, run the full 7-module
    pipeline on each frame, reject bad frames (blinks, outliers, low confidence),
    and return a statistically averaged result.

    **Request:** `multipart/form-data` with:
    - `images[]`     — list of JPEG frames (typically 20, captured at 2 fps)
    - `patient_name` — patient's name
    - `patient_age`  — patient's age in years
    - `user_agent`   — browser's navigator.userAgent (optional, for device calibration)

    **Response 200 — SUCCESS:**
    Aggregated report with mean deviation ± std dev, per-frame readings,
    and intermediate images from the best-quality frame.

    **Response 200 — INCONCLUSIVE:**
    Fewer than 3 usable frames — user needs to retry with better positioning.
    """
    from typing import List as TList
    patient_name = patient_name.strip()
    timestamp    = datetime.now(timezone.utc).isoformat()

    # ── Device fingerprinting (Phase 1 calibration) ───────────
    device_model = parse_device_model(user_agent)
    logger.info(
        f"[API] /analyse-stream — patient='{patient_name}' age={patient_age} "
        f"frames={len(images)} device='{device_model}'"
    )

    if not images:
        return JSONResponse(
            content={
                "status": "INCONCLUSIVE",
                "reason": "no_frames",
                "reason_human": "No frames were received. Please try again.",
                "frames_total": 0, "frames_accepted": 0, "frames_rejected": 0,
                "per_frame_readings": [],
                "flags": ["no_frames"],
                "timestamp": timestamp,
            },
            status_code=200,
        )

    # ── Process frames sequentially with live calibration ─────
    # MediaPipe + OpenCV are CPU-bound and block the event loop —
    # asyncio.gather gives no speedup here and causes GIL contention.
    # Sequential processing is simpler and reliably within timeout.
    #
    # Session calibration strategy:
    #   • Frames 1–3: use default thresholds; collect CLR measurements.
    #   • After the 3rd successful frame: compute SessionCalibration.
    #   • Frames 4+: pass calibrated thresholds into detect_clr().
    frame_reports:    TList[dict]                  = []
    cal_measurements: TList[CalibrationMeasurement] = []
    calibration:      Optional[SessionCalibration]  = None

    for i, upload in enumerate(images):
        try:
            img_rgb = await _load_image(upload)
        except HTTPException:
            frame_reports.append({
                "status": "INCONCLUSIVE",
                "reason": "invalid_image",
                "reason_human": f"Frame {i} could not be decoded.",
            })
            continue

        report, clr_result = await _run_single_frame_pipeline(
            img_rgb, patient_name, patient_age, calibration=calibration
        )
        frame_reports.append(report)

        # ── Collect calibration measurements from successful frames ──
        if clr_result is not None and len(cal_measurements) < 3:
            cal_measurements.append(CalibrationMeasurement(
                peak_brightness_left=clr_result.left_peak_brightness,
                peak_brightness_right=clr_result.right_peak_brightness,
                blob_area_ratio_left=clr_result.left_blob_area_ratio,
                blob_area_ratio_right=clr_result.right_blob_area_ratio,
            ))
            # Compute calibration as soon as we have 3 good frames
            if len(cal_measurements) == 3 and calibration is None:
                calibration = calibrate_from_measurements(cal_measurements, device_model)
                logger.info(
                    f"[API] Session calibration ready after frame {i} — "
                    f"bloom={calibration.bloom_factor:.2f}× "
                    f"peak_threshold={calibration.session_min_peak_brightness:.1f}"
                )

        logger.debug(
            f"[API] Frame {i}: status={report.get('status')} "
            f"deviation={report.get('result', {}).get('deviation_degrees', 'N/A')} "
            f"calibrated={'yes' if (calibration and calibration.calibrated) else 'no'}"
        )

    # ── Aggregate across frames ───────────────────────────────
    try:
        aggregated = aggregate_frame_results(frame_reports)
    except Exception as e:
        logger.exception(f"[API] Aggregation crash: {e}")
        return JSONResponse(
            content={
                "status": "ERROR",
                "message": "An unexpected error occurred during aggregation. Please retry.",
                "patient": {"name": patient_name, "age": patient_age},
                "timestamp": timestamp,
            },
            status_code=500,
        )

    aggregated["patient"]   = {"name": patient_name, "age": patient_age}
    aggregated["timestamp"] = timestamp

    logger.info(
        f"[API] /analyse-stream DONE — status={aggregated['status']} "
        f"accepted={aggregated.get('frames_accepted', 0)}/{len(images)} "
        f"avg_deg={aggregated.get('deviation_avg_deg', 'N/A')} "
        f"std={aggregated.get('deviation_std_deg', 'N/A')} "
        f"confidence={aggregated.get('aggregate_confidence', 'N/A')}"
    )

    return JSONResponse(content=aggregated, status_code=200)


# ─────────────────────────────────────────────────────────────
# POST /analyse-test — development single-image analysis
#
# Bypasses the live-torch flash check so that the pipeline can be
# validated against saved research/clinical strabismus images that
# did not originate from a live phone-torch capture.
#
# THIS ENDPOINT MUST NOT BE USED FOR CLINICAL DECISIONS.
# ─────────────────────────────────────────────────────────────

@app.post(
    "/analyse-test",
    tags=["Analysis"],
    summary="DEV ONLY — single-image analysis with relaxed flash threshold",
    response_description="Single-frame report wrapped in StreamSuccessResponse shape",
)
async def analyse_test(
    image:        UploadFile = File(..., description="JPEG/PNG strabismus image (research dataset, case study, etc.)"),
    patient_name: str        = Form(..., min_length=1, max_length=100),
    patient_age:  int        = Form(..., ge=1, le=120),
) -> JSONResponse:
    """
    Development/research endpoint for validating the CLR pipeline against
    saved strabismus images that were NOT captured by a live phone torch.

    Differences from /analyse and /analyse-stream:
      • CLR_MIN_PEAK_BRIGHTNESS lowered from 235 → 180 — saved JPEGs and
        screen-rendered images cannot reproduce sensor-saturated highlights,
        so the standard flash check would reject every input.
      • Single-frame mode — one image goes through one pass of the pipeline.
        Inter-frame variance gating is bypassed (frames_total=1, std=0).
      • Response is wrapped in the StreamSuccessResponse shape so the
        existing TriageReport component renders it without modification.
      • Adds "test_capture" to technical.flags so the report shows a clear
        TEST CAPTURE banner — this result cannot be confused with a
        production screening.
    """
    patient_name = patient_name.strip()
    img_rgb      = await _load_image(image)

    logger.info(
        f"[API] /analyse-test — patient='{patient_name}' age={patient_age} "
        f"image={img_rgb.shape[1]}×{img_rgb.shape[0]}px (TEST MODE, no live torch)"
    )

    # Synthetic calibration: only override the flash threshold.  Area ratios
    # stay at the production defaults — a real CLR in the source image will
    # still be in the same relative-size range as a live capture.
    test_calibration = SessionCalibration(
        device_model="test-mode-upload",
        calibrated=True,
        calibration_frames=0,
        session_min_area_ratio=0.004,        # production CLR_MIN_AREA_RATIO
        session_max_area_ratio=0.25,         # production CLR_MAX_AREA_RATIO
        session_min_peak_brightness=180.0,   # ← the key relaxation
        bloom_factor=1.0,
        peak_brightness_avg=0.0,
    )

    # First pass: FaceMesh, then (if it fails) a Hough-based eyes-only fallback
    # for tightly-cropped research photos that have no full face to detect.
    report, _ = await _run_single_frame_pipeline(
        img_rgb, patient_name, patient_age,
        calibration=test_calibration, allow_eye_only_fallback=True,
    )

    # Second pass: for near-full-face crops where the face fills the frame,
    # border-padding gives FaceMesh the margin it needs. Re-run the whole
    # pipeline on the padded image (so annotation coords stay consistent).
    detection_failures = {"no_face", "eyes_not_visible", "not_frontal", "crop_too_small"}
    if report.get("status") != "SUCCESS" and report.get("reason") in detection_failures:
        logger.info(
            f"[API] /analyse-test — detection failed ({report.get('reason')}); "
            f"retrying on border-padded image"
        )
        padded = _pad_for_detection(img_rgb)
        retry_report, _ = await _run_single_frame_pipeline(
            padded, patient_name, patient_age,
            calibration=test_calibration, allow_eye_only_fallback=True,
        )
        # Only adopt the retry if it got further than the original attempt.
        if retry_report.get("status") == "SUCCESS" or retry_report.get("reason") not in detection_failures:
            report = retry_report

    timestamp = datetime.now(timezone.utc).isoformat()

    # ── INCONCLUSIVE / ERROR — return as-is with test_capture flag ──
    if report.get("status") != "SUCCESS":
        existing_flags = report.get("flags", []) or []
        report["flags"] = list(set(existing_flags + ["test_capture"]))
        report["patient"]   = {"name": patient_name, "age": patient_age}
        report["timestamp"] = timestamp
        # Provide the minimum multi-frame shape so the frontend INCONCLUSIVE
        # screen has something to render.
        report.setdefault("frames_total",       1)
        report.setdefault("frames_accepted",    0)
        report.setdefault("frames_rejected",    1)
        report.setdefault("per_frame_readings", [None])
        logger.warning(f"[API] /analyse-test INCONCLUSIVE — reason={report.get('reason', '?')}")
        return JSONResponse(content=report, status_code=200)

    # ── SUCCESS — wrap single-frame result in StreamSuccessResponse shape ──
    dev_deg  = report["result"]["deviation_degrees"]
    asym_deg = report["result"].get("asymmetry_degrees", 0.0)
    asym_score = report["result"]["asymmetry_score"]

    # Mark the technical flags so the frontend can show a TEST CAPTURE banner
    tech_flags = list(report.get("technical", {}).get("flags", []))
    if "test_capture" not in tech_flags:
        tech_flags.append("test_capture")
    report["technical"]["flags"] = tech_flags

    wrapped = {
        **report,
        "patient":              {"name": patient_name, "age": patient_age},
        "timestamp":            timestamp,
        # Multi-frame envelope (degenerate — N=1)
        "frames_total":         1,
        "frames_accepted":      1,
        "frames_rejected":      0,
        "per_frame_readings":   [dev_deg],
        "per_frame_rejections": [None],
        "deviation_avg_deg":    dev_deg,
        "deviation_std_deg":    0.0,
        "deviation_min_deg":    dev_deg,
        "deviation_max_deg":    dev_deg,
        "asymmetry_avg":        asym_score,
        "asymmetry_avg_deg":    asym_deg,
        "asymmetry_std_deg":    0.0,
        # Confidence is intentionally MEDIUM, not HIGH — single-frame analysis
        # has no inter-frame variance evidence, so confidence is bounded.
        "aggregate_confidence": "MEDIUM",
        # Extend result with deviation_std_deg for StreamClinicalResult shape
        "result": {**report["result"], "deviation_std_deg": 0.0},
    }

    logger.info(
        f"[API] /analyse-test DONE — urgency={wrapped['result']['urgency_tier']} "
        f"condition={wrapped['result']['condition_name']} "
        f"asym={asym_deg}° dev={dev_deg}°"
    )

    return JSONResponse(content=wrapped, status_code=200)
