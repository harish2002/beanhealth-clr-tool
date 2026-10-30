"""
BeanHealth CLR — Frame landmark extraction (brief §4.3)
=======================================================

One video frame in, one FrameObservation out — the record a capture session
carries. This is the server-side reference for what the phone computes per
frame; the phone uploads the same structure and never the frame itself.

    face mesh (478 landmarks, refine_landmarks) → head pose
    per eye: lid aperture → limbus ellipse → reflex (module 3, then sub-pixel)
             → pupil → fissure, scleral show, limbus coverage → sharpness

An eye value that cannot be measured is left null for frame QC to reject;
nothing is guessed.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from exam.capture.session import EyeObservation, FrameObservation
from exam.kb import load_kb
from exam.kb.loader import KnowledgeBase
from exam.kb.schema import Eye
from exam.landmarks.eye import (
    LidAperture,
    fit_limbus,
    find_pupil,
    lid_metrics,
    refine_reflex,
)
from exam.landmarks.head_pose import head_pose
from exam.qc import sharpness
from pipeline.module3_clr import _detect_clr_one_eye
from utils.exceptions import CLRError

logger = logging.getLogger(__name__)

# Face-mesh indices per PATIENT eye (the right eye is on the image left).
IRIS = {"R": (468, 469, 470, 471, 472), "L": (473, 474, 475, 476, 477)}
UPPER_LID = {"R": (33, 246, 161, 160, 159, 158, 157, 173, 133),
             "L": (362, 398, 384, 385, 386, 387, 388, 466, 263)}
LOWER_LID = {"R": (33, 7, 163, 144, 145, 153, 154, 155, 133),
             "L": (362, 382, 381, 380, 374, 373, 390, 249, 263)}
CANTHI = {"R": (33, 133), "L": (362, 263)}       # (image-left, image-right) on an upright head
_LABEL = {"R": "right", "L": "left"}


def face_landmarks(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    """478 × (x_px, y_px, z) with z scaled like x, or None when no face (or no iris) is found."""
    h, w = image_rgb.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                         refine_landmarks=True, min_detection_confidence=0.5) as mesh:
        res = mesh.process(image_rgb)
    if not res.multi_face_landmarks:
        return None
    lms = res.multi_face_landmarks[0].landmark
    if len(lms) < 478:
        return None
    return np.array([(p.x * w, p.y * h, p.z * w) for p in lms], dtype=float)


def aperture_for(points: np.ndarray, eye: Eye) -> LidAperture:
    def pts(idx: Tuple[int, ...]) -> Tuple[Tuple[float, float], ...]:
        return tuple((float(points[i, 0]), float(points[i, 1])) for i in idx)
    a, b = pts(CANTHI[eye])
    return LidAperture(upper=pts(UPPER_LID[eye]), lower=pts(LOWER_LID[eye]), canthi=(a, b))


def _coarse_reflex(image_rgb: np.ndarray, centre: Tuple[float, float], radius: float,
                   eye: Eye) -> Optional[Tuple[float, float]]:
    """Module 3's reflex detection (percentile threshold + 3-way filter) on a crop round the iris."""
    h, w = image_rgb.shape[:2]
    # floor(v + 0.5) rounds halves up, as the phone does (Python's round() goes to even).
    half = math.floor(2 * radius + 0.5)
    cx, cy = math.floor(centre[0] + 0.5), math.floor(centre[1] + 0.5)
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, cx + half + 1), min(h, cy + half + 1)
    crop = np.ascontiguousarray(image_rgb[y0:y1, x0:x1])
    flags: List[str] = []
    try:
        (rx, ry), *_ = _detect_clr_one_eye(crop, radius, (centre[0] - x0, centre[1] - y0), _LABEL[eye], flags)
    except CLRError as e:
        logger.debug("exam landmarks [%s]: no reflex (%s)", eye, e.code)
        return None
    return (rx + x0, ry + y0)


def eye_observation(image_rgb: np.ndarray, gray: np.ndarray, points: np.ndarray, eye: Eye,
                    kb: KnowledgeBase) -> Tuple[EyeObservation, Optional[float]]:
    """The eye's observation and its sharpness (None when it cannot be measured)."""
    iris = points[list(IRIS[eye]), :2]
    centre_guess = (float(iris[0, 0]), float(iris[0, 1]))
    radius_guess = float(np.mean(np.hypot(*(iris[1:] - iris[0]).T)))
    aperture = aperture_for(points, eye)
    fit = fit_limbus(gray, centre_guess, radius_guess, aperture)
    limbus = fit.ellipse if fit else None
    centre = limbus.centre_px if limbus else centre_guess
    radius = limbus.semi_major_px if limbus else radius_guess

    reflex = _coarse_reflex(image_rgb, centre, radius, eye)
    if reflex is not None:
        reflex = refine_reflex(gray, reflex, radius)

    obs: Dict[str, object] = {"reflex_px": reflex, "limbus": limbus, "iris_radius_px": radius}
    if limbus is not None:
        pupil = find_pupil(gray, limbus, reflex)
        if pupil is not None:
            obs["pupil_px"], obs["pupil_diameter_px"] = pupil
        lids = lid_metrics(aperture, limbus)
        obs.update(fissure_height_px=lids.fissure_height_px, scleral_show_px=lids.scleral_show_px,
                   limbus_covered_fraction=lids.limbus_covered_fraction)
    try:
        sharp: Optional[float] = sharpness(gray, centre, radius, kb)
    except ValueError:
        sharp = None
    return EyeObservation(**obs), sharp           # type: ignore[arg-type]


def extract_frame(image_rgb: np.ndarray, index: int = 0, t_ms: float = 0.0,
                  kb: Optional[KnowledgeBase] = None) -> FrameObservation:
    """Landmarks of one unmirrored RGB frame. No face → a frame with only index and time."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("extract_frame needs an RGB image")
    kb = kb or load_kb()
    points = face_landmarks(image_rgb)
    if points is None:
        return FrameObservation(index=index, t_ms=t_ms)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    eyes, sharps = {}, []
    for eye in ("R", "L"):
        eyes[eye], s = eye_observation(image_rgb, gray, points, eye, kb)   # type: ignore[arg-type]
        sharps.append(s)
    # The blurrier eye decides (QC compares frames by their worst eye).
    frame_sharpness = None if any(s is None for s in sharps) else min(sharps)   # type: ignore[type-var]
    return FrameObservation(index=index, t_ms=t_ms, head_pose=head_pose(points),
                            sharpness=frame_sharpness, eyes=eyes)
