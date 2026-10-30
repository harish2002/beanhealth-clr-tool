"""
BeanHealth CLR — Capture Protocol (brief §4.1)
===============================================

What the phone is asked to capture, and the head pose that reaches each
position. The phone fetches this plan rather than hard-coding it, so the
targets always match the KB version that will judge the frames.

Gaze method: head rotation. The patient keeps fixating a target on the phone
while the operator turns their head; the eyes turn the opposite way in the
orbit. To put the eyes at eye-in-orbit angle (h, v) the head turns to
(−h, −v).

Head-pose convention (face mesh, relative to the camera, phone upright):
  yaw    + = face turned toward the PATIENT's right
  pitch  + = chin up
  roll   + = head tilted toward the patient's right shoulder
Eye-in-orbit angles use the same signs: + horizontal = the patient's right,
+ vertical = up.

Calibration (brief §4.4) comes first: with one eye covered, the other fixes
the light at the lens (its kappa: where the reflex sits when the eye looks
straight at the light), then a target to the patient's right and one to the
left at calibration_target_offset_cm (its Hirschberg ratio: how far the
reflex moves per prism dioptre of eye rotation). Both eyes take a turn.

Known limitations (brief §11): near fixation only; fixation compliance in
children is not guaranteed — the QC rejects frames with a saccade, but cannot
prove a child was looking at the target between them.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import Field

from exam.kb import load_kb
from exam.kb.loader import KnowledgeBase
from exam.kb.schema import Eye
from exam.models import PROTOCOL_POSITIONS, Protocol, Strict

CAPTURE_PROTOCOL_VERSION = "1.1.0"

TiltPosition = Literal["tilt_right", "tilt_left"]
TILT_POSITIONS: Tuple[str, ...] = ("tilt_right", "tilt_left")

# (fixing eye, target) for each calibration position; the other eye is covered.
# target: "lens" or the side of the lens the card target sits on.
CALIBRATION_TARGETS: Dict[str, Tuple[Eye, str]] = {
    f"calib_{eye}_{target}": (eye, target)            # type: ignore[misc]
    for eye in ("R", "L") for target in ("lens", "right", "left")
}
CALIBRATION_POSITIONS: Tuple[str, ...] = tuple(CALIBRATION_TARGETS)
_OTHER_EYE = {"R": "L", "L": "R"}
_EYE_WORD = {"R": "right", "L": "left"}

# Unit eye-in-orbit direction of each gaze position: (horizontal, vertical),
# + = patient's right, up.
GAZE_DIRECTION: Dict[str, Tuple[int, int]] = {
    "primary": (0, 0),
    "right": (1, 0), "left": (-1, 0),
    "up": (0, 1), "down": (0, -1),
    "up_right": (1, 1), "up_left": (-1, 1),
    "down_right": (1, -1), "down_left": (-1, -1),
}

_SIDE_WORD = {1: "right", -1: "left"}
_VERTICAL_WORD = {1: "up", -1: "down"}


class Range(Strict):
    min: float
    max: float


class HeadPoseTarget(Strict):
    """Accepted head pose for a position. A frame outside it is rejected by QC."""
    yaw_deg: float
    pitch_deg: float
    tolerance_deg: float
    # Tilt captures accept a roll range; gaze captures do not check roll.
    roll_deg: Optional[Range] = None


class PositionPlan(Strict):
    position: str
    kind: Literal["calibration", "gaze", "tilt"]
    # Eye-in-orbit angle the head pose produces (deg; + = right / up).
    eye_in_orbit_deg: Tuple[float, float]
    head: HeadPoseTarget
    hold_s: float
    min_valid_frames: int
    instruction: str
    # Calibration only: the eye that fixes (the other is covered) and how far
    # the fixation target sits from the lens (cm; + = the patient's right).
    fixing_eye: Optional[Eye] = None
    target_offset_cm: float = 0.0

    @property
    def observed_eyes(self) -> Tuple[Eye, ...]:
        """The eyes QC and measurement need in every frame of this position."""
        return (self.fixing_eye,) if self.fixing_eye else ("R", "L")


class SettingsPlan(Strict):
    distance_cm: Range
    distance_tolerance_cm: float
    min_fps: float
    max_exposure_s: float
    recommended_frame_height_px: int
    # The phone computes frame sharpness at this iris radius (exam.qc.sharpness).
    sharpness_canonical_iris_radius_px: float
    camera: Literal["rear"] = "rear"
    light: str = "light source next to the lens"


class ProtocolPlan(Strict):
    protocol_version: str
    protocol: Protocol
    settings: SettingsPlan
    positions: List[PositionPlan] = Field(min_length=1)
    conventions: str
    kb_version: str
    kb_sha256: str

    def position(self, name: str) -> PositionPlan:
        for p in self.positions:
            if p.position == name:
                return p
        raise KeyError(f"{name!r} is not in the {self.protocol} plan")


CONVENTIONS = (
    "Right/left are the patient's. Head pose relative to the camera: yaw + = face turned to "
    "the patient's right, pitch + = chin up, roll + = head tilted to the patient's right "
    "shoulder. Eye-in-orbit angle = −head angle while the patient fixates the phone."
)


def _required(kb: KnowledgeBase, entry_id: str) -> float:
    value = kb.scalar(entry_id).value
    if value is None:
        raise ValueError(f"KB entry {entry_id!r} has no value; the capture protocol needs it")
    return value


def _gaze_instruction(position: str, h: int, v: int, angle: float) -> str:
    if position == "primary":
        return "Patient looks at the target with the head straight."
    parts = []
    if h:
        parts.append(f"turn the head to the patient's {_SIDE_WORD[-h]}")
    if v:
        parts.append("lower the chin" if v > 0 else "raise the chin")
    eyes = " and ".join(w for w in (_VERTICAL_WORD.get(v), _SIDE_WORD.get(h)) if w)
    return (f"Patient keeps looking at the target; {' and '.join(parts)} about {angle:g}° "
            f"(eyes end up looking {eyes}).")


def _calibration_instruction(eye: Eye, target: str, offset: float) -> str:
    cover = f"Cover the patient's {_EYE_WORD[_OTHER_EYE[eye]]} eye; head straight."
    if target == "lens":
        return f"{cover} The {_EYE_WORD[eye]} eye looks at the light next to the lens."
    return (f"{cover} The {_EYE_WORD[eye]} eye looks at the card target {offset:g} cm to the "
            f"patient's {target} of the lens.")


def build_plan(protocol: Protocol = "nine_gaze", include_tilt: bool = True,
               kb: Optional[KnowledgeBase] = None, include_calibration: bool = True) -> ProtocolPlan:
    """The ordered capture plan: calibration, primary, the other gaze positions, then the head tilts."""
    kb = kb or load_kb()
    angle = _required(kb, "gaze_eccentricity_deg")
    tol = _required(kb, "qc_head_pose_tolerance_deg")
    hold = _required(kb, "hold_duration_s")
    min_frames = int(_required(kb, "min_valid_frames_per_position"))

    positions: List[PositionPlan] = []
    if include_calibration:
        offset = _required(kb, "calibration_target_offset_cm")
        for name, (eye, target) in CALIBRATION_TARGETS.items():
            positions.append(PositionPlan(
                position=name, kind="calibration", eye_in_orbit_deg=(0.0, 0.0),
                head=HeadPoseTarget(yaw_deg=0.0, pitch_deg=0.0, tolerance_deg=tol),
                hold_s=hold, min_valid_frames=min_frames,
                instruction=_calibration_instruction(eye, target, offset),
                fixing_eye=eye,
                target_offset_cm={"lens": 0.0, "right": offset, "left": -offset}[target],
            ))
    for name in PROTOCOL_POSITIONS[protocol]:
        h, v = GAZE_DIRECTION[name]
        positions.append(PositionPlan(
            position=name, kind="gaze",
            eye_in_orbit_deg=(h * angle, v * angle),
            head=HeadPoseTarget(yaw_deg=-h * angle + 0.0, pitch_deg=-v * angle + 0.0, tolerance_deg=tol),
            hold_s=hold, min_valid_frames=min_frames,
            instruction=_gaze_instruction(name, h, v, angle),
        ))

    if include_tilt:
        lo, hi = _required(kb, "head_tilt_min_deg"), _required(kb, "head_tilt_max_deg")
        for name, sign in (("tilt_right", 1), ("tilt_left", -1)):
            roll = Range(min=lo, max=hi) if sign > 0 else Range(min=-hi, max=-lo)
            positions.append(PositionPlan(
                position=name, kind="tilt", eye_in_orbit_deg=(0.0, 0.0),
                head=HeadPoseTarget(yaw_deg=0.0, pitch_deg=0.0, tolerance_deg=tol, roll_deg=roll),
                hold_s=hold, min_valid_frames=min_frames,
                instruction=(f"Patient keeps looking at the target; tilt the head toward the "
                             f"patient's {_SIDE_WORD[sign]} shoulder, {lo:g}–{hi:g}°, without turning it."),
            ))

    settings = SettingsPlan(
        distance_cm=Range(min=_required(kb, "capture_distance_min_cm"),
                          max=_required(kb, "capture_distance_max_cm")),
        distance_tolerance_cm=_required(kb, "capture_distance_tolerance_cm"),
        min_fps=_required(kb, "capture_min_fps"),
        max_exposure_s=_required(kb, "capture_max_exposure_s"),
        recommended_frame_height_px=int(_required(kb, "capture_recommended_frame_height_px")),
        sharpness_canonical_iris_radius_px=_required(kb, "qc_sharpness_canonical_iris_radius_px"),
    )
    return ProtocolPlan(protocol_version=CAPTURE_PROTOCOL_VERSION, protocol=protocol,
                        settings=settings, positions=positions, conventions=CONVENTIONS,
                        **kb.version_info())
