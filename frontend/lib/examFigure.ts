/**
 * BeanHealth CLR — guided gaze exam: the position figure, in screen terms
 * =======================================================================
 *
 * Turns a position's target (head yaw / pitch / roll, fixing eye, card target)
 * into what a drawing of the patient needs, as the rear camera sees them:
 * unmirrored, so the patient's right is on the image left. Protocol signs:
 * yaw + = face to the patient's right, pitch + = chin up, roll + = toward the
 * right shoulder; eye-in-orbit + = the patient's right / up.
 */

import type { ExamEye, ExamHeadPose, ExamPositionPlan } from "./types";

export interface PoseFigure {
  /** Shift of the face's features, as a fraction of the head's half-width / half-height (+ = image right / down). */
  featureDx: number;
  featureDy: number;
  /** Rotation of the whole head for a tilt (degrees, + = clockwise on screen). */
  rotateDeg: number;
  /** Iris offset inside the eye, −1…1 of the eye's half-width / half-height (+ = image right / down). */
  gazeDx: number;
  gazeDy: number;
  /** The patient's eye that is covered, if any. */
  coveredEye: ExamEye | null;
  /** What the open eye(s) look at. */
  target: "light" | "patient_right" | "patient_left";
}

const RAD = Math.PI / 180;
const clamp = (v: number) => Math.max(-1, Math.min(1, v)) + 0;   // + 0: no negative zero

/** Where a head pose puts the face on screen (also used for the live pose dot). */
export function screenPose(pose: Pick<ExamHeadPose, "yaw_deg" | "pitch_deg">): { x: number; y: number } {
  // Turning to the patient's right moves the face to the image left; chin up moves it up.
  return { x: -pose.yaw_deg, y: -pose.pitch_deg };
}

/** The figure for a position; `distanceCm` turns a card target's offset into an eye angle. */
export function figureFor(p: ExamPositionPlan, distanceCm: number): PoseFigure {
  const roll = p.head.roll_deg ? (p.head.roll_deg.min + p.head.roll_deg.max) / 2 : 0;
  let [h, v] = p.eye_in_orbit_deg;
  if (p.kind === "calibration" && p.target_offset_cm !== 0) {
    h = Math.atan2(p.target_offset_cm, distanceCm > 0 ? distanceCm : 36) / RAD;
    v = 0;
  }
  const s = screenPose({ yaw_deg: p.head.yaw_deg, pitch_deg: p.head.pitch_deg });
  return {
    featureDx: Math.sin(s.x * RAD) * 0.9,
    featureDy: Math.sin(s.y * RAD) * 0.6,
    rotateDeg: -roll,
    // Eyes turned to the patient's right show the iris toward the image left.
    gazeDx: clamp(-Math.sin(h * RAD) * 1.6),
    gazeDy: clamp(-Math.sin(v * RAD) * 1.6),
    coveredEye: p.fixing_eye ? (p.fixing_eye === "R" ? "L" : "R") : null,
    target: p.kind === "calibration" && p.target_offset_cm !== 0
      ? (p.target_offset_cm > 0 ? "patient_right" : "patient_left") : "light",
  };
}

/** One-line description of the position's head and eyes, for under the figure. */
export function figureCaption(p: ExamPositionPlan): string {
  const head: string[] = [];
  const { yaw_deg: yaw, pitch_deg: pitch, roll_deg: roll } = p.head;
  if (roll) head.push(`tilted ${Math.abs(roll.min)}–${Math.abs(roll.max)}° to the ${roll.min > 0 ? "right" : "left"} shoulder`);
  if (yaw) head.push(`turned ${Math.abs(yaw)}° to the patient's ${yaw > 0 ? "right" : "left"}`);
  if (pitch) head.push(`chin ${pitch > 0 ? "up" : "down"} ${Math.abs(pitch)}°`);
  const eyes = p.kind === "calibration" && p.target_offset_cm !== 0
    ? `${p.fixing_eye === "R" ? "right" : "left"} eye on the dot ${Math.abs(p.target_offset_cm)} cm to the patient's ${p.target_offset_cm > 0 ? "right" : "left"}`
    : p.fixing_eye ? `${p.fixing_eye === "R" ? "right" : "left"} eye on the light` : "both eyes on the light";
  const cover = p.fixing_eye ? ` · ${p.fixing_eye === "R" ? "left" : "right"} eye covered` : "";
  return `Head ${head.length ? head.join(", ") : "straight"} · ${eyes}${cover}`;
}
