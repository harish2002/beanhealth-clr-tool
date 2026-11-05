/**
 * BeanHealth CLR — guided gaze-exam capture: pure helpers
 * =======================================================
 *
 * What the operator (or, in the self exam, the patient) is told while holding
 * a position, and how the recorded landmark tracks become the session the
 * server checks (exam.capture.session).
 * Head pose follows the capture protocol: yaw + = face turned to the
 * PATIENT's right, pitch + = chin up, roll + = toward the right shoulder.
 */

import type {
  ExamCaptureSession,
  ExamEye,
  ExamFrameObservation,
  ExamHeadPose,
  ExamPositionPlan,
  ExamProtocolPlan,
} from "./types";

export interface PoseGuidance {
  onTarget: boolean;
  /** Plain instructions for the operator, most important first; empty when on target. */
  hints: string[];
}

/**
 * Head roll beyond this means the picture itself is turned: the phone is held
 * sideways while the screen stays locked upright. An interface threshold, well
 * past the protocol's largest head tilt — not a clinical value.
 */
export const SIDEWAYS_ROLL_DEG = 65;

const sideways = (pose: ExamHeadPose) => Math.abs(pose.roll_deg) > SIDEWAYS_ROLL_DEG;

/** Compare a head pose with a position's target (the same test frame QC applies). */
export function poseGuidance(pose: ExamHeadPose | null, plan: ExamPositionPlan): PoseGuidance {
  if (!pose) return { onTarget: false, hints: ["Bring the patient's face into view"] };
  if (sideways(pose)) return { onTarget: false, hints: ["The picture is sideways: turn on auto-rotate, or hold the phone the way the screen faces"] };
  const { yaw_deg: yaw, pitch_deg: pitch, tolerance_deg: tol, roll_deg: roll } = plan.head;
  const hints: string[] = [];
  const dy = yaw - pose.yaw_deg;
  if (Math.abs(dy) > tol) hints.push(`Turn the head ${Math.round(Math.abs(dy))}° more to the patient's ${dy > 0 ? "right" : "left"}`);
  const dp = pitch - pose.pitch_deg;
  if (Math.abs(dp) > tol) hints.push(`${dp > 0 ? "Raise" : "Lower"} the chin ${Math.round(Math.abs(dp))}°`);
  if (roll) {
    if (pose.roll_deg < roll.min) {
      hints.push(roll.min > 0 ? "Tilt the head further toward the right shoulder" : "Tilt the head less toward the left shoulder");
    } else if (pose.roll_deg > roll.max) {
      hints.push(roll.max < 0 ? "Tilt the head further toward the left shoulder" : "Tilt the head less toward the right shoulder");
    }
  }
  return { onTarget: hints.length === 0, hints };
}

/** The eyes a position observes: the fixing eye during calibration, otherwise both. */
export function observedEyes(p: ExamPositionPlan): ExamEye[] {
  return p.fixing_eye ? [p.fixing_eye] : ["R", "L"];
}

/** Degrees the head is outside the position's accepted pose (0 = on target, null = no face). */
export function poseError(pose: ExamHeadPose | null, plan: ExamPositionPlan): number | null {
  if (!pose) return null;
  const { yaw_deg: yaw, pitch_deg: pitch, tolerance_deg: tol, roll_deg: roll } = plan.head;
  const over = (d: number) => Math.max(0, Math.abs(d) - tol);
  let err = Math.max(over(yaw - pose.yaw_deg), over(pitch - pose.pitch_deg));
  if (roll) err = Math.max(err, roll.min - pose.roll_deg, pose.roll_deg - roll.max);
  return err;
}

// ── Self exam: what the phone says to the patient ─────────────────────────
//
// The patient is the listener, so everything is in the second person; "your
// right" is the patient's right, the same side the protocol calls right.

const SIDE = (sign: number) => (sign > 0 ? "right" : "left");

/** The spoken instruction for a position, given the one spoken before it (null at the start). */
export function spokenInstruction(p: ExamPositionPlan, prev: ExamPositionPlan | null): string {
  if (p.kind === "calibration" && p.fixing_eye) {
    const eye = p.fixing_eye === "R" ? "right" : "left";
    const other = p.fixing_eye === "R" ? "left" : "right";
    const sameEye = prev?.kind === "calibration" && prev.fixing_eye === p.fixing_eye;
    const cover = sameEye ? `Keep your ${other} eye covered.` : `Cover your ${other} eye with your palm.`;
    const look = p.target_offset_cm === 0
      ? `Keep your head straight and look at the light with your ${eye} eye.`
      : `Without moving your head, look at the dot to the ${SIDE(p.target_offset_cm)} of the light.`;
    return `${cover} ${look}`;
  }
  const uncover = prev?.kind === "calibration" ? "Uncover your eye. " : "";
  if (p.kind === "tilt" && p.head.roll_deg) {
    return `${uncover}Keep looking at the light, and tilt your head toward your ${SIDE(p.head.roll_deg.min)} shoulder, without turning it.`;
  }
  const moves: string[] = [];
  if (p.head.yaw_deg !== 0) moves.push(`turn your head to your ${SIDE(p.head.yaw_deg)}`);
  if (p.head.pitch_deg !== 0) moves.push(p.head.pitch_deg > 0 ? "raise your chin" : "lower your chin");
  if (!moves.length) return `${uncover}Keep your head straight and look at the light.`;
  return `${uncover}Keep looking at the light, and slowly ${moves.join(" and ")}.`;
}

/** A short spoken correction toward the position's pose; null when on target. */
export function spokenHint(pose: ExamHeadPose | null, p: ExamPositionPlan): string | null {
  if (!pose) return "I can't see your face. Face the phone.";
  if (sideways(pose)) return "The picture is sideways. Turn on auto-rotate, or hold the phone upright.";
  const { yaw_deg: yaw, pitch_deg: pitch, tolerance_deg: tol, roll_deg: roll } = p.head;
  const parts: string[] = [];
  const dy = yaw - pose.yaw_deg;
  if (Math.abs(dy) > tol) parts.push(Math.abs(dy) <= 2 * tol ? `A little to your ${SIDE(dy)}` : `Turn to your ${SIDE(dy)}`);
  const dp = pitch - pose.pitch_deg;
  if (Math.abs(dp) > tol) {
    const small = Math.abs(dp) <= 2 * tol;
    parts.push(dp > 0 ? (small ? "chin up a little" : "raise your chin") : (small ? "chin down a little" : "lower your chin"));
  }
  if (roll) {
    const side = SIDE(roll.min);
    const toward = roll.min > 0 ? pose.roll_deg < roll.min : pose.roll_deg > roll.max;
    const away = roll.min > 0 ? pose.roll_deg > roll.max : pose.roll_deg < roll.min;
    if (toward) parts.push(`tilt more toward your ${side} shoulder`);
    else if (away) parts.push(`tilt a little less`);
  }
  if (!parts.length) return null;
  const s = parts.join(", ");
  return s[0].toUpperCase() + s.slice(1) + ".";
}

/**
 * Frames the phone itself counts as usable: head on target and every observed
 * eye with a limbus and a reflex. Only decides whether the self exam repeats a
 * position; the server's frame QC remains the judge.
 */
export function usableFrames(frames: ExamFrameObservation[], p: ExamPositionPlan): number {
  return frames.filter((f) => poseError(f.head_pose, p) === 0
    && observedEyes(p).every((e) => f.eyes[e]?.limbus && f.eyes[e]?.reflex_px)).length;
}

export type FrameOrientation = "portrait" | "landscape";

export function frameOrientation(width: number, height: number): FrameOrientation {
  return width >= height ? "landscape" : "portrait";
}

/** What the patient or operator is told when the phone was turned mid-session. */
export function rotateBackMessage(locked: FrameOrientation): string {
  return `The phone was turned. Turn it back to ${locked}.`;
}

/** Frames per second actually recorded (what the session reports; QC flags a low rate). */
export function effectiveFps(frames: ExamFrameObservation[]): number | null {
  if (frames.length < 2) return null;
  const span = frames[frames.length - 1].t_ms - frames[0].t_ms;
  return span > 0 ? ((frames.length - 1) * 1000) / span : null;
}

export interface SessionInput {
  sessionId: string;
  plan: ExamProtocolPlan;
  captures: Record<string, ExamFrameObservation[]>;
  frameSize: { width: number; height: number };
  distanceCm: number | null;
  /** Uncertainty of the operator's distance measurement (cm, one SD). */
  distanceErrorCm: number;
  deviceModel: string;
  startedAt: string;
}

/** The capture session upload (landmark tracks only, never images). */
export function buildSession(input: SessionInput): ExamCaptureSession {
  const order = input.plan.positions.map((p) => p.position);
  const captures = order
    .filter((name) => (input.captures[name]?.length ?? 0) > 0)
    .map((name) => ({ position: name, frames: input.captures[name] }));
  const rates = captures.map((c) => effectiveFps(c.frames)).filter((v): v is number => v !== null);
  // The slowest position sets the rate frame QC reasons about (saccade gaps).
  const fps = rates.length ? Math.min(...rates) : 1;
  return {
    header: {
      session_id: input.sessionId,
      protocol_version: input.plan.protocol_version,
      protocol: input.plan.protocol,
      tier: 1,
      device: { model: input.deviceModel || "unknown", lens: "rear", calibration_profile_id: null },
      settings: { width_px: input.frameSize.width, height_px: input.frameSize.height, fps, exposure_s: null },
      distance_cm: input.distanceCm === null ? null
        : { value: input.distanceCm, error: input.distanceErrorCm, confidence: "HIGH" },
      started_at: input.startedAt,
    },
    captures,
  };
}

/** Short device label from the user agent (the backend keeps the full fingerprint logic). */
export function deviceModel(userAgent: string): string {
  const iphone = /iPhone/.test(userAgent) ? "iPhone" : null;
  const android = /Android[^;]*;\s*([^;)]+?)(?:\s+Build|\))/.exec(userAgent)?.[1]?.trim();
  return iphone ?? android ?? (userAgent.split(" ")[0] || "unknown");
}
