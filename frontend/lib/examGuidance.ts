/**
 * BeanHealth CLR — guided gaze-exam capture: pure helpers
 * =======================================================
 *
 * What the operator is told while holding a position, and how the recorded
 * landmark tracks become the session the server checks (exam.capture.session).
 * Head pose follows the capture protocol: yaw + = face turned to the
 * PATIENT's right, pitch + = chin up, roll + = toward the right shoulder.
 */

import type {
  ExamCaptureSession,
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

/** Compare a head pose with a position's target (the same test frame QC applies). */
export function poseGuidance(pose: ExamHeadPose | null, plan: ExamPositionPlan): PoseGuidance {
  if (!pose) return { onTarget: false, hints: ["Bring the patient's face into view"] };
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
