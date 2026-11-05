/**
 * Unit tests for lib/examGuidance.ts — operator hints, the self-exam voice
 * guide and the session upload.
 *
 *   npm test
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildSession, deviceModel, effectiveFps, poseError, poseGuidance, spokenHint, spokenInstruction, usableFrames,
} from "../lib/examGuidance.ts";
import type { ExamEyeObservation, ExamFrameObservation, ExamPositionPlan, ExamProtocolPlan } from "../lib/types.ts";

function plan(position: string, yaw: number, pitch: number, roll: [number, number] | null = null): ExamPositionPlan {
  return {
    position, kind: roll ? "tilt" : "gaze", eye_in_orbit_deg: [-yaw, -pitch],
    head: { yaw_deg: yaw, pitch_deg: pitch, tolerance_deg: 5, roll_deg: roll ? { min: roll[0], max: roll[1] } : null },
    hold_s: 2, min_valid_frames: 10, instruction: "", fixing_eye: null, target_offset_cm: 0,
  };
}

const pose = (yaw: number, pitch: number, roll = 0) => ({ yaw_deg: yaw, pitch_deg: pitch, roll_deg: roll });

test("on target within the tolerance", () => {
  assert.deepEqual(poseGuidance(pose(-23, 4), plan("right", -25, 0)), { onTarget: true, hints: [] });
});

test("hints say which way to move the head, in the patient's terms", () => {
  // Eyes right: head turned 25° to the patient's left (yaw −25).
  const g = poseGuidance(pose(-10, 0), plan("right", -25, 0));
  assert.equal(g.onTarget, false);
  assert.deepEqual(g.hints, ["Turn the head 15° more to the patient's left"]);
  // Eyes up: chin down (pitch −25).
  assert.deepEqual(poseGuidance(pose(0, -5), plan("up", 0, -25)).hints, ["Lower the chin 20°"]);
  assert.deepEqual(poseGuidance(pose(0, -35), plan("up", 0, -25)).hints, ["Raise the chin 10°"]);
});

test("tilt captures check the roll range", () => {
  const right = plan("tilt_right", 0, 0, [30, 45]);
  assert.equal(poseGuidance(pose(0, 0, 38), right).onTarget, true);
  assert.deepEqual(poseGuidance(pose(0, 0, 20), right).hints, ["Tilt the head further toward the right shoulder"]);
  assert.deepEqual(poseGuidance(pose(0, 0, 50), right).hints, ["Tilt the head less toward the right shoulder"]);
  const left = plan("tilt_left", 0, 0, [-45, -30]);
  assert.deepEqual(poseGuidance(pose(0, 0, -10), left).hints, ["Tilt the head further toward the left shoulder"]);
});

test("no face, no target", () => {
  assert.equal(poseGuidance(null, plan("primary", 0, 0)).onTarget, false);
});

test("pose error: degrees outside the accepted pose", () => {
  assert.equal(poseError(null, plan("primary", 0, 0)), null);
  assert.equal(poseError(pose(-22, 3), plan("right", -25, 0)), 0);
  assert.equal(poseError(pose(-10, 0), plan("right", -25, 0)), 10);
  assert.equal(poseError(pose(0, 0, 20), plan("tilt_right", 0, 0, [30, 45])), 10);
  assert.equal(poseError(pose(0, 0, 50), plan("tilt_right", 0, 0, [30, 45])), 5);
});

const calib = (eye: "R" | "L", offset: number): ExamPositionPlan => ({
  ...plan(`calib_${eye}`, 0, 0), kind: "calibration", fixing_eye: eye, target_offset_cm: offset,
});

test("spoken instructions talk to the patient, in their own right and left", () => {
  assert.equal(spokenInstruction(calib("R", 0), null),
    "Cover your left eye with your palm. Keep your head straight and look at the light with your right eye.");
  assert.equal(spokenInstruction(calib("R", 7), calib("R", 0)),
    "Keep your left eye covered. Without moving your head, look at the dot to the right of the light.");
  assert.equal(spokenInstruction(calib("L", 0), calib("R", -7)),
    "Cover your right eye with your palm. Keep your head straight and look at the light with your left eye.");
  assert.equal(spokenInstruction(plan("primary", 0, 0), calib("L", -7)),
    "Uncover your eye. Keep your head straight and look at the light.");
  // Eyes up-right: head turned to the patient's left, chin down.
  assert.equal(spokenInstruction(plan("up_right", -25, -25), plan("primary", 0, 0)),
    "Keep looking at the light, and slowly turn your head to your left and lower your chin.");
  assert.equal(spokenInstruction(plan("tilt_left", 0, 0, [-45, -30]), plan("down_left", 25, 25)),
    "Keep looking at the light, and tilt your head toward your left shoulder, without turning it.");
});

test("spoken hints: which way, and how far", () => {
  assert.equal(spokenHint(null, plan("primary", 0, 0)), "I can't see your face. Face the phone.");
  assert.equal(spokenHint(pose(-23, 0), plan("right", -25, 0)), null);
  assert.equal(spokenHint(pose(0, 0), plan("right", -25, 0)), "Turn to your left.");
  assert.equal(spokenHint(pose(-17, 0), plan("right", -25, 0)), "A little to your left.");
  assert.equal(spokenHint(pose(0, -15), plan("up", 0, -25)), "Chin down a little.");
  assert.equal(spokenHint(pose(10, 0), plan("primary", 0, 0)), "A little to your left.");
  assert.equal(spokenHint(pose(0, 0, 10), plan("tilt_right", 0, 0, [30, 45])), "Tilt more toward your right shoulder.");
  assert.equal(spokenHint(pose(0, 0, -50), plan("tilt_left", 0, 0, [-45, -30])), "Tilt a little less.");
});

const eyeObs = (ok: boolean): ExamEyeObservation => ({
  reflex_px: ok ? [10, 10] : null, limbus: { centre_px: [10, 10], semi_major_px: 20, semi_minor_px: 19, angle_deg: 0 },
  pupil_px: null, pupil_diameter_px: null, iris_radius_px: null, fissure_height_px: null, scleral_show_px: null,
  limbus_covered_fraction: null,
});

test("usable frames need the pose on target and every observed eye's reflex", () => {
  const p = plan("primary", 0, 0);
  const f = (yaw: number, rOk: boolean, lOk: boolean): ExamFrameObservation =>
    ({ index: 0, t_ms: 0, head_pose: pose(yaw, 0), sharpness: 1, eyes: { R: eyeObs(rOk), L: eyeObs(lOk) } });
  assert.equal(usableFrames([f(0, true, true), f(12, true, true), f(0, true, false)], p), 1);
  // Calibration watches only the fixing eye.
  assert.equal(usableFrames([f(0, true, false)], calib("R", 0)), 1);
});

const frames = (n: number, dtMs: number): ExamFrameObservation[] =>
  Array.from({ length: n }, (_, i) => ({ index: i, t_ms: i * dtMs, head_pose: pose(0, 0), sharpness: 10, eyes: {} }));

test("effective frame rate", () => {
  assert.equal(effectiveFps(frames(1, 33)), null);
  assert.equal(Math.round(effectiveFps(frames(31, 1000 / 15)) as number), 15);
});

test("session: plan order, empty positions left out, slowest rate reported", () => {
  const p: ExamProtocolPlan = {
    protocol_version: "1.1.0", protocol: "five_gaze",
    settings: { distance_cm: { min: 33, max: 40 }, distance_tolerance_cm: 3, min_fps: 30, max_exposure_s: 0.004,
                recommended_frame_height_px: 2160, sharpness_canonical_iris_radius_px: 32, camera: "rear", light: "" },
    positions: [plan("primary", 0, 0), plan("right", -25, 0), plan("left", 25, 0)],
    conventions: "", kb_version: "0.4.0", kb_sha256: "x",
  };
  const s = buildSession({
    sessionId: "s1", plan: p, captures: { left: frames(20, 50), primary: frames(30, 1000 / 30), right: [] },
    frameSize: { width: 3840, height: 2160 }, distanceCm: 36, distanceErrorCm: 1, deviceModel: "Pixel 8",
    startedAt: "2026-10-31T10:00:00Z",
  });
  assert.deepEqual(s.captures.map((c) => c.position), ["primary", "left"]);
  assert.equal(Math.round(s.header.settings.fps), 20);
  assert.equal(s.header.settings.exposure_s, null);
  assert.deepEqual(s.header.distance_cm, { value: 36, error: 1, confidence: "HIGH" });
  assert.equal(s.header.protocol_version, "1.1.0");
});

test("device label", () => {
  assert.equal(deviceModel("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"), "iPhone");
  assert.equal(deviceModel("Mozilla/5.0 (Linux; Android 14; Pixel 8 Build/AP1A) Chrome/120"), "Pixel 8");
  assert.equal(deviceModel("Mozilla/5.0 (Linux; Android 13; SM-S911B) Chrome/120"), "SM-S911B");
});
