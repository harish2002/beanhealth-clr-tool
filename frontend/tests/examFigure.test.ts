/**
 * Unit tests for lib/examFigure.ts — the position figure's on-screen signs.
 *
 *   npm test
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { figureCaption, figureFor, screenPose } from "../lib/examFigure.ts";
import type { ExamPositionPlan } from "../lib/types.ts";

function plan(over: Partial<ExamPositionPlan> & { yaw?: number; pitch?: number; roll?: [number, number] }): ExamPositionPlan {
  const { yaw = 0, pitch = 0, roll, ...rest } = over;
  return {
    position: "p", kind: "gaze", eye_in_orbit_deg: [-yaw, -pitch],
    head: { yaw_deg: yaw, pitch_deg: pitch, tolerance_deg: 5, roll_deg: roll ? { min: roll[0], max: roll[1] } : null },
    hold_s: 2, min_valid_frames: 10, instruction: "", fixing_eye: null, target_offset_cm: 0, ...rest,
  };
}

test("right gaze: head turned to the patient's left shows the face to the image right, irises to the image left", () => {
  const f = figureFor(plan({ yaw: -25 }), 36);
  assert.ok(f.featureDx > 0);
  assert.ok(f.gazeDx < 0);            // eyes look to the patient's right = image left
  assert.equal(f.gazeDy, 0);
  assert.equal(f.coveredEye, null);
  assert.equal(f.target, "light");
});

test("up gaze: chin down moves the face down and the irises up", () => {
  const f = figureFor(plan({ pitch: -25 }), 36);
  assert.ok(f.featureDy > 0);
  assert.ok(f.gazeDy < 0);
});

test("tilt to the right shoulder turns the head anticlockwise on screen", () => {
  assert.equal(figureFor(plan({ kind: "tilt", roll: [30, 45] }), 36).rotateDeg, -37.5);
  assert.equal(figureFor(plan({ kind: "tilt", roll: [-45, -30] }), 36).rotateDeg, 37.5);
});

test("calibration: the other eye is covered, and the dot sets where the open eye looks", () => {
  const lens = figureFor(plan({ kind: "calibration", fixing_eye: "R" }), 36);
  assert.equal(lens.coveredEye, "L");
  assert.equal(lens.target, "light");
  const right = figureFor(plan({ kind: "calibration", fixing_eye: "R", target_offset_cm: 7 }), 36);
  assert.equal(right.target, "patient_right");
  assert.ok(right.gazeDx < 0);
});

test("the live pose dot moves like the face does", () => {
  assert.deepEqual(screenPose({ yaw_deg: 20, pitch_deg: 10 }), { x: -20, y: -10 });
});

test("captions", () => {
  assert.equal(figureCaption(plan({ yaw: -25, pitch: -25 })),
    "Head turned 25° to the patient's left, chin down 25° · both eyes on the light");
  assert.equal(figureCaption(plan({ kind: "calibration", fixing_eye: "L", target_offset_cm: -7 })),
    "Head straight · left eye on the dot 7 cm to the patient's left · right eye covered");
  assert.equal(figureCaption(plan({ kind: "tilt", roll: [30, 45] })),
    "Head tilted 30–45° to the right shoulder · both eyes on the light");
});
