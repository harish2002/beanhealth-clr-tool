/**
 * Parity tests for lib/examLandmarks.ts — the browser port of
 * backend/exam/landmarks.
 *
 * fixtures/examLandmarks.json holds synthetic eyes and what the Python
 * reference found in them (regenerate with
 * `cd backend && venv/bin/python ../tools/parity/dump_exam_fixtures.py`).
 * The port must land within a small fraction of a pixel of the reference,
 * and fail (null) exactly where the reference fails.
 *
 *   npm test
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { inflateSync } from "node:zlib";
import {
  LidAperture,
  LEFT_EYE_RING,
  RIGHT_EYE_RING,
  ellipsePoints,
  findPupil,
  fitConicEllipse,
  fitLimbus,
  headPose,
  lidMetrics,
  mulberry32,
  refineReflex,
  sharpness,
  wholeImage,
} from "../lib/examLandmarks.ts";
import type { ExamEllipse, XY } from "../lib/types.ts";

interface Scene {
  name: string;
  w: number;
  h: number;
  image_zlib_b64: string;
  aperture: { upper: XY[]; lower: XY[]; canthi: [XY, XY] };
  guess: { centre: XY; radius: number };
  reflex_approx: XY;
  iris_radius: number;
  truth: { limbus: ExamEllipse; reflex: XY };
  expect: {
    limbus: ExamEllipse | null;
    reflex: XY | null;
    pupil?: { centre: XY; diameter: number } | null;
    lids?: { fissure_height_px: number | null; scleral_show_px: number | null; limbus_covered_fraction: number };
  };
}

const fixtures = JSON.parse(readFileSync(new URL("./fixtures/examLandmarks.json", import.meta.url), "utf8")) as {
  scenes: Scene[];
  mulberry32: Record<string, number[]>;
};

const close = (a: number, b: number, tol: number, msg = "") =>
  assert.ok(Math.abs(a - b) <= tol, `${msg} expected ${b} ± ${tol}, got ${a}`);
const dist = (a: XY, b: XY) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function image(s: Scene) {
  const data = new Uint8Array(inflateSync(Buffer.from(s.image_zlib_b64, "base64")));
  assert.equal(data.length, s.w * s.h);
  return wholeImage(data, s.w, s.h);
}

function aperture(s: Scene) {
  return new LidAperture(s.aperture.upper, s.aperture.lower, s.aperture.canthi);
}

test("mulberry32 matches the Python generator draw for draw", () => {
  for (const [seed, draws] of Object.entries(fixtures.mulberry32)) {
    const r = mulberry32(Number(seed));
    for (const d of draws) assert.equal(r(), d);
  }
});

test("conic fit is exact on noiseless points", () => {
  const truth: ExamEllipse = { centre_px: [123.4, 56.7], semi_major_px: 60, semi_minor_px: 52, angle_deg: 30 };
  const fit = fitConicEllipse(ellipsePoints(truth, 40));
  assert.ok(fit);
  close(fit.centre_px[0], 123.4, 1e-6);
  close(fit.centre_px[1], 56.7, 1e-6);
  close(fit.semi_major_px, 60, 1e-6);
  close(fit.semi_minor_px, 52, 1e-6);
  close(fit.angle_deg, 30, 1e-4);
  assert.equal(fitConicEllipse(Array.from({ length: 10 }, (_, i): XY => [i, 2 * i])), null);
});

for (const s of fixtures.scenes) {
  test(`limbus matches the reference: ${s.name}`, () => {
    const fit = fitLimbus(image(s), s.guess.centre, s.guess.radius, aperture(s));
    if (s.expect.limbus === null) { assert.equal(fit, null); return; }
    assert.ok(fit, "the reference found a limbus");
    const e = s.expect.limbus;
    // Same samples, same steps: only float rounding differs (measured ~1e-6 px).
    assert.ok(dist(fit.ellipse.centre_px, e.centre_px) < 1e-3, `centre ${fit.ellipse.centre_px} vs ${e.centre_px}`);
    close(fit.ellipse.semi_major_px, e.semi_major_px, 1e-3, "semi-major");
    close(fit.ellipse.semi_minor_px, e.semi_minor_px, 1e-3, "semi-minor");
    // …and both sit on the drawn truth.
    assert.ok(dist(fit.ellipse.centre_px, s.truth.limbus.centre_px) < 0.3);
  });

  test(`reflex matches the reference: ${s.name}`, () => {
    const found = refineReflex(image(s), s.reflex_approx, s.iris_radius);
    if (s.expect.reflex === null) { assert.equal(found, null); return; }
    assert.ok(found);
    assert.ok(dist(found, s.expect.reflex) < 1e-6, `${found} vs ${s.expect.reflex}`);
  });

  if (s.expect.limbus) {
    test(`pupil and lids match the reference: ${s.name}`, () => {
      const limbus = s.expect.limbus as ExamEllipse;
      const pupil = findPupil(image(s), limbus);
      if (!s.expect.pupil) {
        assert.equal(pupil, null);
      } else {
        assert.ok(pupil, "the reference found a pupil");
        assert.ok(dist(pupil[0], s.expect.pupil.centre) < 1e-3, `${pupil[0]} vs ${s.expect.pupil.centre}`);
        close(pupil[1], s.expect.pupil.diameter, 1e-3, "diameter");
      }
      const lids = lidMetrics(aperture(s), limbus);
      const ref = s.expect.lids;
      assert.ok(ref);
      close(lids.fissure_height_px as number, ref.fissure_height_px as number, 1e-9, "fissure");
      close(lids.scleral_show_px as number, ref.scleral_show_px as number, 1e-9, "scleral show");
      close(lids.limbus_covered_fraction, ref.limbus_covered_fraction, 1e-9, "covered");
    });
  }
}

test("a region that does not cover the eye gives no limbus instead of a wrong one", () => {
  const s = fixtures.scenes[0];
  const full = image(s);
  const strip = { ...full, h: 40, data: full.data.subarray(0, full.w * 40) };
  assert.equal(fitLimbus(strip, s.guess.centre, s.guess.radius, aperture(s)), null);
});

test("sharpness drops when the eye is blurred", () => {
  const s = fixtures.scenes[0];
  const img = image(s);
  const blurred = new Uint8Array(img.data.length);
  for (let y = 0; y < s.h; y++) for (let x = 0; x < s.w; x++) {
    let acc = 0;
    for (let k = -3; k <= 3; k++) acc += img.data[y * s.w + Math.min(s.w - 1, Math.max(0, x + k))];
    blurred[y * s.w + x] = Math.round(acc / 7);
  }
  const sharp = sharpness(img, s.truth.limbus.centre_px, 30, 32) as number;
  const soft = sharpness({ ...img, data: blurred }, s.truth.limbus.centre_px, 30, 32) as number;
  assert.ok(soft < 0.6 * sharp, `${soft} vs ${sharp}`);
});

// ── Head pose ───────────────────────────────────────────────────────────────

type V3 = [number, number, number];

/** The rotation exam.landmarks.head_pose.pose_matrix builds: Ry(yaw)·Rx(−pitch)·Rz(−roll). */
function poseMatrix(yaw: number, pitch: number, roll: number): number[][] {
  const r = Math.PI / 180;
  const [y, p, z] = [yaw * r, -pitch * r, -roll * r];
  const ry = [[Math.cos(y), 0, Math.sin(y)], [0, 1, 0], [-Math.sin(y), 0, Math.cos(y)]];
  const rx = [[1, 0, 0], [0, Math.cos(p), -Math.sin(p)], [0, Math.sin(p), Math.cos(p)]];
  const rz = [[Math.cos(z), -Math.sin(z), 0], [Math.sin(z), Math.cos(z), 0], [0, 0, 1]];
  const mul = (a: number[][], b: number[][]) => a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)));
  return mul(mul(ry, rx), rz);
}

function face(yaw = 0, pitch = 0, roll = 0): V3[] {
  const pts: V3[] = Array.from({ length: 478 }, (_, i) => [((i * 37) % 80) - 40, ((i * 53) % 120) - 60, ((i * 17) % 30) - 15]);
  const ring = (idx: number[], cx: number) => idx.forEach((i, k) => {
    const t = (2 * Math.PI * k) / idx.length;
    pts[i] = [cx + 14 * Math.cos(t), 5 * Math.sin(t), 0];
  });
  ring(RIGHT_EYE_RING, -32);
  ring(LEFT_EYE_RING, 32);
  pts[10] = [0, -75, 0]; pts[151] = [0, -60, 0]; pts[9] = [0, -45, 0];
  pts[152] = [0, 95, 0]; pts[175] = [0, 100, 0]; pts[199] = [0, 88, 0];
  const m = poseMatrix(yaw, pitch, roll);
  return pts.map(([x, y, z]) => [
    960 + m[0][0] * x + m[0][1] * y + m[0][2] * z,
    540 + m[1][0] * x + m[1][1] * y + m[1][2] * z,
    m[2][0] * x + m[2][1] * y + m[2][2] * z,
  ]);
}

test("head pose recovers the rotation", () => {
  for (const [y, p, r] of [[0, 0, 0], [25, 0, 0], [0, -25, 0], [25, 25, 0], [0, 0, 40], [10, -5, 35], [-25, -25, -30]]) {
    const pose = headPose(face(y, p, r));
    close(pose.yaw_deg, y, 1e-6, "yaw");
    close(pose.pitch_deg, p, 1e-6, "pitch");
    close(pose.roll_deg, r, 1e-6, "roll");
  }
});

test("head pose signs follow the protocol", () => {
  // Tilted to the right shoulder: the right eye (image left) drops, roll +.
  const f = face(0, 0, 20);
  const meanY = (idx: number[]) => idx.reduce((s, i) => s + f[i][1], 0) / idx.length;
  assert.ok(meanY(RIGHT_EYE_RING) > meanY(LEFT_EYE_RING));
  assert.ok(headPose(f).roll_deg > 0);
  assert.ok(headPose(face(20, 0, 0)).yaw_deg > 0);
  assert.ok(headPose(face(0, 20, 0)).pitch_deg > 0);
  assert.throws(() => headPose(face().slice(0, 100)));
});
