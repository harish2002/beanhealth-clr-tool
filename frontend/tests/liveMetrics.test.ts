/**
 * Unit tests for lib/liveMetrics.ts — geometry, classification and
 * aggregation (browser port of modules 4–6, 8 and module_aggregate).
 *
 *   npm test
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  aggregate,
  measureFrame,
  toPrismDioptres,
  CONDITION_BY_REFLEX,
  SERVER_MIN_FRAMES,
  type EyeInput,
} from "../lib/liveMetrics.ts";

const close = (a: number, b: number, tol: number, msg = "") =>
  assert.ok(Math.abs(a - b) <= tol, `${msg} expected ${b} ± ${tol}, got ${a}`);

// Two eyes 60 px apart, iris radius 10 px → 1 mm of reflex offset = 10/5.75 px.
const R = 10;
const MM = R / 5.75;

function eyes(odReflexDx: number, osReflexDx: number, rotDeg = 0, cornerDrop = 2): [EyeInput, EyeInput] {
  const t = (rotDeg * Math.PI) / 180, c = Math.cos(t), s = Math.sin(t);
  const P = (x: number, y: number) => ({ x: 640 + x * c - y * s, y: 360 + x * s + y * c });
  return [
    { pupil: P(-30, 0), irisR: R, reflex: P(-30 + odReflexDx, 0), inner: P(-12, -cornerDrop), outer: P(-48, -cornerDrop) },
    { pupil: P(30, 0),  irisR: R, reflex: P(30 + osReflexDx, 0),  inner: P(12, -cornerDrop),  outer: P(48, -cornerDrop) },
  ];
}

test("angle kappa (both reflexes equally nasal) cancels", () => {
  close(measureFrame(...eyes(+0.5, -0.5)).asymDeg ?? NaN, 0, 1e-9);
});

test("1 mm of reflex offset in one eye = 7° (Hirschberg)", () => {
  close(measureFrame(...eyes(-MM, 0)).asymDeg ?? NaN, 7, 1e-9);
});

test("inter-pupil distance uses the iris as the ruler", () => {
  close(measureFrame(...eyes(0, 0)).ipdMm, (60 * 5.75) / R, 1e-9);
});

test("head roll changes neither the angle nor the verdict", () => {
  const m = measureFrame(...eyes(-MM, 0, 20));
  close(m.asymDeg ?? NaN, 7, 1e-9);
  close(m.rollDeg, 20, 1e-9);
});

test("roll can be taken from separate reference points (the server uses iris-centre landmarks)", () => {
  const [od, os] = eyes(-MM, 0, 20);
  const m = measureFrame(od, os, [{ x: 0, y: 0 }, { x: 10, y: 0 }]);   // a level reference
  close(m.rollDeg, 0, 1e-9);
});

test("Method B: pupils equally below their corner lines show no vertical asymmetry", () => {
  const { methodB } = measureFrame(...eyes(0, 0, 0, 3));
  close(methodB.vOD, methodB.vOS, 1e-12);
  assert.ok(methodB.vOD > 0, "positive = below the line, in both eyes");
  close(methodB.hOD, methodB.hOS, 1e-12);
});

test("prism dioptres use 100·tan(θ), not a linear factor", () => {
  close(toPrismDioptres(7), 100 * Math.tan((7 * Math.PI) / 180), 1e-12);
  close(toPrismDioptres(45), 100, 1e-9);
});

test("a steady 7° reading → MONITOR, esotropia pattern, high confidence", () => {
  const h = aggregate(Array.from({ length: 12 }, () => measureFrame(...eyes(-MM, 0))))?.hirschberg;
  assert.ok(h);
  assert.equal(h.tier, "MONITOR");
  assert.equal(h.severity, "MILD");
  assert.equal(h.confidence, "HIGH");
  assert.equal(h.deviatingEye, "OD");
  assert.equal(h.reflexDirection, "temporal");
  assert.equal(CONDITION_BY_REFLEX.temporal.name, "Esotropia");
  assert.equal(h.directionReliable, true);
});

test("readings that jump between frames are flagged unstable instead of scored", () => {
  const frames = Array.from({ length: 12 }, (_, i) => measureFrame(...eyes(-MM * (i % 2 ? 0.2 : 3.5), 0)));
  assert.equal(aggregate(frames)?.hirschberg?.unstable, true);
});

test("a pattern that flips between frames is withheld; the tier stands", () => {
  const flip = (i: number) => {
    const [od, os] = eyes(0, 0);
    const d = 1.4 * MM;
    od.reflex = i % 2 ? { x: od.pupil.x + d, y: od.pupil.y } : { x: od.pupil.x, y: od.pupil.y - d };
    return measureFrame(od, os);
  };
  const h = aggregate(Array.from({ length: 12 }, (_, i) => flip(i)))?.hirschberg;
  assert.ok(h);
  assert.equal(h.directionReliable, false);
  assert.equal(h.tier, "MONITOR");
});

test("the frame minimum is a parameter: the live view needs 5, a server comparison uses 3", () => {
  const three = Array.from({ length: 3 }, () => measureFrame(...eyes(-MM, 0)));
  assert.equal(aggregate(three)?.hirschberg, null);
  assert.ok(aggregate(three, SERVER_MIN_FRAMES)?.hirschberg);
});
