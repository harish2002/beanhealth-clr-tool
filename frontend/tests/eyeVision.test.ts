/**
 * Unit tests for lib/eyeVision.ts — the browser port of modules 1–3.
 *
 * Exact agreement with the Python pipeline on real eye crops is checked by
 * tools/parity (it needs the git-ignored test photos); these tests pin the
 * behaviour on synthetic images so they run anywhere.
 *
 *   npm test
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  detectReflex,
  eyeCropBox,
  fitEllipseCentre,
  gaussianBlur5,
  localisePupil,
  otsuThreshold,
  rgbaToGray,
} from "../lib/eyeVision.ts";

const close = (a: number, b: number, tol: number, msg = "") =>
  assert.ok(Math.abs(a - b) <= tol, `${msg} expected ${b} ± ${tol}, got ${a}`);

/** Grey image from a function of (x, y). */
function image(w: number, h: number, f: (x: number, y: number) => number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = Math.max(0, Math.min(255, Math.round(f(x, y))));
  return out;
}

/** Synthetic eye: skin, iris disc, dark pupil (optionally offset), optional reflex spot. */
function eye(opts: { pupil?: { x: number; y: number }; reflex?: { x: number; y: number; sigma?: number }; lidRows?: number } = {}) {
  const w = 120, h = 80, iris = { x: 60, y: 40 }, irisR = 20;
  const pupil = opts.pupil ?? iris;
  const img = image(w, h, (x, y) => {
    let v = 200;                                                          // skin / sclera
    if (Math.hypot(x - iris.x, y - iris.y) <= irisR) v = 110;             // iris
    if (Math.hypot(x - pupil.x, y - pupil.y) <= 7) v = 25;                // pupil
    if (opts.lidRows && y < iris.y - irisR + opts.lidRows) v = 200;       // eyelid covering the top
    if (opts.reflex) {
      const s = opts.reflex.sigma ?? 2.5;   // ≥ the server's minimum reflex area at iris radius 20
      v += 235 * Math.exp(-((x - opts.reflex.x) ** 2 + (y - opts.reflex.y) ** 2) / (2 * s * s));
    }
    return v;
  });
  return { img, w, h, iris, irisR };
}

// ── Pixel primitives match cv2 arithmetic ─────────────────────────────────

test("rgbaToGray uses cv2's fixed-point weights", () => {
  const px = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]);
  assert.deepEqual(Array.from(rgbaToGray(px)), [76, 150, 29, 255]);
});

test("gaussianBlur5 keeps flat images flat and spreads an impulse as [1 4 6 4 1]²/256", () => {
  assert.ok(gaussianBlur5(image(9, 9, () => 137), 9, 9).every((v) => v === 137));
  const out = gaussianBlur5(image(9, 9, (x, y) => (x === 4 && y === 4 ? 255 : 0)), 9, 9);
  assert.equal(out[4 * 9 + 4], Math.round((255 * 36) / 256));   // centre weight 6×6
  assert.equal(out[4 * 9 + 2], Math.round((255 * 6) / 256));    // 1×6 two pixels left
});

test("otsuThreshold splits a two-level image at the lower level, like cv2", () => {
  assert.equal(otsuThreshold([...new Array(100).fill(50), ...new Array(100).fill(200)]), 50);
});

test("fitEllipseCentre recovers the centre from a full ellipse and from a partial arc", () => {
  const cx = 12.3, cy = 7.8, a = 9, b = 5, rot = 0.4;
  const pts = (from: number, to: number) => Array.from({ length: 40 }, (_, i) => {
    const t = from + ((to - from) * i) / 39;
    const x = a * Math.cos(t), y = b * Math.sin(t);
    return { x: cx + x * Math.cos(rot) - y * Math.sin(rot), y: cy + x * Math.sin(rot) + y * Math.cos(rot) };
  });
  for (const [label, p] of [["full", pts(0, 2 * Math.PI)], ["partial", pts(0.2, Math.PI)]] as const) {
    const c = fitEllipseCentre([...p]);
    assert.ok(c, label);
    close(c.x, cx, 1e-6, `${label} x`);
    close(c.y, cy, 1e-6, `${label} y`);
  }
});

// ── Module 1: eye crop ─────────────────────────────────────────────────────

test("eyeCropBox truncates like Python int() and pads 35% / 50%, clamped to the image", () => {
  const box = eyeCropBox([{ x: 100.9, y: 50.7 }, { x: 160.2, y: 70.3 }], 1280, 720);
  // bbox 100..160 × 50..70 → pad 21 × 10
  assert.deepEqual(box, { x1: 79, y1: 40, x2: 181, y2: 80 });
  assert.deepEqual(eyeCropBox([{ x: 2, y: 3 }, { x: 40, y: 20 }], 30, 18), { x1: 0, y1: 0, x2: 30, y2: 18 });
});

// ── Module 2: pupil ────────────────────────────────────────────────────────

test("dark pupil is found at its own centre, not the iris centre", () => {
  const e = eye({ pupil: { x: 64, y: 37 } });
  const r = localisePupil(e.img, e.w, e.h, e.iris, e.irisR);
  assert.equal(r.source, "dark");
  close(r.centre.x, 64, 0.5, "x");
  close(r.centre.y, 37, 0.5, "y");
});

test("pupil is still found with a torch reflex inside it", () => {
  const e = eye({ reflex: { x: 62, y: 39 } });
  const r = localisePupil(e.img, e.w, e.h, e.iris, e.irisR);
  assert.equal(r.source, "dark");
  close(Math.hypot(r.centre.x - 60, r.centre.y - 40), 0, 1.0, "distance from true centre");
});

test("a partly covered pupil keeps its centre (ellipse fit)", () => {
  const e = eye({ lidRows: 17 });   // lid covers the pupil's top 4 rows
  const r = localisePupil(e.img, e.w, e.h, e.iris, e.irisR);
  assert.equal(r.source, "dark");
  close(Math.hypot(r.centre.x - 60, r.centre.y - 40), 0, 1.5, "distance from true centre");
});

test("no dark pupil → falls back to the iris centre at LOW confidence", () => {
  const img = image(120, 80, () => 150);
  assert.deepEqual(localisePupil(img, 120, 80, { x: 60, y: 40 }, 20), { centre: { x: 60, y: 40 }, source: "iris", confidence: "LOW" });
});

test("a dark blob too far from the iris centre is rejected (eyelash / brow shadow)", () => {
  const e = eye({ pupil: { x: 75, y: 40 } });   // 15 px = 0.75 iris radii > 0.6
  assert.equal(localisePupil(e.img, e.w, e.h, e.iris, e.irisR).source, "iris");
});

// ── Module 3: reflex ───────────────────────────────────────────────────────

test("no torch → no_flash, never a guessed position", () => {
  const e = eye();
  assert.equal(detectReflex(e.img, e.w, e.h, e.iris, e.irisR).status, "no_flash");
});

test("reflex located to sub-pixel precision", () => {
  const e = eye({ reflex: { x: 63.4, y: 38.6 } });
  const r = detectReflex(e.img, e.w, e.h, e.iris, e.irisR);
  assert.equal(r.status, "ok");
  if (r.status === "ok") {
    close(r.pos.x, 63.4, 0.25, "x");
    close(r.pos.y, 38.6, 0.25, "y");
  }
});

test("a reflex below the minimum size is rejected, not guessed", () => {
  const e = eye({ reflex: { x: 62, y: 40, sigma: 1.2 } });   // ~4 px < 0.4% of the iris
  assert.equal(detectReflex(e.img, e.w, e.h, e.iris, e.irisR).status, "no_reflex");
});

test("glare on the skin loses to the reflex on the cornea", () => {
  const base = eye({ reflex: { x: 62, y: 41 } });
  const img = Uint8Array.from(base.img);
  for (let y = 18; y < 24; y++) for (let x = 90; x < 100; x++) img[y * base.w + x] = 255;   // glare patch
  const r = detectReflex(img, base.w, base.h, base.iris, base.irisR);
  assert.equal(r.status, "ok");
  if (r.status === "ok") close(Math.hypot(r.pos.x - 62, r.pos.y - 41), 0, 0.5, "picked the corneal reflex");
});

test("a reflex larger than 25% of the iris is recovered by the rescue pass", () => {
  // One smooth bloom: the top 3% of pixels form a ~280 px disc, which against a
  // 15 px iris radius is ~40% of iris area — over the primary ceiling (25%),
  // under the rescue ceiling (45%).
  const img = image(120, 80, (x, y) => 100 + 150 * Math.exp(-((x - 60) ** 2 + (y - 40) ** 2) / 50));
  const r = detectReflex(img, 120, 80, { x: 60, y: 40 }, 15);
  assert.equal(r.status, "ok");
  if (r.status === "ok") assert.equal(r.rescue, true);
});
