/**
 * BeanHealth CLR — eye vision in the browser (port of modules 1–3)
 * ================================================================
 *
 * Mirrors the server's image steps so the live view measures exactly what the
 * server measures:
 *   • module 1 — the padded eye crop around each eye's outline landmarks
 *   • module 2 — the pupil found as the dark pupil disc, not the iris circle
 *     (on a deviated or partly covered eye the visible iris is only an arc and
 *     a circle fit lands off-centre; the pupil stays visible, and Hirschberg
 *     is defined as reflex-vs-pupil)
 *   • module 3 — the corneal light reflex
 *
 * OpenCV steps are reproduced to match cv2's arithmetic where it matters:
 *   • RGB → grey uses cv2's fixed-point weights (bit-exact).
 *   • The 5×5 Gaussian with sigma 0 uses cv2's small-kernel table
 *     [1 4 6 4 1] / 16 with integer rounding and reflect-101 borders.
 *   • Otsu follows cv2's getThreshVal_Otsu_8u loop, quirks included.
 *   • The ellipse centre follows cv2.fitEllipse's first-stage conic fit.
 *   • The reflex threshold is numpy's interpolated percentile, floored as
 *     cv2.threshold does for 8-bit images; circularity uses the traced
 *     contour's length, as cv2.arcLength does.
 *
 * Not ported: the Hough circle estimate. On the server it seeds the
 * dark-pupil search (retrying from the iris landmarks if that fails) and
 * serves as a fallback; here the seed is always the MediaPipe iris centre and
 * the fallback is the iris centre alone. The seed decides which pixels the
 * search thresholds, so when both seeds succeed the two centres can differ by
 * a fraction of a pixel — about 0.9° of asymmetry on the reference photo.
 * Session calibration (module_calibrate) is not ported either.
 *
 * Pure functions: no DOM, no camera. Coordinates are crop pixel indices
 * (a pixel's centre sits on the integer), matching cv2 — and, like the
 * server, landmark-derived points are used as-is in that frame.
 */

import type { Pt } from "./liveMetrics";

/** Dark blob farther than this from the iris centre is not the pupil (constants.py). */
export const DARK_PUPIL_MAX_IRIS_RADII = 0.6;
const PUPIL_AGREEMENT_HIGH_PX = 5;
const FLT_EPSILON = 1.1920929e-7;

// ── Module 1: eye crop ─────────────────────────────────────────────────────

/** Eye-outline landmarks (constants.py LEFT/RIGHT_EYE_BOUNDARY). */
export const EYE_BOUNDARY_A = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
export const EYE_BOUNDARY_B = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
const CROP_PAD_HORIZONTAL = 0.35;
const CROP_PAD_VERTICAL   = 0.50;

export interface Box { x1: number; y1: number; x2: number; y2: number }

/**
 * Padded crop box around one eye — _bounding_box_from_landmarks followed by
 * crop_region. Python int() truncates; coordinates are non-negative here.
 */
export function eyeCropBox(outline: Pt[], imgW: number, imgH: number): Box {
  const xs = outline.map((p) => p.x);
  const ys = outline.map((p) => p.y);
  const x1 = Math.max(0, Math.trunc(Math.min(...xs)));
  const y1 = Math.max(0, Math.trunc(Math.min(...ys)));
  const x2 = Math.min(imgW, Math.trunc(Math.max(...xs)));
  const y2 = Math.min(imgH, Math.trunc(Math.max(...ys)));
  const padX = Math.trunc((x2 - x1) * CROP_PAD_HORIZONTAL);
  const padY = Math.trunc((y2 - y1) * CROP_PAD_VERTICAL);
  return {
    x1: Math.max(0, x1 - padX),
    y1: Math.max(0, y1 - padY),
    x2: Math.min(imgW, x2 + padX),
    y2: Math.min(imgH, y2 + padY),
  };
}

// ── Pixel primitives ───────────────────────────────────────────────────────

/** RGBA → 8-bit grey, bit-exact with cv2.COLOR_RGB2GRAY. */
export function rgbaToGray(rgba: Uint8ClampedArray): Uint8Array {
  const n = rgba.length / 4;
  const out = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[i] = (rgba[j] * 4899 + rgba[j + 1] * 9617 + rgba[j + 2] * 1868 + 8192) >> 14;
  }
  return out;
}

const reflect101 = (i: number, n: number) => (n === 1 ? 0 : i < 0 ? -i : i >= n ? 2 * n - 2 - i : i);

/** cv2.GaussianBlur(gray, (5, 5), 0) on 8-bit input. */
export function gaussianBlur5(src: Uint8Array, w: number, h: number): Uint8Array {
  const k = [1, 4, 6, 4, 1];
  const tmp = new Uint16Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let t = -2; t <= 2; t++) s += k[t + 2] * src[y * w + reflect101(x + t, w)];
      tmp[y * w + x] = s;
    }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let t = -2; t <= 2; t++) s += k[t + 2] * tmp[reflect101(y + t, h) * w + x];
      out[y * w + x] = (s + 128) >> 8;
    }
  }
  return out;
}

/** Otsu threshold of 8-bit values — the loop from cv2's getThreshVal_Otsu_8u. */
export function otsuThreshold(values: ArrayLike<number>): number {
  const n = values.length;
  const hist = new Float64Array(256);
  for (let i = 0; i < n; i++) hist[values[i]]++;
  const scale = 1 / n;
  let mu = 0;
  for (let i = 0; i < 256; i++) mu += i * hist[i];
  mu *= scale;

  let mu1 = 0, q1 = 0, maxSigma = 0, maxVal = 0;
  for (let i = 0; i < 256; i++) {
    const p = hist[i] * scale;
    mu1 *= q1;
    q1 += p;
    const q2 = 1 - q1;
    if (Math.min(q1, q2) < FLT_EPSILON || Math.max(q1, q2) > 1 - FLT_EPSILON) continue;
    mu1 = (mu1 + i * p) / q1;
    const mu2 = (mu - q1 * mu1) / q2;
    const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
    if (sigma > maxSigma) { maxSigma = sigma; maxVal = i; }
  }
  return maxVal;
}

/** numpy.percentile with linear interpolation. */
function percentileLinear(values: number[], pct: number): number {
  const s = [...values].sort((a, b) => a - b);
  const pos = (pct / 100) * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

// 3×3 MORPH_ELLIPSE in OpenCV is the cross.
const CROSS: [number, number][] = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]];

/** Erode (min) or dilate (max) with the 3×3 cross; out-of-image neighbours are ignored, as in cv2. */
function morph(src: Uint8Array, w: number, h: number, dilate: boolean): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = dilate ? 0 : 1;
      for (const [dx, dy] of CROSS) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const s = src[ny * w + nx];
        v = dilate ? Math.max(v, s) : Math.min(v, s);
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

interface Component { pixels: number[]; cx: number; cy: number }

function components8(mask: Uint8Array, w: number, h: number): Component[] {
  const seen = new Uint8Array(w * h);
  const out: Component[] = [];
  const stack: number[] = [];
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    const pixels: number[] = [];
    seen[start] = 1;
    stack.push(start);
    let sx = 0, sy = 0;
    while (stack.length) {
      const idx = stack.pop() as number;
      pixels.push(idx);
      const x = idx % w, y = (idx - x) / w;
      sx += x; sy += y;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if ((dx || dy) && nx >= 0 && ny >= 0 && nx < w && ny < h) {
            const n = ny * w + nx;
            if (mask[n] && !seen[n]) { seen[n] = 1; stack.push(n); }
          }
        }
      }
    }
    out.push({ pixels, cx: sx / pixels.length, cy: sy / pixels.length });
  }
  return out;
}

// ── Contour + ellipse (cv2.findContours EXTERNAL/SIMPLE + cv2.fitEllipse) ──

// Moore neighbourhood, clockwise from west (y grows downward).
const DIRS: [number, number][] = [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]];

/** Outer boundary of one blob, traced in order, with straight runs collapsed to their ends. */
export function traceContour(mask: Uint8Array, w: number, h: number): Pt[] {
  let start = -1;
  for (let i = 0; i < mask.length; i++) if (mask[i]) { start = i; break; }
  if (start < 0) return [];
  const on = (x: number, y: number) => x >= 0 && y >= 0 && x < w && y < h && mask[y * w + x] === 1;

  const sx = start % w, sy = (start - sx) / w;
  const pts: Pt[] = [{ x: sx, y: sy }];
  let cx = sx, cy = sy;
  let back = 0; // the pixel to the west of the first raster pixel is background
  let first: Pt | null = null;

  for (let guard = 0; guard < 4 * mask.length; guard++) {
    let found = -1;
    for (let k = 1; k <= 8; k++) {
      const d = (back + k) % 8;
      if (on(cx + DIRS[d][0], cy + DIRS[d][1])) { found = d; break; }
    }
    if (found < 0) break;                                  // isolated pixel
    const nx = cx + DIRS[found][0], ny = cy + DIRS[found][1];
    if (first && cx === sx && cy === sy && nx === first.x && ny === first.y) break;
    if (!first) first = { x: nx, y: ny };
    // New backtrack: the last background neighbour examined, seen from the new pixel.
    const bpx = cx + DIRS[(found + 7) % 8][0], bpy = cy + DIRS[(found + 7) % 8][1];
    back = DIRS.findIndex(([dx, dy]) => dx === bpx - nx && dy === bpy - ny);
    cx = nx; cy = ny;
    if (cx === sx && cy === sy) continue;
    pts.push({ x: cx, y: cy });
  }

  // CHAIN_APPROX_SIMPLE: keep only points where the step direction changes.
  if (pts.length < 3) return pts;
  const keep: Pt[] = [];
  for (let i = 0; i < pts.length; i++) {
    const a = pts[(i - 1 + pts.length) % pts.length], b = pts[i], c = pts[(i + 1) % pts.length];
    if (i === 0 || b.x - a.x !== c.x - b.x || b.y - a.y !== c.y - b.y) keep.push(b);
  }
  return keep;
}

/** Solve a small dense linear system (Gaussian elimination, partial pivoting). */
function solve(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

/**
 * Ellipse centre as cv2.fitEllipse computes it: least-squares conic
 * −A x² − B y² − C xy + D x + E y = 1e4 on centred, scaled points, then the
 * centre from the conic's gradient. (cv2's second stage only refits the axes.)
 */
export function fitEllipseCentre(points: Pt[]): Pt | null {
  const n = points.length;
  if (n < 5) return null;
  const c = points.reduce((s, p) => ({ x: s.x + p.x / n, y: s.y + p.y / n }), { x: 0, y: 0 });
  const s = points.reduce((acc, p) => acc + Math.abs(p.x - c.x) + Math.abs(p.y - c.y), 0);
  const scale = 100 / (s > FLT_EPSILON ? s : FLT_EPSILON);

  const AtA = Array.from({ length: 5 }, () => new Array(5).fill(0));
  const Atb = new Array(5).fill(0);
  for (const p of points) {
    const px = (p.x - c.x) * scale, py = (p.y - c.y) * scale;
    const row = [-px * px, -py * py, -px * py, px, py];
    for (let i = 0; i < 5; i++) {
      Atb[i] += row[i] * 10000;
      for (let j = 0; j < 5; j++) AtA[i][j] += row[i] * row[j];
    }
  }
  const g = solve(AtA, Atb);
  if (!g) return null;
  const r = solve([[2 * g[0], g[2]], [g[2], 2 * g[1]]], [g[3], g[4]]);
  if (!r || !r.every(Number.isFinite)) return null;
  return { x: r[0] / scale + c.x, y: r[1] / scale + c.y };
}

// ── The dark-pupil method (_dark_pupil_centre) ─────────────────────────────

/**
 * Locate the pupil as the dark disc near `seed`. Returns crop pixel-index
 * coordinates, or null if no plausible pupil blob is found.
 */
export function darkPupilCentre(gray: Uint8Array, w: number, h: number, seed: Pt, irisR: number): Pt | null {
  if (irisR < 4) return null;
  const blurred = gaussianBlur5(gray, w, h);
  const r2 = (irisR * 1.2) ** 2;

  const discVals: number[] = [];
  const inDisc = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if ((x - seed.x) ** 2 + (y - seed.y) ** 2 <= r2) {
        inDisc[y * w + x] = 1;
        discVals.push(blurred[y * w + x]);
      }
    }
  }
  if (discVals.length < 20) return null;

  // Two-stage Otsu: t1 splits eye from skin/sclera, t2 (over the eye pixels
  // only) splits the dark pupil from the lighter iris.
  const t1 = otsuThreshold(discVals);
  const eyeVals = discVals.filter((v) => v < t1);
  const pupilThresh = eyeVals.length >= 20 ? otsuThreshold(eyeVals) : percentileLinear(discVals, 12);

  let dark: Uint8Array = new Uint8Array(w * h);
  for (let i = 0; i < dark.length; i++) dark[i] = inDisc[i] && blurred[i] <= pupilThresh ? 1 : 0;

  // Open once (drop lashes), close twice (fill gaps, e.g. around the reflex).
  dark = morph(morph(dark, w, h, false), w, h, true);
  dark = morph(morph(dark, w, h, true), w, h, true);
  dark = morph(morph(dark, w, h, false), w, h, false);

  const irisArea = Math.PI * irisR * irisR;
  let best: Component | null = null;
  let bestScore = Infinity;
  for (const comp of components8(dark, w, h)) {
    const area = comp.pixels.length;
    if (area < 0.02 * irisArea || area > 0.9 * irisArea) continue;
    const score = Math.hypot(comp.cx - seed.x, comp.cy - seed.y) - 0.02 * Math.sqrt(area);
    if (score < bestScore) { bestScore = score; best = comp; }
  }
  if (!best) return null;

  let cx = best.cx, cy = best.cy;

  // Darkness-weighted centroid: lands on the pupil's intensity minimum instead
  // of wherever the threshold happened to cut. Adopt only if close to the blob.
  let maxV = 0;
  for (const idx of best.pixels) maxV = Math.max(maxV, blurred[idx]);
  let sw = 0, sx = 0, sy = 0;
  for (const idx of best.pixels) {
    const wt = maxV - blurred[idx];
    const x = idx % w;
    sw += wt; sx += x * wt; sy += ((idx - x) / w) * wt;
  }
  if (sw > 0 && Math.hypot(sx / sw - cx, sy / sw - cy) <= irisR * 0.35) { cx = sx / sw; cy = sy / sw; }

  // Ellipse fit recovers the centre from a partly covered pupil.
  const comp = new Uint8Array(w * h);
  for (const idx of best.pixels) comp[idx] = 1;
  const e = fitEllipseCentre(traceContour(comp, w, h));
  if (e && Math.hypot(e.x - cx, e.y - cy) <= irisR * 0.5) { cx = e.x; cy = e.y; }

  if (cx < 0 || cy < 0 || cx > w || cy > h) return null;
  return { x: cx, y: cy };
}

// ── Per-eye decision (_localise_one_eye, without Hough) ───────────────────

export type PupilConfidence = "HIGH" | "MEDIUM" | "LOW";

export interface PupilResult {
  /** Pupil centre in crop pixel-index coordinates. */
  centre: Pt;
  /** "dark" = found as the dark pupil disc; "iris" = fell back to the iris centre. */
  source: "dark" | "iris";
  confidence: PupilConfidence;
}

/**
 * `irisCentre` is the MediaPipe iris-landmark mean in crop coordinates. As on
 * the server, a dark blob more than 0.6 iris radii from it is an eyelash,
 * eyeliner or brow shadow, and is rejected in favour of the iris centre.
 */
export function localisePupil(gray: Uint8Array, w: number, h: number, irisCentre: Pt, irisR: number): PupilResult {
  const dark = darkPupilCentre(gray, w, h, irisCentre, irisR);
  if (dark) {
    const gap = Math.hypot(dark.x - irisCentre.x, dark.y - irisCentre.y);
    if (gap <= DARK_PUPIL_MAX_IRIS_RADII * irisR) {
      return {
        centre: dark,
        source: "dark",
        confidence: gap < PUPIL_AGREEMENT_HIGH_PX ? "HIGH" : "MEDIUM",
      };
    }
  }
  // Server fuses the iris centre with a Hough estimate here; without Hough the
  // iris centre alone is its LOW-confidence fallback.
  return { centre: irisCentre, source: "iris", confidence: "LOW" };
}

// ── Module 3: corneal light reflex ─────────────────────────────────────────

const CLR_PERCENTILE             = 97;     // top 3% brightest pixels
export const CLR_MIN_PEAK        = 235;    // peak below this → no torch reflex
const CLR_MIN_AREA_RATIO         = 0.004;  // of iris area
const CLR_MAX_AREA_RATIO         = 0.25;
const CLR_MIN_CIRCULARITY        = 0.35;
const CLR_LOCATION_MARGIN        = 0.10;   // reflex must sit in the central 80% of the crop
const CLR_MAX_DIST_IRIS_RADII    = 1.5;    // reflex must sit on the cornea
const CLR_RESCUE_MAX_AREA_RATIO  = 0.45;   // rescue pass for bloomed reflexes…
const CLR_RESCUE_MIN_CIRCULARITY = 0.20;   // …which relaxes size and shape, never location

/** numpy.percentile (linear interpolation) over 8-bit values, via a histogram. */
function percentileU8(values: Uint8Array, pct: number): number {
  const hist = new Uint32Array(256);
  for (let i = 0; i < values.length; i++) hist[values[i]]++;
  const valueAt = (k: number) => {
    let cum = 0;
    for (let v = 0; v < 256; v++) { cum += hist[v]; if (cum > k) return v; }
    return 255;
  };
  const pos = (pct / 100) * (values.length - 1);
  const lo = Math.floor(pos);
  const a = valueAt(lo);
  const b = valueAt(Math.min(lo + 1, values.length - 1));
  return a + (b - a) * (pos - lo);
}

/** cv2.arcLength(contour, closed=True). */
function arcLengthClosed(pts: Pt[]): number {
  if (pts.length < 2) return 0;
  let len = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i], q = pts[(i - 1 + pts.length) % pts.length];
    len += Math.hypot(p.x - q.x, p.y - q.y);
  }
  return len;
}

interface ReflexBlob { x: number; y: number; area: number; circularity: number }

function reflexBlobs(mask: Uint8Array, gray: Uint8Array, w: number): ReflexBlob[] {
  const h = mask.length / w;
  return components8(mask, w, h).map((comp) => {
    // Intensity-weighted centroid above the blob's own floor lands on the
    // reflex peak instead of the threshold's geometric middle (as module 3).
    let floor = 255;
    for (const idx of comp.pixels) floor = Math.min(floor, gray[idx]);
    let sw = 0, sx = 0, sy = 0, minX = w, minY = h, maxX = 0, maxY = 0;
    for (const idx of comp.pixels) {
      const x = idx % w, y = (idx - x) / w, wt = gray[idx] - floor;
      sw += wt; sx += x * wt; sy += y * wt;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
    // Circularity from this blob's own outer contour (1 px zero border, like
    // findContours treats the image edge).
    const bw = maxX - minX + 3, bh = maxY - minY + 3;
    const local = new Uint8Array(bw * bh);
    for (const idx of comp.pixels) {
      const x = idx % w, y = (idx - x) / w;
      local[(y - minY + 1) * bw + (x - minX + 1)] = 1;
    }
    const area = comp.pixels.length;
    const perimeter = arcLengthClosed(traceContour(local, bw, bh));
    return {
      x: sw > 0 ? sx / sw : comp.cx,
      y: sw > 0 ? sy / sw : comp.cy,
      area,
      circularity: perimeter > 0 ? Math.min(1, (4 * Math.PI * area) / (perimeter * perimeter)) : 0,
    };
  });
}

export type ReflexResult =
  | { status: "ok";        pos: Pt; area: number; circularity: number; candidates: number; rescue: boolean; peak: number }
  | { status: "no_flash";  peak: number }
  | { status: "no_reflex"; peak: number };

/**
 * Find the corneal light reflex in an eye crop (_detect_clr_one_eye).
 * `pupil` and `irisR` are in crop coordinates.
 */
export function detectReflex(gray: Uint8Array, w: number, h: number, pupil: Pt | null, irisR: number): ReflexResult {
  let peak = 0;
  for (let i = 0; i < gray.length; i++) if (gray[i] > peak) peak = gray[i];
  if (peak < CLR_MIN_PEAK) return { status: "no_flash", peak };

  // Adaptive threshold, never a fixed brightness. cv2.threshold floors the
  // threshold for 8-bit input and keeps pixels strictly above it.
  const thr = Math.floor(percentileU8(gray, CLR_PERCENTILE));
  const mask = new Uint8Array(gray.length);
  if (thr < 255) for (let i = 0; i < gray.length; i++) mask[i] = gray[i] > thr ? 1 : 0;

  const blobs = reflexBlobs(mask, gray, w);
  if (!blobs.length) return { status: "no_reflex", peak };

  const irisArea = Math.PI * irisR * irisR;
  const xMin = w * CLR_LOCATION_MARGIN, xMax = w * (1 - CLR_LOCATION_MARGIN);
  const yMin = h * CLR_LOCATION_MARGIN, yMax = h * (1 - CLR_LOCATION_MARGIN);
  const passes = (b: ReflexBlob, maxAreaRatio: number, minCirc: number) =>
    xMin < b.x && b.x < xMax && yMin < b.y && b.y < yMax &&
    CLR_MIN_AREA_RATIO * irisArea < b.area && b.area < maxAreaRatio * irisArea &&
    b.circularity > minCirc &&
    (!pupil || Math.hypot(b.x - pupil.x, b.y - pupil.y) <= irisR * CLR_MAX_DIST_IRIS_RADII);

  let passing = blobs.filter((b) => passes(b, CLR_MAX_AREA_RATIO, CLR_MIN_CIRCULARITY));
  let rescue = false;
  if (!passing.length) {
    passing = blobs.filter((b) => passes(b, CLR_RESCUE_MAX_AREA_RATIO, CLR_RESCUE_MIN_CIRCULARITY));
    rescue = passing.length > 0;
  }
  if (!passing.length) return { status: "no_reflex", peak };

  // Several candidates → nearest the pupil, so the choice is stable frame to frame.
  const best = passing.length === 1 ? passing[0]
    : pupil ? passing.reduce((a, b) =>
        Math.hypot(b.x - pupil.x, b.y - pupil.y) < Math.hypot(a.x - pupil.x, a.y - pupil.y) ? b : a)
    : passing.reduce((a, b) => (b.area > a.area ? b : a));

  return {
    status: "ok",
    pos: { x: best.x, y: best.y },
    area: best.area,
    circularity: best.circularity,
    candidates: passing.length,
    rescue,
    peak,
  };
}
