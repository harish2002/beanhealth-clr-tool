/**
 * BeanHealth CLR — exam landmarks in the browser (port of backend/exam/landmarks)
 * ==============================================================================
 *
 * Per frame, per eye, the values a gaze-exam session uploads (the phone sends
 * these, never the frame):
 *   • refineReflex  sub-pixel, intensity-weighted reflex centroid
 *   • LidAperture   the lid opening in the eye's own canthal frame
 *   • fitLimbus     limbus ellipse — radial edges inside the lids, robust conic
 *                   fit, refused when independent restarts disagree
 *   • findPupil     dark pupil (null when the iris is too dark to tell)
 *   • lidMetrics    fissure height, scleral show, limbus coverage
 *   • headPose      yaw / pitch / roll from the face-mesh axes
 *   • sharpness     the frame-QC blur measure
 *
 * Same steps and constants as the Python reference; RANSAC draws from the
 * same seeded mulberry32, so both pick the same samples. Values agree to a
 * small fraction of a pixel (tests/examLandmarks.test.ts checks fixtures the
 * Python wrote). Not bit-exact: cv2's float32 blur and its 1/32-pixel remap
 * grid are reproduced, its float rounding is not. Sharpness is only ever
 * compared with other frames of the same position, so its resampling is an
 * approximation of cv2.resize.
 *
 * Images are GrayRegion views: a block of 8-bit grey pixels at (x0, y0) in a
 * frame of fullW × fullH. Every coordinate is in full-frame pixels; a pixel's
 * centre sits on its integer coordinate, as in cv2.
 */

import { detectReflex } from "./eyeVision.ts";
import type { ExamEllipse, ExamEye, ExamEyeObservation, ExamHeadPose, XY } from "./types";

export const LANDMARKS_VERSION = "1.0.0";

const REFLEX_WINDOW_IRIS_RADII = 0.3;
const REFLEX_LEVEL = 0.5;
const LIMBUS_RAYS = 90;
const LIMBUS_SEARCH_BAND: [number, number] = [0.6, 1.5];
const LIMBUS_SAMPLE_STEP_PX = 0.5;
const LIMBUS_MIN_EDGE_SHARE = 0.25;
const LIMBUS_LID_MARGIN_IRIS_RADII = 0.04;
const LIMBUS_INLIER_IRIS_RADII = 0.02;
const LIMBUS_RANSAC_ITERATIONS = 200;     // per restart
const LIMBUS_MIN_INLIERS = 16;
const LIMBUS_RESTARTS = 3;
const LIMBUS_TUKEY_FACTOR = 2.5;
const LIMBUS_REFINE_ITERATIONS = 15;
/** An eye facing the camera cannot look flatter; a flatter fit has caught a lid. */
export const LIMBUS_MAX_AXIS_RATIO = 1.2;
const LIMBUS_SIDE_WINDOW_DEG = 40;
const PUPIL_REGION_IRIS_RADII = 0.8;
const PUPIL_MAX_OFFSET_IRIS_RADII = 0.6;
const PUPIL_MIN_CONTRAST = 12;
const PUPIL_MIN_RELATIVE_CONTRAST = 0.2;
const PUPIL_DIAMETER_RANGE_IRIS_RADII: [number, number] = [0.15, 1.3];

export interface GrayRegion {
  data: Uint8Array;
  w: number;
  h: number;
  x0: number;
  y0: number;
  fullW: number;
  fullH: number;
}

/** A frame-sized region, for whole images (tests, small frames). */
export function wholeImage(data: Uint8Array, w: number, h: number): GrayRegion {
  return { data, w, h, x0: 0, y0: 0, fullW: w, fullH: h };
}

function px(img: GrayRegion, x: number, y: number): number {
  return img.data[(y - img.y0) * img.w + (x - img.x0)];
}

function covers(img: GrayRegion, x0: number, y0: number, x1: number, y1: number): boolean {
  return x0 >= img.x0 && y0 >= img.y0 && x1 <= img.x0 + img.w && y1 <= img.y0 + img.h;
}

// ── Small numerics ──────────────────────────────────────────────────────────

/** numpy.median. */
export function median(values: ArrayLike<number>): number {
  const a = Array.from(values).sort((p, q) => p - q);
  const n = a.length;
  if (!n) return NaN;
  return n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2;
}

function argmax(values: ArrayLike<number>, from = 0, to = values.length): number {
  let best = from;
  for (let i = from + 1; i < to; i++) if (values[i] > values[best]) best = i;
  return best;
}

/** Seeded PRNG, bit-identical to exam.landmarks.eye.mulberry32. */
export function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sample(n: number, k: number, rand: () => number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let j = 0; j < k; j++) {
    const r = j + Math.floor(rand() * (n - j));
    [idx[j], idx[r]] = [idx[r], idx[j]];
  }
  return idx.slice(0, k);
}

/** cv2.BORDER_REFLECT_101 index. */
function reflect101(i: number, n: number): number {
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

/** cv2.GaussianBlur(src, (0, 0), sigma) for a float image (kernel size round(8σ + 1) | 1). */
export function gaussianBlurFloat(src: Float64Array, w: number, h: number, sigma: number): Float64Array {
  const n = Math.round(sigma * 8 + 1) | 1;
  const c = (n - 1) / 2;
  const k = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) { k[i] = Math.exp(-((i - c) ** 2) / (2 * sigma * sigma)); sum += k[i]; }
  for (let i = 0; i < n; i++) k[i] /= sum;
  const tmp = new Float64Array(w * h);
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += k[i] * src[y * w + reflect101(x + i - c, w)];
      tmp[y * w + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += k[i] * tmp[reflect101(y + i - c, h) * w + x];
      out[y * w + x] = Math.fround(s);       // cv2 keeps float32
    }
  }
  return out;
}

/** cv2.remap INTER_LINEAR with BORDER_REPLICATE (coordinates on cv2's 1/32-pixel grid). */
function sampleBilinear(img: Float64Array, w: number, h: number, x: number, y: number): number {
  const X = Math.round(x * 32), Y = Math.round(y * 32);
  const ix = Math.floor(X / 32), iy = Math.floor(Y / 32);
  const fx = (X - ix * 32) / 32, fy = (Y - iy * 32) / 32;
  const cx0 = Math.min(w - 1, Math.max(0, ix)), cx1 = Math.min(w - 1, Math.max(0, ix + 1));
  const cy0 = Math.min(h - 1, Math.max(0, iy)), cy1 = Math.min(h - 1, Math.max(0, iy + 1));
  return (1 - fy) * ((1 - fx) * img[cy0 * w + cx0] + fx * img[cy0 * w + cx1])
       + fy * ((1 - fx) * img[cy1 * w + cx0] + fx * img[cy1 * w + cx1]);
}

// ── Conic fitting ───────────────────────────────────────────────────────────

/** Solve the n×n system in place (partial pivoting); null when singular. */
function solve(a: number[][], b: number[]): number[] | null {
  const n = b.length;
  const scale = Math.max(...a.map((row) => Math.max(...row.map(Math.abs))));
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    if (Math.abs(a[piv][col]) <= 1e-12 * scale) return null;
    [a[col], a[piv]] = [a[piv], a[col]];
    [b[col], b[piv]] = [b[piv], b[col]];
    for (let r = col + 1; r < n; r++) {
      const f = a[r][col] / a[col][col];
      for (let c = col; c < n; c++) a[r][c] -= f * a[col][c];
      b[r] -= f * b[col];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let c = r + 1; c < n; c++) s -= a[r][c] * x[c];
    x[r] = s / a[r][r];
  }
  return x;
}

/**
 * exam.landmarks.eye.fit_conic_ellipse: (weighted) least squares
 * A x² + B xy + C y² + D x + E y = 1 about the points' (weighted) mean.
 */
export function fitConicEllipse(points: XY[], weights: number[] | null = null): ExamEllipse | null {
  const n = points.length;
  if (n < 5) return null;
  const w = weights ?? new Array<number>(n).fill(1);
  if (w.filter((v) => v !== 0).length < 5) return null;
  let mx = 0, my = 0, sw = 0;
  points.forEach(([x, y], i) => { mx += w[i] * x; my += w[i] * y; sw += w[i]; });
  mx /= sw; my /= sw;
  // Columns scaled by the spread for conditioning; the least-squares answer is unchanged.
  let s2 = 0;
  for (const [x, y] of points) s2 += (x - mx) ** 2 + (y - my) ** 2;
  const s = Math.sqrt(s2 / n) || 1;
  const scales = [s * s, s * s, s * s, s, s];
  const ata = Array.from({ length: 5 }, () => new Array<number>(5).fill(0));
  const atb = new Array<number>(5).fill(0);
  points.forEach(([px0, py0], k) => {
    const x = px0 - mx, y = py0 - my;
    const row = [x * x, x * y, y * y, x, y].map((v, i) => v / scales[i]);
    for (let i = 0; i < 5; i++) {
      atb[i] += w[k] * row[i];
      for (let j = 0; j < 5; j++) ata[i][j] += w[k] * row[i] * row[j];
    }
  });
  const sol = solve(ata, atb);
  if (!sol) return null;
  const [a, b, c, d, e] = sol.map((v, i) => v / scales[i]);
  const det = 4 * a * c - b * b;
  if (det <= 0) return null;
  const x0 = (b * e - 2 * c * d) / det;
  const y0 = (b * d - 2 * a * e) / det;
  const f0 = a * x0 * x0 + b * x0 * y0 + c * y0 * y0 + d * x0 + e * y0 - 1;
  const tr = a + c, dif = Math.hypot(a - c, b);
  const lam = [(tr - dif) / 2, (tr + dif) / 2];
  if (f0 >= 0 || lam[0] <= 0) return null;
  const major = Math.sqrt(-f0 / lam[0]);
  const minor = Math.min(Math.sqrt(-f0 / lam[1]), major);
  const theta = 0.5 * Math.atan2(b, a - c);
  const angle = (((theta * 180) / Math.PI + 90) % 180 + 180) % 180;
  return { centre_px: [x0 + mx, y0 + my], semi_major_px: major, semi_minor_px: minor, angle_deg: angle };
}

/** Approximate signed distance from the ellipse (+ outside), px. */
export function ellipseDistance(ell: ExamEllipse, p: XY): number {
  const phi = (ell.angle_deg * Math.PI) / 180;
  const rx = p[0] - ell.centre_px[0], ry = p[1] - ell.centre_px[1];
  const u = rx * Math.cos(phi) + ry * Math.sin(phi);
  const v = -rx * Math.sin(phi) + ry * Math.cos(phi);
  const rho = Math.hypot(u / ell.semi_major_px, v / ell.semi_minor_px);
  return Math.hypot(u, v) * (1 - 1 / Math.max(rho, 1e-9));
}

export function ellipsePoints(ell: ExamEllipse, n = 360): XY[] {
  const phi = (ell.angle_deg * Math.PI) / 180;
  const out: XY[] = [];
  for (let i = 0; i < n; i++) {
    const t = (2 * Math.PI * i) / n;
    const u = ell.semi_major_px * Math.cos(t), v = ell.semi_minor_px * Math.sin(t);
    out.push([ell.centre_px[0] + u * Math.cos(phi) - v * Math.sin(phi),
              ell.centre_px[1] + u * Math.sin(phi) + v * Math.cos(phi)]);
  }
  return out;
}

function ellipseExtent(ell: ExamEllipse, dir: XY): number {
  const phi = (ell.angle_deg * Math.PI) / 180;
  const e1 = dir[0] * Math.cos(phi) + dir[1] * Math.sin(phi);
  const e2 = -dir[0] * Math.sin(phi) + dir[1] * Math.cos(phi);
  return Math.hypot(ell.semi_major_px * e1, ell.semi_minor_px * e2);
}

// ── Reflex ──────────────────────────────────────────────────────────────────

/** exam.landmarks.eye.refine_reflex. */
export function refineReflex(img: GrayRegion, approx: XY, irisRadius: number): XY | null {
  for (const half of [REFLEX_WINDOW_IRIS_RADII * irisRadius, 2 * REFLEX_WINDOW_IRIS_RADII * irisRadius]) {
    const r = Math.max(3, Math.ceil(half));
    const x0 = Math.floor(approx[0] + 0.5) - r, y0 = Math.floor(approx[1] + 0.5) - r;
    const n = 2 * r + 1;
    if (x0 < 0 || y0 < 0 || x0 + n > img.fullW || y0 + n > img.fullH) return null;
    if (!covers(img, x0, y0, x0 + n, y0 + n)) return null;
    const win = new Float64Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) win[y * n + x] = px(img, x0 + x, y0 + y);
    const background = median(win);
    const c0 = r - Math.floor(r / 2), c1 = r + Math.floor(r / 2) + 1;
    let pyx = c0 * n + c0;
    for (let y = c0; y < c1; y++) for (let x = c0; x < c1; x++) if (win[y * n + x] > win[pyx]) pyx = y * n + x;
    const peak = win[pyx];
    if (peak <= background) return null;
    const level = background + REFLEX_LEVEL * (peak - background);
    // The 8-connected spot above the level that holds the peak.
    const spot = new Uint8Array(n * n);
    const stack = [pyx];
    spot[pyx] = 1;
    let spilled = false;
    while (stack.length) {
      const i = stack.pop() as number;
      const x = i % n, y = (i - x) / n;
      if (x === 0 || y === 0 || x === n - 1 || y === n - 1) spilled = true;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= n || yy >= n) continue;
        const j = yy * n + xx;
        if (!spot[j] && win[j] > level) { spot[j] = 1; stack.push(j); }
      }
    }
    if (spilled) continue;
    let sw = 0, sx = 0, sy = 0;
    for (let i = 0; i < n * n; i++) {
      if (!spot[i]) continue;
      const wgt = win[i] - level, x = i % n;
      sw += wgt; sx += wgt * x; sy += wgt * ((i - x) / n);
    }
    if (sw <= 0) return null;
    return [x0 + sx / sw, y0 + sy / sw];
  }
  return null;
}

// ── Lids ────────────────────────────────────────────────────────────────────

/** exam.landmarks.eye.LidAperture: lid contours compared in the canthal frame. */
export class LidAperture {
  readonly upper: XY[];
  readonly lower: XY[];
  private readonly origin: XY;
  private readonly cos: number;
  private readonly sin: number;
  private readonly ux: number[]; private readonly uy: number[];
  private readonly lx: number[]; private readonly ly: number[];

  constructor(upper: XY[], lower: XY[], canthi: [XY, XY]) {
    this.upper = upper;
    this.lower = lower;
    const [[ax, ay], [bx, by]] = canthi;
    const phi = Math.atan2(by - ay, bx - ax);
    this.origin = [(ax + bx) / 2, (ay + by) / 2];
    this.cos = Math.cos(phi);
    this.sin = Math.sin(phi);
    [this.ux, this.uy] = this.curve(upper);
    [this.lx, this.ly] = this.curve(lower);
  }

  toLocal(p: XY): XY {
    const rx = p[0] - this.origin[0], ry = p[1] - this.origin[1];
    return [rx * this.cos + ry * this.sin, -rx * this.sin + ry * this.cos];
  }

  /** The eye's own "down" in image coordinates. */
  vertical(): XY { return [-this.sin, this.cos]; }

  private curve(contour: XY[]): [number[], number[]] {
    const local = contour.map((p) => this.toLocal(p)).sort((p, q) => p[0] - q[0]);
    return [local.map((p) => p[0]), local.map((p) => p[1])];
  }

  /** Upper and lower lid heights (local y, down +) at local x; NaN outside the canthi. */
  lidY(xLocal: number): [number, number] {
    const lo = Math.max(this.ux[0], this.lx[0]);
    const hi = Math.min(this.ux[this.ux.length - 1], this.lx[this.lx.length - 1]);
    if (!(xLocal >= lo && xLocal <= hi)) return [NaN, NaN];
    return [interp(xLocal, this.ux, this.uy), interp(xLocal, this.lx, this.ly)];
  }

  contains(p: XY, marginPx = 0): boolean {
    const [x, y] = this.toLocal(p);
    const [up, dn] = this.lidY(x);
    return y >= up + marginPx && y <= dn - marginPx;
  }
}

/** numpy.interp for increasing xp. */
function interp(x: number, xp: number[], fp: number[]): number {
  if (x <= xp[0]) return fp[0];
  const last = xp.length - 1;
  if (x >= xp[last]) return fp[last];
  let i = 1;
  while (xp[i] < x) i++;
  const t = (x - xp[i - 1]) / (xp[i] - xp[i - 1]);
  return fp[i - 1] + t * (fp[i] - fp[i - 1]);
}

// ── Limbus ──────────────────────────────────────────────────────────────────

export interface LimbusFit { ellipse: ExamEllipse; rms_px: number; inliers: number; rays: number }

interface Blurred { data: Float64Array; w: number; h: number; x0: number; y0: number }

/** exam.landmarks.eye.blurred_region. */
function blurredRegion(img: GrayRegion, centre: XY, radius: number, sigma: number): Blurred | null {
  const half = Math.ceil(2.2 * radius + 4 * sigma) + 2;
  const x0 = Math.max(0, Math.floor(centre[0]) - half), y0 = Math.max(0, Math.floor(centre[1]) - half);
  const x1 = Math.min(img.fullW, Math.floor(centre[0]) + half + 1);
  const y1 = Math.min(img.fullH, Math.floor(centre[1]) + half + 1);
  if (x1 <= x0 || y1 <= y0 || !covers(img, x0, y0, x1, y1)) return null;
  const w = x1 - x0, h = y1 - y0;
  const crop = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) crop[y * w + x] = px(img, x0 + x, y0 + y);
  return { data: gaussianBlurFloat(crop, w, h, sigma), w, h, x0, y0 };
}

function edgePoints(b: Blurred, centre: XY, radius: number, aperture: LidAperture | null): XY[] {
  const [lo, hi] = LIMBUS_SEARCH_BAND;
  const start = lo * radius;
  const count = Math.ceil((hi * radius - start) / LIMBUS_SAMPLE_STEP_PX);
  const step = LIMBUS_SAMPLE_STEP_PX;
  const found: { p: XY; strength: number }[] = [];
  const prof = new Float64Array(count);
  const grad = new Float64Array(count);
  for (let k = 0; k < LIMBUS_RAYS; k++) {
    const th = k * ((2 * Math.PI) / LIMBUS_RAYS);
    const c = Math.cos(th), s = Math.sin(th);
    for (let i = 0; i < count; i++) {
      const r = start + i * step;
      // cv2.remap takes float32 maps.
      prof[i] = Math.fround(sampleBilinear(b.data, b.w, b.h,
        Math.fround(centre[0] - b.x0 + c * r), Math.fround(centre[1] - b.y0 + s * r)));
    }
    for (let i = 0; i < count; i++) {
      grad[i] = i === 0 ? (prof[1] - prof[0]) / step
        : i === count - 1 ? (prof[i] - prof[i - 1]) / step
        : (prof[i + 1] - prof[i - 1]) / (2 * step);
    }
    const m = argmax(grad);
    const g0 = grad[m];
    const gm = grad[Math.max(0, m - 1)], gp = grad[Math.min(count - 1, m + 1)];
    const denom = gm - 2 * g0 + gp;
    const off = m > 0 && m < count - 1 && denom < 0 ? (0.5 * (gm - gp)) / denom : 0;
    const r = start + m * step + off * step;
    found.push({ p: [centre[0] + r * c, centre[1] + r * s], strength: g0 });
  }
  const positive = found.filter((f) => f.strength > 0);
  if (!positive.length) return [];
  const floor = LIMBUS_MIN_EDGE_SHARE * median(positive.map((f) => f.strength));
  const margin = LIMBUS_LID_MARGIN_IRIS_RADII * radius;
  return positive
    .filter((f) => f.strength >= floor && (!aperture || aperture.contains(f.p, margin)))
    .map((f) => f.p);
}

function plausible(ell: ExamEllipse, radius: number): boolean {
  return ell.semi_major_px <= LIMBUS_MAX_AXIS_RATIO * ell.semi_minor_px
    && ell.semi_major_px >= 0.5 * radius && ell.semi_major_px <= 1.6 * radius;
}

/** The 5-point hypothesis with the lowest truncated squared distance (MSAC). */
function msacStart(pts: XY[], radius: number, tol: number, rand: () => number): ExamEllipse | null {
  let best: ExamEllipse | null = null, bestCost = Infinity;
  for (let it = 0; it < LIMBUS_RANSAC_ITERATIONS; it++) {
    const ell = fitConicEllipse(sample(pts.length, 5, rand).map((i) => pts[i]));
    if (!ell || !plausible(ell, radius)) continue;
    let cost = 0;
    for (const p of pts) { const d = ellipseDistance(ell, p); cost += Math.min(d * d, tol * tol); }
    if (cost < bestCost) { best = ell; bestCost = cost; }
  }
  return best;
}

/** Tukey-biweight reweighted conic fits until the ellipse stops moving. */
function refine(pts: XY[], start: ExamEllipse, radius: number, tol: number): ExamEllipse | null {
  const cut = LIMBUS_TUKEY_FACTOR * tol;
  let ell = start;
  for (let it = 0; it < LIMBUS_REFINE_ITERATIONS; it++) {
    const w = pts.map((p) => { const u = ellipseDistance(ell, p) / cut; return Math.abs(u) < 1 ? (1 - u * u) ** 2 : 0; });
    if (w.filter((v) => v !== 0).length < LIMBUS_MIN_INLIERS) return null;
    const next = fitConicEllipse(pts, w);
    if (!next || !plausible(next, radius)) break;
    const moved = Math.max(Math.hypot(next.centre_px[0] - ell.centre_px[0], next.centre_px[1] - ell.centre_px[1]),
                           Math.abs(next.semi_major_px - ell.semi_major_px));
    ell = next;
    if (moved < 1e-4) break;
  }
  return ell;
}

/** exam.landmarks.eye.robust_ellipse: restarts must agree, else null (ambiguous). */
export function robustEllipse(pts: XY[], radius: number, rand: () => number): ExamEllipse | null {
  const tol = Math.max(0.75, LIMBUS_INLIER_IRIS_RADII * radius);
  const fits: ExamEllipse[] = [];
  for (let k = 0; k < LIMBUS_RESTARTS; k++) {
    const start = msacStart(pts, radius, tol, rand);
    const ell = start ? refine(pts, start, radius, tol) : null;
    if (!ell) return null;
    fits.push(ell);
  }
  const [first] = fits;
  const disagree = fits.slice(1).some((f) =>
    Math.hypot(f.centre_px[0] - first.centre_px[0], f.centre_px[1] - first.centre_px[1]) > tol
    || Math.abs(f.semi_major_px - first.semi_major_px) > tol);
  return disagree ? null : first;
}

function bothSides(ell: ExamEllipse, pts: XY[], horizontal: XY): boolean {
  const ref = Math.atan2(horizontal[1], horizontal[0]);
  let right = false, left = false;
  for (const [x, y] of pts) {
    let a = ((Math.atan2(y - ell.centre_px[1], x - ell.centre_px[0]) - ref) * 180) / Math.PI;
    a = ((a + 180) % 360 + 360) % 360 - 180;
    if (Math.abs(a) <= LIMBUS_SIDE_WINDOW_DEG) right = true;
    if (Math.abs(a) >= 180 - LIMBUS_SIDE_WINDOW_DEG) left = true;
  }
  return right && left;
}

/** exam.landmarks.eye.fit_limbus. */
export function fitLimbus(img: GrayRegion, centreGuess: XY, radiusGuess: number,
                          aperture: LidAperture | null = null, seed = 0): LimbusFit | null {
  const sigma = Math.max(1, radiusGuess / 40);
  const b = blurredRegion(img, centreGuess, radiusGuess, sigma);
  if (!b) return null;
  const rand = mulberry32(seed);
  let horizontal: XY = [1, 0];
  if (aperture) { const [dx, dy] = aperture.vertical(); horizontal = [dy, -dx]; }
  let centre = centreGuess, radius = radiusGuess;
  let result: LimbusFit | null = null;
  for (let pass = 0; pass < 2; pass++) {
    const pts = edgePoints(b, centre, radius, aperture);
    if (pts.length < LIMBUS_MIN_INLIERS) return null;
    const ell = robustEllipse(pts, radius, rand);
    if (!ell) return null;
    const tol = Math.max(0.75, LIMBUS_INLIER_IRIS_RADII * radius);
    const inPts = pts.filter((p) => Math.abs(ellipseDistance(ell, p)) < tol);
    if (inPts.length < LIMBUS_MIN_INLIERS || !bothSides(ell, inPts, horizontal)) return null;
    const ss = inPts.reduce((acc, p) => acc + ellipseDistance(ell, p) ** 2, 0);
    result = { ellipse: ell, rms_px: Math.sqrt(ss / inPts.length), inliers: inPts.length, rays: LIMBUS_RAYS };
    centre = ell.centre_px;
    radius = ell.semi_major_px;
  }
  return result;
}

// ── Pupil ───────────────────────────────────────────────────────────────────

function otsu(values: number[]): number {
  const hist = new Float64Array(256);
  for (const v of values) hist[Math.min(255, Math.max(0, Math.trunc(v)))]++;
  const total = values.length;
  let omega = 0, mu = 0, muT = 0;
  for (let i = 0; i < 256; i++) muT += (i * hist[i]) / total;
  let best = -1, bestT = 0;
  for (let i = 0; i < 256; i++) {
    omega += hist[i] / total;
    mu += (i * hist[i]) / total;
    const den = omega * (1 - omega);
    const between = den > 0 ? (muT * omega - mu) ** 2 / den : -1;
    if (between > best) { best = between; bestT = i; }
  }
  return bestT;
}

/** Andrew's monotone chain; collinear points dropped. */
function convexHull(points: XY[]): XY[] {
  const p = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o: XY, a: XY, b: XY) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: XY[] = [], upper: XY[] = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

/** exam.landmarks.eye.find_pupil: [centre, diameter], or null when the pupil cannot be told from the iris. */
export function findPupil(img: GrayRegion, limbus: ExamEllipse): [XY, number] | null {
  const r = limbus.semi_major_px;
  const [cx, cy] = limbus.centre_px;
  const half = Math.ceil(r) + 2;
  const x0 = Math.floor(cx) - half, y0 = Math.floor(cy) - half, n = 2 * half + 1;
  if (x0 < 0 || y0 < 0 || x0 + n > img.fullW || y0 + n > img.fullH || !covers(img, x0, y0, x0 + n, y0 + n)) return null;
  const raw = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) raw[y * n + x] = px(img, x0 + x, y0 + y);
  const win = gaussianBlurFloat(raw, n, n, Math.max(1, r / 40));
  const shrunk: ExamEllipse = { ...limbus, semi_major_px: PUPIL_REGION_IRIS_RADII * limbus.semi_major_px,
                                semi_minor_px: PUPIL_REGION_IRIS_RADII * limbus.semi_minor_px };
  const region = new Uint8Array(n * n);
  const values: number[] = [];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    if (ellipseDistance(shrunk, [x + x0, y + y0]) <= 0) { region[y * n + x] = 1; values.push(win[y * n + x]); }
  }
  if (values.length < 50) return null;
  const t = otsu(values);
  // 8-connected dark components in raster order; keep the largest near the centre.
  const label = new Int32Array(n * n);
  let bestLabel = 0, bestArea = 0, next = 0;
  for (let i = 0; i < n * n; i++) {
    if (!region[i] || win[i] > t || label[i]) continue;
    next++;
    const stack = [i];
    label[i] = next;
    let area = 0, sx = 0, sy = 0;
    while (stack.length) {
      const j = stack.pop() as number;
      const x = j % n, y = (j - x) / n;
      area++; sx += x; sy += y;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= n || yy >= n) continue;
        const k = yy * n + xx;
        if (region[k] && win[k] <= t && !label[k]) { label[k] = next; stack.push(k); }
      }
    }
    const off = Math.hypot(sx / area + x0 - cx, sy / area + y0 - cy);
    if (off <= PUPIL_MAX_OFFSET_IRIS_RADII * r && area > bestArea) { bestArea = area; bestLabel = next; }
  }
  if (!bestLabel) return null;
  const iris: number[] = [], pupil: number[] = [], blob: XY[] = [];
  for (let i = 0; i < n * n; i++) {
    if (!region[i]) continue;
    const x = i % n;
    if (label[i] === bestLabel) { pupil.push(win[i]); blob.push([x, (i - x) / n]); } else iris.push(win[i]);
  }
  const irisMed = median(iris), contrast = irisMed - median(pupil);
  if (contrast < PUPIL_MIN_CONTRAST || contrast < PUPIL_MIN_RELATIVE_CONTRAST * irisMed) return null;
  // Hull of the blob = hull of its outline; densified so bridging edges weigh as much as arcs.
  const hull = convexHull(blob);
  const dense: XY[] = [];
  hull.forEach((a, i) => {
    const b = hull[(i + 1) % hull.length];
    const steps = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1])));
    for (let k = 0; k < steps; k++) dense.push([a[0] + (k / steps) * (b[0] - a[0]) + x0, a[1] + (k / steps) * (b[1] - a[1]) + y0]);
  });
  const ell = fitConicEllipse(dense);
  if (!ell) return null;
  const diameter = 2 * Math.sqrt(ell.semi_major_px * ell.semi_minor_px);
  const [lo, hi] = PUPIL_DIAMETER_RANGE_IRIS_RADII;
  if (diameter < lo * r || diameter > hi * r) return null;
  return [ell.centre_px, diameter];
}

// ── Lid measures ────────────────────────────────────────────────────────────

export interface LidMetrics { fissure_height_px: number | null; scleral_show_px: number | null; limbus_covered_fraction: number }

/** exam.landmarks.eye.lid_metrics. */
export function lidMetrics(aperture: LidAperture, limbus: ExamEllipse): LidMetrics {
  const [lx, ly] = aperture.toLocal(limbus.centre_px);
  const [up, dn] = aperture.lidY(lx);
  const fissure = Number.isNaN(up) ? null : dn - up;
  const show = Number.isNaN(up) ? null : ly - ellipseExtent(limbus, aperture.vertical()) - up;
  const pts = ellipsePoints(limbus);
  const inside = pts.filter((p) => aperture.contains(p)).length;
  return { fissure_height_px: fissure, scleral_show_px: show, limbus_covered_fraction: 1 - inside / pts.length };
}

// ── Head pose ───────────────────────────────────────────────────────────────

export const RIGHT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
export const LEFT_EYE_RING = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
const FOREHEAD = [10, 151, 9];
const CHIN = [152, 175, 199];

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit = (a: V3): V3 => { const n = Math.hypot(...a); if (!n) throw new Error("degenerate face landmarks"); return [a[0] / n, a[1] / n, a[2] / n]; };
const mean3 = (pts: V3[], idx: number[]): V3 => {
  const s: V3 = [0, 0, 0];
  for (const i of idx) { s[0] += pts[i][0]; s[1] += pts[i][1]; s[2] += pts[i][2]; }
  return [s[0] / idx.length, s[1] / idx.length, s[2] / idx.length];
};

/**
 * exam.landmarks.head_pose.head_pose. Points: ≥ 468 face-mesh landmarks as
 * (x px, y px, z) with z scaled like x (MediaPipe z × frame width).
 * yaw + = face to the patient's right, pitch + = chin up, roll + = right shoulder.
 */
export function headPose(points: V3[]): ExamHeadPose {
  if (points.length < 468) throw new Error("headPose needs at least 468 landmarks");
  const ex = unit(sub(mean3(points, LEFT_EYE_RING), mean3(points, RIGHT_EYE_RING)));
  const down = sub(mean3(points, CHIN), mean3(points, FOREHEAD));
  const d = dot(down, ex);
  const ey = unit([down[0] - d * ex[0], down[1] - d * ex[1], down[2] - d * ex[2]]);
  const ez: V3 = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]];
  const yaw = Math.atan2(ez[0], ez[2]);
  const pitch = Math.asin(Math.max(-1, Math.min(1, ez[1])));
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  // Undo yaw (Ry(yaw)ᵀ), then pitch (Rx(−pitch)ᵀ); what remains of x is the roll.
  const a: V3 = [cy * ex[0] - sy * ex[2], ex[1], sy * ex[0] + cy * ex[2]];
  const xr: V3 = [a[0], cp * a[1] - sp * a[2], sp * a[1] + cp * a[2]];
  const roll = Math.atan2(-xr[1], xr[0]);
  const deg = 180 / Math.PI;
  return { yaw_deg: yaw * deg, pitch_deg: pitch * deg, roll_deg: roll * deg };
}

// ── Sharpness (frame QC) ────────────────────────────────────────────────────

/** Area resampling (cv2 INTER_AREA when shrinking) or bilinear (enlarging), rounded to 8 bits. */
function resize(src: Float64Array, w: number, h: number, scale: number): { data: Float64Array; w: number; h: number } {
  const W = Math.max(1, Math.round(w * scale)), H = Math.max(1, Math.round(h * scale));
  const out = new Float64Array(W * H);
  if (scale < 1) {
    const sx = w / W, sy = h / H;
    const weights = (n: number, N: number, s: number) => Array.from({ length: N }, (_, d) => {
      const a = d * s, b = Math.min(n, (d + 1) * s);
      const ws: [number, number][] = [];
      for (let i = Math.floor(a); i < Math.ceil(b); i++) ws.push([i, (Math.min(b, i + 1) - Math.max(a, i)) / s]);
      return ws;
    });
    const wx = weights(w, W, sx), wy = weights(h, H, sy);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let acc = 0;
      for (const [iy, fy] of wy[y]) for (const [ix, fx] of wx[x]) acc += fy * fx * src[iy * w + ix];
      out[y * W + x] = Math.round(acc);
    }
  } else {
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const fx = Math.min(w - 1, Math.max(0, (x + 0.5) / scale - 0.5));
      const fy = Math.min(h - 1, Math.max(0, (y + 0.5) / scale - 0.5));
      const ix = Math.floor(fx), iy = Math.floor(fy);
      const ax = fx - ix, ay = fy - iy, ix1 = Math.min(w - 1, ix + 1), iy1 = Math.min(h - 1, iy + 1);
      out[y * W + x] = Math.round((1 - ay) * ((1 - ax) * src[iy * w + ix] + ax * src[iy * w + ix1])
                                 + ay * ((1 - ax) * src[iy1 * w + ix] + ax * src[iy1 * w + ix1]));
    }
  }
  return { data: out, w: W, h: H };
}

/**
 * exam.qc.sharpness: variance of the 3×3 Laplacian over a square of side
 * 4 × iris radius round the iris, rescaled so the iris radius is
 * `canonicalRadius` px (the KB's qc_sharpness_canonical_iris_radius_px).
 */
export function sharpness(img: GrayRegion, centre: XY, irisRadius: number, canonicalRadius: number): number | null {
  const half = 2 * irisRadius;
  const x0 = Math.max(0, Math.round(centre[0] - half)), x1 = Math.min(img.fullW, Math.round(centre[0] + half));
  const y0 = Math.max(0, Math.round(centre[1] - half)), y1 = Math.min(img.fullH, Math.round(centre[1] + half));
  if (x1 - x0 < 3 || y1 - y0 < 3 || !covers(img, x0, y0, x1, y1)) return null;
  const w = x1 - x0, h = y1 - y0;
  const crop = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) crop[y * w + x] = px(img, x0 + x, y0 + y);
  const r = resize(crop, w, h, canonicalRadius / irisRadius);
  let s = 0, s2 = 0;
  for (let y = 0; y < r.h; y++) for (let x = 0; x < r.w; x++) {
    const at = (xx: number, yy: number) => r.data[reflect101(yy, r.h) * r.w + reflect101(xx, r.w)];
    const v = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4 * at(x, y);
    s += v; s2 += v * v;
  }
  const n = r.w * r.h;
  return s2 / n - (s / n) ** 2;
}

// ── One eye, one frame ──────────────────────────────────────────────────────

export const IRIS: Record<ExamEye, number[]> = { R: [468, 469, 470, 471, 472], L: [473, 474, 475, 476, 477] };
export const UPPER_LID: Record<ExamEye, number[]> = {
  R: [33, 246, 161, 160, 159, 158, 157, 173, 133], L: [362, 398, 384, 385, 386, 387, 388, 466, 263] };
export const LOWER_LID: Record<ExamEye, number[]> = {
  R: [33, 7, 163, 144, 145, 153, 154, 155, 133], L: [362, 382, 381, 380, 374, 373, 390, 249, 263] };
export const CANTHI: Record<ExamEye, [number, number]> = { R: [33, 133], L: [362, 263] };

export function apertureFor(points: V3[], eye: ExamEye): LidAperture {
  const xy = (i: number): XY => [points[i][0], points[i][1]];
  return new LidAperture(UPPER_LID[eye].map(xy), LOWER_LID[eye].map(xy),
                         [xy(CANTHI[eye][0]), xy(CANTHI[eye][1])]);
}

/** The face-mesh iris centre and radius of an eye. */
export function irisGuess(points: V3[], eye: ExamEye): { centre: XY; radius: number } {
  const [c, ...ring] = IRIS[eye].map((i) => points[i]);
  const radius = ring.reduce((acc, p) => acc + Math.hypot(p[0] - c[0], p[1] - c[1]), 0) / ring.length;
  return { centre: [c[0], c[1]], radius };
}

/** Half-size of the square round the iris centre that every step of eyeObservation reads. */
export function eyeRegionHalf(radius: number): number {
  return Math.ceil(3.2 * radius + 8 * Math.max(1, radius / 40)) + 4;
}

/**
 * exam.landmarks.extract.eye_observation: one eye's observation and its
 * sharpness. `img` must cover eyeRegionHalf(radius) round the iris centre.
 */
export function eyeObservation(img: GrayRegion, points: V3[], eye: ExamEye, canonicalRadius: number
                               ): { obs: ExamEyeObservation; sharpness: number | null } {
  const guess = irisGuess(points, eye);
  const aperture = apertureFor(points, eye);
  const fit = fitLimbus(img, guess.centre, guess.radius, aperture);
  const limbus = fit ? fit.ellipse : null;
  const centre = limbus ? limbus.centre_px : guess.centre;
  const radius = limbus ? limbus.semi_major_px : guess.radius;

  // Coarse reflex: module 3 (percentile threshold + 3-way filter) on the iris crop.
  let reflex: XY | null = null;
  const half = Math.round(2 * radius);
  const cx0 = Math.max(0, Math.round(centre[0]) - half), cy0 = Math.max(0, Math.round(centre[1]) - half);
  const cx1 = Math.min(img.fullW, Math.round(centre[0]) + half + 1), cy1 = Math.min(img.fullH, Math.round(centre[1]) + half + 1);
  if (covers(img, cx0, cy0, cx1, cy1)) {
    const cw = cx1 - cx0, ch = cy1 - cy0;
    const crop = new Uint8Array(cw * ch);
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) crop[y * cw + x] = px(img, cx0 + x, cy0 + y);
    const found = detectReflex(crop, cw, ch, { x: centre[0] - cx0, y: centre[1] - cy0 }, radius);
    if (found.status === "ok") reflex = refineReflex(img, [found.pos.x + cx0, found.pos.y + cy0], radius);
  }

  const obs: ExamEyeObservation = {
    reflex_px: reflex, limbus, pupil_px: null, pupil_diameter_px: null, iris_radius_px: radius,
    fissure_height_px: null, scleral_show_px: null, limbus_covered_fraction: null,
  };
  if (limbus) {
    const pupil = findPupil(img, limbus);
    if (pupil) { obs.pupil_px = pupil[0]; obs.pupil_diameter_px = pupil[1]; }
    const lids = lidMetrics(aperture, limbus);
    obs.fissure_height_px = lids.fissure_height_px;
    obs.scleral_show_px = lids.scleral_show_px;
    obs.limbus_covered_fraction = lids.limbus_covered_fraction;
  }
  return { obs, sharpness: sharpness(img, centre, radius, canonicalRadius) };
}
