/**
 * BeanHealth CLR — live measurement maths (browser port of modules 3–6 and 8)
 * ===========================================================================
 *
 * Runs on every camera frame to give the screener a real-time readout.
 *
 * PREVIEW ONLY. The reported result still comes from the server pipeline,
 * which is versioned and covered by the backend test suite. Everything here
 * mirrors the server's maths so the two agree, and the constants below are
 * copied from backend/utils/constants.py — keep them in sync.
 *
 * Pure functions only: no DOM, no camera, no React. The component feeds in
 * landmark coordinates and pixel luminance and gets numbers back.
 *
 * Eye naming: frames from getUserMedia are NOT mirrored, so the eye on the
 * image's left is the patient's RIGHT eye (OD) and the image-right eye is the
 * patient's LEFT eye (OS). This holds for both the rear and front cameras.
 */

export interface Pt {
  x: number;
  y: number;
}

// ── Constants (mirror backend/utils/constants.py) ──────────────────────────

export const HIRSCHBERG_DEG_PER_MM = 7.0;
export const IRIS_RADIUS_MM        = 5.75;

export const SEVERITY_MILD_DEG     = 5.0;
export const SEVERITY_MODERATE_DEG = 15.0;
export const SEVERITY_SEVERE_DEG   = 30.0;

const CLR_PERCENTILE             = 97;     // top 3% brightest pixels
export const CLR_MIN_PEAK        = 235;    // peak below this → no torch reflex
const CLR_MIN_AREA_RATIO         = 0.004;  // of iris area
const CLR_MAX_AREA_RATIO         = 0.25;
const CLR_MIN_CIRCULARITY        = 0.35;
const CLR_MAX_DIST_IRIS_RADII    = 1.5;    // reflex must sit on the cornea
const CLR_RESCUE_MAX_AREA_RATIO  = 0.45;   // rescue pass for bloomed reflexes
const CLR_RESCUE_MIN_CIRCULARITY = 0.20;

const ALIGN_H_ALIGNED_MAX    = 0.05;
const ALIGN_H_BORDERLINE_MAX = 0.10;
const ALIGN_V_ALIGNED_MAX    = 0.06;
const ALIGN_V_BORDERLINE_MAX = 0.12;

const STD_HIGH_THRESHOLD                  = 1.0;  // degrees
const STD_MEDIUM_THRESHOLD                = 3.0;
const VARIANCE_INCONCLUSIVE_STD_THRESHOLD = 2.5;  // unstable when asymmetry ≥ mild

/** Frames with a reflex in both eyes needed before a Hirschberg reading shows. */
export const MIN_REFLEX_FRAMES = 5;

// ── Pixel helpers ───────────────────────────────────────────────────────────

/** RGBA → luminance with the same weights as cv2.COLOR_RGB2GRAY. */
export function toLuminance(rgba: Uint8ClampedArray): Float32Array {
  const n   = rgba.length / 4;
  const out = new Float32Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  }
  return out;
}

/** Percentile of 0–255 values via a histogram — O(n), no sort. */
function percentile255(values: Float32Array, pct: number): number {
  const hist = new Uint32Array(256);
  for (let i = 0; i < values.length; i++) hist[Math.min(255, Math.round(values[i]))]++;
  const target = (pct / 100) * values.length;
  let cum = 0;
  for (let v = 0; v < 256; v++) {
    cum += hist[v];
    if (cum >= target) return v;
  }
  return 255;
}

// ── Module 3 port: corneal light reflex detection ──────────────────────────

interface Blob {
  x: number;
  y: number;
  area: number;
  circularity: number;
}

export type ReflexResult =
  | { status: "ok";        pos: Pt; area: number; circularity: number; candidates: number; peak: number }
  | { status: "no_flash";  peak: number }
  | { status: "no_reflex"; peak: number };

/** 8-connected components with intensity-weighted sub-pixel centroids. */
function findBlobs(mask: Uint8Array, lum: Float32Array, w: number, h: number): Blob[] {
  const labels = new Int32Array(w * h);
  const blobs: Blob[] = [];
  const stack: number[] = [];
  let next = 1;

  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || labels[start]) continue;

    const pixels: number[] = [];
    labels[start] = next;
    stack.push(start);
    while (stack.length) {
      const idx = stack.pop() as number;
      pixels.push(idx);
      const px = idx % w;
      const py = (idx - px) / w;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = px + dx;
          const ny = py + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const nIdx = ny * w + nx;
          if (mask[nIdx] && !labels[nIdx]) {
            labels[nIdx] = next;
            stack.push(nIdx);
          }
        }
      }
    }

    // Perimeter ≈ pixels touching the outside on a 4-neighbour; for a disc
    // this gives 4πA/P² ≈ 1, matching cv2's contour-based circularity.
    let perimeter = 0;
    let floor = Infinity;
    for (const idx of pixels) {
      const px = idx % w;
      const py = (idx - px) / w;
      const edge =
        px === 0 || py === 0 || px === w - 1 || py === h - 1 ||
        !mask[idx - 1] || !mask[idx + 1] || !mask[idx - w] || !mask[idx + w];
      if (edge) perimeter++;
      if (lum[idx] < floor) floor = lum[idx];
    }

    // Weight each pixel by its brightness above the blob's own floor so the
    // centroid lands on the reflex peak, not its geometric middle (as module 3).
    let sw = 0, sx = 0, sy = 0, bx = 0, by = 0;
    for (const idx of pixels) {
      const px = idx % w;
      const py = (idx - px) / w;
      const wt = lum[idx] - floor;
      sw += wt; sx += px * wt; sy += py * wt;
      bx += px; by += py;
    }
    const area = pixels.length;
    blobs.push({
      x: sw > 0 ? sx / sw : bx / area,
      y: sw > 0 ? sy / sw : by / area,
      area,
      circularity: Math.min(1, (4 * Math.PI * area) / Math.max(1, perimeter * perimeter)),
    });
    next++;
  }
  return blobs;
}

/**
 * Find the corneal light reflex in a luminance patch centred on one eye.
 * `pupil` and `irisR` are in patch pixel coordinates.
 */
export function detectReflex(
  lum: Float32Array, w: number, h: number, pupil: Pt, irisR: number,
): ReflexResult {
  let peak = 0;
  for (let i = 0; i < lum.length; i++) if (lum[i] > peak) peak = lum[i];
  if (peak < CLR_MIN_PEAK) return { status: "no_flash", peak };

  // Adaptive threshold, never a fixed brightness. A heavily bloomed reflex
  // can push the 97th percentile to saturation, so include equality there.
  const thr = percentile255(lum, CLR_PERCENTILE);
  const mask = new Uint8Array(lum.length);
  for (let i = 0; i < lum.length; i++) {
    mask[i] = (thr >= 254 ? lum[i] >= thr : lum[i] > thr) ? 1 : 0;
  }

  const blobs    = findBlobs(mask, lum, w, h);
  const irisArea = Math.PI * irisR * irisR;
  const dist     = (b: Blob) => Math.hypot(b.x - pupil.x, b.y - pupil.y);

  const strict = blobs.filter((b) => {
    const ratio = b.area / irisArea;
    return ratio >= CLR_MIN_AREA_RATIO && ratio <= CLR_MAX_AREA_RATIO &&
           b.circularity >= CLR_MIN_CIRCULARITY &&
           dist(b) <= irisR * CLR_MAX_DIST_IRIS_RADII;
  });
  const passing = strict.length ? strict : blobs.filter((b) => {
    const ratio = b.area / irisArea;
    return ratio >= CLR_MIN_AREA_RATIO && ratio <= CLR_RESCUE_MAX_AREA_RATIO &&
           b.circularity >= CLR_RESCUE_MIN_CIRCULARITY;
  });
  if (!passing.length) return { status: "no_reflex", peak };

  // Several candidates → nearest the pupil, so the choice is stable frame to frame.
  const best = passing.reduce((a, b) => (dist(b) < dist(a) ? b : a));
  return {
    status: "ok",
    pos: { x: best.x, y: best.y },
    area: best.area,
    circularity: best.circularity,
    candidates: passing.length,
    peak,
  };
}

// ── Per-frame geometry (modules 4, 5, 8) ───────────────────────────────────

export type Direction = "nasal" | "temporal" | "superior" | "inferior";

export interface EyeInput {
  pupil:  Pt;         // video pixels
  irisR:  number;     // video pixels
  reflex: Pt | null;  // video pixels, null when not found
  inner:  Pt;         // medial canthus
  outer:  Pt;         // lateral canthus
}

export interface EyeReflexReading {
  nasal:    number;   // reflex offset along the nasal axis, in iris radii
  superior: number;   // reflex offset along the superior axis, in iris radii
}

export interface FrameMeasurement {
  ipdPx:    number;
  ipdMm:    number;
  rollDeg:  number;   // tilt of the inter-pupil line (head roll)
  degPerPx: number;   // one pixel of reflex movement, in degrees
  od: EyeReflexReading | null;
  os: EyeReflexReading | null;
  asymDeg:  number | null;
  methodB: { hOD: number; vOD: number; hOS: number; vOS: number };
}

const dot  = (a: Pt, b: Pt) => a.x * b.x + a.y * b.y;
const sub  = (a: Pt, b: Pt): Pt => ({ x: a.x - b.x, y: a.y - b.y });
const norm = (a: Pt) => Math.hypot(a.x, a.y);

/**
 * Pupil position along the inner→outer canthus axis (h: 0 = inner, 1 = outer)
 * and its perpendicular offset from that axis (v, positive = toward the chin),
 * both as fractions of the canthus width.
 *
 * The sign of v is fixed to the face's own "down", so it means the same thing
 * in both eyes. (module8 takes a raw cross product against each eye's own
 * inner→outer axis; those axes point opposite ways, which flips its v sign
 * between the eyes.)
 */
function projectPupil(pupil: Pt, inner: Pt, outer: Pt, down: Pt): { h: number; v: number } {
  const axis  = sub(outer, inner);
  const width = norm(axis);
  if (width < 1e-3) return { h: NaN, v: NaN };
  const rel   = sub(pupil, inner);
  const h     = dot(rel, axis) / (width * width);
  const cross = axis.x * rel.y - axis.y * rel.x;
  const sign  = Math.sign(axis.x * down.y - axis.y * down.x) || 1;
  return { h, v: (sign * cross) / (width * width) };
}

export function measureFrame(od: EyeInput, os: EyeInput): FrameMeasurement {
  // Face-relative axes: u runs from the patient's right eye to their left eye,
  // "up" is perpendicular to it. Measuring along these instead of the image
  // axes keeps a tilted head from leaking into the readings.
  const ipdVec = sub(os.pupil, od.pupil);
  const ipdPx  = norm(ipdVec);
  const u:    Pt = { x: ipdVec.x / ipdPx, y: ipdVec.y / ipdPx };
  const up:   Pt = { x: u.y, y: -u.x };
  const down: Pt = { x: -u.y, y: u.x };

  const rMean = (od.irisR + os.irisR) / 2;

  // Nasal means "toward the other eye", which is independent of mirroring.
  const reading = (eye: EyeInput, nasalDir: Pt): EyeReflexReading | null => {
    if (!eye.reflex) return null;
    const d = sub(eye.reflex, eye.pupil);
    return { nasal: dot(d, nasalDir) / eye.irisR, superior: dot(d, up) / eye.irisR };
  };
  const odR = reading(od, u);
  const osR = reading(os, { x: -u.x, y: -u.y });

  // Bilateral asymmetry vector: kappa is mirrored between the eyes, so it
  // cancels in the difference (module 5).
  const asymDeg = odR && osR
    ? Math.hypot(odR.nasal - osR.nasal, odR.superior - osR.superior) *
      IRIS_RADIUS_MM * HIRSCHBERG_DEG_PER_MM
    : null;

  const pOD = projectPupil(od.pupil, od.inner, od.outer, down);
  const pOS = projectPupil(os.pupil, os.inner, os.outer, down);

  return {
    ipdPx,
    ipdMm:    ipdPx * (IRIS_RADIUS_MM / rMean),
    rollDeg:  (Math.atan2(u.y, u.x) * 180) / Math.PI,
    degPerPx: (IRIS_RADIUS_MM * HIRSCHBERG_DEG_PER_MM) / rMean,
    od: odR,
    os: osR,
    asymDeg,
    methodB: { hOD: pOD.h, vOD: pOD.v, hOS: pOS.h, vOS: pOS.v },
  };
}

// ── Classification (module 6, standard Hirschberg convention) ──────────────

export type Tier     = "NORMAL" | "MONITOR" | "ROUTINE" | "URGENT";
export type Severity = "NORMAL" | "MILD" | "MODERATE" | "SEVERE";

export function severityOf(deg: number): { severity: Severity; tier: Tier } {
  if (deg >= SEVERITY_SEVERE_DEG)   return { severity: "SEVERE",   tier: "URGENT"  };
  if (deg >= SEVERITY_MODERATE_DEG) return { severity: "MODERATE", tier: "ROUTINE" };
  if (deg >= SEVERITY_MILD_DEG)     return { severity: "MILD",     tier: "MONITOR" };
  return { severity: "NORMAL", tier: "NORMAL" };
}

/** Prism dioptres from degrees: 100·tan(θ), not the linear shortcut. */
export const toPrismDioptres = (deg: number) => 100 * Math.tan((deg * Math.PI) / 180);

export function directionOf(r: EyeReflexReading): Direction {
  if (Math.abs(r.nasal) >= Math.abs(r.superior)) return r.nasal >= 0 ? "nasal" : "temporal";
  return r.superior >= 0 ? "superior" : "inferior";
}

/**
 * Standard Hirschberg reading: the reflex stays put while the eye turns, so
 * it lands on the side OPPOSITE the deviation. An inward-turned eye shows a
 * temporally displaced reflex.
 */
export const CONDITION_BY_REFLEX: Record<Direction, { name: string; icd: string; eyeTurns: string }> = {
  temporal: { name: "Esotropia",   icd: "H50.01", eyeTurns: "inward"   },
  nasal:    { name: "Exotropia",   icd: "H50.11", eyeTurns: "outward"  },
  inferior: { name: "Hypertropia", icd: "H50.21", eyeTurns: "upward"   },
  superior: { name: "Hypotropia",  icd: "H50.22", eyeTurns: "downward" },
};

// ── Rolling aggregation (module_aggregate) ─────────────────────────────────

export type Confidence = "HIGH" | "MEDIUM" | "LOW";
export type AlignVerdict = "ALIGNED" | "BORDERLINE" | "ASYMMETRIC";

export interface LiveAggregate {
  frames:       number;
  reflexFrames: number;              // frames with a reflex in both eyes
  odReflexRate: number;              // 0–1
  osReflexRate: number;
  ipdMm:        number;
  ipdPx:        number;
  rollDeg:      number;
  degPerPx:     number;
  od: (EyeReflexReading & { mm: number; deg: number; direction: Direction }) | null;
  os: (EyeReflexReading & { mm: number; deg: number; direction: Direction }) | null;
  hirschberg: null | {
    asymDeg:    number;
    stdDeg:     number;
    pd:         number;
    confidence: Confidence;
    unstable:   boolean;             // high variance at a significant angle
    severity:   Severity;
    tier:       Tier;
    deviatingEye: "OD" | "OS" | null;
    reflexDirection: Direction | null;
    /** False when the pattern flips between frames or both eyes are equally displaced. */
    directionReliable: boolean;
    /** Share of frames that agree with the reported eye + direction (0–1). */
    directionAgreement: number;
  };
  methodB: {
    hOD: number; vOD: number; hOS: number; vOS: number;
    hAsym: number; vAsym: number;
    verdict: AlignVerdict;
  };
}

export function median(xs: number[]): number {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Population standard deviation, as numpy's default. */
export function stdDev(xs: number[]): number {
  if (xs.length < 2) return 0;
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length);
}

export function aggregate(samples: FrameMeasurement[]): LiveAggregate | null {
  if (!samples.length) return null;

  const eyeMedian = (pick: (s: FrameMeasurement) => EyeReflexReading | null) => {
    const rs = samples.map(pick).filter((r): r is EyeReflexReading => r !== null);
    if (!rs.length) return null;
    const r = { nasal: median(rs.map((x) => x.nasal)), superior: median(rs.map((x) => x.superior)) };
    const mm = Math.hypot(r.nasal, r.superior) * IRIS_RADIUS_MM;
    return { ...r, mm, deg: mm * HIRSCHBERG_DEG_PER_MM, direction: directionOf(r) };
  };
  const od = eyeMedian((s) => s.od);
  const os = eyeMedian((s) => s.os);

  const asyms = samples.map((s) => s.asymDeg).filter((a): a is number => a !== null);

  let hirschberg: LiveAggregate["hirschberg"] = null;
  if (asyms.length >= MIN_REFLEX_FRAMES && od && os) {
    const asymDeg = median(asyms);
    const stdDeg  = stdDev(asyms);
    const { severity, tier } = severityOf(asymDeg);

    // Deviating eye = the one whose reflex sits further from its pupil
    // centre (module 5's "dominant eye"); its reflex direction names the pattern.
    const deviatingEye = asymDeg >= SEVERITY_MILD_DEG ? (od.mm >= os.mm ? "OD" : "OS") : null;
    const reflexDirection = deviatingEye === "OD" ? od.direction : deviatingEye === "OS" ? os.direction : null;

    // Module 6 withholds the specific pattern when direction is unreliable
    // (tier is never changed). The live equivalent: the per-frame pattern
    // must agree in most frames, and the two eyes must not be near-equally
    // displaced — module 6's asymmetry_score < 0.05 rule.
    let agreeing = 0;
    let both = 0;
    for (const s of samples) {
      if (!s.od || !s.os) continue;
      both++;
      const odN = Math.hypot(s.od.nasal, s.od.superior);
      const osN = Math.hypot(s.os.nasal, s.os.superior);
      const eye = odN >= osN ? "OD" : "OS";
      const dir = directionOf(eye === "OD" ? s.od : s.os);
      if (eye === deviatingEye && dir === reflexDirection) agreeing++;
    }
    const directionAgreement = both ? agreeing / both : 0;
    const nearEqual = Math.abs(od.mm - os.mm) / IRIS_RADIUS_MM < 0.05;
    const directionReliable = deviatingEye !== null && directionAgreement >= 0.7 && !nearEqual;

    hirschberg = {
      asymDeg,
      stdDeg,
      pd: toPrismDioptres(asymDeg),
      confidence: stdDeg < STD_HIGH_THRESHOLD ? "HIGH" : stdDeg < STD_MEDIUM_THRESHOLD ? "MEDIUM" : "LOW",
      unstable: asymDeg >= SEVERITY_MILD_DEG && stdDeg > VARIANCE_INCONCLUSIVE_STD_THRESHOLD,
      severity,
      tier,
      deviatingEye,
      reflexDirection,
      directionReliable,
      directionAgreement,
    };
  }

  const hOD = median(samples.map((s) => s.methodB.hOD));
  const vOD = median(samples.map((s) => s.methodB.vOD));
  const hOS = median(samples.map((s) => s.methodB.hOS));
  const vOS = median(samples.map((s) => s.methodB.vOS));
  const hAsym = Math.abs(hOD - hOS);
  const vAsym = Math.abs(vOD - vOS);
  const verdict: AlignVerdict =
    hAsym >= ALIGN_H_BORDERLINE_MAX || vAsym >= ALIGN_V_BORDERLINE_MAX ? "ASYMMETRIC" :
    hAsym >= ALIGN_H_ALIGNED_MAX    || vAsym >= ALIGN_V_ALIGNED_MAX    ? "BORDERLINE" :
    "ALIGNED";

  return {
    frames:       samples.length,
    reflexFrames: asyms.length,
    odReflexRate: samples.filter((s) => s.od).length / samples.length,
    osReflexRate: samples.filter((s) => s.os).length / samples.length,
    ipdMm:    median(samples.map((s) => s.ipdMm)),
    ipdPx:    median(samples.map((s) => s.ipdPx)),
    rollDeg:  median(samples.map((s) => s.rollDeg)),
    degPerPx: median(samples.map((s) => s.degPerPx)),
    od,
    os,
    hirschberg,
    methodB: { hOD, vOD, hOS, vOS, hAsym, vAsym, verdict },
  };
}
