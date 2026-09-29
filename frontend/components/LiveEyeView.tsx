"use client";

/**
 * LiveEyeView
 * ===========
 * Real-time eye measurement, drawn on the camera feed.
 *
 *  1. Opens the rear camera (torch on where supported) and runs MediaPipe Face
 *     Mesh on every frame.
 *  2. Once a face is found, the view zooms onto the eye band and follows it.
 *  3. Every frame it runs the server's own image steps on the server's own
 *     eye crop (lib/eyeVision.ts: modules 1–3, verified to match the Python
 *     to within float rounding) and draws the pupils, reflexes and corners —
 *     pupil-to-pupil and pupil-to-reflex lines included.
 *  4. Readings are smoothed over a ~1 s rolling window and shown beside the
 *     view with their clinical meaning.
 *
 * Preview only: the reported result still comes from the server's 8-frame
 * analysis. "Compare with server" sends 8 frames, lossless, together with the
 * browser's own measurement of them; the server measures, compares and logs
 * the difference, which is the evidence needed before the browser result can
 * ever become the reported one.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FaceMesh, NormalizedLandmark, Results } from "@mediapipe/face_mesh";
import {
  aggregate,
  measureFrame,
  CONDITION_BY_REFLEX,
  MIN_REFLEX_FRAMES,
  SERVER_MIN_FRAMES,
  SEVERITY_MILD_DEG,
  SEVERITY_MODERATE_DEG,
  SEVERITY_SEVERE_DEG,
  type Direction,
  type EyeInput,
  type FrameMeasurement,
  type LiveAggregate,
  type Pt,
  type Tier,
} from "@/lib/liveMetrics";
import {
  detectReflex,
  eyeCropBox,
  localisePupil,
  rgbaToGray,
  EYE_BOUNDARY_A,
  EYE_BOUNDARY_B,
  type Box,
} from "@/lib/eyeVision";
import { analyseStream } from "@/lib/api";
import { URGENCY_CONFIG } from "@/lib/types";
import type { ClientMeasurement, StreamAnalyseResponse } from "@/lib/types";

// ── Landmarks ──────────────────────────────────────────────────────────────

interface EyeSpec {
  iris:    number[];            // 5 iris landmarks, centre first
  outline: number[];            // eye outline → module 1 crop box
  corners: [number, number];    // (inner, outer) canthus → Method B
}
/** The server's pairing (constants.py LEFT_* / RIGHT_*). */
const EYES: [EyeSpec, EyeSpec] = [
  { iris: [468, 469, 470, 471, 472], outline: EYE_BOUNDARY_A, corners: [133, 33]  },
  { iris: [473, 474, 475, 476, 477], outline: EYE_BOUNDARY_B, corners: [362, 263] },
];
/** Module 1 rejects eye crops smaller than this (MIN_CROP_WIDTH/HEIGHT). */
const MIN_CROP_W = 60;
const MIN_CROP_H = 40;
/** Corners and lid mid-points that bound the eye band for the zoomed view. */
const EYE_BAND = [33, 133, 159, 145, 362, 263, 386, 374];

// ── View / timing ──────────────────────────────────────────────────────────

const VIEW_ASPECT     = 2;      // zoomed view is a 2:1 eye band
const CROP_SMOOTHING  = 0.25;   // EMA factor for the zoom box (lower = steadier)
const WINDOW_MS       = 1000;   // rolling window for displayed values…
const MIN_KEEP        = 8;      // …but always keep this many frames, so slow devices still get a reading
const MAX_AGE_MS      = 5000;   // hard limit on staleness; long enough for MIN_KEEP frames at 2 fps
const MAX_SAMPLES     = 40;
const PANEL_EVERY_MS  = 150;    // side-panel refresh rate
const FACE_LOST_MS    = 800;    // drop back to the full frame after this

const COMPARE_FRAMES      = 8;    // same as the capture screen
const COMPARE_INTERVAL_MS = 500;  // 2 fps, same as the capture screen
const FACE_CROP_MARGIN    = 0.45; // capture screen's face-crop margin
const PIPELINE_VERSION    = "browser-live-2";

// ── Overlay palette (drawn on video, so it stays light-on-dark in both themes)

const C = {
  iris:    "rgba(110, 231, 183, 0.9)",
  pupil:   "#7DD3FC",
  reflex:  "#FBBF24",
  vector:  "#5EEAD4",
  corner:  "#F0ABFC",
  ipd:     "rgba(255, 255, 255, 0.9)",
  pillBg:  "rgba(10, 16, 25, 0.78)",
  text:    "#FFFFFF",
  muted:   "rgba(255, 255, 255, 0.7)",
};

const TIER_HEX: Record<Tier, string> = {
  NORMAL:  "#34D399",
  MONITOR: "#FBBF24",
  ROUTINE: "#FB923C",
  URGENT:  "#F87171",
};

type Phase  = "idle" | "starting" | "searching" | "tracking" | "error";
type Facing = "environment" | "user";

interface EyeFrame {
  input:       EyeInput;                 // final pupil, reflex, corners (video px)
  status:      "ok" | "no_flash" | "no_reflex" | "too_small";
  pupilSource: "dark" | "iris";
  irisCentre:  Pt;                       // landmark mean, for drawing the iris
  rollRef:     Pt;                       // iris-centre landmark, defines head roll
  box:         Box;                      // module 1 crop
}

type CompareState =
  | { phase: "idle" }
  | { phase: "capturing"; got: number }
  | { phase: "sending" }
  | { phase: "done"; run: CompareRun }
  | { phase: "error"; message: string };

interface CompareRun {
  client: ClientMeasurement;
  server: StreamAnalyseResponse;
}

interface Crop { cx: number; cy: number; w: number }

// ── Geometry helpers ───────────────────────────────────────────────────────

const toPx = (l: NormalizedLandmark, W: number, H: number): Pt => ({ x: l.x * W, y: l.y * H });
const mid  = (a: Pt, b: Pt): Pt => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/**
 * Pixels are copied from the (GPU-backed) frame into a small CPU canvas
 * before reading. Reading straight from a full-size CPU frame canvas measured
 * ~4× slower per frame.
 */
const patchCanvas: HTMLCanvasElement | null =
  typeof document !== "undefined" ? document.createElement("canvas") : null;

/** One eye, exactly as the server sees it: its crop (module 1), pupil (module 2) and reflex (module 3). */
function analyseEye(frame: HTMLCanvasElement, W: number, H: number, lms: NormalizedLandmark[], spec: EyeSpec): EyeFrame {
  const irisPts    = spec.iris.map((i) => toPx(lms[i], W, H));
  const irisCentre = { x: irisPts.reduce((s, p) => s + p.x, 0) / 5, y: irisPts.reduce((s, p) => s + p.y, 0) / 5 };
  const irisR      = irisPts.slice(1).reduce((s, p) => s + dist(p, irisPts[0]), 0) / 4;
  const corners    = { inner: toPx(lms[spec.corners[0]], W, H), outer: toPx(lms[spec.corners[1]], W, H) };
  const box        = eyeCropBox(spec.outline.map((i) => toPx(lms[i], W, H)), W, H);
  const w = box.x2 - box.x1;
  const h = box.y2 - box.y1;
  const common = { irisCentre, rollRef: irisPts[0], box };

  const pctx = patchCanvas?.getContext("2d", { willReadFrequently: true });
  if (w < MIN_CROP_W || h < MIN_CROP_H || !patchCanvas || !pctx) {
    return { ...common, input: { pupil: irisCentre, irisR, reflex: null, ...corners }, status: "too_small", pupilSource: "iris" };
  }
  if (patchCanvas.width < w)  patchCanvas.width  = w;
  if (patchCanvas.height < h) patchCanvas.height = h;
  pctx.drawImage(frame, box.x1, box.y1, w, h, 0, 0, w, h);
  const gray = rgbaToGray(pctx.getImageData(0, 0, w, h).data);

  // Crop coordinates throughout, exactly like the server; shift back to the frame at the end.
  const pupil  = localisePupil(gray, w, h, { x: irisCentre.x - box.x1, y: irisCentre.y - box.y1 }, irisR);
  const reflex = detectReflex(gray, w, h, pupil.centre, irisR);
  return {
    ...common,
    input: {
      pupil:  { x: pupil.centre.x + box.x1, y: pupil.centre.y + box.y1 },
      irisR,
      reflex: reflex.status === "ok" ? { x: reflex.pos.x + box.x1, y: reflex.pos.y + box.y1 } : null,
      ...corners,
    },
    status: reflex.status,
    pupilSource: pupil.source,
  };
}

/**
 * The face crop the capture screen sends (StreamingCapture captureFrame), but
 * PNG and at native resolution, so the server measures the same pixels the
 * browser did — any difference is then the code, not compression or resizing.
 */
function encodeFaceCrop(frame: HTMLCanvasElement, lms: NormalizedLandmark[]): Promise<Blob | null> {
  const W = frame.width, H = frame.height;
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of lms) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  const fw = (maxX - minX) * W, fh = (maxY - minY) * H;
  const padX = fw * FACE_CROP_MARGIN, padY = fh * FACE_CROP_MARGIN;
  const sx = Math.max(0, Math.round(minX * W - padX));
  const sy = Math.max(0, Math.round(minY * H - padY));
  const sw = Math.min(W - sx, Math.round(fw + padX * 2));
  const sh = Math.min(H - sy, Math.round(fh + padY * 2));
  const out = document.createElement("canvas");
  out.width = sw; out.height = sh;
  out.getContext("2d")?.drawImage(frame, sx, sy, sw, sh, 0, 0, sw, sh);   // copied now; the frame canvas is reused next frame
  return new Promise((resolve) => out.toBlob(resolve, "image/png"));
}

// ── Component ──────────────────────────────────────────────────────────────

export default function LiveEyeView({ patientName, patientAge }: {
  /** Used only to label comparison requests to the server. */
  patientName?: string;
  patientAge?:  number | null;
} = {}) {
  const videoRef   = useRef<HTMLVideoElement>(null);
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const frameRef   = useRef<HTMLCanvasElement | null>(null);
  const fmRef      = useRef<FaceMesh | null>(null);
  const streamRef  = useRef<MediaStream | null>(null);
  const busyRef    = useRef(false);
  const cropRef    = useRef<Crop | null>(null);
  const samplesRef = useRef<{ t: number; m: FrameMeasurement; src: [EyeFrame["pupilSource"], EyeFrame["pupilSource"]] }[]>([]);
  const compareRef = useRef<{ active: boolean; lastT: number; frames: { m: FrameMeasurement; png: Promise<Blob | null> }[] }>(
    { active: false, lastT: 0, frames: [] },
  );
  const finishCompareRef = useRef<() => Promise<void>>(async () => undefined);
  const aggRef     = useRef<LiveAggregate | null>(null);
  const lastFaceRef  = useRef(0);
  const lastPanelRef = useRef(0);
  const phaseRef     = useRef<Phase>("idle");
  const fpsRef       = useRef({ last: 0, value: 0 });

  const [phase,   setPhaseState] = useState<Phase>("idle");
  const [error,   setError]      = useState<string | null>(null);
  const [agg,     setAgg]        = useState<LiveAggregate | null>(null);
  const [rate,    setRate]       = useState<{ frames: number; fps: number } | null>(null);
  const [torch,   setTorch]      = useState<boolean | null>(null);   // null = unsupported
  const [facing,  setFacing]     = useState<Facing>("environment");
  const [fmReady, setFmReady]    = useState(false);
  const [compare, setCompare]    = useState<CompareState>({ phase: "idle" });
  const [history, setHistory]    = useState<CompareRun[]>([]);
  const [darkRate, setDarkRate]  = useState<[number, number] | null>(null);

  const setPhase = useCallback((p: Phase) => {
    if (phaseRef.current !== p) {
      phaseRef.current = p;
      setPhaseState(p);
    }
  }, []);

  const running = phase === "searching" || phase === "tracking";

  // ── Drawing ─────────────────────────────────────────────────────────────

  const drawFrame = useCallback((frame: HTMLCanvasElement, eyes: [EyeFrame, EyeFrame] | null) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    const dpr  = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width  = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingQuality = "high";

    const W = frame.width;
    const H = frame.height;
    const crop = cropRef.current;

    // Source rectangle: the tracked eye band, or the whole frame (cover-fit).
    let sx: number, sy: number, sw: number, sh: number;
    if (crop && eyes) {
      sw = crop.w;
      sh = crop.w / VIEW_ASPECT;
      sx = clamp(crop.cx - sw / 2, 0, W - sw);
      sy = clamp(crop.cy - sh / 2, 0, H - sh);
    } else {
      const frameAspect = W / H;
      const viewAspect  = cssW / cssH;
      if (frameAspect > viewAspect) { sh = H; sw = H * viewAspect; sx = (W - sw) / 2; sy = 0; }
      else                          { sw = W; sh = W / viewAspect; sx = 0; sy = (H - sh) / 2; }
    }
    ctx.drawImage(frame, sx, sy, sw, sh, 0, 0, cssW, cssH);

    const k = cssW / sw;
    const M = (p: Pt): Pt => ({ x: (p.x - sx) * k, y: (p.y - sy) * k });

    // Phones get short labels; the full wording is in the side panel.
    const compact = cssW < 560;
    const pill = (text: string, at: Pt, color = C.text, align: "center" | "left" = "center") => {
      ctx.font = `600 ${compact ? 11 : 12}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      const tw = ctx.measureText(text).width;
      const pw = tw + (compact ? 10 : 14);
      const ph = compact ? 19 : 22;
      const x  = clamp(align === "center" ? at.x - pw / 2 : at.x, 4, cssW - pw - 4);
      const y  = clamp(at.y - ph / 2, 4, cssH - ph - 4);
      ctx.fillStyle = C.pillBg;
      ctx.beginPath();
      ctx.roundRect(x, y, pw, ph, 6);
      ctx.fill();
      ctx.fillStyle = color;
      ctx.textBaseline = "middle";
      ctx.fillText(text, x + (compact ? 5 : 7), y + ph / 2 + 0.5);
    };

    // Live badge + frame rate
    const now = performance.now();
    const f = fpsRef.current;
    if (f.last) f.value = f.value * 0.9 + (1000 / Math.max(1, now - f.last)) * 0.1;
    f.last = now;
    pill(`● LIVE  ${Math.round(f.value)} fps`, { x: 10, y: 20 }, "#FCA5A5", "left");
    const cmp = compareRef.current;
    if (cmp.active) pill(`Comparing ${cmp.frames.length}/${COMPARE_FRAMES} — hold still`, { x: cssW - 10, y: 20 }, "#FCD34D");

    if (!eyes) {
      pill(compact ? "Looking for a face…" : "Looking for a face — hold the camera 30–40 cm away", { x: cssW / 2, y: cssH / 2 });
      return;
    }

    const a = aggRef.current;
    const [od, os] = eyes;

    // Eye corners and the canthus axis (Method B)
    for (const e of eyes) {
      const inn = M(e.input.inner);
      const out = M(e.input.outer);
      ctx.setLineDash([3, 4]);
      ctx.strokeStyle = "rgba(240, 171, 252, 0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(inn.x, inn.y); ctx.lineTo(out.x, out.y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = C.corner;
      for (const p of [inn, out]) { ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, Math.PI * 2); ctx.fill(); }
    }

    // Pupil-to-pupil line with distance
    const pOD = M(od.input.pupil);
    const pOS = M(os.input.pupil);
    ctx.setLineDash([7, 6]);
    ctx.strokeStyle = C.ipd;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pOD.x, pOD.y); ctx.lineTo(pOS.x, pOS.y); ctx.stroke();
    ctx.setLineDash([]);

    const centre = mid(pOD, pOS);
    if (a) {
      pill(
        compact ? `IPD ${a.ipdMm.toFixed(1)} mm` : `IPD ${a.ipdMm.toFixed(1)} mm · tilt ${a.rollDeg.toFixed(1)}°`,
        // On phones both pills sit below the pupil line (over the nose
        // bridge), clear of the per-eye tags above each iris.
        { x: centre.x, y: centre.y + (compact ? 13 : -16) },
      );
      const h = a.hirschberg;
      const below = { x: centre.x, y: centre.y + (compact ? 35 : 16) };
      if (h) {
        pill(
          h.unstable
            ? (compact ? "Hold still" : "Unstable — hold still")
            : `${compact ? "" : "Asymmetry "}${h.asymDeg.toFixed(1)}° · ${h.pd.toFixed(0)} PD`,
          below,
          h.unstable ? "#FCD34D" : TIER_HEX[h.tier],
        );
      } else {
        const anyNoFlash = od.status === "no_flash" || os.status === "no_flash";
        pill(
          anyNoFlash ? (compact ? "No torch reflex" : "No torch reflex — turn the torch on") : "Finding reflexes…",
          below,
          C.muted,
        );
      }
    }

    // Per eye: iris, pupil, reflex, and the pupil→reflex vector
    const eyeLabel = (tag: "OD" | "OS") => (tag === "OD" ? "R eye (OD)" : "L eye (OS)");
    ([["OD", od], ["OS", os]] as const).forEach(([tag, e]) => {
      const p = M(e.input.pupil);
      const ic = M(e.irisCentre);
      const r = e.input.irisR * k;

      ctx.strokeStyle = C.iris;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(ic.x, ic.y, r, 0, Math.PI * 2); ctx.stroke();

      ctx.strokeStyle = C.pupil;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(p.x - 6, p.y); ctx.lineTo(p.x + 6, p.y);
      ctx.moveTo(p.x, p.y - 6); ctx.lineTo(p.x, p.y + 6);
      ctx.stroke();

      if (e.input.reflex) {
        const q = M(e.input.reflex);
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = C.vector;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = C.reflex;
        ctx.beginPath(); ctx.arc(q.x, q.y, 3.5, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "rgba(251, 191, 36, 0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(q.x, q.y, 7, 0, Math.PI * 2); ctx.stroke();
      }

      const reading = tag === "OD" ? a?.od : a?.os;
      const name = compact ? (tag === "OD" ? "R" : "L") : eyeLabel(tag);
      const text = reading
        ? compact
          ? `${name} ${reading.mm.toFixed(2)} mm · ${reading.deg.toFixed(1)}°`
          : `${name} · ${reading.mm.toFixed(2)} mm · ${reading.deg.toFixed(1)}° ${reading.direction}`
        : `${name} · ${e.status === "no_flash" ? "no torch" : "no reflex"}`;
      pill(text, { x: p.x, y: p.y - r - (compact ? 14 : 18) }, reading ? C.text : C.muted);
    });
  }, []);

  // ── Per-frame processing ────────────────────────────────────────────────

  const processResults = useCallback((results: Results) => {
    const frame = frameRef.current;
    if (!frame) return;
    const W = frame.width;
    const H = frame.height;
    const now = performance.now();
    const lms = results.multiFaceLandmarks?.[0];

    if (!lms || lms.length < 478) {
      if (now - lastFaceRef.current > FACE_LOST_MS) {
        cropRef.current = null;
        samplesRef.current = [];
        aggRef.current = null;
        setAgg(null);
        setPhase("searching");
      }
      drawFrame(frame, null);
      return;
    }
    lastFaceRef.current = now;
    setPhase("tracking");

    // Both eyes through the server's own steps; the image-left one is the
    // patient's right eye (OD).
    const eyes = EYES.map((spec) => analyseEye(frame, W, H, lms, spec));
    const [od, os] = eyes[0].irisCentre.x <= eyes[1].irisCentre.x ? eyes : [eyes[1], eyes[0]];

    // Module 1 rejects a frame whose eye crops are too small — so do we.
    const samples = samplesRef.current;
    if (od.status !== "too_small" && os.status !== "too_small") {
      const m = measureFrame(od.input, os.input, [od.rollRef, os.rollRef]);
      samples.push({ t: now, m, src: [od.pupilSource, os.pupilSource] });
      while (samples.length > MAX_SAMPLES) samples.shift();
      while (samples.length > MIN_KEEP && now - samples[0].t > WINDOW_MS) samples.shift();
      while (samples.length && now - samples[0].t > MAX_AGE_MS) samples.shift();

      // Comparison capture: 8 frames at 2 fps, like the capture screen.
      const cmp = compareRef.current;
      if (cmp.active && now - cmp.lastT >= COMPARE_INTERVAL_MS) {
        cmp.lastT = now;
        cmp.frames.push({ m, png: encodeFaceCrop(frame, lms) });
        setCompare({ phase: "capturing", got: cmp.frames.length });
        if (cmp.frames.length >= COMPARE_FRAMES) {
          cmp.active = false;
          void finishCompareRef.current();
        }
      }
    }

    // Zoom box follows the eye band, smoothed so the view doesn't shake.
    const band = EYE_BAND.map((i) => toPx(lms[i], W, H));
    const xs = band.map((p) => p.x);
    const ys = band.map((p) => p.y);
    const bw = Math.max(...xs) - Math.min(...xs);
    const bh = Math.max(...ys) - Math.min(...ys);
    let tw = bw * 1.5;
    if (tw / VIEW_ASPECT < bh * 3) tw = bh * 3 * VIEW_ASPECT;
    tw = Math.min(tw, W, H * VIEW_ASPECT);
    const target: Crop = {
      cx: (Math.min(...xs) + Math.max(...xs)) / 2,
      cy: (Math.min(...ys) + Math.max(...ys)) / 2,
      w:  tw,
    };
    const prev = cropRef.current;
    cropRef.current = prev
      ? {
          cx: prev.cx + CROP_SMOOTHING * (target.cx - prev.cx),
          cy: prev.cy + CROP_SMOOTHING * (target.cy - prev.cy),
          w:  prev.w  + CROP_SMOOTHING * (target.w  - prev.w),
        }
      : target;

    if (now - lastPanelRef.current > PANEL_EVERY_MS) {
      lastPanelRef.current = now;
      aggRef.current = aggregate(samples.map((s) => s.m));
      setAgg(aggRef.current);
      if (samples.length) {
        setDarkRate([0, 1].map((i) => samples.filter((x) => x.src[i] === "dark").length / samples.length) as [number, number]);
      }
      const span = samples.length > 1 ? samples[samples.length - 1].t - samples[0].t : 0;
      setRate({ frames: samples.length, fps: span > 0 ? ((samples.length - 1) * 1000) / span : 0 });
    }

    drawFrame(frame, [od, os]);
  }, [drawFrame, setPhase]);

  // ── MediaPipe ───────────────────────────────────────────────────────────

  const processRef = useRef(processResults);
  processRef.current = processResults;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { FaceMesh: FaceMeshCtor } = await import("@mediapipe/face_mesh");
        if (cancelled) return;
        const fm = new FaceMeshCtor({
          locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}`,
        });
        fm.setOptions({
          maxNumFaces: 1,
          refineLandmarks: true,          // iris landmarks 468–477
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        fm.onResults((r) => processRef.current(r));
        fmRef.current = fm;
        setFmReady(true);
      } catch (e) {
        console.warn("[LiveEyeView] Face Mesh failed to load:", e);
        setError("The face-tracking model could not be loaded. Check your connection and reload.");
      }
    })();
    return () => {
      cancelled = true;
      fmRef.current?.close().catch(() => undefined);
      fmRef.current = null;
    };
  }, []);

  // Frame loop: snapshot the video into a canvas, then hand that same
  // snapshot to Face Mesh. Landmarks and pixels always come from one frame.
  useEffect(() => {
    if (!fmReady || !running) return;
    let alive = true;
    let raf = 0;
    const tick = async () => {
      if (!alive) return;
      const video = videoRef.current;
      const fm    = fmRef.current;
      if (video && fm && video.readyState >= 2 && video.videoWidth && !busyRef.current) {
        if (!frameRef.current) frameRef.current = document.createElement("canvas");
        const frame = frameRef.current;
        if (frame.width !== video.videoWidth || frame.height !== video.videoHeight) {
          frame.width  = video.videoWidth;
          frame.height = video.videoHeight;
        }
        frame.getContext("2d")?.drawImage(video, 0, 0);
        busyRef.current = true;
        try {
          await fm.send({ image: frame });
        } catch (e) {
          console.warn("[LiveEyeView] frame failed:", e);
        } finally {
          busyRef.current = false;
        }
      }
      if (alive) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { alive = false; cancelAnimationFrame(raf); };
  }, [fmReady, running]);

  // ── Camera ──────────────────────────────────────────────────────────────

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    cropRef.current = null;
    samplesRef.current = [];
    aggRef.current = null;
    setAgg(null);
    setRate(null);
    setDarkRate(null);
    setTorch(null);
    if (compareRef.current.active) {
      compareRef.current = { active: false, lastT: 0, frames: [] };
      setCompare({ phase: "idle" });
    }
  }, []);

  const startCamera = useCallback(async (which: Facing) => {
    stopCamera();
    setError(null);
    setPhase("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: which }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      const track = stream.getVideoTracks()[0];
      const caps  = (track.getCapabilities?.() ?? {}) as MediaTrackCapabilities & { torch?: boolean };
      if (caps.torch) {
        try {
          await track.applyConstraints({ advanced: [{ torch: true } as MediaTrackConstraintSet] });
          setTorch(true);
        } catch {
          setTorch(false);
        }
      } else {
        setTorch(null);
      }
      setFacing(which);
      setPhase("searching");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Camera access was denied.");
      setPhase("error");
    }
  }, [setPhase, stopCamera]);

  const toggleTorch = useCallback(async () => {
    const track = streamRef.current?.getVideoTracks()[0];
    if (!track || torch === null) return;
    try {
      await track.applyConstraints({ advanced: [{ torch: !torch } as MediaTrackConstraintSet] });
      setTorch(!torch);
    } catch {
      setTorch(null);
    }
  }, [torch]);

  // ── Browser-vs-server comparison ─────────────────────────────────────────

  const startCompare = useCallback(() => {
    compareRef.current = { active: true, lastT: 0, frames: [] };
    setCompare({ phase: "capturing", got: 0 });
  }, []);

  const finishCompare = useCallback(async () => {
    const frames = compareRef.current.frames;
    compareRef.current.frames = [];
    setCompare({ phase: "sending" });
    try {
      const blobs = await Promise.all(frames.map((f) => f.png));
      if (blobs.some((b) => !b)) throw new Error("A frame could not be encoded.");
      const files = blobs.map((b, i) => new File([b as Blob], `frame-${i + 1}.png`, { type: "image/png" }));

      // The browser's reading of exactly these frames, with the server's
      // own minimum-frame rule.
      const ms  = frames.map((f) => f.m);
      const a   = aggregate(ms, SERVER_MIN_FRAMES);
      const h   = a?.hirschberg ?? null;
      const r2  = (v: number) => Math.round(v * 100) / 100;
      const r4  = (v: number) => Math.round(v * 10000) / 10000;
      const client: ClientMeasurement = {
        schema: 1,
        pipeline: PIPELINE_VERSION,
        frames: ms.length,
        reflex_frames: a?.reflexFrames ?? 0,
        per_frame_asymmetry_deg: ms.map((m) => (m.asymDeg === null ? null : r2(m.asymDeg))),
        asymmetry_deg: h ? r2(h.asymDeg) : null,
        asymmetry_std_deg: h ? r2(h.stdDeg) : null,
        tier: h && !h.unstable ? h.tier : null,
        unstable: h?.unstable ?? false,
        method_b: a ? { h_asym: r4(a.methodB.hAsym), v_asym: r4(a.methodB.vAsym), verdict: a.methodB.verdict } : null,
      };

      const server = await analyseStream(files, patientName?.trim() || "Live comparison", patientAge ?? 30, client);
      const run = { client, server };
      setCompare({ phase: "done", run });
      setHistory((hs) => [...hs, run]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "";
      setCompare({
        phase: "error",
        message: msg === "TIMEOUT" ? "The server took too long to answer. Try again."
          : msg || "The comparison could not reach the server.",
      });
    }
  }, [patientName, patientAge]);
  finishCompareRef.current = finishCompare;

  useEffect(() => () => stopCamera(), [stopCamera]);

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <div className="grid lg:grid-cols-[minmax(0,1fr)_380px] gap-6 items-start">

      {/* Camera — pinned under the top bar so it stays in view while the
          readings scroll past (beside it on desktop, beneath it on phones). */}
      <div className="space-y-3 sticky top-[65px] lg:top-20 z-20 bg-white pb-3 lg:pb-0
                      border-b border-ink-100 lg:border-0">
        <div className="relative w-full aspect-[2/1] bg-ink-900 rounded-card overflow-hidden shadow-card">
          <video ref={videoRef} playsInline muted className="absolute w-px h-px opacity-0 pointer-events-none" />
          <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />

          {!running && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 px-6 text-center">
              {phase === "starting" ? (
                <p className="text-white/80 text-[14px]">Opening camera…</p>
              ) : (
                <>
                  <p className="text-white/85 text-[14px] max-w-sm leading-relaxed">
                    The view zooms onto the eyes once a face is found, and measures
                    every frame as it arrives.
                  </p>
                  <button onClick={() => startCamera("environment")} disabled={!fmReady} className="btn-accent">
                    {fmReady ? "Start live view" : "Loading face tracking…"}
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        {error && (
          <div role="alert" className="rounded-field bg-red-50 border border-red-200 px-3.5 py-3 text-[13.5px] text-red-700">
            {error}
          </div>
        )}

        {running && (
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={toggleTorch}
              disabled={torch === null}
              className="btn-secondary btn-sm"
              title={torch === null ? "This camera has no controllable torch" : undefined}
            >
              {torch === null ? "Torch unavailable" : torch ? "Torch on" : "Torch off"}
            </button>
            <button onClick={() => startCamera(facing === "environment" ? "user" : "environment")} className="btn-secondary btn-sm">
              Switch to {facing === "environment" ? "front" : "rear"} camera
            </button>
            <button
              onClick={startCompare}
              disabled={phase !== "tracking" || compare.phase === "capturing" || compare.phase === "sending"}
              className="btn-accent btn-sm"
              title="Capture 8 frames and have the server measure the same frames"
            >
              {compare.phase === "capturing" ? `Capturing ${compare.got}/${COMPARE_FRAMES}…`
                : compare.phase === "sending" ? "Server measuring…"
                : "Compare with server"}
            </button>
            <button onClick={() => { stopCamera(); setPhase("idle"); }} className="btn-ghost btn-sm">
              Stop
            </button>
          </div>
        )}

        <Legend />
      </div>

      {/* Values */}
      <LivePanel agg={agg} rate={rate} phase={phase} torch={torch} darkRate={darkRate} compare={compare} history={history} />
    </div>
  );
}

// ── Side panel ─────────────────────────────────────────────────────────────

function Row({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-2 border-b border-ink-100 last:border-0">
      <span className="text-[13px] text-ink-500">{label}</span>
      <span className="text-right">
        <span className={`font-mono text-[13px] font-semibold tabular ${tone ?? "text-ink-900"}`}>{value}</span>
        {sub && <span className="block text-[11px] text-ink-400">{sub}</span>}
      </span>
    </div>
  );
}

function Section({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <section className="surface p-5">
      <div className="flex items-baseline justify-between gap-3 mb-2">
        <h3 className="eyebrow-muted">{title}</h3>
        {note && <span className="text-[11px] text-ink-400">{note}</span>}
      </div>
      {children}
    </section>
  );
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function LivePanel({ agg, rate, phase, torch, darkRate, compare, history }: {
  agg: LiveAggregate | null;
  rate: { frames: number; fps: number } | null;
  phase: Phase;
  torch: boolean | null;
  darkRate: [number, number] | null;
  compare: CompareState;
  history: CompareRun[];
}) {
  const h = agg?.hirschberg ?? null;
  const pattern = h?.reflexDirection && h.directionReliable ? CONDITION_BY_REFLEX[h.reflexDirection] : null;

  const tiers = useMemo(() => ([
    { tier: "NORMAL"  as Tier, range: `< ${SEVERITY_MILD_DEG}°`,                              action: "No referral" },
    { tier: "MONITOR" as Tier, range: `${SEVERITY_MILD_DEG}–${SEVERITY_MODERATE_DEG}°`,       action: "Re-screen in 3 months" },
    { tier: "ROUTINE" as Tier, range: `${SEVERITY_MODERATE_DEG}–${SEVERITY_SEVERE_DEG}°`,     action: "Refer within 4 weeks" },
    { tier: "URGENT"  as Tier, range: `≥ ${SEVERITY_SEVERE_DEG}°`,                            action: "Refer within 1 week" },
  ]), []);

  const status =
    phase === "tracking"  ? { text: "Tracking",          tone: "text-emerald-700" } :
    phase === "searching" ? { text: "Looking for a face", tone: "text-amber-700"   } :
    phase === "starting"  ? { text: "Starting",          tone: "text-ink-500"     } :
                            { text: "Camera off",        tone: "text-ink-400"     };

  return (
    <aside className="space-y-4">

      {(compare.phase !== "idle" || history.length > 0) && <CompareCard compare={compare} history={history} />}

      <Section title="Signal" note="rolling window">
        <Row label="Face" value={status.text} tone={status.tone} />
        <Row
          label="Frames in window"
          value={rate ? `${rate.frames} · ${rate.fps.toFixed(1)} fps` : "—"}
          sub="every value below is the median of these"
        />
        <Row label="Torch" value={torch === null ? "Not controllable" : torch ? "On" : "Off"} />
        <Row label="Reflex found · R / L" value={agg ? `${pct(agg.odReflexRate)} / ${pct(agg.osReflexRate)}` : "—"} sub="of frames in the window" />
        <Row
          label="Pupil found as dark disc · R / L"
          value={darkRate ? `${pct(darkRate[0])} / ${pct(darkRate[1])}` : "—"}
          sub="otherwise the iris centre is used, as on the server"
        />
        <Row
          label="Stability"
          value={h ? `σ ${h.stdDeg.toFixed(2)}° · ${h.confidence}` : "—"}
          tone={h ? (h.confidence === "HIGH" ? "text-emerald-700" : h.confidence === "MEDIUM" ? "text-amber-700" : "text-red-600") : undefined}
          sub="spread of asymmetry across frames"
        />
      </Section>

      <Section title="Eye geometry" note="no torch needed">
        <Row label="Inter-pupil distance" value={agg ? `${agg.ipdMm.toFixed(1)} mm` : "—"} sub={agg ? `${agg.ipdPx.toFixed(0)} px · iris used as the ruler` : undefined} />
        <Row label="Head tilt" value={agg ? `${agg.rollDeg.toFixed(1)}°` : "—"} sub="measured along the pupil line, so tilt is corrected" />
        <Row
          label="Resolution"
          value={agg ? `1 px ≈ ${agg.degPerPx.toFixed(2)}°` : "—"}
          tone={agg && agg.degPerPx > 2.5 ? "text-amber-700" : undefined}
          sub={agg && agg.degPerPx > 2.5 ? "move closer for a finer reading" : "of reflex movement"}
        />
      </Section>

      <Section title="Hirschberg · torch reflex">
        {agg?.od && <Row label="R eye (OD) reflex" value={`${agg.od.mm.toFixed(2)} mm · ${agg.od.deg.toFixed(1)}°`} sub={`${agg.od.direction} of the pupil centre`} />}
        {agg?.os && <Row label="L eye (OS) reflex" value={`${agg.os.mm.toFixed(2)} mm · ${agg.os.deg.toFixed(1)}°`} sub={`${agg.os.direction} of the pupil centre`} />}

        {h ? (
          <div className="mt-3 pt-3 border-t border-ink-100 space-y-3">
            <div className="flex items-end justify-between gap-3">
              <div>
                <p className="text-[11px] text-ink-400 uppercase tracking-eyebrow font-semibold">Asymmetry</p>
                <p className="font-display text-[30px] leading-none tabular text-ink-900 mt-1">{h.asymDeg.toFixed(1)}°</p>
                <p className="text-[12px] text-ink-500 mt-1">≈ {h.pd.toFixed(0)} prism dioptres</p>
              </div>
              {h.unstable ? (
                <span className="font-mono text-[11px] font-semibold px-2.5 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-700">
                  UNSTABLE
                </span>
              ) : (
                <span className={`font-mono text-[11px] font-semibold px-2.5 py-1 rounded-full text-white ${URGENCY_CONFIG[h.tier].badgeColour}`}>
                  {h.tier} · {h.severity}
                </span>
              )}
            </div>

            {h.unstable ? (
              <p className="text-[12.5px] text-amber-800 leading-relaxed">
                The reading is moving too much between frames to trust — hold the camera
                and the patient&apos;s gaze steady.
              </p>
            ) : pattern && h.deviatingEye ? (
              <p className="text-[12.5px] text-ink-600 leading-relaxed">
                <span className="font-semibold text-ink-900">{pattern.name}</span>{" "}
                <span className="font-mono text-[11px] text-ink-400">{pattern.icd}</span> pattern —
                the {h.deviatingEye === "OD" ? "right" : "left"} eye&apos;s reflex sits {h.reflexDirection},
                consistent with that eye turned {pattern.eyeTurns}.
              </p>
            ) : h.deviatingEye ? (
              <p className="text-[12.5px] text-ink-600 leading-relaxed">
                <span className="font-semibold text-ink-900">Strabismus, unspecified</span>{" "}
                <span className="font-mono text-[11px] text-ink-400">H50.9</span> — the direction
                isn&apos;t consistent enough to name a pattern (agrees in{" "}
                {Math.round(h.directionAgreement * 100)}% of frames). The tier still stands.
              </p>
            ) : (
              <p className="text-[12.5px] text-ink-600">No significant deviation.</p>
            )}
          </div>
        ) : (
          <p className="text-[12.5px] text-ink-500 leading-relaxed mt-1">
            {phase !== "tracking"
              ? "Start the live view to measure."
              : agg && agg.odReflexRate === 0 && agg.osReflexRate === 0
                ? "No torch reflex in either eye. Turn the torch on and dim the room."
                : `Collecting — needs a reflex in both eyes for ${MIN_REFLEX_FRAMES}+ frames.`}
          </p>
        )}
      </Section>

      <Section title="Method B · eye corners" note="no torch needed">
        {agg ? (
          <>
            <Row label="Pupil along corner axis · R / L" value={`${pct(agg.methodB.hOD)} / ${pct(agg.methodB.hOS)}`} sub="0% inner corner → 100% outer corner" />
            <Row label="Horizontal asymmetry" value={pct(agg.methodB.hAsym)} sub="flag at 5% · refer-level at 10%" />
            <Row label="Vertical asymmetry"   value={pct(agg.methodB.vAsym)} sub="flag at 6% · refer-level at 12%" />
            <div className="pt-2">
              <span className={`font-mono text-[11px] font-semibold px-2.5 py-1 rounded-full border ${
                agg.methodB.verdict === "ALIGNED"    ? "bg-emerald-50 border-emerald-200 text-emerald-700" :
                agg.methodB.verdict === "BORDERLINE" ? "bg-amber-50 border-amber-200 text-amber-700" :
                                                       "bg-red-50 border-red-200 text-red-700"
              }`}>
                {agg.methodB.verdict}
              </span>
            </div>
          </>
        ) : (
          <p className="text-[12.5px] text-ink-500">Waiting for a face.</p>
        )}
      </Section>

      <Section title="What the numbers mean">
        <p className="text-[11px] text-ink-400 uppercase tracking-eyebrow font-semibold mb-1.5">Asymmetry → tier</p>
        <ul className="mb-4">
          {tiers.map((t) => {
            const active = h && !h.unstable && h.tier === t.tier;
            return (
              <li key={t.tier} className={`flex items-center gap-2.5 py-1.5 px-2 -mx-2 rounded-md ${active ? "bg-ink-50" : ""}`}>
                <span className={`w-2 h-2 rounded-full shrink-0 ${URGENCY_CONFIG[t.tier].dotColour}`} />
                <span className={`font-mono text-[11px] w-16 shrink-0 ${active ? "font-bold text-ink-900" : "text-ink-600"}`}>{t.tier}</span>
                <span className="font-mono text-[11px] text-ink-500 w-14 shrink-0">{t.range}</span>
                <span className={`text-[12px] ${active ? "text-ink-900 font-medium" : "text-ink-500"}`}>{t.action}</span>
              </li>
            );
          })}
        </ul>

        <p className="text-[11px] text-ink-400 uppercase tracking-eyebrow font-semibold mb-1.5">Reflex position → pattern</p>
        <ul>
          {(Object.keys(CONDITION_BY_REFLEX) as Direction[]).map((dir) => {
            const c = CONDITION_BY_REFLEX[dir];
            const active = h && !h.unstable && h.directionReliable && h.reflexDirection === dir;
            return (
              <li key={dir} className={`flex items-baseline gap-2 py-1.5 px-2 -mx-2 rounded-md ${active ? "bg-ink-50" : ""}`}>
                <span className={`text-[12px] w-[112px] shrink-0 ${active ? "text-ink-900 font-semibold" : "text-ink-500"}`}>Reflex {dir}</span>
                <span className={`text-[12px] ${active ? "text-ink-900 font-semibold" : "text-ink-600"}`}>{c.name}</span>
                <span className="font-mono text-[10.5px] text-ink-400 ml-auto">{c.icd}</span>
              </li>
            );
          })}
        </ul>

        <p className="text-[11.5px] text-ink-400 leading-relaxed mt-4 pt-3 border-t border-ink-100">
          Live preview, not the reported result. Reflexes alone can&apos;t prove which eye
          is deviating — a cover test confirms it.
        </p>
      </Section>
    </aside>
  );
}

const fmtDeg = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${v.toFixed(1)}°`);
const fmtDelta = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(1)}°`);

function CompareCard({ compare, history }: { compare: CompareState; history: CompareRun[] }) {
  const run = compare.phase === "done" ? compare.run : history[history.length - 1] ?? null;
  const p = run?.server.parity;
  const reason = run && run.server.status === "INCONCLUSIVE" ? run.server.reason_human : null;

  // Across this session's runs
  const tiered = history.map((r) => r.server.parity?.tier).filter((t) => t && t.agree !== null);
  const agreeing = tiered.filter((t) => t?.agree).length;
  const deltas = history
    .map((r) => r.server.parity?.asymmetry_deg?.delta)
    .filter((d): d is number => typeof d === "number")
    .map(Math.abs)
    .sort((a, b) => a - b);
  const medianDelta = deltas.length ? deltas[deltas.length >> 1] : null;

  return (
    <Section title="Browser vs server" note="same 8 frames · lossless">
      {compare.phase === "capturing" && (
        <p className="text-[12.5px] text-ink-600 py-1">Capturing frame {compare.got} of {COMPARE_FRAMES} — keep the patient still.</p>
      )}
      {compare.phase === "sending" && (
        <p className="text-[12.5px] text-ink-600 py-1">The server is measuring the same frames…</p>
      )}
      {compare.phase === "error" && (
        <p role="alert" className="text-[12.5px] text-red-700 py-1">{compare.message}</p>
      )}

      {run && (
        p?.error ? (
          <p className="text-[12.5px] text-red-700 py-1">Server could not compare: {p.error}</p>
        ) : p ? (
          <>
            <Row label="Server result" value={p.server_status ?? run.server.status} sub={reason ?? undefined}
                 tone={run.server.status === "SUCCESS" ? "text-ink-900" : "text-amber-700"} />
            <Row
              label="Asymmetry · browser / server"
              value={`${fmtDeg(p.asymmetry_deg?.client)} / ${fmtDeg(p.asymmetry_deg?.server)}`}
              sub={`difference ${fmtDelta(p.asymmetry_deg?.delta)}`}
            />
            <Row
              label="Tier · browser / server"
              value={`${p.tier?.client ?? "—"} / ${p.tier?.server ?? "—"}`}
              tone={p.tier?.agree === true ? "text-emerald-700" : p.tier?.agree === false ? "text-red-600" : undefined}
              sub={p.tier?.agree === true ? "agree" : p.tier?.agree === false ? "disagree" : "not comparable"}
            />
            <Row
              label="Frame by frame"
              value={p.per_frame?.median_abs_delta === null || p.per_frame?.median_abs_delta === undefined
                ? "—" : `±${p.per_frame.median_abs_delta.toFixed(1)}°`}
              sub={`median difference over ${p.per_frame?.frames_compared ?? 0} frames both measured`}
            />
            <Row
              label="Method B · browser / server"
              value={`${p.method_b?.verdict.client ?? "—"} / ${p.method_b?.verdict.server ?? "—"}`}
              tone={p.method_b?.verdict.agree === true ? "text-emerald-700" : p.method_b?.verdict.agree === false ? "text-red-600" : undefined}
            />
          </>
        ) : (
          <p className="text-[12.5px] text-ink-500 py-1">The server didn&apos;t return a comparison — it may be running an older version.</p>
        )
      )}

      {history.length > 1 && (
        <p className="text-[12px] text-ink-500 mt-3 pt-3 border-t border-ink-100">
          This session: {history.length} runs · tiers agree in {agreeing} of {tiered.length}
          {medianDelta !== null && <> · typical difference {medianDelta.toFixed(1)}°</>}
        </p>
      )}
      <p className="text-[11.5px] text-ink-400 leading-relaxed mt-3">
        The server logs every comparison. Browser readings stay a preview until
        these agree consistently.
      </p>
    </Section>
  );
}

function Legend() {
  const items = [
    { swatch: <span className="w-3 h-3 rounded-full border-2" style={{ borderColor: C.iris }} />,   label: "Iris" },
    { swatch: <span className="font-mono text-[12px] font-bold leading-none" style={{ color: "#0284C7" }}>+</span>, label: "Pupil centre" },
    { swatch: <span className="w-2.5 h-2.5 rounded-full" style={{ background: C.reflex }} />,     label: "Torch reflex" },
    { swatch: <span className="w-4 border-t-2 border-dashed" style={{ borderColor: "#14B8A6" }} />, label: "Pupil → reflex" },
    { swatch: <span className="w-4 border-t-2 border-dashed border-ink-400" />,                   label: "Pupil → pupil" },
    { swatch: <span className="w-2.5 h-2.5 rounded-full" style={{ background: "#D946EF" }} />,    label: "Eye corners" },
  ];
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1.5 px-1">
      {items.map((it) => (
        <li key={it.label} className="flex items-center gap-1.5 text-[12px] text-ink-500">
          {it.swatch}
          {it.label}
        </li>
      ))}
    </ul>
  );
}
