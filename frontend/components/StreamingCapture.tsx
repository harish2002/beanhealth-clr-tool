"use client";

/**
 * StreamingCapture
 * ================
 * Live camera feed with real-time MediaPipe eye-tracking overlay.
 *
 * What it does:
 *  1. Opens the back-facing camera with torch enabled
 *  2. Runs @mediapipe/face_mesh in the browser (WASM, ~30 fps) to detect iris
 *  3. Draws iris circles, pupil dots, and "L / R" labels on a canvas overlay
 *  4. When user taps "Start Analysis", captures 2 frames per second for 5 seconds (10 frames)
 *  5. Shows a countdown + per-frame pulse indicator while capturing
 *  6. Sends all frames to POST /analyse-stream and returns the aggregated result
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { analyseStream, checkHealth } from "@/lib/api";
import type { StreamSuccessResponse, StreamInconclusiveResponse } from "@/lib/types";

// ── Pipeline module definitions (for processing animation) ───
const PIPELINE_MODULES = [
  { name: "Eye Detection",       desc: "MediaPipe FaceMesh locating both irises" },
  { name: "Pupil Localisation",  desc: "Cross-validating pupil centre (2 methods)" },
  { name: "CLR Detection",       desc: "Isolating corneal light reflex (top 3% pixels)" },
  { name: "Displacement",        desc: "Measuring vector from pupil to light reflex" },
  { name: "Hirschberg Angle",    desc: "Converting displacement to clinical degrees" },
  { name: "Classification",      desc: "Applying triage criteria + ICD-10 code" },
  { name: "Report",              desc: "Generating annotated result image" },
] as const;

const TOTAL_FRAMES      = 10;    // frames to capture (2 fps × 5 s)
const FRAME_INTERVAL_MS = 500;   // 2 frames per second

// Expected processing time per frame on Railway at 640px (~1.5s), plus aggregation (~3s)
// 10 frames × 1.5s + 3s = ~18s — well within Railway's 60s proxy timeout.
const MS_PER_FRAME      = 1500;
const MS_AGGREGATION    = 3000;
const TOTAL_EXPECTED_MS = TOTAL_FRAMES * MS_PER_FRAME + MS_AGGREGATION;

// ── Iris landmark indices (MediaPipe FaceMesh) ──────────────
const LEFT_IRIS_INDICES  = [468, 469, 470, 471, 472];
const RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477];

type CaptureStatus =
  | "idle"           // waiting for user to start
  | "detecting"      // live preview — MediaPipe running, not yet capturing
  | "capturing"      // countdown active, capturing frames
  | "processing"     // sent to backend, waiting for response
  | "done";          // response received

// ── Live quality metrics ──────────────────────────────────────
interface QualityScore {
  eyesDetected: boolean;
  distance:     "too_close" | "ok" | "too_far" | "unknown";
  pose:         "ok" | "off_axis" | "unknown";
  openness:     "ok" | "closing" | "unknown";
  lighting:     "dark" | "ok" | "bright" | "unknown";
}

const QUALITY_DEFAULT: QualityScore = {
  eyesDetected: false,
  distance:     "unknown",
  pose:         "unknown",
  openness:     "unknown",
  lighting:     "unknown",
};

/** All 4 core checks must pass before capture is allowed. */
function isReadyToCapture(q: QualityScore): boolean {
  return q.eyesDetected && q.distance === "ok" && q.pose === "ok" && q.openness === "ok";
}

// Eyelid landmark indices for eye openness (MediaPipe FaceMesh)
// Subject's left eye  (iris indices 468-472): upper=159, lower=145
// Subject's right eye (iris indices 473-477): upper=386, lower=374
const LEFT_UPPER_LID  = 159;
const LEFT_LOWER_LID  = 145;
const RIGHT_UPPER_LID = 386;
const RIGHT_LOWER_LID = 374;

interface Props {
  patientName: string;
  patientAge: number;
  onSuccess: (result: StreamSuccessResponse) => void;
  onInconclusive: (result: StreamInconclusiveResponse) => void;
  onError: (message: string) => void;
}

export default function StreamingCapture({
  patientName,
  patientAge,
  onSuccess,
  onInconclusive,
  onError,
}: Props) {
  const videoRef   = useRef<HTMLVideoElement>(null);
  const canvasRef  = useRef<HTMLCanvasElement>(null);  // overlay canvas
  const streamRef  = useRef<MediaStream | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const faceMeshRef = useRef<any>(null);

  const [status,        setStatus]        = useState<CaptureStatus>("idle");
  const [eyesDetected,  setEyesDetected]  = useState(false);
  const [countdown,     setCountdown]     = useState(TOTAL_FRAMES);
  const [capturedCount, setCapturedCount] = useState(0);
  const [torchOn,       setTorchOn]       = useState(false);
  const [cameraError,   setCameraError]   = useState<string | null>(null);

  // Server warm-up state
  const [serverReady, setServerReady] = useState<boolean | null>(null); // null=checking

  // Processing animation state
  const [procFrame,    setProcFrame]    = useState(0);   // 0-based frame index
  const [procModule,   setProcModule]   = useState(0);   // 0-6 module index
  const [procProgress, setProcProgress] = useState(0);   // 0-100
  const [isAggregating, setIsAggregating] = useState(false);
  const procTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const procStartRef = useRef<number>(0);

  // Live quality meter
  const [quality, setQuality] = useState<QualityScore>(QUALITY_DEFAULT);
  const qualityTickRef = useRef<number>(0); // throttle quality state updates to 5fps

  // Accumulate captured frames as blobs
  const framesRef    = useRef<Blob[]>([]);
  const capturingRef = useRef(false);
  const intervalRef  = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Wake-up ping on mount ─────────────────────────────────

  useEffect(() => {
    setServerReady(null);
    checkHealth().then((ok) => setServerReady(ok));
  }, []);

  // ── Processing animation timer ────────────────────────────

  useEffect(() => {
    if (status === "processing") {
      procStartRef.current = Date.now();
      setProcFrame(0);
      setProcModule(0);
      setProcProgress(0);
      setIsAggregating(false);

      procTimerRef.current = setInterval(() => {
        const elapsed = Date.now() - procStartRef.current;
        const frameDone = Math.min(
          Math.floor(elapsed / MS_PER_FRAME),
          TOTAL_FRAMES - 1
        );
        const moduleIdx = Math.floor(
          ((elapsed % MS_PER_FRAME) / MS_PER_FRAME) * PIPELINE_MODULES.length
        ) % PIPELINE_MODULES.length;
        const progress = Math.min((elapsed / TOTAL_EXPECTED_MS) * 100, 97);
        const aggregating = elapsed > TOTAL_FRAMES * MS_PER_FRAME;

        setProcFrame(frameDone);
        setProcModule(moduleIdx);
        setProcProgress(progress);
        setIsAggregating(aggregating);
      }, 100);
    } else {
      if (procTimerRef.current) {
        clearInterval(procTimerRef.current);
        procTimerRef.current = null;
      }
    }
    return () => {
      if (procTimerRef.current) clearInterval(procTimerRef.current);
    };
  }, [status]);

  // ── Luminance sampling (2 fps, separate from MediaPipe) ─────
  // Samples a 64×36 thumbnail of the video to estimate ambient brightness.
  // This runs independently of MediaPipe so it still works even if face
  // detection is momentarily lost.

  useEffect(() => {
    if (status !== "detecting" && status !== "capturing") return;

    const id = setInterval(() => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) return;

      const tmp = document.createElement("canvas");
      tmp.width = 64; tmp.height = 36;
      const ctx = tmp.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, 64, 36);
      const data = ctx.getImageData(0, 0, 64, 36).data;
      let lum = 0;
      for (let i = 0; i < data.length; i += 4) {
        // Rec. 709 luminance weights
        lum += data[i] * 0.2126 + data[i + 1] * 0.7152 + data[i + 2] * 0.0722;
      }
      lum /= data.length / 4;

      // With the torch on the eye crop will be bright, but the overall frame
      // is dominated by ambient light. Thresholds (0-255):
      //   < 20 → very dark room, torch may not produce a clean reflex
      //   > 200 → overexposed / bright sunlight, CLR will wash out
      const lighting: QualityScore["lighting"] =
        lum < 20 ? "dark" : lum > 200 ? "bright" : "ok";

      setQuality((q) => ({ ...q, lighting }));
    }, 500);

    return () => clearInterval(id);
  }, [status]);

  // ── Start camera ──────────────────────────────────────────

  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width:  { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // Auto-enable torch on back camera
      const track = stream.getVideoTracks()[0];
      try {
        await track.applyConstraints({ advanced: [{ torch: true } as MediaTrackConstraintSet] });
        setTorchOn(true);
      } catch {
        setTorchOn(false);
      }

      setStatus("detecting");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Camera access denied";
      setCameraError(msg);
    }
  }, []);

  // ── Stop camera ───────────────────────────────────────────

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setQuality(QUALITY_DEFAULT);
  }, []);

  // ── Load MediaPipe FaceMesh dynamically ───────────────────

  useEffect(() => {
    let cancelled = false;

    async function loadFaceMesh() {
      try {
        // Dynamically import to avoid SSR crash
        const { FaceMesh } = await import("@mediapipe/face_mesh");

        if (cancelled) return;

        const fm = new FaceMesh({
          locateFile: (file: string) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}`,
        });

        fm.setOptions({
          maxNumFaces:         1,
          refineLandmarks:     true,   // enables iris (468–477)
          minDetectionConfidence: 0.5,
          minTrackingConfidence:  0.5,
        });

        fm.onResults((results: FaceMeshResults) => {
          if (cancelled) return;
          drawOverlay(results);
        });

        faceMeshRef.current = fm;
      } catch (e) {
        console.warn("[FaceMesh] Failed to load:", e);
      }
    }

    loadFaceMesh();
    return () => { cancelled = true; };
  }, []);

  // ── Run FaceMesh on each video frame ─────────────────────

  useEffect(() => {
    if (status !== "detecting" && status !== "capturing") return;

    let animId: number;
    const tick = async () => {
      if (videoRef.current && faceMeshRef.current &&
          videoRef.current.readyState >= 2) {
        await faceMeshRef.current.send({ image: videoRef.current });
      }
      animId = requestAnimationFrame(tick);
    };
    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
  }, [status]);

  // ── Draw iris overlay on canvas ───────────────────────────

  function drawOverlay(results: FaceMeshResults) {
    const canvas = canvasRef.current;
    const video  = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Use the canvas's CSS display size as internal resolution so coordinates
    // match exactly what the user sees (accounts for object-cover cropping).
    const dispW = canvas.offsetWidth  || video.videoWidth  || 640;
    const dispH = canvas.offsetHeight || video.videoHeight || 480;
    canvas.width  = dispW;
    canvas.height = dispH;
    ctx.clearRect(0, 0, dispW, dispH);

    const lms = results.multiFaceLandmarks?.[0];
    if (!lms) {
      setEyesDetected(false);
      setQuality((q) => ({
        ...q,
        eyesDetected: false,
        distance: "unknown",
        pose:     "unknown",
        openness: "unknown",
      }));
      return;
    }

    setEyesDetected(true);

    // Compute the visible region of the video that object-cover shows.
    // MediaPipe landmarks are normalised to the FULL native video frame,
    // so we need to map them through the crop offset to canvas pixels.
    const videoW = video.videoWidth  || dispW;
    const videoH = video.videoHeight || dispH;
    const videoAspect = videoW / videoH;
    const dispAspect  = dispW  / dispH;

    let srcX = 0, srcY = 0, srcW = videoW, srcH = videoH;
    if (videoAspect > dispAspect) {
      // Video is wider → sides are cropped
      srcH = videoH;
      srcW = videoH * dispAspect;
      srcX = (videoW - srcW) / 2;
    } else {
      // Video is taller → top/bottom are cropped
      srcW = videoW;
      srcH = videoW / dispAspect;
      srcY = (videoH - srcH) / 2;
    }

    const W = dispW;
    const H = dispH;

    // Map a normalised landmark to canvas pixel coordinates
    function px(lm: { x: number; y: number }) {
      const vx = lm.x * videoW;
      const vy = lm.y * videoH;
      return {
        x: (vx - srcX) / srcW * W,
        y: (vy - srcY) / srcH * H,
      };
    }

    // Draw iris for one eye
    function drawIris(indices: number[], label: string) {
      if (!ctx) return;
      const pts = indices.map((i) => px(lms![i]));
      const centre = pts[0]; // index 0 of iris group is the centre point

      // Estimate radius from spread of 4 edge points
      const spread = pts.slice(1).map((p) => Math.hypot(p.x - centre.x, p.y - centre.y));
      const radius = spread.reduce((a, b) => a + b, 0) / spread.length;

      // Iris circle — mint green
      ctx.beginPath();
      ctx.arc(centre.x, centre.y, radius, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(52, 211, 153, 0.85)";
      ctx.lineWidth   = 2;
      ctx.stroke();

      // Pupil dot — white
      ctx.beginPath();
      ctx.arc(centre.x, centre.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
      ctx.fill();

      // L / R label
      ctx.font         = "bold 13px system-ui, sans-serif";
      ctx.fillStyle    = "rgba(255, 255, 255, 0.9)";
      ctx.shadowColor  = "rgba(0,0,0,0.7)";
      ctx.shadowBlur   = 3;
      ctx.fillText(label, centre.x - radius - 18, centre.y + 5);
      ctx.shadowBlur   = 0;
    }

    drawIris(LEFT_IRIS_INDICES,  "L");
    drawIris(RIGHT_IRIS_INDICES, "R");

    // ── Quality metrics (throttled to 5 fps) ─────────────────
    const now = performance.now();
    if (now - qualityTickRef.current > 200) {
      qualityTickRef.current = now;

      // Iris centres and radii in canvas-pixel space
      const liPts = LEFT_IRIS_INDICES.map((i) => px(lms[i]));
      const riPts = RIGHT_IRIS_INDICES.map((i) => px(lms[i]));
      const lc = liPts[0];
      const rc = riPts[0];
      const lr = liPts.slice(1).map((p) => Math.hypot(p.x - lc.x, p.y - lc.y))
                       .reduce((a, b) => a + b, 0) / 4;
      const rr = riPts.slice(1).map((p) => Math.hypot(p.x - rc.x, p.y - rc.y))
                       .reduce((a, b) => a + b, 0) / 4;
      const avgR = (lr + rr) / 2;

      // ① Distance — iris diameter as % of canvas height
      // Typical phone at 30–40 cm: iris ≈ 7–13 % of frame height
      const irisHRatio = (avgR * 2) / H;
      const distance: QualityScore["distance"] =
        irisHRatio > 0.15 ? "too_close" :
        irisHRatio < 0.04 ? "too_far"   : "ok";

      // ② Head pose — horizontal yaw + vertical tilt from iris symmetry
      // For a frontal face: lx + rx ≈ W  (mirrored symmetry)
      const yawErr  = Math.abs((lc.x + rc.x) / W - 1.0);        // ideal = 0
      const tiltErr = Math.abs(lc.y - rc.y) / (avgR * 2);        // in iris-diameters
      const pose: QualityScore["pose"] =
        (yawErr > 0.12 || tiltErr > 0.5) ? "off_axis" : "ok";

      // ③ Eye openness — eyelid gap vs iris diameter (eye aspect ratio)
      const luY = px(lms[LEFT_UPPER_LID]).y;
      const llY = px(lms[LEFT_LOWER_LID]).y;
      const ruY = px(lms[RIGHT_UPPER_LID]).y;
      const rlY = px(lms[RIGHT_LOWER_LID]).y;
      const lEAR = (llY - luY) / (lr * 2);
      const rEAR = (rlY - ruY) / (rr * 2);
      const openness: QualityScore["openness"] =
        Math.min(lEAR, rEAR) > 0.25 ? "ok" : "closing";

      setQuality((q) => ({ ...q, eyesDetected: true, distance, pose, openness }));
    }

    // "Eyes detected" badge
    ctx.font         = "12px system-ui, sans-serif";
    ctx.fillStyle    = "rgba(52, 211, 153, 0.9)";
    ctx.shadowColor  = "rgba(0,0,0,0.5)";
    ctx.shadowBlur   = 4;
    ctx.fillText("● Eyes detected", 12, 20);
    ctx.shadowBlur   = 0;
  }

  // ── Capture a single frame as JPEG blob ───────────────────
  //
  // Frames are downscaled to a max of 640 px wide before encoding.
  // The CLR pipeline only needs the iris to be ~40–60 px wide to
  // work reliably; sending full 1280×720 camera frames (~920 K pixels)
  // forces MediaPipe to process 4× more data per frame, inflating
  // per-frame backend time from ~1.5 s to ~6 s.
  // At 640×360 MediaPipe still detects the iris accurately and the
  // total 10-frame upload + processing stays well under 30 s.

  const CAPTURE_MAX_WIDTH = 640;

  function captureFrame(): Promise<Blob | null> {
    return new Promise((resolve) => {
      const video = videoRef.current;
      if (!video) return resolve(null);

      // Scale to max width, preserve aspect ratio
      const srcW  = video.videoWidth  || 1280;
      const srcH  = video.videoHeight || 720;
      const scale = Math.min(1, CAPTURE_MAX_WIDTH / srcW);
      const dstW  = Math.round(srcW * scale);
      const dstH  = Math.round(srcH * scale);

      const offscreen = document.createElement("canvas");
      offscreen.width  = dstW;
      offscreen.height = dstH;
      const ctx = offscreen.getContext("2d");
      if (!ctx) return resolve(null);
      ctx.drawImage(video, 0, 0, dstW, dstH);
      // Quality 0.88 — indistinguishable from 0.92 for CLR analysis
      // but ~15 % smaller file, slightly faster upload on mobile
      offscreen.toBlob((blob) => resolve(blob), "image/jpeg", 0.88);
    });
  }

  // ── Send frames to /analyse-stream ───────────────────────

  const submitFrames = useCallback(async () => {
    setStatus("processing");
    stopCamera();

    const blobs = framesRef.current;
    if (blobs.length === 0) {
      onError("No frames were captured. Please try again.");
      return;
    }

    try {
      const files = blobs.map(
        (b, i) => new File([b], `frame_${i}.jpg`, { type: "image/jpeg" })
      );

      const result = await analyseStream(files, patientName, patientAge);

      if (result.status === "SUCCESS") {
        onSuccess(result as StreamSuccessResponse);
      } else if (result.status === "INCONCLUSIVE") {
        onInconclusive(result as StreamInconclusiveResponse);
      } else {
        onError("An unexpected error occurred. Please retry.");
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Network error";
      onError(msg);
    }
  }, [patientName, patientAge, onSuccess, onInconclusive, onError, stopCamera]);

  // ── Start the 10-frame capture sequence ───────────────────

  const startCapture = useCallback(async () => {
    if (!eyesDetected) return;

    framesRef.current = [];
    capturingRef.current = true;
    setStatus("capturing");
    setCountdown(TOTAL_FRAMES);
    setCapturedCount(0);

    let remaining = TOTAL_FRAMES;

    intervalRef.current = setInterval(async () => {
      if (!capturingRef.current) return;

      const blob = await captureFrame();
      if (blob) {
        framesRef.current.push(blob);
        setCapturedCount((c) => c + 1);
      }

      remaining -= 1;
      setCountdown(remaining);

      if (remaining <= 0) {
        clearInterval(intervalRef.current!);
        capturingRef.current = false;
        submitFrames();
      }
    }, FRAME_INTERVAL_MS);
  }, [eyesDetected, submitFrames]);

  // ── Cleanup on unmount ────────────────────────────────────

  useEffect(() => {
    return () => {
      capturingRef.current = false;
      if (intervalRef.current) clearInterval(intervalRef.current);
      stopCamera();
    };
  }, [stopCamera]);

  // ── UI ────────────────────────────────────────────────────

  const progressPct = ((TOTAL_FRAMES - countdown) / TOTAL_FRAMES) * 100;

  return (
    <div className="flex flex-col items-center gap-4 w-full max-w-md mx-auto">

      {/* Camera + overlay */}
      <div className="relative w-full aspect-[4/3] bg-black rounded-2xl overflow-hidden shadow-lg">
        <video
          ref={videoRef}
          playsInline
          muted
          className="absolute inset-0 w-full h-full object-cover"
        />
        {/* MediaPipe iris overlay */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {/* Torch badge */}
        {torchOn && (
          <div className="absolute top-3 right-3 bg-amber-400 text-amber-900 text-xs font-semibold px-2 py-0.5 rounded-full">
            Torch ON
          </div>
        )}

        {/* Capturing progress bar */}
        {status === "capturing" && (
          <div className="absolute bottom-0 left-0 right-0 h-1.5 bg-white/20">
            <div
              className="h-full bg-emerald-400 transition-all duration-1000 ease-linear"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        )}

        {/* Processing overlay */}
        {status === "processing" && (
          <div className="absolute inset-0 bg-black/75 flex flex-col items-center justify-center gap-4 px-5">
            {/* Progress bar */}
            <div className="w-full bg-white/20 rounded-full h-1.5 overflow-hidden">
              <div
                className="h-full bg-emerald-400 rounded-full transition-all duration-300 ease-linear"
                style={{ width: `${procProgress}%` }}
              />
            </div>

            {/* Frame counter / aggregating label */}
            <div className="text-center">
              {isAggregating ? (
                <p className="text-emerald-300 text-xs font-semibold uppercase tracking-widest">
                  Aggregating results…
                </p>
              ) : (
                <p className="text-white/60 text-xs">
                  Frame <span className="text-white font-semibold">{procFrame + 1}</span> of {TOTAL_FRAMES}
                </p>
              )}
            </div>

            {/* Module steps */}
            <div className="w-full flex flex-col gap-1.5">
              {PIPELINE_MODULES.map((mod, i) => {
                const done    = !isAggregating && i < procModule;
                const active  = !isAggregating && i === procModule;
                const pending = isAggregating || i > procModule;
                return (
                  <div
                    key={mod.name}
                    className={`flex items-center gap-2.5 px-3 py-1.5 rounded-lg transition-all duration-200 ${
                      active  ? "bg-emerald-500/25 border border-emerald-400/40" :
                      done    ? "opacity-50" : "opacity-25"
                    }`}
                  >
                    {/* Status icon */}
                    <span className="w-4 h-4 flex-shrink-0 flex items-center justify-center">
                      {done   && <span className="text-emerald-400 text-xs">✓</span>}
                      {active && <span className="w-3 h-3 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin block" />}
                      {pending && <span className="w-1.5 h-1.5 rounded-full bg-white/30 block" />}
                    </span>
                    <div className="min-w-0">
                      <p className={`text-xs font-semibold truncate ${active ? "text-emerald-300" : done ? "text-white/70" : "text-white/40"}`}>
                        {i + 1}. {mod.name}
                      </p>
                      {active && (
                        <p className="text-white/50 text-[10px] truncate">{mod.desc}</p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Status bar */}
      <div className="w-full">
        {status === "idle" && !cameraError && (
          <div className="flex flex-col gap-2 w-full">
            {/* Server warm-up status */}
            <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
              serverReady === null ? "bg-amber-50 border border-amber-200 text-amber-700" :
              serverReady         ? "bg-emerald-50 border border-emerald-200 text-emerald-700" :
                                    "bg-red-50 border border-red-200 text-red-600"
            }`}>
              {serverReady === null && <span className="w-3 h-3 border-2 border-amber-400 border-t-transparent rounded-full animate-spin flex-shrink-0" />}
              {serverReady === true  && <span className="text-emerald-500 flex-shrink-0">●</span>}
              {serverReady === false && <span className="text-red-400 flex-shrink-0">●</span>}
              {serverReady === null  ? "Warming up server…"     :
               serverReady          ? "Server ready"           :
                                      "Server unreachable — check connection"}
            </div>
            <button
              onClick={startCamera}
              className="w-full py-3.5 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-xl transition-colors"
            >
              Enable Camera
            </button>
          </div>
        )}

        {cameraError && (
          <div className="w-full p-3 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm text-center">
            {cameraError}
          </div>
        )}

        {status === "detecting" && (
          <div className="flex flex-col gap-3 w-full">
            {/* ── Live quality card ── */}
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
              <p className="text-slate-400 text-[10px] font-semibold uppercase tracking-widest mb-3">
                Live capture quality
              </p>
              <div className="grid grid-cols-2 gap-2">
                {(
                  [
                    {
                      label:   "Eyes found",
                      ok:      quality.eyesDetected,
                      unknown: false,
                      okTip:   "Both irises detected",
                      warnTip: "Aim torch at both eyes",
                    },
                    {
                      label:   "Distance",
                      ok:      quality.distance === "ok",
                      unknown: quality.distance === "unknown",
                      okTip:   "30–40 cm away ✓",
                      warnTip: quality.distance === "too_close"
                                 ? "Move further back"
                                 : quality.distance === "too_far"
                                 ? "Move closer"
                                 : "Detecting…",
                    },
                    {
                      label:   "Head angle",
                      ok:      quality.pose === "ok",
                      unknown: quality.pose === "unknown",
                      okTip:   "Facing straight ✓",
                      warnTip: "Look straight at camera",
                    },
                    {
                      label:   "Eyes open",
                      ok:      quality.openness === "ok",
                      unknown: quality.openness === "unknown",
                      okTip:   "Both eyes open ✓",
                      warnTip: "Open eyes wider",
                    },
                    {
                      label:   "Lighting",
                      ok:      quality.lighting === "ok",
                      unknown: quality.lighting === "unknown",
                      okTip:   "Good ambient light",
                      warnTip: quality.lighting === "dark"
                                 ? "Room too dark"
                                 : quality.lighting === "bright"
                                 ? "Too bright / outdoors"
                                 : "Checking…",
                    },
                  ] as const
                ).map(({ label, ok, unknown, okTip, warnTip }) => (
                  <div
                    key={label}
                    className={`flex items-start gap-2 px-3 py-2 rounded-lg transition-colors duration-300 ${
                      unknown
                        ? "bg-slate-50"
                        : ok
                        ? "bg-emerald-50"
                        : "bg-amber-50"
                    }`}
                  >
                    <span
                      className={`mt-1 w-2 h-2 rounded-full flex-shrink-0 ${
                        unknown
                          ? "bg-slate-300"
                          : ok
                          ? "bg-emerald-500"
                          : "bg-amber-400 animate-pulse"
                      }`}
                    />
                    <div className="min-w-0">
                      <p
                        className={`text-xs font-semibold leading-tight ${
                          unknown
                            ? "text-slate-400"
                            : ok
                            ? "text-emerald-700"
                            : "text-amber-700"
                        }`}
                      >
                        {label}
                      </p>
                      <p className="text-[10px] text-slate-400 leading-tight mt-0.5 truncate">
                        {unknown ? "Waiting…" : ok ? okTip : warnTip}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              {/* Overall readiness summary */}
              {quality.eyesDetected && (
                <p
                  className={`mt-3 text-xs text-center font-medium ${
                    isReadyToCapture(quality)
                      ? "text-emerald-600"
                      : "text-amber-600"
                  }`}
                >
                  {isReadyToCapture(quality)
                    ? "✓ All checks passed — ready to scan"
                    : "Fix the amber items above for best accuracy"}
                </p>
              )}
            </div>

            <button
              onClick={startCapture}
              disabled={!isReadyToCapture(quality)}
              className={`w-full py-3.5 font-semibold rounded-xl transition-all ${
                isReadyToCapture(quality)
                  ? "bg-emerald-600 hover:bg-emerald-700 text-white shadow-md shadow-emerald-600/25"
                  : "bg-slate-200 text-slate-400 cursor-not-allowed"
              }`}
            >
              {isReadyToCapture(quality)
                ? "Start 10-Frame Analysis"
                : "Align for best results…"}
            </button>
          </div>
        )}

        {status === "capturing" && (
          <div className="flex flex-col gap-2 w-full">
            {/* Frame dots */}
            <div className="flex justify-center gap-1.5">
              {Array.from({ length: TOTAL_FRAMES }).map((_, i) => (
                <div
                  key={i}
                  className={`w-2.5 h-2.5 rounded-full transition-colors ${
                    i < capturedCount
                      ? "bg-emerald-500"
                      : i === capturedCount
                      ? "bg-emerald-300 animate-pulse"
                      : "bg-slate-200"
                  }`}
                />
              ))}
            </div>
            <p className="text-center text-slate-600 text-sm font-medium">
              Hold steady — capturing frame {capturedCount + 1} of {TOTAL_FRAMES}
            </p>
          </div>
        )}
      </div>

      {/* Instructions */}
      {(status === "idle" || status === "detecting") && (
        <div className="w-full bg-slate-50 rounded-xl border border-slate-100 p-4">
          <p className="text-slate-700 text-xs font-medium mb-2">How to get the best result</p>
          <ul className="space-y-1 text-slate-500 text-xs">
            <li>• Hold the phone <strong>30 cm</strong> from the patient&apos;s face</li>
            <li>• Ensure <strong>torch is on</strong> — you should see reflections in both eyes</li>
            <li>• Keep <strong>eyes open and looking straight</strong> at the camera</li>
            <li>• The app captures <strong>10 frames over 5 seconds</strong> and averages them</li>
          </ul>
        </div>
      )}
    </div>
  );
}

// ── Types for MediaPipe results (lightweight, no import needed) ──

interface FaceMeshLandmark {
  x: number;
  y: number;
  z: number;
}

interface FaceMeshResults {
  multiFaceLandmarks?: FaceMeshLandmark[][];
}

// Extend window for FaceMesh class loaded via CDN
declare global {
  interface Window {
    __FaceMesh: unknown;
  }
}
