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
 *  4. When user taps "Start Analysis", captures 1 frame per second for N seconds
 *  5. Shows a countdown + per-frame pulse indicator while capturing
 *  6. Sends all frames to POST /analyse-stream and returns the aggregated result
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { analyseStream } from "@/lib/api";
import type { StreamSuccessResponse, StreamInconclusiveResponse } from "@/lib/types";

// ── Iris landmark indices (MediaPipe FaceMesh) ──────────────
const LEFT_IRIS_INDICES  = [468, 469, 470, 471, 472];
const RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477];

const TOTAL_FRAMES  = 10;   // frames to capture
const FRAME_INTERVAL_MS = 1000; // 1 frame per second

type CaptureStatus =
  | "idle"           // waiting for user to start
  | "detecting"      // live preview — MediaPipe running, not yet capturing
  | "capturing"      // countdown active, capturing frames
  | "processing"     // sent to backend, waiting for response
  | "done";          // response received

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

  const [status,       setStatus]       = useState<CaptureStatus>("idle");
  const [eyesDetected, setEyesDetected] = useState(false);
  const [countdown,    setCountdown]    = useState(TOTAL_FRAMES);
  const [capturedCount, setCapturedCount] = useState(0);
  const [torchOn,      setTorchOn]      = useState(false);
  const [cameraError,  setCameraError]  = useState<string | null>(null);

  // Accumulate captured frames as blobs
  const framesRef    = useRef<Blob[]>([]);
  const capturingRef = useRef(false);
  const intervalRef  = useRef<ReturnType<typeof setInterval> | null>(null);

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

    canvas.width  = video.videoWidth  || canvas.offsetWidth;
    canvas.height = video.videoHeight || canvas.offsetHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const lms = results.multiFaceLandmarks?.[0];
    if (!lms) {
      setEyesDetected(false);
      return;
    }

    setEyesDetected(true);

    const W = canvas.width;
    const H = canvas.height;

    function px(lm: { x: number; y: number }) {
      return { x: lm.x * W, y: lm.y * H };
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

    // "Eyes detected" badge
    ctx.font         = "12px system-ui, sans-serif";
    ctx.fillStyle    = "rgba(52, 211, 153, 0.9)";
    ctx.shadowColor  = "rgba(0,0,0,0.5)";
    ctx.shadowBlur   = 4;
    ctx.fillText("● Eyes detected", 12, 20);
    ctx.shadowBlur   = 0;
  }

  // ── Capture a single frame as JPEG blob ───────────────────

  function captureFrame(): Promise<Blob | null> {
    return new Promise((resolve) => {
      const video = videoRef.current;
      if (!video) return resolve(null);

      const offscreen = document.createElement("canvas");
      offscreen.width  = video.videoWidth;
      offscreen.height = video.videoHeight;
      const ctx = offscreen.getContext("2d");
      if (!ctx) return resolve(null);
      ctx.drawImage(video, 0, 0);
      offscreen.toBlob((blob) => resolve(blob), "image/jpeg", 0.92);
    });
  }

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
  }, [eyesDetected]);

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
          <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center gap-3">
            <div className="w-10 h-10 border-4 border-white border-t-emerald-400 rounded-full animate-spin" />
            <p className="text-white text-sm font-medium">Analysing {capturedCount} frames…</p>
          </div>
        )}
      </div>

      {/* Status bar */}
      <div className="w-full">
        {status === "idle" && !cameraError && (
          <button
            onClick={startCamera}
            className="w-full py-3.5 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded-xl transition-colors"
          >
            Enable Camera
          </button>
        )}

        {cameraError && (
          <div className="w-full p-3 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm text-center">
            {cameraError}
          </div>
        )}

        {status === "detecting" && (
          <div className="flex flex-col gap-3 w-full">
            {/* Eye detection status */}
            <div className={`flex items-center gap-2 px-4 py-2.5 rounded-xl border text-sm font-medium transition-colors ${
              eyesDetected
                ? "bg-emerald-50 border-emerald-200 text-emerald-700"
                : "bg-amber-50 border-amber-200 text-amber-700"
            }`}>
              <span className={`w-2 h-2 rounded-full flex-shrink-0 ${eyesDetected ? "bg-emerald-500" : "bg-amber-400"}`} />
              {eyesDetected
                ? "Both eyes detected — ready to scan"
                : "Searching for eyes… point torch at both eyes from ~30 cm"}
            </div>

            <button
              onClick={startCapture}
              disabled={!eyesDetected}
              className={`w-full py-3.5 font-semibold rounded-xl transition-all ${
                eyesDetected
                  ? "bg-emerald-600 hover:bg-emerald-700 text-white shadow-md"
                  : "bg-slate-200 text-slate-400 cursor-not-allowed"
              }`}
            >
              Start 10-Frame Analysis
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
            <li>• Hold the phone <strong>30 cm</strong> from the patient's face</li>
            <li>• Ensure <strong>torch is on</strong> — you should see reflections in both eyes</li>
            <li>• Keep <strong>eyes open and looking straight</strong> at the camera</li>
            <li>• The app captures <strong>10 frames over 10 seconds</strong> and averages them</li>
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
