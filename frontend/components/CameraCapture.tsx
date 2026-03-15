"use client";

/**
 * CameraCapture
 *
 * Handles:
 *   - getUserMedia with facingMode state (back / front)
 *   - Auto-enables torch when back camera starts
 *   - Manual torch toggle (top-right)
 *   - Camera flip button (bottom-left)
 *   - Still-frame capture → base64 data URL + File object
 *   - All camera and torch logic lives exclusively in this component
 */

import { useRef, useEffect, useState } from "react";

interface CameraCaptureProps {
  onCapture: (dataUrl: string, file: File) => void;
}

export default function CameraCapture({ onCapture }: CameraCaptureProps) {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const trackRef  = useRef<MediaStreamTrack | null>(null);

  const [facingMode, setFacingMode] = useState<"environment" | "user">("environment");
  const [initKey,    setInitKey]    = useState(0);   // bump to force re-init on retry
  const [torchOn,    setTorchOn]    = useState(false);
  const [torchAvail, setTorchAvail] = useState(false);
  const [cameraErr,  setCameraErr]  = useState<string | null>(null);
  const [ready,      setReady]      = useState(false);
  const [captured,   setCaptured]   = useState(false);

  // ── Camera init — re-runs when facingMode or initKey changes ────────────
  useEffect(() => {
    let active = true;

    async function init() {
      // Tear down any existing stream
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      trackRef.current  = null;
      setReady(false);
      setTorchOn(false);
      setTorchAvail(false);
      setCameraErr(null);

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: facingMode },
            width:  { ideal: 1920 },
            height: { ideal: 1080 },
          },
          audio: false,
        });

        if (!active) { stream.getTracks().forEach((t) => t.stop()); return; }

        streamRef.current = stream;
        const track = stream.getVideoTracks()[0];
        trackRef.current = track;

        // Detect torch support
        const caps = track.getCapabilities() as MediaTrackCapabilities & { torch?: boolean };
        if (caps.torch) {
          setTorchAvail(true);
          // Auto-enable torch on back camera
          if (facingMode === "environment") {
            try {
              await track.applyConstraints({
                advanced: [{ torch: true } as MediaTrackConstraintSet],
              });
              if (active) setTorchOn(true);
            } catch {
              // torch auto-enable failed — user can toggle manually
            }
          }
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => { if (active) setReady(true); };
        }
      } catch (err) {
        if (!active) return;
        if (err instanceof Error) {
          if (err.name === "NotAllowedError") {
            setCameraErr(
              "Camera permission denied. Please allow camera access in your browser settings.",
            );
          } else if (err.name === "NotFoundError") {
            setCameraErr("No camera found on this device.");
          } else {
            setCameraErr(
              "Could not access the camera. Please check your permissions and try again.",
            );
          }
        }
      }
    }

    init();
    return () => {
      active = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [facingMode, initKey]);

  // ── Torch toggle ──────────────────────────────────────────────────────────
  async function toggleTorch() {
    const track = trackRef.current;
    if (!track || !torchAvail) return;
    const next = !torchOn;
    try {
      await track.applyConstraints({
        advanced: [{ torch: next } as MediaTrackConstraintSet],
      });
      setTorchOn(next);
    } catch (err) {
      console.warn("Torch toggle failed:", err);
    }
  }

  // ── Flip camera ───────────────────────────────────────────────────────────
  function flipCamera() {
    setCaptured(false);
    setFacingMode((prev) => (prev === "environment" ? "user" : "environment"));
  }

  // ── Capture still frame ───────────────────────────────────────────────────
  function captureFrame() {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !ready) return;

    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], "clr-capture.jpg", { type: "image/jpeg" });
        setCaptured(true);
        onCapture(dataUrl, file);
      },
      "image/jpeg",
      0.92,
    );
  }

  // ── Error state ───────────────────────────────────────────────────────────
  if (cameraErr) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 px-6 text-center bg-black">
        <div className="w-14 h-14 rounded-full bg-red-900/60 flex items-center justify-center">
          <svg
            className="w-7 h-7 text-red-400"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.8}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z"
            />
          </svg>
        </div>
        <p className="text-slate-300 text-sm leading-relaxed max-w-xs">{cameraErr}</p>
        <button
          onClick={() => setInitKey((k) => k + 1)}
          className="mt-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }

  return (
    <div className="relative flex flex-col h-full bg-black">
      {/* Live video feed */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover"
      />

      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Face-guide oval */}
      {ready && !captured && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="border-2 border-blue-400/70 rounded-full w-64 h-40 pulse-ring" />
        </div>
      )}

      {/* Top-right: torch toggle */}
      {torchAvail && (
        <div className="absolute top-4 right-4">
          <button
            onClick={toggleTorch}
            className={`w-12 h-12 rounded-full flex items-center justify-center shadow-lg transition-all active:scale-95
              ${torchOn
                ? "bg-yellow-400 text-yellow-900 shadow-yellow-400/50"
                : "bg-black/60 text-white border border-white/25 backdrop-blur-sm"}`}
            aria-label={torchOn ? "Turn torch off" : "Turn torch on"}
          >
            {/* Lightning bolt */}
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path
                fillRule="evenodd"
                clipRule="evenodd"
                d="M13 10.5h6.197L12 21.704V13.5H5.803L13 2.296V10.5Z"
              />
            </svg>
          </button>
        </div>
      )}

      {/* Torch status hint */}
      {ready && !captured && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 pointer-events-none">
          {torchAvail && !torchOn && (
            <span className="bg-yellow-400/90 text-yellow-900 text-xs font-semibold px-3 py-1.5 rounded-full whitespace-nowrap">
              ⚡ Tap torch to enable
            </span>
          )}
          {!torchAvail && (
            <span className="bg-black/60 text-yellow-300 text-xs px-3 py-1.5 rounded-full border border-yellow-400/30 backdrop-blur-sm whitespace-nowrap">
              ⚡ Enable phone torch before capturing
            </span>
          )}
        </div>
      )}

      {/* Bottom controls: flip (left) · shutter (centre) · spacer (right) */}
      {ready && !captured && (
        <div className="absolute bottom-0 left-0 right-0 pb-10 px-8 flex items-center justify-between">
          {/* Flip camera */}
          <button
            onClick={flipCamera}
            className="w-12 h-12 rounded-full bg-black/60 border border-white/25 backdrop-blur-sm
                       flex items-center justify-center text-white transition-all active:scale-95 shadow-lg"
            aria-label="Flip camera"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.8}
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99"
              />
            </svg>
          </button>

          {/* Shutter */}
          <button
            onClick={captureFrame}
            className="w-20 h-20 rounded-full bg-white border-4 border-white/80
                       flex items-center justify-center shadow-2xl
                       active:scale-95 transition-transform"
            aria-label="Capture photo"
          >
            <div className="w-14 h-14 rounded-full bg-white border-2 border-slate-300" />
          </button>

          {/* Right spacer — mirrors flip button width for centering */}
          <div className="w-12" />
        </div>
      )}

      {/* Captured confirmation overlay */}
      {captured && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/70">
          <div className="flex flex-col items-center gap-3">
            <div className="w-16 h-16 rounded-full bg-blue-600 flex items-center justify-center">
              <svg
                className="w-8 h-8 text-white"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={2.5}
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
              </svg>
            </div>
            <p className="text-white font-semibold tracking-wide">Analysing…</p>
          </div>
        </div>
      )}
    </div>
  );
}
