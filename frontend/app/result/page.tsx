"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/useAppStore";
import TriageReport from "@/components/TriageReport";
import type { SuccessResponse, InconclusiveResponse } from "@/lib/types";

// ─── Loading screen ────────────────────────────────────────────────────────────
function LoadingScreen() {
  const steps = ["Detecting eyes", "Locating pupils", "Finding CLR", "Measuring"];
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-slate-50 gap-6 px-6 text-center">
      {/* Spinner */}
      <div className="relative w-20 h-20">
        <div className="absolute inset-0 rounded-full border-4 border-blue-100" />
        <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-blue-600 spin-slow" />
        <div className="absolute inset-0 flex items-center justify-center">
          <svg
            className="w-8 h-8 text-blue-500"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
          </svg>
        </div>
      </div>

      <div className="text-center">
        <h2 className="text-slate-900 font-semibold text-lg">Analysing…</h2>
        <p className="text-slate-500 text-sm mt-1">Running corneal light reflex pipeline</p>
      </div>

      {/* Pipeline steps */}
      <div className="flex items-center gap-1.5">
        {steps.map((step, i) => (
          <span key={step} className="flex items-center gap-1.5 text-xs text-slate-400">
            {i > 0 && <span className="text-slate-300">›</span>}
            {step}
          </span>
        ))}
      </div>
    </div>
  );
}

// ─── Inconclusive screen ───────────────────────────────────────────────────────
const REASON_ICON: Record<string, string> = {
  no_flash:                 "⚡",
  no_face:                  "👤",
  eyes_closed:              "👁",
  not_frontal:              "↩",
  no_reflex_left:           "●",
  no_reflex_right:          "●",
  no_reflex_both:           "●●",
  high_variance_asymmetry:  "📊",
  insufficient_frames:      "🎞",
};

const REASON_TIP: Record<string, string> = {
  high_variance_asymmetry: "The eye position shifted between frames. Ask the patient to stare steadily at the camera dot and try again.",
  no_flash:                "Enable the torch and ensure the bright dot is visible in both eyes.",
  no_face:                 "Move closer and ensure your face is fully in frame.",
  eyes_closed:             "Ask the patient to open their eyes wide and look straight ahead.",
  insufficient_frames:     "Too many frames were unusable — hold the phone steadier and keep eyes open.",
};

function InconclusiveScreen({
  result,
  onRetry,
}: {
  result:  InconclusiveResponse;
  onRetry: () => void;
}) {
  const icon = REASON_ICON[result.reason] ?? "⚠";
  const tip  = REASON_TIP[result.reason];
  const isVariance = result.reason === "high_variance_asymmetry";

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-slate-50 px-6 gap-6 text-center">
      <div className={`w-20 h-20 rounded-full border-2 flex items-center justify-center ${
        isVariance ? "bg-blue-100 border-blue-300" : "bg-amber-100 border-amber-300"
      }`}>
        <span className="text-3xl">{icon}</span>
      </div>

      <div className="max-w-sm">
        <h2 className="text-slate-900 text-xl font-bold mb-2">
          {isVariance ? "Unstable Readings" : "Screening Incomplete"}
        </h2>
        <p className="text-slate-600 text-sm leading-relaxed">{result.reason_human}</p>
      </div>

      {/* Variance-specific stats */}
      {"asymmetry_std_deg" in result && (
        <div className="bg-white border border-blue-100 rounded-xl px-4 py-3 max-w-sm w-full text-left">
          <p className="text-xs font-semibold text-slate-500 mb-2">Frame variance</p>
          <div className="flex justify-between text-sm">
            <span className="text-slate-600">Mean asymmetry</span>
            <span className="font-semibold text-slate-900">{(result as unknown as Record<string, number>)["asymmetry_avg_deg"]}°</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-600">Std deviation</span>
            <span className="font-semibold text-red-500">{(result as unknown as Record<string, number>)["asymmetry_std_deg"]}°</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-600">Frames accepted</span>
            <span className="font-semibold text-slate-900">{result.frames_accepted} / {result.frames_total}</span>
          </div>
        </div>
      )}

      {/* Specific tip */}
      {tip && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 max-w-sm w-full text-left">
          <p className="text-xs font-semibold text-amber-700 mb-1">How to fix this</p>
          <p className="text-xs text-amber-600 leading-relaxed">{tip}</p>
        </div>
      )}

      {result.flags.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-xl px-4 py-3 text-xs text-slate-500 font-mono shadow-sm max-w-sm w-full">
          {result.flags.join(" · ")}
        </div>
      )}

      <button
        onClick={onRetry}
        className="w-full max-w-sm rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold
                   py-3.5 transition-colors shadow-md shadow-blue-600/25 focus:outline-none
                   focus:ring-2 focus:ring-blue-400 focus:ring-offset-2"
      >
        Try Again
      </button>

      <p className="text-slate-400 text-xs max-w-sm leading-relaxed">
        Common fixes: enable your torch, ensure both eyes are open, face the camera
        directly, and hold the camera 30–40 cm away.
      </p>
    </div>
  );
}

// ─── Error screen ──────────────────────────────────────────────────────────────
function ErrorScreen({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-slate-50 px-6 gap-6 text-center">
      <div className="w-16 h-16 rounded-full bg-red-100 border-2 border-red-300 flex items-center justify-center">
        <svg
          className="w-8 h-8 text-red-500"
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

      <div className="max-w-sm">
        <h2 className="text-slate-900 text-xl font-bold mb-2">Something went wrong</h2>
        <p className="text-slate-600 text-sm leading-relaxed">{message}</p>
      </div>

      <button
        onClick={onRetry}
        className="w-full max-w-sm rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold
                   py-3.5 transition-colors shadow-md shadow-blue-600/25 focus:outline-none
                   focus:ring-2 focus:ring-blue-400 focus:ring-offset-2"
      >
        Try Again
      </button>
    </div>
  );
}

// ─── Main result page ──────────────────────────────────────────────────────────
export default function ResultPage() {
  const router         = useRouter();
  const analysisResult = useAppStore((s) => s.analysisResult);
  const isLoading      = useAppStore((s) => s.isLoading);
  const errorMessage   = useAppStore((s) => s.errorMessage);
  const reset          = useAppStore((s) => s.reset);

  // Redirect to home if nothing in store
  useEffect(() => {
    if (!isLoading && !analysisResult && !errorMessage) {
      router.replace("/");
    }
  }, [isLoading, analysisResult, errorMessage, router]);

  function handleRetry() {
    reset();
    router.push("/");
  }

  if (isLoading)      return <LoadingScreen />;
  if (errorMessage)   return <ErrorScreen message={errorMessage} onRetry={handleRetry} />;
  if (!analysisResult) return <LoadingScreen />;

  if (analysisResult.status === "INCONCLUSIVE") {
    return (
      <InconclusiveScreen
        result={analysisResult as InconclusiveResponse}
        onRetry={handleRetry}
      />
    );
  }

  if (analysisResult.status === "ERROR") {
    return (
      <ErrorScreen
        message={analysisResult.message || "An unexpected error occurred. Please retry."}
        onRetry={handleRetry}
      />
    );
  }

  return (
    <TriageReport
      result={analysisResult as SuccessResponse}
      onRetry={handleRetry}
    />
  );
}
