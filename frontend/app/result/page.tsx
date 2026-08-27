"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/useAppStore";
import TriageReport, { AlignmentSection } from "@/components/TriageReport";
import type { PatientMeta } from "@/components/TriageReport";
import { TopBar, EyeGlyph } from "@/components/ui/Chrome";
import type { StreamSuccessResponse, StreamInconclusiveResponse } from "@/lib/types";

// ─── Loading screen ────────────────────────────────────────────────────────────

function LoadingScreen() {
  const steps = ["Detecting eyes", "Locating pupils", "Finding reflex", "Measuring"];
  return (
    <div className="min-h-screen bg-white flex flex-col">
      <TopBar step={3} />
      <div className="flex-1 flex flex-col items-center justify-center gap-7 px-6 text-center">

        <div className="relative w-20 h-20">
          <div className="absolute inset-0 rounded-full border-[3px] border-ink-100" />
          <div className="absolute inset-0 rounded-full border-[3px] border-transparent border-t-clinical-600 spin-slow" />
          <div className="absolute inset-0 flex items-center justify-center">
            <EyeGlyph className="w-7 h-7 text-clinical-600" />
          </div>
        </div>

        <div>
          <h1 className="display-md">Analysing</h1>
          <p className="text-[13.5px] text-ink-500 mt-2">
            Running the corneal light reflex pipeline
          </p>
        </div>

        <ol className="flex flex-wrap items-center justify-center gap-x-2.5 gap-y-1.5 max-w-sm">
          {steps.map((step, i) => (
            <li key={step} className="flex items-center gap-2.5 font-mono text-[11px] text-ink-400">
              {i > 0 && <span className="text-ink-200">→</span>}
              {step}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

// ─── Inconclusive screen ───────────────────────────────────────────────────────

const REASON_TITLE: Record<string, string> = {
  high_variance_asymmetry: "Readings were unstable",
  no_flash:                "No torch reflex found",
  no_face:                 "No face detected",
  eyes_closed:             "Eyes were not open",
  not_frontal:             "Head was turned",
  insufficient_frames:     "Not enough usable frames",
};

const REASON_TIP: Record<string, string> = {
  high_variance_asymmetry: "The eye position shifted between frames. Ask the patient to stare steadily at the target and try again.",
  no_flash:                "Enable the torch and check that a bright dot is visible in both eyes before capturing.",
  no_face:                 "Move closer and make sure the whole face is inside the frame.",
  eyes_closed:             "Ask the patient to open their eyes wide and look straight ahead.",
  not_frontal:             "The patient must face the camera squarely — no tilt, no turn.",
  insufficient_frames:     "Too many frames were unusable. Hold the phone steadier and keep both eyes open.",
};

function InconclusiveScreen({
  result,
  onRetry,
}: {
  result:  StreamInconclusiveResponse;
  onRetry: () => void;
}) {
  const tip        = REASON_TIP[result.reason];
  const title      = REASON_TITLE[result.reason] ?? "Screening incomplete";
  const isVariance = result.reason === "high_variance_asymmetry";

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <TopBar step={3} />

      <div className="flex-1 px-5 sm:px-6 py-10 md:py-14">
        <div className="max-w-lg mx-auto space-y-5 animate-rise-in">

          {/* Verdict header */}
          <div className="text-center">
            <span
              className={`inline-flex items-center justify-center w-14 h-14 rounded-full mb-5
                          ${isVariance ? "bg-clinical-50 text-clinical-600" : "bg-amber-50 text-amber-600"}`}
            >
              <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" strokeWidth={1.6} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z" />
              </svg>
            </span>
            <p className="eyebrow-muted mb-2.5">Inconclusive</p>
            <h1 className="display-md">{title}</h1>
            <p className="text-[14px] text-ink-500 leading-relaxed mt-3">
              {result.reason_human}
            </p>
          </div>

          {/* Torch-free corner-alignment result — still valid without a CLR. */}
          {result.alignment && <AlignmentSection alignment={result.alignment} />}

          {/* Per-frame reading strip */}
          {result.per_frame_readings && result.per_frame_readings.length > 0 && (
            <div className="surface p-5">
              <div className="flex items-baseline justify-between mb-3">
                <p className="eyebrow-muted">Frame readings</p>
                <p className="font-mono text-[11px] text-ink-400 tabular">
                  {result.frames_accepted}/{result.frames_total} accepted
                </p>
              </div>
              <div className="flex gap-1.5 flex-wrap">
                {result.per_frame_readings.map((v, i) => (
                  <div
                    key={i}
                    className={`flex flex-col items-center justify-center w-10 h-11 rounded-lg border
                                font-mono text-[11.5px] font-semibold tabular ${
                      v !== null
                        ? "bg-emerald-50 border-emerald-200 text-emerald-700"
                        : "bg-ink-50 border-ink-200 text-ink-300"
                    }`}
                  >
                    <span className="text-[9px] text-ink-400 font-normal">f{i + 1}</span>
                    <span>{v !== null ? `${v}°` : "✕"}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Variance stats */}
          {result.asymmetry_std_deg !== undefined && (
            <div className="surface p-5">
              <p className="eyebrow-muted mb-3">Frame variance</p>
              <dl className="divide-y divide-ink-100">
                {[
                  {
                    k: "Mean asymmetry",
                    v: result.asymmetry_avg_deg !== undefined ? `${result.asymmetry_avg_deg.toFixed(1)}°` : "–",
                    accent: "text-ink-900",
                  },
                  { k: "Standard deviation", v: `${result.asymmetry_std_deg.toFixed(1)}°`, accent: "text-red-600" },
                  {
                    k: "Frames accepted",
                    v: `${result.frames_accepted} / ${result.frames_total}`,
                    accent: "text-ink-900",
                  },
                ].map(({ k, v, accent }) => (
                  <div key={k} className="flex justify-between items-baseline py-2.5 first:pt-0 last:pb-0">
                    <dt className="text-[13.5px] text-ink-500">{k}</dt>
                    <dd className={`font-mono text-[13px] font-semibold tabular ${accent}`}>{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}

          {/* Fix tip */}
          {tip && (
            <div className="rounded-card border border-amber-200 bg-amber-50/70 p-5">
              <p className="eyebrow-muted !text-amber-700 mb-2">How to fix this</p>
              <p className="text-[13.5px] text-amber-800 leading-relaxed">{tip}</p>
            </div>
          )}

          {/* Raw flags */}
          {result.flags.length > 0 && (
            <div className="surface-subtle px-4 py-3">
              <p className="font-mono text-[11px] text-ink-500 break-words">
                {result.flags.join("  ·  ")}
              </p>
            </div>
          )}

          <button onClick={onRetry} className="btn-primary w-full">
            Screen again
          </button>

          <p className="text-ink-400 text-[12px] leading-relaxed text-center px-2">
            Common fixes: enable the torch, dim the room, keep both eyes open, face the
            camera squarely, and hold the phone 30–40 cm away.
          </p>
        </div>
      </div>
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
    <div className="min-h-screen bg-white flex flex-col">
      <TopBar />
      <div className="flex-1 flex flex-col items-center justify-center px-6 gap-6 text-center">
        <span className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-red-50 text-red-600">
          <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" strokeWidth={1.6} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
          </svg>
        </span>

        <div className="max-w-sm">
          <p className="eyebrow-muted mb-2.5">Error</p>
          <h1 className="display-md">Something went wrong</h1>
          <p className="text-[14px] text-ink-500 leading-relaxed mt-3">{message}</p>
        </div>

        <button onClick={onRetry} className="btn-primary w-full max-w-sm">
          Try again
        </button>
      </div>
    </div>
  );
}

// ─── Main result page ──────────────────────────────────────────────────────────

export default function ResultPage() {
  const router            = useRouter();
  const analysisResult    = useAppStore((s) => s.analysisResult);
  const isLoading         = useAppStore((s) => s.isLoading);
  const errorMessage      = useAppStore((s) => s.errorMessage);
  const reset             = useAppStore((s) => s.reset);
  const patientGender     = useAppStore((s) => s.patientGender);
  const screenerRole      = useAppStore((s) => s.screenerRole);
  const screeningLocation = useAppStore((s) => s.screeningLocation);
  const sessionId         = useAppStore((s) => s.sessionId);

  // Redirect to intake if nothing in store
  useEffect(() => {
    if (!isLoading && !analysisResult && !errorMessage) {
      router.replace("/patient");
    }
  }, [isLoading, analysisResult, errorMessage, router]);

  function handleRetry() {
    reset();
    router.push("/patient");
  }

  if (isLoading)       return <LoadingScreen />;
  if (errorMessage)    return <ErrorScreen message={errorMessage} onRetry={handleRetry} />;
  if (!analysisResult) return <LoadingScreen />;

  if (analysisResult.status === "INCONCLUSIVE") {
    return (
      <InconclusiveScreen
        result={analysisResult as StreamInconclusiveResponse}
        onRetry={handleRetry}
      />
    );
  }

  if (analysisResult.status === "ERROR") {
    return (
      <ErrorScreen
        message={"message" in analysisResult
          ? (analysisResult as { message: string }).message
          : "An unexpected error occurred. Please retry."}
        onRetry={handleRetry}
      />
    );
  }

  const patientMeta: PatientMeta = {
    gender:            patientGender     || undefined,
    screenerRole:      screenerRole      || undefined,
    screeningLocation: screeningLocation || undefined,
    sessionId:         sessionId         || undefined,
  };

  return (
    <TriageReport
      result={analysisResult as StreamSuccessResponse}
      patientMeta={patientMeta}
      onRetry={handleRetry}
    />
  );
}
