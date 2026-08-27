"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/useAppStore";
import StreamingCapture from "@/components/StreamingCapture";
import { TopBar } from "@/components/ui/Chrome";
import type { StreamSuccessResponse, StreamInconclusiveResponse } from "@/lib/types";

const PIPELINE = [
  { n: "1", title: "Eye detection",      desc: "MediaPipe Face Mesh locates both irises in real time, at ~30 fps." },
  { n: "2", title: "Pupil localisation", desc: "Landmark mean and Hough circle cross-validate the pupil centre." },
  { n: "3", title: "CLR detection",      desc: "The brightest 3% of pixels isolate the torch reflection on the cornea." },
  { n: "4", title: "Displacement",       desc: "Vector from pupil to reflex, normalised by iris radius." },
  { n: "5", title: "Hirschberg angle",   desc: "1 mm of displacement ≈ 7° of ocular deviation." },
  { n: "6", title: "Aggregation",        desc: "Frames averaged, IQR outliers dropped, variance scored." },
];

const TIERS = [
  { tier: "URGENT",  dot: "bg-red-500",     desc: "≥ 30° — refer within 1 week" },
  { tier: "ROUTINE", dot: "bg-orange-500",  desc: "15–30° — refer within 4 weeks" },
  { tier: "MONITOR", dot: "bg-amber-400",   desc: "5–15° — re-screen in 3 months" },
  { tier: "NORMAL",  dot: "bg-emerald-500", desc: "< 5° — no referral required" },
];

export default function CapturePage() {
  const router = useRouter();
  const patientName       = useAppStore((s) => s.patientName);
  const patientAge        = useAppStore((s) => s.patientAge);
  const sessionId         = useAppStore((s) => s.sessionId);
  const setAnalysisResult = useAppStore((s) => s.setAnalysisResult);
  const setLoading        = useAppStore((s) => s.setLoading);
  const setError          = useAppStore((s) => s.setError);

  useEffect(() => {
    if (!patientName || patientAge === null) {
      router.replace("/patient");
    }
  }, [patientName, patientAge, router]);

  function handleSuccess(result: StreamSuccessResponse) {
    setAnalysisResult({
      ...result,
      status:              "SUCCESS",
      patient:             result.patient,
      result:              result.result,
      technical:           result.technical,
      intermediate_images: result.intermediate_images,
      annotated_image_b64: result.annotated_image_b64 ?? "",
      timestamp:           result.timestamp,
    });
    setLoading(false);
    router.push("/result");
  }

  function handleInconclusive(result: StreamInconclusiveResponse) {
    setAnalysisResult({
      status:       "INCONCLUSIVE",
      reason:       result.reason,
      reason_human: result.reason_human,
      flags:        result.flags,
      patient:      result.patient,
      timestamp:    result.timestamp,
    });
    setLoading(false);
    router.push("/result");
  }

  function handleError(message: string) {
    setLoading(false);
    if (message === "TIMEOUT") {
      setError("The analysis timed out. Please check your connection and try again.");
    } else {
      setError(message || "Could not reach the server. Please check your connection and try again.");
    }
    router.push("/result");
  }

  if (!patientName || patientAge === null) return null;

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <TopBar step={2} />

      {/* ── Patient strip ────────────────────────────────────────────── */}
      <div className="border-b border-ink-100 bg-ink-50/60">
        <div className="max-w-5xl mx-auto px-5 sm:px-6 py-3.5 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <button
              onClick={() => router.push("/patient")}
              className="text-ink-400 hover:text-ink-900 transition-colors shrink-0 -ml-1 p-1"
              aria-label="Back to patient details"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5 8.25 12l7.5-7.5" />
              </svg>
            </button>
            <div className="min-w-0">
              <p className="text-[14px] font-semibold text-ink-900 truncate">{patientName}</p>
              <p className="text-[11.5px] text-ink-400">
                Age {patientAge}
                {sessionId && <span className="font-mono"> · {sessionId}</span>}
              </p>
            </div>
          </div>

          <p className="text-[11px] sm:text-[11.5px] text-ink-400 font-mono shrink-0 text-right leading-tight">
            8-FRAME<span className="hidden sm:inline"> BILATERAL</span>
            <br className="sm:hidden" /> ANALYSIS
          </p>
        </div>
      </div>

      {/* ── Body ─────────────────────────────────────────────────────── */}
      <div className="flex-1 px-4 sm:px-6 py-6 md:py-10">
        <div className="w-full max-w-5xl mx-auto flex flex-col lg:flex-row lg:gap-10 lg:items-start">

          {/* Camera — phone-width on desktop */}
          <div className="w-full lg:w-[420px] lg:shrink-0">
            <StreamingCapture
              patientName={patientName}
              patientAge={patientAge}
              onSuccess={handleSuccess}
              onInconclusive={handleInconclusive}
              onError={handleError}
            />
          </div>

          {/* Desktop reference rail */}
          <aside className="hidden lg:flex flex-col gap-5 flex-1 pt-1">

            <div className="surface p-6">
              <p className="eyebrow-muted mb-4">Pipeline</p>
              <ol className="space-y-3.5">
                {PIPELINE.map((s) => (
                  <li key={s.n} className="flex gap-3.5">
                    <span className="font-mono text-[11px] text-clinical-600 tabular pt-0.5 shrink-0">
                      0{s.n}
                    </span>
                    <div>
                      <p className="text-ink-800 text-[13px] font-semibold">{s.title}</p>
                      <p className="text-ink-500 text-[12.5px] leading-relaxed mt-0.5">{s.desc}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </div>

            <div className="surface p-6">
              <p className="eyebrow-muted mb-4">Triage tiers</p>
              <ul className="divide-y divide-ink-100">
                {TIERS.map((t) => (
                  <li key={t.tier} className="flex items-center gap-3 py-2.5 first:pt-0 last:pb-0">
                    <span className={`w-2 h-2 rounded-full shrink-0 ${t.dot}`} />
                    <span className="font-mono text-[11px] font-semibold text-ink-800 w-[62px] shrink-0">
                      {t.tier}
                    </span>
                    <span className="text-ink-500 text-[12.5px]">{t.desc}</span>
                  </li>
                ))}
              </ul>
            </div>

            <p className="text-ink-400 text-[12px] leading-relaxed px-1">
              Screening aid only · Not a diagnostic device · Hirschberg corneal light reflex
              method · Always confirm with a qualified ophthalmologist.
            </p>
          </aside>
        </div>
      </div>
    </div>
  );
}
