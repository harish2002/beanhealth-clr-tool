"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/useAppStore";
import StreamingCapture from "@/components/StreamingCapture";
import type { StreamSuccessResponse, StreamInconclusiveResponse } from "@/lib/types";

export default function CapturePage() {
  const router = useRouter();
  const patientName       = useAppStore((s) => s.patientName);
  const patientAge        = useAppStore((s) => s.patientAge);
  const setAnalysisResult = useAppStore((s) => s.setAnalysisResult);
  const setLoading        = useAppStore((s) => s.setLoading);
  const setError          = useAppStore((s) => s.setError);

  useEffect(() => {
    if (!patientName || patientAge === null) {
      router.replace("/");
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
    <div className="min-h-screen bg-slate-50 flex flex-col">

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="bg-white border-b border-slate-100">
        <div className="max-w-5xl mx-auto flex items-center justify-between px-5 py-4">
          <button
            onClick={() => router.push("/")}
            className="text-slate-400 hover:text-slate-700 transition-colors"
            aria-label="Go back"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5 8.25 12l7.5-7.5" />
            </svg>
          </button>

          <div className="text-center">
            <p className="text-slate-800 text-sm font-semibold">{patientName}</p>
            <p className="text-slate-400 text-xs">Age {patientAge} · 15-Frame Analysis</p>
          </div>

          <div className="w-6" />
        </div>
      </div>

      {/* ── Body ──────────────────────────────────────────────────────────── */}
      <div className="flex-1 flex items-start justify-center px-4 py-6">
        <div className="w-full max-w-5xl flex flex-col lg:flex-row lg:gap-8 lg:items-start">

          {/* Camera capture — constrained width on desktop so it feels like a phone */}
          <div className="w-full lg:w-[420px] lg:shrink-0">
            <StreamingCapture
              patientName={patientName}
              patientAge={patientAge}
              onSuccess={handleSuccess}
              onInconclusive={handleInconclusive}
              onError={handleError}
            />
          </div>

          {/* Desktop sidebar — pipeline + triage reference */}
          <div className="hidden lg:flex flex-col gap-4 flex-1 pt-1">

            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
              <h3 className="text-slate-900 font-semibold text-sm mb-3">How the pipeline works</h3>
              <ol className="space-y-3">
                {[
                  { n: "1", title: "Eye Detection",      desc: "MediaPipe Face Mesh locates both irises in real time (30 fps)" },
                  { n: "2", title: "Pupil Localisation", desc: "Two independent methods cross-validate the pupil centre" },
                  { n: "3", title: "CLR Detection",      desc: "Top 3% brightest pixels isolate the torch reflection on the cornea" },
                  { n: "4", title: "Displacement",       desc: "Vector measured from pupil → corneal reflex, normalised by iris radius" },
                  { n: "5", title: "Hirschberg Angle",   desc: "1 mm displacement ≈ 7° ocular deviation (clinical standard)" },
                  { n: "6", title: "Aggregation",        desc: "15 frames averaged · IQR outliers removed · std dev scored" },
                ].map((s) => (
                  <li key={s.n} className="flex gap-3">
                    <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-700 text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                      {s.n}
                    </span>
                    <div>
                      <p className="text-slate-800 text-xs font-semibold">{s.title}</p>
                      <p className="text-slate-500 text-xs leading-relaxed">{s.desc}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </div>

            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
              <h3 className="text-slate-900 font-semibold text-sm mb-3">Triage tiers</h3>
              <div className="space-y-2">
                {[
                  { tier: "URGENT",  colour: "bg-red-100 text-red-700",       desc: "≥ 30° — refer within 1 week" },
                  { tier: "ROUTINE", colour: "bg-orange-100 text-orange-700", desc: "15–30° — refer within 4 weeks" },
                  { tier: "MONITOR", colour: "bg-amber-100 text-amber-700",   desc: "5–15° — re-screen in 3 months" },
                  { tier: "NORMAL",  colour: "bg-green-100 text-green-700",   desc: "< 5° — no referral required" },
                ].map((t) => (
                  <div key={t.tier} className="flex items-center gap-3">
                    <span className={`text-xs font-bold px-2 py-0.5 rounded-md min-w-[64px] text-center ${t.colour}`}>
                      {t.tier}
                    </span>
                    <span className="text-slate-500 text-xs">{t.desc}</span>
                  </div>
                ))}
              </div>
            </div>

            <p className="text-slate-400 text-xs leading-relaxed px-1">
              Screening aid only · Not a diagnostic device · Results based on the Hirschberg corneal
              light reflex method · Always confirm with a qualified ophthalmologist.
            </p>
          </div>

        </div>
      </div>
    </div>
  );
}
