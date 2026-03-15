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

  // Redirect back to form if no patient data
  useEffect(() => {
    if (!patientName || patientAge === null) {
      router.replace("/");
    }
  }, [patientName, patientAge, router]);

  function handleSuccess(result: StreamSuccessResponse) {
    // Spread stream result first, then override the AnalyseResponse required fields
    // so there are no duplicate key conflicts
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
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 bg-white border-b border-slate-100">
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

      {/* Body */}
      <div className="flex-1 px-4 py-6">
        <StreamingCapture
          patientName={patientName}
          patientAge={patientAge}
          onSuccess={handleSuccess}
          onInconclusive={handleInconclusive}
          onError={handleError}
        />
      </div>
    </div>
  );
}
