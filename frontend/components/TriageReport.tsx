"use client";

import type { SuccessResponse, StreamSuccessResponse } from "@/lib/types";
import { URGENCY_CONFIG } from "@/lib/types";
import UrgencyBadge from "./UrgencyBadge";
import AnnotatedEye from "./AnnotatedEye";
import ProcessingSteps from "./ProcessingSteps";

// TriageReport accepts either a single-frame or multi-frame result
type AnySuccessResult = SuccessResponse & Partial<Pick<StreamSuccessResponse,
  | "frames_total" | "frames_accepted" | "frames_rejected"
  | "per_frame_readings" | "deviation_avg_deg" | "deviation_std_deg"
  | "deviation_min_deg" | "deviation_max_deg" | "aggregate_confidence"
>>;

interface TriageReportProps {
  result:  AnySuccessResult;
  onRetry: () => void;
}

const CONF_COLOUR: Record<string, string> = {
  HIGH:   "bg-emerald-100 text-emerald-700 border-emerald-200",
  MEDIUM: "bg-amber-100   text-amber-700   border-amber-200",
  LOW:    "bg-red-100     text-red-700     border-red-200",
};

const URGENCY_PRINT_COLOUR: Record<string, string> = {
  URGENT:  "text-red-700",
  ROUTINE: "text-orange-700",
  MONITOR: "text-yellow-700",
  NORMAL:  "text-green-700",
};

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center py-2.5 border-b border-slate-100 last:border-0">
      <span className="text-slate-500 text-sm">{label}</span>
      <span className="text-slate-900 text-sm font-semibold text-right max-w-[55%]">
        {value}
      </span>
    </div>
  );
}

export default function TriageReport({ result, onRetry }: TriageReportProps) {
  const { patient, result: r, technical, annotated_image_b64 } = result;
  const config          = URGENCY_CONFIG[r.urgency_tier];
  const urgencyPrintCol = URGENCY_PRINT_COLOUR[r.urgency_tier] ?? "text-slate-900";
  const printDate       = new Date().toLocaleDateString("en-GB", {
    day: "2-digit", month: "long", year: "numeric",
  });
  const printTime = new Date().toLocaleTimeString("en-GB", {
    hour: "2-digit", minute: "2-digit",
  });

  return (
    <div className="min-h-screen bg-slate-50 pb-12 print:bg-white print:pb-2">

      {/* ── Print-only header ─────────────────────────────────────────────── */}
      <div className="hidden print:block px-5 pt-4 pb-4 border-b-2 border-slate-900 mb-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-[8pt] font-black tracking-[0.25em] text-slate-400 uppercase mb-0.5">BeanHealth</p>
            <h1 className="text-[18pt] font-bold text-slate-900 leading-tight">CLR Triage Report</h1>
            <p className="text-[8.5pt] text-slate-500 mt-0.5">Corneal Light Reflex Asymmetry Analysis · Hirschberg Method</p>
          </div>
          <div className="text-right text-[8.5pt] text-slate-500 leading-relaxed">
            <p className="font-semibold text-slate-700">{printDate}</p>
            <p>{printTime}</p>
            <p className="mt-1 text-slate-400">EyeQ Innovate Hackathon 2.0</p>
          </div>
        </div>
        <div className="mt-3 pt-2 border-t border-slate-200 flex flex-wrap gap-x-8 gap-y-0.5 text-[9pt]">
          <span><span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">Patient</span><span className="font-semibold text-slate-800">{patient.name}</span></span>
          <span><span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">Age</span><span className="font-semibold text-slate-800">{patient.age}</span></span>
          <span><span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">Urgency</span><span className={`font-bold ${urgencyPrintCol}`}>{r.urgency_tier}</span></span>
          <span><span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">Condition</span><span className="font-semibold text-slate-800">{r.condition_name}</span></span>
          <span><span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">ICD-10</span><span className="font-semibold text-slate-800">{r.icd10_code}</span></span>
          <span><span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">Asymmetry</span><span className="font-semibold text-slate-800">{r.asymmetry_degrees !== undefined ? `${r.asymmetry_degrees}°` : `${r.deviation_degrees}°`}</span></span>
        </div>
      </div>

      {/* ── Screen header ─────────────────────────────────────────────────── */}
      <header className="bg-white border-b border-slate-200 print:hidden">
        <div className="max-w-5xl mx-auto px-5 py-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
            </svg>
          </div>
          <div>
            <p className="text-sm font-bold text-slate-900 leading-none">BeanHealth</p>
            <p className="text-xs text-slate-400 mt-0.5">CLR Screening Tool</p>
          </div>
        </div>
      </header>

      {/* ── Urgency banner ─────────────────────────────────────────────────── */}
      <div className={`${config.bgColour} border-b ${config.borderColour} px-5 py-5 print:hidden`}>
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div>
            <p className="text-slate-500 text-xs mb-1">{patient.name} · Age {patient.age}</p>
            <h1 className={`text-2xl font-bold ${config.colour}`}>{r.condition_name}</h1>
            <p className="text-slate-500 text-sm mt-0.5">{r.icd10_code}</p>
          </div>
          <UrgencyBadge urgency={r.urgency_tier} large />
        </div>
      </div>

      {/* ── Main content ──────────────────────────────────────────────────── */}
      <div className="max-w-5xl mx-auto px-5 pt-6 print:max-w-none print:pt-0 print:px-0">

        {/* Desktop: 2-column grid (left = data, right = visuals)
            Mobile:  single column, visuals come after key clinical info  */}
        <div className="lg:grid lg:grid-cols-5 lg:gap-6 lg:items-start">

          {/* ── LEFT COLUMN: clinical data ──────────────────────────────── */}
          <div className="lg:col-span-2 space-y-4">

            {/* Referral */}
            <div className={`rounded-2xl border ${config.borderColour} ${config.bgColour} p-5 print:bg-slate-50 print:border-slate-300`}>
              <h2 className={`font-semibold mb-1 ${config.colour} print:${urgencyPrintCol}`}>
                Referral Recommendation
              </h2>
              <p className={`text-lg font-bold ${config.colour} print:${urgencyPrintCol}`}>
                {r.referral_recommendation}
              </p>
              <p className="text-slate-500 text-sm mt-1 print:text-slate-600">
                Timeframe: {r.timeframe}
              </p>
            </div>

            {/* Narrative */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
              <h2 className="text-slate-900 font-semibold mb-2">What This Means</h2>
              <p className="text-slate-600 text-sm leading-relaxed">{r.narrative}</p>
            </div>

            {/* Streaming stats — only for multi-frame results */}
            {result.frames_total !== undefined && result.frames_total > 1 && (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 print:hidden">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-slate-900 font-semibold">15-Frame Analysis</h2>
                  {result.aggregate_confidence && (
                    <span className={`text-xs px-2.5 py-1 rounded-full border font-semibold ${CONF_COLOUR[result.aggregate_confidence] ?? ""}`}>
                      {result.aggregate_confidence} confidence
                    </span>
                  )}
                </div>

                <div className="flex items-baseline gap-1 mb-1">
                  <span className="text-3xl font-bold text-slate-900">
                    {result.deviation_avg_deg?.toFixed(1)}°
                  </span>
                  {result.deviation_std_deg !== undefined && (
                    <span className="text-slate-400 text-sm font-medium">
                      ± {result.deviation_std_deg.toFixed(2)}° std dev
                    </span>
                  )}
                </div>
                {result.deviation_min_deg !== undefined && result.deviation_max_deg !== undefined && (
                  <p className="text-slate-400 text-xs mb-4">
                    Range: {result.deviation_min_deg.toFixed(1)}° – {result.deviation_max_deg.toFixed(1)}°
                  </p>
                )}

                {result.per_frame_readings && (
                  <div>
                    <p className="text-slate-500 text-xs mb-2 font-medium">
                      Per-frame readings ({result.frames_accepted} accepted, {result.frames_rejected} rejected)
                    </p>
                    <div className="flex gap-1.5 flex-wrap">
                      {result.per_frame_readings.map((v, i) => (
                        <div
                          key={i}
                          className={`flex flex-col items-center justify-center w-9 h-10 rounded-lg border text-xs font-semibold ${
                            v !== null
                              ? "bg-emerald-50 border-emerald-200 text-emerald-700"
                              : "bg-slate-100 border-slate-200 text-slate-400"
                          }`}
                        >
                          <span className="text-[9px] text-slate-400 font-normal">f{i + 1}</span>
                          <span>{v !== null ? `${v}°` : "✕"}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Clinical measurements */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
              <h2 className="text-slate-900 font-semibold mb-1">Clinical Measurements</h2>
              <div className="mt-3">
                <MetricRow
                  label="Inter-ocular asymmetry"
                  value={
                    r.asymmetry_degrees !== undefined
                      ? `${r.asymmetry_degrees.toFixed(1)}° (classification signal)`
                      : `${r.asymmetry_score.toFixed(3)} normalised`
                  }
                />
                <MetricRow
                  label="Absolute deviation (ref)"
                  value={
                    result.deviation_avg_deg !== undefined
                      ? `${result.deviation_avg_deg.toFixed(1)}° ± ${result.deviation_std_deg?.toFixed(2)}°`
                      : `${r.deviation_degrees}°`
                  }
                />
                <MetricRow label="Displacement (mm)"    value={`${technical.deviation_mm.toFixed(2)} mm`} />
                <MetricRow label="Severity"             value={r.severity} />
                <MetricRow label="Dominant eye"         value={technical.dominant_eye} />
                <MetricRow label="Detection confidence" value={result.aggregate_confidence ?? technical.confidence} />
                {result.frames_total !== undefined && (
                  <MetricRow
                    label="Frames used"
                    value={`${result.frames_accepted} / ${result.frames_total}`}
                  />
                )}
              </div>
            </div>

            {/* Displacement detail */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
              <h2 className="text-slate-900 font-semibold mb-3">Displacement Detail</h2>
              <div className="grid grid-cols-2 gap-4">
                {(["left", "right"] as const).map((eye) => {
                  const norm = eye === "left"
                    ? technical.left_displacement_norm
                    : technical.right_displacement_norm;
                  const dir = eye === "left"
                    ? technical.left_direction
                    : technical.right_direction;
                  const pct = Math.min(norm * 100, 100);
                  return (
                    <div key={eye} className="bg-slate-50 rounded-xl p-4 border border-slate-100 print:bg-slate-100 print:border-slate-200">
                      <p className="text-slate-500 text-xs uppercase tracking-wider mb-1">
                        {eye === "left" ? "Left Eye" : "Right Eye"}
                      </p>
                      <p className="text-slate-900 font-bold text-lg">{(norm * 100).toFixed(1)}%</p>
                      <div className="mt-2 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${
                            norm > 0.5 ? "bg-red-500" : norm > 0.2 ? "bg-yellow-500" : "bg-green-500"
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <p className="text-slate-500 text-xs mt-1.5 capitalize">{dir}</p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Quality flags */}
            {technical.flags.length > 0 && (
              <div className="bg-amber-50 rounded-2xl border border-amber-200 p-5">
                <h2 className="text-amber-800 font-semibold mb-2">⚠ Quality Flags</h2>
                <ul className="space-y-1">
                  {technical.flags.map((flag) => (
                    <li key={flag} className="text-amber-700 text-sm font-mono">
                      · {flag.replace(/_/g, " ")}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Disclaimer */}
            <p className="text-slate-400 text-xs text-center leading-relaxed px-2 print:hidden">
              This tool is a screening aid only. It does not replace a full clinical
              examination by a qualified ophthalmologist. Always follow professional medical advice.
            </p>

            {/* Action buttons */}
            <div className="flex gap-3 print:hidden">
              <button
                onClick={onRetry}
                className="flex-1 rounded-xl bg-slate-200 hover:bg-slate-300 text-slate-800
                           font-medium py-3.5 text-sm transition-colors"
              >
                New Screening
              </button>
              <button
                onClick={() => window.print()}
                className="flex-1 rounded-xl bg-blue-600 hover:bg-blue-700 text-white
                           font-medium py-3.5 text-sm transition-colors shadow-sm shadow-blue-600/25"
              >
                Print / Save PDF
              </button>
            </div>
          </div>

          {/* ── RIGHT COLUMN: visuals ──────────────────────────────────────── */}
          <div className="lg:col-span-3 space-y-4 mt-4 lg:mt-0">

            {/* Annotated scan */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5">
              <h2 className="text-slate-900 font-semibold mb-3">Annotated Scan</h2>
              <AnnotatedEye base64Jpeg={annotated_image_b64} patientName={patient.name} />
            </div>

            {/* Processing steps */}
            {result.intermediate_images && (
              <div className="print:hidden">
                <ProcessingSteps intermediateImages={result.intermediate_images} />
              </div>
            )}
          </div>

        </div>

        {/* ── Print-only footer ──────────────────────────────────────────────── */}
        <div className="hidden print:block pt-3 mt-4 border-t border-slate-300 text-[8pt] text-slate-500 leading-relaxed">
          <p>
            <strong className="text-slate-700">Screening Tool Disclaimer:</strong>{" "}
            This report is generated by BeanHealth CLR Tool v1.0.0, an AI-assisted
            strabismus screening aid. It does not constitute a medical diagnosis and must
            not replace examination by a qualified ophthalmologist. Results are based on
            the Hirschberg corneal light reflex test (7°/mm, iris radius 5.75 mm).
          </p>
          <p className="mt-1 text-slate-400">
            Generated: {new Date().toISOString()} · Status: {result.status} ·
            Confidence: {technical.confidence} · BeanHealth · EyeQ Innovate Hackathon 2.0
          </p>
        </div>

      </div>
    </div>
  );
}
