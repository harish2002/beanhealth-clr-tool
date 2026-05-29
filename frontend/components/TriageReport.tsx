"use client";

import { useState } from "react";
import type { StreamSuccessResponse, UrgencyTier } from "@/lib/types";
import { URGENCY_CONFIG } from "@/lib/types";
import AnnotatedEye from "./AnnotatedEye";
import ProcessingSteps from "./ProcessingSteps";

// ─── Prop types ───────────────────────────────────────────────────────────────

export interface PatientMeta {
  gender?:            string;
  screenerRole?:      string;
  screeningLocation?: string;
  sessionId?:         string;
}

interface TriageReportProps {
  result:       StreamSuccessResponse;
  patientMeta?: PatientMeta;
  onRetry:      () => void;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const TIER_ORDER: UrgencyTier[] = ["NORMAL", "MONITOR", "ROUTINE", "URGENT"];

const TIER_EMOJI: Record<UrgencyTier, string> = {
  NORMAL:  "🟢",
  MONITOR: "🟡",
  ROUTINE: "🟠",
  URGENT:  "🔴",
};

/** Human-readable labels for per-frame rejection reasons */
const REJECTION_LABEL: Record<string, string> = {
  statistical_outlier:    "Outlier",
  ambiguous_reflex:       "Ambig. CLR",
  no_flash:               "No flash",
  pipeline_failed:        "Failed",
  invalid_deviation:      "Invalid",
  deviation_out_of_range: "Range err",
  high_variance_asymmetry:"High var",
  insufficient_frames:    "Insuff.",
};

const CONF_COLOUR: Record<string, string> = {
  HIGH:   "bg-emerald-100 text-emerald-700 border-emerald-200",
  MEDIUM: "bg-amber-100   text-amber-700   border-amber-200",
  LOW:    "bg-red-100     text-red-700     border-red-200",
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Convert degrees to prism dioptres: 1mm = 7° = 22 PD → 1° = 22/7 PD */
function toPD(degrees: number): number {
  return Math.round(degrees * 22 / 7);
}

function rejLabel(reason: string | null | undefined): string {
  if (!reason) return "rejected";
  return REJECTION_LABEL[reason] ?? reason.replace(/_/g, " ");
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-baseline py-2 border-b border-slate-100 last:border-0 gap-4">
      <span className="text-slate-500 text-sm shrink-0">{label}</span>
      <span className="text-slate-900 text-sm font-semibold text-right">{value}</span>
    </div>
  );
}

function SectionCard({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <section className={`bg-white rounded-2xl border border-slate-200 shadow-sm p-5 ${className}`}>
      {children}
    </section>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-slate-900 font-semibold text-sm uppercase tracking-wider mb-4">
      {children}
    </h2>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function TriageReport({ result, patientMeta, onRetry }: TriageReportProps) {
  const [showTechnical, setShowTechnical] = useState(false);
  const { patient, result: r, technical } = result;

  const aggConf = result.aggregate_confidence ?? technical.confidence;

  // §1 confidence-tier locking:
  // LOW confidence on a non-NORMAL finding → render tier as null (uncertain)
  const displayTier: UrgencyTier | null =
    aggConf === "LOW" && r.urgency_tier !== "NORMAL" ? null : r.urgency_tier;

  const config = displayTier ? URGENCY_CONFIG[displayTier] : URGENCY_CONFIG.NORMAL;

  // Clinical angle values
  const asymDeg = r.asymmetry_degrees ?? r.deviation_degrees;
  const asymPD  = toPD(asymDeg);

  // Dates
  const printDate = new Date().toLocaleDateString("en-GB", {
    day: "2-digit", month: "long", year: "numeric",
  });
  const printTime = new Date().toLocaleTimeString("en-GB", {
    hour: "2-digit", minute: "2-digit",
  });

  const sessionId   = patientMeta?.sessionId ?? "–";
  const screenerCtx = [patientMeta?.screenerRole, patientMeta?.screeningLocation]
    .filter(Boolean).join(" · ") || "–";

  // Per-eye PD estimates (displacement_norm × iris_radius_mm × hirschberg_deg/mm → degrees → PD)
  const leftPD  = toPD(technical.left_displacement_norm  * 5.75 * 7.0);
  const rightPD = toPD(technical.right_displacement_norm * 5.75 * 7.0);

  // ── Signal-to-noise ratio (the honest measurement quality indicator) ──
  //  SNR < 2 means the measurement noise is comparable to the signal — the
  //  reading is dominated by capture conditions rather than clinical findings.
  const snr =
    result.asymmetry_avg_deg !== undefined && result.asymmetry_std_deg !== undefined
      ? result.asymmetry_avg_deg / Math.max(result.asymmetry_std_deg, 0.5)
      : null;
  const lowSNR     = technical.flags.includes("low_snr_noise_dominated");
  const testMode   = technical.flags.includes("test_capture");

  return (
    <div className="min-h-screen bg-slate-50 pb-16 print:bg-white print:pb-0">

      {/* ══════════════════════════════════════════════════════════════════
          PRINT-ONLY HEADER
      ══════════════════════════════════════════════════════════════════ */}
      <div className="hidden print:block px-6 pt-5 pb-4 border-b-2 border-slate-800 mb-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-[7.5pt] font-black tracking-[0.25em] text-slate-400 uppercase mb-0.5">BeanHealth</p>
            <h1 className="text-[18pt] font-bold text-slate-900 leading-tight">CLR Triage Report</h1>
            <p className="text-[8.5pt] text-slate-500 mt-0.5">Corneal Light Reflex Asymmetry · Hirschberg Method · 10-Frame Analysis</p>
          </div>
          <div className="text-right text-[8.5pt] text-slate-500 leading-relaxed">
            <p className="font-semibold text-slate-700">{printDate}</p>
            <p>{printTime}</p>
            <p className="mt-1 text-slate-400">Session: {sessionId}</p>
          </div>
        </div>
        <div className="mt-3 pt-2 border-t border-slate-200 flex flex-wrap gap-x-8 gap-y-0.5 text-[9pt]">
          <span>
            <span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">Patient</span>
            <span className="font-semibold text-slate-800">{patient.name}</span>
          </span>
          <span>
            <span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">Age</span>
            <span className="font-semibold text-slate-800">{patient.age}</span>
          </span>
          {patientMeta?.gender && (
            <span>
              <span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">Gender</span>
              <span className="font-semibold text-slate-800">{patientMeta.gender}</span>
            </span>
          )}
          <span>
            <span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">Tier</span>
            <span className="font-bold text-slate-800">{r.urgency_tier}</span>
          </span>
          <span>
            <span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">Condition</span>
            <span className="font-semibold text-slate-800">{r.condition_name}</span>
          </span>
          <span>
            <span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">ICD-10</span>
            <span className="font-semibold text-slate-800">{r.icd10_code}</span>
          </span>
          <span>
            <span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">Asymmetry</span>
            <span className="font-semibold text-slate-800">{asymDeg.toFixed(1)}° ({asymPD} PD)</span>
          </span>
          <span>
            <span className="text-slate-400 uppercase tracking-wide text-[7.5pt] mr-1">Screener</span>
            <span className="font-semibold text-slate-800">{screenerCtx}</span>
          </span>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          §1  HEADER BAND (screen only)
      ══════════════════════════════════════════════════════════════════ */}

      {/* App bar */}
      <header className="bg-white border-b border-slate-100 print:hidden">
        <div className="max-w-2xl mx-auto px-5 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
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
              <p className="text-xs text-slate-400 mt-0.5">CLR Triage Report</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-xs text-slate-400 font-mono">{sessionId}</p>
            <p className="text-xs text-slate-400">{printDate}</p>
          </div>
        </div>
      </header>

      {/* TEST CAPTURE banner — shown when /analyse-test produced this report */}
      {testMode && (
        <div className="bg-amber-100 border-b-2 border-amber-400 print:bg-amber-50 print:border-amber-500">
          <div className="max-w-2xl mx-auto px-5 py-2.5 flex items-center gap-2.5">
            <svg className="w-4 h-4 text-amber-700 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
            </svg>
            <p className="text-amber-800 text-xs font-bold tracking-wider uppercase">
              Test capture
            </p>
            <p className="text-amber-700 text-[11px] leading-tight">
              Single-image upload · flash guard relaxed · NOT a clinical screening
            </p>
          </div>
        </div>
      )}

      {/* Tier band */}
      <div className={`${config.bgColour} border-b ${config.borderColour} print:hidden`}>
        <div className="max-w-2xl mx-auto px-5 py-4">

          {/* Patient info row */}
          <div className="flex items-start justify-between mb-3 gap-3">
            <div>
              <p className="text-slate-800 font-semibold">{patient.name}</p>
              <p className="text-slate-500 text-xs mt-0.5">
                Age {patient.age}
                {patientMeta?.gender           && ` · ${patientMeta.gender}`}
                {patientMeta?.screenerRole     && ` · ${patientMeta.screenerRole}`}
                {patientMeta?.screeningLocation && ` · ${patientMeta.screeningLocation}`}
              </p>
            </div>
            {/* Confidence badge */}
            <span className={`text-xs px-2.5 py-1 rounded-full font-semibold border shrink-0 ${CONF_COLOUR[aggConf] ?? ""}`}>
              {aggConf} confidence
            </span>
          </div>

          {/* 5-tier pill row */}
          <div className="flex gap-1.5 flex-wrap">
            {TIER_ORDER.map((tier) => {
              const tc       = URGENCY_CONFIG[tier];
              const isActive = displayTier === tier;
              return (
                <div
                  key={tier}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold
                    border transition-all ${
                    isActive
                      ? `${tc.badgeColour} text-white border-transparent shadow-sm`
                      : "bg-white/50 text-slate-400 border-slate-200"
                  }`}
                >
                  <span>{TIER_EMOJI[tier]}</span>
                  <span>{tier}</span>
                </div>
              );
            })}
            {/* Show "REVIEW" pill when confidence is LOW on non-NORMAL */}
            {displayTier === null && (
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold
                border bg-slate-500 text-white border-transparent shadow-sm">
                <span>⚪</span>
                <span>REVIEW</span>
              </div>
            )}
          </div>

          {/* LOW confidence warning */}
          {aggConf === "LOW" && (
            <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-1.5 mt-2">
              ⚠ Low detection confidence — re-screen with better fixation before acting on this result.
            </p>
          )}
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          BODY
      ══════════════════════════════════════════════════════════════════ */}
      <div className="max-w-2xl mx-auto px-5 pt-5 space-y-4 print:max-w-none print:pt-0 print:px-0 print:space-y-5">

        {/* ══════════════════════════════════════════════════════════════
            §2  CLINICAL INTERPRETATION
        ══════════════════════════════════════════════════════════════ */}
        <section className={`rounded-2xl border ${config.borderColour} ${config.bgColour} p-5 print:bg-slate-50 print:border-slate-300`}>
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1 min-w-0">
              <h2 className={`text-xl font-bold ${config.colour} mb-0.5`}>{r.condition_name}</h2>
              <span className="inline-block text-xs font-mono font-semibold bg-white/60 border border-current/20
                               px-2 py-0.5 rounded-md text-slate-500 mb-3">
                {r.icd10_code}
              </span>

              {/* Corneal reflex asymmetry — the clinical headline */}
              <div className="mb-3">
                <p className="text-slate-500 text-xs font-semibold uppercase tracking-wide mb-0.5">
                  Corneal Reflex Asymmetry
                </p>
                <p className={`text-3xl font-black ${config.colour} leading-none`}>
                  {asymDeg.toFixed(1)}°
                </p>
                <p className="text-slate-500 text-sm font-semibold mt-0.5">
                  ≈ {asymPD} prism dioptres (PD)
                </p>
              </div>

              {/* Referral action */}
              <div className="bg-white/60 rounded-xl border border-white/80 px-4 py-3 mt-3">
                <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold mb-0.5">Referral Action</p>
                <p className={`text-sm font-bold ${config.colour}`}>{r.referral_recommendation}</p>
                <p className="text-slate-500 text-xs mt-0.5">Timeframe: {r.timeframe}</p>
              </div>
            </div>

            {/* Large tier badge */}
            <div className={`${config.badgeColour} rounded-2xl px-4 py-4 text-white text-center shrink-0 shadow-sm`}>
              <p className="text-3xl leading-none">{displayTier ? TIER_EMOJI[displayTier] : "⚪"}</p>
              <p className="text-[10px] font-bold tracking-widest mt-1.5">
                {displayTier ?? "REVIEW"}
              </p>
            </div>
          </div>

          {/* Narrative */}
          <p className="text-slate-600 text-sm leading-relaxed mt-4 pt-4 border-t border-black/5">
            {r.narrative}
          </p>

          {/* Honest low-SNR disclosure — when measurement noise dominated the signal */}
          {lowSNR && (
            <div className="mt-3 bg-white/70 border border-amber-300 rounded-xl px-4 py-3">
              <p className="text-amber-800 font-bold text-xs uppercase tracking-wider mb-1">
                ⚠ Measurement noise was high
              </p>
              <p className="text-amber-700 text-xs leading-relaxed">
                Capture conditions (torch wobble, ambient reflections, or
                fixation drift) produced inter-frame variation comparable to
                the asymmetry signal itself. The result is reported as
                <strong> NORMAL </strong> because no clinically significant
                deviation was reliably detected — but the screening was not
                ideal. If you have any concern about the patient&apos;s eye
                alignment, recommend a clinical cover test by an eye
                specialist rather than relying on this scan alone.
              </p>
            </div>
          )}
        </section>

        {/* ══════════════════════════════════════════════════════════════
            §3  CLINICAL MEASUREMENTS
        ══════════════════════════════════════════════════════════════ */}
        <SectionCard>
          <SectionTitle>Clinical Measurements</SectionTitle>

          {/* Primary: Bilateral asymmetry */}
          <div className="mb-5">
            <p className="text-slate-400 text-xs font-semibold uppercase tracking-wide mb-2">
              Bilateral Asymmetry (classification signal)
            </p>
            <div className="grid grid-cols-3 gap-2">
              <div className="bg-slate-50 rounded-xl p-3 border border-slate-100 text-center">
                <p className="text-slate-400 text-xs mb-1">Asymmetry</p>
                <p className="text-slate-900 font-bold text-lg leading-none">{asymDeg.toFixed(1)}°</p>
              </div>
              <div className="bg-slate-50 rounded-xl p-3 border border-slate-100 text-center">
                <p className="text-slate-400 text-xs mb-1">PD</p>
                <p className="text-slate-900 font-bold text-lg leading-none">{asymPD}</p>
              </div>
              <div className="bg-slate-50 rounded-xl p-3 border border-slate-100 text-center">
                <p className="text-slate-400 text-xs mb-1">Severity</p>
                <p className="text-slate-900 font-bold text-lg leading-none">{r.severity}</p>
              </div>
            </div>

            {/* Avg std if available */}
            {result.asymmetry_std_deg !== undefined && (
              <p className="text-slate-400 text-xs mt-2">
                Inter-frame variance: <span className="text-slate-600 font-semibold">
                  ±{result.asymmetry_std_deg.toFixed(1)}°
                </span> std dev across {result.frames_accepted} accepted frames
              </p>
            )}
          </div>

          {/* Dominant eye + direction */}
          <div className="mb-5">
            <MetricRow label="Dominant eye"     value={`${technical.dominant_eye} (${r.severity.toLowerCase()} deviation)`} />
            <MetricRow label="Absolute deviation" value={
              result.deviation_avg_deg !== undefined
                ? `${result.deviation_avg_deg.toFixed(1)}° ± ${result.deviation_std_deg?.toFixed(1)}°`
                : `${r.deviation_degrees.toFixed(1)}°`
            } />
          </div>

          {/* Per-eye displacement */}
          <div>
            <p className="text-slate-400 text-xs font-semibold uppercase tracking-wide mb-2">
              Per-Eye Displacement
            </p>
            <div className="grid grid-cols-2 gap-3">
              {(["left", "right"] as const).map((eye) => {
                const norm = eye === "left"
                  ? technical.left_displacement_norm
                  : technical.right_displacement_norm;
                const dir = eye === "left"
                  ? technical.left_direction
                  : technical.right_direction;
                const pd  = eye === "left" ? leftPD : rightPD;
                const pct = Math.min(norm * 100, 100);
                return (
                  <div key={eye} className="bg-slate-50 rounded-xl p-4 border border-slate-100">
                    <p className="text-slate-400 text-xs uppercase tracking-wide mb-1 font-semibold">
                      {eye} eye
                    </p>
                    <p className="text-slate-900 font-bold text-base leading-none">
                      {(norm * 100).toFixed(1)}%
                    </p>
                    <p className="text-slate-400 text-xs mt-0.5 capitalize">
                      {dir} · ≈{pd} PD
                    </p>
                    <div className="mt-2 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${
                          norm > 0.5 ? "bg-red-400" : norm > 0.2 ? "bg-amber-400" : "bg-emerald-400"
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </SectionCard>

        {/* ══════════════════════════════════════════════════════════════
            §4  QUALITY & PROVENANCE
        ══════════════════════════════════════════════════════════════ */}
        <SectionCard>
          <SectionTitle>Quality &amp; Provenance</SectionTitle>

          {/* Confidence bar */}
          <div className={`flex items-center gap-2 rounded-xl px-4 py-3 border mb-4 ${CONF_COLOUR[aggConf] ?? ""}`}>
            <span className="text-lg">
              {aggConf === "HIGH" ? "✅" : aggConf === "MEDIUM" ? "⚠️" : "🔴"}
            </span>
            <div>
              <p className="font-bold text-sm">{aggConf} Confidence</p>
              <p className="text-xs opacity-80 leading-tight">
                {aggConf === "HIGH"
                  ? "Low inter-frame variance — result is reliable"
                  : aggConf === "MEDIUM"
                  ? "Moderate variance — consider re-screening to confirm"
                  : "High variance — re-screen before clinical action"}
              </p>
            </div>
          </div>

          <MetricRow
            label="Frames"
            value={`${result.frames_accepted} / ${result.frames_total} accepted${
              result.frames_rejected > 0 ? ` (${result.frames_rejected} rejected)` : ""
            }`}
          />
          {result.asymmetry_std_deg !== undefined && (
            <MetricRow
              label="Asymmetry std dev"
              value={`±${result.asymmetry_std_deg.toFixed(2)}°`}
            />
          )}
          {snr !== null && (
            <MetricRow
              label="Signal-to-noise ratio"
              value={
                snr >= 3.0 ? `${snr.toFixed(1)} (strong)` :
                snr >= 2.0 ? `${snr.toFixed(1)} (adequate)` :
                snr >= 1.0 ? `${snr.toFixed(1)} (weak — noise-dominated)` :
                             `${snr.toFixed(1)} (below noise floor)`
              }
            />
          )}
          {result.deviation_avg_deg !== undefined && result.deviation_std_deg !== undefined && (
            <MetricRow
              label="Deviation avg ± std"
              value={`${result.deviation_avg_deg.toFixed(1)}° ± ${result.deviation_std_deg.toFixed(2)}°`}
            />
          )}
          {result.deviation_min_deg !== undefined && result.deviation_max_deg !== undefined && (
            <MetricRow
              label="Deviation range"
              value={`${result.deviation_min_deg.toFixed(1)}° – ${result.deviation_max_deg.toFixed(1)}°`}
            />
          )}
          <MetricRow label="Screener"       value={screenerCtx} />
          <MetricRow label="Session ID"     value={sessionId} />
          <MetricRow label="Timestamp"      value={new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC"} />
          <MetricRow label="Pipeline"       value="BeanHealth CLR v1.0 · Hirschberg (7°/mm · 22 PD/mm)" />
        </SectionCard>

        {/* ══════════════════════════════════════════════════════════════
            §5  PER-FRAME AUDIT TRAIL
        ══════════════════════════════════════════════════════════════ */}
        {result.per_frame_readings && result.per_frame_readings.length > 0 && (
          <SectionCard className="print:hidden">
            <SectionTitle>
              Frame Audit ({result.frames_accepted}/{result.frames_total} accepted)
            </SectionTitle>

            <div className="grid grid-cols-5 gap-1.5">
              {result.per_frame_readings.map((reading, i) => {
                const rejection = result.per_frame_rejections?.[i];
                const accepted  = reading !== null;
                return (
                  <div
                    key={i}
                    className={`rounded-xl border px-1.5 py-2 text-center ${
                      accepted
                        ? "bg-emerald-50 border-emerald-200"
                        : "bg-slate-100 border-slate-200"
                    }`}
                  >
                    <p className="text-[9px] text-slate-400 font-mono mb-0.5">F{i + 1}</p>
                    {accepted ? (
                      <p className="text-emerald-700 text-xs font-bold">{reading?.toFixed(1)}°</p>
                    ) : (
                      <>
                        <p className="text-slate-400 text-xs font-bold">✕</p>
                        <p className="text-[8px] text-slate-400 leading-tight mt-0.5 font-medium">
                          {rejLabel(rejection)}
                        </p>
                      </>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Mini deviation sparkline */}
            {result.frames_accepted >= 2 && (
              <div className="mt-3 pt-3 border-t border-slate-100">
                <p className="text-slate-400 text-xs mb-1.5">Accepted readings:</p>
                <div className="flex items-end gap-1 h-8">
                  {result.per_frame_readings.map((v, i) => {
                    if (v === null) return null;
                    const maxVal = Math.max(...result.per_frame_readings.filter((x) => x !== null) as number[]);
                    const h = maxVal > 0 ? Math.max(4, Math.round((v / maxVal) * 28)) : 4;
                    return (
                      <div
                        key={i}
                        title={`F${i + 1}: ${v?.toFixed(1)}°`}
                        className="bg-blue-400 rounded-sm flex-1 min-w-0"
                        style={{ height: `${h}px` }}
                      />
                    );
                  })}
                </div>
              </div>
            )}
          </SectionCard>
        )}

        {/* ══════════════════════════════════════════════════════════════
            §6  ANNOTATED VISUALS
        ══════════════════════════════════════════════════════════════ */}

        {result.annotated_image_b64 && (
          <SectionCard>
            <SectionTitle>Annotated Scan</SectionTitle>
            <AnnotatedEye base64Jpeg={result.annotated_image_b64} patientName={patient.name} />

            {/* Legend */}
            <div className="flex flex-wrap gap-x-5 gap-y-1.5 mt-4 pt-3 border-t border-slate-100">
              {[
                { cls: "bg-blue-500",            label: "Pupil centre" },
                { cls: "bg-amber-400",            label: "Corneal reflex (CLR)" },
                { cls: "bg-white border border-slate-300", label: "Hough estimate" },
              ].map((l) => (
                <div key={l.label} className="flex items-center gap-1.5">
                  <div className={`w-2.5 h-2.5 rounded-full shrink-0 ${l.cls}`} />
                  <span className="text-xs text-slate-500">{l.label}</span>
                </div>
              ))}
              <div className="flex items-center gap-1.5">
                <div className="w-5 h-[2px] bg-slate-400 shrink-0" />
                <span className="text-xs text-slate-500">Displacement vector</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className={`w-3 h-3 rounded-sm border-2 shrink-0 ${
                  r.urgency_tier === "URGENT"  ? "border-red-500"    :
                  r.urgency_tier === "ROUTINE" ? "border-orange-500" :
                  r.urgency_tier === "MONITOR" ? "border-amber-500"  : "border-emerald-500"
                }`} />
                <span className="text-xs text-slate-500">Urgency border</span>
              </div>
            </div>
          </SectionCard>
        )}

        {/* Processing pipeline steps */}
        {result.intermediate_images && (
          <div className="print:hidden">
            <ProcessingSteps intermediateImages={result.intermediate_images} />
          </div>
        )}

        {/* Detection flags (if any) */}
        {technical.flags.length > 0 && (
          <section className="bg-amber-50 rounded-2xl border border-amber-200 p-5">
            <h2 className="text-amber-800 font-semibold text-sm uppercase tracking-wider mb-2">
              ⚠ Detection Flags
            </h2>
            <div className="flex flex-wrap gap-1.5">
              {technical.flags.map((flag) => (
                <span
                  key={flag}
                  className="text-xs font-mono bg-amber-100 border border-amber-200 text-amber-700 px-2 py-0.5 rounded-md"
                >
                  {flag}
                </span>
              ))}
            </div>
          </section>
        )}

        {/* ══════════════════════════════════════════════════════════════
            §7  FOOTER — LIMITATIONS + TECHNICAL + ACTIONS
        ══════════════════════════════════════════════════════════════ */}

        {/* Screening limitations */}
        <section className="bg-slate-100 rounded-2xl border border-slate-200 p-5 print:bg-slate-50">
          <h2 className="text-slate-500 font-semibold text-xs uppercase tracking-wider mb-3">
            Screening Limitations
          </h2>
          <ul className="space-y-2 text-slate-500 text-xs leading-relaxed">
            <li>
              <strong className="text-slate-600">Microtropia (&lt;8 PD):</strong>{" "}
              This tool cannot reliably detect strabismus below 8 PD. Small-angle cases require a
              cover test or synoptophore examination by a trained examiner.
            </li>
            <li>
              <strong className="text-slate-600">Intermittent strabismus:</strong>{" "}
              If the deviation only manifests at certain distances, times of day, or fatigue states,
              a single-session CLR capture may miss it entirely.
            </li>
            <li>
              <strong className="text-slate-600">Pseudostrabismus:</strong>{" "}
              Wide epicanthal folds (common in children under 2 years) can simulate medial esotropia.
              This tool may over-flag these cases — age context is critical.
            </li>
          </ul>
        </section>

        {/* Collapsible technical details (screen only) */}
        <div className="print:hidden">
          <button
            onClick={() => setShowTechnical((s) => !s)}
            className="w-full text-left text-slate-400 text-xs font-medium px-1 py-1.5 flex items-center gap-2
                       hover:text-slate-600 transition-colors"
          >
            <span>{showTechnical ? "▲" : "▼"}</span>
            <span>{showTechnical ? "Hide" : "Show"} technical details (JSON)</span>
          </button>
          {showTechnical && (
            <pre className="text-[10px] bg-slate-900 text-emerald-300 rounded-xl p-4 overflow-auto max-h-72 mt-1
                           font-mono leading-relaxed">
              {JSON.stringify(
                {
                  result:    r,
                  technical,
                  aggregate: {
                    confidence:          aggConf,
                    frames_total:        result.frames_total,
                    frames_accepted:     result.frames_accepted,
                    frames_rejected:     result.frames_rejected,
                    deviation_avg_deg:   result.deviation_avg_deg,
                    deviation_std_deg:   result.deviation_std_deg,
                    asymmetry_avg_deg:   result.asymmetry_avg_deg,
                    asymmetry_std_deg:   result.asymmetry_std_deg,
                    per_frame_readings:  result.per_frame_readings,
                    per_frame_rejections:result.per_frame_rejections,
                  },
                },
                null,
                2
              )}
            </pre>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex gap-3 print:hidden pb-4">
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

        {/* Print-only footer */}
        <div className="hidden print:block pt-4 mt-4 border-t border-slate-300 text-[8pt] text-slate-500 leading-relaxed">
          <p>
            <strong className="text-slate-700">Disclaimer:</strong>{" "}
            This report is generated by BeanHealth CLR Tool v1.0.0, an AI-assisted strabismus
            screening aid. It does not constitute a medical diagnosis and must not replace
            examination by a qualified ophthalmologist. Results are based on the Hirschberg corneal
            light reflex test (7°/mm, 22 PD/mm, iris radius 5.75 mm). Cannot detect strabismus &lt;8 PD.
          </p>
          <p className="mt-1 text-slate-400">
            Generated: {new Date().toISOString()} · Status: {result.status} · Confidence: {aggConf} ·
            BeanHealth · EyeQ Innovate Hackathon 2.0
          </p>
        </div>

      </div>
    </div>
  );
}
