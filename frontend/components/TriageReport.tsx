"use client";

import { useState } from "react";
import type { AlignmentResult, AlignmentVerdict, StreamSuccessResponse, UrgencyTier } from "@/lib/types";
import { URGENCY_CONFIG } from "@/lib/types";
import AnnotatedEye from "./AnnotatedEye";
import ProcessingSteps from "./ProcessingSteps";

// ─── Corner-alignment display config ────────────────────────────────────────

const ALIGN_CONFIG: Record<
  AlignmentVerdict,
  { label: string; emoji: string; text: string; bg: string; border: string }
> = {
  ALIGNED: {
    label: "Aligned", emoji: "🟢", text: "text-emerald-700",
    bg: "bg-emerald-50", border: "border-emerald-200",
  },
  BORDERLINE: {
    label: "Borderline", emoji: "🟡", text: "text-amber-700",
    bg: "bg-amber-50", border: "border-amber-200",
  },
  ASYMMETRIC: {
    label: "Asymmetric — flag", emoji: "🔴", text: "text-red-700",
    bg: "bg-red-50", border: "border-red-200",
  },
  UNAVAILABLE: {
    label: "Not assessed", emoji: "⚪", text: "text-ink-500",
    bg: "bg-ink-50", border: "border-ink-100",
  },
};

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

/**
 * Convert degrees to prism dioptres. A prism dioptre is defined as
 * 100 × tan(angle), so the relationship is trigonometric — the linear
 * "PD per degree" shortcuts (15/7, 22/7) overstate larger angles.
 * Fallback only — prefer the backend's `asymmetry_pd` field, which is the
 * single source of truth for this conversion.
 */
function toPD(degrees: number): number {
  return Math.round(100 * Math.tan((degrees * Math.PI) / 180));
}

function rejLabel(reason: string | null | undefined): string {
  if (!reason) return "rejected";
  return REJECTION_LABEL[reason] ?? reason.replace(/_/g, " ");
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-baseline py-2.5 border-b border-ink-100 last:border-0 gap-4">
      <span className="text-ink-500 text-[13.5px] shrink-0">{label}</span>
      <span className="text-ink-900 text-[13.5px] font-semibold text-right tabular">{value}</span>
    </div>
  );
}

function SectionCard({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <section className={`bg-white rounded-card border border-ink-100 shadow-card p-6 print:shadow-none ${className}`}>
      {children}
    </section>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="eyebrow-muted mb-5">{children}</h2>
  );
}

/** A boxed statistic — the report's basic quantitative unit. */
function StatBox({ label, value, tone = "text-ink-900" }: { label: string; value: string; tone?: string }) {
  return (
    <div className="bg-ink-50 rounded-field px-3 py-3.5 border border-ink-100 text-center">
      <p className="text-ink-400 text-[10.5px] font-semibold uppercase tracking-eyebrow mb-1.5">{label}</p>
      <p className={`font-display text-[24px] leading-none tabular ${tone}`}>{value}</p>
    </div>
  );
}

/** §2B — CLR-free pupil-vs-corner alignment (the second, independent view). */
export function AlignmentSection(
  { alignment, imageB64 }: { alignment: AlignmentResult; imageB64?: string | null },
) {
  const cfg = ALIGN_CONFIG[alignment.verdict] ?? ALIGN_CONFIG.UNAVAILABLE;
  const pct = (v: number | null) => (v === null ? "–" : `${(v * 100).toFixed(1)}%`);

  return (
    <section className={`rounded-card border ${cfg.border} ${cfg.bg} p-6`}>
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <p className="eyebrow-muted">Method B · No torch required</p>
          <h2 className="font-display text-[20px] text-ink-900 leading-snug mt-1">
            Pupil-versus-corner alignment
          </h2>
        </div>
        <span className={`font-mono text-[11px] px-3 py-1.5 rounded-full font-semibold border
                          bg-white/80 ${cfg.text} ${cfg.border} shrink-0`}>
          {cfg.label}
        </span>
      </div>

      <p className="text-ink-500 text-[12.5px] leading-relaxed mb-4">
        This independent check measures where each pupil sits between its own eye
        corners (inner→outer canthus) and compares the two eyes. It needs no
        corneal light reflex, so it works on ordinary photos — but it gives a
        geometric flag only, not a clinical angle.
      </p>

      {imageB64 && (
        <figure className="mb-4">
          <img
            src={`data:image/jpeg;base64,${imageB64}`}
            alt="Each pupil projected onto its own inner-to-outer eye-corner axis"
            className="w-full rounded-field border border-black/10 block"
          />
          <figcaption className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-[11px] text-ink-500">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 rounded-full" style={{ background: "#ff78c8" }} />
              Eye corners
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-px" style={{ background: "#c8c8c8" }} />
              Corner-to-corner axis
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 rounded-full" style={{ background: "#1464eb" }} />
              Pupil
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 rounded-full" style={{ background: "#ffc878" }} />
              Projection onto axis
            </span>
          </figcaption>
        </figure>
      )}

      {alignment.available ? (
        <>
          <div className="grid grid-cols-2 gap-3 mb-3">
            {(["left", "right"] as const).map((eye) => {
              const h = eye === "left" ? alignment.left_h_ratio : alignment.right_h_ratio;
              return (
                <div key={eye} className="bg-white/70 rounded-field p-3.5 border border-black/5">
                  <p className="text-ink-400 text-[11px] uppercase tracking-eyebrow font-semibold mb-1.5">
                    {eye} eye
                  </p>
                  <p className="font-display text-ink-900 text-[20px] leading-none tabular">{pct(h)}</p>
                  <p className="text-ink-400 text-xs mt-0.5">inner→outer position</p>
                </div>
              );
            })}
          </div>
          <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-ink-500 mb-3">
            <span>Horizontal asymmetry: <strong className="text-ink-700">{pct(alignment.h_asymmetry)}</strong></span>
            <span>Vertical asymmetry: <strong className="text-ink-700">{pct(alignment.v_asymmetry)}</strong></span>
          </div>
        </>
      ) : null}

      <p className={`text-sm leading-relaxed ${alignment.available ? "text-ink-600" : "text-ink-500"}`}>
        {alignment.interpretation}
      </p>
    </section>
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

  // Clinical angle values. Prefer the backend's authoritative prism-dioptre
  // value; fall back to local conversion only for older responses.
  const asymDeg = r.asymmetry_degrees ?? r.deviation_degrees;
  const asymPD  = r.asymmetry_pd ?? toPD(asymDeg);

  // Dates
  // Derived from the analysis timestamp, formatted in UTC so the server and the
  // client always render the same string (no hydration mismatch, no midnight
  // timezone drift) and so the printed report agrees with the UTC stamp below.
  const capturedAt = new Date(result.timestamp);
  const printDate  = capturedAt.toLocaleDateString("en-GB", {
    day: "2-digit", month: "long", year: "numeric", timeZone: "UTC",
  });
  const printTime  = capturedAt.toLocaleTimeString("en-GB", {
    hour: "2-digit", minute: "2-digit", timeZone: "UTC",
  }) + " UTC";

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
    <div className="min-h-screen bg-white pb-16 print:pb-0">

      {/* ══════════════════════════════════════════════════════════════════
          PRINT-ONLY HEADER
      ══════════════════════════════════════════════════════════════════ */}
      <div className="hidden print:block px-6 pt-5 pb-4 border-b-2 border-ink-900 mb-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-[7.5pt] font-black tracking-[0.25em] text-ink-400 uppercase mb-0.5">BeanHealth</p>
            <h1 className="text-[18pt] font-bold text-ink-900 leading-tight">CLR Triage Report</h1>
            <p className="text-[8.5pt] text-ink-500 mt-0.5">Corneal Light Reflex Asymmetry · Hirschberg Method · 10-Frame Analysis</p>
          </div>
          <div className="text-right text-[8.5pt] text-ink-500 leading-relaxed">
            <p className="font-semibold text-ink-700">{printDate}</p>
            <p>{printTime}</p>
            <p className="mt-1 text-ink-400">Session: {sessionId}</p>
          </div>
        </div>
        <div className="mt-3 pt-2 border-t border-ink-100 flex flex-wrap gap-x-8 gap-y-0.5 text-[9pt]">
          <span>
            <span className="text-ink-400 uppercase tracking-wide text-[7.5pt] mr-1">Patient</span>
            <span className="font-semibold text-ink-800">{patient.name}</span>
          </span>
          <span>
            <span className="text-ink-400 uppercase tracking-wide text-[7.5pt] mr-1">Age</span>
            <span className="font-semibold text-ink-800">{patient.age}</span>
          </span>
          {patientMeta?.gender && (
            <span>
              <span className="text-ink-400 uppercase tracking-wide text-[7.5pt] mr-1">Gender</span>
              <span className="font-semibold text-ink-800">{patientMeta.gender}</span>
            </span>
          )}
          <span>
            <span className="text-ink-400 uppercase tracking-wide text-[7.5pt] mr-1">Tier</span>
            <span className="font-bold text-ink-800">{r.urgency_tier}</span>
          </span>
          <span>
            <span className="text-ink-400 uppercase tracking-wide text-[7.5pt] mr-1">Condition</span>
            <span className="font-semibold text-ink-800">{r.condition_name}</span>
          </span>
          <span>
            <span className="text-ink-400 uppercase tracking-wide text-[7.5pt] mr-1">ICD-10</span>
            <span className="font-semibold text-ink-800">{r.icd10_code}</span>
          </span>
          <span>
            <span className="text-ink-400 uppercase tracking-wide text-[7.5pt] mr-1">Asymmetry</span>
            <span className="font-semibold text-ink-800">{asymDeg.toFixed(1)}° ({asymPD} PD)</span>
          </span>
          <span>
            <span className="text-ink-400 uppercase tracking-wide text-[7.5pt] mr-1">Screener</span>
            <span className="font-semibold text-ink-800">{screenerCtx}</span>
          </span>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          §1  HEADER BAND (screen only)
      ══════════════════════════════════════════════════════════════════ */}

      {/* App bar */}
      <header className="bg-white border-b border-ink-100 sticky top-0 z-30 print:hidden">
        <div className="max-w-2xl mx-auto px-5 py-3.5 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <span className="w-9 h-9 rounded-[10px] bg-ink-900 flex items-center justify-center shrink-0">
              <svg className="w-[18px] h-[18px] text-white" fill="none" viewBox="0 0 24 24" strokeWidth={1.6} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
              </svg>
            </span>
            <div className="leading-none">
              <p className="text-[14px] font-semibold text-ink-900">
                BeanHealth <span className="text-ink-400 font-normal">CLR</span>
              </p>
              <p className="text-[10.5px] text-ink-400 mt-1">Triage report</p>
            </div>
          </div>
          <div className="text-right leading-none">
            <p className="font-mono text-[11px] text-ink-800 tabular">{sessionId}</p>
            <p className="text-[10.5px] text-ink-400 mt-1">{printDate}</p>
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
      <div className="bg-white border-b border-ink-100 print:hidden">
        <div className="max-w-2xl mx-auto px-5 py-5">

          {/* Patient info row */}
          <div className="flex items-start justify-between mb-4 gap-3">
            <div className="min-w-0">
              <p className="font-display text-[22px] text-ink-900 leading-tight truncate">{patient.name}</p>
              <p className="text-ink-400 text-[12px] mt-1">
                Age {patient.age}
                {patientMeta?.gender            && ` · ${patientMeta.gender}`}
                {patientMeta?.screenerRole      && ` · ${patientMeta.screenerRole}`}
                {patientMeta?.screeningLocation && ` · ${patientMeta.screeningLocation}`}
              </p>
            </div>
            <span className={`text-[11px] px-2.5 py-1 rounded-full font-semibold border shrink-0 ${CONF_COLOUR[aggConf] ?? ""}`}>
              {aggConf} confidence
            </span>
          </div>

          {/* Tier rail — the active tier is filled, the rest are hairline ghosts */}
          <div className="flex gap-1.5 flex-wrap">
            {TIER_ORDER.map((tier) => {
              const tc       = URGENCY_CONFIG[tier];
              const isActive = displayTier === tier;
              return (
                <span
                  key={tier}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full font-mono text-[11px] font-semibold
                              border transition-all duration-200 ${
                    isActive
                      ? `${tc.badgeColour} text-white border-transparent shadow-card`
                      : "bg-white text-ink-300 border-ink-100"
                  }`}
                >
                  <span className={`w-1.5 h-1.5 rounded-full ${isActive ? "bg-white" : tc.dotColour + " opacity-30"}`} />
                  {tier}
                </span>
              );
            })}
            {displayTier === null && (
              <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-full font-mono text-[11px] font-semibold
                               border bg-ink-500 text-white border-transparent shadow-card">
                <span className="w-1.5 h-1.5 rounded-full bg-white" />
                REVIEW
              </span>
            )}
          </div>

          {/* LOW confidence warning */}
          {aggConf === "LOW" && (
            <p className="text-[12.5px] text-amber-800 bg-amber-50 border border-amber-200 rounded-field px-3.5 py-2.5 mt-3 leading-relaxed">
              Low detection confidence — re-screen with better fixation before acting on this result.
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
        <section className={`rounded-card border ${config.borderColour} ${config.bgColour} overflow-hidden print:bg-white print:border-ink-100`}>

          {/* Verdict header — condition on the left, tier block on the right */}
          <div className="flex items-stretch">
            <div className="flex-1 min-w-0 p-6">
              <p className="eyebrow-muted mb-2">Method A · Hirschberg corneal light reflex</p>
              <h2 className={`font-display text-[28px] leading-tight ${config.colour}`}>
                {r.condition_name}
              </h2>
              <span className="inline-block font-mono text-[11px] font-medium bg-white/70 border border-black/5
                               px-2 py-0.5 rounded mt-2 text-ink-600">
                {r.icd10_code}
              </span>
            </div>

            <div className={`${config.badgeColour} px-6 py-6 text-white text-center shrink-0
                             flex flex-col justify-center min-w-[128px]`}>
              <p className="font-display text-[26px] leading-none">
                {displayTier ?? "Review"}
              </p>
              <p className="font-mono text-[10px] font-semibold tracking-[0.18em] mt-2.5 opacity-80">
                {r.severity}
              </p>
            </div>
          </div>

          {/* Measurement + referral action */}
          <div className="px-6 pb-6 space-y-4">
            <div className="grid sm:grid-cols-2 gap-3">
              <div className="bg-white/70 rounded-field border border-black/5 px-4 py-3.5">
                <p className="eyebrow-muted mb-1.5">Corneal reflex asymmetry</p>
                <p className={`font-display text-[26px] leading-none tabular ${config.colour}`}>
                  {asymDeg.toFixed(1)}&deg;
                </p>
                <p className="text-ink-500 text-[12.5px] mt-1.5">
                  &asymp; <span className="font-mono tabular">{asymPD}</span> prism dioptres
                </p>
              </div>

              <div className="bg-white/70 rounded-field border border-black/5 px-4 py-3.5">
                <p className="eyebrow-muted mb-1.5">Referral action</p>
                <p className={`text-[14px] font-semibold leading-snug ${config.colour}`}>
                  {r.referral_recommendation}
                </p>
                <p className="text-ink-500 text-[12.5px] mt-1.5">Timeframe: {r.timeframe}</p>
              </div>
            </div>

            {/* Narrative */}
            <p className="text-ink-600 text-[13.5px] leading-relaxed pt-4 border-t border-black/5">
              {r.narrative}
            </p>

            {/* Honest low-SNR disclosure */}
            {lowSNR && (
              <div className="bg-white/80 border border-amber-300 rounded-field px-4 py-3.5">
                <p className="text-amber-800 font-semibold text-[11px] uppercase tracking-eyebrow mb-1.5">
                  Measurement noise was high
                </p>
                <p className="text-amber-800/90 text-[12.5px] leading-relaxed">
                  Capture conditions (torch wobble, ambient reflections, or fixation drift)
                  produced inter-frame variation comparable to the asymmetry signal itself.
                  The result is reported as <strong>NORMAL</strong> because no clinically
                  significant deviation was reliably detected &mdash; but the screening was not
                  ideal. If you have any concern about the patient&apos;s eye alignment,
                  recommend a clinical cover test by an eye specialist rather than relying
                  on this scan alone.
                </p>
              </div>
            )}
          </div>
        </section>

        {/* ══════════════════════════════════════════════════════════════
            §2B  PUPIL-VS-CORNER ALIGNMENT (CLR-free, second view)
        ══════════════════════════════════════════════════════════════ */}
        {result.alignment && (
          <AlignmentSection
            alignment={result.alignment}
            imageB64={result.intermediate_images?.module8_alignment}
          />
        )}

        {/* ══════════════════════════════════════════════════════════════
            §3  CLINICAL MEASUREMENTS
        ══════════════════════════════════════════════════════════════ */}
        <SectionCard>
          <SectionTitle>Clinical Measurements</SectionTitle>

          {/* Primary: Bilateral asymmetry */}
          <div className="mb-5">
            <p className="text-ink-400 text-[11px] font-semibold uppercase tracking-eyebrow mb-2.5">
              Bilateral asymmetry &middot; classification signal
            </p>
            <div className="grid grid-cols-3 gap-2">
              <StatBox label="Asymmetry" value={`${asymDeg.toFixed(1)}°`} />
              <StatBox label="Prism D"   value={String(asymPD)} />
              <StatBox label="Severity"  value={r.severity} />
            </div>

            {/* Avg std if available */}
            {result.asymmetry_std_deg !== undefined && (
              <p className="text-ink-400 text-xs mt-2">
                Inter-frame variance: <span className="text-ink-600 font-semibold">
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
            <p className="text-ink-400 text-[11px] font-semibold uppercase tracking-eyebrow mb-2.5">
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
                  <div key={eye} className="bg-ink-50 rounded-field p-4 border border-ink-100">
                    <p className="text-ink-400 text-[11px] uppercase tracking-eyebrow mb-1.5 font-semibold">
                      {eye} eye
                    </p>
                    <p className="font-display text-ink-900 text-[20px] leading-none tabular">
                      {(norm * 100).toFixed(1)}%
                    </p>
                    <p className="text-ink-400 text-xs mt-0.5 capitalize">
                      {dir} · ≈{pd} PD
                    </p>
                    <div className="mt-2 h-1.5 bg-ink-200 rounded-full overflow-hidden">
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
          <div className={`flex items-start gap-3 rounded-field px-4 py-3.5 border mb-5 ${CONF_COLOUR[aggConf] ?? ""}`}>
            <span className={`w-2 h-2 rounded-full shrink-0 mt-1.5 ${
              aggConf === "HIGH" ? "bg-emerald-500" : aggConf === "MEDIUM" ? "bg-amber-500" : "bg-red-500"
            }`} />
            <div>
              <p className="font-semibold text-[13.5px]">{aggConf} confidence</p>
              <p className="text-[12.5px] opacity-80 leading-snug mt-0.5">
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
          <MetricRow label="Timestamp"      value={`${result.timestamp.replace("T", " ").slice(0, 19)} UTC`} />
          <MetricRow label="Pipeline"       value="BeanHealth CLR v1.0 · Hirschberg 7°/mm · PD = 100·tan(θ)" />
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
                    className={`rounded-field border px-1.5 py-2 text-center ${
                      accepted
                        ? "bg-emerald-50 border-emerald-200"
                        : "bg-ink-100 border-ink-100"
                    }`}
                  >
                    <p className="text-[9px] text-ink-400 font-mono mb-0.5">F{i + 1}</p>
                    {accepted ? (
                      <p className="text-emerald-700 text-xs font-bold">{reading?.toFixed(1)}°</p>
                    ) : (
                      <>
                        <p className="text-ink-400 text-xs font-bold">✕</p>
                        <p className="text-[8px] text-ink-400 leading-tight mt-0.5 font-medium">
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
              <div className="mt-3 pt-3 border-t border-ink-100">
                <p className="text-ink-400 text-xs mb-1.5">Accepted readings:</p>
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
            <div className="flex flex-wrap gap-x-5 gap-y-1.5 mt-4 pt-3 border-t border-ink-100">
              {[
                { cls: "bg-clinical-500",            label: "Pupil centre" },
                { cls: "bg-amber-400",            label: "Corneal reflex (CLR)" },
                { cls: "bg-white border border-ink-200", label: "Hough estimate" },
              ].map((l) => (
                <div key={l.label} className="flex items-center gap-1.5">
                  <div className={`w-2.5 h-2.5 rounded-full shrink-0 ${l.cls}`} />
                  <span className="text-xs text-ink-500">{l.label}</span>
                </div>
              ))}
              <div className="flex items-center gap-1.5">
                <div className="w-5 h-[2px] bg-ink-400 shrink-0" />
                <span className="text-xs text-ink-500">Displacement vector</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className={`w-3 h-3 rounded-sm border-2 shrink-0 ${
                  r.urgency_tier === "URGENT"  ? "border-red-500"    :
                  r.urgency_tier === "ROUTINE" ? "border-orange-500" :
                  r.urgency_tier === "MONITOR" ? "border-amber-500"  : "border-emerald-500"
                }`} />
                <span className="text-xs text-ink-500">Urgency border</span>
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
          <section className="bg-amber-50 rounded-card border border-amber-200 p-5">
            <h2 className="text-amber-700 text-[10.5px] font-semibold uppercase tracking-eyebrow mb-3">
              Detection flags
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
        <section className="bg-ink-50 rounded-card border border-ink-100 p-6 print:bg-white">
          <h2 className="eyebrow-muted mb-4">Screening limitations</h2>
          <ul className="space-y-3 text-ink-500 text-[12.5px] leading-relaxed">
            <li>
              <strong className="text-ink-600">Microtropia (&lt;8 PD):</strong>{" "}
              This tool cannot reliably detect strabismus below 8 PD. Small-angle cases require a
              cover test or synoptophore examination by a trained examiner.
            </li>
            <li>
              <strong className="text-ink-600">Intermittent strabismus:</strong>{" "}
              If the deviation only manifests at certain distances, times of day, or fatigue states,
              a single-session CLR capture may miss it entirely.
            </li>
            <li>
              <strong className="text-ink-600">Pseudostrabismus:</strong>{" "}
              Wide epicanthal folds (common in children under 2 years) can simulate medial esotropia.
              This tool may over-flag these cases — age context is critical.
            </li>
          </ul>
        </section>

        {/* Collapsible technical details (screen only) */}
        <div className="print:hidden">
          <button
            onClick={() => setShowTechnical((s) => !s)}
            className="w-full text-left text-ink-400 text-xs font-medium px-1 py-1.5 flex items-center gap-2
                       hover:text-ink-600 transition-colors"
          >
            <span>{showTechnical ? "▲" : "▼"}</span>
            <span>{showTechnical ? "Hide" : "Show"} technical details (JSON)</span>
          </button>
          {showTechnical && (
            <pre className="text-[10.5px] bg-ink-900 text-emerald-300 rounded-field p-4 overflow-auto max-h-72 mt-1
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
        <div className="flex flex-col sm:flex-row gap-3 print:hidden pb-4">
          <button onClick={onRetry} className="btn-secondary flex-1">
            New screening
          </button>
          <button onClick={() => window.print()} className="btn-primary flex-1">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M6.72 13.829c-.24.03-.48.062-.72.096m.72-.096a42.415 42.415 0 0 1 10.56 0m-10.56 0L6.34 18m10.94-4.171c.24.03.48.062.72.096m-.72-.096L17.66 18m0 0 .229 2.523a1.125 1.125 0 0 1-1.12 1.227H7.231c-.662 0-1.18-.568-1.12-1.227L6.34 18m11.318 0h1.091A2.25 2.25 0 0 0 21 15.75V9.456c0-1.081-.768-2.015-1.837-2.175a48.055 48.055 0 0 0-1.913-.247M6.34 18H5.25A2.25 2.25 0 0 1 3 15.75V9.456c0-1.081.768-2.015 1.837-2.175a48.041 48.041 0 0 1 1.913-.247m10.5 0a48.536 48.536 0 0 0-10.5 0m10.5 0V3.375c0-.621-.504-1.125-1.125-1.125h-8.25c-.621 0-1.125.504-1.125 1.125v3.659M18 10.5h.008v.008H18V10.5Z" />
            </svg>
            Print / save PDF
          </button>
        </div>

        {/* Print-only footer */}
        <div className="hidden print:block pt-4 mt-4 border-t border-ink-200 text-[8pt] text-ink-500 leading-relaxed">
          <p>
            <strong className="text-ink-700">Disclaimer:</strong>{" "}
            This report is generated by BeanHealth CLR Tool v1.0.0, an AI-assisted strabismus
            screening aid. It does not constitute a medical diagnosis and must not replace
            examination by a qualified ophthalmologist. Results are based on the Hirschberg corneal
            light reflex test (7°/mm, iris radius 5.75 mm; prism dioptres = 100&middot;tan&thinsp;&theta;). Cannot detect strabismus &lt;8 PD.
          </p>
          <p className="mt-1 text-ink-400">
            Generated: {result.timestamp} · Status: {result.status} · Confidence: {aggConf} ·
            BeanHealth · EyeQ Innovate Hackathon 2.0
          </p>
        </div>

      </div>
    </div>
  );
}
