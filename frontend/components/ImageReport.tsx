"use client";

/**
 * BeanHealth CLR — single-image analysis report (printable)
 * =========================================================
 *
 * One page for a saved photo run through /analyse-test: the eyes with the
 * pupil and reflex marks, Method A (Hirschberg corneal light reflex) and
 * Method B (pupil-versus-eye-corner alignment, no flash needed) side by side,
 * whether the two agree, quality notes, and blank fields for the clinician's
 * prism cover test so a printed copy doubles as a validation record.
 */

import type {
  AlignmentResult,
  AlignmentVerdict,
  InconclusiveResponse,
  SuccessResponse,
  UrgencyTier,
} from "@/lib/types";
import { URGENCY_CONFIG } from "@/lib/types";

// Hirschberg conversion used by the pipeline (backend utils/constants.py).
const DEG_PER_MM = 7.0;
const IRIS_RADIUS_MM = 5.75;

const toPD = (deg: number) => 100 * Math.tan((deg * Math.PI) / 180);
const pct = (v: number | null | undefined) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(1)}%`);

const VERDICT: Record<AlignmentVerdict, { label: string; tone: string }> = {
  ALIGNED:     { label: "Aligned",     tone: "bg-emerald-50 text-emerald-700 border-emerald-200" },
  BORDERLINE:  { label: "Borderline",  tone: "bg-amber-50 text-amber-700 border-amber-200" },
  ASYMMETRIC:  { label: "Asymmetric",  tone: "bg-red-50 text-red-700 border-red-200" },
  UNAVAILABLE: { label: "Not assessed", tone: "bg-ink-50 text-ink-500 border-ink-100" },
};

type Eye = "left" | "right";

/** Plain-language note for a flag a reader of the printed report needs, and the eye it is about. */
function flagNote(flag: string): { text: string; eye: Eye | null } | null {
  const m = flag.match(/^(.*)_(left|right)$/);
  const base = m ? m[1] : flag;
  const eye = m ? (m[2] as Eye) : null;
  const note = (text: string) => ({ text, eye });
  switch (base) {
    case "hough_rejected":
    case "darkblob_reseeded":
      return note("the iris-circle detector missed the iris (common on dark irises), so the pupil was found from the iris landmarks.");
    case "ambiguous_reflex":
      return note("more than one bright spot; the one nearest the pupil was used. Check the amber ring sits on the torch reflex.");
    case "pupil_disagreement":
      return note("pupil and iris centres disagree. Check the blue cross sits on the pupil.");
    case "darkblob_rejected":
    case "pupil_darkblob_failed":
      return note("the dark pupil could not be isolated, so the iris-circle centre was used.");
    case "borderline_asymmetry":
      return note("the reading is close to a decision threshold.");
    case "low_snr_noise_dominated":
      return note("measurement noise is comparable to the reading.");
    case "pupil_from_darkblob":
    case "bav_mode":
    case "test_capture":
    case "no_flash":          // Method A already states why it has no reading
      return null;
    default:
      return note(flag.replace(/_/g, " "));
  }
}

/** One line per note, naming the eye(s) it applies to. */
function qualityNotes(flags: string[]): string[] {
  const byText = new Map<string, Set<Eye | null>>();
  for (const f of flags) {
    const n = flagNote(f);
    if (!n) continue;
    const eyes = byText.get(n.text) ?? new Set<Eye | null>();
    eyes.add(n.eye);
    byText.set(n.text, eyes);
  }
  return Array.from(byText, ([text, eyes]) => {
    const who = eyes.has("left") && eyes.has("right") ? "Both eyes"
      : eyes.has("right") ? "Right eye" : eyes.has("left") ? "Left eye" : null;
    return who ? `${who}: ${text}` : text.charAt(0).toUpperCase() + text.slice(1);
  });
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-[10px] uppercase tracking-[0.14em] text-ink-400 font-semibold">{label}</p>
      <p className="text-[13.5px] text-ink-900 font-semibold mt-0.5">{value}</p>
    </div>
  );
}

function Panel({ eyebrow, title, badge, children }: {
  eyebrow: string; title: string; badge: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <section className="rounded-xl border border-ink-100 p-4 print:p-3 break-inside-avoid flex flex-col">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <p className="text-[10px] uppercase tracking-[0.14em] text-ink-400 font-semibold">{eyebrow}</p>
          <h2 className="text-[15px] font-semibold text-ink-900 leading-snug mt-0.5">{title}</h2>
        </div>
        {badge}
      </div>
      {children}
    </section>
  );
}

function Pill({ text, tone }: { text: string; tone: string }) {
  return (
    <span className={`shrink-0 text-[11px] font-semibold px-2.5 py-1 rounded-full border ${tone}`}>{text}</span>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-4 py-1.5 print:py-1 border-b border-ink-100 last:border-0 text-[12.5px]">
      <span className="text-ink-500">{k}</span>
      <span className="text-ink-900 font-semibold text-right tabular">{v}</span>
    </div>
  );
}

type Finding = "normal" | "abnormal" | "borderline" | "none";

function agreement(a: Finding, b: Finding): { text: string; tone: string } {
  if (a === "none" && b === "none") {
    return { text: "Neither method could measure this photo. Retake it with the torch on, both eyes open and the face square to the camera.", tone: "border-ink-200 bg-ink-50 text-ink-700" };
  }
  if (a === "none" || b === "none") {
    const only = a === "none" ? "Method B" : "Method A";
    return { text: `Only ${only} could assess this photo, so there is nothing to cross-check it against.`, tone: "border-ink-200 bg-ink-50 text-ink-700" };
  }
  if (a === b) {
    return a === "normal"
      ? { text: "Both methods agree: no manifest deviation in this photo.", tone: "border-emerald-200 bg-emerald-50 text-emerald-800" }
      : { text: "Both methods agree: the eyes are not aligned in this photo.", tone: "border-red-200 bg-red-50 text-red-800" };
  }
  if (a === "borderline" || b === "borderline") {
    return { text: "One method is borderline. Read the eye picture and the numbers together before drawing a conclusion.", tone: "border-amber-200 bg-amber-50 text-amber-800" };
  }
  return { text: "The methods disagree. Check the marks in the eye picture: a misplaced pupil or reflex mark explains most disagreements.", tone: "border-amber-200 bg-amber-50 text-amber-800" };
}

function methodBFinding(al: AlignmentResult | undefined): Finding {
  if (!al || !al.available) return "none";
  return al.verdict === "ALIGNED" ? "normal" : al.verdict === "BORDERLINE" ? "borderline" : "abnormal";
}

export default function ImageReport({ report, patientName, patientAge, onNewImage, onFullReport }: {
  report: SuccessResponse | InconclusiveResponse;
  patientName: string;
  patientAge: number | null;
  onNewImage: () => void;
  onFullReport?: () => void;
}) {
  const ok = report.status === "SUCCESS" ? report : null;
  const inc = report.status === "INCONCLUSIVE" ? report : null;
  const al = report.alignment;
  const images = report.intermediate_images;
  const eyesImg = ok?.intermediate_images?.report_eyes ?? images?.module8_alignment ?? null;
  const flags: string[] = ok ? ok.technical.flags : inc ? inc.flags : [];
  const notes = qualityNotes(flags);

  const tier: UrgencyTier | null = ok ? ok.result.urgency_tier : null;
  const confidence = ok?.technical.confidence ?? null;
  const asymDeg = ok ? (ok.result.asymmetry_degrees ?? ok.result.deviation_degrees) : null;
  const asymPD = ok ? (ok.result.asymmetry_pd ?? toPD(asymDeg ?? 0)) : null;
  const findingA: Finding = !ok ? "none"
    : tier === "NORMAL" ? (flags.includes("borderline_asymmetry") ? "borderline" : "normal") : "abnormal";
  const agree = agreement(findingA, methodBFinding(al));

  const when = new Date(report.timestamp);
  const date = when.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" });
  const time = when.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "UTC" });

  const perEye = (eye: "right" | "left") => {
    if (!ok) return null;
    const t = ok.technical;
    const norm = eye === "left" ? t.left_displacement_norm : t.right_displacement_norm;
    const dir = eye === "left" ? t.left_direction : t.right_direction;
    return { norm, dir, pd: toPD(norm * IRIS_RADIUS_MM * DEG_PER_MM) };
  };

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-[820px] mx-auto px-5 py-6 print:p-0 print:max-w-none space-y-4 print:space-y-3 text-ink-900">

        {/* Header */}
        <header className="flex items-start justify-between gap-4 border-b-2 border-ink-900 pb-3 print:pb-2">
          <div>
            <p className="text-[10px] font-black tracking-[0.25em] text-ink-400 uppercase">BeanHealth</p>
            <h1 className="text-[22px] font-bold leading-tight">Squint photo analysis</h1>
            <p className="text-[12px] text-ink-500 mt-0.5">Single image · corneal light reflex + eye-corner alignment</p>
          </div>
          <div className="text-right text-[12px] text-ink-500 leading-relaxed shrink-0">
            <p className="font-semibold text-ink-800">{date}</p>
            <p>{time} UTC</p>
          </div>
        </header>

        <div className="grid grid-cols-3 gap-4">
          <Field label="Patient" value={patientName || "–"} />
          <Field label="Age" value={patientAge !== null ? String(patientAge) : "–"} />
          <Field label="Source" value="Uploaded photo" />
        </div>

        <p className="text-[11.5px] print:text-[10px] text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 print:py-1">
          Research / validation use. Saved photo, flash check relaxed. A screening aid, not a diagnosis.
        </p>

        {/* Eyes */}
        {eyesImg && (
          <figure className="break-inside-avoid">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={`data:image/jpeg;base64,${eyesImg}`} alt="Both eyes with the detected pupil and corneal reflex marked"
                 className="w-full print:w-[85%] print:mx-auto rounded-lg border border-ink-100 block" />
            <figcaption className="flex flex-wrap gap-x-4 gap-y-1 mt-1.5 text-[11px] text-ink-500">
              {ok ? (
                <>
                  <span><span className="font-semibold" style={{ color: "#64dc64" }}>◯</span> Iris edge</span>
                  <span><span className="font-semibold" style={{ color: "#1464eb" }}>+</span> Pupil centre</span>
                  <span><span className="font-semibold" style={{ color: "#ffb414" }}>◯</span> Corneal light reflex</span>
                </>
              ) : (
                <span>Eye corners (pink), corner-to-corner axis, pupil (blue)</span>
              )}
              <span className="ml-auto">Patient&apos;s right eye on the left of the picture</span>
            </figcaption>
          </figure>
        )}

        <div className="grid sm:grid-cols-2 print:grid-cols-2 gap-3">
          {/* Method A */}
          <Panel
            eyebrow="Method A · torch reflex"
            title="Corneal light reflex"
            badge={tier
              ? <Pill text={URGENCY_CONFIG[tier].label} tone={`${URGENCY_CONFIG[tier].bgColour} ${URGENCY_CONFIG[tier].borderColour} ${URGENCY_CONFIG[tier].colour}`} />
              : <Pill text="Not measured" tone={VERDICT.UNAVAILABLE.tone} />}
          >
            {ok && asymDeg !== null && asymPD !== null ? (
              <>
                <p className="text-[30px] font-bold leading-none tabular">
                  {asymPD.toFixed(1)} <span className="text-[15px] font-semibold text-ink-500">PD</span>
                </p>
                <p className="text-[12px] text-ink-500 mt-1">
                  {asymDeg.toFixed(1)}° between the two eyes&apos; reflexes
                </p>
                <p className="text-[13.5px] font-semibold mt-2.5">
                  {tier === "NORMAL" ? "No manifest deviation measured" : `${ok.result.condition_name} pattern`}
                  <span className="font-mono text-[11px] text-ink-400 font-normal ml-2">{ok.result.icd10_code}</span>
                </p>
                <div className="mt-2">
                  {(["right", "left"] as const).map((eye) => {
                    const e = perEye(eye);
                    return e && (
                      <Row key={eye} k={`${eye === "right" ? "Right eye (OD)" : "Left eye (OS)"} reflex`}
                           v={`${(e.norm * 100).toFixed(0)}% iris radius · ${e.dir}`} />
                    );
                  })}
                  <Row k="Confidence" v={confidence ?? "–"} />
                  <Row k="Action" v={ok.result.referral_recommendation} />
                </div>
              </>
            ) : (
              <p className="text-[12.5px] text-ink-600 leading-relaxed">
                {inc?.reason_human ?? "No reading."}
              </p>
            )}
          </Panel>

          {/* Method B */}
          <Panel
            eyebrow="Method B · no flash needed"
            title="Pupil between the eye corners"
            badge={<Pill text={VERDICT[al?.verdict ?? "UNAVAILABLE"].label} tone={VERDICT[al?.verdict ?? "UNAVAILABLE"].tone} />}
          >
            {al && al.available ? (
              <>
                {ok && images?.module8_alignment && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={`data:image/jpeg;base64,${images.module8_alignment}`}
                       alt="Each pupil projected onto its own eye-corner axis"
                       className="w-full rounded-lg border border-ink-100 block" />
                )}
                {ok && images?.module8_alignment && (
                  <p className="text-[10.5px] text-ink-400 mt-1 mb-2">Pink: eye corners · blue: pupil · amber: its place on the corner axis</p>
                )}
                <div className="grid grid-cols-2 gap-2 mb-2">
                  {(["right", "left"] as const).map((eye) => (
                    <div key={eye} className="rounded-lg bg-ink-50 border border-ink-100 px-3 py-2">
                      <p className="text-[10px] uppercase tracking-[0.12em] text-ink-400 font-semibold">
                        {eye === "right" ? "Right eye (OD)" : "Left eye (OS)"}
                      </p>
                      <p className="text-[18px] font-bold tabular leading-tight">
                        {pct(eye === "right" ? al.right_h_ratio : al.left_h_ratio)}
                      </p>
                      <p className="text-[10.5px] text-ink-400">inner → outer corner</p>
                    </div>
                  ))}
                </div>
                <Row k="Horizontal difference" v={pct(al.h_asymmetry)} />
                <Row k="Vertical difference" v={pct(al.v_asymmetry)} />
                <p className="text-[11.5px] text-ink-500 leading-relaxed mt-2 print:hidden">{al.interpretation}</p>
              </>
            ) : (
              <p className="text-[12.5px] text-ink-600 leading-relaxed">
                {al?.interpretation ?? "The eye corners could not be located in this photo."}
              </p>
            )}
          </Panel>
        </div>

        {/* Agreement */}
        <p className={`rounded-lg border px-3.5 py-2.5 text-[13px] font-semibold break-inside-avoid ${agree.tone}`}>
          {agree.text}
        </p>

        {/* Quality */}
        {notes.length > 0 && (
          <section className="break-inside-avoid">
            <h2 className="text-[10px] uppercase tracking-[0.14em] text-ink-400 font-semibold mb-1">Quality notes</h2>
            <ul className="list-disc pl-5 text-[12px] print:text-[10.5px] text-ink-600 space-y-0.5">
              {notes.map((n) => <li key={n}>{n}</li>)}
            </ul>
          </section>
        )}

        {/* Clinician comparison — filled in by hand on the printed copy */}
        <section className="rounded-xl border border-ink-200 p-4 print:p-3 break-inside-avoid">
          <h2 className="text-[10px] uppercase tracking-[0.14em] text-ink-400 font-semibold mb-3">
            Clinician reference (prism cover test)
          </h2>
          <div className="grid grid-cols-2 gap-x-6 gap-y-4 print:gap-y-2 text-[12.5px] print:text-[11.5px] text-ink-700">
            <p>Deviation: <span className="inline-block w-24 border-b border-ink-300 align-bottom" /> PD</p>
            <p>☐ Eso&nbsp;&nbsp; ☐ Exo&nbsp;&nbsp; ☐ Hyper / hypo&nbsp;&nbsp; ☐ None</p>
            <p>☐ Near&nbsp;&nbsp; ☐ Distance&nbsp;&nbsp; ☐ Constant&nbsp;&nbsp; ☐ Intermittent</p>
            <p>Fixing eye: ☐ R&nbsp;&nbsp; ☐ L&nbsp;&nbsp; ☐ Alternating</p>
            <p className="col-span-2">Examiner / notes: <span className="inline-block w-[70%] border-b border-ink-300 align-bottom" /></p>
          </div>
        </section>

        {/* Method and limits */}
        <footer className="text-[10.5px] print:text-[9px] text-ink-500 leading-relaxed border-t border-ink-200 pt-3 print:pt-2 break-inside-avoid">
          <p>
            <strong className="text-ink-700">Method and limits.</strong> Method A compares where the torch reflex sits
            against the pupil in each eye; comparing the two eyes cancels the normal (kappa) offset they share
            ({DEG_PER_MM}° per mm of reflex shift, iris radius {IRIS_RADIUS_MM} mm, PD = 100·tan θ). Method B places
            each pupil between its own eye corners and flags misalignment without an angle. A photo shows only the
            deviation present at that moment; a prism cover test measures the full deviation and is usually larger,
            and latent or intermittent squints can read near zero. Not reliable under about 8 PD or above about
            45 PD. Screening aid, not a diagnosis. <span className="text-ink-400">BeanHealth CLR · test mode · {report.timestamp}</span>
          </p>
        </footer>

        {/* Actions — screen only */}
        <div className="no-print print:hidden flex flex-col sm:flex-row gap-3 pt-2">
          <button onClick={() => window.print()} className="btn-primary flex-1">Print / save PDF</button>
          <button onClick={onNewImage} className="btn-secondary flex-1">Analyse another image</button>
          {onFullReport && (
            <button onClick={onFullReport} className="btn-secondary flex-1">Full technical report</button>
          )}
        </div>
      </div>
    </div>
  );
}
