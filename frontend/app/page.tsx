"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/useAppStore";

const SCREENER_ROLES = [
  "Parent / Guardian",
  "ASHA Worker",
  "Nurse / ANM",
  "General Physician",
  "Ophthalmologist",
  "Other",
];

const TRIAGE_TIERS = [
  { label: "Urgent referral",   range: "≥ 30°",  dot: "bg-red-500"     },
  { label: "Routine referral",  range: "15–30°", dot: "bg-orange-500"  },
  { label: "Monitor",           range: "5–15°",  dot: "bg-amber-500"   },
  { label: "Normal",            range: "< 5°",   dot: "bg-emerald-500" },
  { label: "Inconclusive",      range: "re-capture", dot: "bg-slate-400" },
];

const ICD_CODES = [
  { name: "Esotropia",                code: "H50.01"    },
  { name: "Exotropia",                code: "H50.11"    },
  { name: "Hypertropia / Hypotropia", code: "H50.21–22" },
  { name: "Amblyopia (risk)",         code: "H53.0"     },
];

const WORKFLOW_STEPS = [
  {
    step:  "01",
    title: "Patient details",
    body:  "Capture identifiers and screener context. A session ID is generated for the audit trail.",
  },
  {
    step:  "02",
    title: "Torch capture",
    body:  "Rear camera with torch, 30–40 cm. Live quality meter gates capture until distance, pose, openness, and torch reflex are all in range.",
  },
  {
    step:  "03",
    title: "Triage report",
    body:  "10-frame bilateral asymmetry, ICD-10 code, severity in degrees and prism dioptres, urgency tier, and referral letter.",
  },
];

export default function LandingPage() {
  const router          = useRouter();
  const setPatient      = useAppStore((s) => s.setPatient);
  const setPatientMeta  = useAppStore((s) => s.setPatientMeta);
  const reset           = useAppStore((s) => s.reset);

  const [name,     setName]     = useState("");
  const [age,      setAge]      = useState("");
  const [gender,   setGender]   = useState<"" | "Male" | "Female" | "Other">("");
  const [role,     setRole]     = useState("");
  const [location, setLocation] = useState("");
  const [error,    setError]    = useState<string | null>(null);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    const trimmedName = name.trim();
    const parsedAge   = parseInt(age, 10);

    if (!trimmedName) {
      setError("Patient name is required.");
      return;
    }
    if (isNaN(parsedAge) || parsedAge < 1 || parsedAge > 120) {
      setError("Enter a valid age between 1 and 120.");
      return;
    }

    reset();
    setPatient(trimmedName, parsedAge);
    setPatientMeta(gender, role, location.trim());
    router.push("/capture");
  }

  return (
    <main className="min-h-screen bg-white text-slate-900">

      {/* ════════════════════════════════════════════════════════════
          TOP NAV
      ════════════════════════════════════════════════════════════ */}
      <nav className="border-b border-slate-200 bg-white sticky top-0 z-20 backdrop-blur supports-[backdrop-filter]:bg-white/85">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">

          {/* Brand */}
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center shadow-sm shadow-blue-600/20">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
              </svg>
            </div>
            <div className="leading-none">
              <p className="text-sm font-bold text-slate-900">BeanHealth CLR</p>
              <p className="text-[10px] text-slate-400 mt-0.5">Pediatric Strabismus Screening</p>
            </div>
          </div>

          {/* Credibility badges */}
          <div className="hidden sm:flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 text-[10px] font-semibold tracking-widest text-slate-500 uppercase
                             border border-slate-200 rounded-full px-2.5 py-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              CDSCO Class A SaMD
            </span>
            <span className="hidden md:inline-flex items-center text-[10px] font-semibold tracking-widest text-slate-500 uppercase
                             border border-slate-200 rounded-full px-2.5 py-1">
              LVPEI BIONEST research partner
            </span>
          </div>
        </div>
      </nav>

      {/* ════════════════════════════════════════════════════════════
          HERO + FORM
      ════════════════════════════════════════════════════════════ */}
      <section className="relative overflow-hidden bg-gradient-to-b from-white via-white to-slate-50/60 border-b border-slate-100">

        {/* Subtle background grid for texture */}
        <div
          aria-hidden
          className="absolute inset-0 pointer-events-none opacity-[0.025]"
          style={{
            backgroundImage:
              "linear-gradient(to right, #0f172a 1px, transparent 1px), linear-gradient(to bottom, #0f172a 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />

        <div className="relative max-w-6xl mx-auto px-6 py-12 md:py-20 grid md:grid-cols-5 gap-10 md:gap-14">

          {/* ── LEFT: copy + form ────────────────────────────────────── */}
          <div className="md:col-span-3">
            <p className="text-[11px] font-bold uppercase tracking-[0.22em] text-blue-600 mb-3">
              New patient screening
            </p>
            <h1 className="text-3xl md:text-[42px] font-bold text-slate-900 tracking-tight leading-[1.1]">
              Pediatric strabismus screening<br className="hidden md:block" />
              <span className="text-blue-600">in under 30 seconds.</span>
            </h1>
            <p className="text-slate-500 text-base md:text-[17px] mt-4 leading-relaxed max-w-xl">
              Smartphone-based bilateral corneal light reflex analysis with
              multi-frame stability gating. Designed for parents, ASHA workers,
              school nurses, and primary-care physicians — no specialist
              required, no hardware purchase, no clinic visit unless flagged.
            </p>

            {/* Inline form card */}
            <div className="mt-8 bg-white rounded-2xl border border-slate-200 shadow-sm">
              <div className="border-b border-slate-100 px-6 py-4 flex items-center justify-between">
                <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-slate-400">
                  Patient details
                </p>
                <p className="text-[10px] text-slate-400 font-medium">
                  Step 1 of 3
                </p>
              </div>

              <form onSubmit={handleSubmit} className="px-6 py-6 space-y-5">

                {/* Name + Age — two columns on md+ */}
                <div className="grid sm:grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="name" className="block text-xs font-semibold text-slate-700 mb-1.5">
                      Patient name <span className="text-red-500">*</span>
                    </label>
                    <input
                      id="name"
                      type="text"
                      autoComplete="name"
                      placeholder="e.g. Aarav Sharma"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      className="w-full rounded-lg border border-slate-300 bg-white px-3.5 py-2.5
                                 text-slate-900 placeholder-slate-400 text-[15px]
                                 focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500
                                 transition"
                    />
                  </div>
                  <div>
                    <label htmlFor="age" className="block text-xs font-semibold text-slate-700 mb-1.5">
                      Age <span className="text-red-500">*</span>
                    </label>
                    <input
                      id="age"
                      type="number"
                      inputMode="numeric"
                      min={1}
                      max={120}
                      placeholder="years"
                      value={age}
                      onChange={(e) => setAge(e.target.value)}
                      className="w-full rounded-lg border border-slate-300 bg-white px-3.5 py-2.5
                                 text-slate-900 placeholder-slate-400 text-[15px]
                                 focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500
                                 transition"
                    />
                  </div>
                </div>

                {/* Gender */}
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                    Sex <span className="text-slate-400 font-normal">(optional)</span>
                  </label>
                  <div className="grid grid-cols-3 gap-2">
                    {(["Male", "Female", "Other"] as const).map((g) => (
                      <button
                        key={g}
                        type="button"
                        onClick={() => setGender(gender === g ? "" : g)}
                        className={`py-2 rounded-lg border text-sm font-medium transition-colors ${
                          gender === g
                            ? "bg-blue-600 border-blue-600 text-white shadow-sm shadow-blue-600/20"
                            : "bg-white border-slate-300 text-slate-600 hover:border-slate-400"
                        }`}
                      >
                        {g}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Screener role */}
                <div>
                  <label htmlFor="role" className="block text-xs font-semibold text-slate-700 mb-1.5">
                    Screener role <span className="text-slate-400 font-normal">(optional)</span>
                  </label>
                  <select
                    id="role"
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    className="w-full rounded-lg border border-slate-300 bg-white px-3.5 py-2.5
                               text-slate-900 text-[15px]
                               focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500
                               transition"
                  >
                    <option value="">Not specified</option>
                    {SCREENER_ROLES.map((r) => (
                      <option key={r} value={r}>{r}</option>
                    ))}
                  </select>
                </div>

                {/* Screening location */}
                <div>
                  <label htmlFor="location" className="block text-xs font-semibold text-slate-700 mb-1.5">
                    Screening location <span className="text-slate-400 font-normal">(optional)</span>
                  </label>
                  <input
                    id="location"
                    type="text"
                    placeholder="e.g. PHC Bangalore, Home, School"
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                    className="w-full rounded-lg border border-slate-300 bg-white px-3.5 py-2.5
                               text-slate-900 placeholder-slate-400 text-[15px]
                               focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500
                               transition"
                  />
                </div>

                {/* Error */}
                {error && (
                  <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg px-3 py-2.5">
                    <svg className="w-4 h-4 text-red-500 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round"
                        d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
                    </svg>
                    <p className="text-red-700 text-sm">{error}</p>
                  </div>
                )}

                {/* Submit */}
                <div className="pt-2">
                  <button
                    type="submit"
                    className="w-full rounded-lg bg-slate-900 hover:bg-slate-800 active:bg-black
                               text-white font-semibold py-3 text-[15px] transition-colors
                               focus:outline-none focus:ring-2 focus:ring-slate-900/40 focus:ring-offset-2
                               shadow-sm flex items-center justify-center gap-2"
                  >
                    <span>Continue to capture</span>
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2.2} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
                    </svg>
                  </button>
                  <p className="text-[11px] text-slate-400 text-center mt-3 leading-relaxed">
                    No patient images are stored on our servers beyond the screening session.
                    Screening aid only — not a diagnostic device.
                  </p>
                </div>
              </form>
            </div>
          </div>

          {/* ── RIGHT: clinical sidebar ──────────────────────────────── */}
          <aside className="md:col-span-2 space-y-4">

            {/* What it detects (ICD-10) */}
            <div className="rounded-2xl border border-slate-200 bg-white p-6">
              <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-slate-400 mb-3">
                What it screens for
              </p>
              <dl className="space-y-2.5">
                {ICD_CODES.map(({ name, code }) => (
                  <div key={code} className="flex items-baseline justify-between gap-3 text-sm">
                    <dt className="text-slate-700">{name}</dt>
                    <dd className="font-mono text-[11px] text-slate-500 bg-slate-50 border border-slate-200 px-1.5 py-0.5 rounded">
                      {code}
                    </dd>
                  </div>
                ))}
              </dl>
            </div>

            {/* Output tiers */}
            <div className="rounded-2xl border border-slate-200 bg-white p-6">
              <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-slate-400 mb-3">
                Triage output
              </p>
              <ul className="space-y-2.5">
                {TRIAGE_TIERS.map(({ label, range, dot }) => (
                  <li key={label} className="flex items-center gap-3 text-sm">
                    <span className={`w-2 h-2 rounded-full shrink-0 ${dot}`} />
                    <span className="text-slate-700 flex-1">{label}</span>
                    <span className="text-[11px] text-slate-400 font-mono">{range}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Method */}
            <div className="rounded-2xl border border-slate-200 bg-white p-6">
              <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-slate-400 mb-3">
                Method
              </p>
              <p className="text-sm text-slate-600 leading-relaxed">
                Bilateral corneal light reflex (Hirschberg). 10-frame asymmetry
                vector with inter-frame variance gating, per-device calibration,
                and PACT-validated triage thresholds.
              </p>
              <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
                <span>Hirschberg ratio</span>
                <span className="font-mono text-slate-600">22 PD / mm</span>
              </div>
            </div>
          </aside>

        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════
          HOW IT WORKS
      ════════════════════════════════════════════════════════════ */}
      <section className="bg-white border-b border-slate-100">
        <div className="max-w-6xl mx-auto px-6 py-14 md:py-20">

          <div className="max-w-2xl mb-10">
            <p className="text-[11px] font-bold uppercase tracking-[0.22em] text-blue-600 mb-3">
              Capture workflow
            </p>
            <h2 className="text-2xl md:text-3xl font-bold text-slate-900 tracking-tight">
              Three steps from capture to referral
            </h2>
            <p className="text-slate-500 text-base mt-3 leading-relaxed">
              The platform separates capture, analysis, and triage into a single
              linear flow with quality gating at every stage. INCONCLUSIVE is a
              first-class output — the system refuses to score when conditions
              cannot support a confident measurement.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-5">
            {WORKFLOW_STEPS.map(({ step, title, body }) => (
              <div
                key={step}
                className="rounded-2xl border border-slate-200 bg-white p-6 hover:border-slate-300 transition-colors"
              >
                <p className="text-[10px] font-bold tracking-[0.22em] text-blue-600">
                  STEP {step}
                </p>
                <h3 className="text-lg font-bold text-slate-900 mt-1.5">{title}</h3>
                <p className="text-sm text-slate-500 leading-relaxed mt-2">{body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════
          REGULATORY / TRUST STRIP
      ════════════════════════════════════════════════════════════ */}
      <section className="bg-slate-50 border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-6 py-10">
          <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { stat: "≥ 85% / ≥ 85%",     label: "Sens / Spec target vs PACT" },
              { stat: "300–600 frames",     label: "Per session" },
              { stat: "5 tiers",            label: "ICD-10 mapped output" },
              { stat: "Closed-loop",        label: "Tele-consult referral" },
            ].map((m) => (
              <div key={m.label}>
                <p className="text-2xl md:text-3xl font-bold text-slate-900 tracking-tight tabular-nums">
                  {m.stat}
                </p>
                <p className="text-xs text-slate-500 mt-1 leading-snug">{m.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════
          FOOTER
      ════════════════════════════════════════════════════════════ */}
      <footer className="bg-white">
        <div className="max-w-6xl mx-auto px-6 py-8 flex flex-col md:flex-row gap-4 justify-between items-start md:items-center">

          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-md bg-slate-900 flex items-center justify-center">
              <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
              </svg>
            </div>
            <div className="text-xs text-slate-500 leading-snug">
              <p className="font-semibold text-slate-700">BeanHealth Private Limited</p>
              <p>DPIIT-recognised · CDSCO Class A SaMD (pre-registration) · Coimbatore</p>
            </div>
          </div>

          <p className="text-[11px] text-slate-400 leading-relaxed md:text-right max-w-md">
            Screening aid only. Not a diagnostic device. Results must be confirmed
            by a qualified ophthalmologist before any clinical action is taken.
          </p>
        </div>
      </footer>
    </main>
  );
}
