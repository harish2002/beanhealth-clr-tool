"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAppStore } from "@/store/useAppStore";
import { TopBar, SiteFooter, ArrowRight } from "@/components/ui/Chrome";

/**
 * Step 1 of the screening flow — patient intake.
 *
 * Writes name/age (required) and optional metadata into the store, mints a
 * session ID, then hands off to /capture. The capture route guards on
 * patientName + patientAge, so this page is the only legitimate entry point.
 */

const SCREENER_ROLES = [
  "Parent / Guardian",
  "ASHA Worker",
  "Nurse / ANM",
  "General Physician",
  "Ophthalmologist",
  "Other",
];

const PREP_CHECKLIST = [
  {
    title: "Find a dim room",
    body:  "Bright ambient light washes out the corneal reflex. Curtains drawn is ideal.",
  },
  {
    title: "Have the torch ready",
    body:  "The next screen turns on the rear torch automatically. Hold the phone 30–40 cm from the face.",
  },
  {
    title: "Get their attention",
    body:  "The child must look straight at the lens for about four seconds. An animated target appears on screen to help.",
  },
  {
    title: "Remove glasses",
    body:  "Lenses create a second bright spot that competes with the true corneal reflex.",
  },
];

export default function PatientPage() {
  const router         = useRouter();
  const setPatient     = useAppStore((s) => s.setPatient);
  const setPatientMeta = useAppStore((s) => s.setPatientMeta);
  const reset          = useAppStore((s) => s.reset);

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
    <main className="min-h-screen bg-white text-ink-900 flex flex-col">
      <TopBar step={1} />

      <div className="flex-1 border-b border-ink-100">
        <div className="max-w-5xl mx-auto px-5 sm:px-6 py-10 md:py-16
                        grid lg:grid-cols-12 gap-10 lg:gap-14 items-start">

          {/* ── Form ─────────────────────────────────────────────── */}
          <div className="lg:col-span-7 animate-rise-in">

            <Link
              href="/"
              className="inline-flex items-center gap-1.5 text-[13px] text-ink-400
                         hover:text-ink-800 transition-colors mb-6"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" strokeWidth={2.2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5 8.25 12l7.5-7.5" />
              </svg>
              Back to overview
            </Link>

            <p className="eyebrow mb-3">Step 1 of 3</p>
            <h1 className="display-lg">Who are we screening?</h1>
            <p className="lede mt-4 max-w-lg">
              These details appear on the triage report and the referral letter.
              Only name and age are required — everything else sharpens the audit trail.
            </p>

            <form onSubmit={handleSubmit} className="mt-9 surface-raised p-6 sm:p-8 space-y-6">

              {/* Name + Age */}
              <div className="grid sm:grid-cols-3 gap-4">
                <div className="sm:col-span-2">
                  <label htmlFor="name" className="field-label">
                    Patient name <span className="text-red-500">*</span>
                  </label>
                  <input
                    id="name"
                    type="text"
                    autoComplete="name"
                    autoFocus
                    placeholder="e.g. Aarav Sharma"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="field"
                  />
                </div>
                <div>
                  <label htmlFor="age" className="field-label">
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
                    className="field tabular"
                  />
                </div>
              </div>

              <div className="rule" />

              {/* Sex */}
              <div>
                <label className="field-label">
                  Sex <span className="field-hint">optional</span>
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {(["Male", "Female", "Other"] as const).map((g) => (
                    <button
                      key={g}
                      type="button"
                      aria-pressed={gender === g}
                      onClick={() => setGender(gender === g ? "" : g)}
                      className={gender === g ? "segment-on" : "segment-off"}
                    >
                      {g}
                    </button>
                  ))}
                </div>
              </div>

              {/* Screener role */}
              <div>
                <label htmlFor="role" className="field-label">
                  Screener role <span className="field-hint">optional</span>
                </label>
                <select
                  id="role"
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  className="field"
                >
                  <option value="">Not specified</option>
                  {SCREENER_ROLES.map((r) => (
                    <option key={r} value={r}>{r}</option>
                  ))}
                </select>
              </div>

              {/* Location */}
              <div>
                <label htmlFor="location" className="field-label">
                  Screening location <span className="field-hint">optional</span>
                </label>
                <input
                  id="location"
                  type="text"
                  placeholder="e.g. PHC Bangalore, Home, School"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  className="field"
                />
              </div>

              {/* Error */}
              {error && (
                <div
                  role="alert"
                  className="flex items-start gap-2.5 bg-red-50 border border-red-200 rounded-field px-3.5 py-3 animate-fade-in"
                >
                  <svg className="w-4 h-4 text-red-500 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round"
                      d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
                  </svg>
                  <p className="text-red-700 text-[13.5px]">{error}</p>
                </div>
              )}

              {/* Submit */}
              <div className="pt-1">
                <button type="submit" className="btn-primary w-full">
                  Continue to capture
                  <ArrowRight />
                </button>
                <p className="text-[11.5px] text-ink-400 text-center mt-4 leading-relaxed">
                  No patient images are stored on our servers beyond the screening session.
                  Screening aid only — not a diagnostic device.
                </p>
              </div>
            </form>
          </div>

          {/* ── Prep checklist ───────────────────────────────────── */}
          <aside className="lg:col-span-5 lg:sticky lg:top-24 animate-fade-in">
            <div className="surface p-6 sm:p-7">
              <p className="eyebrow-muted mb-1">Before you continue</p>
              <h2 className="font-display text-[22px] text-ink-900 leading-snug mb-5">
                Four things that decide whether the reading works.
              </h2>
              <ol className="divide-y divide-ink-100">
                {PREP_CHECKLIST.map(({ title, body }, i) => (
                  <li key={title} className="flex gap-4 py-4 first:pt-0 last:pb-0">
                    <span className="font-mono text-[11px] text-clinical-600 tabular pt-0.5 shrink-0">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <div>
                      <p className="text-[14px] font-semibold text-ink-800">{title}</p>
                      <p className="text-[13px] text-ink-500 leading-relaxed mt-1">{body}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </div>

            <p className="text-[12px] text-ink-400 leading-relaxed mt-5 px-1">
              If conditions are not good enough, the capture screen will hold and tell you
              why rather than producing an unreliable score.
            </p>
          </aside>
        </div>
      </div>

      <SiteFooter />
    </main>
  );
}
