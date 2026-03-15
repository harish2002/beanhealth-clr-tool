"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/useAppStore";

export default function PatientFormPage() {
  const router     = useRouter();
  const setPatient = useAppStore((s) => s.setPatient);
  const reset      = useAppStore((s) => s.reset);

  const [name,  setName]  = useState("");
  const [age,   setAge]   = useState("");
  const [error, setError] = useState<string | null>(null);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    const trimmedName = name.trim();
    const parsedAge   = parseInt(age, 10);

    if (!trimmedName) {
      setError("Please enter the patient's name.");
      return;
    }
    if (isNaN(parsedAge) || parsedAge < 1 || parsedAge > 120) {
      setError("Please enter a valid age between 1 and 120.");
      return;
    }

    reset();
    setPatient(trimmedName, parsedAge);
    router.push("/capture");
  }

  return (
    <main className="min-h-screen flex flex-col bg-slate-50">

      {/* ── Top header bar ────────────────────────────────────────────────── */}
      <header className="bg-white border-b border-slate-200 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center shadow-sm shadow-blue-600/30">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor">
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

      {/* ── Body ──────────────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col items-center justify-center px-5 py-10">
        <div className="w-full max-w-5xl">

          {/* Hero */}
          <div className="text-center mb-8">
            <h1 className="text-2xl md:text-3xl font-bold text-slate-900 tracking-tight">
              Eye Alignment Check
            </h1>
            <p className="text-slate-500 text-sm md:text-base mt-2 leading-relaxed max-w-md mx-auto">
              Quick corneal light reflex screening for strabismus triage — no specialist needed.
            </p>
          </div>

          {/* Progress steps */}
          <div className="flex items-stretch gap-0 max-w-sm md:max-w-md mx-auto mb-8">
            {[
              { n: "1", label: "Patient info",  icon: "👤" },
              { n: "2", label: "Capture photo", icon: "📷" },
              { n: "3", label: "View results",  icon: "📋" },
            ].map((step, i) => (
              <div key={step.n} className="flex-1 flex flex-col items-center">
                <div className="flex items-center w-full">
                  {i > 0 && <div className="flex-1 h-0.5 bg-blue-200" />}
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shrink-0
                    ${i === 0 ? "bg-blue-600 text-white ring-4 ring-blue-100" : "bg-slate-200 text-slate-500"}`}>
                    {step.n}
                  </div>
                  {i < 2 && <div className="flex-1 h-0.5 bg-blue-200" />}
                </div>
                <p className={`text-xs mt-2 font-medium ${i === 0 ? "text-blue-700" : "text-slate-400"}`}>
                  {step.label}
                </p>
              </div>
            ))}
          </div>

          {/* Two-column on md+: form left, tips right */}
          <div className="flex flex-col md:flex-row md:items-start md:gap-6 max-w-2xl mx-auto">

            {/* Patient form card */}
            <div className="flex-1 bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
              <h2 className="text-base font-semibold text-slate-900 mb-0.5">Patient Details</h2>
              <p className="text-slate-400 text-xs mb-5">Enter details before capturing the photo.</p>

              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1.5" htmlFor="name">
                    Patient Name
                  </label>
                  <input
                    id="name"
                    type="text"
                    inputMode="text"
                    autoComplete="name"
                    placeholder="e.g. Emma Wilson"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3
                               text-slate-900 placeholder-slate-400 text-base
                               focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                               transition"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1.5" htmlFor="age">
                    Age (years)
                  </label>
                  <input
                    id="age"
                    type="number"
                    inputMode="numeric"
                    min={1}
                    max={120}
                    placeholder="e.g. 5"
                    value={age}
                    onChange={(e) => setAge(e.target.value)}
                    className="w-full rounded-xl border border-slate-200 bg-slate-50 px-4 py-3
                               text-slate-900 placeholder-slate-400 text-base
                               focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                               transition"
                  />
                </div>

                {error && (
                  <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-xl px-3 py-2.5">
                    <svg className="w-4 h-4 text-red-500 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round"
                        d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
                    </svg>
                    <p className="text-red-700 text-sm">{error}</p>
                  </div>
                )}

                <button
                  type="submit"
                  className="w-full rounded-xl bg-blue-600 hover:bg-blue-700 active:bg-blue-800
                             text-white font-semibold py-3.5 text-base transition-colors
                             focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2
                             shadow-md shadow-blue-600/25 mt-2"
                >
                  Continue to Camera →
                </button>
              </form>
            </div>

            {/* Tips — beside the form on md+ */}
            <div className="mt-4 md:mt-0 md:w-64 shrink-0 space-y-4">
              <div className="bg-amber-50 border border-amber-200 rounded-2xl px-4 py-4">
                <p className="text-xs font-semibold text-amber-800 mb-2">📸 Before you start</p>
                <ul className="space-y-2 text-xs text-amber-700 leading-relaxed">
                  <li>· Use the <strong>back camera</strong> — torch enables automatically</li>
                  <li>· Hold the phone <strong>30–40 cm</strong> from the patient&apos;s face</li>
                  <li>· Ensure <strong>both eyes are fully open</strong></li>
                  <li>· A second person can hold the phone for a better angle</li>
                </ul>
              </div>

              {/* What it detects panel */}
              <div className="bg-blue-50 border border-blue-200 rounded-2xl px-4 py-4">
                <p className="text-xs font-semibold text-blue-800 mb-2">🔬 What we screen for</p>
                <ul className="space-y-2 text-xs text-blue-700 leading-relaxed">
                  <li>· <strong>Esotropia</strong> — inward eye turn</li>
                  <li>· <strong>Exotropia</strong> — outward eye turn</li>
                  <li>· <strong>Hypertropia</strong> — vertical misalignment</li>
                  <li>· Uses the Hirschberg corneal reflex method (7°/mm)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="py-4 text-center">
        <p className="text-slate-400 text-xs">
          Screening aid only · Not a medical diagnosis · EyeQ Innovate Hackathon 2.0
        </p>
      </footer>
    </main>
  );
}
