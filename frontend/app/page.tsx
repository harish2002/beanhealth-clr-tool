"use client";

import Link from "next/link";
import BatchScreening from "@/components/BatchScreening";
import HirschbergDiagram from "@/components/ui/HirschbergDiagram";
import {
  TopBar,
  SiteFooter,
  SectionHeading,
  ArrowRight,
} from "@/components/ui/Chrome";

// ─── Content ──────────────────────────────────────────────────────────────────

const CONDITIONS = [
  { name: "Esotropia",   note: "eye turns inward",   code: "H50.01"    },
  { name: "Exotropia",   note: "eye turns outward",  code: "H50.11"    },
  { name: "Hypertropia", note: "eye turns upward",   code: "H50.21"    },
  { name: "Hypotropia",  note: "eye turns downward", code: "H50.22"    },
  { name: "Amblyopia",   note: "risk indicator",     code: "H53.0"     },
];

const TRIAGE_TIERS = [
  { label: "Urgent referral",  range: "≥ 30°",      action: "See ophthalmology within 1 week",  dot: "bg-red-500"     },
  { label: "Routine referral", range: "15 – 30°",   action: "See ophthalmology within 4 weeks", dot: "bg-orange-500"  },
  { label: "Monitor",          range: "5 – 15°",    action: "Re-screen in 3 months",            dot: "bg-amber-400"   },
  { label: "Normal",           range: "< 5°",       action: "No referral required",             dot: "bg-emerald-500" },
  { label: "Inconclusive",     range: "no reading", action: "Re-capture under better light",    dot: "bg-ink-300"     },
];

const WORKFLOW_STEPS = [
  {
    step:  "01",
    title: "Patient details",
    body:  "Name, age, and screener context. A session ID is generated so every reading has an audit trail.",
    meta:  "~20 seconds",
  },
  {
    step:  "02",
    title: "Torch capture",
    body:  "Rear camera with the torch on, held 30–40 cm away. A live quality meter holds capture until distance, head pose, eye openness, and the torch reflex are all in range.",
    meta:  "8 frames · ~30 seconds",
  },
  {
    step:  "03",
    title: "Triage report",
    body:  "Bilateral asymmetry in degrees and prism dioptres, an ICD-10 code, an urgency tier, and a printable referral letter.",
    meta:  "Instant",
  },
];

const AUDIENCE = [
  "Parents and guardians",
  "ASHA workers and ANMs",
  "School and community nurses",
  "General physicians",
];

const PRINCIPLES = [
  {
    title: "No specialist required",
    body:  "The measurement an ophthalmologist makes by eye with a pen torch, made instead by the phone already in your pocket.",
  },
  {
    title: "Refuses to guess",
    body:  "INCONCLUSIVE is a first-class result. When the torch reflex, head pose, or frame stability cannot support a confident reading, the tool says so rather than scoring.",
  },
  {
    title: "Nothing is stored",
    body:  "Patient images live only for the duration of the request. No image leaves the screening session, and nothing is retained on our servers.",
  },
];

const STATS = [
  { stat: "≥ 85%",       label: "Sensitivity / specificity target, benchmarked against PACT" },
  { stat: "8 frames",    label: "Per session, IQR-filtered with inter-frame variance gating" },
  { stat: "5 tiers",     label: "Triage outputs, each mapped to an ICD-10 code" },
  { stat: "7° / mm",     label: "Hirschberg conversion ratio, the clinical standard" },
];

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-white text-ink-900">
      <TopBar
        right={
          <Link href="/patient" className="btn-primary btn-sm">
            Start screening
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        }
      />

      {/* ═══════════════════════════════════════════════════════════
          HERO — what this is
      ═══════════════════════════════════════════════════════════ */}
      <section className="relative overflow-hidden border-b border-ink-100">
        <div aria-hidden className="absolute inset-0 bg-grid opacity-40 pointer-events-none" />
        <div
          aria-hidden
          className="absolute inset-x-0 bottom-0 h-48 bg-gradient-to-t from-white to-transparent pointer-events-none"
        />

        <div className="relative max-w-6xl mx-auto px-5 sm:px-6 py-16 md:py-24
                        grid md:grid-cols-12 gap-12 md:gap-10 items-center">

          {/* Copy */}
          <div className="md:col-span-7 animate-rise-in">
            <p className="eyebrow mb-4">Corneal light reflex analysis</p>

            <h1 className="display-xl">
              Catch a child&rsquo;s squint
              <br className="hidden sm:block" />
              <span className="italic text-clinical-600"> before it costs them sight.</span>
            </h1>

            <p className="lede mt-6 max-w-xl">
              Strabismus is treatable when it is found early and permanently
              sight-limiting when it is not. BeanHealth CLR performs the
              Hirschberg corneal light reflex test — the same measurement an
              ophthalmologist makes with a pen torch — using an ordinary
              smartphone camera, and returns a referral decision in about
              thirty seconds.
            </p>

            <div className="mt-9 flex flex-col sm:flex-row gap-3">
              <Link href="/patient" className="btn-primary">
                Start screening
                <ArrowRight />
              </Link>
              <a href="#method" className="btn-secondary">
                How the measurement works
              </a>
            </div>

            <div className="mt-10 pt-8 border-t border-ink-100">
              <p className="eyebrow-muted mb-3">Built for</p>
              <ul className="flex flex-wrap gap-x-5 gap-y-2">
                {AUDIENCE.map((a) => (
                  <li key={a} className="flex items-center gap-2 text-[13.5px] text-ink-600">
                    <span className="w-1 h-1 rounded-full bg-clinical-600 shrink-0" />
                    {a}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Diagram */}
          <div className="md:col-span-5 animate-fade-in">
            <figure className="surface-raised p-6 sm:p-7">
              <figcaption className="flex items-baseline justify-between gap-3 mb-2">
                <span className="eyebrow-muted">The signal</span>
                <span className="code-chip">Hirschberg</span>
              </figcaption>
              <HirschbergDiagram className="w-full h-auto" />
              <p className="text-[12.5px] text-ink-500 leading-relaxed mt-3 pt-4 border-t border-ink-100">
                A torch held at arm&rsquo;s length casts a bright pinpoint on each
                cornea. In a straight eye that pinpoint lands on the pupil centre.
                In a deviating eye it sits off-centre — and how far off-centre it
                sits is the angle of deviation.
              </p>
            </figure>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          PRINCIPLES — how the tool behaves
      ═══════════════════════════════════════════════════════════ */}
      <section className="border-b border-ink-100 bg-white">
        <div className="max-w-6xl mx-auto px-5 sm:px-6 py-14 md:py-20">
          <div className="grid md:grid-cols-3 gap-x-10 gap-y-9 stagger">
            {PRINCIPLES.map(({ title, body }) => (
              <div key={title}>
                <h3 className="font-display text-[21px] text-ink-900 leading-snug">{title}</h3>
                <p className="text-[14px] text-ink-500 leading-relaxed mt-2.5">{body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          METHOD — the screening we do
      ═══════════════════════════════════════════════════════════ */}
      <section id="method" className="border-b border-ink-100 bg-ink-50/60 scroll-mt-16">
        <div className="max-w-6xl mx-auto px-5 sm:px-6 py-14 md:py-20">

          <SectionHeading
            eyebrow="The screening we do"
            title={<>One measurement, taken carefully, repeated eight times.</>}
            body={
              <>
                Every session runs the same pipeline: locate both irises, cross-validate
                the pupil centre with two independent methods, isolate the torch reflex
                from the brightest 3% of pixels, then measure the vector between them and
                normalise it by iris radius so the reading is independent of how far away
                the phone was held.
              </>
            }
          />

          <div className="mt-12 grid lg:grid-cols-2 gap-6 items-start">

            {/* What it screens for */}
            <div className="surface p-6 sm:p-7">
              <div className="flex items-baseline justify-between mb-5">
                <p className="eyebrow-muted">What it screens for</p>
                <p className="text-[11px] text-ink-400">ICD-10</p>
              </div>
              <dl className="divide-y divide-ink-100">
                {CONDITIONS.map(({ name, note, code }) => (
                  <div key={code} className="flex items-baseline justify-between gap-4 py-3 first:pt-0 last:pb-0">
                    <dt className="text-[14.5px] text-ink-800">
                      {name}
                      <span className="text-ink-400 font-normal"> — {note}</span>
                    </dt>
                    <dd className="code-chip shrink-0">{code}</dd>
                  </div>
                ))}
              </dl>
            </div>

            {/* Triage output */}
            <div className="surface p-6 sm:p-7">
              <div className="flex items-baseline justify-between mb-5">
                <p className="eyebrow-muted">What comes out</p>
                <p className="text-[11px] text-ink-400">Deviation</p>
              </div>
              <ul className="divide-y divide-ink-100">
                {TRIAGE_TIERS.map(({ label, range, action, dot }) => (
                  <li key={label} className="flex items-start gap-3 py-3 first:pt-0 last:pb-0">
                    <span className={`w-2 h-2 rounded-full shrink-0 mt-1.5 ${dot}`} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline justify-between gap-3">
                        <span className="text-[14.5px] text-ink-800 font-medium">{label}</span>
                        <span className="font-mono text-[11.5px] text-ink-400 tabular shrink-0">{range}</span>
                      </div>
                      <p className="text-[12.5px] text-ink-400 mt-0.5">{action}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          WORKFLOW — three steps
      ═══════════════════════════════════════════════════════════ */}
      <section className="border-b border-ink-100 bg-white">
        <div className="max-w-6xl mx-auto px-5 sm:px-6 py-14 md:py-20">

          <SectionHeading
            eyebrow="Capture workflow"
            title="Three steps from capture to referral."
            body="Capture, analysis, and triage stay separate, with a quality gate between each one. Nothing is scored on a frame the system does not trust."
          />

          <ol className="mt-12 grid md:grid-cols-3 gap-px bg-ink-100 border border-ink-100 rounded-card overflow-hidden">
            {WORKFLOW_STEPS.map(({ step, title, body, meta }) => (
              <li key={step} className="bg-white p-7 flex flex-col">
                <div className="flex items-baseline justify-between mb-4">
                  <span className="step-index">STEP {step}</span>
                  <span className="text-[11px] text-ink-400 font-mono">{meta}</span>
                </div>
                <h3 className="font-display text-[22px] text-ink-900 leading-snug">{title}</h3>
                <p className="text-[13.5px] text-ink-500 leading-relaxed mt-2.5 flex-1">{body}</p>
              </li>
            ))}
          </ol>

          <div className="mt-10 flex flex-col sm:flex-row sm:items-center gap-4 justify-between
                          surface-subtle px-6 py-5">
            <div>
              <p className="text-[15px] font-semibold text-ink-900">Ready to screen a patient?</p>
              <p className="text-[13px] text-ink-500 mt-0.5">
                You&rsquo;ll enter patient details next, then move straight to torch capture.
              </p>
            </div>
            <Link href="/patient" className="btn-primary shrink-0">
              Start screening
              <ArrowRight />
            </Link>
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          BULK SCREENING
      ═══════════════════════════════════════════════════════════ */}
      <section className="border-b border-ink-100 bg-ink-50/60">
        <div className="max-w-6xl mx-auto px-5 sm:px-6 py-14 md:py-20
                        grid md:grid-cols-12 gap-10 md:gap-12">

          <div className="md:col-span-5">
            <SectionHeading
              eyebrow="Bulk pre-screen"
              title="Screen a whole class at once."
              body="Already have a folder of ID-card or frontal face photos? Upload them together and get a per-child referral list in minutes — no torch capture needed up front."
            />
            <ul className="mt-7 space-y-4">
              {[
                "Full Hirschberg corneal-reflex measurement wherever a torch reflection happens to be present.",
                "Torch-free pupil-versus-eye-corner alignment check on every remaining photo.",
                "Flagged children are re-captured with a torch for a proper measurement.",
              ].map((t) => (
                <li key={t} className="flex items-start gap-3 text-[13.5px] text-ink-600 leading-relaxed">
                  <svg className="w-4 h-4 text-clinical-600 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2.4} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
                  </svg>
                  <span>{t}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="md:col-span-7">
            <BatchScreening />
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════
          FIGURES
      ═══════════════════════════════════════════════════════════ */}
      <section className="border-b border-ink-100 bg-white">
        <div className="max-w-6xl mx-auto px-5 sm:px-6 py-12 md:py-16">
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-x-8 gap-y-9">
            {STATS.map(({ stat, label }) => (
              <div key={label} className="border-t border-ink-900/80 pt-4">
                <p className="font-display text-[34px] leading-none text-ink-900 tabular">{stat}</p>
                <p className="text-[12.5px] text-ink-500 mt-2.5 leading-relaxed">{label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <SiteFooter showDevLink />
    </main>
  );
}
