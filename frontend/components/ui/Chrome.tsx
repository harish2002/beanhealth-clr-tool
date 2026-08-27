"use client";

import Link from "next/link";

/**
 * Shared application chrome — brand mark, top bar, flow stepper, footer.
 *
 * These are presentation-only. No store access, no side effects, so they can be
 * dropped into any route in the screening flow.
 */

// ─── Brand mark ───────────────────────────────────────────────────────────────

export function EyeGlyph({ className = "w-5 h-5" }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.6} stroke="currentColor" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
    </svg>
  );
}

export function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <Link href="/" className="flex items-center gap-2.5 group" aria-label="BeanHealth CLR — home">
      <span className="w-9 h-9 rounded-[10px] bg-ink-900 flex items-center justify-center
                       transition-transform duration-200 group-hover:scale-[1.04]">
        <EyeGlyph className="w-[18px] h-[18px] text-white" />
      </span>
      {!compact && (
        <span className="leading-none">
          <span className="block text-[14px] font-semibold text-ink-900 tracking-[-0.01em]">
            BeanHealth <span className="text-ink-400 font-normal">CLR</span>
          </span>
          <span className="block text-[10.5px] text-ink-400 mt-1">
            Pediatric strabismus screening
          </span>
        </span>
      )}
    </Link>
  );
}

// ─── Flow stepper ─────────────────────────────────────────────────────────────

export const FLOW_STEPS = ["Patient", "Capture", "Report"] as const;

/**
 * Three-dot progress rail for the screening flow.
 * @param current 1-based index of the active step.
 */
export function FlowStepper({ current }: { current: 1 | 2 | 3 }) {
  return (
    <ol className="flex items-center gap-0 select-none" aria-label="Screening progress">
      {FLOW_STEPS.map((label, i) => {
        const index = i + 1;
        const done    = index < current;
        const active  = index === current;

        return (
          <li key={label} className="flex items-center">
            <span
              className="flex items-center gap-2"
              aria-current={active ? "step" : undefined}
            >
              <span
                className={`w-[18px] h-[18px] rounded-full flex items-center justify-center
                            text-[10px] font-semibold transition-colors duration-200
                            ${done   ? "bg-clinical-600 text-white"
                            : active ? "bg-ink-900 text-white"
                                     : "bg-white text-ink-300 border border-ink-200"}`}
              >
                {done ? (
                  <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" strokeWidth={3.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
                  </svg>
                ) : index}
              </span>
              <span className={`text-[12px] font-medium hidden sm:inline
                                ${active ? "text-ink-900" : done ? "text-ink-500" : "text-ink-300"}`}>
                {label}
              </span>
            </span>

            {index < FLOW_STEPS.length && (
              <span className={`mx-2.5 sm:mx-3 h-px w-5 sm:w-8 ${done ? "bg-clinical-600/40" : "bg-ink-200"}`} />
            )}
          </li>
        );
      })}
    </ol>
  );
}

// ─── Top bar ──────────────────────────────────────────────────────────────────

export function TopBar({
  step,
  right,
}: {
  /** Renders the flow stepper when provided. */
  step?: 1 | 2 | 3;
  /** Optional right-hand slot; overrides the stepper when both are given. */
  right?: React.ReactNode;
}) {
  return (
    <header
      className="sticky top-0 z-30 border-b border-ink-100 bg-white/85 backdrop-blur-md
                 supports-[backdrop-filter]:bg-white/75 no-print"
    >
      <div className="max-w-6xl mx-auto px-5 sm:px-6 h-16 flex items-center justify-between gap-4">
        <BrandMark />
        {right ?? (step ? <FlowStepper current={step} /> : null)}
      </div>
    </header>
  );
}

// ─── Footer ───────────────────────────────────────────────────────────────────

export function SiteFooter({ showDevLink = false }: { showDevLink?: boolean }) {
  return (
    <footer className="border-t border-ink-100 bg-white no-print">
      <div className="max-w-6xl mx-auto px-5 sm:px-6 py-9
                      flex flex-col md:flex-row gap-6 justify-between md:items-center">

        <div className="flex items-start gap-3">
          <span className="w-8 h-8 rounded-lg bg-ink-900 flex items-center justify-center shrink-0">
            <EyeGlyph className="w-4 h-4 text-white" />
          </span>
          <div className="text-[12px] text-ink-500 leading-relaxed">
            <p className="font-semibold text-ink-800">BeanHealth Private Limited</p>
            <p className="text-ink-400">
              DPIIT-recognised · CDSCO Class A SaMD (pre-registration) · Coimbatore
            </p>
          </div>
        </div>

        <div className="text-[11.5px] text-ink-400 leading-relaxed md:text-right max-w-md space-y-1.5">
          <p>
            Screening aid only. Not a diagnostic device. Results must be confirmed by a
            qualified ophthalmologist before any clinical action is taken.
          </p>
          {showDevLink && (
            <p>
              <a
                href="/test"
                className="text-ink-500 hover:text-ink-900 underline underline-offset-4 decoration-ink-200
                           hover:decoration-ink-400 font-medium transition-colors"
              >
                Dev · analyse a saved image
              </a>
            </p>
          )}
        </div>
      </div>
    </footer>
  );
}

// ─── Small shared bits ────────────────────────────────────────────────────────

export function ArrowRight({ className = "w-4 h-4" }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={2.2} stroke="currentColor" aria-hidden>
      <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
    </svg>
  );
}

export function SectionHeading({
  eyebrow,
  title,
  body,
  className = "",
}: {
  eyebrow: string;
  title: React.ReactNode;
  body?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`max-w-2xl ${className}`}>
      <p className="eyebrow mb-3">{eyebrow}</p>
      <h2 className="display-lg">{title}</h2>
      {body && <p className="lede mt-4">{body}</p>}
    </div>
  );
}
