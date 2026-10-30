"use client";

import GuidedExam from "@/components/GuidedExam";
import { TopBar } from "@/components/ui/Chrome";

/**
 * Gaze examination — guided multi-position capture (Phase 3).
 * Captures and quality-checks the landmark tracks; the gaze grid and the
 * clinician report are built from them in later milestones.
 */
export default function ExamPage() {
  return (
    <div className="min-h-screen bg-white flex flex-col">
      <TopBar />
      <div className="border-b border-ink-100 bg-ink-50/60">
        <div className="max-w-6xl mx-auto px-5 sm:px-6 py-3.5">
          <p className="text-[14px] font-semibold text-ink-900">Gaze examination</p>
          <p className="text-[11.5px] text-ink-400">
            Calibration, 9 gaze positions and head tilts · measurements only, no video leaves the phone
          </p>
        </div>
      </div>
      <main className="flex-1 px-4 sm:px-6 py-6 md:py-8">
        <div className="max-w-6xl mx-auto">
          <GuidedExam />
        </div>
      </main>
    </div>
  );
}
