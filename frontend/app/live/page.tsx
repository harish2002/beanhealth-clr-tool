"use client";

import Link from "next/link";
import { useAppStore } from "@/store/useAppStore";
import LiveEyeView from "@/components/LiveEyeView";
import { TopBar } from "@/components/ui/Chrome";

/**
 * Live eye view — real-time measurement on the camera feed.
 * A preview beside the capture step; the reported result still comes from
 * the 8-frame analysis on /capture.
 */
export default function LivePage() {
  const patientName = useAppStore((s) => s.patientName);
  const patientAge  = useAppStore((s) => s.patientAge);
  const hasPatient  = Boolean(patientName) && patientAge !== null;

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <TopBar step={2} />

      <div className="border-b border-ink-100 bg-ink-50/60">
        <div className="max-w-6xl mx-auto px-5 sm:px-6 py-3.5 flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[14px] font-semibold text-ink-900 truncate">
              Live eye view
              {hasPatient && <span className="font-normal text-ink-500"> · {patientName}, {patientAge}</span>}
            </p>
            <p className="text-[11.5px] text-ink-400">
              Real-time preview — the report comes from the 8-frame analysis
            </p>
          </div>
          <Link href={hasPatient ? "/capture" : "/patient"} className="btn-primary btn-sm whitespace-nowrap">
            {hasPatient ? "Run 8-frame analysis" : "Enter patient details"}
          </Link>
        </div>
      </div>

      <main className="flex-1 px-4 sm:px-6 py-6 md:py-8">
        <div className="max-w-6xl mx-auto">
          <LiveEyeView patientName={patientName} patientAge={patientAge} />
        </div>
      </main>
    </div>
  );
}
