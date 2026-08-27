"use client";

import { useRef, useState } from "react";
import { analyseBatch, estimateBatchSeconds, BATCH_MAX_PHOTOS } from "@/lib/api";
import type { BatchItemResult, BatchResponse } from "@/lib/types";

/**
 * BatchScreening — bulk pre-screen many photos in one pass.
 *
 * For the school / camp use-case: upload a folder of ordinary photos (e.g. the
 * ones taken for ID cards) and get back a per-photo referral list in minutes.
 * Each photo is run through BOTH the Hirschberg CLR pipeline (only yields an
 * angle when a torch reflex is present) and the CLR-free corner-alignment net.
 *
 * Screening aid only — flagged children must be re-captured with a torch for a
 * real Hirschberg measurement.
 */

function fmtDuration(seconds: number): string {
  if (seconds < 90) return `~${seconds}s`;
  const m = Math.round(seconds / 60);
  return `~${m} min`;
}

function VerdictPill({ item }: { item: BatchItemResult }) {
  const base =
    "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold shrink-0";

  if (item.referral_flag) {
    return (
      <span className={`${base} bg-red-50 border-red-200 text-red-700`}>
        <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
        Refer
      </span>
    );
  }
  if (item.status === "INCONCLUSIVE" || item.status === "ERROR") {
    return (
      <span className={`${base} bg-ink-50 border-ink-200 text-ink-500`}>
        <span className="w-1.5 h-1.5 rounded-full bg-ink-300" />
        Inconclusive
      </span>
    );
  }
  return (
    <span className={`${base} bg-emerald-50 border-emerald-200 text-emerald-700`}>
      <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
      Clear
    </span>
  );
}

export default function BatchScreening() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles]         = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult]       = useState<BatchResponse | null>(null);
  const [error, setError]         = useState<string | null>(null);

  function addFiles(fileList: FileList | null) {
    if (!fileList) return;
    const incoming = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
    setFiles((prev) => {
      const merged = [...prev, ...incoming];
      return merged.slice(0, BATCH_MAX_PHOTOS);
    });
    setError(null);
    setResult(null);
  }

  function clearAll() {
    setFiles([]);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function handleRun() {
    if (files.length === 0) {
      setError("Add at least one photo first.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const res = await analyseBatch(files, files.map((f) => f.name));
      setResult(res);
      if (res.status === "ERROR") setError(res.message ?? "Batch failed.");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Batch failed.";
      setError(
        msg === "TIMEOUT" ? "The batch timed out. Try fewer photos per batch."
        : msg === "NETWORK_ERROR" ? "Could not reach the server. Check your connection."
        : msg
      );
    } finally {
      setSubmitting(false);
    }
  }

  const estSeconds = estimateBatchSeconds(files.length);

  return (
    <div className="surface-raised overflow-hidden">

      {/* Header */}
      <div className="border-b border-ink-100 px-6 py-4 flex items-start justify-between gap-4">
        <div>
          <p className="eyebrow">Batch pre-screen</p>
          <p className="text-[15px] font-semibold text-ink-900 mt-1">
            Screen a whole class in one pass
          </p>
        </div>
        <span className="font-mono text-[11px] text-ink-400 shrink-0 pt-1">
          MAX {BATCH_MAX_PHOTOS}
        </span>
      </div>

      <div className="px-6 py-6 space-y-5">
        <p className="text-[13.5px] text-ink-500 leading-relaxed">
          Upload the photos taken for ID cards, or any frontal face photos. Each one is
          checked two ways — a Hirschberg reflex measurement where a torch reflection is
          present, and a torch-free pupil-versus-corner alignment check on every photo.
        </p>

        {/* Drop zone */}
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); addFiles(e.dataTransfer.files); }}
          className="group cursor-pointer rounded-card border border-dashed border-ink-200 bg-ink-50/50
                     hover:border-clinical-600/50 hover:bg-clinical-50 transition-all duration-200
                     px-4 py-10 text-center"
        >
          <svg
            className="w-8 h-8 text-ink-300 group-hover:text-clinical-600 transition-colors mx-auto mb-3"
            fill="none" viewBox="0 0 24 24" strokeWidth={1.4} stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
          </svg>
          <p className="text-[14px] text-ink-800 font-medium">
            Click to add photos, or drag them here
          </p>
          <p className="font-mono text-[11px] text-ink-400 mt-1.5">JPEG · PNG · MULTI-SELECT</p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png"
            multiple
            onChange={(e) => addFiles(e.target.files)}
            className="hidden"
          />
        </div>

        {/* Selection summary */}
        {files.length > 0 && (
          <div className="flex items-center justify-between gap-4 rounded-field
                          bg-clinical-50 border border-clinical-100 px-4 py-3 animate-fade-in">
            <div>
              <p className="text-[13.5px] font-semibold text-ink-900 tabular">
                {files.length} photo{files.length === 1 ? "" : "s"} selected
              </p>
              <p className="text-[12px] text-ink-500 mt-0.5">
                Estimated time <span className="font-mono text-ink-800">{fmtDuration(estSeconds)}</span>
                <span className="text-ink-400"> · ~2.5s per photo</span>
              </p>
            </div>
            <button
              type="button"
              onClick={clearAll}
              className="text-[12px] font-medium text-ink-500 hover:text-ink-900 transition-colors shrink-0"
            >
              Clear
            </button>
          </div>
        )}

        {error && (
          <div role="alert" className="rounded-field bg-red-50 border border-red-200 px-3.5 py-3 text-[13.5px] text-red-700">
            {error}
          </div>
        )}

        {/* Run */}
        <button
          type="button"
          onClick={handleRun}
          disabled={submitting || files.length === 0}
          className="btn-accent w-full"
        >
          {submitting ? (
            <>
              <span className="w-4 h-4 border-2 border-white/40 border-t-transparent rounded-full animate-spin" />
              Analysing {files.length} photo{files.length === 1 ? "" : "s"}…
            </>
          ) : (
            <>Pre-screen {files.length > 0 ? files.length : ""} photo{files.length === 1 ? "" : "s"}</>
          )}
        </button>

        {/* Results */}
        {result && result.status === "DONE" && (
          <div className="space-y-4 pt-2 animate-rise-in">

            <div className="grid grid-cols-3 gap-px bg-ink-100 border border-ink-100 rounded-card overflow-hidden">
              {[
                { label: "Photos",  value: String(result.total),  tone: "text-ink-900" },
                { label: "Flagged", value: String(result.flagged), tone: result.flagged > 0 ? "text-red-600" : "text-ink-900" },
                { label: "Took",    value: fmtDuration(Math.round(result.elapsed_seconds)), tone: "text-ink-900" },
              ].map((s) => (
                <div key={s.label} className="bg-white px-3 py-4 text-center">
                  <p className="eyebrow-muted mb-1.5">{s.label}</p>
                  <p className={`font-display text-[26px] leading-none tabular ${s.tone}`}>{s.value}</p>
                </div>
              ))}
            </div>

            <div className="rounded-card border border-ink-100 divide-y divide-ink-100 overflow-hidden">
              {result.items.map((item) => {
                const hb = item.hirschberg;
                const al = item.alignment;
                return (
                  <div key={item.index} className="flex items-center gap-3 px-4 py-3 hover:bg-ink-50/60 transition-colors">
                    <div className="flex-1 min-w-0">
                      <p className="text-[13.5px] text-ink-800 font-medium truncate">{item.label}</p>
                      <p className="font-mono text-[10.5px] text-ink-400 mt-1 truncate">
                        {hb?.available && hb.urgency_tier
                          ? `HB ${hb.urgency_tier} · ${hb.condition_name} · ${hb.asymmetry_degrees?.toFixed(1)}°`
                          : "HB no reflex"}
                        {al?.available ? ` · ALIGN ${al.verdict}` : " · ALIGN n/a"}
                        {item.reason && !hb?.available && !al?.available ? ` · ${item.reason}` : ""}
                      </p>
                    </div>
                    <VerdictPill item={item} />
                  </div>
                );
              })}
            </div>

            <p className="text-[11.5px] text-ink-400 leading-relaxed">
              Screening aid only — not a diagnosis. Children marked{" "}
              <span className="font-semibold text-red-600">Refer</span> should be re-captured
              with a torch for a proper Hirschberg measurement and referred to an ophthalmologist.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
