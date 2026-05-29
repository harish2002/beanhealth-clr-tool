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
  if (item.referral_flag) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-red-100 border border-red-200 text-red-700 text-[11px] font-bold px-2 py-0.5">
        🔴 Refer
      </span>
    );
  }
  if (item.status === "INCONCLUSIVE" || item.status === "ERROR") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 border border-slate-200 text-slate-500 text-[11px] font-semibold px-2 py-0.5">
        ⚪ Inconclusive
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 border border-emerald-200 text-emerald-700 text-[11px] font-semibold px-2 py-0.5">
      🟢 Clear
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
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
      <div className="border-b border-slate-100 px-6 py-4 flex items-center justify-between">
        <div>
          <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-blue-600">
            Batch pre-screen
          </p>
          <p className="text-sm font-semibold text-slate-800 mt-0.5">
            Screen a whole class in one pass
          </p>
        </div>
        <span className="text-[10px] text-slate-400 font-medium">
          Up to {BATCH_MAX_PHOTOS} photos
        </span>
      </div>

      <div className="px-6 py-6 space-y-5">
        <p className="text-sm text-slate-500 leading-relaxed">
          Upload the photos taken for ID cards (or any frontal face photos). Each
          one is checked two ways: a Hirschberg reflex measurement where a torch
          reflection is present, and a torch-free pupil-vs-corner alignment check
          on every photo. We return a per-child referral list.
        </p>

        {/* Drop zone */}
        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); addFiles(e.dataTransfer.files); }}
          className="cursor-pointer rounded-xl border-2 border-dashed border-slate-300 bg-slate-50
                     hover:border-slate-400 hover:bg-slate-100 transition-colors px-4 py-8 text-center"
        >
          <svg className="w-9 h-9 text-slate-400 mx-auto mb-2" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
          </svg>
          <p className="text-sm text-slate-600 font-medium">Click to add photos or drag &amp; drop</p>
          <p className="text-xs text-slate-400 mt-1">JPEG or PNG · select multiple</p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png"
            multiple
            onChange={(e) => addFiles(e.target.files)}
            className="hidden"
          />
        </div>

        {/* Selection summary + time estimate */}
        {files.length > 0 && (
          <div className="flex items-center justify-between rounded-xl bg-blue-50 border border-blue-200 px-4 py-3">
            <div className="text-sm">
              <p className="font-semibold text-blue-900">
                {files.length} photo{files.length === 1 ? "" : "s"} selected
              </p>
              <p className="text-blue-700 text-xs mt-0.5">
                Estimated analysis time: <strong>{fmtDuration(estSeconds)}</strong>{" "}
                <span className="text-blue-500">(~2.5s per photo)</span>
              </p>
            </div>
            <button
              type="button"
              onClick={clearAll}
              className="text-blue-600 hover:text-blue-800 text-xs font-medium shrink-0"
            >
              Clear
            </button>
          </div>
        )}

        {error && (
          <div className="rounded-lg bg-red-50 border border-red-200 px-3 py-2.5 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Run button */}
        <button
          type="button"
          onClick={handleRun}
          disabled={submitting || files.length === 0}
          className={`w-full rounded-lg font-semibold py-3 text-[15px] transition-colors
                      flex items-center justify-center gap-2 ${
            submitting || files.length === 0
              ? "bg-slate-200 text-slate-400 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700 text-white shadow-sm"
          }`}
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
          <div className="space-y-3 pt-1">
            <div className="grid grid-cols-3 gap-2">
              <div className="bg-slate-50 rounded-xl p-3 border border-slate-100 text-center">
                <p className="text-slate-400 text-xs mb-1">Photos</p>
                <p className="text-slate-900 font-bold text-lg leading-none">{result.total}</p>
              </div>
              <div className="bg-red-50 rounded-xl p-3 border border-red-100 text-center">
                <p className="text-red-400 text-xs mb-1">Flagged</p>
                <p className="text-red-600 font-bold text-lg leading-none">{result.flagged}</p>
              </div>
              <div className="bg-slate-50 rounded-xl p-3 border border-slate-100 text-center">
                <p className="text-slate-400 text-xs mb-1">Took</p>
                <p className="text-slate-900 font-bold text-lg leading-none">{fmtDuration(Math.round(result.elapsed_seconds))}</p>
              </div>
            </div>

            <div className="rounded-xl border border-slate-200 divide-y divide-slate-100 overflow-hidden">
              {result.items.map((item) => {
                const hb = item.hirschberg;
                const al = item.alignment;
                return (
                  <div key={item.index} className="flex items-center gap-3 px-3 py-2.5">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-slate-800 font-medium truncate">{item.label}</p>
                      <p className="text-[11px] text-slate-400 mt-0.5">
                        {hb?.available && hb.urgency_tier
                          ? `Hirschberg: ${hb.urgency_tier} · ${hb.condition_name} · ${hb.asymmetry_degrees?.toFixed(1)}°`
                          : "Hirschberg: no reflex"}
                        {al?.available ? ` · Alignment: ${al.verdict}` : " · Alignment: n/a"}
                        {item.reason && !hb?.available && !al?.available ? ` · ${item.reason}` : ""}
                      </p>
                    </div>
                    <VerdictPill item={item} />
                  </div>
                );
              })}
            </div>

            <p className="text-[11px] text-slate-400 leading-relaxed">
              Screening aid only — not a diagnosis. Children marked{" "}
              <span className="font-semibold text-red-500">Refer</span> should be
              re-captured with a torch for a proper Hirschberg measurement and
              referred to an ophthalmologist.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
