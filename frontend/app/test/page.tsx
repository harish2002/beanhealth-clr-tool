"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAppStore } from "@/store/useAppStore";
import { analyseTest } from "@/lib/api";
import type { StreamSuccessResponse, StreamInconclusiveResponse } from "@/lib/types";

/**
 * /test — Development / research single-image upload.
 *
 * Bypasses the live torch capture so the CV pipeline can be validated
 * on saved strabismus images (research datasets, case studies, etc.).
 * Hits the backend /analyse-test endpoint which lowers the flash detection
 * threshold and returns the result in StreamSuccessResponse shape so the
 * existing /result page renders it without modification.
 *
 * Results carry a "test_capture" flag — the report shows a banner so this
 * cannot be confused with a real clinical screening.
 */
export default function TestUploadPage() {
  const router            = useRouter();
  const reset             = useAppStore((s) => s.reset);
  const setPatient        = useAppStore((s) => s.setPatient);
  const setPatientMeta    = useAppStore((s) => s.setPatientMeta);
  const setAnalysisResult = useAppStore((s) => s.setAnalysisResult);
  const setLoading        = useAppStore((s) => s.setLoading);
  const setError          = useAppStore((s) => s.setError);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const [file,       setFile]       = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [name,       setName]       = useState("Test Patient");
  const [age,        setAge]        = useState("8");
  const [dragging,   setDragging]   = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  function handleFile(f: File | null) {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setLocalError("Please select a JPEG or PNG image.");
      return;
    }
    setLocalError(null);
    setFile(f);
    const reader = new FileReader();
    reader.onload = () => setPreviewUrl(reader.result as string);
    reader.readAsDataURL(f);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLocalError(null);

    if (!file) {
      setLocalError("Please select an image first.");
      return;
    }
    const parsedAge = parseInt(age, 10);
    if (!name.trim() || isNaN(parsedAge) || parsedAge < 1 || parsedAge > 120) {
      setLocalError("Enter a valid name and age (1–120).");
      return;
    }

    setSubmitting(true);
    try {
      reset();
      setPatient(name.trim(), parsedAge);
      setPatientMeta("", "Test Mode", "Image upload");

      const result = await analyseTest(file, name.trim(), parsedAge);

      setLoading(false);
      if (result.status === "INCONCLUSIVE") {
        const inc = result as StreamInconclusiveResponse;
        setAnalysisResult({
          status:             "INCONCLUSIVE",
          reason:             inc.reason,
          reason_human:       inc.reason_human,
          flags:              inc.flags,
          patient:            inc.patient,
          timestamp:          inc.timestamp,
          // Torch-free corner-alignment still runs even when no CLR is found
          alignment:          inc.alignment,
        });
      } else {
        const ok = result as StreamSuccessResponse;
        setAnalysisResult({
          ...ok,
          status:              "SUCCESS",
          patient:             ok.patient,
          result:              ok.result,
          technical:           ok.technical,
          intermediate_images: ok.intermediate_images,
          annotated_image_b64: ok.annotated_image_b64 ?? "",
          timestamp:           ok.timestamp,
        });
      }
      router.push("/result");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Upload failed.";
      setLocalError(
        msg === "TIMEOUT"      ? "Server timed out. Try again." :
        msg === "NETWORK_ERROR" ? "Could not reach the server. Check connection." :
        msg
      );
      setError(msg);
      setSubmitting(false);
    }
  }

  return (
    <main className="min-h-screen bg-white text-ink-900">

      {/* ── Top nav ─────────────────────────────────────────────────────── */}
      <nav className="border-b border-ink-100 bg-white sticky top-0 z-20">
        <div className="max-w-3xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-lg bg-clinical-600 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
              </svg>
            </div>
            <div className="leading-none">
              <p className="text-sm font-bold text-ink-900">BeanHealth CLR</p>
              <p className="text-[10px] text-ink-400 mt-0.5">Pediatric Strabismus Screening</p>
            </div>
          </Link>

          <span className="text-[10px] font-bold uppercase tracking-[0.22em] text-amber-600 border border-amber-300 rounded-full px-2.5 py-1 bg-amber-50">
            Dev / research mode
          </span>
        </div>
      </nav>

      {/* ── Body ────────────────────────────────────────────────────────── */}
      <section className="max-w-3xl mx-auto px-6 py-12">

        <p className="text-[11px] font-bold uppercase tracking-[0.22em] text-clinical-600 mb-3">
          Image upload
        </p>
        <h1 className="text-3xl font-bold text-ink-900 tracking-tight leading-tight">
          Test the pipeline on a saved image
        </h1>
        <p className="text-ink-500 text-base mt-3 leading-relaxed">
          Upload a single eye photo (research dataset, published case study, etc.)
          to run it through the CLR pipeline without the live-torch requirement.
          The flash check is lowered from 235 → 180 luminance so screen-rendered
          or saved JPEGs can be analysed.
        </p>

        {/* Warning strip */}
        <div className="mt-5 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-xs text-amber-800 leading-relaxed">
          <strong className="font-bold">Test mode — not for clinical use.</strong>{" "}
          Results are single-frame, lack inter-frame variance evidence, and the flash
          guard is relaxed. The report will show a TEST CAPTURE banner.
        </div>

        <form onSubmit={handleSubmit} className="mt-8 space-y-6">

          {/* ── Drop zone / preview ─────────────────────────────────────── */}
          <div>
            <label className="block text-xs font-semibold text-ink-700 mb-1.5">
              Eye image <span className="text-red-500">*</span>
            </label>
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                handleFile(e.dataTransfer.files?.[0] ?? null);
              }}
              className={`relative cursor-pointer rounded-xl border-2 border-dashed transition-colors
                          flex items-center justify-center min-h-[200px] overflow-hidden ${
                dragging
                  ? "border-blue-500 bg-clinical-50"
                  : previewUrl
                    ? "border-ink-100 bg-ink-50"
                    : "border-ink-200 bg-ink-50 hover:border-ink-300 hover:bg-ink-100"
              }`}
            >
              {previewUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={previewUrl}
                  alt="Selected eye photo"
                  className="max-h-[300px] max-w-full object-contain"
                />
              ) : (
                <div className="text-center py-8 px-4">
                  <svg className="w-10 h-10 text-ink-400 mx-auto mb-2" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
                  </svg>
                  <p className="text-sm text-ink-600 font-medium">
                    Click to upload or drag and drop
                  </p>
                  <p className="text-xs text-ink-400 mt-1">JPEG or PNG · max 10 MB</p>
                </div>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png"
                onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
            </div>

            {file && (
              <p className="text-xs text-ink-500 mt-2 flex items-center justify-between">
                <span className="truncate">{file.name}</span>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setFile(null);
                    setPreviewUrl(null);
                    if (fileInputRef.current) fileInputRef.current.value = "";
                  }}
                  className="text-clinical-600 hover:text-clinical-700 font-medium shrink-0 ml-3"
                >
                  Remove
                </button>
              </p>
            )}
          </div>

          {/* ── Name + Age ──────────────────────────────────────────────── */}
          <div className="grid sm:grid-cols-2 gap-4">
            <div>
              <label htmlFor="name" className="block text-xs font-semibold text-ink-700 mb-1.5">
                Label <span className="text-ink-400 font-normal">(case ID, dataset name, etc.)</span>
              </label>
              <input
                id="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Case-42 Eso 25°"
                className="w-full rounded-lg border border-ink-200 bg-white px-3.5 py-2.5
                           text-ink-900 placeholder-ink-300 text-[15px]
                           focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500"
              />
            </div>
            <div>
              <label htmlFor="age" className="block text-xs font-semibold text-ink-700 mb-1.5">
                Age <span className="text-ink-400 font-normal">(demo)</span>
              </label>
              <input
                id="age"
                type="number"
                min={1}
                max={120}
                value={age}
                onChange={(e) => setAge(e.target.value)}
                className="w-full rounded-lg border border-ink-200 bg-white px-3.5 py-2.5
                           text-ink-900 placeholder-ink-300 text-[15px]
                           focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500"
              />
            </div>
          </div>

          {/* ── Error ───────────────────────────────────────────────────── */}
          {localError && (
            <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg px-3 py-2.5">
              <svg className="w-4 h-4 text-red-500 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
              </svg>
              <p className="text-red-700 text-sm">{localError}</p>
            </div>
          )}

          {/* ── Submit ──────────────────────────────────────────────────── */}
          <div className="flex gap-3">
            <Link
              href="/"
              className="flex-1 rounded-lg bg-ink-100 hover:bg-ink-200 text-ink-700 font-medium
                         py-3 text-[15px] text-center transition-colors"
            >
              Back
            </Link>
            <button
              type="submit"
              disabled={submitting || !file}
              className={`flex-[2] rounded-lg font-semibold py-3 text-[15px] transition-colors
                         flex items-center justify-center gap-2 ${
                submitting || !file
                  ? "bg-ink-200 text-ink-400 cursor-not-allowed"
                  : "bg-ink-900 hover:bg-ink-800 text-white shadow-card"
              }`}
            >
              {submitting ? (
                <>
                  <span className="w-4 h-4 border-2 border-ink-200 border-t-transparent rounded-full animate-spin" />
                  Analysing…
                </>
              ) : (
                <>
                  Run test analysis
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2.2} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5 21 12m0 0-7.5 7.5M21 12H3" />
                  </svg>
                </>
              )}
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}
