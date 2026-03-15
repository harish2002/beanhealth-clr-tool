"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAppStore } from "@/store/useAppStore";
import { analyseImage } from "@/lib/api";
import CameraCapture from "@/components/CameraCapture";

export default function CapturePage() {
  const router = useRouter();
  const patientName = useAppStore((s) => s.patientName);
  const patientAge  = useAppStore((s) => s.patientAge);
  const setCapturedImage  = useAppStore((s) => s.setCapturedImage);
  const setAnalysisResult = useAppStore((s) => s.setAnalysisResult);
  const setLoading        = useAppStore((s) => s.setLoading);
  const setError          = useAppStore((s) => s.setError);

  // Redirect back to form if no patient data
  useEffect(() => {
    if (!patientName || patientAge === null) {
      router.replace("/");
    }
  }, [patientName, patientAge, router]);

  async function handleCapture(dataUrl: string, file: File) {
    if (!patientName || patientAge === null) return;

    setCapturedImage(dataUrl, file);
    setLoading(true);
    setError(null);

    try {
      const result = await analyseImage(file, patientName, patientAge);
      setAnalysisResult(result);
      setLoading(false);
      router.push("/result");
    } catch (err) {
      setLoading(false);
      if (err instanceof Error) {
        if (err.message === "TIMEOUT") {
          setError("The analysis timed out. Please check your connection and try again.");
        } else {
          setError("Could not reach the server. Please check your connection and try again.");
        }
      }
      router.push("/result");
    }
  }

  return (
    <div className="h-screen flex flex-col bg-black">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 bg-black/80 backdrop-blur-sm z-10">
        <button
          onClick={() => router.push("/")}
          className="text-slate-400 hover:text-white transition-colors"
          aria-label="Go back"
        >
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5 8.25 12l7.5-7.5" />
          </svg>
        </button>

        <div className="text-center">
          <p className="text-white text-sm font-medium">{patientName}</p>
          <p className="text-slate-400 text-xs">Age {patientAge}</p>
        </div>

        <div className="w-6" />  {/* spacer */}
      </div>

      {/* Camera — fills remaining height */}
      <div className="flex-1 overflow-hidden">
        <CameraCapture onCapture={handleCapture} />
      </div>

      {/* Instructions strip */}
      <div className="bg-black px-5 py-3 text-center">
        <p className="text-slate-400 text-xs">
          Align both eyes inside the oval guide. Tap the shutter button to capture.
        </p>
      </div>
    </div>
  );
}
