/**
 * BeanHealth CLR Tool — Global State (Zustand)
 *
 * Single source of truth for:
 *   - Patient info (name, age)
 *   - Captured image (preview URL + raw File)
 *   - Analysis result from the API
 *   - Loading / error state
 */

import { create } from "zustand";
import type { AnalyseResponse } from "@/lib/types";

interface AppStore {
  // ── Patient ──────────────────────────────────────────────────────────────
  patientName: string;
  patientAge:  number | null;
  setPatient:  (name: string, age: number) => void;

  // ── Captured image ────────────────────────────────────────────────────────
  capturedImageDataUrl: string | null;
  capturedImageFile:    File | null;
  setCapturedImage:     (dataUrl: string, file: File) => void;

  // ── Analysis result ───────────────────────────────────────────────────────
  analysisResult:    AnalyseResponse | null;
  setAnalysisResult: (result: AnalyseResponse) => void;

  // ── UI state ──────────────────────────────────────────────────────────────
  isLoading:    boolean;
  setLoading:   (v: boolean) => void;
  errorMessage: string | null;
  setError:     (msg: string | null) => void;

  // ── Full reset (for "try again") ──────────────────────────────────────────
  reset: () => void;
}

const initialState = {
  patientName:          "",
  patientAge:           null,
  capturedImageDataUrl: null,
  capturedImageFile:    null,
  analysisResult:       null,
  isLoading:            false,
  errorMessage:         null,
};

export const useAppStore = create<AppStore>((set) => ({
  ...initialState,

  setPatient: (name, age) =>
    set({ patientName: name, patientAge: age }),

  setCapturedImage: (dataUrl, file) =>
    set({ capturedImageDataUrl: dataUrl, capturedImageFile: file }),

  setAnalysisResult: (result) =>
    set({ analysisResult: result }),

  setLoading: (v) =>
    set({ isLoading: v }),

  setError: (msg) =>
    set({ errorMessage: msg }),

  reset: () => set(initialState),
}));
