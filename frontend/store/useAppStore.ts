/**
 * BeanHealth CLR Tool — Global State (Zustand)
 *
 * Single source of truth for:
 *   - Patient info (name, age, optional metadata)
 *   - Session ID (generated at intake, stable for the lifetime of one screening)
 *   - Captured image (preview URL + raw File)
 *   - Analysis result from the API
 *   - Loading / error state
 */

import { create } from "zustand";
import type { AnalyseResponse } from "@/lib/types";

interface AppStore {
  // ── Patient ──────────────────────────────────────────────────────────────
  patientName:        string;
  patientAge:         number | null;
  /** M / F / Other / "" (not specified) */
  patientGender:      string;
  /** e.g. "Parent", "ASHA Worker", "Nurse / ANM", "General Physician" */
  screenerRole:       string;
  /** Free-text location, e.g. "PHC Bangalore" */
  screeningLocation:  string;
  /** Unique per-screening ID generated client-side, e.g. BH-X4F7A2 */
  sessionId:          string;

  setPatient:     (name: string, age: number) => void;
  setPatientMeta: (gender: string, role: string, location: string) => void;

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

function generateSessionId(): string {
  return `BH-${Date.now().toString(36).toUpperCase().slice(-6)}`;
}

const initialState = {
  patientName:          "",
  patientAge:           null,
  patientGender:        "",
  screenerRole:         "",
  screeningLocation:    "",
  sessionId:            "",
  capturedImageDataUrl: null,
  capturedImageFile:    null,
  analysisResult:       null,
  isLoading:            false,
  errorMessage:         null,
};

export const useAppStore = create<AppStore>((set) => ({
  ...initialState,

  setPatient: (name, age) =>
    set({
      patientName: name,
      patientAge:  age,
      sessionId:   generateSessionId(),
    }),

  setPatientMeta: (gender, role, location) =>
    set({
      patientGender:     gender,
      screenerRole:      role,
      screeningLocation: location,
    }),

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
