/**
 * BeanHealth CLR Tool — API Client
 *
 * Sends multipart/form-data to POST /analyse and returns a typed response.
 * The API_URL is read from the environment — never hardcoded.
 *
 * During Phase 3 development the MOCK flag returns a hardcoded response
 * so the UI can be built without a live backend.
 * Set NEXT_PUBLIC_USE_MOCK_API=false to use the real FastAPI server.
 */

import axios, { AxiosError } from "axios";
import type { AnalyseResponse, StreamAnalyseResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK_API === "true";

// ─── Mock response (Phase 3 dev — no live server needed) ────────────────────

const MOCK_SUCCESS: AnalyseResponse = {
  status: "SUCCESS",
  patient: { name: "Emma Wilson", age: 5 },
  result: {
    urgency_tier:            "ROUTINE",
    condition_name:          "Esotropia",
    icd10_code:              "H50.01",
    deviation_degrees:       14.2,
    asymmetry_degrees:       12.1,
    asymmetry_score:         0.35,
    severity:                "MODERATE",
    referral_recommendation: "Refer to ophthalmology within 4 weeks",
    timeframe:               "4 weeks",
    narrative:
      "This screening detected a moderate asymmetry in the corneal light reflex, " +
      "which may indicate Esotropia. An ophthalmology assessment is recommended " +
      "within 4 weeks for a full evaluation.",
  },
  technical: {
    left_pupil:              [125, 60],
    right_pupil:             [125, 60],
    left_clr:                [140, 60],
    right_clr:               [110, 60],
    left_displacement_norm:  0.5,
    right_displacement_norm: 0.0,
    left_direction:          "nasal",
    right_direction:         "nasal",
    deviation_mm:            2.875,
    dominant_eye:            "left",
    confidence:              "HIGH",
    flags:                   [],
  },
  annotated_image_b64: "", // empty in mock — AnnotatedEye handles gracefully
  timestamp: new Date().toISOString(),
};

// Swap MOCK_SUCCESS → MOCK_INCONCLUSIVE here to test the INCONCLUSIVE UI path
export const MOCK_INCONCLUSIVE: AnalyseResponse = {
  status:       "INCONCLUSIVE",
  reason:       "no_flash",
  reason_human: "No torch/flash detected. Enable the torch and retry.",
  flags:        ["no_flash"],
  patient:      { name: "Mock Patient", age: 5 },
  timestamp:    new Date().toISOString(),
};

// ─── Real API call ───────────────────────────────────────────────────────────

export async function analyseImage(
  image:       File,
  patientName: string,
  patientAge:  number,
): Promise<AnalyseResponse> {
  if (USE_MOCK) {
    // Simulate network delay in mock mode
    await new Promise((r) => setTimeout(r, 1800));
    return MOCK_SUCCESS;
  }

  const form = new FormData();
  form.append("image",        image);
  form.append("patient_name", patientName);
  form.append("patient_age",  String(patientAge));

  try {
    const response = await axios.post<AnalyseResponse>(
      `${API_URL}/analyse`,
      form,
      {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 30_000,   // 30 s — pipeline can be slow on first warm-up
      },
    );
    return response.data;
  } catch (err) {
    const axiosErr = err as AxiosError;

    if (axiosErr.response) {
      // Server responded with non-2xx — forward the body if it looks like our schema
      const data = axiosErr.response.data as AnalyseResponse;
      if (data?.status) return data;
    }

    if (axiosErr.code === "ECONNABORTED" || axiosErr.message.toLowerCase().includes("timeout")) {
      throw new Error("TIMEOUT");
    }

    throw new Error("NETWORK_ERROR");
  }
}

// ─── Multi-frame streaming analysis ─────────────────────────────────────────

export async function analyseStream(
  frames:      File[],
  patientName: string,
  patientAge:  number,
): Promise<StreamAnalyseResponse> {
  const form = new FormData();
  frames.forEach((f) => form.append("images", f));
  form.append("patient_name", patientName);
  form.append("patient_age",  String(patientAge));

  try {
    const response = await axios.post<StreamAnalyseResponse>(
      `${API_URL}/analyse-stream`,
      form,
      {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120_000,  // 2 min — processing 10 frames takes longer
      },
    );
    return response.data;
  } catch (err) {
    const axiosErr = err as AxiosError;
    if (axiosErr.response) {
      const data = axiosErr.response.data as StreamAnalyseResponse;
      if (data?.status) return data;
    }
    if (axiosErr.code === "ECONNABORTED" || axiosErr.message.toLowerCase().includes("timeout")) {
      throw new Error("TIMEOUT");
    }
    throw new Error("NETWORK_ERROR");
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const r = await axios.get(`${API_URL}/health`, { timeout: 5_000 });
    return r.data?.status === "ok";
  } catch {
    return false;
  }
}
