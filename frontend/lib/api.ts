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
import type { AnalyseResponse, BatchResponse, ClientMeasurement, StreamAnalyseResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK_API === "true";

/** Conservative per-photo processing estimate (mirrors backend BATCH_SECONDS_PER_PHOTO). */
export const BATCH_SECONDS_PER_PHOTO = 2.5;
/** Max photos accepted per batch request (mirrors backend BATCH_MAX_PHOTOS). */
export const BATCH_MAX_PHOTOS = 60;

/** Human-readable estimate of how long a batch of `n` photos will take. */
export function estimateBatchSeconds(n: number): number {
  // Add a small fixed overhead for upload + cold-start.
  return Math.ceil(n * BATCH_SECONDS_PER_PHOTO + 3);
}

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
    asymmetry_pd:            25.9,
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
    bav_nasal:               0.35,
    bav_vertical:            0.02,
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
  /** The browser's own measurement of these frames; the server compares and logs it. */
  clientMeasurement?: ClientMeasurement,
): Promise<StreamAnalyseResponse> {
  const form = new FormData();
  frames.forEach((f) => form.append("images", f));
  form.append("patient_name", patientName);
  form.append("patient_age",  String(patientAge));
  if (clientMeasurement) form.append("client_measurement", JSON.stringify(clientMeasurement));
  // Send User-Agent for session-level device calibration (Phase 1)
  form.append("user_agent",   typeof navigator !== "undefined" ? navigator.userAgent : "");

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

// ─── Test-mode single-image analysis ───────────────────────────────────────
//
// Calls /analyse-test which bypasses the live-torch flash check so that the
// pipeline can be validated against research/case-study images that were
// not captured by a live phone torch. Returns the result in the same shape
// as analyseStream so the existing TriageReport component renders it.

export async function analyseTest(
  image:       File,
  patientName: string,
  patientAge:  number,
): Promise<StreamAnalyseResponse> {
  const form = new FormData();
  form.append("image",        image);
  form.append("patient_name", patientName);
  form.append("patient_age",  String(patientAge));

  try {
    const response = await axios.post<StreamAnalyseResponse>(
      `${API_URL}/analyse-test`,
      form,
      {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 60_000,
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

// ─── Batch pre-screen (school / camp) ────────────────────────────────────────
//
// Uploads N photos to /analyse-batch and returns a compact per-photo referral
// list. Runs both the Hirschberg CLR pipeline and the CLR-free corner-alignment
// net on each photo. Screening aid only — flagged children must be re-captured
// with a torch for a real measurement.

export async function analyseBatch(
  files:  File[],
  labels?: string[],
): Promise<BatchResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("images", f));
  if (labels && labels.length) form.append("labels", labels.join(","));

  // Scale the timeout with the batch size — processing is sequential server-side.
  const timeout = Math.max(60_000, estimateBatchSeconds(files.length) * 1000 * 1.5);

  try {
    const response = await axios.post<BatchResponse>(
      `${API_URL}/analyse-batch`,
      form,
      { headers: { "Content-Type": "multipart/form-data" }, timeout },
    );
    return response.data;
  } catch (err) {
    const axiosErr = err as AxiosError;
    if (axiosErr.response) {
      const data = axiosErr.response.data as BatchResponse;
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
