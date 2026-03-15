/**
 * BeanHealth CLR Tool — Shared TypeScript Types
 *
 * These mirror the Pydantic response models in backend/models/response.py.
 * No `any` types — ever.
 */

// ─── Urgency / severity constants ───────────────────────────────────────────

export type UrgencyTier  = "URGENT" | "ROUTINE" | "MONITOR" | "NORMAL";
export type SeverityTier = "NORMAL" | "MILD"    | "MODERATE" | "SEVERE";
export type Confidence   = "HIGH"   | "MEDIUM"  | "LOW";
export type EyeDirection = "nasal"  | "temporal" | "superior" | "inferior";
export type AnalysisStatus = "SUCCESS" | "INCONCLUSIVE" | "ERROR";

// ─── Sub-types ───────────────────────────────────────────────────────────────

export interface PatientInfo {
  name: string;
  age:  number;
}

export interface ClinicalResult {
  urgency_tier:            UrgencyTier;
  condition_name:          string;
  icd10_code:              string;
  deviation_degrees:       number;
  asymmetry_score:         number;
  severity:                SeverityTier;
  referral_recommendation: string;
  timeframe:               string;
  narrative:               string;
}

export interface TechnicalDetail {
  left_pupil:               [number, number];
  right_pupil:              [number, number];
  left_clr:                 [number, number];
  right_clr:                [number, number];
  left_displacement_norm:   number;
  right_displacement_norm:  number;
  left_direction:           EyeDirection;
  right_direction:          EyeDirection;
  deviation_mm:             number;
  dominant_eye:             string;
  confidence:               Confidence;
  flags:                    string[];
}

// ─── Top-level response shapes ───────────────────────────────────────────────

export interface IntermediateImages {
  module1_crops:  string;
  module2_pupil:  string;
  module3_clr:    string;
  module4_vector: string;
}

export interface SuccessResponse {
  status:                "SUCCESS";
  patient:               PatientInfo;
  result:                ClinicalResult;
  technical:             TechnicalDetail;
  intermediate_images?:  IntermediateImages;
  annotated_image_b64:   string;
  timestamp:             string;
}

export interface InconclusiveResponse {
  status:       "INCONCLUSIVE";
  reason:       string;
  reason_human: string;
  flags:        string[];
  patient?:     PatientInfo;
  timestamp:    string;
}

export interface ErrorResponse {
  status:    "ERROR";
  message:   string;
  patient?:  PatientInfo;
  timestamp: string;
}

export type AnalyseResponse = SuccessResponse | InconclusiveResponse | ErrorResponse;

// ─── UI state ────────────────────────────────────────────────────────────────

export interface AppState {
  patientName:          string;
  patientAge:           number | null;
  capturedImageDataUrl: string | null;
  capturedImageFile:    File | null;
  analysisResult:       AnalyseResponse | null;
  isLoading:            boolean;
  errorMessage:         string | null;
}

// ─── Urgency display config (light theme) ────────────────────────────────────

export const URGENCY_CONFIG: Record<
  UrgencyTier,
  {
    label:        string;
    colour:       string;   // text colour for headings on light bg
    bgColour:     string;   // card/banner background
    borderColour: string;   // card border
    badgeColour:  string;   // pill badge background
    dotColour:    string;   // status dot
  }
> = {
  URGENT: {
    label:        "URGENT",
    colour:       "text-red-600",
    bgColour:     "bg-red-50",
    borderColour: "border-red-200",
    badgeColour:  "bg-red-600",
    dotColour:    "bg-red-500",
  },
  ROUTINE: {
    label:        "ROUTINE",
    colour:       "text-orange-600",
    bgColour:     "bg-orange-50",
    borderColour: "border-orange-200",
    badgeColour:  "bg-orange-500",
    dotColour:    "bg-orange-400",
  },
  MONITOR: {
    label:        "MONITOR",
    colour:       "text-amber-600",
    bgColour:     "bg-amber-50",
    borderColour: "border-amber-200",
    badgeColour:  "bg-amber-500",
    dotColour:    "bg-amber-400",
  },
  NORMAL: {
    label:        "NORMAL",
    colour:       "text-green-600",
    bgColour:     "bg-green-50",
    borderColour: "border-green-200",
    badgeColour:  "bg-green-600",
    dotColour:    "bg-green-500",
  },
};
