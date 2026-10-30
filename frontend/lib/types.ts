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
  name:               string;
  age:                number;
  // Optional context captured at intake — not sent to backend, display-only
  gender?:            string;
  screener_role?:     string;
  screening_location?: string;
}

export interface ClinicalResult {
  urgency_tier:            UrgencyTier;
  condition_name:          string;
  icd10_code:              string;
  /** Dominant eye absolute displacement — includes kappa angle, for reference only */
  deviation_degrees:       number;
  asymmetry_score:         number;
  /** Inter-ocular asymmetry in clinical degrees — the primary classification signal */
  asymmetry_degrees:       number;
  /** Inter-ocular asymmetry in prism dioptres (asymmetry_degrees × 15/7) */
  asymmetry_pd:            number;
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
  bav_nasal:                number;
  bav_vertical:             number;
  confidence:               Confidence;
  flags:                    string[];
}

// ─── Corner-alignment (CLR-free screening net) ───────────────────────────────

export type AlignmentVerdict = "ALIGNED" | "BORDERLINE" | "ASYMMETRIC" | "UNAVAILABLE";

export interface AlignmentResult {
  available:       boolean;
  verdict:         AlignmentVerdict;
  referral_flag:   boolean;
  left_h_ratio:    number | null;
  right_h_ratio:   number | null;
  left_v_ratio:    number | null;
  right_v_ratio:   number | null;
  h_asymmetry:     number | null;
  v_asymmetry:     number | null;
  interpretation:  string;
}

// ─── Top-level response shapes ───────────────────────────────────────────────

export interface IntermediateImages {
  module1_crops:  string;  // Raw eye crops
  module2_clahe:  string;  // Grayscale + CLAHE enhancement
  module3_pupil:  string;  // Pupil centre (blue dot + iris ring)
  module4_clr:    string;  // CLR bright spot (amber dot)
  module5_vector: string;  // Displacement vector + measurement
  module6_result: string;  // Final annotated image with clinical overlay
  /** Method B — canthus axis + pupil projection. Null when corners unavailable. */
  module8_alignment?: string | null;
}

export interface SuccessResponse {
  status:                "SUCCESS";
  patient:               PatientInfo;
  result:                ClinicalResult;
  technical:             TechnicalDetail;
  alignment?:            AlignmentResult;
  intermediate_images?:  IntermediateImages;
  annotated_image_b64:   string;
  timestamp:             string;
}

export interface InconclusiveResponse {
  status:             "INCONCLUSIVE";
  reason:             string;
  reason_human:       string;
  flags:              string[];
  patient?:           PatientInfo;
  timestamp:          string;
  // Torch-free corner-alignment net — present when eye detection + pupil
  // localisation succeeded before the halt (e.g. a no-flash CLR failure).
  alignment?:         AlignmentResult;
  // Present when reason = "high_variance_asymmetry" or "insufficient_frames"
  frames_total?:      number;
  frames_accepted?:   number;
  frames_rejected?:   number;
  per_frame_readings?: (number | null)[];
  // Present when reason = "high_variance_asymmetry"
  asymmetry_avg_deg?: number;
  asymmetry_std_deg?: number;
  deviation_avg_deg?: number;
  deviation_std_deg?: number;
}

export interface ErrorResponse {
  status:    "ERROR";
  message:   string;
  patient?:  PatientInfo;
  timestamp: string;
}

export type AnalyseResponse = SuccessResponse | InconclusiveResponse | ErrorResponse;

// ─── Stream (multi-frame) response shapes ────────────────────────────────────

export interface StreamClinicalResult extends ClinicalResult {
  deviation_std_deg: number;
}

export interface StreamSuccessResponse {
  status:                "SUCCESS";
  patient:               PatientInfo;
  frames_total:          number;
  frames_accepted:       number;
  frames_rejected:       number;
  per_frame_readings:    (number | null)[];
  /** Rejection reason for each frame — null = accepted, string = why rejected */
  per_frame_rejections?: (string | null)[];
  deviation_avg_deg:     number;
  deviation_std_deg:     number;
  deviation_min_deg:     number;
  deviation_max_deg:     number;
  asymmetry_avg:         number;
  /** Mean inter-ocular asymmetry in clinical degrees across accepted frames */
  asymmetry_avg_deg:     number;
  /** Std dev of inter-ocular asymmetry across accepted frames */
  asymmetry_std_deg:     number;
  aggregate_confidence:  Confidence;
  result:                StreamClinicalResult;
  technical:             TechnicalDetail;
  alignment?:            AlignmentResult;
  intermediate_images?:  IntermediateImages;
  annotated_image_b64?:  string;
  timestamp:             string;
  /** Present only when the request carried a client_measurement. */
  parity?:               ParityReport;
}

export interface StreamInconclusiveResponse {
  status:             "INCONCLUSIVE";
  reason:             string;
  reason_human:       string;
  patient?:           PatientInfo;
  frames_total:       number;
  frames_accepted:    number;
  frames_rejected:    number;
  per_frame_readings: (number | null)[];
  flags:              string[];
  timestamp:          string;
  // Present when reason === "high_variance_asymmetry"
  asymmetry_avg_deg?: number;
  asymmetry_std_deg?: number;
  deviation_avg_deg?: number;
  deviation_std_deg?: number;
  // Torch-free corner-alignment net — present when detection + pupil
  // localisation succeeded before the halt (e.g. a no-flash CLR failure).
  alignment?:         AlignmentResult;
  // Carries module8_alignment so Method B can be shown with its geometry even
  // when Method A produced nothing.
  intermediate_images?: IntermediateImages;
  /** Present only when the request carried a client_measurement. */
  parity?:            ParityReport;
}

export type StreamAnalyseResponse = StreamSuccessResponse | StreamInconclusiveResponse;

// ─── Browser / server parity (live-view comparison) ─────────────────────────

/** The browser's own measurement of the frames it sends, for side-by-side comparison. */
export interface ClientMeasurement {
  schema:                   1;
  pipeline:                 string;
  frames:                   number;
  reflex_frames:            number;
  per_frame_asymmetry_deg:  (number | null)[];
  asymmetry_deg:            number | null;
  asymmetry_std_deg:        number | null;
  tier:                     UrgencyTier | null;
  unstable:                 boolean;
  method_b: {
    h_asym:  number;
    v_asym:  number;
    verdict: "ALIGNED" | "BORDERLINE" | "ASYMMETRIC";
  } | null;
}

interface ParityPair<T> { client: T | null; server: T | null }

/** Returned by /analyse-stream when a client_measurement was sent. Never affects the result. */
export interface ParityReport {
  error?:          string;
  client_pipeline?: string;
  server_status?:  AnalysisStatus;
  asymmetry_deg?:  ParityPair<number> & { delta: number | null };
  tier?:           ParityPair<UrgencyTier> & { agree: boolean | null };
  per_frame?: {
    client:           (number | null)[];
    server:           (number | null)[];
    delta:            (number | null)[];
    median_abs_delta: number | null;
    frames_compared:  number;
  };
  method_b?: {
    verdict: ParityPair<string> & { agree: boolean | null };
    h_asym:  ParityPair<number> & { delta: number | null };
    v_asym:  ParityPair<number> & { delta: number | null };
  };
}

// ─── Batch pre-screen (school / camp) ────────────────────────────────────────

export interface BatchItemHirschberg {
  available:          boolean;
  reason?:            string;
  urgency_tier?:      UrgencyTier;
  condition_name?:    string;
  icd10_code?:        string;
  asymmetry_degrees?: number;
  deviation_degrees?: number;
  severity?:          SeverityTier;
}

export interface BatchItemAlignment {
  available:       boolean;
  verdict:         AlignmentVerdict;
  referral_flag:   boolean;
  h_asymmetry:     number | null;
  v_asymmetry:     number | null;
  left_h_ratio:    number | null;
  right_h_ratio:   number | null;
  interpretation:  string;
}

export interface BatchItemResult {
  index:         number;
  label:         string;
  status:        "SUCCESS" | "INCONCLUSIVE" | "ERROR";
  reason:        string | null;
  hirschberg:    BatchItemHirschberg | null;
  alignment:     BatchItemAlignment | null;
  referral_flag: boolean;
}

export interface BatchResponse {
  status:          "DONE" | "ERROR";
  message?:        string;
  total:           number;
  processed:       number;
  flagged:         number;
  elapsed_seconds: number;
  items:           BatchItemResult[];
  timestamp:       string;
}

// ─── Phase 3 gaze examination (/exam) ────────────────────────────────────────
// Mirrors backend/exam/capture, exam/qc and exam/calibration. "R" / "L" are the
// PATIENT's eyes; pixel coordinates are in the full, unmirrored frame.

export type ExamEye        = "R" | "L";
export type ExamProtocol   = "nine_gaze" | "five_gaze";
export type ExamConfidence = "HIGH" | "MODERATE" | "LOW";
export type XY = [number, number];

export interface ExamMeasured { value: number; error: number; confidence: ExamConfidence }

export interface ExamEllipse {
  centre_px:     XY;
  semi_major_px: number;
  semi_minor_px: number;
  angle_deg:     number;      // major axis, image coordinates, [0, 180)
}

export interface ExamEyeObservation {
  reflex_px:               XY | null;
  limbus:                  ExamEllipse | null;
  pupil_px:                XY | null;
  pupil_diameter_px:       number | null;
  iris_radius_px:          number | null;
  fissure_height_px:       number | null;
  scleral_show_px:         number | null;
  limbus_covered_fraction: number | null;
}

export interface ExamHeadPose { yaw_deg: number; pitch_deg: number; roll_deg: number }

export interface ExamFrameObservation {
  index:      number;
  t_ms:       number;
  head_pose:  ExamHeadPose | null;   // null = no face
  sharpness:  number | null;
  eyes:       Partial<Record<ExamEye, ExamEyeObservation>>;
}

export interface ExamPositionCapture { position: string; frames: ExamFrameObservation[] }

export interface ExamSessionHeader {
  session_id:       string;
  protocol_version: string;
  protocol:         ExamProtocol;
  tier:             1;
  device:           { model: string; lens: string | null; calibration_profile_id: string | null };
  settings:         { width_px: number; height_px: number; fps: number; exposure_s: number | null };
  distance_cm:      ExamMeasured | null;
  started_at:       string;
}

export interface ExamCaptureSession { header: ExamSessionHeader; captures: ExamPositionCapture[] }

export interface ExamRange { min: number; max: number }

export interface ExamPositionPlan {
  position:          string;
  kind:              "calibration" | "gaze" | "tilt";
  eye_in_orbit_deg:  XY;
  head:              { yaw_deg: number; pitch_deg: number; tolerance_deg: number; roll_deg: ExamRange | null };
  hold_s:            number;
  min_valid_frames:  number;
  instruction:       string;
  fixing_eye:        ExamEye | null;
  target_offset_cm:  number;
}

export interface ExamProtocolPlan {
  protocol_version: string;
  protocol:         ExamProtocol;
  settings: {
    distance_cm: ExamRange; distance_tolerance_cm: number; min_fps: number;
    max_exposure_s: number; recommended_frame_height_px: number;
    sharpness_canonical_iris_radius_px: number; camera: "rear"; light: string;
  };
  positions:   ExamPositionPlan[];
  conventions: string;
  kb_version:  string;
  kb_sha256:   string;
}

export interface ExamRejection { code: string; eye: ExamEye | null; detail: string }

export interface ExamPositionQC {
  position:         string;
  kind:             ExamPositionPlan["kind"];
  status:           "OK" | "INSUFFICIENT_DATA";
  reason:           "not_captured" | "too_few_valid_frames" | null;
  frames_total:     number;
  frames_valid:     number;
  min_valid_frames: number;
  reject_counts:    Record<string, number>;
  frames:           { index: number; valid: boolean; reasons: ExamRejection[] }[];
}

export interface ExamSessionQC {
  session_id:             string;
  protocol:               ExamProtocol;
  positions:              ExamPositionQC[];
  session_flags:          { code: string; detail: string }[];
  insufficient_positions: string[];
  qc_version:             string;
  kb_version:             string;
  kb_sha256:              string;
}

export interface ExamCalibration {
  session_id:       string;
  iris_diameter_mm: number;
  hirschberg: {
    pd_per_mm:       ExamMeasured;
    source:          "fitted" | "kb_default";
    fitted:          ExamMeasured | null;
    residual_rms_mm: number | null;
    points:          number;
  };
  kappa: Record<ExamEye, { temporal_mm: ExamMeasured; inferior_mm: ExamMeasured; frames: number } | null>;
  points: {
    position: string; eye: ExamEye; rotation_asked_pd: number;
    temporal_mm: ExamMeasured; inferior_mm: ExamMeasured; frames: number;
  }[];
  flags:               { code: string; eye: ExamEye | null; detail: string }[];
  calibration_version: string;
  qc_version:          string;
  kb_version:          string;
  kb_sha256:           string;
}

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
    colour:       "text-emerald-600",
    bgColour:     "bg-emerald-50",
    borderColour: "border-emerald-200",
    badgeColour:  "bg-emerald-600",
    dotColour:    "bg-emerald-500",
  },
};
