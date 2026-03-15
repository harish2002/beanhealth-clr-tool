"""
BeanHealth CLR Tool — Global Constants & Thresholds
All magic numbers live here. Never hardcode these in pipeline modules.
"""

# ─────────────────────────────────────────────
# Module 1 — Eye Detection & Crop
# ─────────────────────────────────────────────

# MediaPipe Face Mesh iris landmark indices
# Left eye iris:  landmarks 468–472 (5 points: centre + 4 boundary)
# Right eye iris: landmarks 473–477 (5 points: centre + 4 boundary)
LEFT_IRIS_INDICES  = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

# Left eye boundary landmarks (approximate bounding box)
LEFT_EYE_BOUNDARY  = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                       173, 157, 158, 159, 160, 161, 246]

# Right eye boundary landmarks (approximate bounding box)
RIGHT_EYE_BOUNDARY = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                       466, 388, 387, 386, 385, 384, 398]

# Crop padding ratios (fraction of eye bounding box size added as padding)
CROP_PAD_HORIZONTAL = 0.35   # 35% of eye width added each side
CROP_PAD_VERTICAL   = 0.50   # 50% of eye height added each side

# Minimum acceptable crop size in pixels
MIN_CROP_WIDTH  = 60
MIN_CROP_HEIGHT = 40

# Minimum iris landmark confidence for detection to be valid
MIN_FACE_CONFIDENCE = 0.7

# ─────────────────────────────────────────────
# Module 2 — Pupil Centre Localisation
# ─────────────────────────────────────────────

PUPIL_AGREEMENT_HIGH_PX   = 5    # < 5px difference → HIGH confidence
PUPIL_AGREEMENT_MEDIUM_PX = 15   # 5–15px → MEDIUM, > 15px → LOW

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (4, 4)
GAUSSIAN_KERNEL  = (7, 7)

HOUGH_DP        = 1
HOUGH_MIN_DIST  = 50
HOUGH_PARAM1    = 50
HOUGH_PARAM2    = 30
HOUGH_MIN_RADIUS_RATIO = 0.20   # fraction of crop width
HOUGH_MAX_RADIUS_RATIO = 0.55

# ─────────────────────────────────────────────
# Module 3 — CLR Bright Spot Detection
# ─────────────────────────────────────────────

CLR_PERCENTILE_THRESHOLD  = 97    # top 3% brightest pixels
CLR_MIN_PEAK_BRIGHTNESS   = 240   # if max pixel < this → no flash
CLR_MIN_AREA_RATIO        = 0.005 # of iris area
CLR_MAX_AREA_RATIO        = 0.15
CLR_MIN_CIRCULARITY       = 0.5   # 4π·area / perimeter²
CLR_LOCATION_MARGIN       = 0.10  # 10% safe margin from crop edge

# ─────────────────────────────────────────────
# Module 5 — Hirschberg Angle
# ─────────────────────────────────────────────

HIRSCHBERG_CONSTANT  = 7.0    # degrees per mm of CLR displacement
IRIS_RADIUS_MM       = 5.75   # average adult iris radius in mm

SEVERITY_MILD_DEG     = 5.0
SEVERITY_MODERATE_DEG = 15.0
SEVERITY_SEVERE_DEG   = 30.0

# ─────────────────────────────────────────────
# Module 6 — Clinical Classification
# ─────────────────────────────────────────────

ICD10_CODES = {
    "esotropia":   "H50.01",
    "exotropia":   "H50.11",
    "hypertropia": "H50.21",
    "hypotropia":  "H50.22",
    "orthophoria": "H50.40",
}

# Alias used by modules — prefer ICD10 over ICD10_CODES
ICD10 = ICD10_CODES

URGENCY_TIER = {
    "SEVERE":   "URGENT",
    "MODERATE": "ROUTINE",
    "MILD":     "MONITOR",
    "NORMAL":   "NORMAL",
}

REFERRAL_TEXT = {
    "URGENT":  "Refer to ophthalmology within 1 week",
    "ROUTINE": "Refer to ophthalmology within 4 weeks",
    "MONITOR": "Monitor and re-screen in 3 months",
    "NORMAL":  "No referral required",
}

TIMEFRAME = {
    "URGENT":  "Within 1 week",
    "ROUTINE": "Within 4 weeks",
    "MONITOR": "3 months",
    "NORMAL":  "N/A",
}

# ─────────────────────────────────────────────
# Severity string constants
# ─────────────────────────────────────────────

SEVERITY_NORMAL   = "NORMAL"
SEVERITY_MILD     = "MILD"
SEVERITY_MODERATE = "MODERATE"
SEVERITY_SEVERE   = "SEVERE"

# ─────────────────────────────────────────────
# Urgency string constants
# ─────────────────────────────────────────────

URGENCY_URGENT  = "URGENT"
URGENCY_ROUTINE = "ROUTINE"
URGENCY_MONITOR = "MONITOR"
URGENCY_NORMAL  = "NORMAL"

# ─────────────────────────────────────────────
# Direction string constants (Module 4)
# ─────────────────────────────────────────────

DIRECTION_NASAL    = "nasal"
DIRECTION_TEMPORAL = "temporal"
DIRECTION_SUPERIOR = "superior"
DIRECTION_INFERIOR = "inferior"
