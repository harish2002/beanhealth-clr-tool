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

# Eye corner (canthus) landmarks — used by Module 8 (corner alignment).
# Pipeline "left"  = subject's LEFT eye  = image-RIGHT half.
# Pipeline "right" = subject's RIGHT eye = image-LEFT  half.
# Each tuple is (inner / medial canthus, outer / lateral canthus).
# These MUST match the eye each *_EYE_BOUNDARY set describes, or Module 8
# projects a pupil onto the OTHER eye's canthus axis and h_ratio lands far
# outside its valid 0..1 range.
# LEFT_EYE_BOUNDARY contains 33/133  -> left  eye corners come from {33, 133}
# RIGHT_EYE_BOUNDARY contains 362/263 -> right eye corners come from {362, 263}
# 133 and 362 are the medial (nasal) canthi; 33 and 263 are the lateral ones.
LEFT_EYE_CORNERS  = (133, 33)    # left  eye: inner=133, outer=33
RIGHT_EYE_CORNERS = (362, 263)   # right eye: inner=362, outer=263

# ─────────────────────────────────────────────
# Module 8 — Pupil-vs-corner alignment (CLR-free screening net)
# ─────────────────────────────────────────────
# Horizontal asymmetry = |left_h_ratio - right_h_ratio|, where h_ratio is the
# pupil's fractional position along the inner→outer canthus axis (0=inner/nasal,
# 1=outer/temporal). Symmetric eyes have near-equal ratios.
ALIGN_H_ALIGNED_MAX    = 0.05   # < this → ALIGNED
ALIGN_H_BORDERLINE_MAX = 0.10   # < this → BORDERLINE, else ASYMMETRIC
# Vertical asymmetry uses the same fractions perpendicular to the canthus axis.
ALIGN_V_ALIGNED_MAX    = 0.06
ALIGN_V_BORDERLINE_MAX = 0.12

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
CLR_MIN_PEAK_BRIGHTNESS   = 235   # if max pixel < this → no flash (was 240; phones vary)

# Primary detection pass — balanced precision / recall
CLR_MIN_AREA_RATIO        = 0.004 # of iris area (lower = catch small reflexes)
CLR_MAX_AREA_RATIO        = 0.25  # was 0.15; phone torches bloom at 30 cm
CLR_MIN_CIRCULARITY       = 0.35  # was 0.5; bloomed LED reflex isn't perfectly round
CLR_LOCATION_MARGIN       = 0.10  # 10% safe margin from crop edge

# Rescue pass — only used if primary pass finds 0 blobs
# Catches very bright / large / irregular reflexes that still beat background
CLR_RESCUE_MAX_AREA_RATIO = 0.45  # allow large bloom
CLR_RESCUE_MIN_CIRCULARITY = 0.20 # allow irregular shape
# (rescue uses the same min_area and location margin as primary)

# ─────────────────────────────────────────────
# Module 5 — Hirschberg Angle
# ─────────────────────────────────────────────

HIRSCHBERG_CONSTANT  = 7.0    # degrees per mm of CLR displacement
IRIS_RADIUS_MM       = 5.75   # average adult iris radius in mm

# Prism dioptres are DEFINED as 100 × tan(angle) — 1Δ deflects light 1 cm
# at 1 m.  The conversion is therefore trigonometric, not a fixed ratio:
#   7°  →  12.3Δ        15° →  26.8Δ        30° →  57.7Δ
# A linear "PD per degree" constant (the 15/7 or 22/7 shortcuts that
# circulate in Hirschberg teaching) diverges badly above ~10° and is not
# used here.  See degrees_to_prism_dioptres() in module5_asymmetry.

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

# Module 2 — dark-blob pupil sanity bound.
# The pupil is anatomically concentric with the iris, so its centre cannot sit
# far from the iris centre.  A "dark pupil" found beyond this fraction of the
# iris radius is eyelash, eyeliner, brow shadow or the eye corner — never a
# pupil.  Measuring from such a point produces a large false displacement in
# one eye and hence a false asymmetry, so the candidate is rejected and the
# iris-circle estimate is used instead.
DARK_PUPIL_MAX_IRIS_RADII = 0.6
