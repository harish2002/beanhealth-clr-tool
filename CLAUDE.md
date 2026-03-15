# CLAUDE.md — BeanHealth CLR Tool
## Project Reference & Coding Instructions

> This file is the single source of truth for all coding, architecture, testing, and implementation decisions.
> Claude must read and adhere to every section before generating any code.

---

## 1. PROJECT OVERVIEW

**Product Name:** BeanHealth CLR Tool
**Full Name:** Corneal Light Reflex (CLR) Asymmetry Analysis Tool
**Purpose:** Phone-based, AI-assisted strabismus (squint) triage screening tool for remote use by parents, nurses, and GPs — without needing an ophthalmologist present.
**Event:** EyeQ Innovate Hackathon 2.0
**Company:** BeanHealth (tele-squint triage startup)

### What the App Does (One Flow)
1. User opens app on phone, enters patient name and age
2. User enables phone torch and holds the camera up to both eyes
3. App captures a photo — the torch creates a small bright spot (corneal light reflex / CLR) on each cornea
4. The backend CV pipeline measures whether those bright spots are centred in each pupil
5. In a normal eye the spot is centred; in a squinting eye one spot is displaced
6. The system computes how asymmetric the displacement is, converts to clinical angle (degrees), and outputs a triage tier
7. Output: URGENT / ROUTINE / MONITOR / NORMAL with a referral recommendation

### Clinical Context
- The measurement standard used is the **Hirschberg test** — a well-established ophthalmology technique
- `1mm of CLR displacement ≈ 7° of ocular deviation`
- Outputs use ICD-10 codes for medical record compatibility
- The tool is a **screening aid**, not a diagnostic device

---

## 2. TECH STACK

### Frontend
| Layer | Technology | Notes |
|---|---|---|
| Framework | **Next.js 14** (App Router) | TypeScript, server components |
| Styling | **Tailwind CSS** | Utility-first, dark medical theme |
| Camera API | **MediaDevices API** (navigator.mediaDevices) | getUserMedia for live camera + torch |
| State Management | **Zustand** | Lightweight, no Redux |
| HTTP Client | **Axios** | For FastAPI requests |
| Image Preview | Native `<canvas>` API | For annotated result overlay |
| Deployment | **Vercel** | Auto-deploy from main branch |

### Backend
| Layer | Technology | Notes |
|---|---|---|
| Framework | **FastAPI** (Python 3.11+) | Async, OpenAPI auto-docs |
| CV Library | **OpenCV 4.x** (`cv2`) | All image processing |
| Face/Eye Detection | **Google MediaPipe** (`mediapipe`) | Iris landmark detection |
| Numerical | **NumPy** | Array ops, percentile thresholding |
| Image I/O | **Pillow** (`PIL`) | Image loading and annotation |
| Report Schema | **Pydantic v2** | Request/response validation |
| Testing | **pytest** + `pytest-asyncio` | Unit + integration tests |
| Server | **Uvicorn** | ASGI server |
| Deployment | **Railway / Render** | FastAPI container deployment |

### Dev Tooling
| Tool | Use |
|---|---|
| `python-dotenv` | Environment variables |
| `httpx` | Async HTTP testing |
| `Postman` | API manual testing |
| `pytest-cov` | Coverage reports |
| `ruff` | Python linting |
| `black` | Python formatting |
| ESLint + Prettier | TypeScript/Next.js linting |

---

## 3. PROJECT FOLDER STRUCTURE

```
beanstrabis/
│
├── CLAUDE.md                        ← This file
├── clr-visualiser.html              ← Step 2 & 3 visual explainer
│
├── frontend/                        ← Next.js app
│   ├── app/
│   │   ├── page.tsx                 ← Home / patient entry
│   │   ├── capture/page.tsx         ← Camera + torch capture
│   │   ├── result/page.tsx          ← Triage report display
│   │   └── layout.tsx
│   ├── components/
│   │   ├── CameraCapture.tsx        ← Camera + torch UI
│   │   ├── PatientForm.tsx          ← Name, age input
│   │   ├── TriageReport.tsx         ← Result card with urgency
│   │   ├── AnnotatedEye.tsx         ← Canvas overlay component
│   │   └── UrgencyBadge.tsx         ← URGENT / NORMAL badge
│   ├── lib/
│   │   ├── api.ts                   ← Axios API calls to FastAPI
│   │   └── types.ts                 ← Shared TypeScript types
│   ├── store/
│   │   └── useAppStore.ts           ← Zustand global state
│   └── public/
│       └── test-images/             ← Mock eye images for dev
│
├── backend/                         ← FastAPI app
│   ├── main.py                      ← FastAPI app entry, /analyse route
│   ├── models/
│   │   ├── request.py               ← Pydantic input schema
│   │   └── response.py              ← Pydantic output schema
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── module1_detection.py     ← Eye crop extraction
│   │   ├── module2_pupil.py         ← Pupil centre localisation
│   │   ├── module3_clr.py           ← CLR bright spot detection
│   │   ├── module4_displacement.py  ← Displacement measurement
│   │   ├── module5_asymmetry.py     ← Hirschberg angle + asymmetry
│   │   ├── module6_classify.py      ← Clinical classification
│   │   └── module7_report.py        ← Report + annotated image
│   ├── utils/
│   │   ├── image_utils.py           ← Shared image helpers
│   │   └── constants.py             ← Thresholds, ICD codes, etc.
│   └── tests/
│       ├── test_images/             ← Test image library (see §6)
│       ├── test_module1.py
│       ├── test_module2.py
│       ├── test_module3.py
│       ├── test_module4.py          ← Pure unit tests (maths)
│       ├── test_module5.py          ← Hirschberg formula tests
│       ├── test_module6.py          ← Classification tests
│       ├── test_module7.py          ← Report schema tests
│       └── test_api.py              ← Full endpoint integration tests
│
└── docs/
    └── CLR_Tool_Technical_Spec.docx ← Original spec document
```

---

## 4. THE 7 PIPELINE MODULES

These modules run sequentially in the backend. Each module receives output from the previous and passes its result to the next. If any module fails, the pipeline must return `INCONCLUSIVE` — never a wrong result.

---

### Module 1 — Eye Detection & Crop
**File:** `backend/pipeline/module1_detection.py`

**What it does:**
Uses Google MediaPipe Face Mesh to detect the face and locate both eyes. Extracts a tight crop around each eye with padding. Also identifies the iris landmark set for Module 2.

**Key logic:**
- Run `mp.solutions.face_mesh` on the input image
- Extract landmarks for left eye (indices 33–133) and right eye (indices 362–463)
- Crop with 20% horizontal and 40% vertical padding
- Extract iris landmarks (left: 468–472, right: 473–477) and store for Module 2
- Validate crop is not too small (minimum 60×40px)

**Inputs:** Raw JPEG/PNG image (bytes or numpy array)
**Outputs:**
```python
{
  "left_crop": np.ndarray,
  "right_crop": np.ndarray,
  "left_iris_landmarks": List[Tuple[float,float]],  # 5 points
  "right_iris_landmarks": List[Tuple[float,float]], # 5 points
  "confidence": float
}
```

**Failure conditions:**
- No face detected → raise `DetectionError("no_face")`
- Face detected but eyes not visible → raise `DetectionError("eyes_not_visible")`
- Eyes closed (iris landmarks missing) → raise `DetectionError("eyes_closed")`
- Profile face (not sufficiently frontal) → raise `DetectionError("not_frontal")`

---

### Module 2 — Pupil Centre Localisation
**File:** `backend/pipeline/module2_pupil.py`

**What it does:**
Pinpoints the exact centre of each pupil using two independent methods — MediaPipe landmark mean and Hough Circle Transform — then cross-validates them for a confidence-graded output.

**Key logic (step by step):**
1. Convert eye crop to grayscale: `cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)`
2. Apply CLAHE: `clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))`
3. Apply Gaussian blur: `cv2.GaussianBlur(clahe_out, (7,7), 0)`
4. **Primary estimate (MediaPipe):** Mean of 5 iris landmark (x,y) positions → `landmark_centre`
5. **Secondary estimate (Hough):** `cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=int(w*0.20), maxRadius=int(w*0.55))` → `hough_centre`
6. **Agreement check:**
   - Distance < 5px → `confidence = HIGH`, use averaged centre
   - Distance 5–15px → `confidence = MEDIUM`, use averaged centre, flag `pupil_disagreement`
   - Hough failed or distance > 15px → `confidence = LOW`, use landmark only, flag `pupil_disagreement`

**Inputs:** Eye crop (np.ndarray), iris landmarks (List of 5 points)
**Outputs:**
```python
{
  "left_pupil": Tuple[float, float],   # (x, y) in crop coords
  "right_pupil": Tuple[float, float],
  "left_iris_radius": float,           # in pixels
  "right_iris_radius": float,
  "left_confidence": str,              # HIGH / MEDIUM / LOW
  "right_confidence": str,
  "flags": List[str]                   # e.g. ["pupil_disagreement_left"]
}
```

**Failure conditions:**
- Both methods fail for one eye → raise `DetectionError("pupil_not_found_{left|right}")`

---

### Module 3 — CLR Bright Spot Detection ← Most Critical
**File:** `backend/pipeline/module3_clr.py`

**What it does:**
Locates the corneal light reflex — the tiny bright torch reflection on the cornea surface. Uses adaptive percentile thresholding, connected component analysis, and a 3-way filter to identify the correct blob.

**Key logic (step by step):**
1. Convert crop to grayscale
2. **Adaptive threshold:** `threshold = np.percentile(gray, 97)` — top 3% brightest pixels
3. **Flash validation:** If `np.max(gray) < 240` → raise `CLRError("no_flash")` → INCONCLUSIVE
4. **Binary mask:** `_, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)`
5. **Connected components:** `cv2.connectedComponentsWithStats(mask, connectivity=8)` → labels, stats
6. **3-way filter for each blob:**
   - ① **Location:** Centroid must be within central 80% of crop width AND height
   - ② **Area:** `0.005 * iris_area < blob_area < 0.15 * iris_area`
   - ③ **Circularity:** `(4π * area) / (perimeter²) > 0.5`
7. **Selection:** Among blobs passing all 3 filters, select the largest
8. Return centroid of selected blob as CLR position

**Inputs:** Eye crop (np.ndarray), iris_radius (float) from Module 2
**Outputs:**
```python
{
  "left_clr": Tuple[float, float],    # (x, y) in crop coords
  "right_clr": Tuple[float, float],
  "left_clr_confidence": float,       # circularity score 0–1
  "right_clr_confidence": float,
  "flags": List[str]                  # e.g. ["no_flash", "no_reflex_left"]
}
```

**Failure conditions:**
- Max pixel < 240 → flag `no_flash` → INCONCLUSIVE
- No blob passes filter for one eye → flag `no_reflex_{left|right}` → INCONCLUSIVE
- Multiple high-confidence blobs (ambiguous) → flag `ambiguous_reflex_{left|right}`

---

### Module 4 — Displacement Measurement
**File:** `backend/pipeline/module4_displacement.py`

**What it does:**
Calculates how far the CLR is displaced from the pupil centre for each eye, normalised by iris size so the result is independent of camera distance.

**Key logic:**
```python
dx = clr_x - pupil_x
dy = clr_y - pupil_y
magnitude = sqrt(dx² + dy²)
normalised = magnitude / iris_radius          # scale-invariant
direction = atan2(dy, dx)                     # angle in radians
direction_label = map_to_cardinal(direction)  # nasal/temporal/superior/inferior
```

**Inputs:** Pupil centres, CLR positions, iris radii (all from Modules 2 & 3)
**Outputs:**
```python
{
  "left_displacement_px": float,
  "right_displacement_px": float,
  "left_displacement_norm": float,   # ratio of iris radius (0.0–1.0+)
  "right_displacement_norm": float,
  "left_direction": str,             # nasal / temporal / superior / inferior
  "right_direction": str,
  "left_dx": float, "left_dy": float,
  "right_dx": float, "right_dy": float
}
```

---

### Module 5 — Asymmetry Score & Hirschberg Angle
**File:** `backend/pipeline/module5_asymmetry.py`

**What it does:**
Compares displacement between both eyes to produce an asymmetry score, then converts normalised displacement to a clinical angle in degrees using the Hirschberg formula.

**Key logic:**
```python
# Asymmetry (the red flag — not just absolute displacement)
asymmetry_score = abs(left_displacement_norm - right_displacement_norm)

# Hirschberg conversion (1mm ≈ 7°; iris ≈ 11.5mm diameter ≈ 5.75mm radius)
HIRSCHBERG_CONSTANT = 7.0          # degrees per mm
MM_PER_IRIS_RADIUS = 5.75          # mm
displacement_mm = displacement_norm * MM_PER_IRIS_RADIUS
angle_degrees = displacement_mm * HIRSCHBERG_CONSTANT

# Severity thresholds
if angle_degrees < 5:    severity = "NORMAL"
elif angle_degrees < 15: severity = "MILD"
elif angle_degrees < 30: severity = "MODERATE"
else:                    severity = "SEVERE"
```

**Inputs:** Displacement data from Module 4
**Outputs:**
```python
{
  "asymmetry_score": float,      # 0.0 = perfectly symmetric
  "dominant_eye": str,           # "left" or "right" (more displaced)
  "deviation_degrees": float,    # Hirschberg angle
  "deviation_mm": float,
  "severity": str,               # NORMAL / MILD / MODERATE / SEVERE
}
```

---

### Module 6 — Clinical Classification
**File:** `backend/pipeline/module6_classify.py`

**What it does:**
Names the type of squint based on displacement direction, assigns ICD-10 code, determines urgency tier, and generates a referral recommendation with timeframe.

**Classification table:**
| Displacement Direction | Condition | ICD-10 |
|---|---|---|
| Nasal (inward) | Esotropia | H50.0x |
| Temporal (outward) | Exotropia | H50.1x |
| Superior (upward) | Hypertropia | H50.2x |
| Inferior (downward) | Hypotropia | H50.2x |
| Minimal / symmetric | Orthophoria | H50.40 |

**Urgency tier mapping:**
```python
SEVERE   → URGENT   → "Refer to ophthalmology within 1 week"
MODERATE → ROUTINE  → "Refer to ophthalmology within 4 weeks"
MILD     → MONITOR  → "Monitor and re-screen in 3 months"
NORMAL   → NORMAL   → "No referral required"
```

**Inputs:** Asymmetry score, severity, direction labels from Modules 4 & 5
**Outputs:**
```python
{
  "condition_name": str,          # e.g. "Esotropia"
  "icd10_code": str,              # e.g. "H50.01"
  "urgency_tier": str,            # URGENT / ROUTINE / MONITOR / NORMAL
  "referral_recommendation": str,
  "timeframe": str,
}
```

---

### Module 7 — Report Generation
**File:** `backend/pipeline/module7_report.py`

**What it does:**
Assembles all upstream outputs into a structured JSON report. Generates an annotated image with all detected landmarks drawn on it. Handles INCONCLUSIVE gracefully.

**Report JSON schema:**
```json
{
  "status": "SUCCESS | INCONCLUSIVE | ERROR",
  "patient": {
    "name": "string",
    "age": "integer"
  },
  "result": {
    "urgency_tier": "URGENT | ROUTINE | MONITOR | NORMAL",
    "condition_name": "string",
    "icd10_code": "string",
    "deviation_degrees": "float",
    "asymmetry_score": "float",
    "severity": "NORMAL | MILD | MODERATE | SEVERE",
    "referral_recommendation": "string",
    "timeframe": "string",
    "narrative": "string"
  },
  "technical": {
    "left_pupil": [x, y],
    "right_pupil": [x, y],
    "left_clr": [x, y],
    "right_clr": [x, y],
    "left_displacement_norm": "float",
    "right_displacement_norm": "float",
    "confidence": "HIGH | MEDIUM | LOW",
    "flags": ["list of warning flags"]
  },
  "annotated_image_b64": "base64 encoded JPEG string",
  "timestamp": "ISO 8601 string"
}
```

**Annotation drawing instructions:**
- Blue dot: pupil centre (both eyes)
- Amber dot: CLR position (both eyes)
- Green circle: Hough pupil estimate
- White line: displacement vector (pupil → CLR)
- Coloured border: Red=URGENT, Orange=ROUTINE, Yellow=MONITOR, Green=NORMAL

---

## 5. IMPLEMENTATION PLAN — 4 PHASES

### Phase 1 — Backend Core (FastAPI + All 7 Modules)
> Goal: A working Python pipeline that processes a real eye image and returns a structured dict

```
Task 1.1  Set up FastAPI project, venv, requirements.txt
Task 1.2  Implement Module 1 (eye crop) — test on 5 face photos
Task 1.3  Implement Module 2 (pupil centre) — visualise output on test crops
Task 1.4  Implement Module 3 (CLR detection) — most critical, test with flash + no flash
Task 1.5  Implement Module 4 (displacement) — pure maths, unit test immediately
Task 1.6  Implement Module 5 (Hirschberg angle + asymmetry)
Task 1.7  Implement Module 6 (classification + ICD codes)
Task 1.8  Implement Module 7 (report generator + image annotation)
Task 1.9  Wire all 7 modules into a single pipeline runner function
Task 1.10 Run pipeline end-to-end on 10 test images, review all outputs visually
```

**Exit criteria for Phase 1:**
- Pipeline processes a real photo end-to-end without crashing
- Annotated image visually shows correct pupil and CLR positions
- INCONCLUSIVE returned correctly for no-flash photos

---

### Phase 2 — API Contract
> Goal: FastAPI endpoint that accepts an image and returns the report JSON

```
Task 2.1  Create POST /analyse endpoint
Task 2.2  Create Pydantic request/response models (see §4 Module 7 schema)
Task 2.3  Add error handling middleware (INCONCLUSIVE, DetectionError, etc.)
Task 2.4  Add CORS middleware for Next.js frontend
Task 2.5  Create GET /health endpoint
Task 2.6  Test all endpoints with Postman using test image library
Task 2.7  Write test_api.py integration tests
Task 2.8  Freeze requirements.txt, document API in OpenAPI (auto-generated)
```

**Exit criteria for Phase 2:**
- `POST /analyse` returns valid JSON for all 6 test image categories
- INCONCLUSIVE returned for no-flash, no-face, eyes-closed images
- 422 returned for malformed requests

---

### Phase 3 — Next.js Frontend
> Goal: A working mobile-first UI that captures a photo, sends it to the API, and shows the report
> Use a **mock JSON response** during this phase — do NOT wait for Phase 2 to be perfect

```
Task 3.1  Set up Next.js 14 project with Tailwind CSS and Zustand
Task 3.2  Build PatientForm page (name + age input)
Task 3.3  Build CameraCapture component (getUserMedia + torch toggle + capture)
Task 3.4  Build mock API response in lib/api.ts (returns hardcoded report JSON)
Task 3.5  Build TriageReport page (urgency badge, condition, referral recommendation)
Task 3.6  Build AnnotatedEye canvas component (overlays pupil/CLR dots on image)
Task 3.7  Build UrgencyBadge component (colour coded: Red/Orange/Yellow/Green)
Task 3.8  Handle INCONCLUSIVE state — show retry screen with reason
Task 3.9  Handle loading state — show processing animation
Task 3.10 Mobile-first testing: open on phone browser, test camera + torch
```

**Exit criteria for Phase 3:**
- Full UI flow works on mobile browser (iOS Safari + Android Chrome)
- All 4 urgency states (URGENT/ROUTINE/MONITOR/NORMAL) render correctly
- INCONCLUSIVE screen shows clear reason and retry button
- Non-technical person can understand the result screen

---

### Phase 4 — Integration, Wire & Polish
> Goal: Real API connected, edge cases handled, demo-ready

```
Task 4.1  Update lib/api.ts to call real FastAPI /analyse endpoint
Task 4.2  Test full flow on real device with real eye photos
Task 4.3  Handle all network error states (timeout, server down)
Task 4.4  Stress test with: different lighting, glasses, dark irises, shaky hands
Task 4.5  Human ground truth test (see §6 Phase 4 testing)
Task 4.6  Final UI polish — typography, spacing, transitions
Task 4.7  Deploy FastAPI to Railway/Render
Task 4.8  Deploy Next.js to Vercel
Task 4.9  Full end-to-end demo run — record a working demo video
Task 4.10 Code freeze
```

---

## 6. TESTING PROTOCOLS

### Test Image Library (required before Phase 1)
```
backend/tests/test_images/
├── flash_on_normal/        → expect: NORMAL, asymmetry < 0.1
├── flash_on_mild/          → expect: MONITOR, 5–15°
├── flash_on_moderate/      → expect: ROUTINE, 15–30°
├── flash_on_severe/        → expect: URGENT, 30°+
├── flash_off/              → expect: INCONCLUSIVE (no_flash flag)
├── glasses/                → expect: INCONCLUSIVE or low confidence
├── eyes_closed/            → expect: DetectionError (eyes_closed)
├── no_face/                → expect: DetectionError (no_face)
└── simulated_squint/       → person deliberately crossing one eye
```

---

### Module 1 Testing
```
Pass criteria: 90%+ of frontal face photos produce clean, centred eye crops

✓ Iris fully inside the crop
✓ Padding present on all sides
✓ Both eyes detected independently
✓ Crops saved to disk and visually reviewed

Edge cases to verify:
✗ Glasses → no_detection or low confidence
✗ Profile face → no_detection
✗ Eyes closed → eyes_closed flag
✗ Child face (smaller iris) → still detects correctly
```

---

### Module 2 Testing
```
Pass criteria: Pupil centre dot falls visually inside the pupil on 90%+ of test crops
Method: Draw detected centre as dot on crop, save annotated image, review manually

✓ Blue dot (landmark) and green circle (Hough) within 5px for normal images
✓ Works on dark brown irises (not just light irises)
✓ HIGH confidence on clear, well-lit images
✓ Correct confidence downgrade for difficult cases

Edge cases:
✗ Partial eyelid closure
✗ Strong glare on iris
✗ Motion blur from shaky hands
```

---

### Module 3 Testing ← Most Critical
```
Pass criteria:
  - Amber dot lands on corneal bright spot in 85%+ of flash-on photos
  - INCONCLUSIVE returned for ALL flash-off photos (never a wrong position)

Method: Annotate CLR position on crop image, review every single one

✓ Rejects glasses glare (larger, irregular shape → fails circularity check)
✓ Rejects eyelid edge highlights (outside central 80% → fails location check)
✓ Rejects small noise blobs (below 0.5% iris area → fails size check)
✓ Picks correct blob when multiple candidates exist

Edge cases:
✗ No flash → must return INCONCLUSIVE (not a guessed position)
✗ Two bright spots (glasses + cornea) → must pick corneal one
✗ Very bright outdoor daylight washing out CLR
✗ Dark skin tone (different sclera/cornea contrast ratio)
```

---

### Module 4 Testing (Pure Unit Tests)
```python
# test_module4.py
def test_zero_displacement():
    # CLR exactly at pupil centre
    result = measure_displacement(pupil=(40,40), clr=(40,40), iris_r=30)
    assert result["magnitude"] == 0.0

def test_known_displacement():
    # 10px right, iris=40px → norm = 0.25
    result = measure_displacement(pupil=(40,40), clr=(50,40), iris_r=40)
    assert abs(result["normalised"] - 0.25) < 0.001
    assert result["direction"] == "temporal"

def test_scale_invariance():
    # Same physical displacement, different image sizes
    r1 = measure_displacement(pupil=(40,40), clr=(50,40), iris_r=40)   # small
    r2 = measure_displacement(pupil=(80,80), clr=(100,80), iris_r=80)  # large
    assert abs(r1["normalised"] - r2["normalised"]) < 0.001

def test_direction_all_quadrants():
    cases = [
        ((40,40), (50,40), "temporal"),
        ((40,40), (30,40), "nasal"),
        ((40,40), (40,30), "superior"),
        ((40,40), (40,50), "inferior"),
    ]
    for pupil, clr, expected_dir in cases:
        r = measure_displacement(pupil=pupil, clr=clr, iris_r=40)
        assert r["direction"] == expected_dir
```

---

### Module 5 Testing (Hirschberg Validation)
```python
# test_module5.py — validate against known clinical values
def test_hirschberg_1mm():
    # 1mm displacement → ~7°
    result = compute_angle(displacement_norm=1/5.75)  # 1mm / 5.75mm radius
    assert abs(result["degrees"] - 7.0) < 0.5

def test_severity_thresholds():
    assert compute_angle_severity(4.9)["severity"] == "NORMAL"
    assert compute_angle_severity(5.0)["severity"] == "MILD"
    assert compute_angle_severity(15.0)["severity"] == "MODERATE"
    assert compute_angle_severity(30.0)["severity"] == "SEVERE"

def test_symmetric_eyes_normal():
    # Both eyes displaced identically → asymmetry should be near 0
    result = compute_asymmetry(left_norm=0.2, right_norm=0.2)
    assert result["asymmetry_score"] < 0.01
```

---

### Module 6 Testing (Classification)
```python
# test_module6.py — parametrized direction → condition mapping
@pytest.mark.parametrize("direction,expected_condition,expected_icd", [
    ("nasal",    "Esotropia",   "H50.01"),
    ("temporal", "Exotropia",   "H50.11"),
    ("superior", "Hypertropia", "H50.21"),
    ("inferior", "Hypotropia",  "H50.22"),
])
def test_condition_mapping(direction, expected_condition, expected_icd):
    result = classify(direction=direction, severity="MODERATE")
    assert result["condition_name"] == expected_condition
    assert result["icd10_code"] == expected_icd

@pytest.mark.parametrize("severity,expected_tier", [
    ("SEVERE",   "URGENT"),
    ("MODERATE", "ROUTINE"),
    ("MILD",     "MONITOR"),
    ("NORMAL",   "NORMAL"),
])
def test_urgency_mapping(severity, expected_tier):
    result = classify(direction="nasal", severity=severity)
    assert result["urgency_tier"] == expected_tier
```

---

### Module 7 Testing
```python
# test_module7.py
def test_report_schema_complete():
    report = generate_report(mock_pipeline_output)
    required_fields = ["status", "patient", "result", "technical",
                       "annotated_image_b64", "timestamp"]
    for field in required_fields:
        assert field in report

def test_inconclusive_report():
    # When upstream module raises DetectionError → status must be INCONCLUSIVE
    report = generate_report(error=DetectionError("no_flash"))
    assert report["status"] == "INCONCLUSIVE"
    assert "result" not in report or report["result"] is None

def test_annotated_image_is_valid_base64():
    import base64
    report = generate_report(mock_pipeline_output)
    decoded = base64.b64decode(report["annotated_image_b64"])
    assert decoded[:2] == b'\xff\xd8'  # JPEG magic bytes
```

---

### Phase 2 API Integration Tests
```python
# test_api.py
@pytest.mark.asyncio
async def test_analyse_normal_photo():
    async with AsyncClient(app=app, base_url="http://test") as client:
        with open("test_images/flash_on_normal/img1.jpg", "rb") as f:
            response = await client.post("/analyse",
                files={"image": f},
                data={"patient_name": "Test", "patient_age": "5"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["result"]["urgency_tier"] == "NORMAL"

@pytest.mark.asyncio
async def test_analyse_no_flash_returns_inconclusive():
    async with AsyncClient(app=app, base_url="http://test") as client:
        with open("test_images/flash_off/img1.jpg", "rb") as f:
            response = await client.post("/analyse", files={"image": f},
                data={"patient_name": "Test", "patient_age": "5"})
    assert response.status_code == 200
    assert response.json()["status"] == "INCONCLUSIVE"

@pytest.mark.asyncio
async def test_analyse_malformed_input():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/analyse",
            files={"image": b"not an image"},
            data={"patient_name": "Test", "patient_age": "5"})
    assert response.status_code == 422
```

---

### Phase 4 Human Ground Truth Test
```
Protocol:
  Step 1: Find 5–10 adult volunteers
  Step 2: Ask each to look straight at the phone camera with torch on, from ~30cm
  Step 3: Run tool — record urgency tier and deviation angle
  Step 4: Compare against visual inspection (are their eyes straight?)
  Step 5: Ask 2–3 volunteers to deliberately cross one eye
  Step 6: Verify tool detects the asymmetry

Pass criteria:
  ✓ All straight-eyed volunteers → NORMAL or MONITOR (< 5° false positive rate)
  ✓ Simulated squint → MONITOR or higher detected
  ✓ Tool returns INCONCLUSIVE (not wrong result) when flash is off
  ✓ Tool processes image in under 3 seconds
```

---

## 7. CODING CONVENTIONS & RULES

### Python (Backend)
- Python 3.11+ only
- All functions must have type hints
- All modules must have docstrings
- Use Pydantic v2 for all API models — never raw dicts in route handlers
- Never swallow exceptions silently — always log and raise appropriate CLR errors
- Each pipeline module is a standalone function — no side effects, no global state
- All image arrays are `np.ndarray` with `dtype=uint8` unless stated otherwise
- Image coordinates are always `(x, y)` — x is horizontal, y is vertical

### TypeScript (Frontend)
- Strict mode enabled
- All API response types defined in `lib/types.ts`
- No `any` types — ever
- Use `async/await` — no raw `.then()` chains
- Camera and torch logic lives exclusively in `CameraCapture.tsx`
- Never hardcode API URLs — always use `process.env.NEXT_PUBLIC_API_URL`

### Error Handling Rules
```
Backend error hierarchy:
  DetectionError     → No face / eyes not found → INCONCLUSIVE
  CLRError           → No flash / no reflex found → INCONCLUSIVE
  PipelineError      → Unexpected crash mid-pipeline → 500 + ERROR status
  ValidationError    → Bad request input → 422

Frontend error handling:
  INCONCLUSIVE → Show retry screen with specific reason
  Network error → Show "check connection" screen with retry
  Timeout (>10s) → Show timeout screen
  500 error → Show generic error, log to console
```

---

## 8. CONSTANTS & THRESHOLDS

```python
# backend/utils/constants.py

# Module 2 — Pupil localisation
PUPIL_AGREEMENT_THRESHOLD_PX = 5      # pixels — HIGH confidence
PUPIL_WARNING_THRESHOLD_PX = 15       # pixels — MEDIUM confidence
HOUGH_DP = 1
HOUGH_MIN_DIST = 50
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30
HOUGH_MIN_RADIUS_RATIO = 0.20         # fraction of crop width
HOUGH_MAX_RADIUS_RATIO = 0.55
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (4, 4)
GAUSSIAN_KERNEL = (7, 7)

# Module 3 — CLR detection
CLR_PERCENTILE_THRESHOLD = 97         # top 3% brightest pixels
CLR_MIN_PEAK_BRIGHTNESS = 240         # if max < this → no flash
CLR_MIN_AREA_RATIO = 0.005            # of iris area
CLR_MAX_AREA_RATIO = 0.15
CLR_MIN_CIRCULARITY = 0.5             # 4π·area/perimeter²
CLR_LOCATION_MARGIN = 0.10            # 10% margin from edge

# Module 5 — Hirschberg formula
HIRSCHBERG_CONSTANT = 7.0             # degrees per mm
IRIS_RADIUS_MM = 5.75                 # mm (average adult)
SEVERITY_MILD_DEG = 5.0
SEVERITY_MODERATE_DEG = 15.0
SEVERITY_SEVERE_DEG = 30.0

# Module 6 — ICD-10 codes
ICD10 = {
    "esotropia":   "H50.01",
    "exotropia":   "H50.11",
    "hypertropia": "H50.21",
    "hypotropia":  "H50.22",
    "orthophoria": "H50.40",
}
```

---

## 9. API SPECIFICATION

### POST /analyse
**Request:** `multipart/form-data`
```
image        : File (JPEG/PNG, required)
patient_name : str (required)
patient_age  : int (required, 1–120)
```

**Response 200 — SUCCESS:**
```json
{
  "status": "SUCCESS",
  "patient": { "name": "string", "age": 5 },
  "result": {
    "urgency_tier": "NORMAL",
    "condition_name": "Orthophoria",
    "icd10_code": "H50.40",
    "deviation_degrees": 2.3,
    "asymmetry_score": 0.04,
    "severity": "NORMAL",
    "referral_recommendation": "No referral required",
    "timeframe": "N/A",
    "narrative": "No significant corneal light reflex asymmetry detected..."
  },
  "technical": { ... },
  "annotated_image_b64": "...",
  "timestamp": "2026-03-14T10:00:00Z"
}
```

**Response 200 — INCONCLUSIVE:**
```json
{
  "status": "INCONCLUSIVE",
  "reason": "no_flash",
  "reason_human": "No torch/flash detected in the image. Please enable torch and retry.",
  "flags": ["no_flash"],
  "timestamp": "2026-03-14T10:00:00Z"
}
```

**Response 422:** Malformed input (Pydantic validation error)
**Response 500:** Internal pipeline crash (log full traceback server-side)

### GET /health
```json
{ "status": "ok", "version": "1.0.0" }
```

---

## 10. DO NOT DO (Hard Rules)

```
✗ Never return a triage result when INCONCLUSIVE conditions are met
✗ Never use a fixed brightness threshold — always use np.percentile(img, 97)
✗ Never skip the 3-way CLR filter — all 3 checks (location, area, circularity) are mandatory
✗ Never hardcode image sizes — always compute radii as ratios of iris_radius
✗ Never expose raw tracebacks in API responses — log server-side, return generic error
✗ Never process images larger than 4000×3000 without downscaling first (performance)
✗ Never store patient images beyond the request lifecycle (privacy)
✗ Never commit .env files or secrets
✗ Never use any types in TypeScript
✗ Never auto-submit the report without the user pressing a button
```

---

## 11. QUICK REFERENCE — PIPELINE DATA FLOW

```
[Phone Camera JPEG]
        │
        ▼
  Module 1: Eye Crop
  └─ left_crop, right_crop, iris_landmarks
        │
        ▼
  Module 2: Pupil Centre
  └─ left_pupil(x,y), right_pupil(x,y), iris_radii, confidence
        │
        ▼
  Module 3: CLR Detection  ← TORCH MUST BE ON
  └─ left_clr(x,y), right_clr(x,y), confidence, flags
        │
        ▼
  Module 4: Displacement
  └─ displacement_norm (left, right), direction (left, right)
        │
        ▼
  Module 5: Asymmetry + Angle
  └─ asymmetry_score, deviation_degrees, severity
        │
        ▼
  Module 6: Clinical Classification
  └─ condition_name, icd10_code, urgency_tier, referral
        │
        ▼
  Module 7: Report + Annotated Image
  └─ Full JSON report + base64 annotated JPEG
        │
        ▼
  POST /analyse response → Next.js frontend → Triage Report UI
```

---

*Last updated: 2026-03-14*
*Project: BeanHealth · EyeQ Innovate Hackathon 2.0*
