"""
Browser/server parity — step 1 of 2: dump eye crops with the server's answers.

Runs the server's module 1 on every face photo in backend/tests/test_images,
then for each eye crop records the server's dark-pupil centre (module 2) and
corneal reflex (module 3). Each real crop also gets synthetic reflexes painted
on — random position, size and brightness, some with glare decoys, one bloomed
enough to need the rescue pass — so module 3 is exercised well beyond the few
real reflexes in the photos.

Step 2 (tools/parity/compare_crops.ts) runs the browser port on the same pixels.

    cd backend && venv/bin/python ../tools/parity/dump_crops.py /tmp/parity
    node ../tools/parity/compare_crops.ts /tmp/parity

The pupil is compared from the iris-landmark seed on both sides (the server
normally seeds from Hough, which the browser does not port); this isolates the
two implementations of the same dark-pupil method.
"""

import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

from pipeline.module1_detection import detect_and_crop_eyes          # noqa: E402
from pipeline.module2_pupil import (                                   # noqa: E402
    _dark_pupil_centre, _iris_radius_in_crop, _landmark_centre, _map_landmarks_to_crop,
)
from pipeline.module3_clr import _detect_clr_one_eye                  # noqa: E402
from utils.exceptions import CLRError                                  # noqa: E402
from utils.image_utils import load_image_from_path                    # noqa: E402

VARIANTS_PER_CROP = 6


def server_reflex(crop, iris_r, pupil, eye):
    try:
        (x, y), _circ, _peak, _ratio = _detect_clr_one_eye(crop, iris_r, pupil, eye, [])
        return {"status": "ok", "x": x, "y": y}
    except CLRError as e:
        return {"status": "no_flash" if "no_flash" in e.code else "no_reflex"}


def main(out: Path) -> None:
    logging.disable(logging.WARNING)   # the pipeline's per-crop warnings drown the result
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    photos = sorted(p for p in glob.glob(str(BACKEND / "tests/test_images/**/*"), recursive=True)
                    if p.lower().endswith((".jpg", ".jpeg", ".png")))
    cases = []
    for path in photos:
        try:
            det = detect_and_crop_eyes(load_image_from_path(path))
        except Exception:        # no face, eyes closed, unreadable — nothing to compare
            continue
        for eye in ("left", "right"):
            crop = getattr(det, f"{eye}_crop")
            lms = _map_landmarks_to_crop(getattr(det, f"{eye}_iris_landmarks"), getattr(det, f"{eye}_crop_box"))
            seed, r = _landmark_centre(lms), _iris_radius_in_crop(lms)
            dark = _dark_pupil_centre(crop, seed, r, eye)
            pupil = dark or seed

            h, w = crop.shape[:2]
            yy, xx = np.mgrid[:h, :w]
            variants = [("real", crop)]
            for k in range(VARIANTS_PER_CROP):
                img = crop.astype(np.float64)
                ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.8) * r
                cx, cy = pupil[0] + rad * np.cos(ang), pupil[1] + rad * np.sin(ang)
                sigma = rng.uniform(0.8, 4.5 if k < VARIANTS_PER_CROP - 1 else 9.0)
                spot = rng.uniform(150, 260) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
                if k in (2, 4):     # glare decoy away from the pupil
                    gx, gy = rng.uniform(0.1 * w, 0.9 * w), rng.uniform(0.1 * h, 0.9 * h)
                    spot += 220 * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * rng.uniform(1.5, 5) ** 2))
                variants.append((f"synthetic{k}", np.clip(np.rint(img + spot[..., None]), 0, 255).astype(np.uint8)))

            for tag, img in variants:
                name = f"crop{len(cases):04d}"
                rgba = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
                (out / f"{name}.rgba").write_bytes(rgba.tobytes())
                cases.append({
                    "name": name, "photo": Path(path).name, "eye": eye, "variant": tag,
                    "w": w, "h": h, "seed": list(map(float, seed)), "iris_r": float(r),
                    "server_pupil": list(map(float, dark)) if dark and tag == "real" else None,
                    "pupil_for_reflex": list(map(float, pupil)),
                    "server_reflex": server_reflex(img, r, pupil, eye),
                })
    (out / "cases.json").write_text(json.dumps(cases))
    print(f"{len(cases)} crops from {len(photos)} photos → {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/parity"))
