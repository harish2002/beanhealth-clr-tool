"""
Write frontend/tests/fixtures/examLandmarks.json: synthetic eyes and what the
Python reference (backend/exam/landmarks) finds in them, for the browser
port's parity tests (frontend/tests/examLandmarks.test.ts).

    cd backend && venv/bin/python ../tools/parity/dump_exam_fixtures.py
"""

import base64
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from exam.capture.session import Ellipse  # noqa: E402
from exam.landmarks.eye import (  # noqa: E402
    find_pupil, fit_limbus, lid_metrics, mulberry32, refine_reflex,
)
from exam.sim.eye_image import EyeScene, render_eye  # noqa: E402

SMALL = dict(size=(200, 150), pupil_radius=9.0, reflex=(106.2, 71.9), reflex_radius=1.6,
             upper_lid=25.0, lower_lid=31.0, canthi_half_width=70.0)
LIMBUS = Ellipse(centre_px=(100.3, 75.6), semi_major_px=30.0, semi_minor_px=29.25, angle_deg=10.0)

SCENES = {
    "clean": EyeScene(limbus=LIMBUS, noise_sd=0, **SMALL),
    "head_tilt_30": EyeScene(limbus=LIMBUS, noise_sd=0, roll_deg=30, **SMALL),
    "low_lid": EyeScene(limbus=LIMBUS, noise_sd=0, **{**SMALL, "upper_lid": 18.0}),
    "dark_iris": EyeScene(limbus=LIMBUS, noise_sd=0, iris=40.0, pupil=32.0, **SMALL),
    "noisy": EyeScene(limbus=LIMBUS, noise_sd=2.0, seed=7, **SMALL),
    "turned": EyeScene(limbus=Ellipse(centre_px=(99.6, 76.1), semi_major_px=30.0, semi_minor_px=26.5,
                                      angle_deg=93.0), noise_sd=0, **SMALL),
}


def ell(e):
    return None if e is None else {"centre_px": list(e.centre_px), "semi_major_px": e.semi_major_px,
                                   "semi_minor_px": e.semi_minor_px, "angle_deg": e.angle_deg}


def main() -> None:
    out = {"generator": "tools/parity/dump_exam_fixtures.py", "scenes": [],
           "mulberry32": {str(seed): [r() for r in [mulberry32(seed)] for _ in range(5)]
                          for seed in (0, 12345)}}
    for name, sc in SCENES.items():
        img = render_eye(sc)
        ap = sc.aperture()
        guess = (sc.limbus.centre_px[0] + 1.5, sc.limbus.centre_px[1] - 1.0)
        radius = 0.95 * sc.limbus.semi_major_px
        fit = fit_limbus(img, guess, radius, ap)
        approx = (round(sc.reflex[0]), round(sc.reflex[1]))
        expect = {"limbus": ell(fit.ellipse) if fit else None,
                  "reflex": refine_reflex(img, approx, sc.limbus.semi_major_px)}
        if fit:
            pupil = find_pupil(img, fit.ellipse)
            expect["pupil"] = None if pupil is None else {"centre": list(pupil[0]), "diameter": pupil[1]}
            expect["lids"] = lid_metrics(ap, fit.ellipse).__dict__
        out["scenes"].append({
            "name": name, "w": sc.size[0], "h": sc.size[1],
            "image_zlib_b64": base64.b64encode(zlib.compress(img.tobytes(), 9)).decode(),
            "aperture": {"upper": [list(p) for p in ap.upper], "lower": [list(p) for p in ap.lower],
                         "canthi": [list(p) for p in ap.canthi]},
            "guess": {"centre": list(guess), "radius": radius},
            "reflex_approx": list(approx), "iris_radius": sc.limbus.semi_major_px,
            "truth": {"limbus": ell(sc.limbus), "reflex": list(sc.reflex)},
            "expect": expect,
        })
    path = ROOT / "frontend" / "tests" / "fixtures" / "examLandmarks.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
