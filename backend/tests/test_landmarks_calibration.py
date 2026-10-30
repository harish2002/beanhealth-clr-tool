"""
BeanHealth CLR Tool — Landmark & Calibration Tests (Phase 3, milestone 5)
=========================================================================

  • Per-eye landmarks against synthetic eyes with known truth: sub-pixel
    reflex, limbus ellipse (lids, tilt, dark iris, foreshortening), pupil,
    fissure / scleral show / limbus coverage — and the cases that must fail
    closed instead of returning a wrong value.
  • Head pose: the decomposition and its sign conventions.
  • Whole-frame extraction on the repository's real photos.
  • Calibration: per-patient Hirschberg ratio and kappa recovered from
    simulated sessions; every fallback to the KB default is flagged; the
    rotation signs (eso +, right hyper +) through to a deviation.
  • Monocular calibration captures in frame QC; the limbus-size gate.
  • The /exam/calibrate and /exam/landmarks endpoints.

Synthetic eyes come from exam.sim.eye_image, sessions from exam.sim.frames —
test fixtures, not physiology.

Run:
    cd backend && pytest tests/test_landmarks_calibration.py -v
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exam.calibration import (  # noqa: E402
    calibrate,
    decentration_mm,
    eye_rotation_pd,
    robust_location,
)
from exam.capture import CALIBRATION_POSITIONS, CaptureSession, Ellipse, EyeObservation, build_plan  # noqa: E402
from exam.kb import load_kb  # noqa: E402
import exam.landmarks.eye as eye_module  # noqa: E402
from exam.landmarks.eye import (  # noqa: E402
    ellipse_points,
    find_pupil,
    fit_conic_ellipse,
    fit_limbus,
    lid_metrics,
    mulberry32,
    refine_reflex,
    robust_ellipse,
)
from exam.landmarks.extract import IRIS, extract_frame, face_landmarks  # noqa: E402
from exam.landmarks.head_pose import (  # noqa: E402
    CHIN,
    FOREHEAD,
    LEFT_EYE_RING,
    RIGHT_EYE_RING,
    head_pose,
    pose_matrix,
)
from exam.models import degrees_to_pd  # noqa: E402
from exam.qc import run_qc  # noqa: E402
from exam.sim import FrameNoise, SessionSpec, SimEyes, simulate_session  # noqa: E402
from exam.sim.eye_image import EyeScene, dark_iris, render_eye, true_covered_fraction  # noqa: E402

IMAGES = Path(__file__).parent / "test_images"
KB = load_kb()
DEFAULT_RATIO = degrees_to_pd(KB.scalar("hirschberg_ratio_deg_per_mm").value)


def rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        # Some photos carry a macOS privacy attribute (com.apple.macl) that blocks
        # other programs from opening them.
        pytest.skip(f"cannot read {path.name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def first_image(folder: str) -> Path:
    return sorted(p for p in (IMAGES / folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[0]


def guess(scene: EyeScene):
    """A face-mesh-quality starting point: a few px off, radius 5% small."""
    cx, cy = scene.limbus.centre_px
    return (cx + 3.0, cy - 2.0), 0.95 * scene.limbus.semi_major_px


# ─────────────────────────────────────────────────────────────
# Conic fit
# ─────────────────────────────────────────────────────────────

class TestConicFit:
    @pytest.mark.parametrize("a,b,angle", [(60, 60, 0), (60, 52, 30), (40, 35, 95), (25, 24.5, 170)])
    def test_exact_on_noiseless_points(self, a, b, angle):
        truth = Ellipse(centre_px=(123.4, 56.7), semi_major_px=a, semi_minor_px=b, angle_deg=angle)
        fit = fit_conic_ellipse(ellipse_points(truth, 40))
        assert fit.centre_px == pytest.approx(truth.centre_px, abs=1e-6)
        assert (fit.semi_major_px, fit.semi_minor_px) == pytest.approx((a, b), abs=1e-6)
        if a != b:
            assert min(abs(fit.angle_deg - angle), 180 - abs(fit.angle_deg - angle)) < 1e-4

    def test_not_an_ellipse(self):
        line = np.stack([np.arange(10.0), 2 * np.arange(10.0)], axis=1)
        assert fit_conic_ellipse(line) is None
        assert fit_conic_ellipse(np.zeros((4, 2))) is None


# ─────────────────────────────────────────────────────────────
# Limbus
# ─────────────────────────────────────────────────────────────

SCENES = {
    "default": EyeScene(),
    "head_tilt_30": EyeScene(roll_deg=30),
    "head_tilt_minus_45": EyeScene(roll_deg=-45),
    "low_upper_lid": EyeScene(upper_lid=35),
    "dark_iris": dark_iris(),
    "turned_eye": EyeScene(limbus=Ellipse(centre_px=(201.7, 149.2), semi_major_px=60,
                                          semi_minor_px=52, angle_deg=95)),
    "small_iris": EyeScene(size=(160, 120), limbus=Ellipse(centre_px=(80.4, 60.3), semi_major_px=20,
                                                           semi_minor_px=19.6, angle_deg=0),
                           pupil_radius=6, reflex=(84.2, 57.7), reflex_radius=1.4,
                           upper_lid=17, lower_lid=21, canthi_half_width=48),
}


class TestLimbus:
    @pytest.mark.parametrize("name", list(SCENES))
    def test_recovers_the_drawn_limbus(self, name):
        scene = SCENES[name]
        c, r = guess(scene)
        fit = fit_limbus(render_eye(scene), c, r, scene.aperture())
        assert fit is not None
        tol = 0.25 if scene.limbus.semi_major_px >= 40 else 0.35
        assert math.dist(fit.ellipse.centre_px, scene.limbus.centre_px) < tol
        assert fit.ellipse.semi_major_px == pytest.approx(scene.limbus.semi_major_px, abs=tol)
        assert fit.ellipse.semi_minor_px == pytest.approx(scene.limbus.semi_minor_px, abs=2 * tol)

    def test_seeded_and_repeatable(self):
        scene = SCENES["default"]
        img, (c, r) = render_eye(scene), guess(scene)
        assert fit_limbus(img, c, r, scene.aperture()) == fit_limbus(img, c, r, scene.aperture())

    def test_no_edge_no_fit(self):
        flat = np.full((300, 400), 120, np.uint8)
        assert fit_limbus(flat, (200, 150), 60) is None

    def test_one_side_hidden_no_fit(self):
        # The horizontal diameter is the scale: an iris whose right-hand side lies past the
        # canthus (outside the lid opening) gives no limbus.
        limbus = Ellipse(centre_px=(260.0, 150.6), semi_major_px=60, semi_minor_px=58.5, angle_deg=10)
        scene = EyeScene(limbus=limbus)
        c, r = guess(scene)
        aperture = EyeScene(limbus=Ellipse(centre_px=(200.0, 150.6), semi_major_px=60, semi_minor_px=58.5,
                                           angle_deg=10), canthi_half_width=70).aperture()
        assert fit_limbus(render_eye(scene), c, r, aperture) is None

    def test_flattened_outline_is_refused(self):
        # An eye facing the camera cannot look this flat (60 × 43 px); such an outline
        # means a lid or the pupil was caught, so no limbus is returned.
        scene = EyeScene(limbus=Ellipse(centre_px=(200.3, 150.6), semi_major_px=60, semi_minor_px=43,
                                        angle_deg=0), upper_lid=70, lower_lid=75)
        c, r = guess(scene)
        assert fit_limbus(render_eye(scene), c, r, scene.aperture()) is None

    def test_restarts_that_disagree_give_no_limbus(self, monkeypatch):
        # Two ellipses fit the arcs about equally well: whichever a restart lands on,
        # the reference is not trustworthy, so there is no limbus at all.
        small = Ellipse(centre_px=(100.0, 100.0), semi_major_px=30, semi_minor_px=29, angle_deg=0)
        large = Ellipse(centre_px=(100.0, 102.5), semi_major_px=32, semi_minor_px=31, angle_deg=0)
        pts = np.vstack([ellipse_points(small, 40), ellipse_points(large, 40)])
        starts = iter([small, large, small])
        monkeypatch.setattr(eye_module, "_msac_start", lambda *a: next(starts))
        assert robust_ellipse(pts, 31.0, mulberry32(0)) is None
        # The same points, every restart in one basin: a fit.
        starts = iter([small, small, small])
        assert robust_ellipse(pts, 31.0, mulberry32(0)) is not None

    def test_real_eye_does_not_depend_on_the_seed(self):
        image = rgb(first_image("success_reference"))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        points = face_landmarks(image)
        from exam.landmarks.extract import aperture_for
        iris = points[list(IRIS["R"]), :2]
        radius = float(np.mean(np.hypot(*(iris[1:] - iris[0]).T)))
        fits = [fit_limbus(gray, tuple(iris[0]), radius, aperture_for(points, "R"), seed=s) for s in range(5)]
        assert all(f is not None for f in fits)
        assert max(math.dist(f.ellipse.centre_px, fits[0].ellipse.centre_px) for f in fits) < 0.01

    def test_misplaced_lid_contour_never_flattens_the_fit(self):
        # Face-mesh lid contours drawn too high let lid-edge points in; the fit must
        # either ignore them or give up — never return a flattened limbus.
        scene = EyeScene(upper_lid=32)
        c, r = guess(scene)
        fit = fit_limbus(render_eye(scene), c, r, EyeScene(upper_lid=58).aperture())
        if fit is not None:
            assert math.dist(fit.ellipse.centre_px, scene.limbus.centre_px) < 0.5
            assert fit.ellipse.semi_major_px / fit.ellipse.semi_minor_px <= 1.2


# ─────────────────────────────────────────────────────────────
# Reflex, pupil, lids
# ─────────────────────────────────────────────────────────────

class TestReflex:
    @pytest.mark.parametrize("fx,fy", [(0.0, 0.0), (0.25, 0.5), (0.5, 0.75), (0.83, 0.17), (0.4, 0.95)])
    def test_sub_pixel(self, fx, fy):
        true = (211.0 + fx, 144.0 + fy)
        img = render_eye(EyeScene(reflex=true, seed=int(10 * fx + 100 * fy)))
        found = refine_reflex(img, (round(true[0]), round(true[1])), 60.0)
        assert math.dist(found, true) < 0.08

    def test_coarse_start_a_pixel_off(self):
        true = (212.37, 143.81)
        found = refine_reflex(render_eye(EyeScene(reflex=true)), (214, 142), 60.0)
        assert math.dist(found, true) < 0.08

    def test_large_saturated_reflex_widens_the_window(self):
        true = (205.6, 151.3)
        found = refine_reflex(render_eye(EyeScene(reflex=true, reflex_radius=14)), (206, 151), 60.0)
        assert math.dist(found, true) < 0.1

    def test_no_reflex(self):
        flat = np.full((100, 100), 90, np.uint8)
        assert refine_reflex(flat, (50, 50), 30.0) is None

    def test_at_the_image_edge(self):
        assert refine_reflex(render_eye(EyeScene()), (3, 3), 60.0) is None


class TestPupil:
    @pytest.mark.parametrize("offset,radius,reflex", [
        ((0.0, 0.0), 18.0, (212.37, 143.81)),     # reflex outside the pupil
        ((-2.3, 1.1), 12.0, (203.0, 146.0)),      # reflex on the pupil edge
        ((0.0, 0.0), 18.0, (201.0, 151.0)),       # reflex inside the pupil
        ((1.5, -0.7), 30.0, (212.37, 143.81)),    # dilated
    ])
    def test_centre_and_diameter(self, offset, radius, reflex):
        scene = EyeScene(pupil_offset=offset, pupil_radius=radius, reflex=reflex)
        img = render_eye(scene)
        c, r = guess(scene)
        limbus = fit_limbus(img, c, r, scene.aperture()).ellipse
        centre, diameter = find_pupil(img, limbus, reflex)
        truth = (scene.limbus.centre_px[0] + offset[0], scene.limbus.centre_px[1] + offset[1])
        assert math.dist(centre, truth) < 0.4
        # The outline runs through boundary pixel centres: about half a pixel small.
        assert diameter == pytest.approx(2 * radius, abs=1.0)

    def test_dark_iris_gives_none(self):
        scene = dark_iris()
        assert find_pupil(render_eye(scene), scene.limbus) is None

    def test_no_pupil_gives_none(self):
        scene = EyeScene(pupil_radius=0)
        assert find_pupil(render_eye(scene), scene.limbus) is None


class TestLids:
    @pytest.mark.parametrize("roll", [0, 30, -45])
    def test_measures_hold_on_a_tilted_head(self, roll):
        scene = EyeScene(roll_deg=roll)
        m = lid_metrics(scene.aperture(), scene.limbus)
        assert m.fissure_height_px == pytest.approx(scene.upper_lid + scene.lower_lid, abs=0.5)
        # Limbus top along the eye's own vertical, minus the lid height there.
        lim = scene.limbus
        phi = math.radians(lim.angle_deg - roll)
        extent = math.hypot(lim.semi_major_px * math.sin(phi), lim.semi_minor_px * math.cos(phi))
        assert m.scleral_show_px == pytest.approx(scene.upper_lid - extent, abs=0.5)
        assert m.limbus_covered_fraction == pytest.approx(true_covered_fraction(scene), abs=0.02)

    def test_scleral_show_positive_when_the_lid_clears_the_limbus(self):
        scene = EyeScene(upper_lid=70, lower_lid=75)
        m = lid_metrics(scene.aperture(), scene.limbus)
        assert m.scleral_show_px > 10
        assert m.limbus_covered_fraction == 0.0

    def test_outside_the_canthi_is_outside_the_aperture(self):
        ap = EyeScene().aperture()
        assert not ap.contains(np.array([[200.3 - 150, 150.6]])).any()
        assert ap.contains(np.array([[200.3, 150.6]])).all()


# ─────────────────────────────────────────────────────────────
# Head pose
# ─────────────────────────────────────────────────────────────

def synthetic_face(yaw=0.0, pitch=0.0, roll=0.0, seed=0) -> np.ndarray:
    """478 face-mesh points of a schematic face, rotated by the protocol's pose convention."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 20, (478, 3))
    for ring, cx in ((RIGHT_EYE_RING, -32.0), (LEFT_EYE_RING, 32.0)):
        t = np.linspace(0, 2 * math.pi, len(ring), endpoint=False)
        pts[list(ring)] = np.stack([cx + 14 * np.cos(t), 5 * np.sin(t), np.zeros_like(t)], axis=1)
    pts[list(FOREHEAD)] = [[0, -75, 0], [0, -60, 0], [0, -45, 0]]
    pts[list(CHIN)] = [[0, 95, 0], [0, 100, 0], [0, 88, 0]]
    rotated = pts @ pose_matrix(yaw, pitch, roll).T
    return rotated + [960.0, 540.0, 0.0]


class TestHeadPose:
    @pytest.mark.parametrize("yaw,pitch,roll", [
        (0, 0, 0), (25, 0, 0), (-25, 0, 0), (0, 25, 0), (0, -25, 0),
        (25, 25, 0), (-25, -25, 0), (0, 0, 40), (0, 0, -40), (10, -5, 35)])
    def test_recovers_the_pose(self, yaw, pitch, roll):
        pose = head_pose(synthetic_face(yaw, pitch, roll))
        assert (pose.yaw_deg, pose.pitch_deg, pose.roll_deg) == pytest.approx((yaw, pitch, roll), abs=1e-6)

    def test_signs_follow_the_protocol(self):
        def toward_camera(p):           # the face's forward direction in camera coordinates
            return -(pose_matrix(*p) @ [0, 0, 1])
        # Face turned to the patient's right = toward image left.
        assert toward_camera((20, 0, 0))[0] < 0 and head_pose(synthetic_face(yaw=20)).yaw_deg > 0
        # Chin up = the face looks up the image.
        assert toward_camera((0, 20, 0))[1] < 0 and head_pose(synthetic_face(pitch=20)).pitch_deg > 0
        # Tilted to the right shoulder = the right eye (image left) drops.
        face = synthetic_face(roll=20)
        assert face[list(RIGHT_EYE_RING), 1].mean() > face[list(LEFT_EYE_RING), 1].mean()
        assert head_pose(face).roll_deg > 0

    def test_needs_the_full_mesh(self):
        with pytest.raises(ValueError):
            head_pose(np.zeros((100, 3)))


# ─────────────────────────────────────────────────────────────
# Whole frames (real photos)
# ─────────────────────────────────────────────────────────────

class TestExtract:
    @pytest.fixture(scope="class")
    def reference(self):
        image = rgb(first_image("success_reference"))
        return image, extract_frame(image, index=4, t_ms=133.3), face_landmarks(image)

    def test_reference_photo(self, reference):
        _, frame, points = reference
        assert (frame.index, frame.t_ms) == (4, 133.3)
        assert frame.head_pose is not None and frame.sharpness > 0
        for eye, obs in frame.eyes.items():
            mesh_radius = float(np.mean(np.hypot(*(points[list(IRIS[eye][1:]), :2]
                                                    - points[IRIS[eye][0], :2]).T)))
            assert obs.limbus.semi_major_px == pytest.approx(mesh_radius, rel=0.1)
            assert obs.iris_radius_px == obs.limbus.semi_major_px
            # The torch reflex sits inside the iris, near its centre.
            assert math.dist(obs.reflex_px, obs.limbus.centre_px) < 0.5 * obs.limbus.semi_major_px
            assert obs.fissure_height_px > obs.limbus.semi_major_px
            assert 0 <= obs.limbus_covered_fraction < 0.4

    def test_reflex_is_refined_not_moved(self, reference):
        image, frame, _ = reference
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        for obs in frame.eyes.values():
            x, y = obs.reflex_px
            assert gray[int(round(y)), int(round(x))] >= 200

    def test_no_torch_no_reflex(self):
        frame = extract_frame(rgb(first_image("flash_off")))
        assert frame.head_pose is not None
        assert all(obs.reflex_px is None for obs in frame.eyes.values())

    def test_no_face(self):
        frame = extract_frame(rgb(first_image("no_face")), index=2, t_ms=66.7)
        assert frame.head_pose is None and frame.eyes == {} and frame.sharpness is None

    def test_needs_rgb(self):
        with pytest.raises(ValueError):
            extract_frame(np.zeros((100, 100), np.uint8))


# ─────────────────────────────────────────────────────────────
# Calibration geometry
# ─────────────────────────────────────────────────────────────

def obs_at(dx, dy, radius=86.0):
    """An eye whose reflex sits (dx, dy) px from the limbus centre."""
    return EyeObservation(reflex_px=(1000.0 + dx, 500.0 + dy),
                          limbus=Ellipse(centre_px=(1000.0, 500.0), semi_major_px=radius,
                                         semi_minor_px=radius, angle_deg=0.0))


class TestDecentration:
    MM = 11.5 / (2 * 86.0)    # mm per px with the KB iris diameter

    def test_temporal_is_toward_each_eye_s_own_side(self):
        # The right eye's temporal side is the image left; the left eye's, the image right.
        assert decentration_mm(obs_at(-10, 0), "R", 0, 11.5) == pytest.approx((10 * self.MM, 0))
        assert decentration_mm(obs_at(10, 0), "L", 0, 11.5) == pytest.approx((10 * self.MM, 0))
        assert decentration_mm(obs_at(0, 7), "R", 0, 11.5) == pytest.approx((0, 7 * self.MM))

    def test_scale_uses_the_major_axis(self):
        o = EyeObservation(reflex_px=(1010.0, 500.0), limbus=Ellipse(
            centre_px=(1000.0, 500.0), semi_major_px=86.0, semi_minor_px=75.0, angle_deg=90.0))
        assert decentration_mm(o, "L", 0, 11.5)[0] == pytest.approx(10 * self.MM)

    def test_head_roll_is_undone(self):
        # Head tilted 90° to the right shoulder: the head's horizontal (right eye → left eye)
        # points up the image, so a reflex above the centre is on the patient's left.
        t, i = decentration_mm(obs_at(0, -10), "L", 90, 11.5)
        assert (t, i) == pytest.approx((10 * self.MM, 0), abs=1e-12)
        t, i = decentration_mm(obs_at(10 * math.cos(math.radians(30)), -10 * math.sin(math.radians(30))),
                               "L", 30, 11.5)
        assert (t, i) == pytest.approx((10 * self.MM, 0), abs=1e-12)

    def test_unmeasured(self):
        assert decentration_mm(EyeObservation(limbus=obs_at(0, 0).limbus), "R", 0, 11.5) is None
        assert decentration_mm(EyeObservation(reflex_px=(1.0, 1.0)), "R", 0, 11.5) is None

    def test_robust_location(self):
        m = robust_location([1.0, 1.1, 0.9, 1.0, 50.0])
        assert m.value == 1.0 and m.error < 0.2
        assert robust_location([2.0, 2.0, 2.0]).error > 0


def session(**kw) -> CaptureSession:
    return simulate_session(SessionSpec(**kw))


PATIENT = SimEyes(hirschberg_pd_per_mm=16.0, kappa_mm={"R": (0.30, -0.10), "L": (0.45, 0.05)},
                  deviation_pd=(20.0, -6.0))


class TestCalibration:
    def test_recovers_ratio_and_kappa(self):
        cal = calibrate(session(sim=PATIENT, noise=FrameNoise(landmark_px=0.5)))
        assert cal.flags == []
        h = cal.hirschberg
        assert h.source == "fitted" and h.points == 6
        assert abs(h.pd_per_mm.value - 16.0) < 3 * h.pd_per_mm.error
        assert h.pd_per_mm.value == pytest.approx(16.0, rel=0.03)
        for eye, (t, i) in PATIENT.kappa_mm.items():
            k = cal.kappa[eye]
            assert k.temporal_mm.value == pytest.approx(t, abs=0.02)
            assert k.inferior_mm.value == pytest.approx(i, abs=0.02)
            assert k.frames >= 10
        assert (cal.kb_version, cal.calibration_version) == (KB.version, "1.0.0")

    def test_default_patient_fits_the_kb_ratio(self):
        cal = calibrate(session())
        assert cal.hirschberg.pd_per_mm.value == pytest.approx(DEFAULT_RATIO, rel=0.02)

    def test_distance_error_widens_the_ratio(self):
        near = calibrate(session(sim=PATIENT, distance_error_cm=0.5)).hirschberg.pd_per_mm
        far = calibrate(session(sim=PATIENT, distance_error_cm=2.0)).hirschberg.pd_per_mm
        assert far.error > 2 * near.error
        assert far.error >= 16.0 * 2.0 / 36.0 * 0.99

    def test_wrong_distance_scales_the_ratio(self):
        # Target angles come from the header distance: a patient at 33 cm reported
        # at 40 cm shrinks every angle asked, and the ratio with it.
        cal = calibrate(session(sim=PATIENT, distance_cm=40.0, true_distance_cm=33.0))
        assert cal.hirschberg.fitted.value == pytest.approx(16.0 * 33 / 40, rel=0.03)

    def test_rotations_give_the_deviation(self):
        s = session(sim=PATIENT)
        cal = calibrate(s)
        for position in ("primary", "tilt_right", "tilt_left"):
            dev_h, dev_v = [], []
            for f in s.capture(position).frames:
                r = eye_rotation_pd(f.eyes["R"], "R", f.head_pose.roll_deg, cal)
                l_ = eye_rotation_pd(f.eyes["L"], "L", f.head_pose.roll_deg, cal)
                dev_h.append(r[0] + l_[0])          # both eyes turned in = eso +
                dev_v.append(r[1] - l_[1])          # right eye higher = right hyper +
            assert np.median(dev_h) == pytest.approx(20.0, abs=0.6)
            assert np.median(dev_v) == pytest.approx(-6.0, abs=0.6)


class TestCalibrationFallbacks:
    def _codes(self, cal):
        return {(f.code, f.eye) for f in cal.flags}

    def test_no_distance(self):
        cal = calibrate(session(sim=PATIENT, distance_cm=None))
        assert ("distance_not_estimated", None) in self._codes(cal)
        assert cal.hirschberg.source == "kb_default" and cal.hirschberg.fitted is None
        assert cal.kappa["R"] is not None          # kappa needs no distance

    def test_kb_default_is_marked(self):
        cal = calibrate(session(sim=PATIENT, distance_cm=None))
        d = cal.hirschberg.pd_per_mm
        assert d.value == pytest.approx(DEFAULT_RATIO)
        assert d.error == KB.scalar("hirschberg_ratio_default_sd_pd_per_mm").value
        assert d.confidence == "MODERATE"

    def test_missing_kappa_capture(self):
        s = session(sim=PATIENT, skip=("calib_L_lens",))
        cal = calibrate(s)
        assert ("kappa_not_measured", "L") in self._codes(cal)
        assert cal.kappa["L"] is None
        # Without its kappa the eye's rotation is not measured — never measured from zero.
        f = s.capture("primary").frames[0]
        assert eye_rotation_pd(f.eyes["L"], "L", f.head_pose.roll_deg, cal) is None
        assert eye_rotation_pd(f.eyes["R"], "R", f.head_pose.roll_deg, cal) is not None
        # The other eye's targets still fit the ratio.
        assert cal.hirschberg.source == "fitted"

    def test_no_targets(self):
        cal = calibrate(session(sim=PATIENT, skip=tuple(p for p in CALIBRATION_POSITIONS if not p.endswith("lens"))))
        assert ("hirschberg_not_fitted", None) in self._codes(cal)
        assert cal.hirschberg.source == "kb_default"

    def test_implausible_fit_falls_back(self):
        cal = calibrate(session(sim=SimEyes(hirschberg_pd_per_mm=30.0)))
        assert ("hirschberg_implausible", None) in self._codes(cal)
        assert cal.hirschberg.source == "kb_default"
        assert cal.hirschberg.fitted.value == pytest.approx(30.0, rel=0.05)

    def test_imprecise_fit_falls_back(self):
        cal = calibrate(session(sim=PATIENT, distance_error_cm=8.0))
        assert ("hirschberg_imprecise", None) in self._codes(cal)
        assert cal.hirschberg.source == "kb_default"

    def test_bad_frames_do_not_enter_the_fit(self):
        from exam.sim import events
        ev = {"calib_R_right": events(glance=range(20, 25), glance_deg=10)}
        clean = calibrate(session(sim=PATIENT)).hirschberg.pd_per_mm.value
        with_glance = calibrate(session(sim=PATIENT, events=ev)).hirschberg.pd_per_mm.value
        assert with_glance == pytest.approx(clean, rel=0.01)


# ─────────────────────────────────────────────────────────────
# Frame QC on calibration captures; limbus-size gate
# ─────────────────────────────────────────────────────────────

class TestCalibrationQC:
    def test_plan(self):
        plan = build_plan("five_gaze", include_tilt=False)
        calib = [p for p in plan.positions if p.kind == "calibration"]
        assert [p.position for p in calib] == list(CALIBRATION_POSITIONS)
        offset = KB.scalar("calibration_target_offset_cm").value
        assert [(p.fixing_eye, p.target_offset_cm) for p in calib] == [
            ("R", 0), ("R", offset), ("R", -offset), ("L", 0), ("L", offset), ("L", -offset)]
        assert all(p.observed_eyes == (p.fixing_eye,) for p in calib)
        assert "Cover the patient's left eye" in calib[0].instruction

    def test_the_covered_eye_is_not_required(self):
        s = session()
        assert set(s.capture("calib_R_lens").frames[0].eyes) == {"R"}
        q = run_qc(s)
        assert all(p.status == "OK" for p in q.positions if p.kind == "calibration")

    def test_the_fixing_eye_is_required(self):
        raw = session(n_frames=12).model_dump()
        cap = next(c for c in raw["captures"] if c["position"] == "calib_L_right")
        cap["frames"][4]["eyes"] = {"R": cap["frames"][4]["eyes"]["L"]}
        pqc = next(p for p in run_qc(CaptureSession.model_validate(raw)).positions
                   if p.position == "calib_L_right")
        bad = {v.index: {r.code for r in v.reasons} for v in pqc.frames if not v.valid}
        assert "eye_not_found" in bad[4]

    def test_limbus_size_mismatch(self):
        raw = session(n_frames=12).model_dump()
        prim = next(c for c in raw["captures"] if c["position"] == "primary")
        lim = prim["frames"][5]["eyes"]["L"]["limbus"]
        lim["semi_major_px"] = lim["semi_minor_px"] = 0.7 * lim["semi_major_px"]
        pqc = next(p for p in run_qc(CaptureSession.model_validate(raw)).positions if p.position == "primary")
        bad = {v.index: {r.code for r in v.reasons} for v in pqc.frames if not v.valid}
        assert bad == {5: {"limbus_size_mismatch"}}


# ─────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)

    def test_protocol_starts_with_calibration(self, client):
        positions = [p["position"] for p in client.get("/exam/protocol/nine_gaze").json()["positions"]]
        assert positions[:6] == list(CALIBRATION_POSITIONS) and positions[6] == "primary"

    def test_calibrate(self, client):
        s = session(sim=PATIENT, include_tilt=False, n_frames=20)
        r = client.post("/exam/calibrate", content=s.model_dump_json(),
                        headers={"content-type": "application/json"})
        assert r.status_code == 200
        body = r.json()
        assert body["hirschberg"]["source"] == "fitted"
        assert body["kb_version"] == KB.version and body["qc_version"] == "1.1.0"

    def test_landmarks(self, client):
        path = first_image("success_reference")
        with open(path, "rb") as f:
            r = client.post("/exam/landmarks", files={"image": ("frame.jpg", f, "image/jpeg")},
                            params={"index": 3, "t_ms": 100})
        assert r.status_code == 200
        body = r.json()
        assert body["frame"]["index"] == 3 and set(body["frame"]["eyes"]) == {"R", "L"}
        assert body["landmarks_version"] == "1.0.0" and body["width_px"] > 0

    def test_landmarks_rejects_non_images(self, client):
        r = client.post("/exam/landmarks", files={"image": ("x.jpg", b"not an image", "image/jpeg")})
        assert r.status_code == 422
