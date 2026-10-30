"""
BeanHealth CLR — Per-eye landmarks (brief §4.3)
================================================

Reference implementation of the per-eye values a capture session carries.
The phone port (frontend/lib/examLandmarks.ts) follows the same steps.

  refine_reflex   corneal light reflex centroid, sub-pixel, intensity-weighted
  LidAperture     the lid opening from the face-mesh lid contours, in the eye's
                  own frame (canthus to canthus = horizontal), so it holds on
                  a tilted head
  fit_limbus      limbus ellipse: radial edge search inside the lid opening,
                  RANSAC over a least-squares conic fit
  find_pupil      dark pupil inside the limbus (None when the iris is too dark
                  to separate it, as is common under visible light)
  lid_metrics     fissure height, scleral show and the share of the limbus the
                  lids hide

Every coordinate is (x, y) in full-frame pixels, unmirrored. The constants
below are image-processing parameters, not clinical values; clinical numbers
live in the KB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from exam.capture.session import Ellipse

Point = Tuple[float, float]

LANDMARKS_VERSION = "1.0.0"

# Sub-pixel reflex: window half-size (× iris radius) and the share of the
# peak above background a pixel needs to count as part of the reflex.
REFLEX_WINDOW_IRIS_RADII = 0.3
REFLEX_LEVEL = 0.5

# Limbus search.
LIMBUS_RAYS = 90
LIMBUS_SEARCH_BAND = (0.6, 1.5)           # × the radius guess
LIMBUS_SAMPLE_STEP_PX = 0.5
LIMBUS_MIN_EDGE_SHARE = 0.25              # edge weaker than this × the median edge is ignored
LIMBUS_LID_MARGIN_IRIS_RADII = 0.04       # edge points this close to a lid are dropped
LIMBUS_INLIER_IRIS_RADII = 0.02           # RANSAC inlier distance (min 0.75 px)
LIMBUS_RANSAC_ITERATIONS = 300
LIMBUS_MIN_INLIERS = 16
# The eye faces the camera (it fixes the light, or deviates from it by at most
# a few tens of degrees: cos 25° ≈ 0.9), so a flatter fit has caught a lid.
LIMBUS_MAX_AXIS_RATIO = 1.2
# The horizontal diameter is the scale ruler: both sides must carry edge points.
LIMBUS_SIDE_WINDOW_DEG = 40.0

# Pupil.
PUPIL_REGION_IRIS_RADII = 0.8
PUPIL_MAX_OFFSET_IRIS_RADII = 0.6
PUPIL_MIN_CONTRAST = 12.0                 # grey levels between iris and pupil medians
PUPIL_MIN_RELATIVE_CONTRAST = 0.2         # … and as a share of the iris median
PUPIL_DIAMETER_RANGE_IRIS_RADII = (0.15, 1.3)


# ─────────────────────────────────────────────────────────────
# Conic fitting (shared by limbus and pupil)
# ─────────────────────────────────────────────────────────────

def fit_conic_ellipse(points: np.ndarray) -> Optional[Ellipse]:
    """
    Least-squares ellipse through ≥ 5 points: A x² + B xy + C y² + D x + E y = 1
    in coordinates centred on the points' mean (so the conic cannot pass
    through the origin). None when the conic is not a real ellipse.
    """
    if len(points) < 5:
        return None
    origin = points.mean(axis=0)
    x, y = (points - origin).T
    design = np.stack([x * x, x * y, y * y, x, y], axis=1)
    try:
        coef, _, rank, _ = np.linalg.lstsq(design, np.ones(len(points)), rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 5:                              # collinear or repeated points: no unique conic
        return None
    a, b, c, d, e = coef
    det = 4 * a * c - b * b
    if det <= 0:
        return None
    x0 = (b * e - 2 * c * d) / det
    y0 = (b * d - 2 * a * e) / det
    f0 = a * x0 * x0 + b * x0 * y0 + c * y0 * y0 + d * x0 + e * y0 - 1.0
    lam = np.linalg.eigvalsh(np.array([[a, b / 2], [b / 2, c]]))
    if f0 >= 0 or np.any(lam <= 0):
        return None
    semi = np.sqrt(-f0 / lam)                 # lam ascending → semi descending
    # The smaller coefficient lies along the major axis.
    theta = 0.5 * math.atan2(b, a - c)        # direction of the larger coefficient
    major_angle = math.degrees(theta) + 90.0
    return Ellipse(centre_px=(float(x0 + origin[0]), float(y0 + origin[1])),
                   semi_major_px=float(semi[0]), semi_minor_px=float(min(semi[1], semi[0])),
                   angle_deg=float(major_angle % 180.0))


def ellipse_distance(ell: Ellipse, points: np.ndarray) -> np.ndarray:
    """Approximate signed distance of points from the ellipse (+ outside), px."""
    phi = math.radians(ell.angle_deg)
    rel = points - np.asarray(ell.centre_px)
    u = rel[:, 0] * math.cos(phi) + rel[:, 1] * math.sin(phi)
    v = -rel[:, 0] * math.sin(phi) + rel[:, 1] * math.cos(phi)
    rho = np.sqrt((u / ell.semi_major_px) ** 2 + (v / ell.semi_minor_px) ** 2)
    r = np.hypot(u, v)
    return r * (1.0 - 1.0 / np.maximum(rho, 1e-9))


def ellipse_points(ell: Ellipse, n: int = 360) -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    phi = math.radians(ell.angle_deg)
    u, v = ell.semi_major_px * np.cos(t), ell.semi_minor_px * np.sin(t)
    return np.stack([ell.centre_px[0] + u * math.cos(phi) - v * math.sin(phi),
                     ell.centre_px[1] + u * math.sin(phi) + v * math.cos(phi)], axis=1)


def ellipse_extent(ell: Ellipse, direction: Point) -> float:
    """Half-width of the ellipse along a unit direction (centre to its supporting line)."""
    phi = math.radians(ell.angle_deg)
    e1 = direction[0] * math.cos(phi) + direction[1] * math.sin(phi)
    e2 = -direction[0] * math.sin(phi) + direction[1] * math.cos(phi)
    return math.sqrt((ell.semi_major_px * e1) ** 2 + (ell.semi_minor_px * e2) ** 2)


# ─────────────────────────────────────────────────────────────
# Corneal light reflex
# ─────────────────────────────────────────────────────────────

def refine_reflex(gray: np.ndarray, approx: Point, iris_radius_px: float) -> Optional[Point]:
    """
    Intensity-weighted centroid of the reflex around a coarse position
    (module 3's blob centroid). Pixels above background + REFLEX_LEVEL ×
    (peak − background) that connect to the peak are weighted by their height
    above that level. None when the reflex fills the whole window (not a
    small spot) or rises nowhere above the background.
    """
    img = gray.astype(np.float64)
    h, w = img.shape
    for half in (REFLEX_WINDOW_IRIS_RADII * iris_radius_px, 2 * REFLEX_WINDOW_IRIS_RADII * iris_radius_px):
        r = max(3, int(math.ceil(half)))
        x0, y0 = int(round(approx[0])) - r, int(round(approx[1])) - r
        x1, y1 = x0 + 2 * r + 1, y0 + 2 * r + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        win = img[y0:y1, x0:x1]
        background = float(np.median(win))
        # Peak: the brightest pixel near the coarse position.
        core = win[r - r // 2:r + r // 2 + 1, r - r // 2:r + r // 2 + 1]
        py, px = np.unravel_index(int(np.argmax(core)), core.shape)
        py, px = py + r - r // 2, px + r - r // 2
        peak = float(win[py, px])
        if peak <= background:
            return None
        level = background + REFLEX_LEVEL * (peak - background)
        mask = (win > level).astype(np.uint8)
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        spot = labels == labels[py, px]
        if spot[0, :].any() or spot[-1, :].any() or spot[:, 0].any() or spot[:, -1].any():
            continue                                  # spilled over the window: try a wider one
        weight = np.where(spot, win - level, 0.0)
        total = float(weight.sum())
        if total <= 0:
            return None
        ys, xs = np.mgrid[0:win.shape[0], 0:win.shape[1]]
        return (x0 + float((weight * xs).sum()) / total, y0 + float((weight * ys).sum()) / total)
    return None


# ─────────────────────────────────────────────────────────────
# Lids
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LidAperture:
    """
    The opening between the lids, from the face-mesh lid contours. Both
    contours run between the two canthi; the frame they are compared in has
    the canthal line as its x axis (pointing image-right on an upright head)
    and y pointing down, so a tilted head does not distort the measures.
    """
    upper: Tuple[Point, ...]
    lower: Tuple[Point, ...]
    canthi: Tuple[Point, Point]                  # (image-left, image-right) on an upright head

    @property
    def _frame(self) -> Tuple[np.ndarray, float, float]:
        (ax, ay), (bx, by) = self.canthi
        phi = math.atan2(by - ay, bx - ax)
        return np.array([(ax + bx) / 2, (ay + by) / 2]), math.cos(phi), math.sin(phi)

    def to_local(self, pts: np.ndarray) -> np.ndarray:
        origin, c, s = self._frame
        rel = np.atleast_2d(pts) - origin
        return np.stack([rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c], axis=1)

    def vertical(self) -> Point:
        """Unit vector of the eye's own 'down' in image coordinates."""
        _, c, s = self._frame
        return (-s, c)

    def _curve(self, contour: Sequence[Point]) -> Tuple[np.ndarray, np.ndarray]:
        local = self.to_local(np.asarray(contour, dtype=float))
        order = np.argsort(local[:, 0])
        return local[order, 0], local[order, 1]

    def lid_y(self, x_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Upper and lower lid heights (local y, down +) at local x; NaN outside the canthi."""
        ux, uy = self._curve(self.upper)
        lx, ly = self._curve(self.lower)
        lo, hi = max(ux[0], lx[0]), min(ux[-1], lx[-1])
        x = np.asarray(x_local, dtype=float)
        inside = (x >= lo) & (x <= hi)
        up = np.where(inside, np.interp(x, ux, uy), np.nan)
        dn = np.where(inside, np.interp(x, lx, ly), np.nan)
        return up, dn

    def contains(self, pts: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
        local = self.to_local(np.asarray(pts, dtype=float))
        up, dn = self.lid_y(local[:, 0])
        with np.errstate(invalid="ignore"):
            return (local[:, 1] >= up + margin_px) & (local[:, 1] <= dn - margin_px)


# ─────────────────────────────────────────────────────────────
# Limbus
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LimbusFit:
    ellipse: Ellipse
    rms_px: float
    inliers: int
    rays: int


def _edge_points(img: np.ndarray, centre: Point, radius: float, aperture: Optional[LidAperture]) -> np.ndarray:
    """Strongest dark-to-bright step along each ray from the centre (sub-pixel), inside the lids."""
    theta = np.linspace(0, 2 * math.pi, LIMBUS_RAYS, endpoint=False)
    radii = np.arange(LIMBUS_SEARCH_BAND[0] * radius, LIMBUS_SEARCH_BAND[1] * radius, LIMBUS_SAMPLE_STEP_PX)
    map_x = (centre[0] + np.outer(np.cos(theta), radii)).astype(np.float32)
    map_y = (centre[1] + np.outer(np.sin(theta), radii)).astype(np.float32)
    prof = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    grad = np.gradient(prof.astype(np.float64), LIMBUS_SAMPLE_STEP_PX, axis=1)
    k = np.argmax(grad, axis=1)
    strength = grad[np.arange(len(theta)), k]
    # Parabolic sub-sample peak.
    km, kp = np.clip(k - 1, 0, len(radii) - 1), np.clip(k + 1, 0, len(radii) - 1)
    gm, g0, gp = grad[np.arange(len(theta)), km], strength, grad[np.arange(len(theta)), kp]
    denom = gm - 2 * g0 + gp
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where((k > 0) & (k < len(radii) - 1) & (denom < 0), 0.5 * (gm - gp) / denom, 0.0)
    r = radii[k] + offset * LIMBUS_SAMPLE_STEP_PX
    pts = np.stack([centre[0] + r * np.cos(theta), centre[1] + r * np.sin(theta)], axis=1)
    positive = strength > 0
    if not positive.any():
        return np.empty((0, 2))
    keep = positive & (strength >= LIMBUS_MIN_EDGE_SHARE * np.median(strength[positive]))
    if aperture is not None:
        keep &= aperture.contains(pts, LIMBUS_LID_MARGIN_IRIS_RADII * radius)
    return pts[keep]


def _ransac_ellipse(pts: np.ndarray, radius: float, rng: np.random.Generator
                    ) -> Optional[Tuple[Ellipse, np.ndarray]]:
    tol = max(0.75, LIMBUS_INLIER_IRIS_RADII * radius)
    best: Optional[np.ndarray] = None
    best_rms = math.inf
    for _ in range(LIMBUS_RANSAC_ITERATIONS):
        sample = pts[rng.choice(len(pts), 5, replace=False)]
        ell = fit_conic_ellipse(sample)
        if ell is None or not _plausible(ell, radius):
            continue
        d = np.abs(ellipse_distance(ell, pts))
        inl = d < tol
        n = int(inl.sum())
        rms = float(np.sqrt(np.mean(d[inl] ** 2))) if n else math.inf
        if best is None or n > int(best.sum()) or (n == int(best.sum()) and rms < best_rms):
            best, best_rms = inl, rms
    if best is None or best.sum() < LIMBUS_MIN_INLIERS:
        return None
    ell = fit_conic_ellipse(pts[best])
    if ell is None or not _plausible(ell, radius):
        return None
    inl = np.abs(ellipse_distance(ell, pts)) < tol
    return ell, inl


def _plausible(ell: Ellipse, radius: float) -> bool:
    return (ell.semi_major_px <= LIMBUS_MAX_AXIS_RATIO * ell.semi_minor_px
            and 0.5 * radius <= ell.semi_major_px <= 1.6 * radius)


def _both_sides(ell: Ellipse, pts: np.ndarray, horizontal: Point) -> bool:
    rel = pts - np.asarray(ell.centre_px)
    ang = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]) - math.atan2(horizontal[1], horizontal[0]))
    ang = (ang + 180.0) % 360.0 - 180.0
    return bool((np.abs(ang) <= LIMBUS_SIDE_WINDOW_DEG).any()
                and (np.abs(ang) >= 180.0 - LIMBUS_SIDE_WINDOW_DEG).any())


def fit_limbus(gray: np.ndarray, centre_guess: Point, radius_guess: float,
               aperture: Optional[LidAperture] = None, seed: int = 0) -> Optional[LimbusFit]:
    """
    Fit the limbus as an ellipse. Two passes: rays from the face-mesh iris
    centre, then again from the first fit's centre so the rays cross the
    edge square-on. None when too few edge points survive, or none survive
    on one side of the horizontal diameter (the scale ruler).
    """
    sigma = max(1.0, radius_guess / 40.0)
    img = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    rng = np.random.default_rng(seed)
    horizontal: Point = (1.0, 0.0)
    if aperture is not None:
        down = aperture.vertical()
        horizontal = (down[1], -down[0])
    centre, radius = centre_guess, radius_guess
    result: Optional[LimbusFit] = None
    for _ in range(2):
        pts = _edge_points(img, centre, radius, aperture)
        if len(pts) < LIMBUS_MIN_INLIERS:
            return None
        found = _ransac_ellipse(pts, radius, rng)
        if found is None:
            return None
        ell, inl = found
        if not _both_sides(ell, pts[inl], horizontal):
            return None
        d = ellipse_distance(ell, pts[inl])
        result = LimbusFit(ellipse=ell, rms_px=float(np.sqrt(np.mean(d ** 2))),
                           inliers=int(inl.sum()), rays=LIMBUS_RAYS)
        centre, radius = ell.centre_px, ell.semi_major_px
    return result


# ─────────────────────────────────────────────────────────────
# Pupil
# ─────────────────────────────────────────────────────────────

def _otsu(values: np.ndarray) -> float:
    hist = np.bincount(np.clip(values, 0, 255).astype(np.int64), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    between[~np.isfinite(between)] = -1
    return float(np.argmax(between))


def find_pupil(gray: np.ndarray, limbus: Ellipse, reflex: Optional[Point] = None
               ) -> Optional[Tuple[Point, float]]:
    """
    (centre, diameter) of the dark pupil inside the limbus, or None when it
    cannot be told from the iris. Where the reflex overlaps the pupil it
    leaves a hole or a notch in the dark region; fitting the outline's convex
    hull bridges both. `reflex` is accepted for the phone port's signature
    and future use; the hull makes masking it unnecessary.
    """
    r = limbus.semi_major_px
    cx, cy = limbus.centre_px
    half = int(math.ceil(r)) + 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    h, w = gray.shape
    if x0 < 0 or y0 < 0 or x0 + 2 * half + 1 > w or y0 + 2 * half + 1 > h:
        return None
    win = cv2.GaussianBlur(gray[y0:y0 + 2 * half + 1, x0:x0 + 2 * half + 1].astype(np.float32),
                           (0, 0), max(1.0, r / 40.0))
    ys, xs = np.mgrid[0:win.shape[0], 0:win.shape[1]]
    pts = np.stack([xs.ravel() + x0, ys.ravel() + y0], axis=1).astype(float)
    shrunk = Ellipse(centre_px=limbus.centre_px, semi_major_px=PUPIL_REGION_IRIS_RADII * limbus.semi_major_px,
                     semi_minor_px=PUPIL_REGION_IRIS_RADII * limbus.semi_minor_px, angle_deg=limbus.angle_deg)
    region = (ellipse_distance(shrunk, pts) <= 0).reshape(win.shape)
    values = win[region]
    if values.size < 50:
        return None
    t = _otsu(values)
    dark = region & (win <= t)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    best = None
    for i in range(1, n):
        off = math.hypot(cents[i][0] + x0 - cx, cents[i][1] + y0 - cy)
        if off <= PUPIL_MAX_OFFSET_IRIS_RADII * r and (best is None or stats[i, cv2.CC_STAT_AREA] >
                                                        stats[best, cv2.CC_STAT_AREA]):
            best = i
    if best is None:
        return None
    blob = labels == best
    iris_med, pupil_med = float(np.median(win[region & ~blob])), float(np.median(win[blob]))
    contrast = iris_med - pupil_med
    if contrast < PUPIL_MIN_CONTRAST or contrast < PUPIL_MIN_RELATIVE_CONTRAST * iris_med:
        return None
    contours, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(contours, key=len)).reshape(-1, 2).astype(float)
    # Densify the hull so long bridging edges carry as much weight as the arc they replace.
    dense: List[np.ndarray] = []
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        steps = max(1, int(math.ceil(np.hypot(*(b - a)))))
        dense.append(a + np.outer(np.arange(steps) / steps, b - a))
    ell = fit_conic_ellipse(np.vstack(dense) + [x0, y0])
    if ell is None:
        return None
    diameter = 2.0 * math.sqrt(ell.semi_major_px * ell.semi_minor_px)
    lo, hi = PUPIL_DIAMETER_RANGE_IRIS_RADII
    if not lo * r <= diameter <= hi * r:
        return None
    return ell.centre_px, diameter


# ─────────────────────────────────────────────────────────────
# Lid measures
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LidMetrics:
    fissure_height_px: Optional[float]
    scleral_show_px: Optional[float]
    limbus_covered_fraction: float


def lid_metrics(aperture: LidAperture, limbus: Ellipse) -> LidMetrics:
    """
    Fissure height and scleral show through the limbus centre, measured
    perpendicular to the canthal line; covered fraction = share of the
    limbus outline outside the lid opening.
    """
    centre_local = aperture.to_local(np.asarray([limbus.centre_px]))[0]
    up, dn = aperture.lid_y(np.asarray([centre_local[0]]))
    fissure = None if np.isnan(up[0]) else float(dn[0] - up[0])
    show = None
    if not np.isnan(up[0]):
        top = centre_local[1] - ellipse_extent(limbus, aperture.vertical())
        show = float(top - up[0])
    covered = 1.0 - float(aperture.contains(ellipse_points(limbus)).mean())
    return LidMetrics(fissure_height_px=fissure, scleral_show_px=show, limbus_covered_fraction=covered)
