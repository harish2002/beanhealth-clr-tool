"""
BeanHealth CLR — Synthetic eye images (test fixture, not physiology).

Renders one eye with known ground truth — limbus ellipse, pupil, corneal
reflex, lids — supersampled so edges fall at sub-pixel positions, then adds
sensor noise. exam.landmarks is tested against the truth it was drawn from.

Grey levels: skin 150, sclera 205, iris 90 (dark-iris preset: 40), pupil 25,
reflex 255. The lids are parabolas through the canthi; the face-mesh
contours handed to LidAperture are samples of those parabolas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from exam.capture.session import Ellipse
from exam.landmarks.eye import LidAperture

Point = Tuple[float, float]
SUPERSAMPLE = 4


@dataclass(frozen=True)
class EyeScene:
    size: Tuple[int, int] = (400, 300)                 # width, height
    limbus: Ellipse = field(default_factory=lambda: Ellipse(
        centre_px=(200.3, 150.6), semi_major_px=60.0, semi_minor_px=58.5, angle_deg=10.0))
    pupil_offset: Point = (0.0, 0.0)                    # from the limbus centre, px
    pupil_radius: float = 18.0                          # 0 = no pupil drawn
    reflex: Optional[Point] = (212.37, 143.81)
    reflex_radius: float = 3.2
    # Lid heights above / below the limbus centre at the middle of the eye (px).
    upper_lid: float = 50.0
    lower_lid: float = 62.0
    canthi_half_width: float = 140.0
    roll_deg: float = 0.0                               # rotates the lids about the limbus centre
    skin: float = 150.0
    sclera: float = 205.0
    iris: float = 90.0
    pupil: float = 25.0
    noise_sd: float = 2.0
    seed: int = 0

    def roll(self, p: Point) -> Point:
        """Rotate a point about the limbus centre by the roll (image coords, roll + = clockwise on screen)."""
        cx, cy = self.limbus.centre_px
        r = math.radians(self.roll_deg)
        dx, dy = p[0] - cx, p[1] - cy
        return (cx + dx * math.cos(r) - dy * math.sin(r), cy + dx * math.sin(r) + dy * math.cos(r))

    def aperture(self, n: int = 9) -> LidAperture:
        cx, cy = self.limbus.centre_px
        w = self.canthi_half_width
        xs = np.linspace(-w, w, n)
        upper = [self.roll((cx + x, cy - self.upper_lid * (1 - (x / w) ** 2))) for x in xs]
        lower = [self.roll((cx + x, cy + self.lower_lid * (1 - (x / w) ** 2))) for x in xs]
        return LidAperture(upper=tuple(upper), lower=tuple(lower),
                           canthi=(self.roll((cx - w, cy)), self.roll((cx + w, cy))))


def dark_iris(**kw) -> EyeScene:
    """Brown iris under visible light: the pupil barely differs from the iris."""
    return EyeScene(iris=40.0, pupil=32.0, **kw)


def render_eye(scene: EyeScene = EyeScene()) -> np.ndarray:
    w, h = scene.size
    s = SUPERSAMPLE
    ys, xs = np.mgrid[0:h * s, 0:w * s]
    # Supersample pixel centres: big pixel (i, j) covers [i − 0.5, i + 0.5).
    x = (xs + 0.5) / s - 0.5
    y = (ys + 0.5) / s - 0.5
    img = np.full((h * s, w * s), scene.sclera, dtype=np.float64)

    lim = scene.limbus
    phi = math.radians(lim.angle_deg)
    u = (x - lim.centre_px[0]) * math.cos(phi) + (y - lim.centre_px[1]) * math.sin(phi)
    v = -(x - lim.centre_px[0]) * math.sin(phi) + (y - lim.centre_px[1]) * math.cos(phi)
    img[(u / lim.semi_major_px) ** 2 + (v / lim.semi_minor_px) ** 2 <= 1] = scene.iris
    if scene.pupil_radius > 0:
        px, py = lim.centre_px[0] + scene.pupil_offset[0], lim.centre_px[1] + scene.pupil_offset[1]
        img[(x - px) ** 2 + (y - py) ** 2 <= scene.pupil_radius ** 2] = scene.pupil
    if scene.reflex is not None:
        img[(x - scene.reflex[0]) ** 2 + (y - scene.reflex[1]) ** 2 <= scene.reflex_radius ** 2] = 255.0

    # Lids in the rolled eye frame.
    cx, cy = lim.centre_px
    r = math.radians(-scene.roll_deg)
    lx = (x - cx) * math.cos(r) - (y - cy) * math.sin(r)
    ly = (x - cx) * math.sin(r) + (y - cy) * math.cos(r)
    shape = np.clip(1 - (lx / scene.canthi_half_width) ** 2, 0, None)
    outside = (ly < -scene.upper_lid * shape) | (ly > scene.lower_lid * shape)
    img[outside] = scene.skin

    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(scene.seed)
    small = small + rng.normal(0, scene.noise_sd, small.shape)
    return np.clip(np.round(small), 0, 255).astype(np.uint8)


def true_covered_fraction(scene: EyeScene, n: int = 3600) -> float:
    """Share of the limbus outline hidden by the drawn lids."""
    lim = scene.limbus
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    phi = math.radians(lim.angle_deg)
    u, v = lim.semi_major_px * np.cos(t), lim.semi_minor_px * np.sin(t)
    x = lim.centre_px[0] + u * math.cos(phi) - v * math.sin(phi)
    y = lim.centre_px[1] + u * math.sin(phi) + v * math.cos(phi)
    cx, cy = lim.centre_px
    r = math.radians(-scene.roll_deg)
    lx = (x - cx) * math.cos(r) - (y - cy) * math.sin(r)
    ly = (x - cx) * math.sin(r) + (y - cy) * math.cos(r)
    shape = np.clip(1 - (lx / scene.canthi_half_width) ** 2, 0, None)
    return float(((ly < -scene.upper_lid * shape) | (ly > scene.lower_lid * shape)).mean())
