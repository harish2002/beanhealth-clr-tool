"""
BeanHealth CLR Tool — Synthetic Case Generator Tests (Phase 3, milestone 3)
===========================================================================

The generator is a test fixture, so these tests check it is self-consistent
and agrees with the KB's own muscle derivations — not that it is physiology.

Run:
    cd backend && pytest tests/test_sim.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from exam.kb import load_kb
from exam.kb.derive import derive_parks_rows
from exam.models import PROTOCOL_POSITIONS
from exam.sim import CASES, CYCLOVERTICAL, Noise, parks_case, simulate, true_deviation, true_duction, true_tilt

pytestmark = pytest.mark.unit

GAZES = PROTOCOL_POSITIONS["nine_gaze"]


def test_normal_is_zero_everywhere():
    case = CASES["normal"]
    for gaze in GAZES:
        assert true_deviation(case, gaze) == pytest.approx((0.0, 0.0), abs=1e-9)
    for side in ("right", "left"):
        assert true_tilt(case, side) == pytest.approx(0.0, abs=1e-9)
    for eye in ("R", "L"):
        for d in ("abduction", "adduction", "elevation", "depression"):
            assert true_duction(case, eye, d) == pytest.approx(1.0)


def test_comitant_offset_is_the_same_in_every_gaze():
    for gaze in GAZES:
        assert true_deviation(CASES["comitant_eso_20"], gaze) == pytest.approx((20.0, 0.0))


class TestLaterality:
    def test_right_lr_palsy_eso_grows_on_right_gaze(self):
        case = CASES["r_cn6"]
        right, primary, left = (true_deviation(case, g)[0] for g in ("right", "primary", "left"))
        assert primary > 0                       # eso in primary
        assert right > primary + 10              # much worse looking right
        assert left == pytest.approx(primary)    # the left eye's LR is fine
        assert true_duction(case, "R", "abduction") == pytest.approx(0.2)
        assert true_duction(case, "L", "abduction") == pytest.approx(1.0)

    def test_mirror_image_palsy_mirrors_the_grid(self):
        right = parks_case("R-LR")
        left = parks_case("L-LR")
        mirror = {"right": "left", "left": "right", "up_right": "up_left", "up_left": "up_right",
                  "down_right": "down_left", "down_left": "down_right"}
        for gaze in GAZES:
            h_r, v_r = true_deviation(right, gaze)
            h_l, v_l = true_deviation(left, mirror.get(gaze, gaze))
            assert h_r == pytest.approx(h_l)
            assert v_r == pytest.approx(-v_l)    # right hyper ↔ left hyper


@pytest.mark.parametrize("eye", ["R", "L"])
@pytest.mark.parametrize("code", CYCLOVERTICAL)
def test_weak_muscle_reproduces_the_kb_parks_signature(eye, code):
    """The simulator's physics and the KB's derivation agree on all 8 muscles."""
    muscle = f"{eye}-{code}"
    hyper, worse_gaze, worse_tilt = derive_parks_rows(load_kb().muscles)[muscle]
    case = parks_case(muscle)
    sign = 1 if hyper == "right" else -1
    primary = sign * true_deviation(case, "primary")[1]
    on = {g: sign * true_deviation(case, g)[1] for g in ("right", "left")}
    tilt = {s: sign * true_tilt(case, s) for s in ("right", "left")}
    assert primary > 0
    assert on[worse_gaze] > on["left" if worse_gaze == "right" else "right"]
    assert tilt[worse_tilt] > tilt["left" if worse_tilt == "right" else "right"]


@pytest.mark.parametrize("eye", ["R", "L"])
@pytest.mark.parametrize("code", CYCLOVERTICAL)
def test_single_palsy_never_reverses_the_hyper_eye(eye, code):
    """Basis for reading a reversal as 'steps conflict' in the 3-step test."""
    case = parks_case(f"{eye}-{code}")
    sign = 1 if true_deviation(case, "primary")[1] > 0 else -1
    for gaze in GAZES:
        assert sign * true_deviation(case, gaze)[1] >= -1e-9
    for side in ("right", "left"):
        assert sign * true_tilt(case, side) >= -1e-9


class TestSimulate:
    def test_exact_values_without_a_seed(self):
        m = simulate(CASES["comitant_eso_20"], Noise(pd=1.5))
        cell = m.grid.cells["up"]
        assert cell.horizontal_pd.value == pytest.approx(20.0)
        assert cell.horizontal_pd.error == 1.5

    def test_seeded_noise_is_reproducible_and_present(self):
        a = simulate(CASES["normal"], Noise(seed=7))
        b = simulate(CASES["normal"], Noise(seed=7))
        c = simulate(CASES["normal"], Noise(seed=8))
        assert a == b
        assert a != c
        assert any(abs(cell.horizontal_pd.value) > 0 for cell in a.grid.cells.values())

    def test_five_gaze_protocol(self):
        m = simulate(CASES["normal"], protocol="five_gaze")
        assert set(m.grid.cells) == set(PROTOCOL_POSITIONS["five_gaze"])

    def test_insufficient_positions_and_no_tilt(self):
        m = simulate(CASES["normal"], insufficient=("up",), tilt=False)
        assert m.grid.cells["up"].status == "INSUFFICIENT_DATA"
        assert m.tilt is None

    def test_confidence_is_carried(self):
        m = simulate(CASES["normal"], Noise(confidence="LOW"))
        assert m.grid.cells["primary"].horizontal_pd.confidence == "LOW"
        assert m.eyes["R"].ductions["abduction"].confidence == "LOW"
