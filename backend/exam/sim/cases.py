"""
BeanHealth CLR — Synthetic Cases
=================================

Named synthetic patients for engine tests (brief §9). Strengths and sizes are
fixture values picked to make each pattern unmistakable — not clinical data.
"""

from __future__ import annotations

from typing import Dict

from exam.sim.model import Noise, SimCase

CASES: Dict[str, SimCase] = {
    c.name: c for c in [
        SimCase("normal"),
        SimCase("comitant_eso_20", comitant_pd=(20.0, 0.0)),
        SimCase("r_cn6", weak={"R-LR": 0.2}),
        SimCase("r_cn4", weak={"R-SO": 0.2}),
        SimCase("l_cn3_complete", weak={"L-MR": 0.1, "L-SR": 0.1, "L-IR": 0.1, "L-IO": 0.1},
                fissure_mm={"R": 10.0, "L": 6.0}),
        SimCase("l_cn3_pupil_involving", weak={"L-MR": 0.1, "L-SR": 0.1, "L-IR": 0.1, "L-IO": 0.1},
                fissure_mm={"R": 10.0, "L": 6.0}, pupil_mm={"R": 4.0, "L": 6.0}),
        SimCase("a_pattern", h_per_up_pd=8.0),
        SimCase("v_pattern", h_per_up_pd=-10.0),
        # Limited abduction without the primary eso a CN VI palsy brings, and
        # the fissure narrowing on adduction.
        SimCase("r_duane_like", weak={"R-LR": 0.2}, tonic_h_deg=0.0,
                fissure_ratio_adduction={"R": 0.7, "L": 1.0}),
        # Lid retraction with limited elevation (restriction modelled as weakness —
        # Tier 1 cannot tell them apart).
        SimCase("r_ted_like", weak={"R-SR": 0.2}, scleral_show_mm={"R": 2.0, "L": 0.0}),
        # Up / down difference sitting exactly on the A-pattern threshold.
        SimCase("noisy_borderline", h_per_up_pd=5.0),
    ]
}

# Noise used for each case unless a test says otherwise.
CASE_NOISE: Dict[str, Noise] = {"noisy_borderline": Noise(pd=2.0)}

CYCLOVERTICAL = ("SR", "IR", "SO", "IO")


def parks_case(eye_muscle: str, strength: float = 0.2) -> SimCase:
    """A single weak cyclovertical muscle, e.g. parks_case("R-SO")."""
    return SimCase(f"weak_{eye_muscle}", weak={eye_muscle: strength})
