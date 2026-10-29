"""
BeanHealth CLR — Muscle-Model Derivations
==========================================

Derive yoke pairs and the Parks 3-step table from the muscle table alone
(actions + diagnostic positions). The loader uses these to cross-check the
hand-written KB files, so a typo in either one fails at load time instead of
producing a wrong finding. The rules engine reuses them.

Conventions: eyes are "R" / "L" (the patient's), sides are "right" / "left"
(the patient's), and an eye muscle id is "R-SO", "L-IR", ...

Logic, per eye muscle:
  • Field of action — the gaze where it is the main mover. A diagnostic
    horizontal position of "in" is toward the nose: for the right eye that is
    the patient's left gaze, for the left eye their right gaze.
  • Yoke pair — the two muscles, one per eye, sharing a field of action.
  • Parks 3-step for a weak cyclovertical muscle:
      step 1  a weak depressor leaves its eye hyper; a weak elevator leaves
              the fellow eye hyper;
      step 2  the deviation is worse in the muscle's field of action;
      step 3  a weak intorter is worse on head tilt toward its own side, a
              weak extorter on tilt toward the fellow side (Bielschowsky).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from exam.kb.schema import Muscle

EYES = ("R", "L")
CYCLOVERTICAL = ("SR", "IR", "SO", "IO")


def side_of(eye: str) -> str:
    return "right" if eye == "R" else "left"


def fellow(eye: str) -> str:
    return "L" if eye == "R" else "R"


def eye_muscle_id(eye: str, code: str) -> str:
    return f"{eye}-{code}"


def split_eye_muscle(eye_muscle: str) -> Tuple[str, str]:
    eye, code = eye_muscle.split("-")
    return eye, code


def _all_actions(muscle: Muscle) -> List[str]:
    a = muscle.actions
    return [*a.primary, *a.secondary, *a.tertiary]


def vertical_action(muscle: Muscle) -> Optional[str]:
    return next((x for x in _all_actions(muscle) if x in ("elevation", "depression")), None)


def torsional_action(muscle: Muscle) -> Optional[str]:
    return next((x for x in _all_actions(muscle) if x in ("intorsion", "extorsion")), None)


def gaze_side(eye: str, horizontal: str) -> str:
    """The patient's gaze side that moves this eye in ("in") or out ("out")."""
    if horizontal == "out":
        return side_of(eye)
    return side_of(fellow(eye))


def field_of_action(eye: str, muscle: Muscle) -> str:
    pos = muscle.diagnostic_position
    side = gaze_side(eye, pos.horizontal)
    return f"{pos.vertical}_{side}" if pos.vertical else side


def fields_of_action(muscles: Dict[str, Muscle]) -> Dict[str, List[str]]:
    """Gaze → every eye muscle whose field of action it is."""
    by_gaze: Dict[str, List[str]] = {}
    for code, muscle in muscles.items():
        for eye in EYES:
            by_gaze.setdefault(field_of_action(eye, muscle), []).append(eye_muscle_id(eye, code))
    return {gaze: sorted(ms) for gaze, ms in by_gaze.items()}


def derive_yoke_pairs(muscles: Dict[str, Muscle]) -> Dict[str, Tuple[str, str]]:
    """Gaze → (right-eye muscle, left-eye muscle), for gazes with exactly one of each."""
    pairs: Dict[str, Tuple[str, str]] = {}
    for gaze, ms in fields_of_action(muscles).items():
        right = [m for m in ms if m.startswith("R-")]
        left = [m for m in ms if m.startswith("L-")]
        if len(right) == 1 and len(left) == 1:
            pairs[gaze] = (right[0], left[0])
    return pairs


def parks_signature(eye: str, muscle: Muscle) -> Optional[Tuple[str, str, str]]:
    """(hyper eye, worse-on gaze, worse-on tilt) for a weak muscle; None if not cyclovertical."""
    vertical, torsion = vertical_action(muscle), torsional_action(muscle)
    if vertical is None or torsion is None:
        return None
    hyper = side_of(eye) if vertical == "depression" else side_of(fellow(eye))
    worse_gaze = gaze_side(eye, muscle.diagnostic_position.horizontal)
    worse_tilt = side_of(eye) if torsion == "intorsion" else side_of(fellow(eye))
    return hyper, worse_gaze, worse_tilt


def derive_parks_rows(muscles: Dict[str, Muscle]) -> Dict[str, Tuple[str, str, str]]:
    """Eye muscle → its Parks signature, for every cyclovertical muscle of both eyes."""
    rows: Dict[str, Tuple[str, str, str]] = {}
    for code, muscle in muscles.items():
        for eye in EYES:
            sig = parks_signature(eye, muscle)
            if sig is not None:
                rows[eye_muscle_id(eye, code)] = sig
    return rows


def derive_step1_candidates(muscles: Dict[str, Muscle], hyper_side: str) -> List[str]:
    return sorted(m for m, sig in derive_parks_rows(muscles).items() if sig[0] == hyper_side)


def describe(rows: Iterable[Tuple[str, Tuple[str, str, str]]]) -> str:
    return "; ".join(f"{m}: hyper {h}, gaze {g}, tilt {t}" for m, (h, g, t) in rows)
