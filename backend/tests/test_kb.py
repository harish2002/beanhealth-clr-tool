"""
BeanHealth CLR Tool — Clinical Knowledge Base Tests (Phase 3, milestone 2)
==========================================================================

  • The shipped KB loads, validates and is version-stamped.
  • Every validation layer catches what it claims to (each mutation test
    breaks a copy of the KB in one way and checks the problem is named).
  • The muscle-model derivations reproduce the brief's Parks table and the
    standard yoke pairs on their own.
  • The KB agrees with the values the screening pipeline uses today.

Run:
    cd backend && pytest tests/test_kb.py -v
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from exam.kb import DEFAULT_KB_DIR, KBValidationError, load_kb
from exam.kb import derive
from exam.kb.loader import MANIFEST, content_sha256, read_raw

BACKEND = Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────
# Helpers — a mutable copy of the KB
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def kb_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "kb"
    shutil.copytree(DEFAULT_KB_DIR, dst)
    return dst


def _stamp(kb_dir: Path) -> None:
    manifest_path = kb_dir / MANIFEST
    manifest = yaml.safe_load(manifest_path.read_text())
    _, raw = read_raw(kb_dir)
    manifest["content_sha256"] = content_sha256(raw)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))


def mutate(kb_dir: Path, file_name: str, change: Callable[[Dict[str, Any]], None], restamp: bool = True) -> None:
    path = kb_dir / file_name
    data = yaml.safe_load(path.read_text())
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    if restamp:
        _stamp(kb_dir)


def problems_of(kb_dir: Path) -> str:
    with pytest.raises(KBValidationError) as exc:
        load_kb(kb_dir)
    return "\n".join(exc.value.problems)


def _find(entries, entry_id):
    return next(e for e in entries if e["id"] == entry_id)


# ─────────────────────────────────────────────────────────────
# The shipped KB
# ─────────────────────────────────────────────────────────────

class TestShippedKB:

    def test_loads_and_is_stamped(self):
        kb = load_kb()
        assert kb.version_info() == {"kb_version": kb.manifest.kb_version, "kb_sha256": kb.sha256}
        assert kb.sha256 == kb.manifest.content_sha256

    def test_default_load_is_cached(self):
        assert load_kb() is load_kb()

    def test_every_entry_has_provenance(self):
        for file_name, e in load_kb().entries():
            assert e.source.strip(), f"{file_name}: {e.id} has no source"

    def test_nothing_is_approved_yet(self):
        """No clinician has reviewed this KB; approval must be recorded, not assumed."""
        approved = [e.id for _, e in load_kb().entries() if e.review_status == "approved"]
        assert approved == []

    def test_lookups(self):
        kb = load_kb()
        assert kb.threshold("a_pattern_pd").value == 10
        assert kb.threshold("v_pattern_pd").value == 15
        assert kb.threshold("incomitance_horizontal_pd").value == 5
        assert kb.threshold("decision_margin_k").value == 2
        assert kb.scalar("iris_diameter_mm").unit == "mm"
        assert kb.muscles["SO"].nerve == "CN IV"
        assert kb.condition("cn3_pupil_involving").urgency == "URGENT"
        with pytest.raises(KeyError):
            kb.threshold("no_such_threshold")

    def test_unset_thresholds_are_null_not_guessed(self):
        kb = load_kb()
        for tid in ("anisocoria_tolerance_mm", "duane_fissure_narrowing_ratio",
                    "duction_limit_fraction", "scleral_show_mm"):
            t = kb.threshold(tid)
            assert t.value is None and t.review_status == "draft"

    def test_conflicting_values_keep_the_alternative(self):
        kb = load_kb()
        assert kb.scalar("iris_diameter_mm").alternatives[0].value == 11.7
        assert kb.scalar("hirschberg_ratio_deg_per_mm").alternatives[0].unit == "PD/mm"

    def test_cli_check(self):
        r = subprocess.run([sys.executable, "-m", "exam.kb", "check"], cwd=BACKEND,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert load_kb().version in r.stdout


# ─────────────────────────────────────────────────────────────
# Derivations, checked against the brief independently
# ─────────────────────────────────────────────────────────────

BRIEF_PARKS = {   # brief §5.3: muscle → (hyper, worse on gaze, worse on tilt)
    "R-SO": ("right", "left", "right"), "L-SR": ("right", "left", "left"),
    "L-IO": ("right", "right", "right"), "R-IR": ("right", "right", "left"),
    "L-SO": ("left", "right", "left"),  "R-SR": ("left", "right", "right"),
    "R-IO": ("left", "left", "left"),   "L-IR": ("left", "left", "right"),
}


class TestDerivations:

    def test_parks_rows_match_the_brief(self):
        assert derive.derive_parks_rows(load_kb().muscles) == BRIEF_PARKS

    def test_step1_candidates_match_the_brief(self):
        muscles = load_kb().muscles
        assert derive.derive_step1_candidates(muscles, "right") == sorted(["R-IR", "R-SO", "L-SR", "L-IO"])
        assert derive.derive_step1_candidates(muscles, "left") == sorted(["L-IR", "L-SO", "R-SR", "R-IO"])

    def test_yoke_pairs(self):
        assert derive.derive_yoke_pairs(load_kb().muscles) == {
            "right": ("R-LR", "L-MR"), "left": ("R-MR", "L-LR"),
            "up_right": ("R-SR", "L-IO"), "up_left": ("R-IO", "L-SR"),
            "down_right": ("R-IR", "L-SO"), "down_left": ("R-SO", "L-IR"),
        }

    def test_horizontal_recti_have_no_parks_signature(self):
        muscles = load_kb().muscles
        assert derive.parks_signature("R", muscles["MR"]) is None
        assert derive.parks_signature("L", muscles["LR"]) is None


# ─────────────────────────────────────────────────────────────
# Each validation layer catches what it claims to
# ─────────────────────────────────────────────────────────────

class TestValidation:

    def test_copy_is_valid(self, kb_copy):
        load_kb(kb_copy)

    def test_wrong_parks_muscle(self, kb_copy):
        mutate(kb_copy, "parks_3_step.yaml",
               lambda d: _find(d["rows"], "parks_r_so")["value"].update(muscle="L-SR"))
        p = problems_of(kb_copy)
        assert "parks_r_so: row says L-SR" in p
        assert "no row for ['R-SO']" in p

    def test_wrong_parks_tilt(self, kb_copy):
        mutate(kb_copy, "parks_3_step.yaml",
               lambda d: _find(d["rows"], "parks_l_io")["value"].update(worse_on_tilt="left"))
        assert "parks_l_io" in problems_of(kb_copy)

    def test_muscle_table_typo_breaks_derived_tables(self, kb_copy):
        mutate(kb_copy, "muscles.yaml",
               lambda d: _find(d["muscles"], "SO")["value"]["diagnostic_position"].update(horizontal="out"))
        p = problems_of(kb_copy)
        assert "yoke" in p and "parks_r_so" in p

    def test_wrong_yoke_pair(self, kb_copy):
        mutate(kb_copy, "muscles.yaml",
               lambda d: _find(d["yoke_pairs"], "yoke_up_right")["value"].update(muscles=["R-SR", "L-SR"]))
        assert "yoke_up_right" in problems_of(kb_copy)

    def test_wrong_step1_candidates(self, kb_copy):
        mutate(kb_copy, "parks_3_step.yaml",
               lambda d: _find(d["step1"], "parks_step1_right_hyper")["value"].update(
                   candidates=["R-IR", "R-SO", "L-SR", "R-IO"]))
        assert "parks_step1_right_hyper" in problems_of(kb_copy)

    def test_unknown_feature(self, kb_copy):
        mutate(kb_copy, "conditions.yaml",
               lambda d: _find(d["conditions"], "cn6_palsy")["value"]["required"].append({"feature": "telepathy"}))
        assert "unknown feature 'telepathy'" in problems_of(kb_copy)

    def test_bad_feature_param(self, kb_copy):
        def change(d):
            _find(d["conditions"], "cn6_palsy")["value"]["required"][0]["params"]["direction"] = "sideways"
        mutate(kb_copy, "conditions.yaml", change)
        assert "direction='sideways'" in problems_of(kb_copy)

    def test_unknown_threshold_reference(self, kb_copy):
        mutate(kb_copy, "conditions.yaml",
               lambda d: _find(d["features"], "a_pattern")["value"]["uses_thresholds"].append("nope_pd"))
        assert "unknown threshold 'nope_pd'" in problems_of(kb_copy)

    def test_duplicate_id(self, kb_copy):
        mutate(kb_copy, "thresholds.yaml", lambda d: d["entries"].append(dict(d["entries"][0])))
        assert "already used" in problems_of(kb_copy)

    def test_misspelt_key_is_rejected(self, kb_copy):
        mutate(kb_copy, "thresholds.yaml", lambda d: d["entries"][0].update(sauce="typo of source"))
        assert "sauce" in problems_of(kb_copy)

    def test_approved_needs_reviewer(self, kb_copy):
        mutate(kb_copy, "thresholds.yaml",
               lambda d: _find(d["entries"], "a_pattern_pd").update(review_status="approved"))
        assert "reviewed_by" in problems_of(kb_copy)

    def test_approved_needs_value(self, kb_copy):
        mutate(kb_copy, "thresholds.yaml",
               lambda d: _find(d["entries"], "scleral_show_mm").update(
                   review_status="approved", reviewed_by="Dr X", reviewed_on="2026-11-01"))
        assert "needs a value" in problems_of(kb_copy)

    def test_value_outside_plausible_range(self, kb_copy):
        mutate(kb_copy, "measurement.yaml",
               lambda d: _find(d["entries"], "hirschberg_ratio_plausible_pd_per_mm").update(value=30))
        assert "outside plausible_range" in problems_of(kb_copy)

    def test_edit_without_restamp_is_rejected(self, kb_copy):
        mutate(kb_copy, "thresholds.yaml",
               lambda d: _find(d["entries"], "a_pattern_pd").update(value=12), restamp=False)
        assert "content changed" in problems_of(kb_copy)
        load_kb(kb_copy, verify_hash=False)   # the content itself is still valid

    def test_comments_do_not_change_the_hash(self, kb_copy):
        path = kb_copy / "thresholds.yaml"
        path.write_text("# a new comment\n" + path.read_text())
        load_kb(kb_copy)

    def test_missing_file(self, kb_copy):
        (kb_copy / "conditions.yaml").unlink()
        assert "conditions.yaml: missing" in problems_of(kb_copy)

    def test_all_problems_reported_together(self, kb_copy):
        mutate(kb_copy, "parks_3_step.yaml",
               lambda d: _find(d["rows"], "parks_l_io")["value"].update(worse_on_tilt="left"))
        mutate(kb_copy, "conditions.yaml",
               lambda d: _find(d["conditions"], "cn6_palsy")["value"]["required"].append({"feature": "telepathy"}))
        p = problems_of(kb_copy)
        assert "parks_l_io" in p and "telepathy" in p


# ─────────────────────────────────────────────────────────────
# Agreement with the screening pipeline's values
# ─────────────────────────────────────────────────────────────

class TestScreeningParity:
    """Until screening reads the KB directly, the two must not drift apart."""

    def test_measurement_constants(self):
        from utils import constants as c
        kb = load_kb()
        assert kb.scalar("iris_diameter_mm").value == 2 * c.IRIS_RADIUS_MM
        assert kb.scalar("hirschberg_ratio_deg_per_mm").value == c.HIRSCHBERG_CONSTANT

    def test_severity_thresholds(self):
        from utils import constants as c
        kb = load_kb()
        assert kb.scalar("severity_mild_deg").value == c.SEVERITY_MILD_DEG
        assert kb.scalar("severity_moderate_deg").value == c.SEVERITY_MODERATE_DEG
        assert kb.scalar("severity_severe_deg").value == c.SEVERITY_SEVERE_DEG

    def test_icd10_codes(self):
        from utils import constants as c
        kb_codes = {e.value.condition: e.value.code for e in load_kb().screening.icd10}
        assert kb_codes == c.ICD10

    def test_reflex_reading_matches_module6(self):
        from pipeline.module6_classify import _DIRECTION_TO_CONDITION
        word = {"eso": "Esotropia", "exo": "Exotropia", "hyper": "Hypertropia", "hypo": "Hypotropia"}
        reading = load_kb().conventions.reflex_reading.value.model_dump()
        assert {d: word[t] for d, t in reading.items()} == {
            d: name for d, (name, _code) in _DIRECTION_TO_CONDITION.items()
        }
