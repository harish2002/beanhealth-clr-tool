"""
BeanHealth CLR — Knowledge Base Loader
=======================================

Loads the versioned clinical knowledge base (exam/kb/data/*.yaml), validates
it, and exposes it read-only. Rules read every clinical value from here —
none are hard-coded.

Validation runs in three layers and reports every problem at once:
  1. Schema    — each file against its pydantic model (unknown keys rejected).
  2. Version   — the content hash must match manifest.content_sha256, so a
                 KB version always identifies exactly one set of values.
                 After editing, run `python -m exam.kb stamp` and bump
                 kb_version.
  3. Cross-file — ids unique; references resolve; yoke pairs and every Parks
                 row re-derived from the muscle table and compared.

Usage:
    from exam.kb import load_kb
    kb = load_kb()
    kb.threshold("a_pattern_pd").value
    kb.version_info()        # stamp this on every output
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

import yaml
from pydantic import BaseModel, ValidationError

from exam.kb import derive
from exam.kb.schema import (
    Condition,
    ConditionsFile,
    ConventionsFile,
    Entry,
    Feature,
    FeatureRef,
    Manifest,
    Muscle,
    MusclesFile,
    ParksFile,
    Scalar,
    ScalarsFile,
    ScreeningFile,
)

DEFAULT_KB_DIR = Path(__file__).parent / "data"
MANIFEST = "manifest.yaml"

FILE_MODELS: Dict[str, Type[BaseModel]] = {
    "conventions.yaml":  ConventionsFile,
    "measurement.yaml":  ScalarsFile,
    "thresholds.yaml":   ScalarsFile,
    "muscles.yaml":      MusclesFile,
    "parks_3_step.yaml": ParksFile,
    "conditions.yaml":   ConditionsFile,
    "screening.yaml":    ScreeningFile,
    "capture.yaml":      ScalarsFile,
}

# Cardinal positions with exactly one yoke pair each.
YOKE_GAZES = ("right", "left", "up_right", "up_left", "down_right", "down_left")


class KBValidationError(Exception):
    """The KB failed validation. `problems` lists every issue found."""

    def __init__(self, problems: List[str]):
        self.problems = problems
        super().__init__(
            f"Knowledge base invalid ({len(problems)} problem(s)):\n  - " + "\n  - ".join(problems)
        )


@dataclass(frozen=True)
class KnowledgeBase:
    manifest:    Manifest
    conventions: ConventionsFile
    measurement: ScalarsFile
    thresholds:  ScalarsFile
    muscles_file: MusclesFile
    parks:       ParksFile
    conditions_file: ConditionsFile
    screening:   ScreeningFile
    capture:     ScalarsFile
    sha256:      str
    _scalars:    Dict[str, Scalar] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        scalars = {
            e.id: e
            for e in [
                *self.measurement.entries, *self.thresholds.entries,
                *self.muscles_file.plane_angles, *self.screening.severity_thresholds,
                *self.capture.entries,
            ]
        }
        object.__setattr__(self, "_scalars", scalars)

    # ── Identity ──
    @property
    def version(self) -> str:
        return self.manifest.kb_version

    def version_info(self) -> Dict[str, str]:
        """Stamp this on every measurement, finding and report."""
        return {"kb_version": self.version, "kb_sha256": self.sha256}

    # ── Lookups ──
    def scalar(self, entry_id: str) -> Scalar:
        try:
            return self._scalars[entry_id]
        except KeyError:
            raise KeyError(f"no numeric KB entry {entry_id!r}") from None

    def threshold(self, entry_id: str) -> Scalar:
        for e in self.thresholds.entries:
            if e.id == entry_id:
                return e
        raise KeyError(f"no threshold {entry_id!r}")

    @property
    def muscles(self) -> Dict[str, Muscle]:
        return {m.id: m.value for m in self.muscles_file.muscles}

    def feature(self, feature_id: str) -> Feature:
        for f in self.conditions_file.features:
            if f.id == feature_id:
                return f.value
        raise KeyError(f"no feature {feature_id!r}")

    def condition(self, condition_id: str) -> Condition:
        for c in self.conditions_file.conditions:
            if c.id == condition_id:
                return c.value
        raise KeyError(f"no condition {condition_id!r}")

    def entries(self) -> Iterator[Tuple[str, Entry[Any]]]:
        """Every entry in the KB, as (file, entry)."""
        yield from _entries(self)


# ─────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────

def _read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _format_errors(file_name: str, err: ValidationError) -> List[str]:
    return [
        f"{file_name}: {'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}"
        for e in err.errors()
    ]


def content_sha256(raw_files: Dict[str, Any]) -> str:
    """Hash of the parsed data (not the text) — comments and layout don't count."""
    canonical = json.dumps(raw_files, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_raw(kb_dir: Path) -> Tuple[Any, Dict[str, Any]]:
    """The raw manifest and the raw content of every data file."""
    manifest_raw = _read_yaml(kb_dir / MANIFEST)
    raw: Dict[str, Any] = {}
    for name in FILE_MODELS:
        path = kb_dir / name
        if path.exists():
            raw[name] = _read_yaml(path)
    return manifest_raw, raw


def load_kb(kb_dir: Optional[Path] = None, verify_hash: bool = True) -> KnowledgeBase:
    """
    Load and fully validate the KB. The default KB is cached after the first load.

    Raises:
        KBValidationError: listing every schema, version and cross-file problem.
    """
    if kb_dir is None and verify_hash:
        return _load_default()
    return _load(Path(kb_dir or DEFAULT_KB_DIR), verify_hash)


@lru_cache(maxsize=1)
def _load_default() -> KnowledgeBase:
    return _load(DEFAULT_KB_DIR, verify_hash=True)


def _load(kb_dir: Path, verify_hash: bool) -> KnowledgeBase:
    problems: List[str] = []

    try:
        manifest_raw, raw = read_raw(kb_dir)
    except (OSError, yaml.YAMLError) as e:
        raise KBValidationError([f"cannot read KB at {kb_dir}: {e}"]) from e

    manifest: Optional[Manifest] = None
    try:
        manifest = Manifest.model_validate(manifest_raw)
    except ValidationError as e:
        problems += _format_errors(MANIFEST, e)

    if manifest is not None and set(manifest.files) != set(FILE_MODELS):
        problems.append(
            f"{MANIFEST}: files must be exactly {sorted(FILE_MODELS)}, got {sorted(manifest.files)}"
        )

    parsed: Dict[str, BaseModel] = {}
    for name, model in FILE_MODELS.items():
        if name not in raw:
            problems.append(f"{name}: missing")
            continue
        try:
            parsed[name] = model.model_validate(raw[name])
        except ValidationError as e:
            problems += _format_errors(name, e)

    if problems:
        raise KBValidationError(problems)
    assert manifest is not None

    sha = content_sha256(raw)
    if verify_hash and sha != manifest.content_sha256:
        problems.append(
            f"{MANIFEST}: content changed since version {manifest.kb_version} was stamped "
            f"(content_sha256 {manifest.content_sha256[:12]}…, actual {sha[:12]}…). "
            "Bump kb_version and run `python -m exam.kb stamp`."
        )

    kb = KnowledgeBase(
        manifest=manifest,
        conventions=parsed["conventions.yaml"],           # type: ignore[arg-type]
        measurement=parsed["measurement.yaml"],           # type: ignore[arg-type]
        thresholds=parsed["thresholds.yaml"],             # type: ignore[arg-type]
        muscles_file=parsed["muscles.yaml"],              # type: ignore[arg-type]
        parks=parsed["parks_3_step.yaml"],                # type: ignore[arg-type]
        conditions_file=parsed["conditions.yaml"],        # type: ignore[arg-type]
        screening=parsed["screening.yaml"],               # type: ignore[arg-type]
        capture=parsed["capture.yaml"],                   # type: ignore[arg-type]
        sha256=sha,
    )
    problems += cross_validate(kb)
    if problems:
        raise KBValidationError(problems)
    return kb


# ─────────────────────────────────────────────────────────────
# Cross-file validation
# ─────────────────────────────────────────────────────────────

def _entries(kb: KnowledgeBase) -> Iterator[Tuple[str, Entry[Any]]]:
    c = kb.conventions
    for e in (c.laterality, c.sign_convention, c.reflex_reading, c.prism_dioptre_conversion):
        yield "conventions.yaml", e
    for e in kb.measurement.entries:
        yield "measurement.yaml", e
    for e in kb.thresholds.entries:
        yield "thresholds.yaml", e
    for e in [*kb.muscles_file.muscles, *kb.muscles_file.plane_angles, *kb.muscles_file.yoke_pairs]:
        yield "muscles.yaml", e
    for e in [*kb.parks.step1, *kb.parks.rows, kb.parks.inconclusive]:
        yield "parks_3_step.yaml", e
    for e in [*kb.conditions_file.features, *kb.conditions_file.conditions]:
        yield "conditions.yaml", e
    for e in [*kb.screening.severity_thresholds, *kb.screening.icd10]:
        yield "screening.yaml", e
    for e in kb.capture.entries:
        yield "capture.yaml", e


def cross_validate(kb: KnowledgeBase) -> List[str]:
    problems: List[str] = []
    problems += _check_unique_ids(kb)
    problems += _check_scalars(kb)
    problems += _check_muscles(kb)
    problems += _check_parks(kb)
    problems += _check_conditions(kb)
    return problems


def _check_unique_ids(kb: KnowledgeBase) -> List[str]:
    seen: Dict[str, str] = {}
    problems = []
    for file_name, e in _entries(kb):
        if e.id in seen:
            problems.append(f"{file_name}: id {e.id!r} already used in {seen[e.id]}")
        seen.setdefault(e.id, file_name)
    return problems


def _check_scalars(kb: KnowledgeBase) -> List[str]:
    problems = []
    for e in kb._scalars.values():
        if e.value is not None and e.plausible_range is not None:
            lo, hi = e.plausible_range
            if not lo <= e.value <= hi:
                problems.append(f"{e.id}: value {e.value} outside plausible_range {e.plausible_range}")
    return problems


def _check_muscles(kb: KnowledgeBase) -> List[str]:
    problems = []
    codes = [m.id for m in kb.muscles_file.muscles]
    if sorted(codes) != sorted({"MR", "LR", "SR", "IR", "SO", "IO"}) or len(codes) != 6:
        problems.append(f"muscles.yaml: need each of MR, LR, SR, IR, SO, IO once, got {codes}")
        return problems

    derived = derive.derive_yoke_pairs(kb.muscles)
    fields = derive.fields_of_action(kb.muscles)
    declared = {y.value.gaze: y for y in kb.muscles_file.yoke_pairs}
    for gaze in YOKE_GAZES:
        if gaze not in declared:
            problems.append(f"muscles.yaml: no yoke pair for gaze {gaze!r}")
            continue
        expected = derived.get(gaze)
        if expected is None or set(declared[gaze].value.muscles) != set(expected):
            problems.append(
                f"muscles.yaml: yoke pair {declared[gaze].id} for {gaze} is "
                f"{list(declared[gaze].value.muscles)}, but the muscle table puts "
                f"{fields.get(gaze, [])} in that field of action"
            )
    for gaze, entry in declared.items():
        if gaze not in YOKE_GAZES:
            problems.append(f"muscles.yaml: {entry.id}: {gaze!r} has no single yoke pair")
    return problems


def _check_parks(kb: KnowledgeBase) -> List[str]:
    problems = []
    muscles = kb.muscles
    derived = derive.derive_parks_rows(muscles)

    step1 = {s.value.hyper_eye: s for s in kb.parks.step1}
    for side in ("right", "left"):
        if side not in step1:
            problems.append(f"parks_3_step.yaml: no step-1 candidate set for {side} hyper")
            continue
        want = derive.derive_step1_candidates(muscles, side)
        got = sorted(step1[side].value.candidates)
        if got != want:
            problems.append(
                f"parks_3_step.yaml: {step1[side].id}: candidates {got}, but the muscle table gives {want}"
            )

    rows = kb.parks.rows
    if len(rows) != 8:
        problems.append(f"parks_3_step.yaml: expected 8 rows, got {len(rows)}")
    combos: Dict[Tuple[str, str, str], str] = {}
    for r in rows:
        v = r.value
        key = (v.hyper_eye, v.worse_on_gaze, v.worse_on_tilt)
        if key in combos:
            problems.append(f"parks_3_step.yaml: {r.id} repeats the pattern of {combos[key]}")
        combos[key] = r.id
        if v.muscle not in derived:
            problems.append(f"parks_3_step.yaml: {r.id}: {v.muscle} is not a cyclovertical muscle")
            continue
        if derived[v.muscle] != key:
            h, g, t = derived[v.muscle]
            problems.append(
                f"parks_3_step.yaml: {r.id}: row says {v.muscle} = hyper {v.hyper_eye}, gaze "
                f"{v.worse_on_gaze}, tilt {v.worse_on_tilt}; the muscle table gives hyper {h}, "
                f"gaze {g}, tilt {t}"
            )
        cands = step1.get(v.hyper_eye)
        if cands is not None and v.muscle not in cands.value.candidates:
            problems.append(f"parks_3_step.yaml: {r.id}: {v.muscle} is not a step-1 candidate for {v.hyper_eye} hyper")
    missing = set(derived) - {r.value.muscle for r in rows}
    if missing:
        problems.append(f"parks_3_step.yaml: no row for {sorted(missing)}")
    return problems


def _check_ref(where: str, ref: FeatureRef, features: Dict[str, Feature]) -> List[str]:
    feat = features.get(ref.feature)
    if feat is None:
        return [f"conditions.yaml: {where}: unknown feature {ref.feature!r}"]
    problems = []
    if set(ref.params) != set(feat.params):
        problems.append(
            f"conditions.yaml: {where}: {ref.feature} takes params {sorted(feat.params)}, got {sorted(ref.params)}"
        )
    for k, v in ref.params.items():
        allowed = feat.params.get(k)
        if allowed is not None and v not in allowed:
            problems.append(f"conditions.yaml: {where}: {ref.feature}.{k}={v!r} not one of {allowed}")
    return problems


def _check_conditions(kb: KnowledgeBase) -> List[str]:
    problems = []
    features = {f.id: f.value for f in kb.conditions_file.features}
    thresholds = {t.id for t in kb.thresholds.entries}
    for f in kb.conditions_file.features:
        for t in f.value.uses_thresholds:
            if t not in thresholds:
                problems.append(f"conditions.yaml: feature {f.id}: unknown threshold {t!r}")
    for c in kb.conditions_file.conditions:
        refs = [("required", r) for r in c.value.required] + [("supporting", r) for r in c.value.supporting]
        for i, group in enumerate(c.value.any_of):
            if not group:
                problems.append(f"conditions.yaml: {c.id}: any_of group {i} is empty")
            refs += [(f"any_of[{i}]", r) for r in group]
        for where, ref in refs:
            problems += _check_ref(f"{c.id}.{where}", ref, features)
    return problems
