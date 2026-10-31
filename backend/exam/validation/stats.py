"""
BeanHealth CLR — Validation statistics (brief §9)
=================================================

  bland_altman   device vs reference: bias, SD of differences, 95% limits
                 of agreement, and 95% confidence intervals for the bias and
                 both limits (Bland & Altman 1999, approximate SE of a limit
                 = SD·√(3/n)).
  agreement      device pattern calls vs clinician: sensitivity,
                 specificity, PPV, NPV and Cohen's κ. INCONCLUSIVE device
                 calls are counted, reported as a rate and left out of the
                 2×2 table — they are neither right nor wrong.
  cohen_kappa    chance-corrected agreement of two raters (any categories).

Pure functions over plain lists; the database queries live in from_db.py.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import NormalDist
from typing import Hashable, List, Optional, Sequence

from pydantic import Field

from exam.models import Strict

_Z = NormalDist().inv_cdf(0.975)


def _t975(dof: int) -> float:
    """Two-sided 95% Student t quantile (normal for large dof)."""
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, dof))
    except ImportError:                       # scipy is optional
        return _Z


class BlandAltman(Strict):
    n: int
    bias: Optional[float] = None              # mean(device − reference)
    sd: Optional[float] = None
    loa_low: Optional[float] = None           # bias − 1.96·SD
    loa_high: Optional[float] = None
    bias_ci95: Optional[List[float]] = None
    loa_low_ci95: Optional[List[float]] = None
    loa_high_ci95: Optional[List[float]] = None
    mean_abs_error: Optional[float] = None


def bland_altman(device: Sequence[float], reference: Sequence[float]) -> BlandAltman:
    if len(device) != len(reference):
        raise ValueError("device and reference need the same length")
    d = [a - b for a, b in zip(device, reference)]
    n = len(d)
    if n == 0:
        return BlandAltman(n=0)
    bias = sum(d) / n
    mae = sum(abs(x) for x in d) / n
    if n < 2:
        return BlandAltman(n=n, bias=bias, mean_abs_error=mae)
    sd = math.sqrt(sum((x - bias) ** 2 for x in d) / (n - 1))
    t = _t975(n - 1)
    se_bias, se_loa = sd / math.sqrt(n), sd * math.sqrt(3.0 / n)
    lo, hi = bias - _Z * sd, bias + _Z * sd
    return BlandAltman(n=n, bias=bias, sd=sd, loa_low=lo, loa_high=hi,
                       bias_ci95=[bias - t * se_bias, bias + t * se_bias],
                       loa_low_ci95=[lo - t * se_loa, lo + t * se_loa],
                       loa_high_ci95=[hi - t * se_loa, hi + t * se_loa], mean_abs_error=mae)


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> Optional[float]:
    if len(a) != len(b):
        raise ValueError("raters need the same number of ratings")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


class Agreement(Strict):
    n: int                                    # pairs, inconclusive included
    inconclusive: int = 0
    inconclusive_rate: Optional[float] = None
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    ppv: Optional[float] = None
    npv: Optional[float] = None
    kappa: Optional[float] = None
    sensitivity_ci95: Optional[List[float]] = Field(default=None)
    specificity_ci95: Optional[List[float]] = Field(default=None)


def _ratio(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def wilson(k: int, n: int) -> Optional[List[float]]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return None
    p, z2 = k / n, _Z * _Z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = _Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def agreement(device: Sequence[Optional[bool]], reference: Sequence[bool]) -> Agreement:
    """device: True (reported), False (absent) or None (INCONCLUSIVE); reference: the clinician's call."""
    if len(device) != len(reference):
        raise ValueError("device and reference need the same length")
    decided = [(d, r) for d, r in zip(device, reference) if d is not None]
    tp = sum(d and r for d, r in decided)
    fp = sum(d and not r for d, r in decided)
    tn = sum((not d) and (not r) for d, r in decided)
    fn = sum((not d) and r for d, r in decided)
    n = len(device)
    return Agreement(
        n=n, inconclusive=n - len(decided), inconclusive_rate=_ratio(n - len(decided), n),
        tp=tp, fp=fp, tn=tn, fn=fn,
        sensitivity=_ratio(tp, tp + fn), specificity=_ratio(tn, tn + fp),
        ppv=_ratio(tp, tp + fp), npv=_ratio(tn, tn + fn),
        kappa=cohen_kappa([d for d, _ in decided], [r for _, r in decided]),
        sensitivity_ci95=wilson(tp, tp + fn), specificity_ci95=wilson(tn, tn + fp),
    )
