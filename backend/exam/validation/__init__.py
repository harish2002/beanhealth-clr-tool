"""
BeanHealth CLR — Validation (brief §9): Bland–Altman device vs orthoptist, pattern agreement
(sensitivity / specificity / Cohen's κ), per-device error breakdown.

    python -m exam.validation --db exam.sqlite
"""

from exam.validation.from_db import VALIDATION_VERSION, ValidationReport, summary, validate
from exam.validation.stats import Agreement, BlandAltman, agreement, bland_altman, cohen_kappa, wilson

__all__ = ["VALIDATION_VERSION", "Agreement", "BlandAltman", "ValidationReport", "agreement", "bland_altman",
           "cohen_kappa", "summary", "validate", "wilson"]
