"""
BeanHealth CLR — Clinician report (brief §8): JSON, HTML and PDF.

    from exam.report import build_report, render_html, render_pdf
    report = build_report(session, examine(session))
"""

from exam.report.build import REPORT_VERSION, ExamReport, ReportPatient, build_report, grid_layout
from exam.report.html import render_html
from exam.report.pdf import render_pdf

__all__ = ["REPORT_VERSION", "ExamReport", "ReportPatient", "build_report", "grid_layout", "render_html",
           "render_pdf"]
