"""
BeanHealth CLR — Phase 3 gaze examination.

Characterises ocular deviation across gaze positions: guided capture → frame
QC → landmarks → calibration → per-gaze measurement → rules engine → report.

Positioning: measures and characterizes ocular deviation; suggests likely
patterns. Decision support — not a diagnosis.

Sub-packages are added milestone by milestone; `exam.kb` (the clinical
knowledge base) comes first, because every later stage reads its values from
there rather than hard-coding them.
"""
