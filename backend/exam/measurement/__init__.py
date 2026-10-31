"""
BeanHealth CLR — Per-gaze measurement (brief §4.5): the deviation grid with errors, head-tilt
deviations, ductions, lids and pupils — the rules engine's input.

    from exam.measurement import measure
    grid = measure(session)               # runs QC and calibration if not given
    result = evaluate(grid.measurements)  # exam.engine
"""

from exam.measurement.measure import (
    MEASUREMENT_VERSION,
    EyeRotation,
    GridMeasurement,
    MeasurementFlag,
    PositionMeasurement,
    effective_n,
    fuse,
    measure,
)

__all__ = ["MEASUREMENT_VERSION", "EyeRotation", "GridMeasurement", "MeasurementFlag", "PositionMeasurement",
           "effective_n", "fuse", "measure"]
