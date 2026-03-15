"""
BeanHealth CLR Tool — Custom Exception Hierarchy

All pipeline errors inherit from CLRPipelineError.
Each error carries a machine-readable code and a human-readable message
so the API layer can return the right INCONCLUSIVE reason to the frontend.
"""


class CLRPipelineError(Exception):
    """Base class for all CLR pipeline errors."""

    def __init__(self, code: str, human_message: str):
        self.code = code
        self.human_message = human_message
        super().__init__(f"[{code}] {human_message}")


class DetectionError(CLRPipelineError):
    """
    Raised by Module 1 when the face or eyes cannot be reliably detected.
    Always results in INCONCLUSIVE output — never a guessed result.
    """

    CODES = {
        "no_face":          "No face detected in the image. Ensure the face is fully visible.",
        "eyes_not_visible": "Eyes could not be located. Look directly at the camera.",
        "eyes_closed":      "Eyes appear to be closed. Please open both eyes fully.",
        "not_frontal":      "Face is not facing the camera. Look straight ahead.",
        "crop_too_small":   "Eye region is too small to analyse. Move closer to the camera.",
        "low_confidence":   "Face detection confidence is too low. Improve lighting and retry.",
    }

    def __init__(self, code: str):
        message = self.CODES.get(code, f"Detection failed: {code}")
        super().__init__(code, message)


class CLRError(CLRPipelineError):
    """
    Raised by Module 3 when the corneal light reflex cannot be located.
    Always results in INCONCLUSIVE — a missing CLR must never be guessed.
    """

    CODES = {
        "no_flash":             "No torch/flash detected. Enable the torch and retry.",
        "no_reflex_left":       "No corneal light reflex found in the left eye.",
        "no_reflex_right":      "No corneal light reflex found in the right eye.",
        "no_reflex_both":       "No corneal light reflex found in either eye.",
        "ambiguous_reflex_left":  "Multiple bright spots in left eye — cannot determine CLR.",
        "ambiguous_reflex_right": "Multiple bright spots in right eye — cannot determine CLR.",
    }

    def __init__(self, code: str):
        message = self.CODES.get(code, f"CLR detection failed: {code}")
        super().__init__(code, message)


class PupilError(CLRPipelineError):
    """Raised by Module 2 when pupil centre cannot be determined."""

    CODES = {
        "pupil_not_found_left":  "Pupil not found in left eye crop.",
        "pupil_not_found_right": "Pupil not found in right eye crop.",
        "pupil_not_found_both":  "Pupil not found in either eye.",
    }

    def __init__(self, code: str):
        message = self.CODES.get(code, f"Pupil detection failed: {code}")
        super().__init__(code, message)


class PipelineError(CLRPipelineError):
    """
    Raised by Modules 4–7 for unexpected arithmetic or logic failures
    (e.g. division by zero, invalid inputs).
    Results in a 500 ERROR status — not INCONCLUSIVE.
    """

    def __init__(self, message: str, code: str = "pipeline_error"):
        super().__init__(code, message)
