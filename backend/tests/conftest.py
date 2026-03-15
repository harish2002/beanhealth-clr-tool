"""
BeanHealth CLR Tool — Pytest Configuration
Shared fixtures for all test modules.
"""

import sys
from pathlib import Path

import pytest

# Ensure backend/ root is on the Python path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def blank_rgb_image():
    """128x128 solid grey RGB image — no face, no features."""
    import numpy as np
    return np.full((128, 128, 3), 128, dtype=np.uint8)


@pytest.fixture
def test_image_dir():
    """Path to test_images/ directory."""
    return Path(__file__).parent / "test_images"


@pytest.fixture
def test_output_dir():
    """Path to test_output/ directory (created if needed)."""
    out = Path(__file__).parent / "test_output"
    out.mkdir(exist_ok=True)
    return out
