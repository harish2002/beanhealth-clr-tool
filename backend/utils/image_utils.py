"""
BeanHealth CLR Tool — Shared Image Utilities

Reusable helpers for image loading, conversion, resizing and annotation.
All pipeline modules import from here — never duplicate image ops.
"""

import base64
import logging
from pathlib import Path
from typing import Tuple, Optional, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Maximum input image dimension (longer side) — downsized before processing
MAX_IMAGE_DIMENSION = 2000


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes (JPEG/PNG) to an RGB numpy array.

    Args:
        image_bytes: Raw bytes from file upload or disk.

    Returns:
        np.ndarray of shape (H, W, 3) in RGB, dtype uint8.

    Raises:
        ValueError: If bytes cannot be decoded as a valid image.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Could not decode image bytes. Ensure the file is a valid JPEG or PNG.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def load_image_from_path(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from disk as an RGB numpy array.

    Args:
        path: Path to JPEG or PNG file.

    Returns:
        np.ndarray of shape (H, W, 3) in RGB, dtype uint8.
    """
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def downscale_if_needed(img: np.ndarray, max_dim: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """
    Downscale an image so its longest side does not exceed max_dim.
    Preserves aspect ratio. Returns original if already within limits.

    Args:
        img: RGB numpy array.
        max_dim: Maximum allowed dimension (pixels).

    Returns:
        Possibly downscaled RGB numpy array.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img

    scale = max_dim / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.debug(f"Downscaled image from ({w}x{h}) to ({new_w}x{new_h})")
    return resized


def crop_region(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    pad_h: float = 0.0,
    pad_v: float = 0.0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop a rectangular region from an image with optional relative padding.

    Args:
        img:   RGB numpy array (H, W, 3).
        x1,y1: Top-left corner of the region (before padding).
        x2,y2: Bottom-right corner of the region (before padding).
        pad_h: Horizontal padding as fraction of region width (each side).
        pad_v: Vertical padding as fraction of region height (each side).

    Returns:
        Tuple of:
          - Cropped RGB numpy array.
          - (x1_padded, y1_padded, x2_padded, y2_padded) actual pixel coords used.
    """
    h_img, w_img = img.shape[:2]

    width  = x2 - x1
    height = y2 - y1

    pad_x = int(width  * pad_h)
    pad_y = int(height * pad_v)

    x1_p = max(0, x1 - pad_x)
    y1_p = max(0, y1 - pad_y)
    x2_p = min(w_img, x2 + pad_x)
    y2_p = min(h_img, y2 + pad_y)

    crop = img[y1_p:y2_p, x1_p:x2_p].copy()
    return crop, (x1_p, y1_p, x2_p, y2_p)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to single-channel grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def encode_image_to_base64(img: np.ndarray, quality: int = 85) -> str:  # type: ignore
    """
    Encode an RGB numpy array to a base64-encoded JPEG string.

    Args:
        img:     RGB numpy array.
        quality: JPEG quality (0–100).

    Returns:
        Base64 string (no data URI prefix).
    """
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")


def save_debug_image(img: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save an RGB numpy array to disk as JPEG (for debug / test annotation review).

    Args:
        img:  RGB numpy array.
        path: Destination file path (will create parent dirs if needed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)
    logger.debug(f"Debug image saved: {path}")


def draw_dot(
    img: np.ndarray,
    x: float,
    y: float,
    color: Tuple[int, int, int],
    radius: int = 4,
    thickness: int = -1,
) -> np.ndarray:
    """Draw a filled dot on a copy of the image."""
    out = img.copy()
    cv2.circle(out, (int(x), int(y)), radius, color, thickness)
    return out


def draw_circle(
    img: np.ndarray,
    x: float,
    y: float,
    radius: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw an unfilled circle on a copy of the image."""
    out = img.copy()
    cv2.circle(out, (int(x), int(y)), int(radius), color, thickness)
    return out


def draw_crosshair(
    img: np.ndarray,
    x: float,
    y: float,
    color: Tuple[int, int, int],
    size: int = 10,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a crosshair (+ marker) on a copy of the image."""
    out = img.copy()
    xi, yi = int(x), int(y)
    cv2.line(out, (xi - size, yi), (xi + size, yi), color, thickness)
    cv2.line(out, (xi, yi - size), (xi, yi + size), color, thickness)
    return out


def image_shape_str(img: np.ndarray) -> str:
    """Return a readable shape string, e.g. '640x480x3'."""
    return "x".join(str(d) for d in img.shape)


def combine_crops_to_base64(left_crop: np.ndarray, right_crop: np.ndarray, quality: int = 85) -> str:
    """
    Concatenate left and right eye crops horizontally and encode as base64 JPEG.
    Used for creating side-by-side intermediate pipeline visualizations.
    
    Args:
        left_crop:  RGB numpy array of left eye.
        right_crop: RGB numpy array of right eye.
        quality:    JPEG quality (0-100).
        
    Returns:
        Base64 string.
    """
    # Ensure both crops have the same height before horizontal concatenation
    h_l, w_l = left_crop.shape[:2]
    h_r, w_r = right_crop.shape[:2]
    
    target_h = max(h_l, h_r)
    
    # Pad images if heights don't match
    if h_l != target_h:
        pad_top = (target_h - h_l) // 2
        pad_bottom = target_h - h_l - pad_top
        left_padded = cv2.copyMakeBorder(left_crop, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    else:
        left_padded = left_crop
        
    if h_r != target_h:
        pad_top = (target_h - h_r) // 2
        pad_bottom = target_h - h_r - pad_top
        right_padded = cv2.copyMakeBorder(right_crop, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    else:
        right_padded = right_crop
        
    # Add a thin white divider column between them (e.g., 4 pixels)
    divider = np.full((target_h, 4, 3), 255, dtype=np.uint8)
    
    # Horizontally concatenate: Left Eye | Divider | Right Eye
    combined = cv2.hconcat([left_padded, divider, right_padded])
    
    return encode_image_to_base64(combined, quality=quality)
