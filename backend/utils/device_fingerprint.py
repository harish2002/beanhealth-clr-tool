"""
BeanHealth CLR Tool — Device Fingerprinting
============================================

Parses the User-Agent header from the mobile browser to extract a human-readable
device model string.

Phase 1: Lightweight regex parsing — no external library dependency.
Phase 2 (future): Matched against a cloud calibration database to apply per-model
                  torch offset and bloom corrections.

Supported patterns:
  • iPhone iOS  (e.g. "iPhone; CPU iPhone OS 17_0")
  • iPad
  • Android with explicit model  (e.g. "Linux; Android 13; SM-G991B")
  • Android generic  (e.g. "Android 13")
  • Desktop fallback  (Windows / Mac)
  • Unknown
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ─── Compiled patterns (ordered: most-specific first) ───────────────────────

_IPHONE_RE = re.compile(
    r"iPhone(?:;\s*CPU iPhone OS\s*([\d_]+))?",
    re.IGNORECASE,
)
_IPAD_RE = re.compile(r"iPad", re.IGNORECASE)

# Android with a model string: "Linux; Android 12; Redmi Note 10 Pro)"
# or "Linux; Android 13; SM-G991B Build/TP1A"
_ANDROID_MODEL_RE = re.compile(
    r"Linux;\s*Android\s*[\d.]+;\s*([^)]+)\)",
    re.IGNORECASE,
)

_ANDROID_GENERIC_RE = re.compile(r"Android\s*([\d.]+)", re.IGNORECASE)


# ─── Public API ──────────────────────────────────────────────────────────────

def parse_device_model(user_agent: str) -> str:
    """
    Extract a human-readable device model string from a User-Agent header.

    Args:
        user_agent: Raw User-Agent header value (may be empty).

    Returns:
        A non-empty string such as:
          "iPhone iOS 17.0"
          "Samsung SM-G991B"
          "Redmi Note 10 Pro"
          "Android 13"
          "iPad"
          "Windows Desktop"
          "Mac Desktop"
          "Unknown"
    """
    if not user_agent or not user_agent.strip():
        return "Unknown"

    # ── iPhone ──────────────────────────────────────────────────
    m = _IPHONE_RE.search(user_agent)
    if m:
        os_ver = m.group(1)
        if os_ver:
            os_ver = os_ver.replace("_", ".")
            return f"iPhone iOS {os_ver}"
        return "iPhone"

    # ── iPad ────────────────────────────────────────────────────
    if _IPAD_RE.search(user_agent):
        return "iPad"

    # ── Android with explicit model string ──────────────────────
    m = _ANDROID_MODEL_RE.search(user_agent)
    if m:
        model = m.group(1).strip()
        # Remove trailing "Build/..." suffix that some UAs include
        model = re.sub(r"\s+Build.*", "", model, flags=re.IGNORECASE).strip()
        if model:
            logger.debug(f"[Fingerprint] Parsed Android model: '{model}'")
            return model

    # ── Android generic (no model) ──────────────────────────────
    m = _ANDROID_GENERIC_RE.search(user_agent)
    if m:
        return f"Android {m.group(1)}"

    # ── Desktop browsers ────────────────────────────────────────
    if "Windows" in user_agent:
        return "Windows Desktop"
    if "Macintosh" in user_agent or "Mac OS X" in user_agent:
        return "Mac Desktop"

    return "Unknown"
