"""
BeanHealth CLR Tool — Browser/Server Parity Check
==================================================

Responsibility:
    The live view in the browser runs its own port of the measurement pipeline.
    When it sends frames to /analyse-stream it can attach its own reading of
    those same frames (`client_measurement`). This module compares that reading
    with the server's and returns a structured report that main.py logs as a
    [PARITY] line and returns alongside the result.

    It is the evidence-gathering step before the browser's reading could ever
    become the reported one: every comparison is recorded, so agreement can be
    measured across many real captures instead of assumed.

Guarantees:
    • Never raises on bad client input — a malformed payload yields
      {"error": ...} and the analysis result is untouched.
    • Never changes the server's result, tier, or any clinical field.

Author: BeanHealth
"""

from __future__ import annotations

import json
import math
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

# A client payload is a few hundred bytes; anything far larger is not ours.
MAX_CLIENT_PAYLOAD_CHARS = 20_000

VALID_TIERS = {"URGENT", "ROUTINE", "MONITOR", "NORMAL"}
VALID_VERDICTS = {"ALIGNED", "BORDERLINE", "ASYMMETRIC"}


def parse_client_measurement(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse the optional `client_measurement` form field.

    Returns:
        (payload, None) on success, (None, None) when absent,
        or (None, error_message) when present but unusable.
    """
    if not raw:
        return None, None
    if len(raw) > MAX_CLIENT_PAYLOAD_CHARS:
        return None, "client_measurement is too large"
    try:
        data = json.loads(raw)
    except ValueError:
        return None, "client_measurement is not valid JSON"
    if not isinstance(data, dict):
        return None, "client_measurement must be a JSON object"
    return data, None


def _num(value: Any) -> Optional[float]:
    """A finite number, or None — booleans and non-numbers are rejected."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _pick(value: Any, allowed: set) -> Optional[str]:
    return value if isinstance(value, str) and value in allowed else None


def _delta(client: Optional[float], server: Optional[float], digits: int) -> Optional[float]:
    return round(client - server, digits) if client is not None and server is not None else None


def _agree(client: Optional[str], server: Optional[str]) -> Optional[bool]:
    return client == server if client is not None and server is not None else None


def compare_with_client(
    client: Dict[str, Any],
    aggregated: Dict[str, Any],
    frame_reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare the browser's reading with the server's for the same frames.

    Args:
        client:        Parsed client_measurement payload.
        aggregated:    The server's aggregated response for the request.
        frame_reports: The server's per-frame reports, in upload order.

    Returns:
        A JSON-serialisable parity report. Deltas are client − server.
    """
    server_status = aggregated.get("status")
    succeeded = server_status == "SUCCESS"

    # ── Asymmetry: the classification signal ──
    c_asym = _num(client.get("asymmetry_deg"))
    s_asym = _num(aggregated.get("asymmetry_avg_deg"))

    # ── Tier: only meaningful when the server produced one ──
    c_tier = _pick(client.get("tier"), VALID_TIERS)
    s_tier = _pick((aggregated.get("result") or {}).get("urgency_tier"), VALID_TIERS) if succeeded else None

    # ── Frame by frame: the tightest test of the two implementations ──
    raw_client_frames = client.get("per_frame_asymmetry_deg")
    client_frames = [_num(v) for v in raw_client_frames] if isinstance(raw_client_frames, list) else []
    server_frames = [
        _num((report.get("result") or {}).get("asymmetry_degrees")) if report.get("status") == "SUCCESS" else None
        for report in frame_reports
    ]
    frame_deltas = [_delta(c, s, 2) for c, s in zip(client_frames, server_frames)]
    abs_deltas = [abs(d) for d in frame_deltas if d is not None]

    # ── Method B ──
    client_b = client.get("method_b") if isinstance(client.get("method_b"), dict) else {}
    alignment = aggregated.get("alignment") or {}
    server_b_ok = bool(alignment.get("available"))
    c_verdict = _pick(client_b.get("verdict"), VALID_VERDICTS)
    s_verdict = _pick(alignment.get("verdict"), VALID_VERDICTS) if server_b_ok else None
    c_h = _num(client_b.get("h_asym"))
    c_v = _num(client_b.get("v_asym"))
    s_h = _num(alignment.get("h_asymmetry")) if server_b_ok else None
    s_v = _num(alignment.get("v_asymmetry")) if server_b_ok else None

    return {
        "client_pipeline": str(client.get("pipeline", "unknown"))[:40],
        "server_status": server_status,
        "asymmetry_deg": {"client": c_asym, "server": s_asym, "delta": _delta(c_asym, s_asym, 2)},
        "tier": {"client": c_tier, "server": s_tier, "agree": _agree(c_tier, s_tier)},
        "per_frame": {
            "client": client_frames,
            "server": server_frames,
            "delta": frame_deltas,
            "median_abs_delta": round(median(abs_deltas), 2) if abs_deltas else None,
            "frames_compared": len(abs_deltas),
        },
        "method_b": {
            "verdict": {"client": c_verdict, "server": s_verdict, "agree": _agree(c_verdict, s_verdict)},
            "h_asym": {"client": c_h, "server": s_h, "delta": _delta(c_h, s_h, 4)},
            "v_asym": {"client": c_v, "server": s_v, "delta": _delta(c_v, s_v, 4)},
        },
    }
