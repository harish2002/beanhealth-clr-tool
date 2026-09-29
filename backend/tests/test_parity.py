"""
BeanHealth CLR Tool — Browser/Server Parity Test Suite
=======================================================

Unit tests for pipeline/module_parity.py, plus endpoint tests that the
optional client_measurement on /analyse-stream is compared and logged without
ever changing the result.

Run:
    cd backend && pytest tests/test_parity.py -v
"""

import json
import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app
from pipeline.module_parity import (
    MAX_CLIENT_PAYLOAD_CHARS,
    compare_with_client,
    parse_client_measurement,
)

SUCCESS_REFERENCE = Path(__file__).parent / "test_images" / "success_reference" / "clr_both_eyes.jpg"


def _client(**overrides):
    payload = {
        "schema": 1,
        "pipeline": "browser-live-2",
        "frames": 3,
        "reflex_frames": 3,
        "per_frame_asymmetry_deg": [10.0, 11.0, None],
        "asymmetry_deg": 10.5,
        "asymmetry_std_deg": 0.5,
        "tier": "MONITOR",
        "unstable": False,
        "method_b": {"h_asym": 0.02, "v_asym": 0.01, "verdict": "ALIGNED"},
    }
    payload.update(overrides)
    return payload


def _server(status="SUCCESS", tier="MONITOR", asym=11.0):
    agg = {
        "status": status,
        "asymmetry_avg_deg": asym,
        "alignment": {"available": True, "verdict": "ALIGNED", "h_asymmetry": 0.03, "v_asymmetry": 0.01},
    }
    if status == "SUCCESS":
        agg["result"] = {"urgency_tier": tier}
    return agg


def _frames(*asyms):
    return [
        {"status": "SUCCESS", "result": {"asymmetry_degrees": a}} if a is not None else {"status": "INCONCLUSIVE"}
        for a in asyms
    ]


# ─────────────────────────────────────────────────────────────
# parse_client_measurement
# ─────────────────────────────────────────────────────────────

class TestParse:

    def test_absent(self):
        assert parse_client_measurement("") == (None, None)

    def test_valid(self):
        data, err = parse_client_measurement(json.dumps(_client()))
        assert err is None and data["tier"] == "MONITOR"

    def test_invalid_json(self):
        data, err = parse_client_measurement("{not json")
        assert data is None and "not valid JSON" in err

    def test_not_an_object(self):
        data, err = parse_client_measurement("[1, 2, 3]")
        assert data is None and "JSON object" in err

    def test_too_large(self):
        data, err = parse_client_measurement(" " * (MAX_CLIENT_PAYLOAD_CHARS + 1))
        assert data is None and "too large" in err


# ─────────────────────────────────────────────────────────────
# compare_with_client
# ─────────────────────────────────────────────────────────────

class TestCompare:

    def test_asymmetry_delta_is_client_minus_server(self):
        p = compare_with_client(_client(asymmetry_deg=10.5), _server(asym=11.0), _frames(10.2, 11.3, None))
        assert p["asymmetry_deg"] == {"client": 10.5, "server": 11.0, "delta": -0.5}

    def test_tier_agreement(self):
        assert compare_with_client(_client(tier="MONITOR"), _server(tier="MONITOR"), [])["tier"]["agree"] is True
        assert compare_with_client(_client(tier="ROUTINE"), _server(tier="MONITOR"), [])["tier"]["agree"] is False

    def test_no_server_tier_when_inconclusive(self):
        p = compare_with_client(_client(), _server(status="INCONCLUSIVE"), [])
        assert p["tier"]["server"] is None and p["tier"]["agree"] is None
        assert p["server_status"] == "INCONCLUSIVE"

    def test_per_frame_only_where_both_measured(self):
        # client [10.0, 11.0, None] vs server [10.5, None, 12.0] → only frame 0 compares
        p = compare_with_client(_client(), _server(), _frames(10.5, None, 12.0))
        assert p["per_frame"]["delta"] == [-0.5, None, None]
        assert p["per_frame"]["frames_compared"] == 1
        assert p["per_frame"]["median_abs_delta"] == 0.5

    def test_method_b(self):
        p = compare_with_client(_client(), _server(), [])
        assert p["method_b"]["verdict"]["agree"] is True
        assert p["method_b"]["h_asym"]["delta"] == pytest.approx(-0.01)

    def test_hostile_values_are_ignored(self):
        """Wrong types, unknown tiers and non-finite numbers never crash and never compare."""
        p = compare_with_client(
            _client(asymmetry_deg="12", tier="SEVERE!", per_frame_asymmetry_deg="nope",
                    method_b=["x"], pipeline=123),
            _server(), _frames(10.0),
        )
        assert p["asymmetry_deg"]["client"] is None and p["asymmetry_deg"]["delta"] is None
        assert p["tier"]["client"] is None and p["tier"]["agree"] is None
        assert p["per_frame"]["client"] == [] and p["per_frame"]["frames_compared"] == 0
        assert p["method_b"]["verdict"]["client"] is None
        json.dumps(p)   # must stay serialisable

    def test_booleans_are_not_numbers(self):
        p = compare_with_client(_client(asymmetry_deg=True), _server(), [])
        assert p["asymmetry_deg"]["client"] is None


# ─────────────────────────────────────────────────────────────
# /analyse-stream end to end
# ─────────────────────────────────────────────────────────────

async def _stream(extra=None):
    img = SUCCESS_REFERENCE.read_bytes()
    files = [("images", (f"f{i}.jpg", img, "image/jpeg")) for i in range(3)]
    data = {"patient_name": "Parity Test", "patient_age": "8", **(extra or {})}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/analyse-stream", files=files, data=data)
    assert r.status_code == 200
    return r.json()


needs_reference = pytest.mark.skipif(
    not SUCCESS_REFERENCE.exists(),
    reason=f"Reference image not present: {SUCCESS_REFERENCE.name} (test images are git-ignored)",
)


@needs_reference
@pytest.mark.asyncio
async def test_stream_without_client_measurement_has_no_parity():
    body = await _stream()
    assert "parity" not in body


@needs_reference
@pytest.mark.asyncio
async def test_stream_returns_parity_and_leaves_result_unchanged():
    plain = await _stream()
    compared = await _stream({"client_measurement": json.dumps(_client(per_frame_asymmetry_deg=[11.0, 11.0, 11.0]))})

    parity = compared["parity"]
    assert parity["server_status"] == plain["status"] == "SUCCESS"
    assert parity["asymmetry_deg"]["server"] == plain["asymmetry_avg_deg"]
    assert len(parity["per_frame"]["server"]) == 3
    assert parity["per_frame"]["frames_compared"] >= 1

    # The comparison is observational only
    for key in ("status", "asymmetry_avg_deg", "result"):
        assert compared[key] == plain[key]


@needs_reference
@pytest.mark.asyncio
async def test_stream_with_garbage_client_measurement_still_succeeds():
    body = await _stream({"client_measurement": "{broken"})
    assert body["status"] == "SUCCESS"
    assert body["parity"] == {"error": "client_measurement is not valid JSON"}
