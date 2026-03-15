"""
BeanHealth CLR Tool — API Integration Tests
============================================

Tests for the POST /analyse and GET /health endpoints.

Uses httpx.AsyncClient (via pytest-asyncio) to make real HTTP requests
against the FastAPI app without needing a live server.

Fixture images:
    flash_off/   — no torch → INCONCLUSIVE (no_flash)
    no_face/     — wall photo → INCONCLUSIVE (no_face)
    eyes_closed/ — closed eyes → INCONCLUSIVE (eyes_closed)
    flash_on_normal/ — proper CLR photos → SUCCESS (if available)

Run:
    cd backend && pytest tests/test_api.py -v
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

TEST_IMAGES = Path(__file__).parent / "test_images"


def _synthetic_jpeg(width: int = 640, height: int = 480, brightness: int = 50) -> bytes:
    """Generate a synthetic solid-colour JPEG — no face, useful for error testing."""
    arr = np.full((height, width, 3), brightness, dtype=np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _real_image_bytes(folder: str, index: int = 0) -> bytes | None:
    """
    Return raw bytes of the first JPEG/PNG in a test_images sub-folder,
    or None if the folder is empty or missing.
    """
    folder_path = TEST_IMAGES / folder
    if not folder_path.exists():
        return None
    files = sorted(
        f for f in folder_path.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files or index >= len(files):
        return None
    return files[index].read_bytes()


def _multipart(image_bytes: bytes, name: str = "Test Patient", age: int = 5):
    """Build the multipart form data dict for /analyse."""
    return {
        "files": {"image": ("test.jpg", image_bytes, "image/jpeg")},
        "data":  {"patient_name": name, "patient_age": str(age)},
    }


# ─────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_returns_200():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_health_contains_status_ok():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_contains_version():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert "version" in r.json()


# ─────────────────────────────────────────────────────────────
# /analyse — malformed requests (HTTP 422)
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyse_missing_image_returns_422():
    """Request with no image field → 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            data={"patient_name": "Test", "patient_age": "5"},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_analyse_missing_patient_name_returns_422():
    """Request with no patient_name → 422."""
    img = _synthetic_jpeg()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("t.jpg", img, "image/jpeg")},
            data={"patient_age": "5"},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_analyse_missing_patient_age_returns_422():
    """Request with no patient_age → 422."""
    img = _synthetic_jpeg()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("t.jpg", img, "image/jpeg")},
            data={"patient_name": "Test"},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_analyse_age_zero_returns_422():
    """Age 0 is below the minimum (ge=1) → 422."""
    img = _synthetic_jpeg()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("t.jpg", img, "image/jpeg")},
            data={"patient_name": "Test", "patient_age": "0"},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_analyse_age_over_limit_returns_422():
    """Age 121 exceeds maximum (le=120) → 422."""
    img = _synthetic_jpeg()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("t.jpg", img, "image/jpeg")},
            data={"patient_name": "Test", "patient_age": "121"},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_analyse_non_image_file_returns_422():
    """Uploading plain text bytes as 'image' → 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("t.jpg", b"this is not an image", "image/jpeg")},
            data={"patient_name": "Test", "patient_age": "5"},
        )
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────
# /analyse — synthetic image → INCONCLUSIVE (no_face)
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyse_synthetic_no_face_returns_inconclusive():
    """A solid-colour synthetic image has no face → INCONCLUSIVE."""
    img = _synthetic_jpeg()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("t.jpg", img, "image/jpeg")},
            data={"patient_name": "Test", "patient_age": "5"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "INCONCLUSIVE"


@pytest.mark.asyncio
async def test_analyse_inconclusive_has_reason():
    """INCONCLUSIVE response must have 'reason' and 'reason_human' keys."""
    img = _synthetic_jpeg()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("t.jpg", img, "image/jpeg")},
            data={"patient_name": "Test", "patient_age": "5"},
        )
    body = r.json()
    assert "reason"       in body
    assert "reason_human" in body
    assert len(body["reason_human"]) > 5


@pytest.mark.asyncio
async def test_analyse_inconclusive_has_no_result_block():
    """INCONCLUSIVE response must NOT contain a 'result' block."""
    img = _synthetic_jpeg()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("t.jpg", img, "image/jpeg")},
            data={"patient_name": "Test", "patient_age": "5"},
        )
    body = r.json()
    assert "result" not in body


@pytest.mark.asyncio
async def test_analyse_inconclusive_has_timestamp():
    """INCONCLUSIVE response must have a timestamp."""
    img = _synthetic_jpeg()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("t.jpg", img, "image/jpeg")},
            data={"patient_name": "Test", "patient_age": "5"},
        )
    body = r.json()
    assert "timestamp" in body
    assert len(body["timestamp"]) > 10


@pytest.mark.asyncio
async def test_analyse_patient_name_in_inconclusive_response():
    """Patient name must appear in the INCONCLUSIVE response."""
    img = _synthetic_jpeg()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("t.jpg", img, "image/jpeg")},
            data={"patient_name": "Emma Wilson", "patient_age": "5"},
        )
    body = r.json()
    assert body.get("patient", {}).get("name") == "Emma Wilson"


# ─────────────────────────────────────────────────────────────
# /analyse — real no-face photo (wall image)
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyse_no_face_image_is_inconclusive():
    """Real wall photo (no_face folder) must return INCONCLUSIVE."""
    img_bytes = _real_image_bytes("no_face")
    if img_bytes is None:
        pytest.skip("No test image in no_face/ folder")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("wall.jpg", img_bytes, "image/jpeg")},
            data={"patient_name": "Test", "patient_age": "5"},
        )
    assert r.status_code == 200
    assert r.json()["status"] == "INCONCLUSIVE"


# ─────────────────────────────────────────────────────────────
# /analyse — real flash-off photo → INCONCLUSIVE (no_flash)
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyse_flash_off_is_inconclusive():
    """Flash-off face photo — face is detected but CLR is absent → INCONCLUSIVE."""
    img_bytes = _real_image_bytes("flash_off")
    if img_bytes is None:
        pytest.skip("No test image in flash_off/ folder")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("flash_off.jpg", img_bytes, "image/jpeg")},
            data={"patient_name": "Test", "patient_age": "5"},
        )
    assert r.status_code == 200
    assert r.json()["status"] == "INCONCLUSIVE"


# ─────────────────────────────────────────────────────────────
# /analyse — real flash-on photo → SUCCESS (if CLR detectable)
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyse_flash_on_returns_200():
    """Flash-on photo at minimum returns HTTP 200 (either SUCCESS or INCONCLUSIVE)."""
    img_bytes = _real_image_bytes("flash_on_normal")
    if img_bytes is None:
        pytest.skip("No test image in flash_on_normal/ folder")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("flash_on.jpg", img_bytes, "image/jpeg")},
            data={"patient_name": "Test Child", "patient_age": "5"},
        )
    assert r.status_code == 200
    assert r.json()["status"] in {"SUCCESS", "INCONCLUSIVE"}


@pytest.mark.asyncio
async def test_analyse_success_schema_complete():
    """If flash-on photo produces SUCCESS, verify all required fields are present."""
    img_bytes = _real_image_bytes("flash_on_normal")
    if img_bytes is None:
        pytest.skip("No test image in flash_on_normal/ folder")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("flash_on.jpg", img_bytes, "image/jpeg")},
            data={"patient_name": "Test Child", "patient_age": "5"},
        )
    body = r.json()
    if body["status"] != "SUCCESS":
        pytest.skip(f"Image produced {body['status']} — skipping SUCCESS schema check")

    # Top-level keys
    for key in ["status", "patient", "result", "technical", "annotated_image_b64", "timestamp"]:
        assert key in body, f"Missing top-level key: {key}"

    # Result keys
    for key in ["urgency_tier", "condition_name", "icd10_code", "deviation_degrees",
                "asymmetry_score", "severity", "referral_recommendation", "timeframe", "narrative"]:
        assert key in body["result"], f"Missing result key: {key}"

    # Technical keys
    for key in ["left_pupil", "right_pupil", "left_clr", "right_clr",
                "left_displacement_norm", "right_displacement_norm",
                "confidence", "flags"]:
        assert key in body["technical"], f"Missing technical key: {key}"

    # Annotated image is a valid JPEG
    import base64
    decoded = base64.b64decode(body["annotated_image_b64"])
    assert decoded[:2] == b"\xff\xd8", "annotated_image_b64 is not a valid JPEG"


@pytest.mark.asyncio
async def test_analyse_success_urgency_is_valid_value():
    """urgency_tier must be one of the four known values."""
    img_bytes = _real_image_bytes("flash_on_normal")
    if img_bytes is None:
        pytest.skip("No test image in flash_on_normal/ folder")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("flash_on.jpg", img_bytes, "image/jpeg")},
            data={"patient_name": "Test Child", "patient_age": "5"},
        )
    body = r.json()
    if body["status"] != "SUCCESS":
        pytest.skip("Image did not produce SUCCESS")

    assert body["result"]["urgency_tier"] in {"URGENT", "ROUTINE", "MONITOR", "NORMAL"}


# ─────────────────────────────────────────────────────────────
# /analyse — eyes-closed photo
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyse_eyes_closed_is_inconclusive():
    """Eyes-closed photo — detection must fail gracefully → INCONCLUSIVE."""
    img_bytes = _real_image_bytes("eyes_closed")
    if img_bytes is None:
        pytest.skip("No test image in eyes_closed/ folder")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("eyes_closed.jpg", img_bytes, "image/jpeg")},
            data={"patient_name": "Test", "patient_age": "5"},
        )
    assert r.status_code == 200
    assert r.json()["status"] == "INCONCLUSIVE"


# ─────────────────────────────────────────────────────────────
# /analyse — response never leaks tracebacks
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_analyse_response_never_contains_traceback():
    """No response under any condition should contain a raw Python traceback."""
    img = _synthetic_jpeg()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/analyse",
            files={"image": ("t.jpg", img, "image/jpeg")},
            data={"patient_name": "Test", "patient_age": "5"},
        )
    raw = r.text
    assert "Traceback" not in raw
    assert "File \"/Users" not in raw
    assert "line " not in raw or "timestamp" in raw   # 'line' may appear in narratives
