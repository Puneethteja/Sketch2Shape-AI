"""
tests/test_server.py
====================
Test suite for the unified FastAPI application server and SPA routing.
"""

import pytest
from starlette.testclient import TestClient
from server import app
import numpy as np
import base64
import io
from PIL import Image


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "Sketch2Shape AI"
    assert "version" in data


def test_system_diagnostics(client):
    response = client.get("/api/system/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert "os_name" in data
    assert "python_version" in data
    assert "pytorch_version" in data
    assert "cuda_available" in data


def test_samples_manifest(client):
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data


def test_spa_root_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Sketch2Shape" in response.text


def test_spa_static_assets(client):
    response = client.get("/css/styles.css")
    assert response.status_code == 200


def test_preprocess_api(client):
    # Create simple dummy base64 canvas image
    img = Image.new("RGB", (512, 512), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    response = client.post("/api/preprocess", json={"sketch": b64, "return_lineart": True})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["lineart_url"].startswith("data:image/png;base64,")
    assert data["preprocessed_url"].startswith("data:image/png;base64,")


def test_export_api_invalid_format(client):
    response = client.post("/api/export", json={"format": "invalid_format"})
    assert response.status_code == 400


def test_export_api_valid_format(client):
    response = client.post("/api/export", json={"format": "glb"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "download_url" in data
    assert data["format"] == "GLB"


def test_generate_api(client):
    # Create simple dummy circle sketch
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    # Draw simple dark box in center
    img[200:300, 200:300] = 0
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    response = client.post(
        "/api/generate",
        json={
            "sketch": b64,
            "prompt": "smooth porcelain ceramic vase",
            "controlnet_scale": 0.85,
            "inference_steps": 2,
            "seed": 42,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "glb_url" in data
    assert data["glb_url"].startswith("/outputs/")
    assert data["vertices"] > 0
    assert data["faces"] > 0
    assert isinstance(data["is_watertight"], bool)
    assert data["seed"] == 42
