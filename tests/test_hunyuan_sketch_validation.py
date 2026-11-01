import io
import asyncio

import pytest
from PIL import Image, ImageDraw

from api.main import _validate_glb
from utils.sketch_crop import prepare_sketch
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile


def test_screenshot_with_header_footer_is_not_guessed_automatically():
    image = Image.new("RGB", (700, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((300, 130, 470, 300), outline="black", width=8)
    draw.line((320, 300, 305, 420), fill="black", width=8)
    draw.line((445, 300, 460, 420), fill="black", width=8)
    draw.line((40, 30, 250, 30), fill="#222222", width=5)
    draw.rectangle((60, 455, 650, 480), fill="#dddddd")

    prepared, metadata = prepare_sketch(image)

    assert prepared.size == (512, 512)
    assert metadata["uncertain"] is True
    assert metadata["reason"]


def test_clean_sketch_is_cropped_and_centered():
    image = Image.new("RGB", (700, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((300, 130, 470, 300), outline="black", width=8)
    draw.line((320, 300, 305, 420), fill="black", width=8)
    draw.line((445, 300, 460, 420), fill="black", width=8)

    prepared, metadata = prepare_sketch(image)

    assert prepared.size == (512, 512)
    assert metadata["uncertain"] is False
    assert metadata["bbox"][1] > 80
    assert metadata["bbox"][3] < 460


def test_uncertain_crop_preserves_full_input_for_manual_review():
    image = Image.new("RGB", (512, 512), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 490, 490), outline="white", width=4)

    prepared, metadata = prepare_sketch(image)

    assert metadata["uncertain"] is True
    assert prepared.size == (512, 512)


def test_glb_validation_requires_geometry_and_embedded_pbr_texture():
    trimesh = pytest.importorskip("trimesh")
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    mesh = trimesh.creation.box()
    mesh.visual = TextureVisuals(material=PBRMaterial(baseColorTexture=Image.new("RGB", (4, 4), (180, 40, 30))))
    blob = mesh.export(file_type="glb")

    validated = _validate_glb(blob)

    assert validated["mesh_primitives"] > 0
    assert validated["pbr_textured"] is True
    with pytest.raises(ValueError, match="PBR materials"):
        _validate_glb(trimesh.creation.box().export(file_type="glb"))


def test_missing_tokenhub_key_returns_actionable_error_without_loading_local_pipeline(monkeypatch):
    import api.main as api

    monkeypatch.delenv("TENCENT_TOKENHUB_API_KEY", raising=False)
    monkeypatch.delenv("HUNYUAN3D_API_KEY", raising=False)
    monkeypatch.setenv("SKETCH2SHAPE_PROVIDER", "tokenhub")
    image = io.BytesIO()
    Image.new("RGB", (128, 128), "white").save(image, "PNG")
    image.seek(0)
    upload = UploadFile(file=image, filename="sketch.png", headers=Headers({"content-type": "image/png"}))

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api.generate(upload, "", .8, 30, -1, "glb", "Auto-Detect from Sketch", "draw"))

    assert exc_info.value.status_code == 503
    assert "configure_tencent.ps1" in exc_info.value.detail
