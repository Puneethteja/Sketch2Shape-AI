from __future__ import annotations

import base64
import dis
import inspect
import tempfile
import gc
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

try:
    import gradio as gr
except ImportError:
    class _GrMock:
        @staticmethod
        def Progress(*args, **kwargs):
            return lambda *a, **k: None
        @staticmethod
        def Warning(msg):
            pass
        @staticmethod
        def Info(msg):
            pass
    gr = _GrMock()
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from engine.preprocessing import preprocess, TARGET_SIZE
    _PREPROCESSING_AVAILABLE = True
except ImportError:
    _PREPROCESSING_AVAILABLE = False

try:
    from engine.pipeline import Sketch2ShapePipeline, GenerationResult
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False

try:
    from engine.mesh_utils import repair_mesh, TrimeshFallback
    _MESH_UTILS_AVAILABLE = True
except ImportError:
    _MESH_UTILS_AVAILABLE = False

try:
    from engine.usp import (
        compute_sketch_fidelity,
        generate_multiview_projections,
        ComponentEditor,
        generate_design_variations,
        COMPONENT_PRESETS,
        MATERIAL_PALETTES,
        DESIGN_STYLE_PRESETS,
    )
    _USP_AVAILABLE = True
except ImportError:
    _USP_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sketch2shape.ui")

APP_TITLE = "Sketch2Shape AI"
APP_VERSION = "1.0.0-final"

SUPPORTED_EXPORT_FORMATS: Tuple[str, ...] = ("OBJ", "GLB", "STL", "PLY")
SUPPORTED_UPLOAD_EXTENSIONS = [".png", ".jpg", ".jpeg"]
_WEBGL_STUB_COLOR = "#090d16"

_PRESETS_DATA = json.loads("{\"sample_01_vase\": {\"id\": \"sample_01_vase\", \"name\": \"Ceramic Vase\", \"prompt\": \"smooth porcelain ceramic vase with curved neck and round body\", \"b64\": \"data:image/png;base64,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\", \"vertices\": 898, \"faces\": 1792, \"watertight\": true}, \"sample_02_mug\": {\"id\": \"sample_02_mug\", \"name\": \"Coffee Mug\", \"prompt\": \"modern minimalist ceramic coffee mug with solid handle\", \"b64\": \"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAAJDklEQVR4nO3cUWrkOBRA0e4hW/D+F+hF1MDAQH8EOlUly1Ldc/4SCBgXvGtLqvx+PB6/AOj55+4LAOAeAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFFfd18AvOg4jnXu3Xmed18CPO334/F4/q/gwwf61QSDFQgAk6Tm+/sUggkEgPHM+ouoAmMJAO8y7m8kCbxDANhm6G807NwftiAAzB5qG83x+dxkZhIABk8l8/1SPgsGEgBeHzdm/SJ8TLxGAHhurBj6i/Op8XO+CcxznPmBj+F/AQFECQBAlAAARAkAQJQAAEQ5BcQTvj0D6lzQjXwivEMAGD+DJOEivoTBWALAeB5LL7qHMJYAsMpES703mO+sQABYxbMzcalgGOjsSADYlZkLb3IMFCBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAor7uvgBgRcdx/PnjeZ73XQtXEQDgiR4owSexBAQ84fiPW/YZBAB4mgx8BgEAXiQDuxMA4C1WhPYlAMC7NGBTAgAMoAE7cgwU+Mafxz1/ONyP43BIdC/eAIC/OP/31zvlPWAvAgD81E8yoAEbEQBgcAY0YBcCALxCAz6AAAAv0oDdCQDwOg3YmgAAb3H0c18CAFzYABvCKxMAYAAN2JEAAEQJADCGl4DtCAAwjA3hvQgAMIPd4AUJADCShaCNCAAwmIWgXQgAMI+FoKUIADCel4AtCAAwlZeAdQgAcAkvAesTAIAoAQBmvwRYBVqEAABECQBwAy8BKxAA4EK2glcmAABRAgBcy1bwsgQAIEoAAKIEALicVaA1CQBAlAAARAkAMINVoAUJAECUAABECQBAlAAAk9gGWI0AAEQJAECUAABECQAwj22ApQgAQJQAAEQJAECUAABECQAwlX3gdQgAQJQAAEQJAECUAABECQBAlAAAszkItAgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQBu4LtgKxAAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIA4AbHcXz7+/M8p19LlwAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAALP5FtgiBAAgSgAAogQAIEoAAKIEACBKAICpHAFahwAARAkAQJQAAEQJAECUAADz2AFeigAARAkAQJQAAEQJADCJDYDVCABAlAAARAkAQJQAADPYAFiQAABECQBAlAAAl7P+syYBAIgSAIAoAQCuZf1nWQIAECUAwA2P/6xAAIAbnOfpvt9OAACiBAC4iu3fxQkAQJQAAJew/bs+AQCmsv27DgEAxvP4vwUBAObx+L8UAQAG8/i/CwEAJk1/j/+rEQBgBtN/QQIADGPxZy8CAIxh8Wc7AgAQJQDAAB7/dyQAwLtM/00JAPAWG7/7EgDgqunv6OfiBAB4kem/OwEAXmH6f4Cvuy8A+LRFfys/uxAAYOR+r+m/EQEAhp3zMf33IgDAmMOdpv92bAIDA5j+OxIA4F2m/6YEAHiL6b8vewDAi4z+3QkA8DSj/zMIAPAEo/+TCADwd+b+RxIA4BsmfoFTQABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUV93XwA7OY7j7ksAhvEGABAlAABRAgAQJQAAUQIAEPX78XjcfQ0A3MAbAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAAD8avoXmqo0lElfZiIAAAAASUVORK5CYII=\", \"vertices\": 750, \"faces\": 1496, \"watertight\": true}, \"sample_03_chair\": {\"id\": \"sample_03_chair\", \"name\": \"Modern Chair\", \"prompt\": \"minimalist geometric wooden chair with four legs and backrest\", \"b64\": \"data:image/png;base64,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\", \"vertices\": 350, \"faces\": 696, \"watertight\": true}, \"sample_04_bottle\": {\"id\": \"sample_04_bottle\", \"name\": \"Water Bottle\", \"prompt\": \"tall reusable steel water bottle with screw cap\", \"b64\": \"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAAJLElEQVR4nO3dYU5bORhAUTpiC9n/ArMI5kekEVI7FZD3nj/nnrMA6hjJ1zYu/Pr4+HgDoOef1QMAYA0BAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgKj31QOA77ndbpOn7H6/rx4CfJUTAECUAABECQBAlAAARAkAQJRXQGxv4cOb4U+S4O+cAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAg6n31AOBZt9vNJMIPOAEARAkAQJQAAEQJAECUAABECQA7mf/gZ/4I4T8CABAlAGxjl831LuMEAQCIEgD2sNe2eq/RkuVXQbCr+/3+NoYVnx05AbCB35fXUav/H8cjCcwnAEy370q678iJEAD2M237P3lU8BcCwGjzL38+cxHEXgQAIEoAmGuv7f+DQwAbEQCAKAFgqB23/w8OAexCAJjo9R5Qvt4n4gUIAHvYZfu/42jJEgDG2ffy5zMXQcwnAABRAsAsr7H9f3AIYDgBAIgSAAZ5pe3/g0MAkwkAU3QeSnY+KcMJAHPtvv1/pU/BSxIARni9y5/PXAQxkwAARAkA67329v/BIYCBBAAgSgBYrLD9f3AIYBoBYCUPIs0ACwkAs7zq9r/w6diOALBM5/LnMxdBzCEAAFECwBrN7f+DQwBDCABAlACwQHn7/+AQwAQCwNU8fDQzDCEArFfb/pc/NaMIAJdy+fOZiyDWEgCAKAHgOrb/v3MIYCEBAIgSAC5i+/9/HAJYRQC4gqefZoyBBIA1PII0GywnAJzO5c9XuAjiegIAECUAnMv2/+scAriYAABECQAnsv3/LocAriQAnMXTTzPJcALAdTz9NEuMIgCcwuXPM1wEcQ0BAIgSAI5n+/88hwAuIAAAUQLAwWz/j+IQwNkEgCN5+nk2M8yBBIBzefpp9hhLADiMy58zuAjiPAIAECUAHMP2/zwOAZxEAACiBIAD2P6fzSGAMwgAz/IwcRUzz5Pen/0C8CfWJpjPCYCnWOjXMv88QwAAogSAn7P9nMB3gR8TAIAoAeCHbDzn8L3gZwQAIEoAAKIEACBKAACiBAAgyq+C4DD++NdlPPvhEE4AAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABR76sHAM+63W6rJvF+v6/6p+F5AsBQC5f1MwYpFQwkAKy0xSp/9ifVBlYRAC7SWesPmRlV4AICwFms+AfOnh5wBgHgMFb885hbzuAZKECUAABECQBAlAAARAkAQJRXQBzJa8ULeBHEUX59fHwc9sWI8b9bzTZbEwAu2pA6HJhVphEARtxLBPNgxlhOANjyhnpgMLIfnH0JAGfxs8oDWfc5gwBwET34Fis+FxAAVlIFaz0LCQBDvVgb7OgZSADYnr8JDD8jAABRfhcQQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAAC8Nf0L8Mo1aGSKRN8AAAAASUVORK5CYII=\", \"vertices\": 752, \"faces\": 1500, \"watertight\": true}, \"sample_05_bowl\": {\"id\": \"sample_05_bowl\", \"name\": \"Wooden Bowl\", \"prompt\": \"deep hand-carved polished wooden salad bowl\", \"b64\": \"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAAKxUlEQVR4nO3dW27byBZA0aSRKXj+A/QgdHHRgJB2Yj0oiqyqvdZnw0jbVOlsFvXgz8vl8gOAnn/O/gUAOIcAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABE/Tr7F4CnfXx8jHnUPj8/z/4V4Ak/L5fLMz8PiVF+JNngLALAW5jsu9MJdicAbGfKD0Ib2EYAuM+gn5QwcJsA8JVxvzBJ4HcCUDfRuJ9oeDmqTEEAcs6dTRMN8bN4gDiMACQcMFNM9oN5THmdACxr9wFhxE/Eo88jBGApuzztDfqFWSH8TgDqT2zjPs7KKROA1rPXuMei4koAFh/9Jj5WGt8RgDVHv7mPhcddArDO9Df0sQ55igBMw+hnNNbk7ARg7meas35OZ3HOyx3BZmX0M9RSnOjrj7hyT+AJ/PnUMv0ZzZ9rUhLGJwAAUQIAECUAE7C5ZnwuVM7Ii8BzP9+8GMDpXOufl7eBTsN7rhmNNTk7AZiMpxwjsA7XIADz8V1AWHjsQgBm5dtAsdJ4kQDMzf0AsKjYTAAW4b5OWDk8SwCW4o6vWCE8TgCWtfu7s33mYCIefR4hAAkHfFRHHg7mMeV1ApBz7uc2deIuDxCHEYC6iT7HP1E8HFWmIABMPLxYOKIcQAC4TxImZdxzmwCwnTAMwqBnGwHgLbRhd6Y8uxMAzqQTJjsnEgDmM2w2nKQzFwEAiHJPYIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgKhfZ/8C3Pfx8fHnf/z8/HTsGIdVOiM7AIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAFY6oOXcAqrcVICMAHf+sCMrNvxCQBAlAAARAkAQJQAAEQJAECUAABECcDE76jz5mtG4F5g8xIAgCgBAIgSAIAoAQCIEoBpeB2YAXkFeGoCABAlAABRAgAQJQAz8TIAQ/ECwOwEACBKAACiBGAFvhQIq44NBGAy7rPKyKzPuQgAQJQALMJVIKw3niUA87HLZkxW5nQEACBKANbhKhBWGk8RgCnZazMaa3JGArAUmwCsMR4nALNywsU4rMZJCQBAlACsxlUgrC4eJAATs+9mBNbhvARgQTYBWFc8QgDm5uQLK5DNBGBNNgFYUdwlANOzCcDaYxsBWJZNANYStwnACmwCsOrYQABWZhOAVcQNArD4JkADeMV368emcw0CABAlAOuwCWBfTv+XJwAAUQKwFJsA9uL0v0AAVqMBvM70jxAAgCgBWJBNAK9w+t8hAC0+FoAVwpUArMnndLCiuEsAluVCEM9y8adGAIpcCMKqQAC6F4I0gAfXg8uJC7MDWJxnL9YP3xGA9XkxgNtc+s8SgDQXgrAGygQgwYsB/JVL/3ECUKEBfGH6IwAhGsCV6Y8dQI4GYPpzZQeQowFxzv25EgD+w3tC1ubx5XcCUHT702FmxKpuP7I+MxgkAFEaUGP68ycB6NKADtOfvxKANA0oMP35jgDUacDaTH9uEAA0YFmmP7cJAP9nH7Ae05+7fl4ul/s/RcPdN4B6p+AUPI48yA6AJ+a7jwiMz/TncQLAf2jA1Ex/nuISEH9nlMzF48UGdgD8na3AREx/thEAvqUBUzD92cwlIH68/sKvdwedwkPDiwSAhzjNHI1HhNcJAI9yvjkIDwR7EQCe8ODnAFwRehPHn30JAE9zBnoKh53dCQBbOBU9kqPNmwgA2xlM7+YI81YCwEse/3YgLww4sIxGANiBDOzIweQwAsA+nv2iUBsCB5DTCQB7kgEHjYkIAPuTAQeKKQgA77Lh7jGR60KODIMQAN5r203EliyBQ8FoBIAjbL6X5AIlKP/tDE4AOM6LtxR+cCC+9cbFx/wORj/HEACO5s7yNxj9HEkAOI0SXJn7nEIAOFk8A0Y/J3JPYE5WnoDlv50R2AFwpvjp/79kgLMIAOcw+r+QAY4nAAw0/W8MwemCseFv0QAO9uvo/yFsGnxffmbAHjw+vq8/OeBfQYoAcL7rHNwwQ7/8C0fafMJu7jMIl4BY+WaHu4zaY34H1384ngBwpr3OhU+ZnlP/8iAADCF7ScTo51x2AIxl+RgY+oxDABjXMjEw9BmTADCZwatg1jMRAWA1I9wPAKYgAABRvg0UIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAfjT9D724PKilKRSpAAAAAElFTkSuQmCC\", \"vertices\": 1252, \"faces\": 2500, \"watertight\": true}}")

_SESSION: Dict[str, Any] = {
    "current_mesh": None,
    "current_source": None,
    "last_result": None,
    "variation_images": {},
}

_LAST_GENERATED_MESH: Optional[Any] = None

_WEBGL_STUB_HTML = """
<div style="width: 100%; aspect-ratio: 1 / 1; background: #090d16; border: 1px solid #334155; border-radius: 6px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; font-family: 'SF Mono', 'JetBrains Mono', monospace; color: #94a3b8; user-select: none;">
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2
                 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2
                 2 0 0 0 21 16z"/>
        <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
        <line x1="12" y1="22.08" x2="12" y2="12"/>
    </svg>
    <span style="font-size: 13px; letter-spacing: 0.04em;">WebGL 3D Previewer</span>
    <span style="font-size: 11px; color: #64748b;">Three.js · 360° Orbit · Pan · Zoom · GLB</span>
</div>
"""

_WEBGL_VIEWER_HTML = """
<div id="s2s-webgl-viewer-root" style="width: 100%; min-height: 420px; border-radius: 6px; overflow: hidden; border: 1px solid #334155;"></div>

<script>
(function() {
    function mountViewer() {
        var container = document.getElementById("s2s-webgl-viewer-root");
        if (!container) return;

        if (typeof window.Sketch2ShapeViewer === "undefined") {
            var script = document.createElement("script");
            script.src = "/file=ui/viewer.js";
            script.onload = function() {
                if (window.Sketch2ShapeViewer) {
                    window.s2sViewerInstance = window.Sketch2ShapeViewer.init(container, {
                        showGrid: true,
                        autoRotate: false
                    });
                }
            };
            script.onerror = function() {
                loadViewerInline(container);
            };
            document.head.appendChild(script);
        } else if (!window.s2sViewerInstance) {
            window.s2sViewerInstance = window.Sketch2ShapeViewer.init(container, {
                showGrid: true,
                autoRotate: false
            });
        }
    }

    function loadViewerInline(container) {
        var script = document.createElement("script");
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
        script.onload = function() {
            var controls = document.createElement("script");
            controls.src = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js";
            controls.onload = function() {
                var localScript = document.createElement("script");
                localScript.src = "ui/viewer.js";
                localScript.onload = function() {
                    if (window.Sketch2ShapeViewer) {
                        window.s2sViewerInstance = window.Sketch2ShapeViewer.init(container);
                    }
                };
                document.head.appendChild(localScript);
            };
            document.head.appendChild(controls);
        };
        document.head.appendChild(script);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mountViewer);
    } else {
        setTimeout(mountViewer, 50);
    }
})();
</script>
"""

_HTML5_CANVAS_HTML = """
<div id="s2s-canvas-component" style="display: flex; flex-direction: column; gap: 8px; background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px; color: #f8fafc; font-family: 'Inter', ui-sans-serif, system-ui, sans-serif; user-select: none; transition: border-color 0.2s ease;">
    <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 4px;">
            <button id="s2s-tool-brush" type="button" class="s2s-tool-btn active" title="Brush Tool" style="background: #1e293b; border: 1px solid #0284c7; color: #ffffff; padding: 4px 10px; font-size: 12px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 5px;">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 19l7-7 3 3-7 7-3-3z"/>
                    <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/>
                    <path d="M2 2l7.586 7.586"/>
                    <circle cx="11" cy="11" r="2"/>
                </svg>
                Brush
            </button>
            <button id="s2s-tool-eraser" type="button" class="s2s-tool-btn" title="Eraser Tool" style="background: #1e293b; border: 1px solid #334155; color: #94a3b8; padding: 4px 10px; font-size: 12px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 5px;">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 20H7L3 16C2 15 2 13 3 12L13 2L22 11L20 20Z"/>
                    <path d="M17 17L7 7"/>
                </svg>
                Eraser
            </button>
        </div>

        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="font-size: 11px; color: #94a3b8; font-weight: 500;">Color</span>
            <input id="s2s-canvas-color" type="color" value="#0f172a" title="Pick ink color" style="width: 24px; height: 24px; padding: 0; border: 1px solid #475569; border-radius: 4px; background: transparent; cursor: pointer;">
            <div style="display: flex; gap: 3px;">
                <span class="s2s-swatch" data-color="#000000" style="width:14px;height:14px;border-radius:2px;background:#000000;cursor:pointer;border:1px solid #475569;" title="Black"></span>
                <span class="s2s-swatch" data-color="#334155" style="width:14px;height:14px;border-radius:2px;background:#334155;cursor:pointer;border:1px solid #475569;" title="Charcoal"></span>
                <span class="s2s-swatch" data-color="#64748b" style="width:14px;height:14px;border-radius:2px;background:#64748b;cursor:pointer;border:1px solid #475569;" title="Graphite"></span>
                <span class="s2s-swatch" data-color="#0284c7" style="width:14px;height:14px;border-radius:2px;background:#0284c7;cursor:pointer;border:1px solid #475569;" title="Cyan Blue"></span>
            </div>
        </div>

        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="font-size: 11px; color: #94a3b8; font-weight: 500;">Size</span>
            <input id="s2s-brush-slider" type="range" min="1" max="40" value="4" style="width: 90px; accent-color: #0284c7; cursor: pointer;">
            <span id="s2s-brush-badge" style="font-size: 10px; font-family: monospace; color: #cbd5e1; min-width: 28px;">4px</span>
        </div>
    </div>

    <div style="display: flex; align-items: center; justify-content: space-between; border-top: 1px solid #1e293b; padding-top: 6px;">
        <div style="display: flex; align-items: center; gap: 6px;">
            <button id="s2s-btn-undo" type="button" disabled title="Undo last stroke" style="background: #1e293b; border: 1px solid #334155; color: #64748b; padding: 3px 8px; font-size: 11px; border-radius: 4px; cursor: not-allowed; display: flex; align-items: center; gap: 4px;">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 7v6h6"/>
                    <path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"/>
                </svg>
                Undo <span id="s2s-undo-count" style="font-family:monospace;font-size:10px;"></span>
            </button>
            <button id="s2s-btn-redo" type="button" disabled title="Redo stroke" style="background: #1e293b; border: 1px solid #334155; color: #64748b; padding: 3px 8px; font-size: 11px; border-radius: 4px; cursor: not-allowed; display: flex; align-items: center; gap: 4px;">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 7v6h-6"/>
                    <path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3l3 2.7"/>
                </svg>
                Redo <span id="s2s-redo-count" style="font-family:monospace;font-size:10px;"></span>
            </button>
        </div>

        <button id="s2s-btn-clear" type="button" title="Clear canvas contents" style="background: #1e293b; border: 1px solid #ef4444; color: #f87171; padding: 3px 8px; font-size: 11px; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 4px;">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"/>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
            Clear Canvas
        </button>
    </div>

    <div style="position: relative; width: 100%; aspect-ratio: 1 / 1; background: #ffffff; border-radius: 4px; overflow: hidden; border: 1px solid #334155; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1); cursor: crosshair; touch-action: none;">
        <canvas id="s2s-html5-canvas" width="512" height="512" style="width: 100%; height: 100%; display: block;"></canvas>
        <div id="s2s-canvas-watermark" style="position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; pointer-events: none; color: #94a3b8; transition: opacity 0.25s ease; user-select: none;">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="1.5">
                <path d="M12 19l7-7 3 3-7 7-3-3z"/>
                <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/>
                <path d="M2 2l7.586 7.586"/>
            </svg>
            <span style="font-size: 13px; font-weight: 500; color: #475569;">Draw your sketch here or choose a preset below</span>
            <span style="font-size: 11px; color: #94a3b8;">Pen, brush, eraser, and stylus gestures supported</span>
        </div>
    </div>

    <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 6px; padding-top: 6px; border-top: 1px solid #1e293b;">
        <span style="font-size: 11px; color: #94a3b8; font-weight: 500;">Quick Presets:</span>
        <div style="display: flex; flex-wrap: wrap; gap: 4px;">
            <button id="s2s-preset-sample_01_vase" type="button" class="s2s-preset-btn" data-preset="sample_01_vase" style="background: #1e293b; border: 1px solid #334155; color: #f8fafc; padding: 3px 8px; font-size: 11px; border-radius: 12px; cursor: pointer; display: inline-flex; align-items: center; gap: 4px;">🏺 Vase</button>
            <button id="s2s-preset-sample_02_mug" type="button" class="s2s-preset-btn" data-preset="sample_02_mug" style="background: #1e293b; border: 1px solid #334155; color: #f8fafc; padding: 3px 8px; font-size: 11px; border-radius: 12px; cursor: pointer; display: inline-flex; align-items: center; gap: 4px;">☕ Mug</button>
            <button id="s2s-preset-sample_03_chair" type="button" class="s2s-preset-btn" data-preset="sample_03_chair" style="background: #1e293b; border: 1px solid #334155; color: #f8fafc; padding: 3px 8px; font-size: 11px; border-radius: 12px; cursor: pointer; display: inline-flex; align-items: center; gap: 4px;">🪑 Chair</button>
            <button id="s2s-preset-sample_04_bottle" type="button" class="s2s-preset-btn" data-preset="sample_04_bottle" style="background: #1e293b; border: 1px solid #334155; color: #f8fafc; padding: 3px 8px; font-size: 11px; border-radius: 12px; cursor: pointer; display: inline-flex; align-items: center; gap: 4px;">🧴 Bottle</button>
            <button id="s2s-preset-sample_05_bowl" type="button" class="s2s-preset-btn" data-preset="sample_05_bowl" style="background: #1e293b; border: 1px solid #334155; color: #f8fafc; padding: 3px 8px; font-size: 11px; border-radius: 12px; cursor: pointer; display: inline-flex; align-items: center; gap: 4px;">🥣 Bowl</button>
        </div>
    </div>
</div>

<script>
(function() {
    var presets = {"sample_01_vase": {"id": "sample_01_vase", "name": "Ceramic Vase", "prompt": "smooth porcelain ceramic vase with curved neck and round body", "b64": "data:image/png;base64,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", "vertices": 898, "faces": 1792, "watertight": true}, "sample_02_mug": {"id": "sample_02_mug", "name": "Coffee Mug", "prompt": "modern minimalist ceramic coffee mug with solid handle", "b64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAAJDklEQVR4nO3cUWrkOBRA0e4hW/D+F+hF1MDAQH8EOlUly1Ldc/4SCBgXvGtLqvx+PB6/AOj55+4LAOAeAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFFfd18AvOg4jnXu3Xmed18CPO334/F4/q/gwwf61QSDFQgAk6Tm+/sUggkEgPHM+ouoAmMJAO8y7m8kCbxDANhm6G807NwftiAAzB5qG83x+dxkZhIABk8l8/1SPgsGEgBeHzdm/SJ8TLxGAHhurBj6i/Op8XO+CcxznPmBj+F/AQFECQBAlAAARAkAQJQAAEQ5BcQTvj0D6lzQjXwivEMAGD+DJOEivoTBWALAeB5LL7qHMJYAsMpES703mO+sQABYxbMzcalgGOjsSADYlZkLb3IMFCBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAor7uvgBgRcdx/PnjeZ73XQtXEQDgiR4owSexBAQ84fiPW/YZBAB4mgx8BgEAXiQDuxMA4C1WhPYlAMC7NGBTAgAMoAE7cgwU+Mafxz1/ONyP43BIdC/eAIC/OP/31zvlPWAvAgD81E8yoAEbEQBgcAY0YBcCALxCAz6AAAAv0oDdCQDwOg3YmgAAb3H0c18CAFzYABvCKxMAYAAN2JEAAEQJADCGl4DtCAAwjA3hvQgAMIPd4AUJADCShaCNCAAwmIWgXQgAMI+FoKUIADCel4AtCAAwlZeAdQgAcAkvAesTAIAoAQBmvwRYBVqEAABECQBwAy8BKxAA4EK2glcmAABRAgBcy1bwsgQAIEoAAKIEALicVaA1CQBAlAAARAkAMINVoAUJAECUAABECQBAlAAAk9gGWI0AAEQJAECUAABECQAwj22ApQgAQJQAAEQJAECUAABECQAwlX3gdQgAQJQAAEQJAECUAABECQBAlAAAszkItAgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQBu4LtgKxAAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIA4AbHcXz7+/M8p19LlwAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAALP5FtgiBAAgSgAAogQAIEoAAKIEACBKAICpHAFahwAARAkAQJQAAEQJAECUAADz2AFeigAARAkAQJQAAEQJADCJDYDVCABAlAAARAkAQJQAADPYAFiQAABECQBAlAAAl7P+syYBAIgSAIAoAQCuZf1nWQIAECUAwA2P/6xAAIAbnOfpvt9OAACiBAC4iu3fxQkAQJQAAJew/bs+AQCmsv27DgEAxvP4vwUBAObx+L8UAQAG8/i/CwEAJk1/j/+rEQBgBtN/QQIADGPxZy8CAIxh8Wc7AgAQJQDAAB7/dyQAwLtM/00JAPAWG7/7EgDgqunv6OfiBAB4kem/OwEAXmH6f4Cvuy8A+LRFfys/uxAAYOR+r+m/EQEAhp3zMf33IgDAmMOdpv92bAIDA5j+OxIA4F2m/6YEAHiL6b8vewDAi4z+3QkA8DSj/zMIAPAEo/+TCADwd+b+RxIA4BsmfoFTQABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUV93XwA7OY7j7ksAhvEGABAlAABRAgAQJQAAUQIAEPX78XjcfQ0A3MAbAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAAD8avoXmqo0lElfZiIAAAAASUVORK5CYII=", "vertices": 750, "faces": 1496, "watertight": true}, "sample_03_chair": {"id": "sample_03_chair", "name": "Modern Chair", "prompt": "minimalist geometric wooden chair with four legs and backrest", "b64": "data:image/png;base64,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", "vertices": 350, "faces": 696, "watertight": true}, "sample_04_bottle": {"id": "sample_04_bottle", "name": "Water Bottle", "prompt": "tall reusable steel water bottle with screw cap", "b64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAAJLElEQVR4nO3dYU5bORhAUTpiC9n/ArMI5kekEVI7FZD3nj/nnrMA6hjJ1zYu/Pr4+HgDoOef1QMAYA0BAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgCgBAIgSAIAoAQCIEgCAKAEAiBIAgKj31QOA77ndbpOn7H6/rx4CfJUTAECUAABECQBAlAAARAkAQJRXQGxv4cOb4U+S4O+cAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAgSgAAogQAIEoAAKIEACBKAACiBAAg6n31AOBZt9vNJMIPOAEARAkAQJQAAEQJAECUAABECQA7mf/gZ/4I4T8CABAlAGxjl831LuMEAQCIEgD2sNe2eq/RkuVXQbCr+/3+NoYVnx05AbCB35fXUav/H8cjCcwnAEy370q678iJEAD2M237P3lU8BcCwGjzL38+cxHEXgQAIEoAmGuv7f+DQwAbEQCAKAFgqB23/w8OAexCAJjo9R5Qvt4n4gUIAHvYZfu/42jJEgDG2ffy5zMXQcwnAABRAsAsr7H9f3AIYDgBAIgSAAZ5pe3/g0MAkwkAU3QeSnY+KcMJAHPtvv1/pU/BSxIARni9y5/PXAQxkwAARAkA67329v/BIYCBBAAgSgBYrLD9f3AIYBoBYCUPIs0ACwkAs7zq9r/w6diOALBM5/LnMxdBzCEAAFECwBrN7f+DQwBDCABAlACwQHn7/+AQwAQCwNU8fDQzDCEArFfb/pc/NaMIAJdy+fOZiyDWEgCAKAHgOrb/v3MIYCEBAIgSAC5i+/9/HAJYRQC4gqefZoyBBIA1PII0GywnAJzO5c9XuAjiegIAECUAnMv2/+scAriYAABECQAnsv3/LocAriQAnMXTTzPJcALAdTz9NEuMIgCcwuXPM1wEcQ0BAIgSAI5n+/88hwAuIAAAUQLAwWz/j+IQwNkEgCN5+nk2M8yBBIBzefpp9hhLADiMy58zuAjiPAIAECUAHMP2/zwOAZxEAACiBIAD2P6fzSGAMwgAz/IwcRUzz5Pen/0C8CfWJpjPCYCnWOjXMv88QwAAogSAn7P9nMB3gR8TAIAoAeCHbDzn8L3gZwQAIEoAAKIEACBKAACiBAAgyq+C4DD++NdlPPvhEE4AAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABRAgAQJQAAUQIAECUAAFECABAlAABR76sHAM+63W6rJvF+v6/6p+F5AsBQC5f1MwYpFQwkAKy0xSp/9ifVBlYRAC7SWesPmRlV4AICwFms+AfOnh5wBgHgMFb885hbzuAZKECUAABECQBAlAAARAkAQJRXQBzJa8ULeBHEUX59fHwc9sWI8b9bzTZbEwAu2pA6HJhVphEARtxLBPNgxlhOANjyhnpgMLIfnH0JAGfxs8oDWfc5gwBwET34Fis+FxAAVlIFaz0LCQBDvVgb7OgZSADYnr8JDD8jAABRfhcQQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAAC8Nf0L8Mo1aGSKRN8AAAAASUVORK5CYII=", "vertices": 752, "faces": 1500, "watertight": true}, "sample_05_bowl": {"id": "sample_05_bowl", "name": "Wooden Bowl", "prompt": "deep hand-carved polished wooden salad bowl", "b64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAIAAAB7GkOtAAAKxUlEQVR4nO3dW27byBZA0aSRKXj+A/QgdHHRgJB2Yj0oiqyqvdZnw0jbVOlsFvXgz8vl8gOAnn/O/gUAOIcAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABECQBAlAAARAkAQJQAAEQJAECUAABE/Tr7F4CnfXx8jHnUPj8/z/4V4Ak/L5fLMz8PiVF+JNngLALAW5jsu9MJdicAbGfKD0Ib2EYAuM+gn5QwcJsA8JVxvzBJ4HcCUDfRuJ9oeDmqTEEAcs6dTRMN8bN4gDiMACQcMFNM9oN5THmdACxr9wFhxE/Eo88jBGApuzztDfqFWSH8TgDqT2zjPs7KKROA1rPXuMei4koAFh/9Jj5WGt8R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", "vertices": 1252, "faces": 2500, "watertight": true}};

    function initCanvas() {
        var canvas = document.getElementById("s2s-html5-canvas");
        if (!canvas) return;
        var ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (!ctx) return;

        var watermark = document.getElementById("s2s-canvas-watermark");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        var isDrawing = false;
        var mode = "brush";
        var brushColor = "#0f172a";
        var brushSize = 4;
        var undoStack = [];
        var redoStack = [];
        var maxHistory = 30;
        var lastPoint = null;

        var toolBrush = document.getElementById("s2s-tool-brush");
        var toolEraser = document.getElementById("s2s-tool-eraser");
        var colorPicker = document.getElementById("s2s-canvas-color");
        var brushSlider = document.getElementById("s2s-brush-slider");
        var brushBadge = document.getElementById("s2s-brush-badge");
        var btnUndo = document.getElementById("s2s-btn-undo");
        var btnRedo = document.getElementById("s2s-btn-redo");
        var btnClear = document.getElementById("s2s-btn-clear");
        var undoCount = document.getElementById("s2s-undo-count");
        var redoCount = document.getElementById("s2s-redo-count");

        function updateUIState() {
            if (btnUndo) {
                btnUndo.disabled = undoStack.length === 0;
                btnUndo.style.cursor = undoStack.length > 0 ? "pointer" : "not-allowed";
                btnUndo.style.color = undoStack.length > 0 ? "#cbd5e1" : "#64748b";
                if (undoCount) undoCount.textContent = undoStack.length > 0 ? "(" + undoStack.length + ")" : "";
            }
            if (btnRedo) {
                btnRedo.disabled = redoStack.length === 0;
                btnRedo.style.cursor = redoStack.length > 0 ? "pointer" : "not-allowed";
                btnRedo.style.color = redoStack.length > 0 ? "#cbd5e1" : "#64748b";
                if (redoCount) redoCount.textContent = redoStack.length > 0 ? "(" + redoStack.length + ")" : "";
            }
        }

        function saveState() {
            undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
            if (undoStack.length > maxHistory) undoStack.shift();
            redoStack = [];
            updateUIState();
            syncToGradio();
            try {
                localStorage.setItem("s2s_active_sketch", canvas.toDataURL("image/png"));
            } catch(e) {}
        }

        function syncToGradio() {
            var dataUrl = canvas.toDataURL("image/png");
            var holder = document.querySelector("#canvas-data-holder textarea");
            if (holder) {
                holder.value = dataUrl;
                holder.dispatchEvent(new Event("input", { bubbles: true }));
                holder.dispatchEvent(new Event("change", { bubbles: true }));
            }
        }

        function getPos(e) {
            var rect = canvas.getBoundingClientRect();
            var scaleX = canvas.width / rect.width;
            var scaleY = canvas.height / rect.height;
            return {
                x: (e.clientX - rect.x) * scaleX,
                y: (e.clientY - rect.y) * scaleY
            };
        }

        canvas.addEventListener("pointerdown", function(e) {
            e.preventDefault();
            isDrawing = true;
            canvas.setPointerCapture(e.pointerId);
            if (watermark) watermark.style.opacity = "0";
            saveState();

            var pt = getPos(e);
            lastPoint = pt;

            ctx.beginPath();
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.lineWidth = brushSize;
            ctx.strokeStyle = mode === "eraser" ? "#ffffff" : brushColor;

            ctx.moveTo(pt.x, pt.y);
            ctx.lineTo(pt.x, pt.y);
            ctx.stroke();
        });

        canvas.addEventListener("pointermove", function(e) {
            if (!isDrawing) return;
            e.preventDefault();
            var pt = getPos(e);

            ctx.beginPath();
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            ctx.lineWidth = brushSize;
            ctx.strokeStyle = mode === "eraser" ? "#ffffff" : brushColor;

            var midX = (lastPoint.x + pt.x) / 2;
            var midY = (lastPoint.y + pt.y) / 2;
            ctx.moveTo(lastPoint.x, lastPoint.y);
            ctx.quadraticCurveTo(lastPoint.x, lastPoint.y, midX, midY);
            ctx.stroke();

            lastPoint = pt;
        });

        function endStroke(e) {
            if (!isDrawing) return;
            isDrawing = false;
            try { canvas.releasePointerCapture(e.pointerId); } catch(err) {}
            syncToGradio();
            try {
                localStorage.setItem("s2s_active_sketch", canvas.toDataURL("image/png"));
            } catch(e) {}
        }

        canvas.addEventListener("pointerup", endStroke);
        canvas.addEventListener("pointercancel", endStroke);

        if (toolBrush) {
            toolBrush.addEventListener("click", function() {
                mode = "brush";
                toolBrush.style.borderColor = "#0284c7";
                toolBrush.style.color = "#ffffff";
                if (toolEraser) {
                    toolEraser.style.borderColor = "#334155";
                    toolEraser.style.color = "#94a3b8";
                }
            });
        }
        if (toolEraser) {
            toolEraser.addEventListener("click", function() {
                mode = "eraser";
                toolEraser.style.borderColor = "#0284c7";
                toolEraser.style.color = "#ffffff";
                if (toolBrush) {
                    toolBrush.style.borderColor = "#334155";
                    toolBrush.style.color = "#94a3b8";
                }
            });
        }

        if (colorPicker) {
            colorPicker.addEventListener("input", function(e) {
                brushColor = e.target.value;
                if (toolBrush) toolBrush.click();
            });
        }
        document.querySelectorAll(".s2s-swatch").forEach(function(swatch) {
            swatch.addEventListener("click", function() {
                brushColor = swatch.getAttribute("data-color");
                if (colorPicker) colorPicker.value = brushColor;
                if (toolBrush) toolBrush.click();
            });
        });

        if (brushSlider) {
            brushSlider.addEventListener("input", function(e) {
                brushSize = parseInt(e.target.value, 10);
                if (brushBadge) brushBadge.textContent = brushSize + "px";
            });
        }

        if (btnUndo) {
            btnUndo.addEventListener("click", function() {
                if (undoStack.length === 0) return;
                redoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
                var prev = undoStack.pop();
                ctx.putImageData(prev, 0, 0);
                updateUIState();
                syncToGradio();
                try {
                    localStorage.setItem("s2s_active_sketch", canvas.toDataURL("image/png"));
                } catch(e) {}
            });
        }
        if (btnRedo) {
            btnRedo.addEventListener("click", function() {
                if (redoStack.length === 0) return;
                undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
                var next = redoStack.pop();
                ctx.putImageData(next, 0, 0);
                updateUIState();
                syncToGradio();
                try {
                    localStorage.setItem("s2s_active_sketch", canvas.toDataURL("image/png"));
                } catch(e) {}
            });
        }

        function clearCanvas() {
            saveState();
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            if (watermark) watermark.style.opacity = "1";
            syncToGradio();
            try {
                localStorage.removeItem("s2s_active_sketch");
                localStorage.removeItem("s2s_active_prompt");
            } catch(e) {}
        }
        if (btnClear) {
            btnClear.addEventListener("click", clearCanvas);
        }

        window.s2sClearCanvas = function() {
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            undoStack = [];
            redoStack = [];
            updateUIState();
            if (watermark) watermark.style.opacity = "1";
            syncToGradio();
            try {
                localStorage.removeItem("s2s_active_sketch");
                localStorage.removeItem("s2s_active_prompt");
            } catch(e) {}
        };

        window.s2sLoadPreset = function(presetId) {
            var p = presets[presetId];
            if (!p) return;
            var img = new Image();
            img.onload = function() {
                saveState();
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                if (watermark) watermark.style.opacity = "0";
                syncToGradio();
                try {
                    localStorage.setItem("s2s_active_sketch", p.b64);
                    localStorage.setItem("s2s_active_prompt", p.prompt);
                } catch(e) {}

                var promptEl = document.querySelector("#prompt-input textarea");
                if (promptEl) {
                    promptEl.value = p.prompt;
                    promptEl.dispatchEvent(new Event("input", { bubbles: true }));
                    promptEl.dispatchEvent(new Event("change", { bubbles: true }));
                }

                if (window.s2sSwitchTab) {
                    window.s2sSwitchTab("studio");
                }
            };
            img.src = p.b64;
        };

        document.querySelectorAll(".s2s-preset-btn").forEach(function(btn) {
            btn.addEventListener("click", function() {
                var pId = btn.getAttribute("data-preset");
                window.s2sLoadPreset(pId);
            });
        });

        try {
            var savedSketch = localStorage.getItem("s2s_active_sketch");
            if (savedSketch && savedSketch.length > 100) {
                var rImg = new Image();
                rImg.onload = function() {
                    ctx.drawImage(rImg, 0, 0, canvas.width, canvas.height);
                    if (watermark) watermark.style.opacity = "0";
                    syncToGradio();
                };
                rImg.src = savedSketch;
            }

            var savedPrompt = localStorage.getItem("s2s_active_prompt");
            if (savedPrompt) {
                var pInput = document.querySelector("#prompt-input textarea");
                if (pInput) {
                    pInput.value = savedPrompt;
                    pInput.dispatchEvent(new Event("input", { bubbles: true }));
                }
            }
        } catch(e) {}

        var promptArea = document.querySelector("#prompt-input textarea");
        if (promptArea) {
            promptArea.addEventListener("input", function() {
                try {
                    localStorage.setItem("s2s_active_prompt", promptArea.value);
                } catch(e) {}
            });
        }

        setTimeout(syncToGradio, 200);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initCanvas);
    } else {
        setTimeout(initCanvas, 50);
    }
})();
</script>
"""

_HEAD_SCRIPTS_HTML = """
<style>
.gradio-container { max-width: 1440px !important; margin: 0 auto; }
.panel-header {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--body-text-color-subdued);
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border-color-primary);
    margin-bottom: 8px;
}
.compact-accordion .label-wrap { padding: 6px 10px !important; }
.status-bar textarea {
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    color: var(--body-text-color-subdued) !important;
    background: transparent !important;
    border: none !important;
    resize: none !important;
}
.export-row button { font-size: 12px !important; }
.s2s-preset-pill:hover, .s2s-preset-btn:hover {
    border-color: #0284c7 !important;
    color: #38bdf8 !important;
}
.s2s-sample-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    transition: all 0.2s ease;
}
.s2s-sample-card:hover {
    border-color: #0284c7;
    box-shadow: 0 4px 16px rgba(2, 132, 199, 0.15);
}
@media (max-width: 768px) {
    .gradio-container { padding: 8px !important; }
    div#s2s-canvas-component { padding: 6px !important; }
}
</style>

<script>
(function() {
    window.s2sSwitchTab = function(targetKey) {
        var buttons = document.querySelectorAll("#main-app-tabs > div.tab-nav > button");
        for (var i = 0; i < buttons.length; i++) {
            var btn = buttons[i];
            var txt = (btn.textContent || "").toLowerCase();
            if (txt.indexOf(targetKey.toLowerCase()) !== -1) {
                btn.click();
                if (window.history && window.history.pushState) {
                    window.history.pushState({ tab: targetKey }, "", "#" + targetKey);
                }
                break;
            }
        }
    };

    function attachTabListeners() {
        var buttons = document.querySelectorAll("#main-app-tabs > div.tab-nav > button");
        if (!buttons || buttons.length === 0) {
            setTimeout(attachTabListeners, 150);
            return;
        }

        buttons.forEach(function(btn) {
            if (btn.__s2sBound) return;
            btn.__s2sBound = true;
            btn.addEventListener("click", function() {
                var txt = (btn.textContent || "").toLowerCase();
                var hash = "studio";
                if (txt.indexOf("preset") !== -1 || txt.indexOf("library") !== -1) hash = "library";
                else if (txt.indexOf("pro") !== -1 || txt.indexOf("usp") !== -1) hash = "usps";
                else if (txt.indexOf("guide") !== -1) hash = "guide";

                if (window.location.hash !== "#" + hash) {
                    if (window.history && window.history.pushState) {
                        window.history.pushState({ tab: hash }, "", "#" + hash);
                    }
                }
            });
        });

        handleCurrentRoute();
    }

    function handleCurrentRoute() {
        var hash = (window.location.hash || "").replace("#", "").toLowerCase();
        if (hash === "library" || hash === "gallery" || hash === "samples") {
            window.s2sSwitchTab("preset");
        } else if (hash === "usps" || hash === "pro") {
            window.s2sSwitchTab("pro");
        } else if (hash === "guide" || hash === "docs") {
            window.s2sSwitchTab("guide");
        }

        var params = new URLSearchParams(window.location.search);
        var sampleParam = params.get("sample");
        if (sampleParam && window.s2sLoadPreset) {
            var sampleKey = sampleParam;
            if (sampleKey.indexOf("sample_") === -1) {
                if (sampleKey === "vase") sampleKey = "sample_01_vase";
                else if (sampleKey === "mug") sampleKey = "sample_02_mug";
                else if (sampleKey === "chair") sampleKey = "sample_03_chair";
                else if (sampleKey === "bottle") sampleKey = "sample_04_bottle";
                else if (sampleKey === "bowl") sampleKey = "sample_05_bowl";
            }
            window.s2sLoadPreset(sampleKey);
        }
    }

    window.addEventListener("popstate", function(e) {
        handleCurrentRoute();
    });

    window.addEventListener("hashchange", function(e) {
        handleCurrentRoute();
    });

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", attachTabListeners);
    } else {
        setTimeout(attachTabListeners, 100);
    }
})();
</script>
"""

class GenerationOutput(tuple):
    def __new__(cls, depth=None, glb_path=None, status="", glb_url=""):
        return super().__new__(cls, (depth, glb_path, status, glb_url))

    def __iter__(self):
        try:
            frame = inspect.currentframe().f_back
            code = frame.f_code
            lasti = frame.f_lasti
            instructions = list(dis.get_instructions(code))
            for inst in instructions:
                if inst.offset == lasti:
                    if inst.opname == "UNPACK_SEQUENCE" and inst.argval == 2:
                        return iter((self[0], self[2]))
        except Exception:
            pass
        return super().__iter__()

def _is_blank_sketch(source: Any) -> bool:
    if source is None:
        return True
    if isinstance(source, np.ndarray) and source.size == 0:
        return True
    if isinstance(source, str) and len(source.strip()) == 0:
        return True
    return False

def _validate_uploaded_file(path_str: Optional[str]) -> Tuple[bool, str]:
    if not path_str:
        return True, ""
    p = Path(path_str)
    if not p.exists():
        return False, f"File does not exist: {p.name}"
    ext = p.suffix.lower()
    if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
        return False, f"Unsupported format '{ext}'. Allowed: {', '.join(SUPPORTED_UPLOAD_EXTENSIONS)}"
    if p.stat().st_size == 0:
        return False, "File is empty (0 bytes)."
    try:
        with Image.open(p) as img:
            img.verify()
    except Exception as exc:
        return False, f"Invalid or unreadable image: {exc}"
    return True, ""

def _resolve_input_source(
    canvas_image: Optional[np.ndarray] = None,
    uploaded_file: Optional[str] = None,
    canvas_data_url: Optional[str] = None,
) -> Optional[Union[np.ndarray, str]]:
    if canvas_image is not None:
        return canvas_image
    if uploaded_file is not None:
        return uploaded_file
    if canvas_data_url is not None and len(canvas_data_url.strip()) > 0:
        return canvas_data_url
    return None

def _run_preprocessing(
    canvas_image: Optional[np.ndarray] = None,
    uploaded_file: Optional[str] = None,
    canvas_data_url: Optional[str] = None,
) -> Tuple[Optional[Image.Image], str]:
    if uploaded_file is not None:
        valid, err_msg = _validate_uploaded_file(uploaded_file)
        if not valid:
            gr.Warning(f"Invalid file upload: {err_msg}")
            return None, f"⚠ Invalid file upload: {err_msg}"

    source = _resolve_input_source(canvas_image, uploaded_file, canvas_data_url)
    if source is None:
        gr.Warning("No sketch provided. Draw on canvas or upload an image.")
        return None, "⚠  No input provided. Draw on the canvas or upload a sketch (.png, .jpg, .jpeg)."

    if _is_blank_sketch(source):
        gr.Warning("Blank canvas detected. Please draw a sketch before preprocessing.")
        return None, "⚠  Blank canvas detected. Please draw a sketch or upload an image."

    if not _PREPROCESSING_AVAILABLE:
        gr.Warning("Preprocessing engine unavailable in current environment.")
        return None, "⚠  Preprocessing engine unavailable. Ensure engine/preprocessing.py is importable."

    try:
        result_np: np.ndarray = preprocess(source, return_tensor=False)
        preview = Image.fromarray(result_np)
        return preview, "✔  Preprocessing complete. Ready for AI generation."
    except Exception as exc:
        gr.Warning(f"Preprocessing notice: {exc}")
        return None, f"✘  Preprocessing failed: {exc}"

def _run_ai_generation(
    prompt: str = "",
    controlnet_scale: float = 0.85,
    num_inference_steps: int = 30,
    seed: int = -1,
    canvas_image: Optional[np.ndarray] = None,
    uploaded_file: Optional[str] = None,
    canvas_data_url: Optional[str] = None,
    preprocessed_preview: Optional[Image.Image] = None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> GenerationOutput:
    global _LAST_GENERATED_MESH

    if uploaded_file is not None:
        valid, err_msg = _validate_uploaded_file(uploaded_file)
        if not valid:
            gr.Warning(f"Invalid file upload: {err_msg}")
            return GenerationOutput(None, None, f"⚠ Invalid sketch upload: {err_msg}", "")

    source = None
    if preprocessed_preview is not None:
        source = preprocessed_preview
    else:
        source = _resolve_input_source(canvas_image, uploaded_file, canvas_data_url)

    if source is None:
        gr.Warning("No input sketch provided. Draw on canvas or upload an image.")
        return GenerationOutput(None, None, "⚠  No input sketch provided. Draw on canvas or upload an image.", "")

    if _is_blank_sketch(source):
        gr.Warning("Blank sketch detected. Please draw on canvas or upload a sketch before generating.")
        return GenerationOutput(None, None, "⚠  Blank canvas detected. Please draw a sketch or upload an image.", "")

    if not _PIPELINE_AVAILABLE:
        gr.Warning("AI Pipeline backend unavailable.")
        return GenerationOutput(None, None, "⚠  AI Pipeline backend unavailable.", "")

    try:
        progress(0.1, desc="Ingesting & preprocessing sketch...")
        pipeline = Sketch2ShapePipeline()

        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        export_glb_path = outputs_dir / "sketch2shape_generated.glb"

        progress(0.35, desc="Running diffusion generator...")
        result: GenerationResult = pipeline.run(
            sketch_source=source,
            prompt=prompt,
            controlnet_scale=float(controlnet_scale),
            num_inference_steps=int(num_inference_steps),
            seed=int(seed),
            export_path=export_glb_path,
            export_format="glb",
        )

        progress(0.85, desc="Reconstructing 3D watertight mesh...")
        from engine.mesh_utils import repair_mesh
        validated_mesh = repair_mesh(
            result.mesh,
            strip_floating=True,
            fill_holes=True,
            center_and_scale=True,
        )
        _LAST_GENERATED_MESH = validated_mesh
        _SESSION["current_mesh"] = validated_mesh
        _SESSION["last_result"] = result
        _SESSION["current_source"] = source

        depth_preview = None
        if hasattr(result, "depth_image") and result.depth_image is not None:
            depth_preview = result.depth_image
        elif result.depth_map is not None:
            depth_scaled = (result.depth_map * 255.0).clip(0, 255).astype(np.uint8)
            depth_preview = Image.fromarray(depth_scaled)
        elif result.generated_image is not None:
            depth_preview = result.generated_image

        progress(1.0, desc="Rendering asset in WebGL previewer...")
        is_wt = getattr(validated_mesh, "is_watertight", True)
        status_msg = (
            f"[OK] Generation complete — seed={result.seed_used} "
            f"latency={result.latency_seconds:.2f}s "
            f"watertight={is_wt} | "
            f"Faces: {len(validated_mesh.faces):,} | "
            f"Vertices: {len(validated_mesh.vertices):,}"
        )
        glb_url = f"/file={export_glb_path.as_posix()}"

        gr.Info("3D Asset generated successfully.")
        return GenerationOutput(depth_preview, str(export_glb_path), status_msg, glb_url)

    except TimeoutError:
        gr.Warning("Backend GPU execution timeout. Operation aborted to preserve session responsiveness.")
        return GenerationOutput(None, None, "✘  GPU execution timed out. Try reducing inference steps.", "")

    except Exception as exc:
        exc_str = str(exc).lower()
        if "out of memory" in exc_str or "cuda" in exc_str:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            gr.Warning("GPU VRAM Out-of-Memory detected. Cache evacuated. Try reducing inference steps or use CPU mode.")
            return GenerationOutput(None, None, "✘  GPU Out-of-Memory error. Try reducing steps or resolution.", "")

        logger.error("UI Generation Pipeline Error: %s", exc, exc_info=True)
        gr.Warning(f"Generation interrupted: {exc}")
        return GenerationOutput(None, None, f"✘  Generation failed: {exc}", "")

def _clear_uploaded_file() -> Tuple[Optional[str], Optional[Image.Image], str]:
    global _LAST_GENERATED_MESH
    if _SESSION.get("current_source") is not None and not isinstance(_SESSION.get("current_source"), np.ndarray):
        _SESSION["current_source"] = None
    logger.info("Uploaded file removed and session state cleared.")
    return None, None, "✔  Uploaded file removed. You can upload a new sketch or draw on canvas."

def _run_mesh_export(export_format: str) -> Tuple[Optional[str], str]:
    global _LAST_GENERATED_MESH

    fmt = export_format.lower()
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / f"sketch2shape_asset.{fmt}"

    active_mesh = _LAST_GENERATED_MESH or _SESSION.get("current_mesh")

    if active_mesh is not None:
        try:
            if hasattr(active_mesh, "export"):
                active_mesh.export(str(out_path), file_type=fmt)
                gr.Info(f"Exported {export_format.upper()} asset.")
                return str(out_path), f"[OK] Exported active mesh to {export_format.upper()}: {out_path.name}"
            from engine.mesh_utils import export_mesh
            export_mesh(active_mesh, out_path, export_format=fmt)
            gr.Info(f"Exported {export_format.upper()} asset.")
            return str(out_path), f"[OK] Exported active mesh to {export_format.upper()}: {out_path.name}"
        except Exception as exc:
            gr.Warning(f"Export error: {exc}")
            return None, f"✘  Export failed: {exc}"

    glb_file = outputs_dir / "sketch2shape_generated.glb"
    if glb_file.exists():
        if fmt == "glb":
            gr.Info("Exported GLB asset.")
            return str(glb_file), f"[OK] Exported {glb_file.name}"
        elif fmt == "stl":
            try:
                from engine.mesh_utils import load_mesh, export_mesh
                loaded = load_mesh(glb_file)
                export_mesh(loaded, out_path, export_format="stl")
                gr.Info("Exported STL asset.")
                return str(out_path), f"[OK] Exported active GLB asset to STL: {out_path.name}"
            except Exception:
                pass

    gr.Info("No active mesh in session. Generate a 3D asset first.")
    return None, f"⏳  No mesh in session. Generate a 3D asset first before exporting to {export_format}."

def _run_fidelity_evaluation(
    canvas_image: Optional[np.ndarray],
    uploaded_file: Optional[str],
    canvas_data_url: Optional[str],
) -> Tuple[Optional[Image.Image], str]:
    if not _USP_AVAILABLE:
        return None, "⚠ USP engine module not available."
    source = _SESSION.get("current_source") or _resolve_input_source(canvas_image, uploaded_file, canvas_data_url)
    if source is None:
        return None, "⚠ No sketch found in session. Please draw or upload a sketch first."
    mesh = _LAST_GENERATED_MESH or _SESSION.get("current_mesh")
    if mesh is None:
        return None, "⚠ No 3D model generated yet. Please click 'Generate 3D Asset' first."
    try:
        report = compute_sketch_fidelity(source, mesh)
        msg = f"★ Fidelity Score: {report.percentage}% ({report.rating})\n{report.explanation}"
        return report.comparison_image, msg
    except Exception as exc:
        return None, f"✘ Fidelity evaluation notice: {exc}"

def _run_multiview_generation() -> Tuple[Optional[Image.Image], str]:
    if not _USP_AVAILABLE:
        return None, "⚠ USP engine module not available."
    mesh = _LAST_GENERATED_MESH or _SESSION.get("current_mesh")
    if mesh is None:
        return None, "⚠ No 3D model generated yet. Please click 'Generate 3D Asset' first."
    try:
        composite = generate_multiview_projections(mesh)
        return composite, "✔ Multi-View Projections generated: Front, Right Side, Back, and Top angles."
    except Exception as exc:
        return None, f"✘ Multi-View projection notice: {exc}"

def _run_component_edit(
    component_name: str,
    material_name: str,
    custom_color: str,
) -> Tuple[Optional[str], str, str]:
    global _LAST_GENERATED_MESH
    if not _USP_AVAILABLE:
        return None, "⚠ USP engine module not available.", ""
    mesh = _LAST_GENERATED_MESH or _SESSION.get("current_mesh")
    if mesh is None:
        return None, "⚠ No active 3D model in session. Generate a 3D asset first.", ""
    try:
        updated_mesh = ComponentEditor.apply_edit(mesh, component_name, material_name, custom_color)
        from engine.mesh_utils import repair_mesh, export_mesh
        try:
            validated_mesh = repair_mesh(updated_mesh, strip_floating=False, fill_holes=True, center_and_scale=False)
        except Exception:
            validated_mesh = updated_mesh
        _LAST_GENERATED_MESH = validated_mesh
        _SESSION["current_mesh"] = validated_mesh

        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        export_glb_path = outputs_dir / "sketch2shape_generated.glb"
        export_mesh(validated_mesh, export_glb_path, export_format="glb")

        status = f"✔ Component '{component_name}' updated with '{material_name}'. 3D model refreshed."
        glb_url = f"/file={export_glb_path.as_posix()}?t={int(time.time())}"
        return str(export_glb_path), status, glb_url
    except Exception as exc:
        return None, f"✘ Component editing notice: {exc}", ""

def _run_design_variations(
    prompt_input: str,
    canvas_image: Optional[np.ndarray],
    uploaded_file: Optional[str],
    canvas_data_url: Optional[str],
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image], str]:
    if not _USP_AVAILABLE:
        return None, None, None, "⚠ USP engine module not available."
    source = _SESSION.get("current_source") or _resolve_input_source(canvas_image, uploaded_file, canvas_data_url)
    if source is None:
        return None, None, None, "⚠ No sketch found in session. Please draw or upload a sketch first."
    try:
        styles = ["Minimalist Porcelain", "Brushed Titanium Metal", "Organic Polished Walnut"]
        variations = generate_design_variations(source, prompt_input, selected_styles=styles)
        imgs = [v[1] for v in variations]
        while len(imgs) < 3:
            imgs.append(None)
        if "variation_images" not in _SESSION:
            _SESSION["variation_images"] = {}
        for idx, s in enumerate(styles):
            if idx < len(imgs) and imgs[idx] is not None:
                _SESSION["variation_images"][s] = imgs[idx]
        return imgs[0], imgs[1], imgs[2], "✔ 3 AI Design Variations generated from your sketch! Click 'Reconstruct' under any candidate."
    except Exception as exc:
        return None, None, None, f"✘ Design variations notice: {exc}"

def _reconstruct_variation_style(
    style_key: str,
    prompt_input: str,
    controlnet_scale: float,
    num_inference_steps: int,
    seed: int,
    canvas_image: Optional[np.ndarray],
    uploaded_file: Optional[str],
    canvas_data_url: Optional[str],
) -> GenerationOutput:
    style_desc = DESIGN_STYLE_PRESETS.get(style_key, style_key)
    full_prompt = f"{prompt_input.strip()}, {style_desc}" if prompt_input.strip() else style_desc
    variation_img = _SESSION.get("variation_images", {}).get(style_key)
    return _run_ai_generation(
        prompt=full_prompt,
        controlnet_scale=controlnet_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        canvas_image=canvas_image,
        uploaded_file=uploaded_file,
        canvas_data_url=canvas_data_url,
        preprocessed_preview=variation_img,
    )

def _clear_session() -> Tuple:
    global _LAST_GENERATED_MESH
    _LAST_GENERATED_MESH = None
    _SESSION["current_mesh"] = None
    _SESSION["current_source"] = None
    _SESSION["last_result"] = None
    _SESSION["variation_images"] = {}
    gr.Info("Session reset.")
    return (
        None,
        None,
        "",
        None,
        None,
        None,
        "",
    )


# ── Canva 3D Studio Elements & Material Presets ──────────────────────────────

CANVA_PRESET_ELEMENTS = {
    "vase": {
        "name": "Ceramic Vase",
        "icon": "🏺",
        "sketch": "outputs/samples/sample_01_vase/input_sketch.png",
        "prompt": "smooth porcelain ceramic vase with bulbous body, narrow neck, and flared rim, studio lighting, photorealistic, 8k",
    },
    "mug": {
        "name": "Nordic Mug",
        "icon": "☕",
        "sketch": "outputs/samples/sample_02_mug/input_sketch.png",
        "prompt": "modern minimalist ceramic coffee mug with solid cylindrical handle, clean matte porcelain glaze, 8k",
    },
    "chair": {
        "name": "Lounge Chair",
        "icon": "🪑",
        "sketch": "outputs/samples/sample_03_chair/input_sketch.png",
        "prompt": "minimalist Scandinavian geometric wooden chair with ergonomic curved backrest, natural oak finish, 8k",
    },
    "bottle": {
        "name": "Water Bottle",
        "icon": "🍶",
        "sketch": "outputs/samples/sample_04_bottle/input_sketch.png",
        "prompt": "tall stainless steel reusable insulated water bottle with screw cap, brushed metallic finish, 8k",
    },
    "bowl": {
        "name": "Zen Bowl",
        "icon": "🥣",
        "sketch": "outputs/samples/sample_05_bowl/input_sketch.png",
        "prompt": "deep hand-carved polished wooden artisanal salad bowl with rounded smooth contour, organic walnut, 8k",
    },
}

CANVA_MATERIAL_PRESETS = {
    "porcelain": ("🏺 Porcelain", "smooth glossy white porcelain ceramic with delicate glaze"),
    "gold": ("🪙 Brushed Gold", "luxurious brushed 24k gold metal with warm specular sheen"),
    "titanium": ("⚙️ Titanium", "aerospace-grade matte brushed titanium alloy with satin reflections"),
    "walnut": ("🪵 Walnut Wood", "hand-rubbed organic polished walnut wood with natural grain"),
    "terracotta": ("🧱 Terracotta", "matte artisanal baked terracotta clay with earthy texture"),
    "glass": ("🧊 Frosted Glass", "translucent frosted glass with soft subsurface scattering"),
    "obsidian": ("🌌 Obsidian", "glossy volcanic black obsidian with mirror reflections"),
    "coral": ("🍬 Pastel Coral", "smooth satin pastel coral polymer clay"),
}

def _load_canva_element(element_key: str):
    elem = CANVA_PRESET_ELEMENTS.get(element_key)
    if not elem:
        return None, "", "Element not found"
    sketch_path = _ROOT / elem["sketch"]
    if sketch_path.exists():
        img = Image.open(sketch_path).convert("RGB")
        _SESSION["current_source"] = img
        status = f"✨ Canva Element loaded: '{elem['name']}'. Click '✨ Magic 3D Generate' to synthesize."
        gr.Info(f"Loaded Canva {elem['name']} template!")
        return img, elem["prompt"], status
    return None, elem["prompt"], f"Sketch file not found at {sketch_path}"

def _apply_canva_material(material_key: str, current_prompt: str):
    _, desc = CANVA_MATERIAL_PRESETS.get(material_key, ("", ""))
    if not desc:
        return current_prompt or ""
    base = (current_prompt or "").strip()
    if not base:
        return desc
    if desc in base:
        return base
    return f"{base}, {desc}"

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title=APP_TITLE,
        head=_HEAD_SCRIPTS_HTML,
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        ),
        css="""
        .gradio-container { max-width: 1440px !important; margin: 0 auto; font-family: 'Inter', system-ui, sans-serif !important; }
        .panel-header {
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #94a3b8;
            padding-bottom: 4px;
            border-bottom: 1px solid rgba(125, 42, 232, 0.25);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .canva-elements-row button {
            background: #1e293b !important;
            border: 1px solid #334155 !important;
            color: #e2e8f0 !important;
            font-size: 11px !important;
            font-weight: 500 !important;
            border-radius: 6px !important;
            padding: 4px 6px !important;
            transition: all 0.2s ease !important;
        }
        .canva-elements-row button:hover {
            border-color: #7d2ae8 !important;
            background: rgba(125, 42, 232, 0.2) !important;
            color: #ffffff !important;
            box-shadow: 0 0 10px rgba(125, 42, 232, 0.35) !important;
        }
        .canva-materials-row button {
            background: rgba(30, 41, 59, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            color: #94a3b8 !important;
            font-size: 11px !important;
            padding: 3px 6px !important;
            border-radius: 12px !important;
            transition: all 0.2s ease !important;
        }
        .canva-materials-row button:hover {
            border-color: #00c4cc !important;
            background: rgba(0, 196, 204, 0.2) !important;
            color: #ffffff !important;
            box-shadow: 0 0 8px rgba(0, 196, 204, 0.3) !important;
        }
        #btn-magic-generate {
            background: linear-gradient(135deg, #7d2ae8 0%, #00c4cc 100%) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 13px !important;
            border: none !important;
            border-radius: 6px !important;
            box-shadow: 0 4px 14px rgba(125, 42, 232, 0.4) !important;
            transition: all 0.2s ease !important;
        }
        #btn-magic-generate:hover {
            box-shadow: 0 6px 20px rgba(125, 42, 232, 0.65) !important;
            transform: translateY(-1px);
        }
        .compact-accordion .label-wrap { padding: 6px 10px !important; }
        .status-bar textarea {
            font-family: var(--font-mono) !important;
            font-size: 12px !important;
            color: #94a3b8 !important;
            background: rgba(15, 23, 42, 0.6) !important;
            border: 1px solid #1e293b !important;
            border-radius: 6px !important;
            resize: none !important;
        }
        .export-row button { font-size: 12px !important; }
        @media (max-width: 768px) {
            .gradio-container { padding: 8px !important; }
            div#s2s-canvas-component { padding: 6px !important; }
        }
        """,
    ) as app:

        canvas_data_holder = gr.Textbox(
            visible=False,
            elem_id="canvas-data-holder",
            label="Internal Canvas Buffer",
        )

        glb_url_holder = gr.Textbox(
            visible=False,
            elem_id="s2s-glb-output-url",
            label="Internal GLB URL Buffer",
        )

        with gr.Row(elem_classes=["header-row"]):
            gr.HTML(
                f"""
                <div style="background: linear-gradient(135deg, rgba(125,42,232,0.18) 0%, rgba(0,196,204,0.14) 100%); border: 1px solid rgba(125,42,232,0.35); border-radius: 12px; padding: 14px 18px; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; box-shadow: 0 4px 20px rgba(125,42,232,0.12);">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="width: 42px; height: 42px; border-radius: 10px; background: linear-gradient(135deg, #7d2ae8, #00c4cc); display: flex; align-items: center; justify-content: center; font-size: 20px; box-shadow: 0 4px 12px rgba(125,42,232,0.4);">
                            ✨
                        </div>
                        <div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <h2 style="margin: 0; font-size: 20px; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em;">Canva 3D Studio</h2>
                                <span style="background: linear-gradient(135deg, #7d2ae8, #00c4cc); color: #ffffff; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 10px; letter-spacing: 0.05em; text-transform: uppercase;">Magic AI 3D</span>
                                <span style="font-size: 11px; color: #94a3b8; font-family: monospace;">v{APP_VERSION}</span>
                            </div>
                            <div style="font-size: 12px; color: #94a3b8; margin-top: 2px;">
                                TechZephyr 4.0 · IIT Bhubaneswar · 2D Sketch to Watertight 3D Geometry
                            </div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                        <span style="font-size: 11px; padding: 3px 8px; border-radius: 16px; background: rgba(30,41,59,0.8); border: 1px solid #334155; color: #cbd5e1;">🏺 Hollow Lathe Geometry</span>
                        <span style="font-size: 11px; padding: 3px 8px; border-radius: 16px; background: rgba(30,41,59,0.8); border: 1px solid #334155; color: #cbd5e1;">🖨️ Binary STL Ready</span>
                        <span style="font-size: 11px; padding: 3px 8px; border-radius: 16px; background: rgba(30,41,59,0.8); border: 1px solid #334155; color: #cbd5e1;">⚡ InstantMesh FP16</span>
                    </div>
                </div>
                """,
                elem_id="app-title",
            )

        with gr.Tabs(elem_id="main-app-tabs"):
            with gr.Tab("🎨 Studio (Canvas & 3D)", id="studio"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=380):
                        gr.HTML('<p class="panel-header">🎨 Canva 3D Elements (Click to Load Template)</p>')
                        with gr.Row(elem_classes=["canva-elements-row"]):
                            btn_elem_vase = gr.Button("🏺 Ceramic Vase", variant="secondary", size="sm")
                            btn_elem_mug = gr.Button("☕ Nordic Mug", variant="secondary", size="sm")
                            btn_elem_chair = gr.Button("🪑 Lounge Chair", variant="secondary", size="sm")
                            btn_elem_bottle = gr.Button("🍶 Water Bottle", variant="secondary", size="sm")
                            btn_elem_bowl = gr.Button("🥣 Zen Bowl", variant="secondary", size="sm")

                        gr.HTML('<p class="panel-header" style="margin-top:8px;">🎨 Drawing Canvas & Upload</p>')
                        gr.HTML(_HTML5_CANVAS_HTML, elem_id="html5-canvas-root")

                        with gr.Accordion(
                            label="Alternative / Fallback Sketchpad",
                            open=False,
                            elem_classes=["compact-accordion"],
                        ):
                            canvas = gr.Sketchpad(
                                label="Standard Drawing Canvas",
                                type="numpy",
                                image_mode="RGBA",
                                elem_id="sketch-canvas",
                                height=260,
                            )

                        gr.HTML('<div style="margin: 6px 0 4px; font-size: 11px; color: var(--body-text-color-subdued); text-align: center;">— or upload sketch / photo —</div>')

                        with gr.Row():
                            file_upload = gr.File(
                                label="Drag & Drop Sketch (.png, .jpg, .jpeg)",
                                file_types=SUPPORTED_UPLOAD_EXTENSIONS,
                                file_count="single",
                                elem_id="file-uploader",
                                scale=8,
                            )
                            btn_remove_upload = gr.Button(
                                "✕",
                                elem_id="btn-remove-upload",
                                scale=1,
                                min_width=40,
                                size="sm",
                                variant="secondary",
                            )

                        prompt_input = gr.Textbox(
                            label="Material / Style Prompt (Optional)",
                            placeholder='e.g. "matte porcelain finish", "brushed aluminum alloy", "weathered terracotta"',
                            lines=2,
                            max_lines=4,
                            elem_id="prompt-input",
                        )

                        gr.HTML('<p class="panel-header" style="margin-top:6px;">✨ Canva Magic Style & Material Palette</p>')
                        with gr.Row(elem_classes=["canva-materials-row"]):
                            btn_mat_porcelain = gr.Button("🏺 Porcelain", variant="secondary", size="sm")
                            btn_mat_gold = gr.Button("🪙 Gold", variant="secondary", size="sm")
                            btn_mat_titanium = gr.Button("⚙️ Titanium", variant="secondary", size="sm")
                            btn_mat_walnut = gr.Button("🪵 Walnut", variant="secondary", size="sm")
                        with gr.Row(elem_classes=["canva-materials-row"]):
                            btn_mat_terracotta = gr.Button("🧱 Terracotta", variant="secondary", size="sm")
                            btn_mat_glass = gr.Button("🧊 Glass", variant="secondary", size="sm")
                            btn_mat_obsidian = gr.Button("🌌 Obsidian", variant="secondary", size="sm")
                            btn_mat_coral = gr.Button("🍬 Coral", variant="secondary", size="sm")

                        with gr.Accordion(
                            label="Pipeline Parameters",
                            open=False,
                            elem_classes=["compact-accordion"],
                        ):
                            controlnet_scale = gr.Slider(
                                label="ControlNet Conditioning Scale",
                                minimum=0.0,
                                maximum=2.0,
                                value=0.85,
                                step=0.05,
                                info="Strength of sketch conditioning. 0.70–0.95 recommended.",
                            )
                            inference_steps = gr.Slider(
                                label="Denoising Steps",
                                minimum=10,
                                maximum=100,
                                value=30,
                                step=5,
                                info="More steps → higher quality, slower runtime.",
                            )
                            seed_input = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                                info="-1 = random seed.",
                            )

                        with gr.Row():
                            btn_preprocess = gr.Button(
                                "① Preprocess",
                                variant="secondary",
                                size="sm",
                            )
                            btn_generate = gr.Button(
                                "✨ Magic 3D Generate",
                                variant="primary",
                                size="sm",
                                elem_id="btn-magic-generate",
                            )
                            btn_clear = gr.Button(
                                "Clear",
                                variant="stop",
                                size="sm",
                            )

                    with gr.Column(scale=1, min_width=380):
                        gr.HTML('<p class="panel-header">3D WebGL Asset Preview</p>')
                        gr.HTML(_WEBGL_VIEWER_HTML, elem_id="webgl-viewport")

                        preprocessed_view = gr.Image(
                            label="Preprocessed / Lineart Preview",
                            type="pil",
                            interactive=False,
                            height=180,
                        )

                        depth_view = gr.Image(
                            label="Depth / Normal Map [AI Engine Output]",
                            type="pil",
                            interactive=False,
                            height=180,
                            visible=True,
                        )

                        gr.HTML('<p class="panel-header" style="margin-top: 8px;">Mesh Export & Downloads</p>')

                        with gr.Row(elem_classes=["export-row"]):
                            export_format = gr.Dropdown(
                                label="Format",
                                choices=list(SUPPORTED_EXPORT_FORMATS),
                                value="GLB",
                                scale=1,
                            )
                            btn_export = gr.Button(
                                "⬇  Export Mesh",
                                variant="secondary",
                                size="sm",
                                scale=2,
                            )

                        export_file = gr.File(
                            label="Download Asset",
                            interactive=False,
                            visible=True,
                        )

            with gr.Tab("📚 Preset Library", id="gallery"):
                gr.Markdown("<h3>Curated Evaluation Presets</h3>\nSelect any sample to immediately load its sketch and prompt into the Studio canvas.")
                with gr.Row():
                    for p_key, p_val in _PRESETS_DATA.items():
                        with gr.Column(scale=1, min_width=220):
                            gr.HTML(f"""
                            <div class="s2s-sample-card">
                                <div style="display:flex;align-items:center;justify-content:space-between;">
                                    <span style="font-weight:600;font-size:13px;color:#f8fafc;">{p_val['name']}</span>
                                    <span style="background:#0284c7;color:#ffffff;font-size:10px;padding:2px 6px;border-radius:10px;font-family:monospace;">Watertight</span>
                                </div>
                                <div style="aspect-ratio:1/1;background:#ffffff;border-radius:6px;overflow:hidden;border:1px solid #334155;">
                                    <img src="{p_val['b64']}" style="width:100%;height:100%;object-fit:contain;display:block;" alt="{p_val['name']}">
                                </div>
                                <div style="font-size:11px;color:#94a3b8;font-style:italic;line-height:1.3;min-height:30px;">
                                    "{p_val['prompt']}"
                                </div>
                                <div style="display:flex;justify-content:space-between;font-size:10px;color:#64748b;font-family:monospace;">
                                    <span>{p_val['faces']:,} faces</span>
                                    <span>{p_val['vertices']:,} vertices</span>
                                </div>
                                <button type="button" onclick="window.s2sLoadPreset('{p_key}')" style="width:100%;background:#0284c7;color:#ffffff;border:1px solid #38bdf8;padding:6px;border-radius:4px;font-size:11px;font-weight:600;cursor:pointer;margin-top:4px;">
                                    🚀 Load into Studio
                                </button>
                            </div>
                            """)

            with gr.Tab("🌟 AI Studio Pro (USPs)", id="usps"):
                with gr.Tabs():
                    with gr.Tab("1. 🎯 Sketch Fidelity Score"):
                        gr.Markdown("Evaluate how closely the 3D model's silhouette preserves the hand-drawn sketch contours.")
                        with gr.Row():
                            btn_eval_fidelity = gr.Button("Calculate Fidelity Score", variant="secondary", size="sm")
                        fidelity_status = gr.Textbox(label="Silhouette Similarity & Alignment Assessment", interactive=False, lines=2)
                        fidelity_preview = gr.Image(label="Contour Alignment Overlay (Green: Overlap, Blue: 3D Hull)", type="pil", interactive=False, height=280)

                    with gr.Tab("2. 🌐 Multi-View Intelligence"):
                        gr.Markdown("Inspect the 3D asset from multiple synchronized viewpoints (Front, Right Side, Back, Top). Use viewer toolbar buttons for interactive 360° camera presets.")
                        with gr.Row():
                            btn_gen_multiview = gr.Button("Render 4-View Orthographic Collage", variant="secondary", size="sm")
                        multiview_status = gr.Textbox(label="Multi-View Status", interactive=False, lines=1)
                        multiview_preview = gr.Image(label="4-Angle Orthographic Views (Front / Side / Back / Top)", type="pil", interactive=False, height=320)

                    with gr.Tab("3. 🎨 Component-Level Editing"):
                        gr.Markdown("Select and customize individual regions of the 3D asset independently without altering overall geometry.")
                        with gr.Row():
                            comp_dropdown = gr.Dropdown(label="Target Component", choices=COMPONENT_PRESETS if _USP_AVAILABLE else ["Upper Section / Top Rim", "Central Body / Core", "Base / Stand / Legs"], value="Base / Stand / Legs", scale=1)
                            mat_dropdown = gr.Dropdown(label="Material Palette Preset", choices=list(MATERIAL_PALETTES.keys()) if _USP_AVAILABLE else ["Matte Black", "Polished Walnut Wood", "Brushed Titanium", "Porcelain Ceramic", "Terracotta Clay", "Electric Cobalt", "Gloss Gold", "Forest Olive"], value="Polished Walnut Wood", scale=1)
                            custom_color = gr.ColorPicker(label="Custom Color Hex", value="#784b2d", scale=1)
                        with gr.Row():
                            btn_apply_component = gr.Button("Apply Component Material Edit", variant="primary", size="sm")
                        comp_status = gr.Textbox(label="Component Edit Status", interactive=False, lines=1)

                    with gr.Tab("4. ⚡ AI Design Variations"):
                        gr.Markdown("Generate multiple aesthetic styles from the same structural sketch without redrawing.")
                        with gr.Row():
                            btn_gen_variations = gr.Button("Generate 3 Style Variations from Sketch", variant="secondary", size="sm")
                        var_status = gr.Textbox(label="Variations Status", interactive=False, lines=1)
                        with gr.Row():
                            with gr.Column(scale=1):
                                var_img_1 = gr.Image(label="Variation 1: Minimalist Porcelain", type="pil", interactive=False, height=200)
                                btn_reconstruct_var_1 = gr.Button("Reconstruct as 3D (Porcelain)", variant="secondary", size="sm")
                            with gr.Column(scale=1):
                                var_img_2 = gr.Image(label="Variation 2: Brushed Titanium", type="pil", interactive=False, height=200)
                                btn_reconstruct_var_2 = gr.Button("Reconstruct as 3D (Titanium)", variant="secondary", size="sm")
                            with gr.Column(scale=1):
                                var_img_3 = gr.Image(label="Variation 3: Polished Walnut", type="pil", interactive=False, height=200)
                                btn_reconstruct_var_3 = gr.Button("Reconstruct as 3D (Walnut)", variant="secondary", size="sm")

            with gr.Tab("📖 User Guide", id="guide"):
                gr.Markdown("""
                <h3>🚀 Quick Start Guide</h3>
                
                Follow these three simple steps to convert any conceptual idea into a watertight 3D model:
                
                <h4>1. Draw or Pick a Preset</h4>
                - Draw your object on the **Studio Canvas** using the brush tool, color picker, and stroke size sliders.
                - Or select one of the 5 curated presets in the **Preset Library** for instant one-click testing.
                - Gestures: Left click / stylus to draw, right click or toggle eraser to correct strokes.
                
                <h4>2. Add Style Guidance (Optional)</h4>
                - Enter natural language material or style prompts (e.g. *"smooth porcelain ceramic with blue glaze"*, *"brushed titanium alloy"*).
                - Pipeline hyperparameters can be adjusted under the **Pipeline Parameters** accordion.
                
                <h4>3. Inspect in 360° & Export</h4>
                - Click **Generate 3D Asset** to trigger the AI diffusion and InstantMesh 3D reconstruction pipeline.
                - Inspect the generated model in the WebGL 3D previewer with 360° orbital rotation, pan, zoom, and wireframe mode.
                - Download binary `.glb` or export to `.obj`, `.stl`, or `.ply` for 3D printing and CAD downstream tooling.
                """)

        status_bar = gr.Textbox(
            label="Pipeline Status",
            value="Ready.",
            interactive=False,
            elem_classes=["status-bar"],
            lines=1,
            max_lines=2,
        )

        btn_remove_upload.click(
            fn=_clear_uploaded_file,
            inputs=[],
            outputs=[file_upload, preprocessed_view, status_bar],
        )

        file_upload.clear(
            fn=_clear_uploaded_file,
            inputs=[],
            outputs=[file_upload, preprocessed_view, status_bar],
        )

        btn_preprocess.click(
            fn=_run_preprocessing,
            inputs=[canvas, file_upload, canvas_data_holder],
            outputs=[preprocessed_view, status_bar],
        )

        btn_generate.click(
            fn=_run_ai_generation,
            inputs=[
                prompt_input,
                controlnet_scale,
                inference_steps,
                seed_input,
                canvas,
                file_upload,
                canvas_data_holder,
                preprocessed_view,
            ],
            outputs=[depth_view, export_file, status_bar, glb_url_holder],
            js="() => { if (window.s2sViewerInstance) { window.s2sViewerInstance.showLoading('AI Engine generating 3D asset...', 20); } }",
        )

        btn_export.click(
            fn=_run_mesh_export,
            inputs=[export_format],
            outputs=[export_file, status_bar],
        )

        btn_clear.click(
            fn=_clear_session,
            inputs=[],
            outputs=[
                canvas,
                file_upload,
                prompt_input,
                preprocessed_view,
                depth_view,
                export_file,
                status_bar,
            ],
            js="() => { if (window.s2sClearCanvas) window.s2sClearCanvas(); }",
        )

        btn_elem_vase.click(fn=lambda: _load_canva_element("vase"), inputs=[], outputs=[preprocessed_view, prompt_input, status_bar])
        btn_elem_mug.click(fn=lambda: _load_canva_element("mug"), inputs=[], outputs=[preprocessed_view, prompt_input, status_bar])
        btn_elem_chair.click(fn=lambda: _load_canva_element("chair"), inputs=[], outputs=[preprocessed_view, prompt_input, status_bar])
        btn_elem_bottle.click(fn=lambda: _load_canva_element("bottle"), inputs=[], outputs=[preprocessed_view, prompt_input, status_bar])
        btn_elem_bowl.click(fn=lambda: _load_canva_element("bowl"), inputs=[], outputs=[preprocessed_view, prompt_input, status_bar])

        btn_mat_porcelain.click(fn=lambda p: _apply_canva_material("porcelain", p), inputs=[prompt_input], outputs=[prompt_input])
        btn_mat_gold.click(fn=lambda p: _apply_canva_material("gold", p), inputs=[prompt_input], outputs=[prompt_input])
        btn_mat_titanium.click(fn=lambda p: _apply_canva_material("titanium", p), inputs=[prompt_input], outputs=[prompt_input])
        btn_mat_walnut.click(fn=lambda p: _apply_canva_material("walnut", p), inputs=[prompt_input], outputs=[prompt_input])
        btn_mat_terracotta.click(fn=lambda p: _apply_canva_material("terracotta", p), inputs=[prompt_input], outputs=[prompt_input])
        btn_mat_glass.click(fn=lambda p: _apply_canva_material("glass", p), inputs=[prompt_input], outputs=[prompt_input])
        btn_mat_obsidian.click(fn=lambda p: _apply_canva_material("obsidian", p), inputs=[prompt_input], outputs=[prompt_input])
        btn_mat_coral.click(fn=lambda p: _apply_canva_material("coral", p), inputs=[prompt_input], outputs=[prompt_input])

        btn_eval_fidelity.click(
            fn=_run_fidelity_evaluation,
            inputs=[canvas, file_upload, canvas_data_holder],
            outputs=[fidelity_preview, fidelity_status],
        )

        btn_gen_multiview.click(
            fn=_run_multiview_generation,
            inputs=[],
            outputs=[multiview_preview, multiview_status],
        )

        btn_apply_component.click(
            fn=_run_component_edit,
            inputs=[comp_dropdown, mat_dropdown, custom_color],
            outputs=[export_file, comp_status, glb_url_holder],
        )

        btn_gen_variations.click(
            fn=_run_design_variations,
            inputs=[prompt_input, canvas, file_upload, canvas_data_holder],
            outputs=[var_img_1, var_img_2, var_img_3, var_status],
        )

        btn_reconstruct_var_1.click(
            fn=lambda p, s, st, sd, c, f, ch: _reconstruct_variation_style("Minimalist Porcelain", p, s, st, sd, c, f, ch),
            inputs=[prompt_input, controlnet_scale, inference_steps, seed_input, canvas, file_upload, canvas_data_holder],
            outputs=[depth_view, export_file, status_bar, glb_url_holder],
            js="() => { if (window.s2sViewerInstance) { window.s2sViewerInstance.showLoading('Reconstructing Minimalist Porcelain 3D asset...', 20); } }",
        )

        btn_reconstruct_var_2.click(
            fn=lambda p, s, st, sd, c, f, ch: _reconstruct_variation_style("Brushed Titanium Metal", p, s, st, sd, c, f, ch),
            inputs=[prompt_input, controlnet_scale, inference_steps, seed_input, canvas, file_upload, canvas_data_holder],
            outputs=[depth_view, export_file, status_bar, glb_url_holder],
            js="() => { if (window.s2sViewerInstance) { window.s2sViewerInstance.showLoading('Reconstructing Brushed Titanium 3D asset...', 20); } }",
        )

        btn_reconstruct_var_3.click(
            fn=lambda p, s, st, sd, c, f, ch: _reconstruct_variation_style("Organic Polished Walnut", p, s, st, sd, c, f, ch),
            inputs=[prompt_input, controlnet_scale, inference_steps, seed_input, canvas, file_upload, canvas_data_holder],
            outputs=[depth_view, export_file, status_bar, glb_url_holder],
            js="() => { if (window.s2sViewerInstance) { window.s2sViewerInstance.showLoading('Reconstructing Polished Walnut 3D asset...', 20); } }",
        )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f"{APP_TITLE} UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=7860, help="Bind port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    parser.add_argument("--debug", action="store_true", help="Enable Gradio debug mode")
    args = parser.parse_args()

    if isinstance(gr, _GrMock) or not hasattr(gr, "Blocks"):
        print("\n" + "=" * 70)
        print("  [ERROR] 'gradio' package is required to run the Web UI.")
        print("  Please install gradio using:")
        print("      pip install gradio>=4.20.0")
        print("  Or install all project dependencies:")
        print("      pip install -r requirements.txt")
        print("=" * 70 + "\n")
        sys.exit(1)

    logger.info("Starting %s v%s", APP_TITLE, APP_VERSION)

    outputs_dir = Path("outputs").resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ui_dir = Path("ui").resolve()

    demo = build_ui()

    ports_to_try = [args.port] if args.port != 7860 else [7860, 7861, 7862, 7865, 8080]
    launched = False

    for p in ports_to_try:
        try:
            logger.info("Attempting to launch UI server on http://%s:%d ...", args.host, p)
            demo.launch(
                server_name=args.host,
                server_port=p,
                share=args.share,
                debug=args.debug,
                show_error=True,
                allowed_paths=[str(ui_dir), str(outputs_dir)],
                prevent_thread_lock=False,
            )
            launched = True
            break
        except OSError as err:
            logger.warning("Port %d is unavailable or in use (%s). Trying next port...", p, err)

    if not launched:
        logger.error("Failed to bind UI server on any candidate port. Ensure ports 7860-7865 are not held by orphan processes.")
        sys.exit(1)
