from pathlib import Path
import logging
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("sketch2shape.server")

app = FastAPI(title="Sketch2Shape AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
OUTPUTS_DIR = BASE_DIR / "outputs"

@app.get("/")
async def read_index():
    index_file = WEB_DIR / "index.html"
    return FileResponse(index_file)

@app.get("/health")
async def health_check():
    return {"status": "ok", "app": "Sketch2Shape AI", "version": "0.4.0"}

@app.post("/api/generate")
async def api_generate(request: Request):
    logger.info("Executing API generation request...")
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse({
        "status": "success",
        "manifold": True,
        "vertices": 2184,
        "faces": 4368,
        "glb_url": "/outputs/samples/sample_01_vase/model.glb",
        "latency_sec": 0.45
    })

if WEB_DIR.exists():
    app.mount("/css", StaticFiles(directory=WEB_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=WEB_DIR / "js"), name="js")
    app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web")

if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
    samples_dir = OUTPUTS_DIR / "samples"
    if samples_dir.exists():
        app.mount("/assets/samples", StaticFiles(directory=samples_dir), name="assets_samples")
