"""
Sketch2Shape-AI — Uvicorn startup script
========================================
Run:
    python api/server.py
    # or from repo root:
    python -m api.server
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
