"""
app.py
======
Sketch2Shape AI  ·  TechZephyr 4.0 Hackathon  ·  IIT Bhubaneswar
Main application entry point. Launches the original Single Page Application (SPA)
Web Interface frontend connected to the generative diffusion and 3D reconstruction backend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sketch2Shape AI — Final SPA Web Application")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code change")
    args = parser.parse_args()

    import uvicorn
    print("\n" + "=" * 70)
    print(" 🚀 Sketch2Shape AI — Single Page Application (SPA)")
    print(f" 🌐 Web Interface : http://{args.host}:{args.port}")
    print(f" 📚 API Docs      : http://{args.host}:{args.port}/docs")
    print("=" * 70 + "\n")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
