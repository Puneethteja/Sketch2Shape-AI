import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

if DIST.exists():
    shutil.rmtree(DIST)

DIST.mkdir(parents=True, exist_ok=True)

WEB = ROOT / "web"
shutil.copy2(WEB / "index.html", DIST / "index.html")
shutil.copytree(WEB / "css", DIST / "css")
shutil.copytree(WEB / "js", DIST / "js")

OUTPUTS = ROOT / "outputs" / "samples"
if OUTPUTS.exists():
    shutil.copytree(OUTPUTS, DIST / "outputs" / "samples")
    shutil.copytree(OUTPUTS, DIST / "assets" / "samples")

print(f"Successfully assembled distribution bundle at: {DIST}")
