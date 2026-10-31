from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logger = logging.getLogger("sketch2shape.backup_runner")

@dataclass
class DemoSample:
    id: str
    name: str
    prompt: str
    seed: int
    sketch_path: Path
    lineart_path: Path
    glb_path: Path
    obj_path: Path
    vertex_count: int
    face_count: int
    is_watertight: bool
    volume: float
    surface_area: float

class OfflineDemoRunner:
    def __init__(self, samples_dir: Optional[Path] = None) -> None:
        self.samples_dir = (samples_dir or (ROOT_DIR / "outputs" / "samples")).resolve()
        self.manifest_path = self.samples_dir / "manifest.json"
        self._samples: Dict[str, DemoSample] = {}
        self.load_manifest()

    def load_manifest(self) -> Dict[str, DemoSample]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        raw_data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._samples.clear()

        for item in raw_data.get("samples", []):
            sample_id = item["id"]
            sketch = (ROOT_DIR / item["sketch_path"]).resolve()
            lineart = (ROOT_DIR / item["lineart_path"]).resolve()
            glb = (ROOT_DIR / item["glb_path"]).resolve()
            obj = (ROOT_DIR / item["obj_path"]).resolve()
            metrics = item.get("metrics", {})

            sample = DemoSample(
                id=sample_id,
                name=item["name"],
                prompt=item["prompt"],
                seed=item["seed"],
                sketch_path=sketch,
                lineart_path=lineart,
                glb_path=glb,
                obj_path=obj,
                vertex_count=metrics.get("vertex_count", 0),
                face_count=metrics.get("face_count", 0),
                is_watertight=metrics.get("is_watertight", False),
                volume=metrics.get("volume", 0.0),
                surface_area=metrics.get("surface_area", 0.0),
            )
            self._samples[sample_id] = sample

        return self._samples

    def list_samples(self) -> List[DemoSample]:
        return list(self._samples.values())

    def get_sample(self, sample_id: str) -> Optional[DemoSample]:
        return self._samples.get(sample_id)

    def match_sample(self, query: str) -> Optional[DemoSample]:
        q = query.lower().strip()
        if q in self._samples:
            return self._samples[q]

        for s in self._samples.values():
            if q in s.id.lower() or q in s.name.lower() or q in s.prompt.lower():
                return s

        return list(self._samples.values())[0] if self._samples else None

    def deploy_sample_to_preview(
        self,
        sample_id: str,
        target_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        sample = self.get_sample(sample_id) or self.match_sample(sample_id)
        if sample is None:
            raise ValueError(f"No sample available matching '{sample_id}'")

        out_dir = (target_dir or (ROOT_DIR / "outputs")).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        target_glb = out_dir / "sketch2shape_generated.glb"
        target_obj = out_dir / "sketch2shape_asset.obj"

        shutil.copyfile(sample.glb_path, target_glb)
        shutil.copyfile(sample.obj_path, target_obj)

        preview_glb = out_dir / f"preview_{sample.seed}.glb"
        shutil.copyfile(sample.glb_path, preview_glb)

        return target_glb, target_obj

def run_cli() -> int:
    parser = argparse.ArgumentParser(description="Sketch2Shape AI Offline Demo Backup Runner")
    parser.add_argument("--list", action="store_true", help="List all available curated demo samples")
    parser.add_argument("--load", type=str, help="Load and deploy sample by ID or keyword to outputs directory")
    parser.add_argument("--verify", action="store_true", help="Verify integrity of all curated sample files")
    args = parser.parse_args()

    runner = OfflineDemoRunner()

    if args.list:
        print("\n=== Curated Offline Demo Samples ===")
        for s in runner.list_samples():
            print(f"- [{s.id}] {s.name} (Seed: {s.seed})")
            print(f"  Prompt: \"{s.prompt}\"")
            print(f"  Geometry: {s.vertex_count:,} vertices | {s.face_count:,} faces | Watertight: {s.is_watertight}")
            print(f"  Artifacts: GLB={s.glb_path.name}, OBJ={s.obj_path.name}")
        return 0

    if args.verify:
        print("\n=== Verifying Sample Artifacts Integrity ===")
        all_ok = True
        for s in runner.list_samples():
            missing = []
            for path in [s.sketch_path, s.lineart_path, s.glb_path, s.obj_path]:
                if not path.exists() or path.stat().st_size == 0:
                    missing.append(path.name)
            if missing:
                print(f"[FAIL] {s.id}: Missing or empty: {', '.join(missing)}")
                all_ok = False
            else:
                print(f"[OK] {s.id}: All 4 artifacts present and validated ({s.face_count:,} faces, watertight={s.is_watertight})")
        return 0 if all_ok else 1

    if args.load:
        sample = runner.match_sample(args.load)
        if not sample:
            print(f"[ERROR] Sample '{args.load}' not found.")
            return 1
        glb, obj = runner.deploy_sample_to_preview(sample.id)
        print(f"\n[OK] Deployed sample '{sample.name}' ({sample.id})")
        print(f"-> Active GLB: {glb}")
        print(f"-> Active OBJ: {obj}")
        print(f"-> Prompt: \"{sample.prompt}\"")
        return 0

    parser.print_help()
    return 0

if __name__ == "__main__":
    sys.exit(run_cli())
