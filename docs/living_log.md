# Sketch2Shape AI — Living Development Log

> **TechZephyr 4.0 Hackathon · IIT Bhubaneswar**  
> Chronological record of development milestones, design decisions, and cross-team integration notes.

---

## Log Format

Each entry follows the structure:

```
### [YYYY-MM-DD] · <Author> · <Phase> · <Title>
```

---

### [2026-10-31] · Laxmi · Phase 1 · UI Skeleton & Technical Architecture Documentation

**Engineer:** Laxmi — Lead Frontend, WebGL & Technical Documentation  
**Scope:** UI skeleton (`ui/app.py`) + architecture spec (`docs/technical_architecture.md`)

#### Deliverables shipped

| Artifact | Path | Description |
|----------|------|-------------|
| UI skeleton | [`ui/app.py`](../ui/app.py) | Gradio Blocks dual-viewport interface |
| Architecture spec | [`docs/technical_architecture.md`](./technical_architecture.md) | Full system dataflow, algorithmic specs, integration contracts |
| Living log (this file) | [`docs/living_log.md`](./living_log.md) | Phase 1 entry |

#### UI Architecture decisions

- **Framework choice — Gradio Blocks:** Selected over Streamlit for its native
  component-level event wiring (`click`, `change`) without requiring full page
  reloads, and first-class support for image canvas / sketchpad widgets. Gradio
  `gr.Sketchpad` provides RGBA output directly consumable by
  `engine.preprocessing.ingest()`.

- **Dual-viewport grid:** Implemented as a two-column `gr.Row` with `scale=1`
  on both sides, giving equal visual weight to input controls and output
  previews. The layout avoids decorative elements; all space is functional.

- **WebGL viewport stub:** Rendered as a raw HTML `<div>` block with a
  cube-icon SVG placeholder. Will be replaced with a Three.js `<canvas>` in
  Phase 2 once GLB export is operational. The stub is styled to match the
  dark editor aesthetic of the Gradio default theme.

- **Pipeline parameter accordion:** Collapsed by default to keep the interface
  clean for non-technical demo audiences. Power users can expand to tune
  `controlnet_conditioning_scale` (0–2), `num_inference_steps` (10–100), and
  RNG `seed`.

- **Handler stubs:** `_run_ai_generation()` and `_run_mesh_export()` are wired
  to the UI but return stub responses until Mona's engine module (`engine/ai_engine.py`)
  lands. This prevents broken state in the live demo while still exercising
  the full event graph.

- **Graceful degradation:** `_PREPROCESSING_AVAILABLE` flag gates the import
  of `engine.preprocessing` so the UI launches cleanly in environments where
  OpenCV/PyTorch are not yet installed (e.g., CI preview builds).

#### Architecture document decisions

- **Tensor lifecycle table:** Every shape and dtype transition from raw user
  input to serialised mesh is enumerated to make integration debugging
  unambiguous. Types that have caused silent bugs elsewhere (HWC vs CHW,
  uint8 vs float32, RGB vs BGR) are called out explicitly.

- **ControlNet scale bounds documented:** Hard bounds (< 0.5 too weak,
  > 1.5 causes aliasing) were derived from empirical ControlNet community
  benchmarks and codified here to prevent trial-and-error during the
  hackathon time window.

- **FP16 guard checklist:** The latent scaling constant `0.18215` and the
  CLIP text encoder FP32 exception are both documented as commonly missed
  gotchas when assembling custom diffusion pipelines from scratch.

- **`GenerationResult` dataclass contract:** Defined a typed return type for
  `engine.ai_engine.generate()` so Laxmi's UI and Mona's engine can be
  developed and tested in parallel against a shared interface spec without
  waiting for the other side to be complete.

- **Non-manifold recovery sequence:** The 7-step repair sequence in §3.4 is
  deterministic and idempotent — running it twice on an already-repaired mesh
  produces no change. This was a deliberate design constraint to make the
  repair safe to call defensively at any point in the export path.

#### Open items / handoffs

| Item | Owner | Target phase | Status |
|------|-------|-------------|--------|
| `engine/ai_engine.py` — implement `generate()` + `GenerationResult` | Mona | Phase 2 | Pending |
| `engine/mesh_utils.py` — implement `repair_mesh()` | Mona + Laxmi | Phase 2 | In Progress |
| Three.js WebGL viewport integration | Laxmi | Phase 2 | **Completed** |
| HTML5 Drawing Canvas & Toolbar | Laxmi | Phase 2 | **Completed** |
| Drag-and-drop raster uploader (.png, .jpg, .jpeg) | Laxmi | Phase 2 | **Completed** |
| Session state management & base64 sync | Laxmi | Phase 2 | **Completed** |
| `gradio` added to `requirements.txt` | Puneeth / Laxmi | Immediate | **Completed** |

#### Integration notes for Puneeth

The UI consumes `engine.preprocessing.preprocess()` exactly as documented
in Contract A (§4.1 of the architecture spec). No API changes to
`preprocessing.py` are required for Phase 2. The `ingest()` function's
handling of base64 data-URLs is particularly important — the Gradio
`gr.Sketchpad` widget delivers canvas data as a NumPy RGBA array, which
`ingest()` already handles via the `np.ndarray` branch.

#### Integration notes for Mona

- `GenerationResult` dataclass is defined in `docs/technical_architecture.md`
  §4.2. Please implement it in `engine/ai_engine.py` as the return type for
  `generate()`.
- The UI imports `engine.ai_engine` lazily (inside handler functions) so
  missing the module does not crash the server at startup — this allows the
  UI to be demoed before the AI engine is ready.
- `result.mesh.is_watertight` **must** be `True` before `GenerationResult` is
  returned. The repair sequence in §3.4 should be called inside `generate()`
  before returning.

---

### [2026-10-31] · Laxmi · Phase 2 · Interactive HTML5 Drawing Canvas & WebGL 3D Viewer Integration

**Engineer:** Laxmi — Lead Frontend, WebGL & Technical Documentation  
**Scope:** HTML5 drawing canvas (`ui/app.py`), WebGL 3D viewer component (`ui/viewer.js`), file upload zone, style prompt field, and smoke-test suite (`tests/test_ui.py`)

#### Deliverables shipped

| Artifact | Path | Description |
|----------|------|-------------|
| Interactive UI | [`ui/app.py`](../ui/app.py) | Full Gradio Blocks interface featuring custom HTML5 drawing canvas, file uploader (.png/jpg/jpeg), prompt inputs, and dual-viewport layout |
| WebGL 3D Viewer | [`ui/viewer.js`](../ui/viewer.js) | Three.js / @google/model-viewer 3D component with 360° orbit, pan, zoom, loading spinners, and binary GLB download triggers |
| UI Smoke Test Suite | [`tests/test_ui.py`](../tests/test_ui.py) | 13 test cases covering components, formats, base64 data-URL preprocessing, viewer script validation, and canvas controls |
| Living Log (this file) | [`docs/living_log.md`](./living_log.md) | Phase 2 completion milestone |

#### HTML5 Canvas Architecture decisions

- **PointerEvents input abstraction:** Implemented `pointerdown`, `pointermove`,
  `pointerup`, and `pointercancel` with `setPointerCapture()` to provide uniform
  pen pressure, stylus, mouse, and touch interaction. Multi-touch gestures are
  isolated to prevent browser viewport scrolling while drawing.

- **High-DPI resolution scaling:** Canvas uses a native 512×512 resolution backing
  mapped to display dimensions, preventing blurriness on Retina and high-DPI
  laptop displays.

- **Midpoint Quadratic Bézier curve smoothing:** Rather than connecting raw discrete
  pointer samples with straight line segments, midpoints between sequential
  samples are computed and interpolated via `ctx.quadraticCurveTo()`. This produces
  silky smooth strokes and eliminates aliasing artifacts before lineart extraction.

- **Undo / Redo buffer stacks:** Implemented dual history stacks (`undoStack` and
  `redoStack`) capturing full-resolution `ImageData` snapshots before each stroke,
  capped at 30 history states to preserve memory. Undo and Redo buttons are
  dynamically enabled/disabled with real-time stack depth counters.

- **Eraser & Clear canvas integration:** The eraser tool switches stroke colors
  to match the clean white canvas background (`#ffffff`), ensuring the exported
  raster remains an opaque RGB field easily thresholded by OpenCV in Stage 1.
  Clearing the canvas snapshots current state to the undo buffer so accidental
  resets can be immediately reversed.

- **Bidirectional Gradio synchronization:** Canvas modifications trigger automated
  `toDataURL("image/png")` serialisation into a hidden Gradio buffer element
  (`#canvas-data-holder`), dispatching standard DOM `input` and `change` events.
  `_run_preprocessing()` ingests these base64 data-URLs directly via
  `engine.preprocessing.ingest()`.

- **Retaining backward compatibility:** Gradio's standard `gr.Sketchpad` is
  preserved inside a collapsed fallback accordion to guarantee that existing
  component inventory tests and headless environments function without disruption.

#### WebGL 3D Viewer Architecture decisions

- **Three.js & OrbitControls integration:** Embedded in `ui/viewer.js` with full
  360-degree orbit rotation, screen-space panning (right-click / 2-finger touch),
  and distance-clamped perspective zoom (0.5m to 20m). Damping is enabled
  (`dampingFactor = 0.05`) for smooth physical deceleration.

- **Dynamic dependency resolution with offline fallback:** Automatically resolves
  Three.js, OrbitControls, and GLTFLoader via CDN, with an integrated native WebGL
  context fallback if the hackathon environment is air-gapped or offline.

- **Studio 3-point lighting rig:** Calibrated for industrial design visualization:
  - Key Light: Directional white light (intensity 1.2 at `[4, 6, 4]`)
  - Fill Light: Directional cool slate light (intensity 0.6 at `[-4, 2, -2]`)
  - Rim Light: Directional cyan accent light (intensity 0.8 at `[0, 5, -5]`)
  - Hemisphere Light: Soft ambient illumination (`#f8fafc` sky, `#0f172a` ground)
  - Ground Grid Helper: Slate coordinate plane (`#334155` / `#1e293b`)

- **Bounding-box auto-framing:** Every loaded model is queried with `THREE.Box3`
  to compute spatial extents and geometric center. The model is normalized to the
  origin and the camera is smoothly repositioned to fit the frustum regardless
  of imported scale.

- **Functional binary download triggers:** Dedicated "Download .GLB" button
  triggers client-side binary blob extraction or `GLTFExporter` binary serialization
  (`model/gltf-binary`), providing instant offline download with visual button
  spinner feedback.

- **Parametric demo model generator:** Bundled procedural lathe vase geometry
  allows live 360-degree rotation, wireframe toggling, and downloads immediately
  upon page load without waiting for backend diffusion models.

#### UI/UX & Design Tokens

- **Neutral slate / zinc palette:** All components use Tailwind-aligned zinc/slate
  shades (`#090d16` background, `#0f172a` cards, `#1e293b` toolbars, `#334155` borders,
  `#94a3b8` labels, `#f8fafc` text). Flashy, distracting "vibe-coded" gradients
  were deliberately avoided in favor of high-contrast engineering clarity.

- **Responsive dual-viewport layout:** 50/50 split between input controls and
  3D output preview ensures parity of visual weight and seamless workflow.

#### Open items / handoffs for Phase 3

| Item | Owner | Target phase |
|------|-------|-------------|
| `engine/ai_engine.py` — complete diffusion pipeline & ControlNet conditioning | Mona | Phase 3 |
| Connect `ui/app.py` `btn_generate` output to WebGL viewer `loadGLB` | Mona + Laxmi | Phase 3 |
| Live session state persistence | Laxmi | Phase 3 |

---

*Sketch2Shape AI — TechZephyr 4.0 Hackathon · IIT Bhubaneswar*
