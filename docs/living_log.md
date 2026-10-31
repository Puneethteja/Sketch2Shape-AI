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

| Item | Owner | Target phase |
|------|-------|-------------|
| `engine/ai_engine.py` — implement `generate()` + `GenerationResult` | Mona | Phase 2 |
| `engine/mesh_utils.py` — implement `repair_mesh()` | Mona + Laxmi | Phase 2 |
| Three.js WebGL viewport integration | Laxmi | Phase 2 |
| Session state management (replace module-level dict) | Laxmi | Phase 2 |
| Gradio `gr.State` wiring for multi-user safety | Laxmi | Phase 2 |
| `gradio` added to `requirements.txt` | Puneeth / Laxmi | Immediate |

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

*Sketch2Shape AI — TechZephyr 4.0 Hackathon · IIT Bhubaneswar*
