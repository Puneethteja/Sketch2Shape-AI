/**
 * Sketch2Shape AI — Architecture & Pipeline View
 * Visualized End-to-End Deep Learning Architecture & StageLinker Dataflow
 */

export function createPipelineView() {
  const STAGES = [
    {
      num: "01",
      name: "Sketch Preprocessing & Lineart Ingestion",
      component: "engine/preprocessing.py",
      latency: "0.04s",
      vram: "0 MB",
      desc: "Universal ingestion of base64/files, bilateral edge-preserving smoothing, and adaptive morphological contour thresholding to produce clean single-channel scribble conditioning masks."
    },
    {
      num: "02",
      name: "ControlNet Conditioned Diffusion Pass",
      component: "engine/generator.py",
      latency: "4.82s",
      vram: "3,840 MB (FP16)",
      desc: "Stable Diffusion v1.5 conditioned via ControlNet Scribble adapters with calibrated scale (0.75-0.85) to generate 512×512 photorealistic object renders adhering to sketch contours."
    },
    {
      num: "2.5",
      name: "StageLinker Shading & Depth Tensor Extraction",
      component: "engine/linker.py & utils/segmentation.py",
      latency: "0.12s",
      vram: "0 MB (VRAM Evacuated)",
      desc: "Zero-copy tensor handoff extracting multi-frequency lighting gradients, distance-transform radial depth profiles, and foreground alpha segmentation via U2Net before flushing GPU VRAM."
    },
    {
      num: "03",
      name: "Volumetric 3D Reconstruction Backbone",
      component: "engine/reconstruction.py",
      latency: "1.45s",
      vram: "4,120 MB (FP16)",
      desc: "Deep learning 3D geometric synthesis mapping 2D multi-view shading tensors into signed distance fields and extracting dense manifold polygon meshes."
    },
    {
      num: "04",
      name: "Non-Manifold Mesh Repair Sequence",
      component: "utils/mesh_utils.py",
      latency: "0.35s",
      vram: "0 MB (CPU Trimesh)",
      desc: "Robust repair pipeline resolving degenerate triangles, closing non-manifold boundary holes, stripping unattached floating fragments, and centering geometry to produce valid printable assets."
    }
  ];

  return {
    async render() {
      return `
        <div class="pipeline-container">
          <!-- Header -->
          <div style="margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <span class="badge badge-cyan">System Architecture</span>
              <span class="badge badge-success">FP16 Half-Precision</span>
              <span class="badge badge-purple">Sequential CPU Offloading</span>
            </div>
            <h1 style="font-size: 24px; font-weight: 800; letter-spacing: -0.02em; margin-top: 8px;">
              AI Pipeline & StageLinker Dataflow
            </h1>
            <p style="font-size: 13px; color: var(--text-secondary); margin-top: 4px; max-width: 800px;">
              Sketch2Shape AI links generative 2D diffusion with deep 3D reconstruction using zero-copy tensor passing and sequential VRAM cache evacuation to run on budget 16GB GPUs in under 35 seconds.
            </p>
          </div>

          <!-- Interactive Pipeline Stages List -->
          <div class="pipeline-stages-list">
            ${STAGES.map((s, idx) => `
              <div class="pipeline-stage-card">
                <div class="stage-step-num">${s.num}</div>
                <div>
                  <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <h3 style="font-size: 16px; font-weight: 700;">${s.name}</h3>
                    <code style="font-size: 11px; background: var(--bg-tertiary); padding: 2px 6px; border-radius: 4px; color: var(--accent-cyan); font-family: var(--font-mono);">${s.component}</code>
                  </div>
                  <p style="font-size: 12px; color: var(--text-secondary); margin-top: 6px; line-height: 1.5;">${s.desc}</p>
                </div>
                <div style="text-align: right; min-width: 130px; font-family: var(--font-mono);">
                  <div style="font-size: 13px; font-weight: 700; color: var(--accent-emerald);">⏱️ ${s.latency}</div>
                  <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">VRAM: ${s.vram}</div>
                </div>
              </div>
            `).join("")}
          </div>

          <!-- Tensor Flow Specs Card -->
          <div style="margin-top: 28px; background: var(--bg-card); border: 1px solid var(--border-default); border-radius: var(--radius-lg); padding: 24px;">
            <h3 style="font-size: 16px; font-weight: 700; margin-bottom: 12px;">StageLinker Inter-Process Communication Contract</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px;">
              <div style="background: var(--bg-secondary); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
                <span style="font-size: 11px; color: var(--accent-cyan); font-weight: 600;">STAGE 1 ➔ STAGE 2</span>
                <div style="font-size: 13px; font-family: var(--font-mono); font-weight: 600; margin-top: 4px;">torch.Tensor (1, 3, 512, 512)</div>
                <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Normalized FP16 Scribble conditioning tensor [0.0, 1.0].</div>
              </div>

              <div style="background: var(--bg-secondary); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
                <span style="font-size: 11px; color: var(--accent-cyan); font-weight: 600;">STAGE 2 ➔ STAGELINKER</span>
                <div style="font-size: 13px; font-family: var(--font-mono); font-weight: 600; margin-top: 4px;">PIL.Image (512, 512, 3)</div>
                <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Diffusion product render + Concrete random seed.</div>
              </div>

              <div style="background: var(--bg-secondary); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
                <span style="font-size: 11px; color: var(--accent-cyan); font-weight: 600;">STAGELINKER ➔ STAGE 3</span>
                <div style="font-size: 13px; font-family: var(--font-mono); font-weight: 600; margin-top: 4px;">StageHandoff Dataclass</div>
                <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Fused distance-transform depth prior + alpha cutout.</div>
              </div>

              <div style="background: var(--bg-secondary); padding: 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
                <span style="font-size: 11px; color: var(--accent-cyan); font-weight: 600;">STAGE 4 ➔ OUTPUTS</span>
                <div style="font-size: 13px; font-family: var(--font-mono); font-weight: 600; margin-top: 4px;">trimesh.Trimesh / GLB</div>
                <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Verified closed-manifold mesh binary serialization.</div>
              </div>
            </div>
          </div>
        </div>
      `;
    }
  };
}
