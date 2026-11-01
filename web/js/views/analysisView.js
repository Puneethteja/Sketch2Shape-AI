import { store } from "../state.js";

const safe = (value) => String(value ?? "Unavailable").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));

export function createAnalysisView() {
  let selectedId = null;
  return {
    async render() {
      const models = store.getState().samples.filter(item => item.id.startsWith("gen_") || item.id.startsWith("variation_"));
      const selected = models.find(item => item.id === selectedId) || models[models.length - 1];
      const analysis = selected?.analysis || {};
      const metrics = [
        ["Mesh integrity", analysis.mesh_integrity],
        ["Shape similarity (silhouette IoU)", analysis.shape_similarity],
        ["Geometry completeness", null],
        ["Structural consistency", null]
      ];
      return `<div class="benchmarks-container">
        <h1 style="font-size:24px;font-weight:800">Model &amp; Metrics</h1>
        <p style="color:var(--text-secondary);margin:6px 0 18px">Inspection results come from the generated mesh and its source image. Unavailable measurements are left blank.</p>
        <label class="form-label" for="analysis-model-select">Generated model</label>
        <select id="analysis-model-select" class="form-select" style="max-width:420px">${models.map(item=>`<option value="${safe(item.id)}">${safe(item.title)} — ${safe(item.category)}</option>`).join("")}</select>
        ${selected ? `<section class="panel-card" style="margin-top:18px;padding:18px">
          <h2>${safe(selected.title)}</h2><p>${safe(selected.category)} · ${safe(selected.createdAt || "Generation time unavailable")}</p>
          <p>Validity score: <strong>${analysis.validity_score == null ? "Unavailable" : `${safe(analysis.validity_score)}/100`}</strong></p>
          <div class="mesh-stats-bar">${metrics.map(([name,value])=>`<div class="stat-pill"><span class="stat-pill-label">${safe(name)}</span><span class="stat-pill-val">${value == null ? "Unavailable" : `${safe(value)}/100`}</span></div>`).join("")}</div>
          <p>Vertices: ${safe(analysis.vertices ?? selected.vertices)} · Faces: ${safe(analysis.faces ?? selected.faces)} · Dimensions: ${safe(analysis.dimensions?.join(" × ") || "Unavailable")}</p>
          <h3>Connected mesh parts</h3><p>These are geometric connected components; semantic labels are not inferred.</p>
          ${(analysis.parts || []).length ? `<ul>${analysis.parts.map(part=>`<li>${safe(part.name)} — ${safe(part.vertices)} vertices, ${safe(part.faces)} faces; bounds ${safe(part.dimensions?.join(" × "))}</li>`).join("")}</ul>` : `<p>${analysis.connected_components === 1 ? "The mesh is a single connected surface; semantic parts cannot be separated from this geometry." : "Component analysis unavailable."}</p>`}
          <h3>Issues</h3><p>${(analysis.issues || []).length ? analysis.issues.map(safe).join(" ") : "No mesh issues were detected by the available checks."}</p>
          <p>Export formats supported by the current generation endpoint: GLB, OBJ and STL. Export validation: ${analysis.available ? "mesh topology inspected" : "Unavailable"}.</p>
        </section>` : `<div class="panel-card" style="margin-top:18px;padding:18px">Generate and save a model in Studio to view its analysis here. No sample or placeholder analysis is shown.</div>`}
      </div>`;
    },
    afterRender() {
      const select = document.getElementById("analysis-model-select");
      if (select) select.addEventListener("change", async () => {
        selectedId = select.value;
        document.getElementById("view-container").innerHTML = await this.render();
        this.afterRender();
      });
    }
  };
}
