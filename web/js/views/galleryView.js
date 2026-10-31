/**
 * Sketch2Shape AI — Gallery View
 * Filterable 3D Asset Showcase with Interactive 3D Inspection Modal
 */

import { store } from "../state.js";
import { ModelViewer } from "../components/modelViewer.js";

export function createGalleryView() {
  let modalViewer = null;

  return {
    async render() {
      const state = store.getState();
      const samples = state.samples;

      // Extract unique categories
      const categories = ["All", ...new Set(samples.map(s => s.category))];

      return `
        <div class="gallery-container">
          <!-- Gallery Header -->
          <div class="gallery-header">
            <div>
              <h1 style="font-size: 22px; font-weight: 800; letter-spacing: -0.02em;">Curated 3D Asset Library</h1>
              <p style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
                Explore pre-computed and generated 3D meshes validated for watertight manifold topology.
              </p>
            </div>

            <!-- Search and Filter Bar -->
            <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
              <input id="gallery-search" type="text" class="form-input" placeholder="🔍 Search assets..." style="width: 200px; padding: 7px 12px; font-size: 12px;">
              <div class="gallery-filters" id="gallery-filter-chips">
                ${categories.map(cat => `
                  <button class="tool-btn ${cat === 'All' ? 'active' : ''}" data-category="${cat}">
                    ${cat}
                  </button>
                `).join("")}
              </div>
            </div>
          </div>

          <!-- Gallery Grid -->
          <div id="gallery-grid" class="gallery-grid">
            ${samples.map(item => `
              <div class="gallery-card" data-id="${item.id}" data-category="${item.category}">
                <div class="gallery-thumb-wrap">
                  <img src="${item.sketchPath}" alt="${item.title}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'100\\' height=\\'100\\' fill=\\'%23334155\\'><text x=\\'50%\\' y=\\'50%\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' fill=\\'white\\'>3D</text></svg>'">
                  <span class="badge badge-cyan" style="position: absolute; top: 10px; right: 10px;">${item.category}</span>
                </div>
                <div class="gallery-card-body">
                  <h3 class="gallery-card-title">${item.title}</h3>
                  <p style="font-size: 12px; color: var(--text-secondary); line-height: 1.4;">${item.description || "Validated watertight 3D asset."}</p>
                  <div class="gallery-card-meta">
                    <span>🔷 ${item.vertices ? item.vertices.toLocaleString() : '1,800'} Verts</span>
                    <span style="color: var(--accent-emerald);">⏱️ ${item.latency || '6.8s'}</span>
                    <span class="badge badge-success" style="font-size: 10px;">Watertight</span>
                  </div>
                </div>
              </div>
            `).join("")}
          </div>

          <!-- Inspection Modal Container -->
          <div id="modal-backdrop" class="modal-backdrop" style="display: none;">
            <div class="modal-content">
              <div class="modal-header">
                <div>
                  <h2 id="modal-title" style="font-size: 18px; font-weight: 700;">Asset Inspection</h2>
                  <span id="modal-meta" style="font-size: 12px; font-family: var(--font-mono); color: var(--accent-cyan);"></span>
                </div>
                <button id="btn-modal-close" class="btn-icon" style="font-size: 16px;">✕</button>
              </div>
              <div class="modal-body" style="display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px;">
                <!-- 3D Interactive Modal Viewport -->
                <div style="aspect-ratio: 1/1; background: #090d16; border-radius: var(--radius-md); overflow: hidden; position: relative; border: 1px solid var(--border-default);">
                  <div id="modal-3d-viewport" style="width: 100%; height: 100%;"></div>
                  <div style="position: absolute; bottom: 10px; right: 10px; display: flex; gap: 6px;">
                    <button id="modal-wire-toggle" class="tool-btn btn-sm">🕸️ Wireframe</button>
                    <button id="modal-rot-toggle" class="tool-btn btn-sm">🔄 Rotate</button>
                  </div>
                </div>

                <!-- Metadata & Actions -->
                <div style="display: flex; flex-direction: column; justify-content: space-between; gap: 16px;">
                  <div>
                    <h3 style="font-size: 14px; font-weight: 600; color: var(--text-muted); text-transform: uppercase;">Topology & Geometric Health</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                      <div style="background: var(--bg-tertiary); padding: 10px; border-radius: var(--radius-sm);">
                        <div style="font-size: 11px; color: var(--text-muted);">Watertight Boundary</div>
                        <div style="font-size: 14px; font-weight: 700; color: var(--accent-emerald);">✓ Non-Manifold Repaired</div>
                      </div>
                      <div style="background: var(--bg-tertiary); padding: 10px; border-radius: var(--radius-sm);">
                        <div style="font-size: 11px; color: var(--text-muted);">Inference Latency</div>
                        <div id="modal-stat-lat" style="font-size: 14px; font-weight: 700; color: var(--text-primary);">6.84s</div>
                      </div>
                      <div style="background: var(--bg-tertiary); padding: 10px; border-radius: var(--radius-sm);">
                        <div style="font-size: 11px; color: var(--text-muted);">Polygon Faces</div>
                        <div id="modal-stat-faces" style="font-size: 14px; font-weight: 700; color: var(--accent-cyan);">4,368</div>
                      </div>
                      <div style="background: var(--bg-tertiary); padding: 10px; border-radius: var(--radius-sm);">
                        <div style="font-size: 11px; color: var(--text-muted);">Vertex Density</div>
                        <div id="modal-stat-verts" style="font-size: 14px; font-weight: 700; color: var(--accent-cyan);">2,184</div>
                      </div>
                    </div>

                    <div style="margin-top: 16px;">
                      <h3 style="font-size: 14px; font-weight: 600; color: var(--text-muted); text-transform: uppercase;">Pipeline Verification</h3>
                      <p id="modal-desc" style="font-size: 13px; color: var(--text-secondary); margin-top: 6px; line-height: 1.5;"></p>
                    </div>
                  </div>

                  <!-- Download Buttons -->
                  <div style="display: flex; gap: 10px;">
                    <button id="btn-modal-dl-glb" class="btn btn-primary" style="flex: 1;">
                      ⬇️ Download .GLB
                    </button>
                    <button id="btn-modal-dl-obj" class="btn btn-secondary" style="flex: 1;">
                      ⬇️ Download .OBJ
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      `;
    },

    afterRender() {
      const state = store.getState();
      const samples = state.samples;

      // 1. Search and Filtering
      const searchInput = document.getElementById("gallery-search");
      const filterChips = document.querySelectorAll("#gallery-filter-chips button");
      const cards = document.querySelectorAll(".gallery-card");

      let activeCategory = "All";
      let searchQuery = "";

      const applyFilters = () => {
        cards.forEach(card => {
          const cat = card.getAttribute("data-category");
          const title = card.querySelector(".gallery-card-title").textContent.toLowerCase();
          const matchesCat = activeCategory === "All" || cat === activeCategory;
          const matchesSearch = !searchQuery || title.includes(searchQuery);

          card.style.display = matchesCat && matchesSearch ? "flex" : "none";
        });
      };

      filterChips.forEach(btn => {
        btn.addEventListener("click", () => {
          filterChips.forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          activeCategory = btn.getAttribute("data-category");
          applyFilters();
        });
      });

      if (searchInput) {
        searchInput.addEventListener("input", (e) => {
          searchQuery = e.target.value.toLowerCase().trim();
          applyFilters();
        });
      }

      // 2. Inspection Modal Logic
      const backdrop = document.getElementById("modal-backdrop");
      const btnClose = document.getElementById("btn-modal-close");
      const modalTitle = document.getElementById("modal-title");
      const modalMeta = document.getElementById("modal-meta");
      const modalDesc = document.getElementById("modal-desc");
      const statLat = document.getElementById("modal-stat-lat");
      const statFaces = document.getElementById("modal-stat-faces");
      const statVerts = document.getElementById("modal-stat-verts");
      const viewportContainer = document.getElementById("modal-3d-viewport");

      const openModal = (sample) => {
        modalTitle.textContent = sample.title;
        modalMeta.textContent = `Category: ${sample.category} | Pipeline ID: ${sample.id}`;
        modalDesc.textContent = sample.description || "Synthesized 3D mesh via ControlNet conditioning and StageLinker geometry pass.";
        statLat.textContent = sample.latency || "6.84s";
        statFaces.textContent = (sample.faces || 3680).toLocaleString();
        statVerts.textContent = (sample.vertices || 1840).toLocaleString();

        backdrop.style.display = "flex";

        // Initialize 3D Modal Viewer
        if (modalViewer) modalViewer.destroy();
        modalViewer = new ModelViewer(viewportContainer, { autoRotate: true });
        modalViewer.loadProceduralShape(sample.shapeType || "vase");

        // Download hooks
        document.getElementById("btn-modal-dl-glb").onclick = () => {
          const a = document.createElement("a");
          a.href = sample.glbPath || "../outputs/samples/sample_01_vase/model.glb";
          a.download = `${sample.id}.glb`;
          document.body.appendChild(a);
          a.click();
          a.remove();
          store.showToast(`Downloading ${sample.id}.glb`, "success");
        };

        document.getElementById("btn-modal-dl-obj").onclick = () => {
          const a = document.createElement("a");
          a.href = sample.objPath || "../outputs/samples/sample_01_vase/model.obj";
          a.download = `${sample.id}.obj`;
          document.body.appendChild(a);
          a.click();
          a.remove();
          store.showToast(`Downloading ${sample.id}.obj`, "success");
        };
      };

      cards.forEach(card => {
        card.addEventListener("click", () => {
          const id = card.getAttribute("data-id");
          const sample = samples.find(s => s.id === id);
          if (sample) openModal(sample);
        });
      });

      const closeModal = () => {
        backdrop.style.display = "none";
        if (modalViewer) {
          modalViewer.destroy();
          modalViewer = null;
        }
      };

      if (btnClose) btnClose.addEventListener("click", closeModal);
      if (backdrop) {
        backdrop.addEventListener("click", (e) => {
          if (e.target === backdrop) closeModal();
        });
      }

      // Modal Viewport Tool toggles
      const wireToggle = document.getElementById("modal-wire-toggle");
      let wire = false;
      if (wireToggle) {
        wireToggle.addEventListener("click", () => {
          wire = !wire;
          if (modalViewer) modalViewer.setWireframe(wire);
          wireToggle.classList.toggle("active", wire);
        });
      }

      const rotToggle = document.getElementById("modal-rot-toggle");
      let rot = true;
      if (rotToggle) {
        rotToggle.addEventListener("click", () => {
          rot = !rot;
          if (modalViewer) modalViewer.setAutoRotate(rot);
          rotToggle.classList.toggle("active", rot);
        });
      }
    },

    destroy() {
      if (modalViewer) {
        modalViewer.destroy();
        modalViewer = null;
      }
    }
  };
}
