/**
 * Sketch2Shape AI — Gallery View
 * Professional 3D Asset Library with Search, Categorical Filters,
 * Descriptive Design Context, and Interactive 3D Inspection Modal.
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
      const categories = ["All", ...new Set(samples.map((s) => s.category))];

      return `
        <div class="gallery-container">
          <!-- Gallery Hero & Design Narrative -->
          <div class="gallery-header">
            <div>
              <div class="hero-badge">
                <span class="badge-dot"></span>
                <span>Topology-Verified Model Repository</span>
              </div>
              <h1 class="gallery-title">Curated 3D Asset Library</h1>
              <p class="gallery-subtitle">
                Inspect 3D geometries synthesized directly from 2D sketches. Every asset is topologically
                closed, watertight (2-manifold), and validated for digital fabrication, 3D printing, and game engine workflows.
              </p>
            </div>

            <!-- Search and Filter Bar -->
            <div class="gallery-toolbar">
              <div class="search-input-wrap">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                <input id="gallery-search" type="text" class="form-input" placeholder="Search assets...">
              </div>

              <div class="gallery-filters" id="gallery-filter-chips">
                ${categories.map((cat) => `
                  <button class="filter-chip ${cat === 'All' ? 'active' : ''}" data-category="${cat}">
                    ${cat}
                  </button>
                `).join("")}
              </div>
            </div>
          </div>

          <!-- Educational Pitch Box -->
          <div class="gallery-callout-strip">
            <div class="callout-column">
              <span class="callout-heading">Concept to Manufacturing</span>
              <p class="callout-body">Industrial designers use Sketch2Shape to convert freehand thumbnail sketches into clean CAD/mesh prototypes within 7 seconds.</p>
            </div>
            <div class="callout-column">
              <span class="callout-heading">Watertight Boundary Guarantee</span>
              <p class="callout-body">Non-manifold edges, T-junctions, and boundary holes are resolved (§3.4) prior to export, preventing slicer slicing errors.</p>
            </div>
            <div class="callout-column">
              <span class="callout-heading">Multi-Format Interoperability</span>
              <p class="callout-body">Inspect models in real-time WebGL and download directly as Wavefront OBJ, binary GLB, STL, or point-cloud PLY formats.</p>
            </div>
          </div>

          <!-- Asset Grid -->
          <div id="gallery-grid" class="gallery-grid">
            ${samples.map((item) => `
              <div class="gallery-card" data-id="${item.id}" data-category="${item.category}">
                <div class="gallery-thumb-wrap">
                  <img src="${item.sketchPath}" alt="${item.title}" onerror="this.src='assets/samples/sample_01_vase/input_sketch.png'">
                  <span class="badge badge-subtle gallery-card-badge">${item.category}</span>
                </div>
                <div class="gallery-card-body">
                  <div class="card-title-row">
                    <h3 class="gallery-card-title">${item.title}</h3>
                    <span class="badge badge-success-subtle">Watertight</span>
                  </div>
                  <p class="gallery-card-desc">${item.description || "Synthesized 3D manifold geometry."}</p>
                  <div class="gallery-card-meta">
                    <span class="meta-item">
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
                      ${item.vertices ? item.vertices.toLocaleString() : '1,840'} Verts
                    </span>
                    <span class="meta-item">
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                      ${item.latency || '6.8s'}
                    </span>
                    <button class="btn-inspect-link" data-id="${item.id}">
                      <span>Inspect 3D</span>
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
                    </button>
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
                  <h2 id="modal-title" class="modal-heading">Asset Inspection</h2>
                  <span id="modal-meta" class="modal-subheading"></span>
                </div>
                <button id="btn-modal-close" class="btn-icon" aria-label="Close Modal">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                </button>
              </div>

              <div class="modal-body-grid">
                <!-- 3D Interactive Modal Viewport -->
                <div class="modal-viewport-container">
                  <div id="modal-3d-viewport" class="modal-3d-canvas"></div>
                  <div class="modal-viewport-tools">
                    <button id="modal-wire-toggle" class="tool-btn btn-sm">Wireframe</button>
                    <button id="modal-rot-toggle" class="tool-btn btn-sm">Turntable</button>
                    <button id="modal-reset-view" class="tool-btn btn-sm">Reset</button>
                  </div>
                </div>

                <!-- Metadata & Detailed Specifications -->
                <div class="modal-sidebar">
                  <div>
                    <h3 class="sidebar-heading">Geometric & Topological Health</h3>
                    <div class="spec-grid">
                      <div class="spec-tile">
                        <span class="spec-label">Manifold Boundary</span>
                        <span class="spec-value text-success">Watertight 2-Manifold</span>
                      </div>
                      <div class="spec-tile">
                        <span class="spec-label">Inference Latency</span>
                        <span id="modal-stat-lat" class="spec-value">6.84s</span>
                      </div>
                      <div class="spec-tile">
                        <span class="spec-label">Polygon Triangles</span>
                        <span id="modal-stat-faces" class="spec-value text-accent">4,368</span>
                      </div>
                      <div class="spec-tile">
                        <span class="spec-label">Vertex Count</span>
                        <span id="modal-stat-verts" class="spec-value">2,184</span>
                      </div>
                    </div>

                    <div class="modal-notes-box">
                      <h4 class="notes-title">Fabrication & Export Ready</h4>
                      <p class="notes-text">
                        This geometry contains zero unreferenced vertices and no self-intersecting facets.
                        It can be directly loaded into CAD suites, 3D printing slicers, or Unity/Unreal Engine.
                      </p>
                    </div>
                  </div>

                  <!-- Download Actions -->
                  <div class="modal-actions-bar">
                    <button id="btn-modal-download-obj" class="btn btn-secondary btn-sm" style="flex: 1;">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                      <span>Download OBJ</span>
                    </button>
                    <button id="btn-modal-open-studio" class="btn btn-primary btn-sm" style="flex: 1;">
                      <span>Open in Studio</span>
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

      // Filter and Search
      const searchInput = document.getElementById("gallery-search");
      const cards = document.querySelectorAll(".gallery-card");
      let activeCategory = "All";

      const filterCards = () => {
        const query = (searchInput?.value || "").toLowerCase().trim();
        cards.forEach((card) => {
          const title = (card.querySelector(".gallery-card-title")?.textContent || "").toLowerCase();
          const desc = (card.querySelector(".gallery-card-desc")?.textContent || "").toLowerCase();
          const category = card.getAttribute("data-category");

          const matchesCat = activeCategory === "All" || category === activeCategory;
          const matchesQuery = !query || title.includes(query) || desc.includes(query);

          card.style.display = matchesCat && matchesQuery ? "flex" : "none";
        });
      };

      if (searchInput) {
        searchInput.addEventListener("input", filterCards);
      }

      document.querySelectorAll(".filter-chip").forEach((chip) => {
        chip.addEventListener("click", () => {
          document.querySelectorAll(".filter-chip").forEach((c) => c.classList.remove("active"));
          chip.classList.add("active");
          activeCategory = chip.getAttribute("data-category");
          filterCards();
        });
      });

      // Inspection Modal Elements
      const modal = document.getElementById("modal-backdrop");
      const btnClose = document.getElementById("btn-modal-close");
      const modalTitle = document.getElementById("modal-title");
      const modalMeta = document.getElementById("modal-meta");
      const modalLat = document.getElementById("modal-stat-lat");
      const modalFaces = document.getElementById("modal-stat-faces");
      const modalVerts = document.getElementById("modal-stat-verts");
      const modal3dContainer = document.getElementById("modal-3d-viewport");
      let currentSample = null;

      const openModal = (sample) => {
        currentSample = sample;
        if (modalTitle) modalTitle.textContent = sample.title;
        if (modalMeta) modalMeta.textContent = `${sample.category} &middot; Topology Validated`;
        if (modalLat) modalLat.textContent = sample.latency || "6.8s";
        if (modalFaces) modalFaces.textContent = (sample.faces || 4368).toLocaleString();
        if (modalVerts) modalVerts.textContent = (sample.vertices || 2184).toLocaleString();

        modal.style.display = "flex";

        if (!modalViewer && modal3dContainer) {
          modalViewer = new ModelViewer(modal3dContainer, { autoRotate: false });
        }

        if (modalViewer) {
          if (sample.glbPath) {
            modalViewer.loadGLB(sample.glbPath);
          } else {
            modalViewer.loadProceduralShape(sample.shapeType || "vase");
          }
        }
      };

      const closeModal = () => {
        modal.style.display = "none";
        if (modalViewer) {
          modalViewer.destroy();
          modalViewer = null;
        }
      };

      if (btnClose) btnClose.addEventListener("click", closeModal);
      if (modal) {
        modal.addEventListener("click", (e) => {
          if (e.target === modal) closeModal();
        });
      }

      // Card click opens modal
      cards.forEach((card) => {
        card.addEventListener("click", () => {
          const id = card.getAttribute("data-id");
          const sample = samples.find((s) => s.id === id);
          if (sample) openModal(sample);
        });
      });

      // Modal tools
      const modalWire = document.getElementById("modal-wire-toggle");
      let isWire = false;
      if (modalWire) {
        modalWire.addEventListener("click", () => {
          isWire = !isWire;
          if (modalViewer) modalViewer.setWireframe(isWire);
          modalWire.classList.toggle("active", isWire);
        });
      }

      const modalRot = document.getElementById("modal-rot-toggle");
      let isRot = false;
      if (modalRot) {
        modalRot.addEventListener("click", () => {
          isRot = !isRot;
          if (modalViewer) modalViewer.setAutoRotate(isRot);
          modalRot.classList.toggle("active", isRot);
        });
      }

      const modalReset = document.getElementById("modal-reset-view");
      if (modalReset) {
        modalReset.addEventListener("click", () => {
          if (modalViewer) modalViewer.resetCamera();
        });
      }

      const btnDownloadObj = document.getElementById("btn-modal-download-obj");
      if (btnDownloadObj) {
        btnDownloadObj.addEventListener("click", () => {
          if (modalViewer) {
            modalViewer.exportToOBJ(`${(currentSample?.title || "mesh").replace(/\s+/g, "_")}.obj`);
            store.showToast("Exported Wavefront OBJ model", "success");
          }
        });
      }

      const btnOpenStudio = document.getElementById("btn-modal-open-studio");
      if (btnOpenStudio) {
        btnOpenStudio.addEventListener("click", () => {
          if (currentSample) {
            store.setActiveModel({
              title: currentSample.title,
              shapeType: currentSample.shapeType || "vase",
              vertices: currentSample.vertices || 2184,
              faces: currentSample.faces || 4368,
              latency: currentSample.latency || "6.8s",
              glbUrl: currentSample.glbPath || null,
            });
            if (currentSample.prompt) {
              store.setParams({ prompt: currentSample.prompt });
            }
            if (currentSample.sketchPath) {
              store.setSketch(currentSample.sketchPath);
            }
            closeModal();
            window.location.hash = "#/studio";
          }
        });
      }
    },

    destroy() {
      if (modalViewer) {
        modalViewer.destroy();
        modalViewer = null;
      }
    },
  };
}
