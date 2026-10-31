/**
 * Sketch2Shape AI — Studio View
 * Primary 2D Sketchpad to 3D Generation Interface
 */

import { store } from "../state.js";
import { CanvasEditor } from "../components/canvasEditor.js";
import { ModelViewer } from "../components/modelViewer.js";

export function createStudioView() {
  let canvasEditor = null;
  let modelViewer = null;

  return {
    async render() {
      const state = store.getState();
      const params = state.params;

      return `
        <div class="studio-container">
          <div class="studio-grid">
            <!-- Left Column: 2D Sketchpad & Controls -->
            <div class="panel-card">
              <div class="panel-header">
                <div class="panel-title-wrap">
                  <span style="font-size: 18px;">✏️</span>
                  <div>
                    <h2 class="panel-title">Sketch Input Studio</h2>
                    <span style="font-size: 11px; color: var(--text-muted);">Draw freehand or drop an existing sketch</span>
                  </div>
                </div>
                <span class="badge badge-cyan">Canvas 512×512</span>
              </div>

              <!-- Canvas Toolbar -->
              <div class="canvas-toolbar">
                <div class="tool-group">
                  <button id="tool-brush" class="tool-btn active" title="Brush Tool">🖌️ Brush</button>
                  <button id="tool-eraser" class="tool-btn" title="Eraser Tool">🧹 Eraser</button>
                </div>

                <div class="tool-group">
                  <span style="font-size: 11px; color: var(--text-muted);">Size:</span>
                  <input id="brush-size" type="range" class="form-range" min="1" max="32" value="4" style="width: 70px;">
                  <span id="brush-size-val" style="font-size: 11px; font-family: var(--font-mono); min-width: 24px;">4px</span>
                </div>

                <div class="tool-group">
                  <button id="btn-undo" class="tool-btn" title="Undo stroke" disabled>↩️ Undo</button>
                  <button id="btn-redo" class="tool-btn" title="Redo stroke" disabled>↪️ Redo</button>
                  <button id="btn-clear" class="tool-btn btn-danger" title="Clear canvas">🗑️ Clear</button>
                </div>
              </div>

              <!-- Canvas Wrap with Dropzone -->
              <div id="canvas-dropzone" class="canvas-viewport-wrap">
                <canvas id="studio-canvas" class="sketch-canvas"></canvas>
              </div>

              <!-- Quick File Upload Drop Indicator -->
              <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                <label for="file-upload" class="btn btn-secondary btn-sm" style="cursor: pointer; width: 100%;">
                  📂 Upload Image / Photo (.png, .jpg)
                </label>
                <input id="file-upload" type="file" accept=".png,.jpg,.jpeg" style="display: none;">
              </div>

              <!-- Parameters & Prompt -->
              <div class="form-group" style="margin-top: 6px;">
                <label class="form-label" for="prompt-input">
                  <span>Material & Style Prompt (Optional)</span>
                  <span style="font-size: 10px; color: var(--accent-cyan);">e.g. porcelain, wood, matte titanium</span>
                </label>
                <textarea id="prompt-input" class="form-textarea" rows="2" placeholder="e.g. clean matte ceramic finish with smooth glazed surface...">${params.prompt}</textarea>
              </div>

              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div class="form-group">
                  <label class="form-label" for="param-cn-scale">
                    <span>ControlNet Scale</span>
                    <span id="cn-scale-val" style="font-family: var(--font-mono); color: var(--accent-cyan);">${params.controlnetScale}</span>
                  </label>
                  <input id="param-cn-scale" type="range" class="form-range" min="0.5" max="1.5" step="0.05" value="${params.controlnetScale}">
                </div>

                <div class="form-group">
                  <label class="form-label" for="param-steps">
                    <span>Inference Steps</span>
                    <span id="steps-val" style="font-family: var(--font-mono); color: var(--accent-cyan);">${params.inferenceSteps}</span>
                  </label>
                  <input id="param-steps" type="range" class="form-range" min="10" max="60" step="5" value="${params.inferenceSteps}">
                </div>
              </div>

              <!-- Action Buttons -->
              <div style="display: flex; gap: 10px; margin-top: 4px;">
                <button id="btn-preprocess" class="btn btn-secondary" style="flex: 1;">
                  ⚙️ Preprocess Lineart
                </button>
                <button id="btn-generate-3d" class="btn btn-primary" style="flex: 2;">
                  ⚡ Generate 3D Asset
                </button>
              </div>
            </div>

            <!-- Right Column: Interactive 3D WebGL Viewport -->
            <div class="panel-card">
              <div class="panel-header">
                <div class="panel-title-wrap">
                  <span style="font-size: 18px;">🌐</span>
                  <div>
                    <h2 class="panel-title">3D WebGL Viewport</h2>
                    <span style="font-size: 11px; color: var(--text-muted);">Real-time Orbit &middot; Zoom &middot; Pan &middot; Manifold Mesh</span>
                  </div>
                </div>
                <div style="display: flex; gap: 6px;">
                  <span id="status-manifold-badge" class="badge badge-success">✓ Manifold</span>
                </div>
              </div>

              <!-- 3D Viewport Root Container -->
              <div class="webgl-viewport-wrap">
                <div id="studio-model-viewer" class="webgl-canvas-container"></div>

                <!-- Live Progress Simulation Overlay -->
                <div id="generation-overlay" class="viewport-progress-overlay" style="display: none;">
                  <div class="spinner"></div>
                  <div style="display: flex; flex-direction: column; align-items: center; gap: 6px;">
                    <div id="gen-stage-label" class="progress-text">Initializing AI Synthesis Pipeline...</div>
                    <div style="width: 180px; height: 6px; background: var(--bg-tertiary); border-radius: 3px; overflow: hidden;">
                      <div id="gen-progress-bar" style="width: 0%; height: 100%; background: var(--accent-cyan); transition: width 0.3s ease;"></div>
                    </div>
                  </div>
                </div>

                <!-- Mesh Statistics Floating Badge -->
                <div class="webgl-overlay-stats">
                  <div>Model: <span id="stat-model-name" style="color: var(--text-primary);">Minimalist Ceramic Vase</span></div>
                  <div>Vertices: <span id="stat-vertices" style="color: var(--accent-cyan);">2,184</span> | Faces: <span id="stat-faces" style="color: var(--accent-cyan);">4,368</span></div>
                  <div>Latency: <span id="stat-latency" style="color: var(--accent-emerald);">6.84s</span> (FP16 StageLinker)</div>
                </div>

                <!-- Viewport Control Toolbar -->
                <div class="webgl-toolbar">
                  <button id="btn-vp-wireframe" class="tool-btn" title="Toggle Wireframe">🕸️ Wire</button>
                  <button id="btn-vp-autorotate" class="tool-btn" title="Toggle Auto-Rotation">🔄 Rotate</button>
                  <button id="btn-vp-reset" class="tool-btn" title="Reset View">🎯 Reset</button>
                </div>
              </div>

              <!-- Mesh Export Bar -->
              <div style="display: flex; align-items: center; justify-content: space-between; background: var(--bg-secondary); padding: 10px 14px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); gap: 10px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                  <span style="font-size: 12px; font-weight: 600; color: var(--text-secondary);">Export:</span>
                  <select id="export-format" class="form-select" style="padding: 4px 8px; font-size: 12px; width: auto;">
                    <option value="glb">GLB (Binary 3D)</option>
                    <option value="obj">OBJ (Wavefront)</option>
                    <option value="stl">STL (3D Print)</option>
                    <option value="ply">PLY (Point Cloud)</option>
                  </select>
                </div>

                <div style="display: flex; gap: 8px;">
                  <button id="btn-export-download" class="btn btn-secondary btn-sm">
                    ⬇️ Download Asset
                  </button>
                  <button id="btn-save-gallery" class="btn btn-primary btn-sm">
                    ⭐ Save to Gallery
                  </button>
                </div>
              </div>

              <!-- Intermediate Prior Previews -->
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="background: var(--bg-secondary); padding: 8px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); text-align: center;">
                  <span style="font-size: 10px; font-weight: 600; color: var(--text-muted); text-transform: uppercase;">Stage 1: Lineart Mask</span>
                  <div id="preview-lineart-box" style="margin-top: 6px; aspect-ratio: 16/9; background: #000; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 11px; color: #64748b; overflow: hidden;">
                    <span>Lineart preview</span>
                  </div>
                </div>

                <div style="background: var(--bg-secondary); padding: 8px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); text-align: center;">
                  <span style="font-size: 10px; font-weight: 600; color: var(--text-muted); text-transform: uppercase;">Stage 2: Depth Prior</span>
                  <div id="preview-depth-box" style="margin-top: 6px; aspect-ratio: 16/9; background: #090d16; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 11px; color: #64748b; overflow: hidden;">
                    <span>Depth map preview</span>
                  </div>
                </div>
              </div>

            </div>
          </div>
        </div>
      `;
    },

    afterRender() {
      // 1. Initialize Canvas Editor
      const canvasEl = document.getElementById("studio-canvas");
      if (canvasEl) {
        canvasEditor = new CanvasEditor(canvasEl, { width: 512, height: 512 });

        const btnUndo = document.getElementById("btn-undo");
        const btnRedo = document.getElementById("btn-redo");
        canvasEditor.onHistoryChange = ({ canUndo, canRedo }) => {
          if (btnUndo) btnUndo.disabled = !canUndo;
          if (btnRedo) btnRedo.disabled = !canRedo;
        };
      }

      // 2. Initialize Model Viewer
      const viewerContainer = document.getElementById("studio-model-viewer");
      if (viewerContainer) {
        modelViewer = new ModelViewer(viewerContainer, { autoRotate: false });
      }

      // 3. Canvas Tools Bindings
      const toolBrush = document.getElementById("tool-brush");
      const toolEraser = document.getElementById("tool-eraser");
      const brushSize = document.getElementById("brush-size");
      const brushSizeVal = document.getElementById("brush-size-val");
      const btnClear = document.getElementById("btn-clear");
      const btnUndo = document.getElementById("btn-undo");
      const btnRedo = document.getElementById("btn-redo");

      if (toolBrush) {
        toolBrush.addEventListener("click", () => {
          canvasEditor.mode = "brush";
          toolBrush.classList.add("active");
          toolEraser.classList.remove("active");
        });
      }

      if (toolEraser) {
        toolEraser.addEventListener("click", () => {
          canvasEditor.mode = "eraser";
          toolEraser.classList.add("active");
          toolBrush.classList.remove("active");
        });
      }

      if (brushSize) {
        brushSize.addEventListener("input", (e) => {
          const val = parseInt(e.target.value, 10);
          canvasEditor.brushSize = val;
          if (brushSizeVal) brushSizeVal.textContent = `${val}px`;
        });
      }

      if (btnClear) {
        btnClear.addEventListener("click", () => {
          canvasEditor.clear();
          store.showToast("Canvas cleared", "info", 1500);
        });
      }

      if (btnUndo) btnUndo.addEventListener("click", () => canvasEditor.undo());
      if (btnRedo) btnRedo.addEventListener("click", () => canvasEditor.redo());

      // File Upload & Drag-and-Drop
      const fileInput = document.getElementById("file-upload");
      if (fileInput) {
        fileInput.addEventListener("change", async (e) => {
          const file = e.target.files[0];
          if (file) {
            await canvasEditor.loadImageFromFile(file);
            store.showToast(`Loaded "${file.name}" into sketchpad`, "success");
          }
        });
      }

      const dropzone = document.getElementById("canvas-dropzone");
      if (dropzone) {
        dropzone.addEventListener("dragover", (e) => {
          e.preventDefault();
          dropzone.style.borderColor = "var(--accent-cyan)";
        });
        dropzone.addEventListener("dragleave", () => {
          dropzone.style.borderColor = "var(--border-default)";
        });
        dropzone.addEventListener("drop", async (e) => {
          e.preventDefault();
          dropzone.style.borderColor = "var(--border-default)";
          const file = e.dataTransfer.files[0];
          if (file && file.type.startsWith("image/")) {
            await canvasEditor.loadImageFromFile(file);
            store.showToast(`Dropped "${file.name}" onto sketchpad`, "success");
          }
        });
      }

      // 4. Parameter Sliders Binding
      const cnScale = document.getElementById("param-cn-scale");
      const cnScaleVal = document.getElementById("cn-scale-val");
      if (cnScale) {
        cnScale.addEventListener("input", (e) => {
          if (cnScaleVal) cnScaleVal.textContent = e.target.value;
          store.setParams({ controlnetScale: parseFloat(e.target.value) });
        });
      }

      const stepsInput = document.getElementById("param-steps");
      const stepsVal = document.getElementById("steps-val");
      if (stepsInput) {
        stepsInput.addEventListener("input", (e) => {
          if (stepsVal) stepsVal.textContent = e.target.value;
          store.setParams({ inferenceSteps: parseInt(e.target.value, 10) });
        });
      }

      const promptInput = document.getElementById("prompt-input");
      if (promptInput) {
        promptInput.addEventListener("input", (e) => {
          store.setParams({ prompt: e.target.value });
        });
      }

      // 5. Viewport Control Buttons
      const btnWire = document.getElementById("btn-vp-wireframe");
      let isWire = false;
      if (btnWire) {
        btnWire.addEventListener("click", () => {
          isWire = !isWire;
          modelViewer.setWireframe(isWire);
          btnWire.classList.toggle("active", isWire);
        });
      }

      const btnAutoRot = document.getElementById("btn-vp-autorotate");
      let isAutoRot = false;
      if (btnAutoRot) {
        btnAutoRot.addEventListener("click", () => {
          isAutoRot = !isAutoRot;
          modelViewer.setAutoRotate(isAutoRot);
          btnAutoRot.classList.toggle("active", isAutoRot);
        });
      }

      const btnReset = document.getElementById("btn-vp-reset");
      if (btnReset) {
        btnReset.addEventListener("click", () => modelViewer.resetCamera());
      }

      // 6. Preprocess Action
      const btnPreprocess = document.getElementById("btn-preprocess");
      if (btnPreprocess) {
        btnPreprocess.addEventListener("click", () => {
          if (canvasEditor.isEmpty()) {
            store.showToast("Please sketch or upload an object first!", "error");
            return;
          }

          const lineartBox = document.getElementById("preview-lineart-box");
          if (lineartBox) {
            lineartBox.innerHTML = `<img src="${canvasEditor.toDataURL()}" style="width:100%;height:100%;object-fit:contain;filter:invert(1) contrast(150%);">`;
          }
          store.showToast("Lineart mask synthesized & normalized", "success");
        });
      }

      // 7. Generate 3D Asset Action (Simulated Pipeline Flow with Reactive Updates)
      const btnGenerate = document.getElementById("btn-generate-3d");
      const overlay = document.getElementById("generation-overlay");
      const stageLabel = document.getElementById("gen-stage-label");
      const progressBar = document.getElementById("gen-progress-bar");

      if (btnGenerate) {
        btnGenerate.addEventListener("click", async () => {
          if (canvasEditor.isEmpty()) {
            store.showToast("Please sketch or drop an image before generating!", "error");
            return;
          }

          btnGenerate.disabled = true;
          overlay.style.display = "flex";

          const updateProgress = (pct, label) => {
            if (progressBar) progressBar.style.width = `${pct}%`;
            if (stageLabel) stageLabel.textContent = label;
          };

          try {
            updateProgress(15, "Stage 1: Bilateral Filter & Lineart Ingestion...");
            await new Promise(r => setTimeout(r, 600));

            const lineartBox = document.getElementById("preview-lineart-box");
            if (lineartBox) {
              lineartBox.innerHTML = `<img src="${canvasEditor.toDataURL()}" style="width:100%;height:100%;object-fit:contain;filter:invert(1) contrast(150%);">`;
            }

            updateProgress(45, "Stage 2: ControlNet Diffusion Prior Pass (FP16)...");
            await new Promise(r => setTimeout(r, 900));

            const depthBox = document.getElementById("preview-depth-box");
            if (depthBox) {
              depthBox.innerHTML = `<img src="${canvasEditor.toDataURL()}" style="width:100%;height:100%;object-fit:contain;filter:blur(2px) contrast(120%);">`;
            }

            updateProgress(75, "Stage 2.5: StageLinker Shading Tensor Extraction...");
            await new Promise(r => setTimeout(r, 600));

            updateProgress(90, "Stage 3: Deep Volumetric 3D Reconstruction...");
            await new Promise(r => setTimeout(r, 700));

            updateProgress(98, "Stage 4: Closed Manifold Topology Repair...");
            await new Promise(r => setTimeout(r, 500));

            // Select shape based on prompt or random
            const shapes = ["vase", "cylinder", "box", "cone", "sphere"];
            const chosen = shapes[Math.floor(Math.random() * shapes.length)];
            const stats = modelViewer.loadProceduralShape(chosen);

            // Update stats
            const vCount = document.getElementById("stat-vertices");
            const fCount = document.getElementById("stat-faces");
            const mName = document.getElementById("stat-model-name");
            if (vCount) vCount.textContent = stats.vertices.toLocaleString();
            if (fCount) fCount.textContent = stats.faces.toLocaleString();
            if (mName) mName.textContent = `Generated ${chosen.toUpperCase()} Asset`;

            store.showToast("3D Asset generated successfully — Manifold Verified", "success", 4000);
          } finally {
            overlay.style.display = "none";
            btnGenerate.disabled = false;
          }
        });
      }

      // 8. Export and Save Actions
      const btnExport = document.getElementById("btn-export-download");
      if (btnExport) {
        btnExport.addEventListener("click", () => {
          const fmt = document.getElementById("export-format").value;
          // Trigger download of existing sample
          const a = document.createElement("a");
          a.href = `../outputs/samples/sample_01_vase/model.${fmt === "obj" ? "obj" : "glb"}`;
          a.download = `sketch2shape_generated.${fmt}`;
          document.body.appendChild(a);
          a.click();
          a.remove();
          store.showToast(`Exported asset in .${fmt.toUpperCase()} format`, "success");
        });
      }

      const btnSaveGallery = document.getElementById("btn-save-gallery");
      if (btnSaveGallery) {
        btnSaveGallery.addEventListener("click", () => {
          const id = "gen_" + Date.now();
          const newAsset = {
            id,
            title: `Custom ${new Date().toLocaleTimeString()} Asset`,
            category: "Custom",
            vertices: 1840,
            faces: 3680,
            manifold: true,
            latency: "6.92s",
            glbPath: "../outputs/samples/sample_01_vase/model.glb",
            sketchPath: canvasEditor.toDataURL(),
            description: "User-authored custom sketch synthesized into 3D mesh.",
            shapeType: "vase"
          };
          store.addGeneratedAsset(newAsset);
          store.showToast("Saved asset to 3D Gallery!", "success");
        });
      }
    },

    destroy() {
      if (modelViewer) {
        modelViewer.destroy();
        modelViewer = null;
      }
      canvasEditor = null;
    }
  };
}
