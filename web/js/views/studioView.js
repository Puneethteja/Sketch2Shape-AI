import { store } from "../state.js";
import { CanvasEditor } from "../components/canvasEditor.js";
import { ModelViewer } from "../components/modelViewer.js";

const MATERIAL_PRESETS = [
  { key: "porcelain", name: "Porcelain", desc: "smooth glossy white porcelain ceramic with delicate glaze", bg: "#f8fafc", border: "#cbd5e1" },
  { key: "gold", name: "Gold", desc: "luxurious brushed 24k gold metal with warm specular sheen", bg: "linear-gradient(135deg, #fbbf24, #d97706)" },
  { key: "titanium", name: "Titanium", desc: "aerospace-grade matte brushed titanium alloy with satin reflections", bg: "linear-gradient(135deg, #94a3b8, #475569)" },
  { key: "walnut", name: "Walnut", desc: "hand-rubbed organic polished walnut wood with natural grain", bg: "linear-gradient(135deg, #92400e, #451a03)" },
  { key: "terracotta", name: "Terracotta", desc: "matte artisanal baked terracotta clay with earthy texture", bg: "linear-gradient(135deg, #ea580c, #9a3412)" },
  { key: "glass", name: "Glass", desc: "translucent frosted glass with soft subsurface scattering", bg: "linear-gradient(135deg, #38bdf8, #0284c7)" },
  { key: "obsidian", name: "Obsidian", desc: "glossy volcanic black obsidian with mirror reflections", bg: "linear-gradient(135deg, #334155, #020617)" },
  { key: "coral", name: "Coral", desc: "smooth satin pastel coral polymer clay", bg: "linear-gradient(135deg, #fb7185, #e11d48)" }
];

export function createStudioView() {
  let canvasEditor = null;
  let modelViewer = null;
  let activeMaterial = "porcelain";
  let activeInputMode = "draw";
  let requestController = new AbortController();

  return {
    async render() {
      const state = store.getState();
      const params = state.params;

      return `
        <div class="studio-container">
          <div class="studio-grid">
            <div class="panel-card card-elevated">
              <div class="panel-header">
                <div class="panel-title-wrap">
                  <span style="font-size: 18px;">✏️</span>
                  <div>
                    <h2 class="panel-title">Sketch Input Studio</h2>
                    <span style="font-size: 11px; color: var(--text-muted);">Draw freehand or drop an existing sketch</span>
                  </div>
                </div>
                <span class="badge badge-cyan">Sketch or reference image</span>
              </div>

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

              <div class="tool-group" style="margin-bottom:8px">
                <button id="input-mode-draw" class="tool-btn active">Draw Sketch</button>
                <button id="input-mode-upload" class="tool-btn">Upload Image</button>
                <span id="input-mode-hint" style="font-size:11px;color:var(--text-muted)">Draw with mouse or touch</span>
              </div>

              <div id="canvas-dropzone" class="canvas-viewport-wrap">
                <canvas id="studio-canvas" class="sketch-canvas"></canvas>
              </div>

              <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                <label for="file-upload" class="btn btn-secondary btn-sm" style="cursor: pointer; width: 100%;">
                  📂 Upload Reference Image (PNG, JPG, WEBP)
                </label>
                <input id="file-upload" type="file" accept="image/png,image/jpeg,image/webp" style="display: none;">
              </div>
              <div style="margin-top:8px">
                <button id="btn-open-crop" class="tool-btn">Crop / clean image preview</button>
                <div id="crop-review" class="panel-card" style="display:none;margin-top:8px;padding:12px">
                  <strong>Review detected object crop</strong>
                  <p id="crop-review-message" style="font-size:11px;color:var(--text-muted)">Adjust the crop bounds so every part of the object remains inside.</p>
                  <canvas id="crop-preview-canvas" width="512" height="512" style="width:100%;max-height:300px;object-fit:contain;background:#fff"></canvas>
                  <div class="crop-slider-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px">
                    <label>Left <input id="crop-left" type="range" min="0" max="70" value="0"></label>
                    <label>Top <input id="crop-top" type="range" min="0" max="70" value="0"></label>
                    <label>Width <input id="crop-width" type="range" min="30" max="100" value="100"></label>
                    <label>Height <input id="crop-height" type="range" min="30" max="100" value="100"></label>
                  </div>
                  <button id="btn-apply-crop" class="btn btn-primary btn-sm" style="margin-top:8px">Use this crop</button>
                  <button id="btn-cancel-crop" class="tool-btn btn-sm" style="margin-top:8px">Cancel</button>
                </div>
              </div>

              <div class="form-group" style="margin-top: 10px;">
                <label class="form-label" for="object-category">Object identity hint</label>
                <select id="object-category" class="form-select">
                  <option selected>Auto-Detect from Sketch</option>
                  ${Object.entries({Furniture:["Chair","Table","Sofa","Bed","Desk","Stool","Shelf"],"Household Objects":["Vase","Cup","Bottle","Lamp","Plate","Bowl","Clock"],Accessories:["Shopping Bag","Handbag","Backpack","Shoes","Hat","Wallet"],Animals:["Cat","Dog","Rabbit","Bird","Fish","Elephant","Lion","Horse","Other Animals"],Fruits:["Apple","Banana","Orange","Mango","Grapes","Watermelon","Strawberry","Other Fruits"],Vegetables:["Tomato","Carrot","Potato","Brinjal","Pumpkin","Other Vegetables"],Nature:["Tree"],Vehicles:["Car","Bicycle","Motorcycle","Bus","Truck","Airplane"],Electronics:["Mobile Phone","Laptop","Headphones","Camera","Speaker"],Toys:["Teddy Bear","Doll","Toy Car","Building Blocks"],Other:["Other"]}).map(([group,items]) => `<optgroup label="${group}">${items.map(item=>`<option>${item}</option>`).join("")}</optgroup>`).join("")}
                </select>
              </div>

              <div class="form-group" style="margin-top: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                  <label class="form-label" style="margin-bottom: 0;">
                    <span>🎨 Material & Surface Palette</span>
                  </label>
                  <span id="active-material-badge" class="badge badge-cyan">Porcelain</span>
                </div>
                <div class="material-grid" id="material-presets-grid">
                  ${MATERIAL_PRESETS.map((m) => `
                    <button type="button" class="material-btn ${m.key === 'porcelain' ? 'active' : ''}" data-material="${m.key}" data-desc="${m.desc}" title="${m.name}: ${m.desc}">
                      <span class="material-swatch" style="background: ${m.bg}; ${m.border ? `border: 1px solid ${m.border};` : ''}"></span>
                      <span class="material-label">${m.name}</span>
                    </button>
                  `).join("")}
                </div>
              </div>

              <div class="form-group" style="margin-top: 10px;">
                <label class="form-label" for="prompt-input">
                  <span>Material & Style Prompt (Optional)</span>
                  <span style="font-size: 10px; color: var(--accent-cyan);">Click any material above to sync</span>
                </label>
                <textarea id="prompt-input" class="form-textarea" rows="2" placeholder="Optional style or material guidance; the input image determines object identity">${params.prompt || ""}</textarea>
              </div>

              <div style="display: flex; gap: 10px; margin-top: 8px;">
                <button id="btn-preprocess" class="btn btn-secondary" style="flex: 1;">
                  ⚙️ Preprocess Lineart
                </button>
                <button id="btn-generate-3d" class="btn btn-primary" style="flex: 2;">
                  ⚡ Generate 3D Asset
                </button>
              </div>
              <button id="btn-generate-variations" class="btn btn-secondary" style="width:100%; margin-top:8px; display:none;">Generate Variations</button>
              <div id="variation-gallery" class="preset-strip-grid" style="display:none; margin-top:10px;"></div>
            </div>

            <div class="panel-card card-elevated">
              <div class="panel-header">
                <div class="panel-title-wrap">
                  <span style="font-size: 18px;">🌐</span>
                  <div>
                    <h2 class="panel-title">3D WebGL Viewport</h2>
                    <span style="font-size: 11px; color: var(--text-muted);">Real-time Orbit &middot; Zoom &middot; Pan &middot; Manifold Mesh</span>
                  </div>
                </div>
                <div style="display: flex; gap: 6px;">
                <span id="status-manifold-badge" class="badge">Awaiting model</span>
                </div>
              </div>

              <div class="viewport-toolbar">
                <div class="tool-group">
                  <button id="btn-vp-wireframe" class="tool-btn" title="Toggle wireframe topology">🕸️ Wireframe</button>
                  <button id="btn-vp-autorotate" class="tool-btn" title="Auto orbital rotate">🔄 Orbit</button>
                  <button id="btn-vp-reset" class="tool-btn" title="Reset camera viewpoint">🎯 Reset</button>
                </div>

                <div class="tool-group">
                  <span class="viewport-stat-chip">
                    <span id="stat-model-name" style="color: var(--accent-cyan); font-weight: 600;">No model generated</span>
                  </span>
                </div>
              </div>

              <div class="model-viewport-wrap">
                <div id="studio-webgl-container" class="webgl-canvas-container"></div>

                <div id="generation-overlay" class="generation-overlay" style="display: none;">
                  <div class="spinner-large"></div>
                  <div class="overlay-title">Synthesizing 3D Geometry</div>
                  <div id="gen-stage-label" class="overlay-stage">Executing Neural Handoff...</div>
                  <div class="progress-bar-wrap">
                    <div id="gen-progress-bar" class="progress-bar-fill"></div>
                  </div>
                </div>
              </div>

              <div class="viewport-toolbar" aria-label="Multiple views of this model">
                <span class="preview-mini-title">Multiple Views</span>
                ${["front","back","left","right","top","bottom","iso"].map(view=>`<button class="tool-btn model-view-btn" data-view="${view}">${view === "iso" ? "Isometric" : view[0].toUpperCase()+view.slice(1)}</button>`).join("")}
              </div>
              <div id="model-analysis-summary" class="mesh-stats-bar" style="display:none"></div>

              <div class="intermediate-previews">
                <div class="preview-mini-card">
                  <span class="preview-mini-title">Stage 1 Lineart</span>
                  <div id="preview-lineart-box" class="preview-mini-box">
                    <span style="font-size: 10px; color: var(--text-muted);">Awaiting Run</span>
                  </div>
                </div>
                <div class="preview-mini-card">
                  <span class="preview-mini-title">Stage 2 Diffusion</span>
                  <div id="preview-depth-box" class="preview-mini-box">
                    <span style="font-size: 10px; color: var(--text-muted);">Awaiting Run</span>
                  </div>
                </div>
                <div class="preview-mini-card">
                  <span class="preview-mini-title">Stage 3 Volumetric</span>
                  <div class="preview-mini-box" style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 2px;">
                    <span style="font-size: 11px; font-weight: 700; color: var(--accent-emerald);">512³ Grid</span>
                    <span style="font-size: 9px; color: var(--text-muted);">FP16 Tensors</span>
                  </div>
                </div>
                <div class="preview-mini-card">
                  <span class="preview-mini-title">Stage 4 Manifold</span>
                  <div class="preview-mini-box" style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 2px;">
                    <span style="font-size: 11px; font-weight: 700; color: var(--accent-cyan);">7-Step Repair</span>
                    <span style="font-size: 9px; color: var(--accent-emerald);">Zero Holes</span>
                  </div>
                </div>
              </div>

              <div class="mesh-stats-bar">
                <div class="stat-pill">
                  <span class="stat-pill-label">Vertices</span>
                  <span id="stat-vertices" class="stat-pill-val">—</span>
                </div>
                <div class="stat-pill">
                  <span class="stat-pill-label">Faces</span>
                  <span id="stat-faces" class="stat-pill-val">—</span>
                </div>
                <div class="stat-pill">
                  <span class="stat-pill-label">Topology</span>
                  <span class="stat-pill-val">—</span>
                </div>
                <div class="stat-pill">
                  <span class="stat-pill-label">Format</span>
                  <span class="stat-pill-val">GLTF / GLB 2.0</span>
                </div>
              </div>

              <div class="export-tray">
                <div style="display: flex; gap: 8px; align-items: center;">
                  <span style="font-size: 12px; color: var(--text-secondary);">Format:</span>
                  <select id="export-format" class="form-select" style="width: 90px; padding: 6px 10px; font-size: 12px;">
                    <option value="glb" selected>GLB</option>
                    <option value="obj">OBJ</option>
                    <option value="stl">STL</option>
                  </select>
                </div>

                <div style="display: flex; gap: 8px;">
                  <button id="btn-save-gallery" class="btn btn-secondary btn-sm" title="Save snapshot to Gallery" disabled>
                    💾 Save Asset
                  </button>
                  <button id="btn-export-download" class="btn btn-primary btn-sm" title="Download 3D model asset" disabled>
                    ⬇️ Download Mesh
                  </button>
                </div>
              </div>
            </div>
          </div>

          <div class="preset-strip-wrap card-elevated" style="margin-top: 20px;">
            <div class="panel-header" style="padding-bottom: 8px; margin-bottom: 12px;">
              <div class="panel-title-wrap">
                <span style="font-size: 16px;">📚</span>
                <div>
                  <h3 class="panel-title" style="font-size: 14px;">Instant Curated Sketch Presets</h3>
                  <span style="font-size: 11px; color: var(--text-muted);">Click any benchmark sketch to instantly populate the canvas & 3D mesh</span>
                </div>
              </div>
            </div>

            <div class="preset-strip-grid">
              ${state.samples.map(sample => `
                <div class="preset-thumb-card" data-sample-id="${sample.id}">
                  <div class="preset-img-wrap">
                    <img src="${sample.sketchPath}" alt="${sample.title}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'100\\' height=\\'100\\' fill=\\'%23334155\\'><text x=\\'50%\\' y=\\'50%\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' fill=\\'white\\'>Sketch</text></svg>'">
                    <span class="badge badge-cyan" style="position: absolute; bottom: 4px; right: 4px; font-size: 9px; padding: 2px 6px;">${sample.category}</span>
                  </div>
                  <div class="preset-card-title">${sample.title}</div>
                </div>
              `).join("")}
            </div>
          </div>
        </div>
      `;
    },

    async afterRender() {
      const canvasEl = document.getElementById("studio-canvas");
      if (canvasEl) {
        canvasEditor = new CanvasEditor(canvasEl);
      }

      const webglContainer = document.getElementById("studio-webgl-container");
      if (webglContainer) {
        modelViewer = new ModelViewer(webglContainer, { material: activeMaterial });
      }
      document.querySelectorAll(".model-view-btn").forEach((button) => button.addEventListener("click", () => modelViewer?.setView(button.dataset.view)));

      const toolBrush = document.getElementById("tool-brush");
      const toolEraser = document.getElementById("tool-eraser");
      const brushSize = document.getElementById("brush-size");
      const brushSizeVal = document.getElementById("brush-size-val");
      const btnClear = document.getElementById("btn-clear");
      const btnUndo = document.getElementById("btn-undo");
      const btnRedo = document.getElementById("btn-redo");

      if (toolBrush && toolEraser) {
        toolBrush.addEventListener("click", () => {
          canvasEditor.setMode("brush");
          toolBrush.classList.add("active");
          toolEraser.classList.remove("active");
        });
        toolEraser.addEventListener("click", () => {
          canvasEditor.setMode("eraser");
          toolEraser.classList.add("active");
          toolBrush.classList.remove("active");
        });
      }

      if (canvasEditor) {
        canvasEditor.onHistoryChange = ({ canUndo, canRedo }) => {
          if (btnUndo) btnUndo.disabled = !canUndo;
          if (btnRedo) btnRedo.disabled = !canRedo;
        };
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

      const fileInput = document.getElementById("file-upload");
      const drawMode = document.getElementById("input-mode-draw");
      const uploadMode = document.getElementById("input-mode-upload");
      const modeHint = document.getElementById("input-mode-hint");
      if (drawMode && uploadMode) {
        drawMode.addEventListener("click", () => {
          activeInputMode = "draw";
          drawMode.classList.add("active"); uploadMode.classList.remove("active");
          if (modeHint) modeHint.textContent = "Draw with mouse or touch";
        });
        uploadMode.addEventListener("click", () => {
          activeInputMode = "image";
          uploadMode.classList.add("active"); drawMode.classList.remove("active");
          if (modeHint) modeHint.textContent = "Upload an image to preview it on the canvas";
          fileInput?.click();
        });
      }
      if (fileInput) {
        fileInput.addEventListener("change", async (e) => {
          const file = e.target.files[0];
          if (file) {
            await canvasEditor.loadImageFromFile(file);
            store.showToast(`Loaded "${file.name}" into sketchpad`, "success");
          }
        });
      }

      const cropPanel = document.getElementById("crop-review");
      const cropCanvas = document.getElementById("crop-preview-canvas");
      const cropCtx = cropCanvas?.getContext("2d");
      let cropImage = null;
      const cropSliders = ["crop-left", "crop-top", "crop-width", "crop-height"].map(id => document.getElementById(id));
      const drawCropPreview = () => {
        if (!cropCtx || !cropImage) return;
        const [left, top, width, height] = cropSliders.map(input => Number(input.value));
        const sx = cropImage.width * left / 100;
        const sy = cropImage.height * top / 100;
        const sw = cropImage.width * Math.min(width, 100 - left) / 100;
        const sh = cropImage.height * Math.min(height, 100 - top) / 100;
        cropCtx.fillStyle = "#fff";
        cropCtx.fillRect(0, 0, cropCanvas.width, cropCanvas.height);
        const scale = Math.min(480 / sw, 480 / sh);
        const dw = sw * scale, dh = sh * scale;
        cropCtx.drawImage(cropImage, sx, sy, sw, sh, (512 - dw) / 2, (512 - dh) / 2, dw, dh);
      };
      document.getElementById("btn-open-crop")?.addEventListener("click", async () => {
        if (canvasEditor.isEmpty()) { store.showToast("Draw or upload an image before cropping.", "error"); return; }
        const button = document.getElementById("btn-open-crop");
        button.disabled = true;
        try {
          const blob = await (await fetch(canvasEditor.toDataURL())).blob();
          const form = new FormData(); form.append("file", blob, "source.png");
          const response = await fetch("/prepare-crop", { method: "POST", body: form, signal: requestController.signal });
          const result = await response.json();
          if (!response.ok) throw new Error(result.detail || "Crop preview failed.");
          cropImage = new Image();
          cropImage.onload = () => {
            cropSliders.forEach((slider, index) => { slider.value = [0, 0, 100, 100][index]; });
            drawCropPreview();
            cropPanel.style.display = "block";
          };
          cropImage.src = `data:image/png;base64,${result.image_b64}`;
          const message = document.getElementById("crop-review-message");
          if (message) message.textContent = result.uncertain ? `${result.reason} Adjust the bounds manually; automatic cropping was not applied.` : "Review the detected crop and adjust it to include the complete object.";
        } catch (error) { store.showToast(error.message || "Crop preview failed.", "error", 6000); }
        finally { button.disabled = false; }
      });
      cropSliders.forEach(slider => slider?.addEventListener("input", drawCropPreview));
      document.getElementById("btn-apply-crop")?.addEventListener("click", () => {
        if (!cropCanvas || !canvasEditor) return;
        canvasEditor.saveState();
        canvasEditor.ctx.fillStyle = "#fff";
        canvasEditor.ctx.fillRect(0, 0, canvasEditor.canvas.width, canvasEditor.canvas.height);
        canvasEditor.ctx.drawImage(cropCanvas, 0, 0, canvasEditor.canvas.width, canvasEditor.canvas.height);
        cropPanel.style.display = "none";
        store.showToast("Reviewed crop applied. This image will be sent for generation.", "success");
      });
      document.getElementById("btn-cancel-crop")?.addEventListener("click", () => { cropPanel.style.display = "none"; });

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

      const promptInput = document.getElementById("prompt-input");
      if (promptInput) {
        promptInput.addEventListener("input", (e) => {
          store.setParams({ prompt: e.target.value });
        });
      }

      const matButtons = document.querySelectorAll(".material-btn");
      const matBadge = document.getElementById("active-material-badge");
      matButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
          matButtons.forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          const matKey = btn.getAttribute("data-material");
          const matDesc = btn.getAttribute("data-desc");
          const matLabel = btn.querySelector(".material-label")?.textContent || matKey;

          activeMaterial = matKey;
          if (matBadge) matBadge.textContent = matLabel;
          if (modelViewer) modelViewer.setMaterial(matKey);

          if (promptInput) {
            promptInput.value = matDesc;
            store.setParams({ prompt: matDesc });
          }

          store.showToast(`Applied ${matLabel} surface material`, "info", 1500);
        });
      });

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

      const btnPreprocess = document.getElementById("btn-preprocess");
      if (btnPreprocess) {
        btnPreprocess.addEventListener("click", async () => {
          if (canvasEditor.isEmpty()) {
            store.showToast("Please sketch or upload an object first!", "error");
            return;
          }

          btnPreprocess.disabled = true;
          try {
            const sketchBlob = await (await fetch(canvasEditor.toDataURL())).blob();
            const form = new FormData();
            form.append("file", sketchBlob, "sketch.png");
            form.append("mode", "lineart");
            const response = await fetch("/preview", { method: "POST", body: form, signal: requestController.signal });
            const result = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(result.detail || `Preview failed (${response.status}).`);

            const lineartBox = document.getElementById("preview-lineart-box");
            const depthBox = document.getElementById("preview-depth-box");
            if (lineartBox && result.image_b64) lineartBox.innerHTML = `<img src="data:image/png;base64,${result.image_b64}" alt="Preprocessed line drawing" style="width:100%;height:100%;object-fit:contain;">`;
            if (depthBox) depthBox.innerHTML = `<img src="${canvasEditor.toDataURL()}" alt="Original source image" style="width:100%;height:100%;object-fit:contain;">`;
            store.showToast("Line-art preview is ready.", "success");
          } catch (error) {
            const message = error instanceof TypeError
              ? "Could not reach the backend. Start it with .\\scripts\\start_local.ps1."
              : (error.message || "Could not reach the local backend.");
            store.showToast(message, "error", 5000);
          } finally {
            btnPreprocess.disabled = false;
          }
        });
      }

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
            updateProgress(15, "Preparing sketch…");
            const sketchBlob = await (await fetch(canvasEditor.toDataURL())).blob();
            const form = new FormData();
            const params = store.getState().params;
            form.append("file", sketchBlob, "sketch.png");
            form.append("prompt", promptInput ? promptInput.value.trim() : "");
            form.append("category", document.getElementById("object-category")?.value || "Auto-Detect from Sketch");
            form.append("input_mode", activeInputMode);
            form.append("controlnet_scale", String(params.controlnetScale ?? 0.8));
            form.append("num_inference_steps", String(params.inferenceSteps ?? 30));
            form.append("seed", String(params.seed ?? -1));
            form.append("export_format", document.getElementById("export-format").value);

            updateProgress(35, "Sending the reviewed image to the configured 3D model…");
            const response = await fetch("/generate", { method: "POST", body: form, signal: requestController.signal });
            const result = await response.json().catch(() => ({}));
            if (!response.ok || !result.success) {
              throw new Error(result.detail || `Generation failed (${response.status}).`);
            }

            const meshUrl = result.mesh_url || (result.export_format === "glb" ? result.gltf_url : null);
            const gltfUrl = result.gltf_url || (result.export_format === "glb" ? meshUrl : null);
            if (!meshUrl && !gltfUrl) throw new Error("The backend completed without returning a mesh file.");
            updateProgress(86, "Loading generated mesh…");
            if (gltfUrl) {
              modelViewer.loadGLB(gltfUrl, {
                onError: () => store.showToast("The mesh was generated, but its 3D preview could not be loaded.", "error", 5000)
              });
            }

            const vCount = document.getElementById("stat-vertices");
            const fCount = document.getElementById("stat-faces");
            const mName = document.getElementById("stat-model-name");
            const manifoldBadge = document.getElementById("status-manifold-badge");
            const topologyValue = document.querySelector(".mesh-stats-bar .stat-pill:nth-child(3) .stat-pill-val");
            if (vCount) vCount.textContent = Number(result.vertices || 0).toLocaleString();
            if (fCount) fCount.textContent = Number(result.faces || 0).toLocaleString();
            if (mName) mName.textContent = result.category && !result.category.startsWith("Auto") ? result.category : `Run ${result.run_id || "complete"}`;
            const topology = result.is_watertight ? "Watertight" : "Open / unverified";
            if (manifoldBadge) {
              manifoldBadge.textContent = result.is_watertight ? "✓ Watertight" : "Topology unverified";
              manifoldBadge.classList.toggle("badge-success", Boolean(result.is_watertight));
              manifoldBadge.classList.toggle("badge-amber", !result.is_watertight);
            }
            if (topologyValue) topologyValue.textContent = topology;

            btnGenerate.dataset.meshUrl = meshUrl || gltfUrl;
            btnGenerate.dataset.gltfUrl = gltfUrl || "";
            btnGenerate.dataset.exportFormat = result.export_format || "glb";
            btnGenerate.dataset.runId = result.run_id || "";
            btnGenerate.dataset.vertices = String(result.vertices || 0);
            btnGenerate.dataset.faces = String(result.faces || 0);
            btnGenerate.dataset.watertight = String(Boolean(result.is_watertight));
            btnGenerate.dataset.elapsed = String(result.timing?.total_seconds ?? "");
            btnGenerate.dataset.category = document.getElementById("object-category")?.value || "Auto-Detect from Sketch";
            btnGenerate.dataset.analysis = JSON.stringify(result.analysis || {});
            btnGenerate.dataset.provider = result.generation_provider || "local";
            const saveButton = document.getElementById("btn-save-gallery");
            const exportButton = document.getElementById("btn-export-download");
            if (saveButton) saveButton.disabled = false;
            if (exportButton) exportButton.disabled = false;
            const generatedAsset = {
              id: `gen_${result.run_id || Date.now()}`,
              title: `${document.getElementById("object-category")?.value || "Generated model"} · ${new Date().toLocaleTimeString()}`,
              category: document.getElementById("object-category")?.value || "Auto-Detect from Sketch",
              vertices: Number(result.vertices || 0), faces: Number(result.faces || 0),
              manifold: Boolean(result.is_watertight), latency: result.timing?.total_seconds ? `${result.timing.total_seconds}s` : "—",
              glbPath: gltfUrl || (result.export_format === "glb" ? meshUrl : undefined),
              objPath: result.export_format === "obj" ? meshUrl : undefined,
              sketchPath: canvasEditor.toDataURL(),
              description: promptInput?.value || "Generated from the selected sketch or reference image.",
              createdAt: new Date().toISOString(), analysis: result.analysis || {}, parentId: null
            };
            btnGenerate.dataset.assetId = generatedAsset.id;
            store.addGeneratedAsset(generatedAsset);
            const analysis = result.analysis;
            const summary = document.getElementById("model-analysis-summary");
            if (summary && analysis?.available) {
              summary.style.display = "flex";
              const similarity = analysis.shape_similarity == null ? "Unavailable" : `${analysis.shape_similarity}/100`;
              summary.innerHTML = `<div class="stat-pill"><span class="stat-pill-label">Mesh Integrity</span><span class="stat-pill-val">${analysis.mesh_integrity}/100</span></div><div class="stat-pill"><span class="stat-pill-label">Shape Similarity</span><span class="stat-pill-val">${similarity}</span></div><div class="stat-pill"><span class="stat-pill-label">Connected Parts</span><span class="stat-pill-val">${analysis.connected_components ?? "Unavailable"}</span></div><div class="stat-pill"><span class="stat-pill-label">Validity</span><span class="stat-pill-val">${analysis.validity_score}/100</span></div>`;
              if (analysis.issues?.length) store.showToast(analysis.issues.join(" "), "info", 6000);
            }
            updateProgress(100, "Model ready");
            const variationsButton = document.getElementById("btn-generate-variations");
            if (variationsButton) variationsButton.style.display = "block";
            const settings = result.generation_settings;
            const validation = result.output_validation;
            const mode = settings?.generate_type || (settings?.request_type === 3 ? "textured image-to-3D" : "image-to-3D");
            const detail = settings ? `${settings.model} · ${mode} · ${validation?.textures || 0} texture(s)` : (result.generation_provider || result.device || "backend");
            const creditNote = result.credits_used == null ? "" : ` Meshy credits used: ${result.credits_used}.`;
            store.showToast(`Model generated and validated: ${detail}.${creditNote}`, "success", 6000);
            if (result.export_format !== "glb" && !result.gltf_url) {
              store.showToast("This format can be downloaded. Select GLB to preview it in the studio.", "info", 4500);
            }
          } catch (error) {
            const message = error instanceof TypeError
              ? "Could not reach the backend. Start it with .\\scripts\\start_local.ps1."
              : (error.message || "Generation failed.");
            store.showToast(message, "error", 6000);
          } finally {
            overlay.style.display = "none";
            btnGenerate.disabled = false;
          }
        });
      }

      const btnExport = document.getElementById("btn-export-download");
      if (btnExport) {
        btnExport.addEventListener("click", () => {
          const fmt = btnGenerate?.dataset.exportFormat;
          const meshUrl = btnGenerate?.dataset.meshUrl;
          if (!fmt || !meshUrl) {
            store.showToast("Generate a mesh before downloading.", "error");
            return;
          }
          const a = document.createElement("a");
          a.href = meshUrl;
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
          const meshUrl = btnGenerate?.dataset.meshUrl;
          const fmt = btnGenerate?.dataset.exportFormat;
          if (!meshUrl || (fmt !== "glb" && !btnGenerate?.dataset.gltfUrl)) {
            store.showToast("Generate a GLB mesh before saving it to the 3D Gallery.", "error");
            return;
          }
          const id = btnGenerate.dataset.assetId || ("gen_" + Date.now());
          const newAsset = {
            id,
            title: `Generated model ${new Date().toLocaleTimeString()}`,
            category: btnGenerate.dataset.category || "Auto-Detect from Sketch",
            vertices: Number(btnGenerate.dataset.vertices || 0),
            faces: Number(btnGenerate.dataset.faces || 0),
            manifold: btnGenerate.dataset.watertight === "true",
            latency: btnGenerate.dataset.elapsed ? `${btnGenerate.dataset.elapsed}s` : "—",
            glbPath: btnGenerate.dataset.gltfUrl || meshUrl,
            objPath: btnGenerate.dataset.exportFormat === "obj" ? meshUrl : undefined,
            sketchPath: canvasEditor.toDataURL(),
            description: promptInput?.value || "Generated from the selected sketch or reference image.",
            createdAt: new Date().toISOString(),
            analysis: JSON.parse(btnGenerate.dataset.analysis || "{}")
          };
          store.addGeneratedAsset(newAsset);
          store.showToast("Saved asset to 3D Gallery!", "success");
        });
      }

      const variationsButton = document.getElementById("btn-generate-variations");
      if (variationsButton) variationsButton.addEventListener("click", async () => {
        if (btnGenerate?.dataset.provider !== "Tencent TokenHub HY-3D") {
          store.showToast("Variations require the configured Tencent HY-3D image-conditioned service.", "error", 5000);
          return;
        }
        variationsButton.disabled = true;
        variationsButton.textContent = "Generating structural variations…";
        const gallery = document.getElementById("variation-gallery");
        gallery.style.display = "grid";
        const styles = ["Change the pose and proportions while preserving the same object identity.", "Create a distinct structural design variation of the same object; preserve its identity.", "Create another structural variation with changed shape details, preserving the same object identity."];
        try {
          for (let index = 0; index < styles.length; index++) {
            const form = new FormData();
            const blob = await new Promise(resolve => canvasEditor.canvas.toBlob(resolve, "image/png"));
            form.append("file", blob, "source.png");
            form.append("category", document.getElementById("object-category")?.value || "Auto-Detect from Sketch");
            form.append("input_mode", activeInputMode);
            form.append("prompt", `${promptInput?.value || ""} ${styles[index]}`.trim());
            form.append("export_format", "glb");
            const response = await fetch("/generate", { method: "POST", body: form, signal: requestController.signal });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || "Variation generation failed.");
            const item = { id: `variation_${Date.now()}_${index}`, title: `Variation ${index + 1}`, category: btnGenerate.dataset.category || "Auto-Detect from Sketch", glbPath: result.gltf_url || result.mesh_url, sketchPath: canvasEditor.toDataURL(), parentId: `gen_${btnGenerate.dataset.runId}`, description: "Image-conditioned structural variation", createdAt: new Date().toISOString() };
            store.addGeneratedAsset(item);
            const card = document.createElement("button");
            card.className = "preset-thumb-card";
            card.textContent = `Variation ${index + 1} — open in viewer`;
            card.addEventListener("click", () => modelViewer.loadGLB(item.glbPath, { onError: () => store.showToast("Could not open this variation.", "error") }));
            gallery.appendChild(card);
          }
          store.showToast("Three image-conditioned structural variations added to Gallery.", "success");
        } catch (error) {
          store.showToast(error.message || "Variation generation failed.", "error", 6000);
        } finally {
          variationsButton.disabled = false;
          variationsButton.textContent = "Generate Variations";
        }
      });

      const presetCards = document.querySelectorAll(".preset-thumb-card");
      presetCards.forEach((card) => {
        card.addEventListener("click", async () => {
          const sampleId = card.getAttribute("data-sample-id");
          const samples = store.getState().samples;
          const sample = samples.find(s => s.id === sampleId);
          if (!sample) return;

          await canvasEditor.loadImageFromURL(sample.sketchPath);

          if (promptInput) {
            promptInput.value = sample.description;
            store.setParams({ prompt: sample.description });
          }

          if (sample.shapeType) {
            const stats = modelViewer.loadProceduralShape(sample.shapeType);
            if (activeMaterial) modelViewer.setMaterial(activeMaterial);
            const vCount = document.getElementById("stat-vertices");
            const fCount = document.getElementById("stat-faces");
            const mName = document.getElementById("stat-model-name");
            if (vCount) vCount.textContent = (sample.vertices || stats.vertices).toLocaleString();
            if (fCount) fCount.textContent = (sample.faces || stats.faces).toLocaleString();
            if (mName) mName.textContent = sample.title;
          } else if (sample.glbPath) {
            modelViewer.loadGLB(sample.glbPath);
            const mName = document.getElementById("stat-model-name");
            if (mName) mName.textContent = sample.title;
          }

          const lineartBox = document.getElementById("preview-lineart-box");
          if (lineartBox && sample.lineartPath) {
            lineartBox.innerHTML = `<img src="${sample.lineartPath}" style="width:100%;height:100%;object-fit:contain;">`;
          }

          store.showToast(`Loaded preset: "${sample.title}"`, "info", 2000);
        });
      });
    },

    destroy() {
      requestController.abort();
      if (canvasEditor) {
        canvasEditor.destroy();
        canvasEditor = null;
      }
      if (modelViewer) {
        modelViewer.destroy();
        modelViewer = null;
      }
    }
  };
}
