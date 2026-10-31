/**
 * Sketch2Shape AI — Reactive State Store & Pub/Sub EventBus
 * Robust client-side state management with LocalStorage persistence,
 * template presets, and route-crossing canvas caching.
 */

class EventBus {
  constructor() {
    this.events = new Map();
  }

  on(event, callback) {
    if (!this.events.has(event)) {
      this.events.set(event, new Set());
    }
    this.events.get(event).add(callback);
    return () => this.off(event, callback);
  }

  off(event, callback) {
    if (this.events.has(event)) {
      this.events.get(event).delete(callback);
    }
  }

  emit(event, data) {
    if (this.events.has(event)) {
      this.events.get(event).forEach((cb) => {
        try {
          cb(data);
        } catch (err) {
          console.error(`[EventBus] Error in listener for ${event}:`, err);
        }
      });
    }
  }
}

// Preset Starter Templates for instant 1-click loading
export const PRESET_TEMPLATES = [
  {
    id: "preset_vase",
    name: "Ceramic Vase",
    title: "Ceramic Vase",
    category: "Decor",
    sketchPath: "assets/samples/sample_01_vase/input_sketch.png",
    previewImage: "assets/generated_previews/vase_ceramic_preview.png",
    glbPath: "assets/samples/sample_01_vase/model.glb",
    objPath: "assets/samples/sample_01_vase/model.obj",
    stlPath: "assets/samples/sample_01_vase/model.stl",
    plyPath: "assets/samples/sample_01_vase/model.ply",
    lineartPath: "assets/samples/sample_01_vase/preprocessed_lineart.png",
    prompt: "Smooth porcelain ceramic vase with curved neck and matte glazed surface",
    material: "Porcelain Ceramic",
    shapeType: "vase",
    vertices: 2184,
    faces: 4368,
    description: "Curved radial pottery profile with smooth interior cavity and watertight base.",
  },
  {
    id: "preset_mug",
    name: "Nordic Coffee Mug",
    title: "Nordic Coffee Mug",
    category: "Tableware",
    sketchPath: "assets/samples/sample_02_mug/input_sketch.png",
    previewImage: "assets/generated_previews/mug_slate_preview.png",
    glbPath: "assets/samples/sample_02_mug/model.glb",
    objPath: "assets/samples/sample_02_mug/model.obj",
    stlPath: "assets/samples/sample_02_mug/model.stl",
    plyPath: "assets/samples/sample_02_mug/model.ply",
    lineartPath: "assets/samples/sample_02_mug/preprocessed_lineart.png",
    prompt: "Modern minimalist ceramic coffee mug with solid ergonomic handle",
    material: "Slate Polymer",
    shapeType: "cylinder",
    vertices: 1840,
    faces: 3680,
    description: "Cylindrical beverage vessel with integrated toroidal handle topology.",
  },
  {
    id: "preset_chair",
    name: "Bauhaus Lounge Chair",
    title: "Bauhaus Lounge Chair",
    category: "Furniture",
    sketchPath: "assets/samples/sample_03_chair/input_sketch.png",
    previewImage: "assets/generated_previews/chair_bronze_preview.png",
    glbPath: "assets/samples/sample_03_chair/model.glb",
    objPath: "assets/samples/sample_03_chair/model.obj",
    stlPath: "assets/samples/sample_03_chair/model.stl",
    plyPath: "assets/samples/sample_03_chair/model.ply",
    lineartPath: "assets/samples/sample_03_chair/preprocessed_lineart.png",
    prompt: "Minimalist geometric lounge chair with planar seat and solid frame",
    material: "Cast Bronze",
    shapeType: "box",
    vertices: 960,
    faces: 1920,
    description: "Sharp rectangular seating geometry with clean planar quad-face topology.",
  },
  {
    id: "preset_bottle",
    name: "Apothecary Flask",
    title: "Apothecary Flask",
    category: "Containers",
    sketchPath: "assets/samples/sample_04_bottle/input_sketch.png",
    previewImage: "assets/generated_previews/bottle_slate_preview.png",
    glbPath: "assets/samples/sample_04_bottle/model.glb",
    objPath: "assets/samples/sample_04_bottle/model.obj",
    stlPath: "assets/samples/sample_04_bottle/model.stl",
    plyPath: "assets/samples/sample_04_bottle/model.ply",
    lineartPath: "assets/samples/sample_04_bottle/preprocessed_lineart.png",
    prompt: "Tall reusable steel water bottle with tapered neck and screw cap",
    material: "Slate Polymer",
    shapeType: "cone",
    vertices: 1880,
    faces: 3760,
    description: "Sleek tapered neck bottle with uniform wall thickness and hollow interior.",
  },
  {
    id: "preset_bowl",
    name: "Sculpted Wooden Bowl",
    title: "Sculpted Wooden Bowl",
    category: "Tableware",
    sketchPath: "assets/samples/sample_05_bowl/input_sketch.png",
    previewImage: "assets/generated_previews/bowl_bronze_preview.png",
    glbPath: "assets/samples/sample_05_bowl/model.glb",
    objPath: "assets/samples/sample_05_bowl/model.obj",
    stlPath: "assets/samples/sample_05_bowl/model.stl",
    plyPath: "assets/samples/sample_05_bowl/model.ply",
    lineartPath: "assets/samples/sample_05_bowl/preprocessed_lineart.png",
    prompt: "Deep hand-carved polished wooden salad bowl with smooth curved lip",
    material: "Cast Bronze",
    shapeType: "sphere",
    vertices: 3040,
    faces: 6080,
    description: "Hemispherical lathe geometry with non-intersecting manifold rim boundary.",
  },
];

// Initial Curated Hackathon Samples Manifest
const DEFAULT_SAMPLES = PRESET_TEMPLATES.map((t) => ({
  ...t,
  watertight: true,
  latency: "6.8s",
}));

class StateStore {
  constructor() {
    this.bus = new EventBus();

    // Load persisted theme or default to dark
    const savedTheme = localStorage.getItem("s2s_theme") || "dark";

    // Load custom generated assets from localStorage
    let savedAssets = [];
    try {
      const stored = localStorage.getItem("s2s_user_assets");
      if (stored) savedAssets = JSON.parse(stored);
    } catch (e) {
      savedAssets = [];
    }

    // Persisted sketch in session
    let savedSketch = null;
    try {
      savedSketch = sessionStorage.getItem("s2s_current_sketch");
    } catch (e) {}

    this.state = {
      theme: savedTheme,
      activeRoute: "#/studio",
      isGenerating: false,
      generationProgress: 0,
      generationStage: "idle",

      // Persistent Canvas Drawing state across page transitions
      sketchDataUrl: savedSketch,
      lineartUrl: null,
      depthUrl: null,

      // Active 3D Model Display
      activeModel: {
        title: "Ceramic Vase",
        shapeType: "vase",
        vertices: 2184,
        faces: 4368,
        watertight: true,
        latency: "6.84s",
        material: "ceramic",
        glbUrl: "assets/samples/sample_01_vase/model.glb",
      },

      // Hyperparameters
      params: {
        prompt: "Smooth porcelain ceramic vase with curved neck and matte glazed surface",
        controlnetScale: 0.85,
        inferenceSteps: 30,
        seed: 101,
        precision: "FP16",
        autoRepair: true,
      },

      // Templates & Gallery
      templates: PRESET_TEMPLATES,
      samples: [...DEFAULT_SAMPLES, ...savedAssets],
    };

    // Apply theme attribute to html immediately
    document.documentElement.setAttribute("data-theme", this.state.theme);
  }

  getState() {
    return this.state;
  }

  setTheme(newTheme) {
    this.state.theme = newTheme;
    localStorage.setItem("s2s_theme", newTheme);
    document.documentElement.setAttribute("data-theme", newTheme);
    this.bus.emit("theme:change", newTheme);
  }

  toggleTheme() {
    const next = this.state.theme === "dark" ? "light" : "dark";
    this.setTheme(next);
  }

  setSketch(dataUrl) {
    this.state.sketchDataUrl = dataUrl;
    try {
      if (dataUrl) {
        sessionStorage.setItem("s2s_current_sketch", dataUrl);
      } else {
        sessionStorage.removeItem("s2s_current_sketch");
      }
    } catch (e) {}
    this.bus.emit("sketch:change", dataUrl);
  }

  setStageOutputs(outputs) {
    if (outputs.lineartUrl !== undefined) this.state.lineartUrl = outputs.lineartUrl;
    if (outputs.depthUrl !== undefined) this.state.depthUrl = outputs.depthUrl;
    this.bus.emit("stages:change", {
      lineartUrl: this.state.lineartUrl,
      depthUrl: this.state.depthUrl,
    });
  }

  setParams(partialParams) {
    this.state.params = { ...this.state.params, ...partialParams };
    this.bus.emit("params:change", this.state.params);
  }

  setActiveModel(modelData) {
    this.state.activeModel = { ...this.state.activeModel, ...modelData };
    this.bus.emit("model:change", this.state.activeModel);
  }

  addGeneratedAsset(asset) {
    this.state.samples.unshift(asset);
    try {
      const userOnly = this.state.samples.filter((s) => s.id.startsWith("gen_"));
      localStorage.setItem("s2s_user_assets", JSON.stringify(userOnly));
    } catch (e) {}
    this.bus.emit("gallery:update", this.state.samples);
  }

  setGenerating(isGenerating, stage = "idle", progress = 0) {
    this.state.isGenerating = isGenerating;
    this.state.generationStage = stage;
    this.state.generationProgress = progress;
    this.bus.emit("generation:status", { isGenerating, stage, progress });
  }

  showToast(message, type = "info", duration = 3500) {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;

    let iconSvg = "";
    if (type === "success") {
      iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`;
    } else if (type === "error") {
      iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
    } else {
      iconSvg = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;
    }

    toast.innerHTML = `<span class="toast-icon">${iconSvg}</span><span class="toast-msg">${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("toast-exit");
      setTimeout(() => toast.remove(), 260);
    }, duration);
  }
}

export const store = new StateStore();
