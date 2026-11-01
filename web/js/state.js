/**
 * Sketch2Shape AI — Reactive State Store & Pub/Sub EventBus
 * Pure Vanilla ES6+ State Management with LocalStorage Persistence
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

// Initial Curated Hackathon Samples Manifest
const DEFAULT_SAMPLES = [
  {
    id: "sample_01_vase",
    title: "Minimalist Ceramic Vase",
    category: "Decor",
    vertices: 2184,
    faces: 4368,
    manifold: true,
    latency: "6.84s",
    glbPath: "../outputs/samples/sample_01_vase/model.glb",
    objPath: "../outputs/samples/sample_01_vase/model.obj",
    sketchPath: "../outputs/samples/sample_01_vase/input_sketch.png",
    lineartPath: "../outputs/samples/sample_01_vase/preprocessed_lineart.png",
    shapeType: "vase",
    description: "Curved radial pottery profile with smooth interior cavity and closed manifold base."
  },
  {
    id: "sample_02_mug",
    title: "Nordic Coffee Mug",
    category: "Tableware",
    vertices: 1840,
    faces: 3680,
    manifold: true,
    latency: "7.12s",
    glbPath: "../outputs/samples/sample_02_mug/model.glb",
    objPath: "../outputs/samples/sample_02_mug/model.obj",
    sketchPath: "../outputs/samples/sample_02_mug/input_sketch.png",
    lineartPath: "../outputs/samples/sample_02_mug/preprocessed_lineart.png",
    shapeType: "cylinder",
    description: "Cylindrical beverage vessel with integrated toroidal handle topology."
  },
  {
    id: "sample_03_chair",
    title: "Modern Bauhaus Lounge Chair",
    category: "Furniture",
    vertices: 960,
    faces: 1920,
    manifold: true,
    latency: "5.92s",
    glbPath: "../outputs/samples/sample_03_chair/model.glb",
    objPath: "../outputs/samples/sample_03_chair/model.obj",
    sketchPath: "../outputs/samples/sample_03_chair/input_sketch.png",
    lineartPath: "../outputs/samples/sample_03_chair/preprocessed_lineart.png",
    shapeType: "box",
    description: "Sharp rectangular seating geometry with clean planar quad-face topology."
  },
  {
    id: "sample_04_bottle",
    title: "Apothecary Glass Bottle",
    category: "Containers",
    vertices: 1880,
    faces: 3760,
    manifold: true,
    latency: "6.45s",
    glbPath: "../outputs/samples/sample_04_bottle/model.glb",
    objPath: "../outputs/samples/sample_04_bottle/model.obj",
    sketchPath: "../outputs/samples/sample_04_bottle/input_sketch.png",
    lineartPath: "../outputs/samples/sample_04_bottle/preprocessed_lineart.png",
    shapeType: "cone",
    description: "Sleek tapered neck bottle with uniform wall thickness and hollow interior."
  },
  {
    id: "sample_05_bowl",
    title: "Artisan Stoneware Bowl",
    category: "Tableware",
    vertices: 3040,
    faces: 6080,
    manifold: true,
    latency: "7.40s",
    glbPath: "../outputs/samples/sample_05_bowl/model.glb",
    objPath: "../outputs/samples/sample_05_bowl/model.obj",
    sketchPath: "../outputs/samples/sample_05_bowl/input_sketch.png",
    lineartPath: "../outputs/samples/sample_05_bowl/preprocessed_lineart.png",
    shapeType: "sphere",
    description: "Hemispherical lathe geometry with non-intersecting manifold rim boundary."
  }
];

class StateStore {
  constructor() {
    this.bus = new EventBus();
    
    // Load persisted theme or default to dark
    const savedTheme = localStorage.getItem("s2s_theme") || "dark";
    
    // Load generated assets from localStorage
    let savedAssets = [];
    try {
      const stored = localStorage.getItem("s2s_user_assets");
      if (stored) savedAssets = JSON.parse(stored);
    } catch (e) {
      savedAssets = [];
    }

    this.state = {
      theme: savedTheme,
      activeRoute: "#/studio",
      isGenerating: false,
      generationProgress: 0,
      generationStage: "idle",
      
      // Active 3D Model Display
      activeModel: {
        title: "No model generated",
        shapeType: null,
        vertices: null,
        faces: null,
        manifold: null,
        latency: null,
        glbUrl: null,
      },
      
      // Parameters
      params: {
        prompt: "",
        controlnetScale: 0.85,
        inferenceSteps: 30,
        seed: -1,
        precision: "FP16",
        autoRepair: true,
      },

      // Combined Gallery Items
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

  setParams(partialParams) {
    this.state.params = { ...this.state.params, ...partialParams };
    this.bus.emit("params:change", this.state.params);
  }

  setActiveModel(modelData) {
    this.state.activeModel = { ...this.state.activeModel, ...modelData };
    this.bus.emit("model:change", this.state.activeModel);
  }

  addGeneratedAsset(asset) {
    this.state.samples = this.state.samples.filter(existing => existing.id !== asset.id);
    this.state.samples.unshift(asset);
    try {
      const userOnly = this.state.samples.filter(s => s.id.startsWith("gen_") || s.id.startsWith("variation_"));
      localStorage.setItem("s2s_user_assets", JSON.stringify(userOnly));
    } catch (e) {}
    this.bus.emit("gallery:update", this.state.samples);
  }

  clearGeneratedAssets() {
    this.state.samples = this.state.samples.filter(s => !s.id.startsWith("gen_") && !s.id.startsWith("variation_"));
    try { localStorage.removeItem("s2s_user_assets"); } catch (error) { console.warn("Could not clear persisted assets", error); }
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
    
    let icon = "ℹ️";
    if (type === "success") icon = "✅";
    if (type === "error") icon = "❌";

    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(30px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }
}

export const store = new StateStore();
