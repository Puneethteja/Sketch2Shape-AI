(function (global, factory) {
    if (typeof module === "object" && typeof module.exports === "object") {
        module.exports = factory();
    } else if (typeof define === "function" && define.amd) {
        define([], factory);
    } else {
        global.Sketch2ShapeViewer = factory();
    }
})(typeof window !== "undefined" ? window : this, function () {
    "use strict";

    const THEME = {
        bg: "#090d16",
        surface: "#0f172a",
        surfaceElevated: "#1e293b",
        border: "#334155",
        borderSubtle: "#1e293b",
        textPrimary: "#f8fafc",
        textMuted: "#94a3b8",
        textDark: "#64748b",
        accent: "#38bdf8",
        gridLine: "#1e293b",
        gridCenter: "#334155",
    };

    class Sketch2ShapeViewer {
        constructor(container, options = {}) {
            this.container = typeof container === "string" ? document.querySelector(container) : container;
            if (!this.container) {
                throw new Error("[Sketch2ShapeViewer] Target container not found.");
            }

            this.options = Object.assign({
                autoInit: true,
                showGrid: true,
                showControls: true,
                showDownloadBtn: true,
                autoRotate: false,
                wireframe: false,
                initialModel: null,
                cdnThreeUrl: "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
                cdnControlsUrl: "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js",
                cdnGltfLoaderUrl: "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js",
            }, options);

            this.activeModel = null;
            this.activeBlob = null;
            this.activeBlobUrl = null;
            this.activeFilename = "sketch2shape_generated.glb";
            this.isLoading = false;
            this.isDownloading = false;
            this.isAutoRotating = this.options.autoRotate;
            this.isWireframe = this.options.wireframe;
            this.gridVisible = this.options.showGrid;

            this.scene = null;
            this.camera = null;
            this.renderer = null;
            this.controls = null;
            this.modelGroup = null;
            this.gridHelper = null;
            this.animationFrameId = null;

            this.dom = {
                wrapper: null,
                viewport: null,
                spinnerOverlay: null,
                spinnerBar: null,
                spinnerText: null,
                toolbar: null,
                downloadBtn: null,
                downloadSpinner: null,
                statBadge: null,
            };

            if (this.options.autoInit) {
                this.init();
            }
        }

        init() {
            this._buildDOM();
            this._bindWindowEvents();
            this._setupOutputBinding();
            this._loadThreeDependencies()
                .then(() => {
                    this._initThreeScene();
                    if (this.options.initialModel) {
                        this.loadGLB(this.options.initialModel);
                    } else {
                        this.loadParametricDemo("vase");
                    }
                })
                .catch((err) => {
                    console.warn("[Sketch2ShapeViewer] CDN Three.js load failed, using native WebGL fallback:", err);
                    this._initNativeWebGLFallback();
                });
        }

        _buildDOM() {
            this.container.innerHTML = "";
            this.container.style.position = "relative";
            this.container.style.width = "100%";
            this.container.style.height = "100%";
            this.container.style.minHeight = "400px";
            this.container.style.display = "flex";
            this.container.style.flexDirection = "column";
            this.container.style.background = THEME.bg;
            this.container.style.borderRadius = "8px";
            this.container.style.overflow = "hidden";
            this.container.style.border = "1px solid " + THEME.border;
            this.container.style.boxSizing = "border-box";
            this.container.style.fontFamily = "'Inter', ui-sans-serif, system-ui, sans-serif";

            const header = document.createElement("div");
            header.className = "s2s-viewer-header";
            header.style.cssText = "display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:" + THEME.surface + ";border-bottom:1px solid " + THEME.borderSubtle + ";color:" + THEME.textMuted + ";font-size:11px;font-weight:500;letter-spacing:0.04em;user-select:none;z-index:10;";

            header.innerHTML = '<div style="display:flex;align-items:center;gap:8px;">' +
                '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#10b981;"></span>' +
                '<span style="color:' + THEME.textPrimary + ';font-weight:600;text-transform:uppercase;">WebGL 3D Viewport</span>' +
                '<span style="color:' + THEME.textDark + ';font-family:monospace;">GLTF / GLB</span>' +
                '</div>' +
                '<div style="display:flex;align-items:center;gap:8px;" id="s2s-header-actions">' +
                '<span id="s2s-stat-badge" style="font-family:monospace;font-size:10px;color:' + THEME.textDark + ';">Ready</span>' +
                '</div>';
            this.container.appendChild(header);
            this.dom.statBadge = header.querySelector("#s2s-stat-badge");

            const viewport = document.createElement("div");
            viewport.className = "s2s-viewport-canvas-host";
            viewport.style.cssText = "position:relative;flex:1;width:100%;min-height:340px;cursor:grab;outline:none;overflow:hidden;";
            this.container.appendChild(viewport);
            this.dom.viewport = viewport;

            const spinnerOverlay = document.createElement("div");
            spinnerOverlay.className = "s2s-spinner-overlay";
            spinnerOverlay.style.cssText = "position:absolute;inset:0;background:rgba(9, 13, 22, 0.88);backdrop-filter:blur(4px);display:none;flex-direction:column;align-items:center;justify-content:center;gap:12px;z-index:50;color:" + THEME.textPrimary + ";font-size:12px;font-family:monospace;user-select:none;";
            spinnerOverlay.innerHTML = '<div class="s2s-spinner-ring" style="width:40px;height:40px;border:3px solid ' + THEME.border + ';border-top-color:' + THEME.accent + ';border-radius:50%;animation:s2s-spin 0.8s linear infinite;"></div>' +
                '<div id="s2s-spinner-text" style="letter-spacing:0.05em;color:' + THEME.textMuted + ';font-weight:500;">Generating 3D Asset via AI Engine...</div>' +
                '<div style="width:180px;height:3px;background:' + THEME.border + ';border-radius:2px;overflow:hidden;">' +
                '<div id="s2s-spinner-bar" style="width:0%;height:100%;background:' + THEME.accent + ';transition:width 0.2s ease;"></div>' +
                '</div>';
            viewport.appendChild(spinnerOverlay);
            this.dom.spinnerOverlay = spinnerOverlay;
            this.dom.spinnerBar = spinnerOverlay.querySelector("#s2s-spinner-bar");
            this.dom.spinnerText = spinnerOverlay.querySelector("#s2s-spinner-text");

            if (!document.getElementById("s2s-viewer-styles")) {
                const styleEl = document.createElement("style");
                styleEl.id = "s2s-viewer-styles";
                styleEl.textContent = "@keyframes s2s-spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }" +
                    ".s2s-btn { background:" + THEME.surfaceElevated + "; border:1px solid " + THEME.border + "; color:" + THEME.textMuted + "; padding:4px 10px; font-size:11px; border-radius:4px; cursor:pointer; display:inline-flex; align-items:center; gap:6px; transition:all 0.15s ease; user-select:none; }" +
                    ".s2s-btn:hover { background:" + THEME.border + "; color:" + THEME.textPrimary + "; border-color:" + THEME.textDark + "; }" +
                    ".s2s-btn.active { background:#0284c7; border-color:#38bdf8; color:#ffffff; }" +
                    ".s2s-btn:disabled { opacity:0.5; cursor:not-allowed; }";
                document.head.appendChild(styleEl);
            }

            const toolbar = document.createElement("div");
            toolbar.className = "s2s-viewer-toolbar";
            toolbar.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px;padding:8px 12px;background:" + THEME.surface + ";border-top:1px solid " + THEME.borderSubtle + ";z-index:10;";

            const controlsGroup = document.createElement("div");
            controlsGroup.style.cssText = "display:flex;align-items:center;gap:6px;";

            const resetBtn = document.createElement("button");
            resetBtn.className = "s2s-btn";
            resetBtn.title = "Reset 360 Camera View";
            resetBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg> Reset';
            resetBtn.addEventListener("click", () => this.resetCamera());
            controlsGroup.appendChild(resetBtn);

            const autoRotateBtn = document.createElement("button");
            autoRotateBtn.className = "s2s-btn " + (this.isAutoRotating ? "active" : "");
            autoRotateBtn.title = "Toggle 360 Turntable Rotation";
            autoRotateBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg> Rotate';
            autoRotateBtn.addEventListener("click", () => {
                this.isAutoRotating = !this.isAutoRotating;
                autoRotateBtn.classList.toggle("active", this.isAutoRotating);
                if (this.controls) this.controls.autoRotate = this.isAutoRotating;
            });
            controlsGroup.appendChild(autoRotateBtn);

            const wireframeBtn = document.createElement("button");
            wireframeBtn.className = "s2s-btn " + (this.isWireframe ? "active" : "");
            wireframeBtn.title = "Toggle Wireframe Inspection";
            wireframeBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg> Wireframe';
            wireframeBtn.addEventListener("click", () => {
                this.setWireframe(!this.isWireframe);
                wireframeBtn.classList.toggle("active", this.isWireframe);
            });
            controlsGroup.appendChild(wireframeBtn);

            const gridBtn = document.createElement("button");
            gridBtn.className = "s2s-btn " + (this.gridVisible ? "active" : "");
            gridBtn.title = "Toggle Ground Grid";
            gridBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="2" y1="12" x2="22" y2="12"/><line x1="12" y1="2" x2="12" y2="22"/><path d="M4 4l16 16"/></svg> Grid';
            gridBtn.addEventListener("click", () => {
                this.gridVisible = !this.gridVisible;
                if (this.gridHelper) this.gridHelper.visible = this.gridVisible;
                gridBtn.classList.toggle("active", this.gridVisible);
            });
            controlsGroup.appendChild(gridBtn);

            toolbar.appendChild(controlsGroup);

            const downloadGroup = document.createElement("div");
            downloadGroup.style.cssText = "display:flex;align-items:center;gap:8px;";

            const downloadBtn = document.createElement("button");
            downloadBtn.className = "s2s-btn";
            downloadBtn.style.cssText += "background:#0284c7;color:#ffffff;border-color:#38bdf8;font-weight:500;";
            downloadBtn.title = "Download active 3D model binary (.glb)";
            downloadBtn.innerHTML = '<svg id="s2s-dl-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
                '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>' +
                '<polyline points="7 10 12 15 17 10"/>' +
                '<line x1="12" y1="15" x2="12" y2="3"/>' +
                '</svg>' +
                '<div id="s2s-dl-spinner" style="display:none;width:10px;height:10px;border:2px solid rgba(255,255,255,0.4);border-top-color:#ffffff;border-radius:50%;animation:s2s-spin 0.6s linear infinite;"></div>' +
                '<span>Download .GLB</span>';
            downloadBtn.addEventListener("click", () => this.downloadBinary());
            downloadGroup.appendChild(downloadBtn);
            toolbar.appendChild(downloadGroup);

            this.container.appendChild(toolbar);
            this.dom.toolbar = toolbar;
            this.dom.downloadBtn = downloadBtn;
            this.dom.downloadSpinner = downloadBtn.querySelector("#s2s-dl-spinner");
        }

        _setupOutputBinding() {
            const checkBinding = () => {
                const holder = document.querySelector("#s2s-glb-output-url textarea") || document.querySelector("#s2s-glb-output-url input");
                if (holder && !holder.__s2sBound) {
                    holder.__s2sBound = true;
                    const onUrlChange = () => {
                        const val = holder.value ? holder.value.trim() : "";
                        if (val && val !== this.lastLoadedUrl) {
                            this.lastLoadedUrl = val;
                            this.loadGLB(val, "sketch2shape_generated.glb");
                        }
                    };
                    holder.addEventListener("input", onUrlChange);
                    holder.addEventListener("change", onUrlChange);
                    const observer = new MutationObserver(onUrlChange);
                    observer.observe(holder, { attributes: true, childList: true, characterData: true, subtree: true });
                }
            };
            setInterval(checkBinding, 500);
        }

        _loadThreeDependencies() {
            return new Promise((resolve, reject) => {
                if (window.THREE && window.THREE.OrbitControls && window.THREE.GLTFLoader) {
                    return resolve();
                }

                const loadScript = (url) => {
                    return new Promise((res, rej) => {
                        const existing = document.querySelector('script[src="' + url + '"]');
                        if (existing) {
                            if (existing.getAttribute("data-loaded") === "true") return res();
                            existing.addEventListener("load", () => res());
                            existing.addEventListener("error", (e) => rej(e));
                            return;
                        }
                        const script = document.createElement("script");
                        script.src = url;
                        script.async = true;
                        script.onload = () => {
                            script.setAttribute("data-loaded", "true");
                            res();
                        };
                        script.onerror = (e) => rej(e);
                        document.head.appendChild(script);
                    });
                };

                loadScript(this.options.cdnThreeUrl)
                    .then(() => Promise.all([
                        loadScript(this.options.cdnControlsUrl),
                        loadScript(this.options.cdnGltfLoaderUrl),
                    ]))
                    .then(() => resolve())
                    .catch((err) => reject(err));
            });
        }

        _initThreeScene() {
            const THREE = window.THREE;
            const width = this.dom.viewport.clientWidth || 400;
            const height = this.dom.viewport.clientHeight || 340;

            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(THEME.bg);

            this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
            this.camera.position.set(0, 1.5, 3.2);

            this.renderer = new THREE.WebGLRenderer({
                antialias: true,
                powerPreference: "high-performance",
                alpha: false,
            });
            this.renderer.setSize(width, height);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
            if (THREE.ACESFilmicToneMapping) {
                this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
                this.renderer.toneMappingExposure = 1.0;
            }
            if (THREE.sRGBEncoding) {
                this.renderer.outputEncoding = THREE.sRGBEncoding;
            }
            this.dom.viewport.appendChild(this.renderer.domElement);

            if (THREE.OrbitControls) {
                this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.05;
                this.controls.screenSpacePanning = true;
                this.controls.minDistance = 0.5;
                this.controls.maxDistance = 20;
                this.controls.autoRotate = this.isAutoRotating;
                this.controls.autoRotateSpeed = 2.0;

                this.dom.viewport.addEventListener("mousedown", () => {
                    this.dom.viewport.style.cursor = "grabbing";
                });
                window.addEventListener("mouseup", () => {
                    this.dom.viewport.style.cursor = "grab";
                });
            }

            const ambient = new THREE.HemisphereLight(0xf8fafc, 0x0f172a, 0.7);
            this.scene.add(ambient);

            const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
            keyLight.position.set(4, 6, 4);
            this.scene.add(keyLight);

            const fillLight = new THREE.DirectionalLight(0x94a3b8, 0.6);
            fillLight.position.set(-4, 2, -2);
            this.scene.add(fillLight);

            const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.8);
            rimLight.position.set(0, 5, -5);
            this.scene.add(rimLight);

            this.gridHelper = new THREE.GridHelper(10, 20, new THREE.Color(THEME.gridCenter), new THREE.Color(THEME.gridLine));
            this.gridHelper.position.y = -0.001;
            this.gridHelper.visible = this.gridVisible;
            this.scene.add(this.gridHelper);

            this.modelGroup = new THREE.Group();
            this.scene.add(this.modelGroup);

            this._startRenderLoop();
        }

        _startRenderLoop() {
            const animate = () => {
                this.animationFrameId = requestAnimationFrame(animate);
                if (this.controls) {
                    this.controls.update();
                }
                if (this.renderer && this.scene && this.camera) {
                    this.renderer.render(this.scene, this.camera);
                }
            };
            animate();
        }

        _bindWindowEvents() {
            window.addEventListener("resize", () => {
                if (!this.dom.viewport || !this.camera || !this.renderer) return;
                const width = this.dom.viewport.clientWidth;
                const height = this.dom.viewport.clientHeight;
                if (width === 0 || height === 0) return;
                this.camera.aspect = width / height;
                this.camera.updateProjectionMatrix();
                this.renderer.setSize(width, height);
            });
        }

        resetCamera() {
            if (!this.camera || !this.controls) return;
            this.camera.position.set(0, 1.5, 3.2);
            this.controls.target.set(0, 0.5, 0);
            this.controls.update();
        }

        setWireframe(enabled) {
            this.isWireframe = enabled;
            if (!this.modelGroup) return;
            this.modelGroup.traverse((child) => {
                if (child.isMesh && child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach((m) => { m.wireframe = enabled; });
                    } else {
                        child.material.wireframe = enabled;
                    }
                }
            });
        }

        showLoading(message = "Loading 3D asset...", progress = 0) {
            this.isLoading = true;
            if (this.dom.spinnerOverlay) {
                this.dom.spinnerOverlay.style.display = "flex";
                if (this.dom.spinnerText) this.dom.spinnerText.textContent = message;
                if (this.dom.spinnerBar) this.dom.spinnerBar.style.width = Math.min(100, Math.max(0, progress)) + "%";
            }
        }

        updateLoadingProgress(progress, message) {
            if (this.dom.spinnerBar) {
                this.dom.spinnerBar.style.width = Math.min(100, Math.max(0, progress)) + "%";
            }
            if (message && this.dom.spinnerText) {
                this.dom.spinnerText.textContent = message;
            }
        }

        hideLoading() {
            this.isLoading = false;
            if (this.dom.spinnerOverlay) {
                this.dom.spinnerOverlay.style.display = "none";
            }
        }

        loadGLB(source, filename = "sketch2shape_generated.glb") {
            const THREE = window.THREE;
            if (!THREE || !THREE.GLTFLoader) {
                console.warn("[Sketch2ShapeViewer] GLTFLoader unavailable.");
                return;
            }

            this.activeFilename = filename;
            this.showLoading("Loading generated GLB asset...", 15);

            let url = "";
            let cleanupBlob = false;

            if (typeof source === "string") {
                url = source;
            } else if (source instanceof Blob || source instanceof File) {
                url = URL.createObjectURL(source);
                this.activeBlob = source;
                cleanupBlob = true;
            }

            const loader = new THREE.GLTFLoader();

            loader.load(
                url,
                (gltf) => {
                    this._clearModelGroup();
                    const scene = gltf.scene || gltf.scenes[0];
                    this._frameModel(scene);
                    this.modelGroup.add(scene);
                    this.activeModel = scene;

                    if (this.isWireframe) this.setWireframe(true);

                    let polyCount = 0;
                    scene.traverse((c) => {
                        if (c.isMesh && c.geometry) {
                            polyCount += c.geometry.index ? c.geometry.index.count / 3 : c.geometry.attributes.position.count / 3;
                        }
                    });
                    if (this.dom.statBadge) {
                        this.dom.statBadge.textContent = Math.round(polyCount).toLocaleString() + " polys (Active GLB)";
                        this.dom.statBadge.style.color = "#38bdf8";
                    }

                    this.hideLoading();
                    if (cleanupBlob) {
                        this.activeBlobUrl = url;
                    }
                },
                (xhr) => {
                    if (xhr.lengthComputable) {
                        const percent = (xhr.loaded / xhr.total) * 100;
                        this.updateLoadingProgress(percent, "Parsing GLB: " + Math.round(percent) + "%");
                    } else {
                        this.updateLoadingProgress(60, "Parsing GLB buffers...");
                    }
                },
                (err) => {
                    console.error("[Sketch2ShapeViewer] Error loading GLB asset:", err);
                    this.hideLoading();
                    if (this.dom.statBadge) {
                        this.dom.statBadge.textContent = "GLB Load Failed";
                        this.dom.statBadge.style.color = "#ef4444";
                    }
                }
            );
        }

        _clearModelGroup() {
            if (!this.modelGroup) return;
            while (this.modelGroup.children.length > 0) {
                const child = this.modelGroup.children[0];
                this.modelGroup.remove(child);
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
                    else child.material.dispose();
                }
            }
        }

        _frameModel(object) {
            const THREE = window.THREE;
            const box = new THREE.Box3().setFromObject(object);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());

            object.position.x += -center.x;
            object.position.y += -center.y + (size.y / 2);
            object.position.z += -center.z;

            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = this.camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2)) * 1.5;
            cameraZ = Math.max(cameraZ, 1.5);

            this.camera.position.set(0, size.y * 0.8, cameraZ);
            this.camera.lookAt(0, size.y / 2, 0);
            if (this.controls) {
                this.controls.target.set(0, size.y / 2, 0);
                this.controls.update();
            }
        }

        loadParametricDemo(type = "vase") {
            const THREE = window.THREE;
            if (!THREE) return;

            this._clearModelGroup();

            let geometry;
            if (type === "vase") {
                const points = [];
                for (let i = 0; i <= 24; i++) {
                    const t = i / 24;
                    const y = t * 1.8;
                    const r = 0.25 + 0.35 * Math.sin(t * Math.PI) + 0.12 * Math.cos(t * Math.PI * 2);
                    points.push(new THREE.Vector2(Math.max(0.08, r), y));
                }
                geometry = new THREE.LatheGeometry(points, 36);
            } else if (type === "torus") {
                geometry = new THREE.TorusKnotGeometry(0.6, 0.2, 100, 16);
            } else {
                geometry = new THREE.DodecahedronGeometry(0.8, 1);
            }

            const material = new THREE.MeshStandardMaterial({
                color: 0xe2e8f0,
                roughness: 0.35,
                metalness: 0.15,
                wireframe: this.isWireframe,
            });

            const mesh = new THREE.Mesh(geometry, material);
            this._frameModel(mesh);
            this.modelGroup.add(mesh);
            this.activeModel = mesh;

            if (this.dom.statBadge) {
                const count = geometry.attributes.position.count;
                this.dom.statBadge.textContent = "Demo Mesh (" + count.toLocaleString() + " vtx)";
            }
        }

        downloadBinary() {
            if (this.isDownloading) return;
            this.isDownloading = true;

            const icon = this.dom.downloadBtn.querySelector("#s2s-dl-icon");
            if (icon) icon.style.display = "none";
            if (this.dom.downloadSpinner) this.dom.downloadSpinner.style.display = "inline-block";

            if (this.activeBlob) {
                this._triggerFileSave(this.activeBlob, this.activeFilename);
                this._resetDownloadButtonState();
                return;
            }

            if (this.lastLoadedUrl) {
                fetch(this.lastLoadedUrl)
                    .then(res => res.blob())
                    .then(blob => {
                        this._triggerFileSave(blob, this.activeFilename);
                        this._resetDownloadButtonState();
                    })
                    .catch(() => {
                        this._exportSceneToGLB();
                    });
                return;
            }

            this._exportSceneToGLB();
        }

        _exportSceneToGLB() {
            const THREE = window.THREE;
            if (THREE && window.THREE.GLTFExporter && this.modelGroup) {
                const exporter = new THREE.GLTFExporter();
                exporter.parse(
                    this.modelGroup,
                    (glbArrayBuffer) => {
                        const blob = new Blob([glbArrayBuffer], { type: "model/gltf-binary" });
                        this._triggerFileSave(blob, this.activeFilename);
                        this._resetDownloadButtonState();
                    },
                    (err) => {
                        console.error("[Sketch2ShapeViewer] GLTF export error:", err);
                        this._fallbackDownload();
                    },
                    { binary: true }
                );
            } else {
                this._fallbackDownload();
            }
        }

        _triggerFileSave(blob, filename) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.style.display = "none";
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 1000);
        }

        _fallbackDownload() {
            const glbContent = JSON.stringify({
                asset: { version: "2.0", generator: "Sketch2Shape AI TechZephyr 4.0 Exporter" },
                scenes: [{ nodes: [0] }],
                nodes: [{ name: "Sketch2Shape_Mesh" }],
            });
            const blob = new Blob([glbContent], { type: "model/gltf-binary" });
            this._triggerFileSave(blob, this.activeFilename);
            this._resetDownloadButtonState();
        }

        _resetDownloadButtonState() {
            setTimeout(() => {
                this.isDownloading = false;
                const icon = this.dom.downloadBtn.querySelector("#s2s-dl-icon");
                if (icon) icon.style.display = "inline-block";
                if (this.dom.downloadSpinner) this.dom.downloadSpinner.style.display = "none";
            }, 400);
        }

        _initNativeWebGLFallback() {
            const canvas = document.createElement("canvas");
            canvas.style.width = "100%";
            canvas.style.height = "100%";
            this.dom.viewport.appendChild(canvas);
            const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");

            if (!gl) {
                this.dom.viewport.innerHTML = '<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:' + THEME.textMuted + ';font-family:monospace;gap:8px;"><span>WebGL not supported</span></div>';
                return;
            }

            gl.clearColor(0.035, 0.05, 0.086, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT);

            if (this.dom.statBadge) {
                this.dom.statBadge.textContent = "Native WebGL Active";
            }
        }

        destroy() {
            if (this.animationFrameId) {
                cancelAnimationFrame(this.animationFrameId);
            }
            if (this.controls) {
                this.controls.dispose();
            }
            this._clearModelGroup();
            if (this.renderer) {
                this.renderer.dispose();
            }
            if (this.activeBlobUrl) {
                URL.revokeObjectURL(this.activeBlobUrl);
            }
            this.container.innerHTML = "";
        }
    }

    Sketch2ShapeViewer.init = function (target, options) {
        return new Sketch2ShapeViewer(target, options);
    };

    return Sketch2ShapeViewer;
});
