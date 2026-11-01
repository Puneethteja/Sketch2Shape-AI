const MATERIAL_DEFS = {
  porcelain: { color: 0xf8fafc, roughness: 0.15, metalness: 0.05, name: "Porcelain" },
  gold: { color: 0xf59e0b, roughness: 0.22, metalness: 0.95, name: "Brushed Gold" },
  titanium: { color: 0x94a3b8, roughness: 0.35, metalness: 0.85, name: "Titanium" },
  walnut: { color: 0x78350f, roughness: 0.70, metalness: 0.08, name: "Walnut Wood" },
  terracotta: { color: 0xc2410c, roughness: 0.85, metalness: 0.04, name: "Terracotta" },
  glass: { color: 0x38bdf8, roughness: 0.10, metalness: 0.15, transparent: true, opacity: 0.75, name: "Frosted Glass" },
  obsidian: { color: 0x0f172a, roughness: 0.12, metalness: 0.80, name: "Obsidian" },
  coral: { color: 0xfb7185, roughness: 0.38, metalness: 0.15, name: "Pastel Coral" }
};

export class ModelViewer {
  constructor(containerElement, options = {}) {
    this.container = containerElement;
    this.options = {
      autoRotate: options.autoRotate || false,
      showGrid: options.showGrid !== false,
      wireframe: options.wireframe || false,
      ...options
    };

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.activeMesh = null;
    this.gridHelper = null;
    this.animationFrameId = null;
    this.currentMaterial = options.material || "porcelain";

    this.init();
  }

  init() {
    if (typeof THREE === "undefined") {
      this.container.innerHTML = `
        <div style="display:flex;height:100%;align-items:center;justify-content:center;color:var(--text-muted);font-size:12px;">
          WebGL library loading...
        </div>
      `;
      return;
    }

    const width = this.container.clientWidth || 480;
    const height = this.container.clientHeight || 480;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#090d16");

    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    this.camera.position.set(0, 1.4, 2.8);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    if (typeof THREE.OrbitControls !== "undefined") {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.autoRotate = this.options.autoRotate;
      this.controls.autoRotateSpeed = 1.8;
      this.controls.target.set(0, 0, 0);
    }

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x06b6d4, 1.2);
    dirLight1.position.set(3, 4, 2);
    dirLight1.castShadow = true;
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x8b5cf6, 0.8);
    dirLight2.position.set(-3, -2, -2);
    this.scene.add(dirLight2);

    if (this.options.showGrid) {
      this.gridHelper = new THREE.GridHelper(4, 20, 0x06b6d4, 0x1e293b);
      this.gridHelper.position.y = -0.65;
      this.scene.add(this.gridHelper);
    }

    this.loadProceduralShape("vase");

    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(this.container);

    this.animate = this.animate.bind(this);
    this.animate();
  }

  handleResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate() {
    this.animationFrameId = requestAnimationFrame(this.animate);
    if (this.controls) this.controls.update();
    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  clearActiveMesh() {
    if (this.activeMesh) {
      this.scene.remove(this.activeMesh);
      if (this.activeMesh.geometry) this.activeMesh.geometry.dispose();
      if (this.activeMesh.material) {
        if (Array.isArray(this.activeMesh.material)) {
          this.activeMesh.material.forEach(m => m.dispose());
        } else {
          this.activeMesh.material.dispose();
        }
      }
      this.activeMesh = null;
    }
  }

  getMaterialInstance(key = null) {
    const matKey = key || this.currentMaterial || "porcelain";
    const def = MATERIAL_DEFS[matKey] || MATERIAL_DEFS.porcelain;
    return new THREE.MeshStandardMaterial({
      color: def.color,
      roughness: def.roughness,
      metalness: def.metalness,
      wireframe: this.options.wireframe,
      transparent: Boolean(def.transparent),
      opacity: def.opacity !== undefined ? def.opacity : 1.0
    });
  }

  setMaterial(materialKey) {
    if (!MATERIAL_DEFS[materialKey]) return;
    this.currentMaterial = materialKey;
    if (this.activeMesh) {
      const newMat = this.getMaterialInstance(materialKey);
      this.activeMesh.traverse((child) => {
        if (child.isMesh) {
          child.material = newMat;
        }
      });
    }
  }

  loadProceduralShape(type = "vase") {
    this.clearActiveMesh();

    let geometry;
    const material = this.getMaterialInstance();

    if (type === "vase") {
      const points = [];
      for (let i = 0; i < 20; i++) {
        const y = (i / 19) * 1.3 - 0.65;
        const radius = Math.sin((i / 19) * Math.PI) * 0.45 + 0.15;
        points.push(new THREE.Vector2(radius, y));
      }
      geometry = new THREE.LatheGeometry(points, 32);
    } else if (type === "cylinder" || type === "mug") {
      geometry = new THREE.CylinderGeometry(0.45, 0.45, 0.9, 32);
    } else if (type === "box" || type === "chair") {
      geometry = new THREE.BoxGeometry(0.8, 0.9, 0.8, 4, 4, 4);
    } else if (type === "cone" || type === "bottle") {
      geometry = new THREE.CylinderGeometry(0.18, 0.42, 1.2, 32);
    } else if (type === "sphere" || type === "bowl") {
      geometry = new THREE.SphereGeometry(0.65, 32, 24, 0, Math.PI * 2, 0, Math.PI / 1.8);
    } else {
      geometry = new THREE.TorusKnotGeometry(0.4, 0.14, 64, 16);
    }

    geometry.computeVertexNormals();
    this.activeMesh = new THREE.Mesh(geometry, material);
    this.activeMesh.castShadow = true;
    this.activeMesh.receiveShadow = true;
    this.scene.add(this.activeMesh);

    return {
      vertices: geometry.attributes.position.count,
      faces: geometry.index ? geometry.index.count / 3 : geometry.attributes.position.count / 3,
      manifold: true
    };
  }

  loadGLB(url) {
    if (!url || typeof THREE.GLTFLoader === "undefined") return;

    const loader = new THREE.GLTFLoader();
    loader.load(
      url,
      (gltf) => {
        this.clearActiveMesh();
        const model = gltf.scene;

        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 1.3 / maxDim;

        model.position.sub(center.multiplyScalar(scale));
        model.scale.multiplyScalar(scale);

        model.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            child.material.wireframe = this.options.wireframe;
          }
        });

        this.activeMesh = model;
        this.scene.add(this.activeMesh);
      },
      undefined,
      (error) => {
        this.loadProceduralShape("vase");
      }
    );
  }

  setWireframe(enabled) {
    this.options.wireframe = enabled;
    if (this.activeMesh) {
      this.activeMesh.traverse((child) => {
        if (child.isMesh && child.material) {
          child.material.wireframe = enabled;
        }
      });
    }
  }

  setAutoRotate(enabled) {
    this.options.autoRotate = enabled;
    if (this.controls) {
      this.controls.autoRotate = enabled;
    }
  }

  resetCamera() {
    if (this.camera && this.controls) {
      this.camera.position.set(0, 1.4, 2.8);
      this.controls.target.set(0, 0, 0);
      this.controls.update();
    }
  }

  destroy() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }
    this.clearActiveMesh();
    if (this.renderer && this.renderer.domElement) {
      this.renderer.domElement.remove();
      this.renderer.dispose();
    }
  }
}
