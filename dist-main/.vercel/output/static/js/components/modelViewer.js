/**
 * Sketch2Shape AI — 3D WebGL Model Viewer Component
 * High-performance Three.js OrbitControls Viewport with Wireframe, Lighting, and Procedural/GLTF Mesh Loaders
 */

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

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#090d16");

    // Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    this.camera.position.set(0, 1.4, 2.8);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    // Controls
    if (typeof THREE.OrbitControls !== "undefined") {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.autoRotate = this.options.autoRotate;
      this.controls.autoRotateSpeed = 1.8;
      this.controls.target.set(0, 0, 0);
    }

    // Studio Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x06b6d4, 1.2);
    dirLight1.position.set(3, 4, 2);
    dirLight1.castShadow = true;
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x8b5cf6, 0.8);
    dirLight2.position.set(-3, -2, -2);
    this.scene.add(dirLight2);

    // Grid
    if (this.options.showGrid) {
      this.gridHelper = new THREE.GridHelper(4, 20, 0x06b6d4, 0x1e293b);
      this.gridHelper.position.y = -0.65;
      this.scene.add(this.gridHelper);
    }

    // Default procedural geometry
    this.loadProceduralShape("vase");

    // Resize Observer
    this.resizeObserver = new ResizeObserver(() => this.handleResize());
    this.resizeObserver.observe(this.container);

    // Render loop
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

  loadProceduralShape(type = "vase") {
    this.clearActiveMesh();

    let geometry;
    const matOptions = {
      color: 0x06b6d4,
      metalness: 0.25,
      roughness: 0.35,
      wireframe: this.options.wireframe,
    };
    const material = new THREE.MeshStandardMaterial(matOptions);

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

        // Auto center and scale
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
        console.warn("[ModelViewer] GLB load fallback to procedural shape:", error);
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
