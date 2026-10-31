/**
 * Sketch2Shape AI — Model & Metrics View
 * Based on Microsoft Research's TRELLIS.2: "Native and Compact Structured Latents for 3D Generation"
 * Paper: arXiv:2512.14692 | Weights: microsoft/TRELLIS.2-4B
 */

import { store } from "../state.js";

export function createModelView() {
  return {
    async render() {
      return `
        <div class="model-view-container">
          <!-- Model Header Pitch -->
          <div class="model-hero-card">
            <div class="hero-badge">
              <span class="badge-dot"></span>
              <span>Microsoft Research &middot; arXiv:2512.14692</span>
            </div>
            <h1 class="model-title">Microsoft TRELLIS.2 (TRELLIS.2-4B)</h1>
            <p class="model-description">
              <strong>TRELLIS.2</strong> is a state-of-the-art large 3D generative model with <strong>4 Billion parameters</strong>
              designed for high-fidelity image-to-3D generation. It leverages a novel "field-free" sparse voxel structure
              termed <strong>O-Voxel</strong> with a Sparse 3D VAE utilizing 16&times; spatial downsampling to encode assets
              into a compact latent space. It natively generates arbitrary 3D assets with complex topologies, sharp features,
              and full PBR materials (Base Color, Roughness, Metallic, Opacity).
            </p>
          </div>

          <!-- Official Paper Benchmarks (Tested on NVIDIA H100) -->
          <div class="model-specs-section">
            <h2 class="section-heading">Inference Performance (Official Benchmarks)</h2>
            <div class="specs-table-wrap">
              <table class="specs-table">
                <thead>
                  <tr>
                    <th>Resolution</th>
                    <th>Total Time</th>
                    <th>Breakdown (Shape + Material)</th>
                    <th>Pipeline Stage</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>512&sup3;</strong></td>
                    <td><strong>~3.0s</strong></td>
                    <td>2.0s Shape + 1.0s Material</td>
                    <td>Real-Time Interactive Synthesis</td>
                  </tr>
                  <tr>
                    <td><strong>1024&sup3;</strong></td>
                    <td><strong>~17.0s</strong></td>
                    <td>10.0s Shape + 7.0s Material</td>
                    <td>High-Resolution Production Mesh</td>
                  </tr>
                  <tr>
                    <td><strong>1536&sup3;</strong></td>
                    <td><strong>~60.0s</strong></td>
                    <td>35.0s Shape + 25.0s Material</td>
                    <td>Ultra-Detail Master Asset</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- 4-Format Output Delivery Strip -->
          <div class="formats-overview-grid">
            <div class="format-card">
              <div class="format-badge">3D PRINTING</div>
              <h3 class="format-name">Stereolithography (.STL)</h3>
              <p class="format-desc">Directly sliceable by PrusaSlicer, Cura, and Bambu Studio. Guaranteed watertight and non-manifold free.</p>
            </div>
            <div class="format-card">
              <div class="format-badge">CAD & DCC</div>
              <h3 class="format-name">Wavefront (.OBJ)</h3>
              <p class="format-desc">Polygonal geometry with surface normals, compatible with Blender, Rhino, SolidWorks, and Maya.</p>
            </div>
            <div class="format-card">
              <div class="format-badge">REAL-TIME PBR</div>
              <h3 class="format-name">Binary glTF (.GLB)</h3>
              <p class="format-desc">Embedded PBR material attributes (Base Color, Roughness, Metallic) for Three.js, Unity, and Unreal Engine.</p>
            </div>
            <div class="format-card">
              <div class="format-badge">METROLOGY</div>
              <h3 class="format-name">Polygon File (.PLY)</h3>
              <p class="format-desc">Vertex arrays with color channels and normal vectors for point-cloud processing and quality inspection.</p>
            </div>
          </div>

          <!-- O-Voxel Key Innovations -->
          <div class="model-specs-section">
            <h2 class="section-heading">O-Voxel Architecture Innovations</h2>
            <div class="specs-table-wrap">
              <table class="specs-table">
                <thead>
                  <tr>
                    <th>Capability</th>
                    <th>O-Voxel Advantage</th>
                    <th>Traditional Iso-Surface Limit</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Open Surfaces</strong></td>
                    <td>Natively models thin clothing, leaves, and open shells</td>
                    <td>Forced closure or catastrophic mesh tear</td>
                  </tr>
                  <tr>
                    <td><strong>Non-Manifold Geometry</strong></td>
                    <td>Directly represents T-junctions and shared edges without conversion loss</td>
                    <td>Implicit field reconstruction failure</td>
                  </tr>
                  <tr>
                    <td><strong>Internal Structures</strong></td>
                    <td>Preserves hollow cavities and enclosed internal volumes</td>
                    <td>Internal voids frequently filled or erased</td>
                  </tr>
                  <tr>
                    <td><strong>Full PBR Modeling</strong></td>
                    <td>Predicts Base Color, Roughness, Metallic, and Opacity</td>
                    <td>Only diffuse RGB color supported</td>
                  </tr>
                  <tr>
                    <td><strong>Data Processing Speed</strong></td>
                    <td>Textured Mesh &rarr; O-Voxel &lt; 10s | O-Voxel &rarr; Mesh &lt; 100ms</td>
                    <td>Minutes of iterative marching cubes optimization</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- 3 Native Material Systems -->
          <div class="materials-showcase-section">
            <h2 class="section-heading">3 Native PBR Material Systems</h2>
            <div class="materials-grid">
              <div class="material-spec-card">
                <div class="material-color-swatch" style="background: #f2f4f8; border: 1px solid #cbd5e1;"></div>
                <h3 class="material-title">Porcelain Ceramic</h3>
                <p class="material-details">
                  Smooth specular white glaze with high surface reflectivity (Roughness 0.18, Metalness 0.05, Opacity 1.0).
                  Ideal for homeware, pottery profiles, and tableware.
                </p>
              </div>

              <div class="material-spec-card">
                <div class="material-color-swatch" style="background: #2d3440;"></div>
                <h3 class="material-title">Slate Polymer</h3>
                <p class="material-details">
                  Industrial matte resin with controlled micro-roughness (Roughness 0.72, Metalness 0.12, Opacity 1.0).
                  Accentuates geometric contours and crisp chamfers for product shells and enclosures.
                </p>
              </div>

              <div class="material-spec-card">
                <div class="material-color-swatch" style="background: #b6864c;"></div>
                <h3 class="material-title">Cast Bronze</h3>
                <p class="material-details">
                  Warm antique metallic luster with rich copper undertones (Roughness 0.38, Metalness 0.85, Opacity 1.0).
                  Designed for architectural hardware, artistic sculptures, and precision metal fixtures.
                </p>
              </div>
            </div>
          </div>
        </div>
      `;
    },

    afterRender() {},
  };
}
