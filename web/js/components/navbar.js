import { store } from "../state.js";

export function renderNavbar() {
  const root = document.getElementById("navbar-root");
  if (!root) return;

  const currentTheme = store.getState().theme;

  root.innerHTML = `
    <nav class="sidebar" role="navigation" aria-label="Main Navigation">
      <div class="sidebar-top">
        <a href="#/studio" class="sidebar-brand">
          <div class="brand-icon">📐</div>
          <div class="brand-text">
            <span class="brand-title">Sketch2Shape AI</span>
            <span class="brand-subtitle">Generative 3D Studio</span>
          </div>
        </a>

        <div class="sidebar-section-title">Navigation</div>
        <ul class="sidebar-nav">
          <li>
            <a href="#/studio" class="nav-link">
              <span class="nav-icon">✏️</span>
              <span class="nav-label">Studio</span>
            </a>
          </li>
          <li>
            <a href="#/gallery" class="nav-link">
              <span class="nav-icon">🏛️</span>
              <span class="nav-label">3D Gallery</span>
            </a>
          </li>
          <li>
            <a href="#/analysis" class="nav-link">
              <span class="nav-icon">📐</span>
              <span class="nav-label">Model &amp; Metrics</span>
            </a>
          </li>
          <li>
            <a href="#/pipeline" class="nav-link">
              <span class="nav-icon">⚡</span>
              <span class="nav-label">Architecture</span>
            </a>
          </li>
          <li>
            <a href="#/benchmarks" class="nav-link">
              <span class="nav-icon">📊</span>
              <span class="nav-label">Benchmarks</span>
            </a>
          </li>
          <li>
            <a href="#/settings" class="nav-link">
              <span class="nav-icon">⚙️</span>
              <span class="nav-label">Settings</span>
            </a>
          </li>
        </ul>
      </div>

      <div class="sidebar-bottom">
        <div class="sidebar-status-card">
          <span class="badge badge-success" style="width: 100%; justify-content: center;">
            <span class="badge-dot"></span> Manifold Engine Online
          </span>
        </div>

        <button id="btn-theme-toggle" class="sidebar-theme-btn" title="Toggle Dark / Light Theme" aria-label="Toggle Theme">
          <span id="theme-icon">${currentTheme === "dark" ? "☀️" : "🌙"}</span>
          <span id="theme-label">${currentTheme === "dark" ? "Light Mode" : "Dark Mode"}</span>
        </button>
      </div>
    </nav>
  `;

  const toggleBtn = document.getElementById("btn-theme-toggle");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      store.toggleTheme();
      const nextTheme = store.getState().theme;
      const icon = document.getElementById("theme-icon");
      const label = document.getElementById("theme-label");
      if (icon) icon.textContent = nextTheme === "dark" ? "☀️" : "🌙";
      if (label) label.textContent = nextTheme === "dark" ? "Light Mode" : "Dark Mode";
      store.showToast(`Switched to ${nextTheme} mode`, "info", 1500);
    });
  }
}
