/**
 * Sketch2Shape AI — Navbar Component
 */

import { store } from "../state.js";

export function renderNavbar() {
  const root = document.getElementById("navbar-root");
  if (!root) return;

  const currentTheme = store.getState().theme;

  root.innerHTML = `
    <nav class="navbar" role="navigation" aria-label="Main Navigation">
      <div class="navbar-inner">
        <!-- Brand / Logo -->
        <a href="#/studio" class="navbar-brand">
          <div class="brand-icon">📐</div>
          <div class="brand-text">
            <span class="brand-title">Sketch2Shape AI</span>
            <span class="brand-subtitle">TechZephyr 4.0 &middot; IIT BBS</span>
          </div>
        </a>

        <!-- Nav Links -->
        <ul class="navbar-nav">
          <li>
            <a href="#/studio" class="nav-link">
              <span>✏️</span> Studio
            </a>
          </li>
          <li>
            <a href="#/gallery" class="nav-link">
              <span>🏛️</span> 3D Gallery
            </a>
          </li>
          <li>
            <a href="#/pipeline" class="nav-link">
              <span>⚡</span> Architecture
            </a>
          </li>
          <li>
            <a href="#/benchmarks" class="nav-link">
              <span>📊</span> Benchmarks
            </a>
          </li>
          <li>
            <a href="#/settings" class="nav-link">
              <span>⚙️</span> Settings
            </a>
          </li>
        </ul>

        <!-- Right Actions: Theme Toggle -->
        <div class="navbar-actions">
          <button id="btn-theme-toggle" class="btn-icon" title="Toggle Dark / Light Theme" aria-label="Toggle Theme">
            <span id="theme-icon">${currentTheme === "dark" ? "☀️" : "🌙"}</span>
          </button>
        </div>
      </div>
    </nav>
  `;

  // Attach theme toggle listener
  const toggleBtn = document.getElementById("btn-theme-toggle");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      store.toggleTheme();
      const nextTheme = store.getState().theme;
      const icon = document.getElementById("theme-icon");
      if (icon) icon.textContent = nextTheme === "dark" ? "☀️" : "🌙";
      store.showToast(`Switched to ${nextTheme} mode`, "info", 1500);
    });
  }
}
