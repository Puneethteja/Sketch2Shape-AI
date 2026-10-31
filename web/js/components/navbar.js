/**
 * Sketch2Shape AI — Left Sidebar Navigation Component
 * Sleek, modern vertical navigation bar for Microsoft TRELLIS.2-4B Studio.
 */

import { store } from "../state.js";

export function renderNavbar() {
  const root = document.getElementById("sidebar-root") || document.getElementById("navbar-root");
  if (!root) return;

  const currentTheme = store.getState().theme;
  const hash = window.location.hash || "#/studio";

  root.innerHTML = `
    <div class="sidebar-inner">
      <!-- Brand Logo Header -->
      <div class="sidebar-brand-wrap">
        <a href="#/studio" class="sidebar-brand">
          <div class="brand-cube-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
              <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
              <line x1="12" y1="22.08" x2="12" y2="12"/>
            </svg>
          </div>
          <div class="brand-info">
            <span class="brand-name">Sketch2Shape</span>
            <span class="brand-engine-tag">TRELLIS.2-4B</span>
          </div>
        </a>
      </div>

      <!-- Main Navigation Menu -->
      <nav class="sidebar-nav">
        <div class="nav-section-label">WORKSPACE</div>
        <ul class="nav-links-list">
          <li>
            <a href="#/studio" class="nav-item ${hash === '#/studio' || hash === '' ? 'active' : ''}">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
              <span>Studio</span>
            </a>
          </li>
          <li>
            <a href="#/model" class="nav-item ${hash === '#/model' ? 'active' : ''}">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>
              <span>Model & Metrics</span>
            </a>
          </li>
          <li>
            <a href="#/gallery" class="nav-item ${hash === '#/gallery' ? 'active' : ''}">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
              <span>3D Gallery</span>
            </a>
          </li>
        </ul>

        <div class="nav-section-label" style="margin-top: 24px;">DIAGNOSTICS</div>
        <ul class="nav-links-list">
          <li>
            <a href="#/metrics" class="nav-item ${hash === '#/metrics' ? 'active' : ''}">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
              <span>Verification</span>
            </a>
          </li>
        </ul>
      </nav>

      <!-- Sidebar Footer Details -->
      <div class="sidebar-footer">
        <!-- Supported 4 Formats Chips -->
        <div class="sidebar-formats-pill">
          <span class="pill-label">Formats:</span>
          <div class="pill-badges">
            <span class="fmt-chip">STL</span>
            <span class="fmt-chip">OBJ</span>
            <span class="fmt-chip">GLB</span>
            <span class="fmt-chip">PLY</span>
          </div>
        </div>

        <!-- Theme Toggle -->
        <div class="sidebar-theme-row">
          <button id="btn-theme-toggle" class="sidebar-theme-btn" title="Toggle Light / Dark Interface">
            <span id="theme-icon">
              ${currentTheme === "dark"
                ? `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg><span>Light Mode</span>`
                : `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg><span>Dark Mode</span>`}
            </span>
          </button>
        </div>

        <div class="sidebar-version-tag">
          <span>Microsoft TRELLIS.2-4B</span>
        </div>
      </div>
    </div>
  `;

  // Attach theme toggle listener
  const toggleBtn = document.getElementById("btn-theme-toggle");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      store.toggleTheme();
      renderNavbar();
      store.showToast(`Switched to ${store.getState().theme} theme`, "info", 1500);
    });
  }
}
