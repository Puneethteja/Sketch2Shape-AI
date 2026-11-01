/**
 * Sketch2Shape AI — Settings & Diagnostics View
 */

import { store } from "../state.js";

export function createSettingsView() {
  return {
    async render() {
      const state = store.getState();
      const params = state.params;

      return `
        <div class="settings-container" style="max-width: 800px; margin: 0 auto;">
          <div style="margin-bottom: 24px;">
            <h1 style="font-size: 24px; font-weight: 800; letter-spacing: -0.02em;">Settings & System Diagnostics</h1>
            <p style="font-size: 13px; color: var(--text-secondary); margin-top: 4px;">
              Manage local application preferences, runtime precision modes, and storage cache.
            </p>
          </div>

          <!-- Preferences Card -->
          <div class="panel-card" style="margin-bottom: 20px;">
            <h3 style="font-size: 16px; font-weight: 700; border-bottom: 1px solid var(--border-subtle); padding-bottom: 10px;">
              Appearance & Theme
            </h3>
            <div style="display: flex; align-items: center; justify-content: space-between;">
              <div>
                <div style="font-weight: 600; font-size: 13px;">Color Scheme</div>
                <div style="font-size: 12px; color: var(--text-secondary);">Toggle between Slate Dark and Crisp Light themes</div>
              </div>
              <div style="display: flex; gap: 8px;">
                <button id="btn-set-dark" class="tool-btn ${state.theme === 'dark' ? 'active' : ''}">🌙 Dark Mode</button>
                <button id="btn-set-light" class="tool-btn ${state.theme === 'light' ? 'active' : ''}">☀️ Light Mode</button>
              </div>
            </div>
          </div>

          <!-- Inference Runtime Card -->
          <div class="panel-card" style="margin-bottom: 20px;">
            <h3 style="font-size: 16px; font-weight: 700; border-bottom: 1px solid var(--border-subtle); padding-bottom: 10px;">
              Inference Hardware Configuration
            </h3>
            <div class="form-group">
              <label class="form-label" for="setting-precision">Tensor Compute Precision</label>
              <select id="setting-precision" class="form-select">
                <option value="FP16" ${params.precision === 'FP16' ? 'selected' : ''}>FP16 (Half-Precision) — Recommended for 16GB GPUs</option>
                <option value="FP32" ${params.precision === 'FP32' ? 'selected' : ''}>FP32 (Single-Precision) — Higher fidelity, higher VRAM</option>
              </select>
            </div>

            <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 10px;">
              <div>
                <div style="font-weight: 600; font-size: 13px;">Automated Non-Manifold Mesh Repair</div>
                <div style="font-size: 12px; color: var(--text-secondary);">Automatically fill holes and strip unattached floating polygons</div>
              </div>
              <input type="checkbox" id="setting-autorepair" ${params.autoRepair ? 'checked' : ''} style="width: 18px; height: 18px; accent-color: var(--accent-cyan); cursor: pointer;">
            </div>
          </div>

          <!-- Storage & Cache Reset -->
          <div class="panel-card">
            <h3 style="font-size: 16px; font-weight: 700; border-bottom: 1px solid var(--border-subtle); padding-bottom: 10px;">
              Diagnostics & Local Cache
            </h3>
            <div style="display: flex; align-items: center; justify-content: space-between;">
              <div>
                <div style="font-weight: 600; font-size: 13px;">Clear User Generation History</div>
                <div style="font-size: 12px; color: var(--text-secondary);">Remove all locally saved 3D assets from browser storage</div>
              </div>
              <button id="btn-clear-cache" class="btn btn-danger btn-sm">
                🗑️ Clear History
              </button>
            </div>

            <div style="margin-top: 16px; padding: 12px; background: var(--bg-secondary); border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
              <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 12px; font-weight: 600;">3D Generation Provider</span>
                <button id="btn-refresh-provider-status" class="tool-btn btn-sm">Refresh Status</button>
              </div>
              <div id="backend-status-badge" class="badge" style="margin-top:8px">Checking backend…</div>
              <p id="provider-status-detail" style="font-size:11px;color:var(--text-secondary);margin-top:8px">Provider and model configuration are read from the backend.</p>
              <p style="font-size:11px;color:var(--text-muted);margin-top:8px">Choose a provider in PowerShell: <code>.\\scripts\\configure_meshy.ps1</code> for Meshy or <code>.\\scripts\\configure_tencent.ps1</code> for Tencent, then run <code>.\\scripts\\start_local.ps1</code>. Keys stay in the ignored <code>.env</code> file. Meshy generation can consume account credits.</p>
            </div>
          </div>
        </div>
      `;
    },

    afterRender() {
      const btnDark = document.getElementById("btn-set-dark");
      const btnLight = document.getElementById("btn-set-light");

      if (btnDark) {
        btnDark.addEventListener("click", () => {
          store.setTheme("dark");
          btnDark.classList.add("active");
          btnLight.classList.remove("active");
          const icon = document.getElementById("theme-icon");
          if (icon) icon.textContent = "☀️";
          store.showToast("Set Dark theme", "info", 1500);
        });
      }

      if (btnLight) {
        btnLight.addEventListener("click", () => {
          store.setTheme("light");
          btnLight.classList.add("active");
          btnDark.classList.remove("active");
          const icon = document.getElementById("theme-icon");
          if (icon) icon.textContent = "🌙";
          store.showToast("Set Light theme", "info", 1500);
        });
      }

      const precisionSelect = document.getElementById("setting-precision");
      if (precisionSelect) {
        precisionSelect.addEventListener("change", (e) => {
          store.setParams({ precision: e.target.value });
          store.showToast(`Precision set to ${e.target.value}`, "info", 2000);
        });
      }

      const repairCheck = document.getElementById("setting-autorepair");
      if (repairCheck) {
        repairCheck.addEventListener("change", (e) => {
          store.setParams({ autoRepair: e.target.checked });
          store.showToast(`Auto-repair ${e.target.checked ? "enabled" : "disabled"}`, "info", 2000);
        });
      }

      const btnClearCache = document.getElementById("btn-clear-cache");
      if (btnClearCache) {
        btnClearCache.addEventListener("click", () => {
          store.clearGeneratedAssets();
          store.showToast("User asset cache cleared", "success", 2000);
        });
      }

      const refreshProviderStatus = async () => {
        const badge = document.getElementById("backend-status-badge");
        const detail = document.getElementById("provider-status-detail");
        const button = document.getElementById("btn-refresh-provider-status");
        if (button) button.disabled = true;
        try {
          const response = await fetch("/health", { cache: "no-store" });
          const health = await response.json();
          if (!response.ok) throw new Error(health.detail || `Backend returned ${response.status}`);
          const activeProvider = health.provider_setting || "tokenhub";
          const configured = activeProvider === "meshy"
            ? Boolean(health.meshy_configured)
            : activeProvider === "tokenhub"
              ? Boolean(health.tokenhub_configured)
              : true;
          if (badge) {
            badge.className = configured ? "badge badge-success" : "badge badge-amber";
            badge.textContent = configured
              ? `${health.generation_provider} configured · authentication checked on generation`
              : `${health.generation_provider} API key missing`;
          }
          if (detail) detail.textContent = `Backend: ${health.status} · Provider: ${health.generation_provider} · Model: ${health.model}. Credentials are never returned to the browser.`;
        } catch (error) {
          if (badge) { badge.className = "badge badge-amber"; badge.textContent = "Backend unreachable"; }
          if (detail) detail.textContent = error.message || "Could not reach the Sketch2Shape backend.";
        } finally { if (button) button.disabled = false; }
      };
      document.getElementById("btn-refresh-provider-status")?.addEventListener("click", refreshProviderStatus);
      refreshProviderStatus();
    }
  };
}
