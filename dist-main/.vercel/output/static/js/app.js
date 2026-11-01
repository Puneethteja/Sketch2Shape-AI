/**
 * Sketch2Shape AI — SPA Main Application Entry
 * Boots the router, registers views, and mounts the navigation bar.
 */

import { Router } from "./router.js";
import { renderNavbar } from "./components/navbar.js";
import { createStudioView } from "./views/studioView.js";
import { createGalleryView } from "./views/galleryView.js";
import { createPipelineView } from "./views/pipelineView.js";
import { createBenchmarksView } from "./views/benchmarksView.js";
import { createSettingsView } from "./views/settingsView.js";
import { createNotFoundView } from "./views/notFoundView.js";

// Route Definition Table
const ROUTES = {
  "#/studio": {
    title: "3D Synthesis Studio",
    view: createStudioView,
  },
  "#/gallery": {
    title: "3D Asset Gallery",
    view: createGalleryView,
  },
  "#/pipeline": {
    title: "AI Pipeline Architecture",
    view: createPipelineView,
  },
  "#/benchmarks": {
    title: "Resilience & Benchmarks",
    view: createBenchmarksView,
  },
  "#/settings": {
    title: "Settings & Diagnostics",
    view: createSettingsView,
  },
  "404": {
    title: "Page Not Found",
    view: createNotFoundView,
  },
};

// Initialize on DOMContentLoaded
window.addEventListener("DOMContentLoaded", () => {
  // Render sticky top navigation
  renderNavbar();

  // Instantiate and run the client router
  const router = new Router(ROUTES, "view-container");
  router.init();

  console.log("%c[Sketch2Shape AI] SPA Loaded Successfully", "color: #06b6d4; font-weight: bold; font-size: 14px;");
});
