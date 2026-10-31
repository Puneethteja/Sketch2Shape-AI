/**
 * Sketch2Shape AI — SPA Main Application Entry
 * Boots the router, registers views, and mounts the navigation bar.
 */

import { Router } from "./router.js";
import { renderNavbar } from "./components/navbar.js";
import { createStudioView } from "./views/studioView.js";
import { createModelView } from "./views/modelView.js";
import { createGalleryView } from "./views/galleryView.js";
import { createBenchmarksView } from "./views/benchmarksView.js";
import { createNotFoundView } from "./views/notFoundView.js";

// Route Definition Table
const ROUTES = {
  "#/studio": {
    title: "TRELLIS 3D Studio",
    view: createStudioView,
  },
  "#/model": {
    title: "TRELLIS.2-4B Model Specs",
    view: createModelView,
  },
  "#/gallery": {
    title: "3D Asset Gallery",
    view: createGalleryView,
  },
  "#/metrics": {
    title: "Verification & Telemetry",
    view: createBenchmarksView,
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
