/**
 * Sketch2Shape AI — Client-Side Hash Router
 * Lightweight, vanilla ES6+ routing with transitions and 404 fallback
 */

export class Router {
  constructor(routes, containerId = "view-container") {
    this.routes = routes;
    this.container = document.getElementById(containerId);
    this.currentView = null;
    this.isTransitioning = false;

    // Bind hash change listener
    window.addEventListener("hashchange", () => this.handleRoute());
  }

  init() {
    if (!window.location.hash) {
      window.location.hash = "#/studio";
    } else {
      this.handleRoute();
    }
  }

  async handleRoute() {
    if (this.isTransitioning) return;
    const hash = window.location.hash.split("?")[0] || "#/studio";
    const route = this.routes[hash] || this.routes["404"];

    if (!route) {
      console.error(`[Router] No route definition found for: ${hash}`);
      return;
    }

    this.isTransitioning = true;
    this.updateActiveNavLinks(hash);

    // Transition out
    if (this.container && this.currentView) {
      this.container.classList.add("view-exit");
      await new Promise(r => setTimeout(r, 120));
      if (typeof this.currentView.destroy === "function") {
        this.currentView.destroy();
      }
    }

    // Load and render new view
    this.container.innerHTML = "";
    this.container.classList.remove("view-exit");
    this.container.classList.add("view-enter");

    try {
      this.currentView = route.view();
      const dom = await this.currentView.render();
      if (typeof dom === "string") {
        this.container.innerHTML = dom;
      } else if (dom instanceof HTMLElement) {
        this.container.appendChild(dom);
      }

      if (typeof this.currentView.afterRender === "function") {
        await this.currentView.afterRender();
      }

      // Update document title
      document.title = `${route.title} — Sketch2Shape AI`;
    } catch (err) {
      console.error(`[Router] Render error for route ${hash}:`, err);
      this.container.innerHTML = `
        <div style="padding: 40px; text-align: center;">
          <h2 style="color: var(--accent-rose);">View Loading Error</h2>
          <p style="color: var(--text-secondary); margin-top: 8px;">${err.message}</p>
        </div>
      `;
    }

    // Transition in
    requestAnimationFrame(() => {
      this.container.classList.remove("view-enter");
      this.isTransitioning = false;
    });
  }

  updateActiveNavLinks(currentHash) {
    document.querySelectorAll(".nav-link").forEach(link => {
      const href = link.getAttribute("href");
      if (href === currentHash) {
        link.classList.add("active");
      } else {
        link.classList.remove("active");
      }
    });
  }

  navigate(path) {
    window.location.hash = path;
  }
}
