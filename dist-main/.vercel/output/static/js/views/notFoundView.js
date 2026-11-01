/**
 * Sketch2Shape AI — 404 Not Found View
 */

export function createNotFoundView() {
  return {
    async render() {
      return `
        <div style="padding: 60px 20px; text-align: center; max-width: 600px; margin: 0 auto;">
          <div style="font-size: 64px;">📐</div>
          <h1 style="font-size: 32px; font-weight: 800; margin-top: 16px;">404 &middot; Route Not Found</h1>
          <p style="color: var(--text-secondary); margin-top: 8px; font-size: 14px;">
            The requested viewport coordinate does not exist in the Sketch2Shape topology.
          </p>
          <div style="margin-top: 24px;">
            <a href="#/studio" class="btn btn-primary">
              ↩️ Return to 3D Studio
            </a>
          </div>
        </div>
      `;
    }
  };
}
