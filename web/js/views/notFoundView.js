/**
 * Sketch2Shape AI — 404 Not Found View
 */

export function createNotFoundView() {
  return {
    async render() {
      return `
        <div style="padding: 80px 20px; text-align: center; max-width: 540px; margin: 0 auto;">
          <div style="width: 48px; height: 48px; border-radius: var(--radius-md); background: var(--bg-tertiary); border: 1px solid var(--border-default); display: inline-flex; align-items: center; justify-content: center; color: var(--accent-blue); margin-bottom: 16px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          </div>
          <h1 style="font-size: 24px; font-weight: 700; letter-spacing: -0.02em;">Page Not Found</h1>
          <p style="color: var(--text-secondary); margin-top: 8px; font-size: 13px; line-height: 1.6;">
            The requested route does not exist. Use the navigation bar above or click below to return to the interactive studio.
          </p>
          <div style="margin-top: 24px;">
            <a href="#/studio" class="btn btn-primary">
              <span>Return to 3D Studio</span>
            </a>
          </div>
        </div>
      `;
    },
  };
}
