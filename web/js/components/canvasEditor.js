/**
 * Sketch2Shape AI — Canvas Editor Component
 * High-performance 2D Sketchpad with Bezier Smoothing, History Stack, and Drag-and-Drop
 */

export class CanvasEditor {
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.options = {
      width: options.width || 512,
      height: options.height || 512,
      strokeColor: options.strokeColor || "#0f172a",
      lineWidth: options.lineWidth || 4,
      ...options
    };

    this.isDrawing = false;
    this.mode = "brush"; // "brush" | "eraser"
    this.brushColor = this.options.strokeColor;
    this.brushSize = this.options.lineWidth;
    this.undoStack = [];
    this.redoStack = [];
    this.maxHistory = 30;
    this.lastPoint = null;
    this.abortController = new AbortController();
    this.canvasWrap = this.canvas.closest(".canvas-viewport-wrap") || this.canvas.parentElement;
    this.cursorElement = null;
    if (this.canvasWrap) {
      if (getComputedStyle(this.canvasWrap).position === "static") this.canvasWrap.style.position = "relative";
      this.cursorElement = document.createElement("div");
      this.cursorElement.className = "canvas-brush-cursor";
      this.cursorElement.setAttribute("aria-hidden", "true");
      this.canvasWrap.appendChild(this.cursorElement);
    }

    this.initCanvas();
    this.bindEvents();
  }

  initCanvas() {
    this.canvas.width = this.options.width;
    this.canvas.height = this.options.height;
    this.clear(false);
  }

  clear(save = true) {
    if (save) this.saveState();
    this.ctx.fillStyle = "#ffffff";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  saveState() {
    if (this.undoStack.length >= this.maxHistory) {
      this.undoStack.shift();
    }
    this.undoStack.push(this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height));
    this.redoStack = [];
    this.triggerStateUpdate();
  }

  undo() {
    if (this.undoStack.length === 0) return;
    this.redoStack.push(this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height));
    const previous = this.undoStack.pop();
    this.ctx.putImageData(previous, 0, 0);
    this.triggerStateUpdate();
  }

  redo() {
    if (this.redoStack.length === 0) return;
    this.undoStack.push(this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height));
    const next = this.redoStack.pop();
    this.ctx.putImageData(next, 0, 0);
    this.triggerStateUpdate();
  }

  triggerStateUpdate() {
    if (typeof this.onHistoryChange === "function") {
      this.onHistoryChange({
        canUndo: this.undoStack.length > 0,
        canRedo: this.redoStack.length > 0
      });
    }
  }

  getCanvasPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / Math.max(1, rect.width);
    const scaleY = this.canvas.height / Math.max(1, rect.height);
    return {
      x: Math.max(0, Math.min(this.canvas.width, (e.clientX - rect.left) * scaleX)),
      y: Math.max(0, Math.min(this.canvas.height, (e.clientY - rect.top) * scaleY))
    };
  }

  setMode(mode) {
    if (mode !== "brush" && mode !== "eraser") return;
    this.mode = mode;
    if (this.cursorElement) this.cursorElement.dataset.mode = mode;
  }

  updateCursor(e) {
    if (!this.cursorElement || e.pointerType === "touch") return;
    const canvasRect = this.canvas.getBoundingClientRect();
    const wrapRect = this.canvasWrap.getBoundingClientRect();
    const point = this.getCanvasPos(e);
    const displayScaleX = canvasRect.width / this.canvas.width;
    const displayScaleY = canvasRect.height / this.canvas.height;
    this.cursorElement.style.left = `${canvasRect.left - wrapRect.left + point.x * displayScaleX}px`;
    this.cursorElement.style.top = `${canvasRect.top - wrapRect.top + point.y * displayScaleY}px`;
    this.cursorElement.style.width = `${Math.max(8, this.brushSize * displayScaleX)}px`;
    this.cursorElement.style.height = `${Math.max(8, this.brushSize * displayScaleY)}px`;
    this.cursorElement.style.display = "block";
  }

  bindEvents() {
    const signal = this.abortController.signal;
    this.canvas.addEventListener("pointerenter", (e) => this.updateCursor(e), { signal });
    this.canvas.addEventListener("pointerleave", () => {
      if (!this.isDrawing && this.cursorElement) this.cursorElement.style.display = "none";
    }, { signal });
    this.canvas.addEventListener("pointerdown", (e) => {
      if (e.button !== undefined && e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation();
      this.updateCursor(e);
      this.isDrawing = true;
      this.activePointerId = e.pointerId;
      this.canvas.setPointerCapture(e.pointerId);
      this.saveState();

      const pt = this.getCanvasPos(e);
      this.lastPoint = pt;

      this.ctx.beginPath();
      this.ctx.lineCap = "round";
      this.ctx.lineJoin = "round";
      this.ctx.lineWidth = this.brushSize;
      this.ctx.strokeStyle = this.mode === "eraser" ? "#ffffff" : this.brushColor;

      this.ctx.moveTo(pt.x, pt.y);
      this.ctx.lineTo(pt.x, pt.y);
      this.ctx.stroke();
    }, { signal });

    this.canvas.addEventListener("pointermove", (e) => {
      this.updateCursor(e);
      if (!this.isDrawing) return;
      e.preventDefault();
      e.stopPropagation();
      const pt = this.getCanvasPos(e);

      this.ctx.beginPath();
      this.ctx.lineCap = "round";
      this.ctx.lineJoin = "round";
      this.ctx.lineWidth = this.brushSize;
      this.ctx.strokeStyle = this.mode === "eraser" ? "#ffffff" : this.brushColor;

      const midX = (this.lastPoint.x + pt.x) / 2;
      const midY = (this.lastPoint.y + pt.y) / 2;
      this.ctx.moveTo(this.lastPoint.x, this.lastPoint.y);
      this.ctx.quadraticCurveTo(this.lastPoint.x, this.lastPoint.y, midX, midY);
      this.ctx.stroke();

      this.lastPoint = pt;
    }, { signal });

    const endDrawing = (e) => {
      if (!this.isDrawing) return;
      if (this.activePointerId !== undefined && e.pointerId !== this.activePointerId) return;
      this.isDrawing = false;
      this.activePointerId = undefined;
      try {
        this.canvas.releasePointerCapture(e.pointerId);
      } catch (err) {}
      if (e.type === "pointercancel" && this.cursorElement) this.cursorElement.style.display = "none";
    };

    this.canvas.addEventListener("pointerup", endDrawing, { signal });
    this.canvas.addEventListener("pointercancel", endDrawing, { signal });
    this.canvas.addEventListener("lostpointercapture", endDrawing, { signal });
  }

  loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => { this.drawImageToCanvas(img); resolve(); };
        img.onerror = reject;
        img.src = event.target.result;
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  loadImageFromURL(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => { this.drawImageToCanvas(image); resolve(); };
      image.onerror = reject;
      image.src = url;
    });
  }

  drawImageToCanvas(image) {
    this.saveState();
    this.ctx.fillStyle = "#ffffff";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    const ratio = Math.min(this.canvas.width / image.width, this.canvas.height / image.height);
    const drawW = image.width * ratio;
    const drawH = image.height * ratio;
    this.ctx.drawImage(image, (this.canvas.width - drawW) / 2, (this.canvas.height - drawH) / 2, drawW, drawH);
  }

  destroy() {
    this.isDrawing = false;
    if (this.activePointerId !== undefined) {
      try { this.canvas.releasePointerCapture(this.activePointerId); } catch (err) {}
      this.activePointerId = undefined;
    }
    this.abortController.abort();
    if (this.cursorElement) this.cursorElement.remove();
    this.cursorElement = null;
  }

  toDataURL() {
    return this.canvas.toDataURL("image/png");
  }

  isEmpty() {
    const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height).data;
    // Check if everything is white
    for (let i = 0; i < imgData.length; i += 16) {
      if (imgData[i] < 240 || imgData[i + 1] < 240 || imgData[i + 2] < 240) {
        return false;
      }
    }
    return true;
  }
}
