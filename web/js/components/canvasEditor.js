/**
 * Sketch2Shape AI — Canvas Editor Component
 * High-performance 2D Sketchpad with Bezier Smoothing, History Stack,
 * Template Loading, and Auto-Persistence Callbacks.
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
      ...options,
    };

    this.isDrawing = false;
    this.mode = "brush"; // "brush" | "eraser"
    this.brushColor = this.options.strokeColor;
    this.brushSize = this.options.lineWidth;
    this.undoStack = [];
    this.redoStack = [];
    this.maxHistory = 30;
    this.lastPoint = null;

    // Callbacks
    this.onHistoryChange = null;
    this.onStrokeEnd = null;

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
    if (save && typeof this.onStrokeEnd === "function") {
      this.onStrokeEnd(this.toDataURL());
    }
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
    if (typeof this.onStrokeEnd === "function") {
      this.onStrokeEnd(this.toDataURL());
    }
  }

  redo() {
    if (this.redoStack.length === 0) return;
    this.undoStack.push(this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height));
    const next = this.redoStack.pop();
    this.ctx.putImageData(next, 0, 0);
    this.triggerStateUpdate();
    if (typeof this.onStrokeEnd === "function") {
      this.onStrokeEnd(this.toDataURL());
    }
  }

  triggerStateUpdate() {
    if (typeof this.onHistoryChange === "function") {
      this.onHistoryChange({
        canUndo: this.undoStack.length > 0,
        canRedo: this.redoStack.length > 0,
      });
    }
  }

  getCanvasPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }

  bindEvents() {
    this.canvas.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      this.isDrawing = true;
      try {
        this.canvas.setPointerCapture(e.pointerId);
      } catch (err) {}
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
    });

    this.canvas.addEventListener("pointermove", (e) => {
      if (!this.isDrawing) return;
      e.preventDefault();
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
    });

    const endDrawing = (e) => {
      if (!this.isDrawing) return;
      this.isDrawing = false;
      try {
        this.canvas.releasePointerCapture(e.pointerId);
      } catch (err) {}

      if (typeof this.onStrokeEnd === "function") {
        this.onStrokeEnd(this.toDataURL());
      }
    };

    this.canvas.addEventListener("pointerup", endDrawing);
    this.canvas.addEventListener("pointercancel", endDrawing);
  }

  loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        this.loadImageFromDataURL(event.target.result)
          .then(resolve)
          .catch(reject);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  loadImageFromDataURL(dataUrl) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.saveState();
        this.ctx.fillStyle = "#ffffff";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Letterbox scaling
        const ratio = Math.min(this.canvas.width / img.width, this.canvas.height / img.height);
        const drawW = img.width * ratio;
        const drawH = img.height * ratio;
        const drawX = (this.canvas.width - drawW) / 2;
        const drawY = (this.canvas.height - drawH) / 2;

        this.ctx.drawImage(img, drawX, drawY, drawW, drawH);
        if (typeof this.onStrokeEnd === "function") {
          this.onStrokeEnd(this.toDataURL());
        }
        resolve();
      };
      img.onerror = reject;
      img.src = dataUrl;
    });
  }

  loadImageFromUrl(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        this.saveState();
        this.ctx.fillStyle = "#ffffff";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        const ratio = Math.min(this.canvas.width / img.width, this.canvas.height / img.height);
        const drawW = img.width * ratio;
        const drawH = img.height * ratio;
        const drawX = (this.canvas.width - drawW) / 2;
        const drawY = (this.canvas.height - drawH) / 2;

        this.ctx.drawImage(img, drawX, drawY, drawW, drawH);
        if (typeof this.onStrokeEnd === "function") {
          this.onStrokeEnd(this.toDataURL());
        }
        resolve();
      };
      img.onerror = reject;
      img.src = url;
    });
  }

  toDataURL() {
    return this.canvas.toDataURL("image/png");
  }

  isEmpty() {
    const imgData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height).data;
    for (let i = 0; i < imgData.length; i += 16) {
      if (imgData[i] < 240 || imgData[i + 1] < 240 || imgData[i + 2] < 240) {
        return false;
      }
    }
    return true;
  }
}
