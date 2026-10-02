// Polar plot: Cl, L/D and drag polar vs angle of attack, one curve per configuration. Mirrors class
// PolarPlot of Flowniac.py (a matplotlib window there, three canvases beside the flow picture here).

import { SPAN_EFFICIENCY } from "./config.js";

// categorical colours for the dark panel, in fixed order (checked for colour-blind separation)
const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const INK = "rgb(232 234 238)";
const DIM = "rgb(140 146 158)";
const GRID = "rgba(140 146 158 / 0.25)";
const SURFACE = "rgb(24 26 31)";

function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) {
    lo -= 0.5;
    hi += 0.5;
  }
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw);
  const ticks = [];
  for (let v = Math.floor(lo / step) * step; v <= hi + 1e-9 * step; v += step) ticks.push(Number(v.toPrecision(10)));
  return { lo: ticks[0], hi: ticks.at(-1) < hi ? ticks.at(-1) + step : ticks.at(-1), ticks, step };
}

const label = (v, step) => (step >= 1 ? v.toFixed(0) : v.toFixed(Math.min(3, Math.ceil(-Math.log10(step)))));

export class PolarPlot {
  constructor(root) {
    this.root = root;
    this.canvases = [0, 1, 2].map(() => {
      const c = document.createElement("canvas");
      c.addEventListener("pointermove", (e) => this._hover(c, e));
      c.addEventListener("pointerleave", () => this._hover(c, null));
      root.append(c);
      return c;
    });
    this.data = new Map();    // config label -> Map(aoa -> [cl, cd])
    this.dirty = false;
    this.lastDraw = 0;
    this.current = null;
    this.points = new Map();  // canvas -> drawn points [x, y, text], for the hover readout
    this.hovered = new Map();
    this.ar = 1;
  }

  record(label, aoa, cl, cd) {
    if (!this.data.has(label)) this.data.set(label, new Map());
    const pts = this.data.get(label);
    const key = Math.round(aoa * 2) / 2;
    const old = pts.get(key);
    pts.set(key, [cl, cd]);
    if (old === undefined || Math.abs(old[0] - cl) > 0.01 || Math.abs(old[1] - cd) > 0.002) this.dirty = true;
  }

  clear() {
    this.data = new Map();
    this.dirty = true;
  }

  set visible(on) {
    if (on === !this.root.hidden) return;
    this.root.hidden = !on;
    this.dirty = true;
  }

  /** current = [aoa, cl, cd] of the live (not yet converged) point. */
  update(ar, current) {
    if (this.root.hidden) return;
    const now = performance.now() / 1000;
    if (JSON.stringify(current) !== JSON.stringify(this.current) && now - this.lastDraw > 2.0) {
      this.current = current;
      this.dirty = true;
    }
    if (ar !== this.ar) {
      this.ar = ar;
      this.dirty = true;
    }
    if (this.dirty && now - this.lastDraw > 0.5) {
      this._draw();
      this.dirty = false;
      this.lastDraw = now;
    }
  }

  _draw() {
    const k = 1.0 / (Math.PI * SPAN_EFFICIENCY * this.ar);
    const labels = [...this.data.keys()].sort();
    const shown = labels.slice(0, SERIES.length);          // no recycled colours: clear the polar for more
    const series = (fx, fy, extra = {}) => shown.map((lab, i) => {
      const pts = [...this.data.get(lab)].sort((a, b) => a[0] - b[0]);
      return { label: lab, color: SERIES[i], pts: pts.map(([aoa, [cl, cd]]) => [fx(aoa, cl, cd), fy(aoa, cl, cd)]), ...extra };
    });
    const cur = this.current;
    const note = labels.length > shown.length ? `+${labels.length - shown.length} more: clear polar` : "";
    this._panel(this.canvases[0], {
      title: "Lift coefficient", xlabel: "AoA [deg]", ylabel: "Cl (2D)", legend: true, note,
      series: series((a) => a, (a, cl) => cl), current: cur && [cur[0], cur[1]],
    });
    this._panel(this.canvases[1], {
      title: `L/D: solid 2D, dashed whole sail (AR ${this.ar.toFixed(1)})`, xlabel: "AoA [deg]", ylabel: "L/D",
      series: [...series((a) => a, (a, cl, cd) => cl / cd),
               ...series((a) => a, (a, cl, cd) => cl / (cd + k * cl * cl), { dashed: true })],
      current: cur && [cur[0], cur[1] / cur[2]],
    });
    this._panel(this.canvases[2], {
      title: "Drag polar", xlabel: "Cd (2D)", ylabel: "Cl (2D)",
      series: series((a, cl, cd) => cd, (a, cl) => cl), current: cur && [cur[2], cur[1]],
    });
  }

  _panel(canvas, spec) {
    canvas.spec = spec;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w === 0 || h === 0) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = SURFACE;
    ctx.fillRect(0, 0, w, h);
    const finite = (p) => Number.isFinite(p[0]) && Number.isFinite(p[1]);
    const all = spec.series.flatMap((s) => s.pts).concat(spec.current ? [spec.current] : []).filter(finite);
    const xs = all.map((p) => p[0]);
    const ys = all.map((p) => p[1]);
    const X = niceTicks(Math.min(...xs, 0), Math.max(...xs, 1e-3));
    const Y = niceTicks(Math.min(...ys, 0), Math.max(...ys, 1e-3));
    const box = { l: 46, r: w - 10, t: 24, b: h - 34 };
    const px = (x) => box.l + (x - X.lo) / (X.hi - X.lo) * (box.r - box.l);
    const py = (y) => box.b - (y - Y.lo) / (Y.hi - Y.lo) * (box.b - box.t);

    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = INK;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(spec.title, box.l, 15);
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    ctx.fillStyle = DIM;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const v of X.ticks) {
      ctx.beginPath();
      ctx.moveTo(px(v), box.t);
      ctx.lineTo(px(v), box.b);
      ctx.stroke();
      ctx.fillText(label(v, X.step), px(v), box.b + 4);
    }
    ctx.fillText(spec.xlabel, (box.l + box.r) / 2, box.b + 18);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of Y.ticks) {
      ctx.beginPath();
      ctx.moveTo(box.l, py(v));
      ctx.lineTo(box.r, py(v));
      ctx.stroke();
      ctx.fillText(label(v, Y.step), box.l - 5, py(v));
    }
    ctx.save();
    ctx.translate(11, (box.t + box.b) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText(spec.ylabel, 0, 0);
    ctx.restore();

    const hit = [];
    for (const s of spec.series) {
      const pts = s.pts.filter(finite);
      ctx.strokeStyle = s.color;
      ctx.fillStyle = s.color;
      ctx.globalAlpha = s.dashed ? 0.7 : 1;
      ctx.lineWidth = 2;
      ctx.setLineDash(s.dashed ? [6, 4] : []);
      ctx.beginPath();
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))));
      ctx.stroke();
      ctx.setLineDash([]);
      for (const [x, y] of pts) {
        ctx.beginPath();
        if (s.dashed) ctx.rect(px(x) - 3.5, py(y) - 3.5, 7, 7);
        else ctx.arc(px(x), py(y), 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = SURFACE;                // 2px surface ring keeps overlapping markers apart
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.strokeStyle = s.color;
        hit.push([px(x), py(y), `${s.label}${s.dashed ? " (whole sail)" : ""}\n${spec.xlabel.split(" ")[0]} ` +
                  `${label(x, X.step / 10)}   ${spec.ylabel.split(" ")[0]} ${label(y, Y.step / 10)}`]);
      }
      ctx.globalAlpha = 1;
    }
    if (spec.current && finite(spec.current)) {
      const [x, y] = [px(spec.current[0]), py(spec.current[1])];
      ctx.strokeStyle = INK;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x - 5, y - 5);
      ctx.lineTo(x + 5, y + 5);
      ctx.moveTo(x + 5, y - 5);
      ctx.lineTo(x - 5, y + 5);
      ctx.stroke();
      hit.push([x, y, "current (not yet averaged)"]);
    }
    if (spec.legend) {
      const items = spec.series.map((s) => [s.label, s.color]);
      if (spec.note) items.push([spec.note, null]);
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.font = "11px system-ui, sans-serif";
      items.reverse().forEach(([text, color], i) => {
        const y = box.b - 10 - 15 * i;
        if (color) {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(box.r - 6, y, 4, 0, 2 * Math.PI);
          ctx.fill();
        }
        ctx.fillStyle = color ? INK : DIM;
        ctx.textAlign = "right";
        ctx.fillText(text, box.r - 14, y);
      });
    }
    this.points.set(canvas, hit);
    const hov = this.hovered.get(canvas);
    if (hov) this._tooltip(ctx, hov, w);
  }

  _hover(canvas, e) {
    let best = null;
    if (e) {
      const r = canvas.getBoundingClientRect();
      const x = e.clientX - r.left;
      const y = e.clientY - r.top;
      let d2 = 12 * 12;                             // hit radius larger than the marker
      for (const p of this.points.get(canvas) || []) {
        const d = (p[0] - x) ** 2 + (p[1] - y) ** 2;
        if (d < d2) {
          d2 = d;
          best = p;
        }
      }
    }
    if (best !== (this.hovered.get(canvas) ?? null)) {
      this.hovered.set(canvas, best);
      if (canvas.spec) this._panel(canvas, canvas.spec);
    }
  }

  _tooltip(ctx, [x, y, text], w) {
    const lines = text.split("\n");
    ctx.font = "12px system-ui, sans-serif";
    const tw = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 12;
    const th = 16 * lines.length + 6;
    const bx = Math.min(x + 10, w - tw - 2);
    const by = Math.max(y - th - 8, 2);
    ctx.fillStyle = "rgb(44 48 56)";
    ctx.fillRect(bx, by, tw, th);
    ctx.fillStyle = INK;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    lines.forEach((l, i) => ctx.fillText(l, bx + 6, by + 4 + 16 * i));
  }
}
