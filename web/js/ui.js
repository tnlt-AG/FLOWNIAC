// Control column and keyboard. Mirrors classes Keys and Panel of Flowniac.py: same controls, ranges,
// steps and keys. The column is plain HTML here (it scales with the browser zoom).

import {
  CP_RANGE, FORCE_AXES, HELP_LINES, JIB_MAIN, MAST_SAIL, RHO_AIR, SAIL, SHAPES, VIEWS, VORTICITY_RANGE,
} from "./config.js";
import { LUTS } from "./luts.js";
import { fmtNum } from "./results.js";

const sign = (v, s) => (v >= 0 ? "+" : "") + s;

/** Keyboard handling. Arrow keys auto-repeat while held (own timing, as in the desktop version). */
export class Keys {
  static STEP = { aoa: 1.0, camber: 0.01, draft: 0.05, wind: 1.0, width: 0.25, height: 1.0,
                  mast: 0.01, jib_gap: 0.01, jib_angle: 1.0, jib_overlap: 0.05, heading: 5.0 };
  static LETTER = { d: "draft", w: "wind", c: "width", h: "height", m: "mast", g: "jib_gap",
                    j: "jib_angle", o: "jib_overlap", b: "heading" };
  static HELD = { ArrowUp: ["aoa", 1], ArrowDown: ["aoa", -1], ArrowRight: ["camber", 1], ArrowLeft: ["camber", -1] };

  constructor(st) {
    this.st = st;
    this.down = new Set();
    this.shift = false;
    this.heldSince = new Map();
    this.lastRepeat = new Map();
    window.addEventListener("keydown", (e) => this._keydown(e));
    window.addEventListener("keyup", (e) => {
      this.down.delete(e.key);
      this.shift = e.shiftKey;
    });
    window.addEventListener("blur", () => this.down.clear());
  }

  _step(key) {
    const [name, dir] = Keys.HELD[key];
    this.st[name] += dir * Keys.STEP[name] * (this.shift ? 0.2 : 1.0);
    this.st.clamp();
  }

  _keydown(e) {
    this.shift = e.shiftKey;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target;
    const inControl = t instanceof HTMLSelectElement || t instanceof HTMLInputElement || t instanceof HTMLButtonElement;
    if (inControl && (e.key.startsWith("Arrow") || e.key === " " || e.key === "Enter")) return;   // the control's own keys
    if (e.key in Keys.HELD) {
      e.preventDefault();
      if (!this.down.has(e.key)) {             // first step at once, so a short tap between frames counts
        const now = performance.now() / 1000;
        this.down.add(e.key);
        this.heldSince.set(e.key, now);
        this.lastRepeat.set(e.key, now);
        this._step(e.key);
      }
      return;
    }
    if (e.repeat && !(e.key.toLowerCase() in Keys.LETTER)) return;
    if (this.press(e.key, e.shiftKey)) e.preventDefault();
  }

  /** Returns true if the key did something. */
  press(key, shift) {
    const st = this.st;
    if (key.length === 1 && key !== key.toLowerCase()) {
      key = key.toLowerCase();
      shift = true;
    }
    if (key in Keys.LETTER) {
      const name = Keys.LETTER[key];
      st[name] += (shift ? 1 : -1) * Keys.STEP[name];
    } else if ("1234567".includes(key) && key.length === 1) {
      st.shape = Number(key) - 1;
    } else if (key === "v") {
      st.view = (st.view + 1) % VIEWS.length;
    } else if (key === "t") {
      st.tracers = !st.tracers;
    } else if (key === "l") {
      st.telltales = !st.telltales;
    } else if (key === "f") {
      st.arrows = !st.arrows;
    } else if (key === "n") {
      st.boat = !st.boat;
    } else if (key === "x") {
      st.axes = (st.axes + 1) % FORCE_AXES.length;
    } else if (key === "p") {
      st.polar = !st.polar;
    } else if (key === "r") {
      st.reset_flow = true;
    } else if (key === "Backspace") {
      st.clear_polar = true;
    } else if (key === " ") {
      st.paused = !st.paused;
    } else if (key === "i") {
      st.help = !st.help;
    } else {
      return false;
    }
    st.clamp();
    return true;
  }

  /** Called once per frame: arrow keys keep changing angle of attack and camber while held. */
  held() {
    const now = performance.now() / 1000;
    for (const key of this.down) {
      if (now - this.heldSince.get(key) > 0.35 && now - this.lastRepeat.get(key) > 0.06) {
        this.lastRepeat.set(key, now);
        this._step(key);
      }
    }
  }
}

/** Control column on the left of the window: selectors, sliders, check boxes, buttons and results. */
export class Panel {
  // label, State attribute, min, max, step, display format, shapes it applies to (null = all)
  static SLIDERS = [
    ["Angle of attack", "aoa", -30.0, 90.0, 0.5, (v) => `${v.toFixed(1)}°`, null],
    ["Camber", "camber", 0.0, 0.2, 0.005, (v) => `${(100 * v).toFixed(1)} %`, [SAIL, MAST_SAIL, JIB_MAIN]],
    ["Draft position", "draft", 0.25, 0.65, 0.01, (v) => `${(100 * v).toFixed(0)} %`, [SAIL, MAST_SAIL, JIB_MAIN]],
    ["Mast diameter", "mast", 0.01, 0.12, 0.005, (v) => `${(100 * v).toFixed(1)} %`, [MAST_SAIL, JIB_MAIN]],
    ["Jib slot", "jib_gap", 0.02, 0.25, 0.005, (v) => `${(100 * v).toFixed(1)} %`, [JIB_MAIN]],
    ["Jib overlap", "jib_overlap", -0.2, 0.4, 0.01, (v) => `${(100 * v).toFixed(0)} %`, [JIB_MAIN]],
    ["Jib angle", "jib_angle", -20.0, 15.0, 0.5, (v) => sign(v, `${v.toFixed(1)}°`), [JIB_MAIN]],
    ["Wind", "wind", 1.0, 20.0, 0.5, (v) => `${v.toFixed(1)} m/s`, null],
    ["Sail width", "width", 0.5, 10.0, 0.05, (v) => `${v.toFixed(2)} m`, null],
    ["Sail height", "height", 1.0, 40.0, 0.5, (v) => `${v.toFixed(1)} m`, null],
    ["Boat heading", "heading", 0.0, 180.0, 1.0, (v) => `${v.toFixed(0)}°`, null],
  ];

  constructor(root, st) {
    this.st = st;
    this.root = root;
    this.controls = [];          // [element, update(st)] pairs, refreshed when the state changes
    this.sig = null;
    this.lines = null;
    this.tText = 0;
    this._build();
  }

  _el(tag, props = {}, ...children) {
    const el = Object.assign(document.createElement(tag), props);
    el.append(...children);
    return el;
  }

  /** Give the keyboard back to the app after a mouse click on a control. */
  _release(el) {
    el.addEventListener("pointerup", () => setTimeout(() => el.blur(), 0));
    el.addEventListener("change", () => { if (el.tagName === "SELECT") el.blur(); });
  }

  _build() {
    const st = this.st;
    const add = (el) => this.root.append(el);
    add(this._el("header", { className: "title" }, this._el("h1", { textContent: "FLOWNIAC" }),
                 this._el("span", { textContent: "sail aerodynamics" })));
    add(this._select("Shape", "shape", SHAPES.map(([, name]) => name)));
    for (const sl of Panel.SLIDERS) add(this._slider(sl));
    add(this._select("View", "view", VIEWS, "gap"));
    add(this._scale());
    add(this._select("Forces", "axes", FORCE_AXES));
    add(this._toggles([["Smoke", "tracers"], ["Arrows", "arrows"], ["Telltales", "telltales"]]));
    add(this._toggles([["Boat", "boat"], ["Polar plot", "polar"]]));
    const run = this._button("Pause", () => { st.paused = !st.paused; });
    this.controls.push([run, () => { run.textContent = st.paused ? "Run" : "Pause"; }]);
    add(this._el("div", { className: "buttons" }, run,
                 this._button("Reset flow", () => { st.reset_flow = true; }),
                 this._button("Clear polar", () => { st.clear_polar = true; })));
    add(this._el("hr"));
    this.heading = this._el("h2");
    this.results = this._el("pre", { id: "results" });
    add(this.heading);
    add(this.results);
  }

  _select(label, attr, names, cls = "") {
    const st = this.st;
    const id = `ctl-${attr}`;
    const sel = this._el("select", { id }, ...names.map((name, k) => this._el("option", { value: k, textContent: name })));
    sel.addEventListener("change", () => {
      st[attr] = Number(sel.value);
      st.clamp();
    });
    this._release(sel);
    this.controls.push([sel, () => { if (Number(sel.value) !== st[attr]) sel.value = st[attr]; }]);
    return this._el("div", { className: `row ${cls}` }, this._el("label", { htmlFor: id, textContent: label }), sel);
  }

  _slider([label, attr, lo, hi, step, fmt, shapes]) {
    const st = this.st;
    const id = `ctl-${attr}`;
    const input = this._el("input", { id, type: "range", min: lo, max: hi, step, value: st[attr] });
    const out = this._el("output", { htmlFor: id });
    input.addEventListener("input", () => {
      st[attr] = Number(input.value);
      st.clamp();
      out.value = fmt(st[attr]);
    });
    this._release(input);
    const row = this._el("div", { className: attr === "wind" ? "row gap" : "row" },
                         this._el("label", { htmlFor: id, textContent: label }), input, out);
    this.controls.push([row, () => {
      row.hidden = shapes !== null && !shapes.includes(st.shape);
      if (document.activeElement !== input && Math.abs(Number(input.value) - st[attr]) > 1e-9) input.value = st[attr];
      out.value = fmt(st[attr]);
    }]);
    return row;
  }

  /** Colour scale of the flow picture in real units, so the wind speed shows: m/s, 1/s or Pa. */
  _scale() {
    const st = this.st;
    const canvas = this._el("canvas", { width: 256, height: 1 });
    const ticks = [0, 1, 2].map(() => this._el("span"));
    const note = this._el("span", { className: "note", textContent: "smoke only: no colours" });
    const box = this._el("div", { className: "scale" }, canvas, this._el("div", { className: "ticks" }, ...ticks), note);
    const ctx = canvas.getContext("2d");
    let drawn = -1;
    this.controls.push([box, () => {
      const smoke = st.view === 3;
      canvas.hidden = smoke;
      ticks[0].parentElement.hidden = smoke;
      note.hidden = !smoke;
      if (smoke) return;
      if (drawn !== st.view) {
        drawn = st.view;
        const px = ctx.createImageData(256, 1);
        for (let k = 0; k < 256; k++) {
          for (let ch = 0; ch < 3; ch++) px.data[4 * k + ch] = Math.round(255 * LUTS[3 * (256 * st.view + k) + ch]);
          px.data[4 * k + 3] = 255;
        }
        ctx.putImageData(px, 0, 0);
      }
      const q = 0.5 * RHO_AIR * st.wind ** 2;
      let t;
      if (st.view === 0) {                          // speed: 0 .. 2x the wind
        t = [fmtNum(0.0), `${fmtNum(st.wind)} = wind`, `${fmtNum(2.0 * st.wind)} m/s`];
      } else if (st.view === 1) {                   // vorticity: +-VORTICITY_RANGE wind / chord
        const w = VORTICITY_RANGE * st.wind / st.width;
        t = [fmtNum(-w), "0", `+${fmtNum(w)} 1/s`];
      } else {                                      // pressure: Cp -3 .. +1 times the dynamic pressure
        t = [fmtNum(CP_RANGE[0] * q), "0 = ambient", `+${fmtNum(CP_RANGE[1] * q)} Pa`];
      }
      t.forEach((s, k) => { ticks[k].textContent = s; });
    }]);
    return this._el("div", { className: "row" }, this._el("span", { className: "note", textContent: "Scale" }), box);
  }

  _toggles(items) {
    const st = this.st;
    return this._el("div", { className: "toggles" }, ...items.map(([label, attr]) => {
      const box = this._el("input", { type: "checkbox", checked: st[attr] });
      box.addEventListener("change", () => { st[attr] = box.checked; });
      this._release(box);
      this.controls.push([box, () => { box.checked = st[attr]; }]);
      return this._el("label", {}, box, label);
    }));
  }

  _button(label, onClick) {
    const b = this._el("button", { type: "button", textContent: label });
    b.addEventListener("click", onClick);
    this._release(b);
    return b;
  }

  /** Refresh the controls when the state changed, the results every 0.4 s. */
  render(lines) {
    const st = this.st;
    const now = performance.now() / 1000;
    const sig = JSON.stringify([st.shape, st.view, st.axes, st.tracers, st.arrows, st.telltales, st.boat, st.polar,
                                st.paused, st.help, Panel.SLIDERS.map((sl) => st[sl[1]])]);
    if (sig !== this.sig) {
      for (const [, update] of this.controls) update(st);
    } else if (lines.join("\n") === (this.lines || []).join("\n") || now - this.tText < 0.4) {
      return;
    }
    this.sig = sig;
    this.lines = lines;
    this.tText = now;
    this.heading.textContent = st.help ? "Keyboard" : "Results";
    this.results.replaceChildren(...(st.help ? HELP_LINES : lines).map((line) =>
      this._el("div", { className: line.startsWith("!!") ? "warn" : "", textContent: line })));
  }
}
