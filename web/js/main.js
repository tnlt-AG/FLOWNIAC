// FLOWNIAC web version: main loop, picture and quality selection. Mirrors main(), Front/GGUIFront and
// auto_quality() of Flowniac.py. URL parameters replace the command-line options, e.g.
//   index.html?quality=high&shape=jib_main&aoa=15&wind-from=left&forces=drive&polar

import {
  AUTO_SECONDS_PER_CHORD, DOMAIN_CHORDS, QUALITY_CELLS, SAIL_SHAPES, SHAPES, SOLVER_SHARE, State, TARGET_FPS,
  TELLTALES, UI_SCALE, U_LAT, WIND_FROM,
} from "./config.js";
import { initGPU } from "./gpu.js";
import { PolarPlot } from "./polar.js";
import { Averager, arrowGeometry, playback, resultLines, results, tracerColor } from "./results.js";
import { Solver } from "./solver.js";
import { Keys, Panel } from "./ui.js";

const VIEW_KEYS = ["speed", "vorticity", "pressure", "smoke"];

function parseArgs() {
  const q = new URLSearchParams(location.search);
  const num = (name, def) => (q.has(name) && Number.isFinite(Number(q.get(name))) ? Number(q.get(name)) : def);
  const choice = (name, options, def) => (options.includes(q.get(name)) ? q.get(name) : def);
  return {
    quality: choice("quality", ["auto", ...Object.keys(QUALITY_CELLS)], "auto"),
    shape: choice("shape", SHAPES.map(([key]) => key), "sail"),
    aoa: num("aoa", 12.0),
    camber: num("camber", 10.0),
    draft: num("draft", 45.0),
    wind: num("wind", 8.0),
    width: num("width", 3.0),
    height: num("height", 9.0),
    heading: num("heading", 30.0),
    forces: choice("forces", ["lift", "drive"], "lift"),
    noBoat: q.has("no-boat"),
    view: choice("view", VIEW_KEYS, "speed"),
    polar: q.has("polar"),
    windFrom: choice("wind-from", ["top", "left"], WIND_FROM),
  };
}

function secondsPerChord(n, mlups) {
  const cells = DOMAIN_CHORDS[0] * DOMAIN_CHORDS[1] * n * n;
  return (n / U_LAT) * cells / (mlups * 1e6) * 1.2;     // +20 % for rendering and UI
}

/** Measure the solver speed on the 'medium' grid and pick the finest preset that is fast enough. */
async function autoQuality(device) {
  const sim = await Solver.create(device, QUALITY_CELLS.medium, { nTracerLines: 1 });
  let enc = device.createCommandEncoder();
  sim.setGeometry(enc, new State());
  sim.advance(enc, 10);
  sim.submit(enc);
  await device.queue.onSubmittedWorkDone();
  const t0 = performance.now();
  let steps = 0;
  while (performance.now() - t0 < 1000) {
    enc = device.createCommandEncoder();
    sim.advance(enc, 20);
    sim.submit(enc);
    await device.queue.onSubmittedWorkDone();
    steps += 20;
  }
  const mlups = sim.nx * sim.ny * steps / ((performance.now() - t0) / 1000) / 1e6;
  sim.destroy();
  let choice = "low";
  for (const name of ["low", "medium", "high", "ultra"]) {
    if (secondsPerChord(QUALITY_CELLS[name], mlups) <= AUTO_SECONDS_PER_CHORD) choice = name;
  }
  console.log(`${mlups.toFixed(0)} million cell updates/s -> quality '${choice}' ` +
              `(${secondsPerChord(QUALITY_CELLS[choice], mlups).toFixed(1)} s per chord of flow)`);
  return choice;
}

/** The flow picture: WebGPU canvas (field, hull, bodies, smoke) plus a 2D overlay (telltales, arrows). */
class Front {
  constructor(sim, device, format) {
    this.sim = sim;
    this.flow = document.getElementById("flow");
    this.picture = document.getElementById("picture");
    this.gpuCanvas = document.getElementById("gpu");
    this.overlay = document.getElementById("overlay");
    this.ctx = this.gpuCanvas.getContext("webgpu");
    this.ctx.configure({ device, format, alphaMode: "opaque" });
    this.ctx2d = this.overlay.getContext("2d");
    this.fw = 1;
    this.fh = 1;
    new ResizeObserver(() => this._layout()).observe(this.flow);
    this._layout();
  }

  /** Largest picture with the aspect of the visible domain that fits the free space. */
  _layout() {
    const sim = this.sim;
    const aspect = sim.rotated ? sim.vh / sim.vw : sim.vw / sim.vh;     // width / height
    const W = this.flow.clientWidth;
    const H = this.flow.clientHeight;
    if (W === 0 || H === 0) return;
    const w = Math.min(W, H * aspect);
    const h = w / aspect;
    this.picture.style.width = `${w}px`;
    this.picture.style.height = `${h}px`;
    const dpr = window.devicePixelRatio || 1;
    this.fw = Math.max(1, Math.round(w * dpr));
    this.fh = Math.max(1, Math.round(h * dpr));
    for (const c of [this.gpuCanvas, this.overlay]) {
      c.width = this.fw;
      c.height = this.fh;
    }
  }

  /** Position in cells (or normalized picture coords u along / v across the wind) -> canvas pixels. */
  _pix(u, v) {
    return this.sim.rotated ? [v * this.fw, u * this.fh] : [u * this.fw, (1.0 - v) * this.fh];
  }

  draw(enc, st, arrows, telltales) {
    const sim = this.sim;
    const hullR = 0.6 * UI_SCALE + 0.4;                 // half width of the dinghy outline, pixels
    if (st.boat) sim.setHull(st, this.fw, this.fh, hullR);
    sim.drawPicture(enc, this.ctx.getCurrentTexture().createView(), this.fw, this.fh,
                    { hull: st.boat, hullR, tracers: st.tracers, smoke: tracerColor(st) });
    sim.submit(enc);

    const ctx = this.ctx2d;
    ctx.clearRect(0, 0, this.fw, this.fh);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    const rgb = (c) => `rgb(${c.map((v) => Math.round(255 * v)).join(" ")})`;
    if (telltales) {
      // dark outline first, then the coloured ribbon: red leeward, green windward, yellow on the leech
      const ribbons = telltales.map((pts, k) => [pts.map(([x, y]) => this._pix((x - sim.vx0) / sim.vw,
                                                                                (y - sim.vy0) / sim.vh)), TELLTALES[k][2]]);
      for (const layer of [0, 1]) {
        for (const [pts, side] of ribbons) {
          if (pts.length < 2) continue;
          ctx.lineWidth = 2 * (layer ? 0.8 * UI_SCALE : 0.8 * UI_SCALE + 1.2);
          ctx.strokeStyle = !layer ? rgb([0.05, 0.05, 0.05])
            : rgb(side < -0.5 ? [0.25, 0.95, 0.35] : side < 0.5 ? [1.0, 0.86, 0.1] : [1.0, 0.25, 0.2]);
          ctx.beginPath();
          pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
          ctx.stroke();
        }
      }
    }
    if (arrows) {
      ctx.lineWidth = 1.5 * UI_SCALE;
      for (const [a, b, col] of arrows) {
        ctx.strokeStyle = rgb(col);
        ctx.beginPath();
        ctx.moveTo(...this._pix(...a));
        ctx.lineTo(...this._pix(...b));
        ctx.stroke();
      }
    }
  }
}

function showMessage(text) {
  document.getElementById("message").textContent = text;
}

async function main() {
  const args = parseArgs();
  let device;
  try {
    ({ device } = await initGPU());
  } catch (e) {
    showMessage(e.message);
    return;
  }
  device.lost.then((info) => showMessage(`The graphics device was lost (${info.message}). Reload the page.`));
  showMessage("Measuring GPU speed for quality selection ...");
  const quality = args.quality === "auto" ? await autoQuality(device) : args.quality;
  const n = QUALITY_CELLS[quality];

  const st = new State();
  st.shape = SHAPES.findIndex(([key]) => key === args.shape);
  st.aoa = args.aoa;
  st.camber = args.camber / 100.0;
  st.draft = args.draft / 100.0;
  st.wind = args.wind;
  st.width = args.width;
  st.height = args.height;
  st.view = VIEW_KEYS.indexOf(args.view);
  st.heading = args.heading;
  st.axes = ["lift", "drive"].indexOf(args.forces);
  st.boat = !args.noBoat;
  st.polar = args.polar;
  st.clamp();

  const format = navigator.gpu.getPreferredCanvasFormat();
  const sim = await Solver.create(device, n, { rotated: args.windFrom === "top", format });
  showMessage("");
  document.getElementById("stage").classList.toggle("landscape", !sim.rotated);
  const front = new Front(sim, device, format);
  const panel = new Panel(document.getElementById("panel"), st);
  const keys = new Keys(st);
  const polar = new PolarPlot(document.getElementById("polar"));

  console.log(`Grid ${sim.nx} x ${sim.ny} (${n} cells per chord), simulated Re ${sim.reSim.toPrecision(3)}, ` +
              `tau0 = ${sim.tau0.toFixed(5)}, quality '${quality}'`);
  const avg = new Averager(sim.stepsPerChord);
  let geometry = null;
  let stepsPerFrame = 10;
  let fps = 0.0;
  let sps = 0.0;                                   // solver steps per second, smoothed
  let tLast = performance.now() / 1000;
  let tStep = 0.0;
  let overhead = 0.02;
  let unstableUntil = 0.0;
  let lee = null;                                  // time-averaged lee-side flow of the main sail
  window.flowniac = { sim, st, avg, timing: {} };   // for inspection from the browser console

  async function frame() {
    keys.held();
    const sail = SAIL_SHAPES.includes(st.shape);
    let enc = device.createCommandEncoder();
    if (st.geometry() !== geometry) {
      geometry = st.geometry();
      sim.setGeometry(enc, st);
      avg.reset();
      lee = null;
    }
    if (st.reset_flow) {
      st.reset_flow = false;
      sim.resetFlow(enc);
      sim.resetTracers(enc);
      avg.reset();
    }
    if (st.clear_polar) {
      st.clear_polar = false;
      polar.clear();
    }

    const steps = st.paused ? 0 : stepsPerFrame;
    let tSolver = performance.now();
    sim.advance(enc, steps);
    const probes = { lee: sail, telltales: sail && st.telltales };
    sim.probe(enc, probes);
    sim.submit(enc);
    let out = await sim.readOut();                  // waits for the GPU
    let coeffs = sim.coefficients(out, st.refFraction());
    tSolver = (performance.now() - tSolver) / 1000;
    if (!coeffs.flat().every(Number.isFinite)) {
      // the flow blew up (extreme case for this grid): restart it instead of showing garbage
      console.warn("Flow became unstable and was reset.");
      enc = device.createCommandEncoder();
      sim.resetFlow(enc);
      sim.resetTracers(enc);
      sim.probe(enc, probes);
      sim.submit(enc);
      out = await sim.readOut();
      avg.reset();
      unstableUntil = performance.now() / 1000 + 4.0;
      coeffs = [[0, 0], [0, 0], [0, 0], [0, 0]];
    }
    avg.add(coeffs, steps);
    enc = device.createCommandEncoder();
    if (st.tracers) sim.advectTracers(enc, steps);
    sim.render(enc, st.view);
    if (sail) {
      const prof = sim.leeProfile(out);
      const k = steps > 0 ? 1.0 - Math.exp(-steps / (0.5 * sim.stepsPerChord)) : 0.0;
      lee = lee === null ? prof : lee.map((v, i) => v + k * (prof[i] - v));
    } else {
      lee = null;
    }

    const c = avg.value;
    sim.setFarField(c[0][1], c[0][0], st.refFraction());
    const r = results(st, c);
    if (avg.ready) polar.record(st.configLabel(), st.aoa, r.cl, r.cd);
    polar.visible = st.polar;
    polar.update(r.ar, [st.aoa, r.cl, r.cd]);

    const info = `${quality}, ${fps.toFixed(0)} fps, ${playback(st, sim, sps)}`;
    const lines = resultLines(st, r, avg, sim, info, lee);
    if (performance.now() / 1000 < unstableUntil) lines.unshift("!! flow unstable: restarted");
    panel.render(lines);
    front.draw(enc, st, st.arrows ? arrowGeometry(sim, st, r) : null,
               probes.telltales ? sim.telltalePoints(out) : null);

    // steps per frame: aim for TARGET_FPS, but never let drawing eat more than 1/3 of the time
    const now = performance.now() / 1000;
    const dt = Math.max(now - tLast, 1e-4);
    tLast = now;
    fps = fps > 0 ? 0.9 * fps + 0.1 / dt : 1.0 / dt;
    if (steps > 0) {
      sps = sps > 0 ? 0.9 * sps + 0.1 * steps / dt : steps / dt;
      // smoothed timings; single slow frames (plot redraw, geometry change) are clipped
      tStep = tStep ? 0.8 * tStep + 0.2 * (tSolver / steps) : tSolver / steps;
      overhead = 0.8 * overhead + 0.2 * Math.min(Math.max(dt - tSolver, 0.0), 0.1);
      let want = Math.max((1.0 / TARGET_FPS - overhead) / tStep, SOLVER_SHARE / (1.0 - SOLVER_SHARE) * overhead / tStep);
      want = Math.min(want, 0.2 / tStep);          // keep at least ~5 frames per second
      stepsPerFrame = Math.trunc(Math.min(Math.max(0.7 * stepsPerFrame + 0.3 * want, 2), 2000));
    }
    Object.assign(window.flowniac.timing, { stepsPerFrame, tSolver, tStep, overhead, dt });
  }

  async function loop() {
    try {
      await frame();
      requestAnimationFrame(loop);
    } catch (e) {
      console.error(e);
      showMessage(`Stopped: ${e.message}`);
    }
  }
  requestAnimationFrame(loop);
}

main();
