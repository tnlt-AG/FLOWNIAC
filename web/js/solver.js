// Lattice-Boltzmann solver on the GPU (WebGPU). Ported from class Solver of Flowniac.py: same sizes, same
// kernels (shaders.js), same methods in camelCase. GPU work is recorded into a command encoder that the
// caller submits with submit(), so one frame needs only one wait for the GPU (readOut).

import {
  CYLINDER, CYLINDER_DIAMETER, DOMAIN_CHORDS, HULL_BEAM, HULL_LENGTH, HULL_MAST, HULL_POINTS, HULL_TRANSOM,
  HULL_WIDEST, LEE_SAMPLES, PIVOT_CHORDS, RE_PER_CELL2, SAIL, SIDE_SPONGE_CHORDS, SPONGE_CHORDS, TELLTALES, U_LAT,
  WIND_FROM,
} from "./config.js";
import { LUTS } from "./luts.js";
import { OUT_FORCE, OUT_LEE, OUT_RHO, OUT_SIZE, OUT_TT, PARAMS, TT_STRIDE, buildKernels } from "./shaders.js";

const radians = (deg) => deg * Math.PI / 180;
const clipNum = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

export class Solver {
  /**
   * Python: Solver(cells_per_chord, n_tracer_lines=24, re_sim=None). Pass the canvas format to be able
   * to draw the picture (drawPicture); without it the solver runs headless (speed test, parity test).
   */
  static async create(device, cellsPerChord, { nTracerLines = 24, reSim = null, rotated = WIND_FROM === "top",
                                                format = null } = {}) {
    const sim = new Solver(device, cellsPerChord, nTracerLines, reSim, rotated);
    await sim._compile(format);
    const enc = device.createCommandEncoder();
    sim.resetFlow(enc);
    sim.resetTracers(enc);
    sim.submit(enc);
    return sim;
  }

  constructor(device, cellsPerChord, nTracerLines, reSim, rotated) {
    const n = Math.trunc(cellsPerChord);
    this.device = device;
    this.n = n;
    this.reModel = reSim || RE_PER_CELL2 * n * n;    // simulated Re of the "model size" boundary layer
    this.reSim = this.reModel;
    this.fullSize = false;
    this.nx = Math.trunc(DOMAIN_CHORDS[0] * n);
    this.ny = Math.trunc(DOMAIN_CHORDS[1] * n);
    // off-cell offsets: the half cell in y breaks the up/down symmetry; with the quarter cell in x a
    // straight plate at 0, 45, 90 or 135 deg never runs exactly through cell centres (there the last
    // bit of cos/sin would decide which lattice links it blocks)
    this.px = PIVOT_CHORDS[0] * n + 0.25;
    this.py = PIVOT_CHORDS[1] * n + 0.5;
    this.tau0 = 3.0 * U_LAT * n / this.reSim + 0.5;
    this.sponge = SPONGE_CHORDS * n;
    this.sideSponge = SIDE_SPONGE_CHORDS * n;
    // part of the domain that is shown: everything except the absorbing layers
    this.vx0 = this.sideSponge;
    this.vy0 = this.sideSponge;
    this.vx1 = this.nx - 1 - this.sponge;
    this.vy1 = this.ny - 1 - this.sideSponge;
    this.vw = this.vx1 - this.vx0;
    this.vh = this.vy1 - this.vy0;
    this.rotated = rotated;                       // picture turned 90 deg clockwise: wind from the top
    this.stepsPerChord = n / U_LAT;
    // smoke: nLines rake lines, each releasing 3 particles per cell of free stream so it reads as a streak.
    // A line re-uses its particles in turn; with 4 per cell of domain length a particle is re-used only
    // after 4/3 domain lengths of flow, so smoke slowed down in the wake still reaches the outlet
    this.nLines = nTracerLines;
    this.smokePerCell = 3.0;
    this.perLine = 4 * this.nx;
    this.nTracers = this.nLines * this.perLine;
    this.rakeX = 0.5 * n;
    this.released = 0.0;                          // particles released per line since resetTracers (mod perLine)

    this.cur = 0;
    this.geo = [SAIL, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0];
    this.gamma = 0.0;      // far-field circulation (lattice units), from the measured lift
    this.source = 0.0;     // far-field source strength, from the measured drag
    this.hullSig = null;

    this.paramData = new ArrayBuffer(4 * PARAMS.length);
    this.paramView = new DataView(this.paramData);
    this.paramIndex = Object.fromEntries(PARAMS.map(([name, type], k) => [name, [4 * k, type]]));
    this.setParam("favg", 1);                     // forces: mean of the last two steps (see shaders.js)
    this._setViscosity();
    this._createBuffers();
  }

  _createBuffers() {
    const d = this.device;
    const S = GPUBufferUsage.STORAGE;
    const cells = this.nx * this.ny;
    const make = (bytes, usage) => d.createBuffer({ size: Math.max(16, bytes), usage });
    this.buffers = {
      f0: make(9 * 4 * cells, S),
      f1: make(9 * 4 * cells, S),
      mask: make(4 * cells, S),                   // 0 fluid, 1 main/body, 2 jib, 3 mast
      mask_new: make(4 * cells, S),
      link: make(4 * cells, S),                   // links crossing a membrane: bit k main/plate, 8+k jib (k=1..8)
      cs: make(4 * cells, S),                     // Smagorinsky constant per cell
      wn: make(8 * cells, S),                     // wall normal of the cells next to a wall (full-size boundary layer)
      rho: make(4 * cells, S),
      vel: make(8 * cells, S),
      img: make(16 * cells, S),
      lut: make(16 * 3 * 256, S | GPUBufferUsage.COPY_DST),
      bbox: make(16, S),
      out: make(4 * OUT_SIZE, S | GPUBufferUsage.COPY_SRC),
      tracers: make(24 * this.nTracers, S),
      hull: make(8 * (2 * HULL_POINTS + 1), S | GPUBufferUsage.COPY_DST),
      params: make(this.paramData.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
    };
    this.staging = make(4 * OUT_SIZE, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const lut = new Float32Array(4 * 3 * 256);
    for (let k = 0; k < 3 * 256; k++) {
      lut.set(LUTS.subarray(3 * k, 3 * k + 3), 4 * k);
      lut[4 * k + 3] = 1.0;
    }
    d.queue.writeBuffer(this.buffers.lut, 0, lut);
  }

  /** The GPU buffer behind a kernel's buffer name, for the population pair of step parity cur. */
  _buffer(name, cur) {
    if (name === "fa") return this.buffers[cur === 0 ? "f0" : "f1"];
    if (name === "fb") return this.buffers[cur === 0 ? "f1" : "f0"];
    if (name === "bbox_atomic") return this.buffers.bbox;
    return this.buffers[name];
  }

  async _compile(format) {
    const d = this.device;
    const { compute, render } = buildKernels(this);
    const layoutFor = (kernel, visibility, storageType) => d.createBindGroupLayout({
      entries: [{ binding: 0, visibility, buffer: { type: "uniform" } },
                ...kernel.uses.map((name, k) => ({
                  binding: k + 1, visibility,
                  buffer: { type: name === "bbox_atomic" ? "storage" : storageType } }))],
    });
    // one bind group per step parity (fa/fb swap); kernels without fa/fb use the same group for both
    const groupsFor = (kernel, layout) => [0, 1].map((cur) => d.createBindGroup({
      layout,
      entries: [{ binding: 0, resource: { buffer: this.buffers.params } },
                ...kernel.uses.map((name, k) => ({ binding: k + 1, resource: { buffer: this._buffer(name, cur) } }))],
    }));
    const module = async (name, kernel) => {
      const m = d.createShaderModule({ label: name, code: kernel.wgsl });
      const info = await m.getCompilationInfo();
      const errors = info.messages.filter((msg) => msg.type === "error");
      if (errors.length) {
        throw new Error(`WGSL ${name}: ` + errors.map((e) => `line ${e.lineNum}: ${e.message}`).join("; "));
      }
      return m;
    };

    this.kernels = {};
    await Promise.all(Object.entries(compute).map(async ([name, kernel]) => {
      const layout = layoutFor(kernel, GPUShaderStage.COMPUTE, "storage");
      const pipeline = await d.createComputePipelineAsync({
        label: name,
        layout: d.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: await module(name, kernel), entryPoint: "main" },
      });
      this.kernels[name] = { pipeline, groups: groupsFor(kernel, layout), count: this._groupCount(kernel.size) };
    }));

    this.draws = null;
    if (!format) return;
    this.draws = {};
    // alpha blending for the soft edges of the smoke dots (the flow picture itself is opaque)
    const blend = { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
                    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" } };
    await Promise.all(Object.entries(render).map(async ([name, kernel]) => {
      const layout = layoutFor(kernel, GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, "read-only-storage");
      const m = await module(name, kernel);
      const pipeline = await d.createRenderPipelineAsync({
        label: name,
        layout: d.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module: m, entryPoint: "vs" },
        fragment: { module: m, entryPoint: "fs", targets: [{ format, blend }] },
        primitive: { topology: "triangle-list" },
      });
      this.draws[name] = { pipeline, group: groupsFor(kernel, layout)[0] };
    }));
  }

  _groupCount(size) {
    const up = (v, k) => Math.ceil(v / k);
    if (size === "grid") return [up(this.nx, 16), up(this.ny, 16)];
    if (size === "nx") return [up(this.nx, 64)];
    if (size === "ny") return [up(this.ny, 64)];
    if (size === "tracers") return [up(this.nTracers, 64)];
    if (size === "workgroup") return [1];
    return [up(size, 64)];
  }

  /** Record kernel runs into a compute pass (cur selects which population buffer is fa). */
  _run(pass, name, cur = this.cur) {
    const k = this.kernels[name];
    pass.setPipeline(k.pipeline);
    pass.setBindGroup(0, k.groups[cur]);
    pass.dispatchWorkgroups(...k.count);
  }

  _pass(enc, fn) {
    const pass = enc.beginComputePass();
    fn(pass);
    pass.end();
  }

  setParam(name, value) {
    const [offset, type] = this.paramIndex[name];
    if (type === "i32") this.paramView.setInt32(offset, value, true);
    else if (type === "u32") this.paramView.setUint32(offset, value, true);
    else this.paramView.setFloat32(offset, value, true);
  }

  /** Upload the parameter block and run the recorded GPU work. */
  submit(enc) {
    this.device.queue.writeBuffer(this.buffers.params, 0, this.paramData);
    this.device.queue.submit([enc.finish()]);
  }

  destroy() {
    for (const b of Object.values(this.buffers)) b.destroy();
    this.staging.destroy();
  }

  // ---------------------------------------------------------------- initialisation / geometry
  resetFlow(enc) {
    this._pass(enc, (pass) => this._run(pass, "init_flow", 0));
    this.cur = 0;
  }

  setGeometry(enc, st) {
    const c = this.n;
    this.geo = [st.shape, radians(st.aoa), st.camber, st.draft, 0.5 * st.mast * c,
                st.jib_gap * c, st.jib_overlap * c, radians(st.jib_angle)];
    ["shape", "a", "m", "p", "mast_r", "jgap", "jover", "jang"].forEach((name, k) => this.setParam(name, this.geo[k]));
    this._pass(enc, (pass) => {
      for (const name of ["build_mask", "apply_mask", "build_links", "reset_bbox", "find_bbox", "wall_damping",
                          "build_wall"]) {
        this._run(pass, name);
      }
    });
  }

  _setViscosity() {
    this.tau0 = 3.0 * U_LAT * this.n / this.reSim + 0.5;
    this.setParam("tau0", this.tau0);
    this.setParam("nu", U_LAT * this.n / this.reSim);
    this.setParam("fs", this.fullSize ? 1 : 0);
  }

  /**
   * Boundary layer: model size (the Re of the grid, laminar wall) or full size (approx.): the viscosity of the
   * real Re plus the law of the wall and near-wall mixing (see config.js). The near-wall mixing is part of the
   * geometry set-up, so call setGeometry after switching between the two.
   */
  setBoundaryLayer(fullSize, reReal) {
    this.fullSize = fullSize;
    this.reSim = fullSize ? reReal : this.reModel;
    this._setViscosity();
  }

  // ---------------------------------------------------------------- time stepping
  advance(enc, steps) {
    if (steps <= 0) return;
    this.setParam("gam", this.gamma);
    this.setParam("src", this.source);
    this._pass(enc, (pass) => {
      for (let s = 0; s < steps; s++) {
        this._run(pass, "step");
        this._run(pass, "boundaries_tb");
        this._run(pass, "boundaries_io");
        this.cur = 1 - this.cur;
      }
    });
  }

  /**
   * Circulation and source strength from the (averaged) force coefficients (Kutta-Joukowski).
   * The inlet and the absorbing layers then carry the flow a lifting body induces far away, so a
   * small domain behaves like an unbounded one instead of a narrow wind tunnel.
   */
  setFarField(cl, cd, refFraction) {
    const L = refFraction * this.n;
    cl = Number.isFinite(cl) ? clipNum(cl, -4.0, 4.0) : 0.0;
    cd = Number.isFinite(cd) ? clipNum(cd, 0.0, 4.0) : 0.0;
    this.gamma = -0.5 * cl * U_LAT * L;
    this.source = 0.5 * cd * U_LAT * L;
  }

  // ---------------------------------------------------------------- diagnostics
  /** Record the force sum, macroscopic fields, lee-side probe and telltales, and their copy for readOut. */
  probe(enc, { lee = false, telltales = false } = {}) {
    this._pass(enc, (pass) => {
      this._run(pass, "forces");
      this._run(pass, "macro");
      if (lee) this._run(pass, "leeward_profile");
      if (telltales) this._run(pass, "telltales");
    });
    enc.copyBufferToBuffer(this.buffers.out, 0, this.staging, 0, 4 * OUT_SIZE);
  }

  /** Wait for the GPU and return the read-back buffer (forces, lee profile, telltales). */
  async readOut() {
    await this.staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.staging.getMappedRange().slice(0));
    this.staging.unmap();
    return out;
  }

  /** Per element (index 1..3) and total (index 0): [Cd, Cl] of the 2D section. */
  coefficients(out, refFraction) {
    const f = [0, 1, 2, 3].map((e) => [out[OUT_FORCE + 2 * e], out[OUT_FORCE + 2 * e + 1]]);
    f[0] = [f[1][0] + f[2][0] + f[3][0], f[1][1] + f[2][1] + f[3][1]];
    const q = 0.5 * U_LAT * U_LAT * refFraction * this.n;
    return f.map(([x, y]) => [x / q, y / q]);
  }

  leeProfile(out) {
    return Array.from(out.subarray(OUT_LEE, OUT_LEE + LEE_SAMPLES));
  }

  /** Density 1.5 cells off the main on its lee and windward side, at the stations of the lee profile. */
  surfaceDensity(out) {
    return [Array.from(out.subarray(OUT_RHO, OUT_RHO + LEE_SAMPLES)),
            Array.from(out.subarray(OUT_RHO + LEE_SAMPLES, OUT_RHO + 2 * LEE_SAMPLES))];
  }

  /** Telltale ribbons as point lists in cells (empty for telltales that are not shown). */
  telltalePoints(out) {
    return TELLTALES.map((_, k) => {
      const base = OUT_TT + k * TT_STRIDE;
      const pts = [];
      for (let s = 0; s < out[base]; s++) pts.push([out[base + 1 + 2 * s], out[base + 2 + 2 * s]]);
      return pts;
    });
  }

  // ---------------------------------------------------------------- smoke tracers
  resetTracers(enc) {
    this.released = 0.0;
    this._pass(enc, (pass) => this._run(pass, "reset_tracers"));
  }

  advectTracers(enc, steps) {
    if (steps <= 0) return;
    const nsub = Math.max(1, Math.ceil(steps * U_LAT * 3.0));   // at most ~1/3 cell per sub-step at 1 U
    this.setParam("dt", steps / nsub);
    this.setParam("nsub", nsub);
    // particles each rake line releases in this frame: smokePerCell per cell the free stream moves
    const next = this.released + steps * U_LAT * this.smokePerCell;
    const first = Math.floor(this.released);
    this.setParam("emit0", first % this.perLine);
    this.setParam("emit_n", Math.min(Math.floor(next) - first, this.perLine));
    this.released = next - this.perLine * Math.floor(first / this.perLine);
    this._pass(enc, (pass) => this._run(pass, "advect"));
  }

  // ---------------------------------------------------------------- picture
  render(enc, view) {
    this.setParam("view", view);
    this._pass(enc, (pass) => this._run(pass, "render"));
  }

  /** Front of the object in cells: luff of the main / mast centre, or the upstream point of the cylinder. */
  leadingEdge(st) {
    const c = this.n;
    if (st.shape === CYLINDER) return [this.px - 0.5 * CYLINDER_DIAMETER * c, this.py];
    const a = radians(st.aoa);
    return [this.px - 0.5 * c * Math.cos(a), this.py + 0.5 * c * Math.sin(a)];
  }

  /**
   * Dinghy outline in cells, as a closed polygon (bow, starboard side, transom, port side, bow).
   * Drawing only: the hull is not part of the flow (on a real boat it sits below the sail section).
   * It turns about the leading edge of the object, pointing st.heading to the right of the wind.
   */
  hullOutline(st) {
    const b = radians(st.heading);
    const fwd = [-Math.cos(b), Math.sin(b)];             // towards the bow (the wind blows along +x)
    const across = [Math.sin(b), Math.cos(b)];           // towards starboard
    const length = HULL_LENGTH * this.n;
    const le = this.leadingEdge(st);
    const stb = [];
    const port = [];
    for (let k = 0; k < HULL_POINTS; k++) {
      const t = k / (HULL_POINTS - 1);                   // from the bow to the transom
      let w = t < HULL_WIDEST ? 1.0 - (1.0 - Math.min(t / HULL_WIDEST, 1.0)) ** 2.5
        : 1.0 - (1.0 - HULL_TRANSOM) * ((t - HULL_WIDEST) / (1.0 - HULL_WIDEST)) ** 2;
      w *= 0.5 * HULL_BEAM * length;
      const mid = [le[0] + (HULL_MAST - t) * length * fwd[0], le[1] + (HULL_MAST - t) * length * fwd[1]];
      stb.push([mid[0] + w * across[0], mid[1] + w * across[1]]);
      port.push([mid[0] - w * across[0], mid[1] - w * across[1]]);
    }
    return [...stb, ...port.reverse(), stb[0]];
  }

  /** Points in cells -> pixels of the flow picture, (0, 0) at its top left corner. */
  cellsToPix(pts, fw, fh) {
    return pts.map(([x, y]) => {
      const u = (x - this.vx0) / this.vw;                // along the wind, 0..1
      const v = (y - this.vy0) / this.vh;                // across the wind, 0..1
      return this.rotated ? [v * fw, u * fh] : [u * fw, (1.0 - v) * fh];
    });
  }

  setHull(st, fw, fh, r) {
    const sig = [st.shape, st.aoa, st.heading, fw, fh, r].join();
    if (sig === this.hullSig) return;
    this.hullSig = sig;
    const pix = this.cellsToPix(this.hullOutline(st), fw, fh);
    const xs = pix.map((p) => p[0]);
    const ys = pix.map((p) => p[1]);
    this.setParam("hb0", Math.max(Math.floor(Math.min(...xs) - r - 3.0), 0));
    this.setParam("hb1", Math.max(Math.floor(Math.min(...ys) - r - 3.0), 0));
    this.setParam("hb2", Math.min(Math.ceil(Math.max(...xs) + r + 3.0), fw - 1));
    this.setParam("hb3", Math.min(Math.ceil(Math.max(...ys) + r + 3.0), fh - 1));
    this.device.queue.writeBuffer(this.buffers.hull, 0, new Float32Array(pix.flat()));
  }

  /**
   * Draw the flow picture into a texture of fw x fh pixels: field, dinghy outline and bodies (compose),
   * then the smoke (splat_tracers). Python: Solver.compose + Solver.splat_tracers.
   */
  drawPicture(enc, view, fw, fh, { hull = false, hullR = 1.0, tracers = false, smoke = [1, 1, 1] } = {}) {
    this.setParam("fw", fw);
    this.setParam("fh", fh);
    this.setParam("hull", hull ? 1 : 0);
    this.setParam("hull_r", hullR);
    ["smoke_r", "smoke_g", "smoke_b"].forEach((name, k) => this.setParam(name, smoke[k]));
    // smoke dots grow with the picture: ~1/700 of its length along the wind, and wide enough that the
    // particles overlap into a streak; 2 pixels at least, as before
    const along = this.rotated ? fh : fw;
    const perParticle = along / this.vw / this.smokePerCell;
    this.setParam("dot", clipNum(Math.round(Math.max(along / 700.0, 1.3 * perParticle)), 2, 6));
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: [0.05, 0.05, 0.05, 1] }],
    });
    pass.setPipeline(this.draws.compose.pipeline);
    pass.setBindGroup(0, this.draws.compose.group);
    pass.draw(3);
    if (tracers) {
      pass.setPipeline(this.draws.splat_tracers.pipeline);
      pass.setBindGroup(0, this.draws.splat_tracers.group);
      pass.draw(6, this.nTracers);
    }
    pass.end();
  }
}

