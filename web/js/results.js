// Force averaging and the numbers shown to the user. Mirrors class Averager, results(), result_lines(),
// playback() and arrow_geometry() of Flowniac.py.

import {
  AVERAGE_CHORDS, CYLINDER, ELEMENT_NAMES, FS_FORM_FACTOR, JIB_MAIN, MAST_SAIL, NACA0012, NACA2412, NU_AIR, PLATE,
  QUALITY_CELLS, RHO_AIR, SAIL_SHAPES, SETTLE_CHORDS, SHAPES, SPAN_EFFICIENCY, TELLTALE_LENGTH, TELLTALE_SEGMENTS,
  U_LAT, jibAoaLimit, jibSlot,
} from "./config.js";

const radians = (deg) => deg * Math.PI / 180;

/** Coefficient sets are arrays of [Cd, Cl] pairs: total, main/body, jib, mast, and total with the drag estimated
 *  for the full-size boundary layer (fullSizeDrag; the simulated drag where there is no estimate). */
const scale = (c, k) => c.map(([x, y]) => [x * k, y * k]);
const sum = (a, b) => a.map(([x, y], e) => [x + b[e][0], y + b[e][1]]);

/** Instantaneous forces fluctuate with vortex shedding: settle, then take the long-time mean. */
export class Averager {
  constructor(stepsPerChord) {
    this.spc = stepsPerChord;
    this.reset();
  }

  reset() {
    this.age = 0.0;
    this.ema = null;
    this.sum = null;
    this.weight = 0.0;
  }

  add(c, steps) {
    if (steps <= 0) {
      if (this.ema === null) this.ema = scale(c, 1);
      return;
    }
    this.age += steps;
    const a = 1.0 - Math.exp(-steps / (0.5 * this.spc));
    this.ema = this.ema === null ? scale(c, 1) : sum(this.ema, scale(sum(c, scale(this.ema, -1)), a));
    if (this.age > SETTLE_CHORDS * this.spc) {
      this.sum = this.sum === null ? scale(c, steps) : sum(this.sum, scale(c, steps));
      this.weight += steps;
    }
  }

  get value() {
    return this.weight > 0 ? scale(this.sum, 1 / this.weight) : this.ema;
  }

  get chords() {
    return this.age / this.spc;
  }

  get ready() {
    return this.weight >= AVERAGE_CHORDS * this.spc;
  }

  status() {
    if (this.age <= SETTLE_CHORDS * this.spc) {
      return `flow settling: ${this.chords.toFixed(1)} of ${SETTLE_CHORDS.toFixed(0)} chords`;
    }
    const avg = this.weight / this.spc;
    return `averaged over ${avg.toFixed(1)} chords` + (this.ready ? "" : " (settling)");
  }
}

/** Length of the main sail's camber line / chord. */
function arcLength(st) {
  const m = st.shape === PLATE ? 0.0 : st.camber;
  const p = st.draft;
  let len = 0.0;
  let y0 = 0.0;
  for (let k = 1; k <= 200; k++) {
    const x = k / 200;
    const y = m <= 0.0 ? 0.0 : x < p ? m / (p * p) * (2 * p * x - x * x) : m / ((1 - p) ** 2) * (1 - 2 * p + 2 * p * x - x * x);
    len += Math.hypot(1 / 200, y - y0);
    y0 = y;
  }
  return len;
}

/**
 * Section drag of the full-size boundary layer, estimated: the simulated drag is far too high there (see
 * config.js). Skin friction of a turbulent boundary layer, Cf = 0.074 Re^-0.2 per side, with the local speed
 * outside the boundary layer from the pressure beside the cloth (u_e^2 = U^2 (1 - Cp), wall stress ~ u_e^1.8),
 * none where the lee flow has separated; attached flow adds form drag FS_FORM_FACTOR x friction, separated flow
 * the simulated (pressure) drag in proportion to the separated part of the lee side.
 * rhoLee, rhoWind: density beside the main (Solver.surfaceDensity); lee: lee-side flow (Solver.leeProfile).
 */
export function fullSizeDrag(st, cdSim, rhoLee, rhoWind, lee) {
  const f = (rho) => Math.max(1.0 + (2.0 / 3.0) * (1.0 - rho) / (U_LAT * U_LAT), 0.0) ** 0.9;
  let fl = 0.0;
  let fw = 0.0;
  let att = 0;
  lee.forEach((v, k) => {
    if (v > 0.0) {
      fl += f(rhoLee[k]);
      att += 1;
    }
    fw += f(rhoWind[k]);
  });
  const re = st.wind * st.width / NU_AIR;
  const cdf = 0.074 * re ** -0.2 * (fl + fw) / lee.length * arcLength(st) / st.refFraction();
  const a = att / lee.length;
  return cdf * (1.0 + FS_FORM_FACTOR * a) + cdSim * (1.0 - a);
}

/**
 * Leech telltale of the main in the full-size boundary layer. There the strong near-wall mixing closes a
 * trailing-edge separation again just before the leech, so the simulated air still leaves the leech cleanly and
 * the telltale would stream aft. Instead, when the lee flow (lee: Solver.leeProfile) is reversed in the aft half
 * of the main, the ribbon is drawn curling to leeward and fluttering, the more the larger the separated part, as
 * on a real stalled leech. Returns the ribbon points (cells) or null (no separation: use the computed telltale).
 */
export function leechFlutter(sim, st, lee, time) {
  if (!SAIL_SHAPES.includes(st.shape)) return null;
  const aft = lee.slice(Math.floor(lee.length / 2));
  const stall = aft.filter((v) => v < 0.0).length / aft.length;
  if (stall <= 0.0) return null;
  const k = Math.min(1.0, 1.5 * stall);
  const [x0, y0, tx, ty] = sim.mainFrame(st, 1.0);
  const nx = -ty;                                   // towards the lee side
  const ny = tx;
  const seg = TELLTALE_LENGTH * sim.n / TELLTALE_SEGMENTS;
  let p = [x0 + tx, y0 + ty];                       // tied on just behind the leech, like the computed one
  const pts = [p];
  for (let s = 0; s < TELLTALE_SEGMENTS; s++) {
    const wobble = Math.sin(9.0 * time + 1.3 * s) + 0.5 * Math.sin(23.0 * time + 2.1 * s) + 0.6 * (Math.random() - 0.5);
    const phi = k * radians(40.0 + 18.0 * s + 25.0 * wobble);
    p = [p[0] + seg * (Math.cos(phi) * tx + Math.sin(phi) * nx), p[1] + seg * (Math.cos(phi) * ty + Math.sin(phi) * ny)];
    pts.push(p);
  }
  return pts;
}

/** All numbers shown to the user, from time-averaged 2D coefficients. */
export function results(st, coeffs) {
  const [cd, cl] = st.boundary ? coeffs[4] : coeffs[0];
  const refM = st.refFraction() * st.width;
  const area = refM * st.height;
  const ar = st.height / refM;
  const cdi = cl * cl / (Math.PI * SPAN_EFFICIENCY * ar);
  const q = 0.5 * RHO_AIR * st.wind ** 2;
  const re = st.wind * (st.shape === CYLINDER ? refM : st.width) / NU_AIR;
  const r = {
    cd, cl, ld: Math.abs(cd) > 1e-9 ? cl / cd : NaN, cdi, cd3: cd + cdi, ar,
    ld3: Math.abs(cd + cdi) > 1e-9 ? cl / (cd + cdi) : NaN, q, area, re,
    lift: cl * q * area, drag2: cd * q * area, dragi: cdi * q * area, elements: [],
  };
  r.drag3 = r.drag2 + r.dragi;
  r.total = Math.hypot(r.lift, r.drag3);
  // whole-sail force in boat axes, for a boat heading b to the (apparent) wind
  const b = radians(st.heading);
  r.drive = r.lift * Math.sin(b) - r.drag3 * Math.cos(b);
  r.side = r.lift * Math.cos(b) + r.drag3 * Math.sin(b);
  for (const e of [1, 2, 3]) {
    if ((st.shape === MAST_SAIL || st.shape === JIB_MAIN) && (e !== 2 || st.shape === JIB_MAIN)) {
      r.elements.push([ELEMENT_NAMES[e], coeffs[e][1], coeffs[e][0]]);
    }
  }
  return r;
}

/** Python's format(v, ".Ng"): 2e+04, 1.6e+06, 0.25. */
export function fmtG(v, digits) {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  const exp = Math.floor(Math.log10(Math.abs(Number(v.toPrecision(digits)))));
  const strip = (s) => (s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s);
  if (exp < -4 || exp >= digits) {
    const [m, e] = v.toExponential(digits - 1).split("e");
    const n = Number(e);
    return `${strip(m)}e${n < 0 ? "-" : "+"}${String(Math.abs(n)).padStart(2, "0")}`;
  }
  return strip(v.toFixed(Math.max(0, digits - 1 - exp)));
}

/** Short label for a scale: 118, 16, 2.5, 2. */
export function fmtNum(v) {
  if (Math.abs(v) >= 9.95) return v.toFixed(0);
  const s = v.toFixed(1);
  return s.endsWith(".0") ? s.slice(0, -2) : s;
}

/**
 * How the animation compares with reality: the simulated air moves stepsPerSecond * U_LAT / n chords
 * per second on screen, the real wind st.wind / st.width chords per second.
 */
export function playback(st, sim, stepsPerSecond) {
  if (st.paused) return "paused";
  if (stepsPerSecond <= 0.0) return "–";
  const k = (st.wind / st.width) / (stepsPerSecond * U_LAT / sim.n);
  if (k >= 1.1) return k >= 9.5 ? `${k.toFixed(0)}x slow motion` : `${k.toFixed(1)}x slow motion`;
  if (k > 0.9) return "real time";
  return `${(1.0 / k).toFixed(1)}x faster than real`;
}

export function fmtLd(cl, cd) {
  return cd > 0.005 ? (cl / cd).toFixed(1) : "–";      // drag ~0 only happens while the flow settles
}

export function resultLines(st, r, avg, sim, info, lee = null) {
  const lines = [
    `${SHAPES[st.shape][1]}, AoA ${st.aoa.toFixed(1)}°`,
    `  ${avg.status()}`,
    `2D section  Cl ${r.cl.toFixed(2)}  Cd ${r.cd.toFixed(3)}  L/D ${fmtLd(r.cl, r.cd)}`,
  ];
  if (st.boundary) {
    lines.push(lee !== null ? "  full size (approx.): Cd estimated" : "  full size (approx.): Cd far too high");
  }
  for (const [name, cl, cd] of r.elements) {
    lines.push(`  ${name.padEnd(4)}      Cl ${cl.toFixed(2)}  Cd ${cd.toFixed(3)}`);
  }
  if (st.shape === JIB_MAIN) {
    const { slot, overlap } = jibSlot(st);
    lines.push(`Jib slot ${(100 * slot).toFixed(1)}%  overlap ${(100 * overlap).toFixed(0)}%`);
    if (st.jib_aoa >= jibAoaLimit(st) - 1e-6) lines.push("  jib sheeted in as far as it goes");
  }
  if (lee !== null) {
    const sep = lee.map((v) => v < 0.0);
    const who = st.shape === JIB_MAIN ? "Main leeward" : "Leeward";
    const attached = 1 - sep.filter(Boolean).length / sep.length;
    lines.push(`${who} flow attached ${(100 * attached).toFixed(0)}% of chord`);
    if (sep.some(Boolean)) {
      const note = st.boundary ? "" : " (full size: later)";
      lines.push(`  separated from ${(100 * (sep.indexOf(true) + 0.5) / lee.length).toFixed(0)}%${note}`);
    }
  }
  lines.push(`Whole sail  Cd ${r.cd3.toFixed(3)}  L/D ${fmtLd(r.cl, r.cd3)}  (AR ${r.ar.toFixed(1)})`);
  if ((st.boat || st.axes === 1) && st.shape !== CYLINDER) {
    let boom = `Boom ${(st.heading - st.aoa).toFixed(0)}° off centreline`;
    if (st.shape === JIB_MAIN) boom += `, jib ${(st.heading - st.jib_aoa).toFixed(0)}°`;
    lines.push(boom);
  }
  if (st.axes === 1) {
    lines.push(`Drive ${r.drive.toFixed(0)} N  Side ${r.side.toFixed(0)} N  (${r.area.toFixed(1)} m²)`);
    if (r.drive > 0.0) {
      lines.push(`  side force = ${(r.side / r.drive).toFixed(1)} x drive`);
    } else if (r.drive < 0.0) {
      lines.push("  no drive: " + (st.heading < 60.0 ? "too close to the wind" : "sail force points backwards"));
    } else {
      lines.push("  –");
    }
  } else {
    lines.push(`Lift ${r.lift.toFixed(0)} N  Drag ${r.drag3.toFixed(0)} N  (${r.area.toFixed(1)} m²)`);
    lines.push(`  = ${r.drag2.toFixed(0)} N section + ${r.dragi.toFixed(0)} N induced`);
  }
  if (st.boundary) {
    lines.push(`Re ${fmtG(r.re, 2)} real, full size (approx.)`);
  } else {
    lines.push(`Re ${fmtG(r.re, 2)} real, ${fmtG(sim.reSim, 2)} simulated`);
  }
  if (st.shape === CYLINDER && r.re > 3e5) lines.push("  real cylinder: drag crisis > Re 3e5");
  if (st.shape === NACA0012 || st.shape === NACA2412) {
    lines.push("Note: thick profiles suffer at low Re");
    lines.push("  at full size (Re ~1e6) L/D ~50-100");
    if (sim.n < QUALITY_CELLS.medium) {
      lines.push("  Low grid: nose under-resolved,");
      lines.push("  use quality Medium or higher");
    }
  }
  lines.push(info);
  return lines;
}

export function tracerColor(st) {
  return st.view === 2 ? [0.12, 0.12, 0.12] : [0.92, 0.92, 0.92];   // dark smoke on the light Cp view
}

/**
 * Line segments (normalized picture coords: u along the wind, v across it, from the bottom) for the
 * total force (white) and its two parts: lift (green) and drag (red) of the 2D section, or drive (blue)
 * and side force (orange) of the whole sail. Returns [[from, to, colour], ...].
 */
export function arrowGeometry(sim, st, r) {
  const o = [(sim.px - sim.vx0) / sim.vw, (sim.py - sim.vy0) / sim.vh];
  const scaleLen = 0.45 * st.refFraction() * sim.n;    // arrow length per unit coefficient, in cells
  const cl = r.cl;
  let cd;
  let parts;
  if (st.axes === 1) {
    cd = r.cd3;                                        // boat axes: whole sail, including induced drag
    const b = radians(st.heading);
    const fwd = [-Math.cos(b), Math.sin(b)];
    const across = [Math.sin(b), Math.cos(b)];
    const drive = cl * Math.sin(b) - cd * Math.cos(b);
    const side = cl * Math.cos(b) + cd * Math.sin(b);
    parts = [[[fwd[0] * drive, fwd[1] * drive], [0.3, 0.75, 1.0]],
             [[across[0] * side, across[1] * side], [1.0, 0.6, 0.15]]];
  } else {
    cd = r.cd;
    parts = [[[0.0, cl], [0.2, 0.9, 0.3]], [[cd, 0.0], [1.0, 0.3, 0.2]]];
  }
  const segs = [];
  for (const [vec, col] of [...parts, [[cd, cl], [1.0, 1.0, 1.0]]]) {
    const d = [vec[0] * scaleLen / sim.vw, vec[1] * scaleLen / sim.vh];
    const tip = [o[0] + d[0], o[1] + d[1]];
    segs.push([o, tip, col]);
    const n = Math.hypot(d[0] * sim.vw, d[1] * sim.vh);
    if (n >= 1e-6) {
      const ux = d[0] * sim.vw / n;
      const uy = d[1] * sim.vh / n;
      const h = Math.min(0.25 * n, 0.08 * sim.n);
      const left = [tip[0] + (-ux * 0.9 - uy * 0.5) * h / sim.vw, tip[1] + (-uy * 0.9 + ux * 0.5) * h / sim.vh];
      const right = [tip[0] + (-ux * 0.9 + uy * 0.5) * h / sim.vw, tip[1] + (-uy * 0.9 - ux * 0.5) * h / sim.vh];
      segs.push([tip, left, col], [tip, right, col]);
    }
  }
  return segs;
}
