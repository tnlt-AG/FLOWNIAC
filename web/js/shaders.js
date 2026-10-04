// WGSL kernels of the flow solver and the picture. Ported one by one from the Taichi kernels of
// Flowniac.py (same names without the leading underscore); the solver kernels still match it.
//
// Each kernel declares the buffers it uses; solver.js binds them in that order after the uniform
// parameter block P (binding 0). Loops over the 9 lattice directions are unrolled here in JavaScript,
// like ti.static(range(9)) in Taichi.

import {
  B_LOG, CP_RANGE, CYLINDER, CYLINDER_DIAMETER, DRAW_THICKNESS, E, FS_MIXING, FS_MIXING_MAX, FS_SLEEVE, HULL_POINTS,
  JIB_CHORD, JIB_MAIN, KAPPA, LEE_SAMPLES, MAST_SAIL, NACA0012, NACA2412, OPP, PLATE, SAIL, SIGMA_SPONGE, SMAGORINSKY,
  START_KICK, TAU_SPONGE, TELLTALES, TELLTALE_LENGTH, TELLTALE_SEGMENTS, U_LAT, VORTICITY_RANGE, W, WALL_DAMPING,
} from "./config.js";

/** Uniform parameter block P, shared by all kernels: [name, type], 4 bytes each. */
export const PARAMS = [
  ["shape", "i32"], ["a", "f32"], ["m", "f32"], ["p", "f32"],            // geometry (Solver.geo)
  ["mast_r", "f32"], ["jtx", "f32"], ["jty", "f32"], ["jang", "f32"],
  ["gam", "f32"], ["src", "f32"], ["view", "i32"], ["hull", "i32"],      // far field, picture
  ["hull_r", "f32"], ["fw", "f32"], ["fh", "f32"], ["dt", "f32"],
  ["nsub", "i32"], ["favg", "u32"], ["hb0", "f32"], ["hb1", "f32"],      // tracers, force mode, hull box
  ["hb2", "f32"], ["hb3", "f32"], ["smoke_r", "f32"], ["smoke_g", "f32"],
  ["smoke_b", "f32"], ["dot", "f32"], ["emit0", "u32"], ["emit_n", "i32"],   // smoke dot size, smoke release
  ["tau0", "f32"], ["nu", "f32"], ["fs", "u32"], ["pad1", "u32"],        // viscosity, full-size boundary layer
];

/** Storage buffers: WGSL element type. "bbox_atomic" is the bbox buffer seen as atomics. */
export const BUFFER_TYPES = {
  fa: "array<f32>", fb: "array<f32>",           // populations, k * N + cell (fa = f[cur], fb = f[1 - cur])
  mask: "array<i32>", mask_new: "array<i32>", link: "array<u32>", cs: "array<f32>",
  wn: "array<vec2f>",                           // wall normal (into the air) of cells next to a wall, else 0
  rho: "array<f32>", vel: "array<vec2f>", img: "array<vec4f>", lut: "array<vec4f>",
  bbox: "array<i32, 4>", bbox_atomic: "array<atomic<i32>, 4>",
  out: "array<f32>",                            // read back each frame: forces, lee profile, telltales
  tracers: "array<Tracer>", hull: "array<vec2f>",
};

// layout of the read-back buffer "out" (floats)
export const OUT_FORCE = 0;                         // force[4] (x, y), element 0 unused
export const OUT_LEE = 8;                           // LEE_SAMPLES values
export const OUT_TT = OUT_LEE + LEE_SAMPLES;        // per telltale: point count, then (x, y) points
export const TT_STRIDE = 1 + 2 * (TELLTALE_SEGMENTS + 1);
export const OUT_RHO = OUT_TT + TELLTALES.length * TT_STRIDE;   // density beside the main: LEE_SAMPLES lee, then windward
export const OUT_SIZE = OUT_RHO + 2 * LEE_SAMPLES;

/** Float literal for WGSL with the full double precision (converted to f32 like a Taichi constant). */
const fl = (v) => {
  const s = String(v);
  return /[.eE]/.test(s) || !Number.isFinite(v) ? s : s + ".0";
};
const K9 = [0, 1, 2, 3, 4, 5, 6, 7, 8];
const each = (ks, fn) => ks.map(fn).join("\n");

/** Constants of one grid (the ti.static values of the Python Solver). */
function header(sim) {
  return /* wgsl */ `
const NX: i32 = ${sim.nx};
const NY: i32 = ${sim.ny};
const N: i32 = ${sim.nx * sim.ny};
const C: f32 = ${fl(sim.n)};                      // cells per chord
const PX: f32 = ${fl(sim.px)};
const PY: f32 = ${fl(sim.py)};
const SPONGE: f32 = ${fl(sim.sponge)};
const SIDE_SPONGE: f32 = ${fl(sim.sideSponge)};
const VX0: f32 = ${fl(sim.vx0)};
const VY0: f32 = ${fl(sim.vy0)};
const VW: f32 = ${fl(sim.vw)};
const VH: f32 = ${fl(sim.vh)};
const ROTATED: bool = ${sim.rotated};
const N_LINES: i32 = ${sim.nLines};
const PER_LINE: i32 = ${sim.perLine};
const RAKE_X: f32 = ${fl(sim.rakeX)};
const SMOKE_PER_CELL: f32 = ${fl(sim.smokePerCell)};
const FS_MIX_MAX: f32 = ${fl(FS_MIXING_MAX * sim.n)};     // full-size mixing length cap, cells
const FS_SLEEVE_CELLS: f32 = ${fl(FS_SLEEVE * sim.n)};
const DAMP_R: i32 = ${Math.max(Math.trunc(4 * WALL_DAMPING), Math.ceil(FS_SLEEVE * sim.n) + 1)};

const U_LAT: f32 = ${fl(U_LAT)};
const SMAGORINSKY: f32 = ${fl(SMAGORINSKY)};
const WALL_DAMPING: f32 = ${fl(WALL_DAMPING)};
const TAU_SPONGE: f32 = ${fl(TAU_SPONGE)};
const SIGMA_SPONGE: f32 = ${fl(SIGMA_SPONGE)};
const START_KICK: f32 = ${fl(START_KICK)};
const KAPPA: f32 = ${fl(KAPPA)};
const B_LOG: f32 = ${fl(B_LOG)};
const EKB: f32 = ${fl(Math.exp(-KAPPA * B_LOG))};
const FS_MIXING: f32 = ${fl(FS_MIXING)};
const WALL_D: f32 = 0.5;                          // wall distance of a wall cell (halfway bounce-back)
const JIB_CHORD: f32 = ${fl(JIB_CHORD)};
const CYLINDER_DIAMETER: f32 = ${fl(CYLINDER_DIAMETER)};
const DRAW_THICKNESS: f32 = ${fl(DRAW_THICKNESS)};
const VORTICITY_RANGE: f32 = ${fl(VORTICITY_RANGE)};
const CP_MIN: f32 = ${fl(CP_RANGE[0])};
const CP_MAX: f32 = ${fl(CP_RANGE[1])};
const LEE_SAMPLES: i32 = ${LEE_SAMPLES};
const TELLTALE_SEGMENTS: i32 = ${TELLTALE_SEGMENTS};
const N_TELLTALES: i32 = ${TELLTALES.length};
const TT_STRIDE: i32 = ${TT_STRIDE};
const OUT_LEE: i32 = ${OUT_LEE};
const OUT_TT: i32 = ${OUT_TT};
const OUT_RHO: i32 = ${OUT_RHO};
const HULL_SEGMENTS: i32 = ${2 * HULL_POINTS};
const PI: f32 = ${fl(Math.PI)};

const SAIL: i32 = ${SAIL};
const MAST_SAIL: i32 = ${MAST_SAIL};
const JIB_MAIN: i32 = ${JIB_MAIN};
const PLATE: i32 = ${PLATE};
const CYLINDER: i32 = ${CYLINDER};
const NACA0012: i32 = ${NACA0012};
const NACA2412: i32 = ${NACA2412};

var<private> TT_SAIL: array<i32, ${TELLTALES.length}> = array<i32, ${TELLTALES.length}>(${TELLTALES.map((t) => t[0]).join(", ")});
var<private> TT_POS: array<f32, ${TELLTALES.length}> = array<f32, ${TELLTALES.length}>(${TELLTALES.map((t) => fl(t[1])).join(", ")});
var<private> TT_SIDE: array<f32, ${TELLTALES.length}> = array<f32, ${TELLTALES.length}>(${TELLTALES.map((t) => fl(t[2])).join(", ")});

struct Params {
${PARAMS.map(([name, type]) => `  ${name}: ${type},`).join("\n")}
}

struct Tracer {
  pos: vec2f,
  norm: vec2f,        // 0..1 across the flow picture
  pad0: f32,
  pad1: f32,
}
`;
}

/** Functions without buffer access, shared by all kernels. */
const COMMON = /* wgsl */ `
fn ix(i: i32, j: i32) -> i32 {
  return i + j * NX;
}

fn feq(rho: f32, u: vec2f) -> array<f32, 9> {
  let uu = 1.5 * (u.x * u.x + u.y * u.y);
  var f: array<f32, 9>;
${each(K9, (k) => {
    const eu = k === 0 ? "0.0" : [E[k][0] && `${E[k][0] < 0 ? "-" : ""}u.x`, E[k][1] && `${E[k][1] < 0 ? "-" : "+"}u.y`]
      .filter(Boolean).join(" ").replace(/^\+/, "");
    return `  { let eu = ${eu}; f[${k}] = ${fl(W[k])} * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - uu); }`;
  })}
  return f;
}

// density and velocity of a population vector, packed as (rho, ux, uy)
fn moments(f: array<f32, 9>) -> vec3f {
  let rho = ${K9.map((k) => `f[${k}]`).join(" + ")};
  let ux = f[1] - f[3] + f[5] - f[6] - f[7] + f[8];
  let uy = f[2] - f[4] + f[5] + f[6] - f[7] - f[8];
  return vec3f(rho, ux / rho, uy / rho);
}

// NACA 4-digit mean line: returns (y/c, dy/dx) at x/c = xn for camber m at position p
fn mean_line(xn: f32, m: f32, p: f32) -> vec2f {
  var yc = 0.0;
  var dyc = 0.0;
  if (m > 0.0) {
    if (xn < p) {
      yc = m / (p * p) * (2.0 * p * xn - xn * xn);
      dyc = 2.0 * m / (p * p) * (p - xn);
    } else {
      let q = (1.0 - p) * (1.0 - p);
      yc = m / q * (1.0 - 2.0 * p + 2.0 * p * xn - xn * xn);
      dyc = 2.0 * m / q * (p - xn);
    }
  }
  return vec2f(yc, dyc);
}

// thin cambered membrane from (0, 0) to (c, 0) with thickness th (all in cells)
fn in_membrane(x: f32, y: f32, c: f32, m: f32, p: f32, th: f32) -> bool {
  var inside = false;
  if (x >= -0.5 * th && x <= c + 0.5 * th) {
    let ml = mean_line(clamp(x / c, 0.0, 1.0), m, p);
    inside = abs(y - ml.x * c) / sqrt(1.0 + ml.y * ml.y) <= 0.5 * th;
  }
  return inside;
}

fn in_naca(x: f32, y: f32, c: f32, m: f32, p: f32, t: f32) -> bool {
  var inside = false;
  if (x >= 0.0 && x <= c) {
    let xn = x / c;
    let yt = 5.0 * t * c * (0.2969 * sqrt(xn) - 0.1260 * xn - 0.3516 * xn * xn + 0.2843 * xn * xn * xn
                            - 0.1036 * xn * xn * xn * xn);
    let yc = mean_line(xn, m, p).x * c;
    inside = abs(y - yc) <= yt;
  }
  return inside;
}

// point in the body frame: luff of the main at (0, 0), chord along +x (all in cells)
fn to_body(x: f32, y: f32, a: f32) -> vec2f {
  let dx = x - PX;
  let dy = y - PY;
  let ca = cos(a);
  let sa = sin(a);
  return vec2f(dx * ca - dy * sa + 0.5 * C, dx * sa + dy * ca);
}

// body-frame point in the jib's own frame (luff at 0, chord along +x). The jib tack (jtx, jty) is at the bow
// of the boat; jang is the angle of the jib chord to the main chord (config.js, jibPlacement).
fn jib_frame(xb: f32, yb: f32, jtx: f32, jty: f32, jang: f32) -> vec2f {
  let cj = cos(jang);
  let sj = sin(jang);
  return vec2f((xb - jtx) * cj - (yb - jty) * sj, (xb - jtx) * sj + (yb - jty) * cj);
}

// position of the point along, and signed offset from, each zero-thickness membrane (cells):
// (x along main / plate, offset, x along jib, offset)
fn membranes(x: f32, y: f32, shape: i32, a: f32, m: f32, p: f32, jtx: f32, jty: f32, jang: f32) -> vec4f {
  let b = to_body(x, y, a);
  var mm = m;
  var pp = p;
  if (shape == PLATE || shape == NACA0012) {
    mm = 0.0;
  } else if (shape == NACA2412) {
    mm = 0.02;
    pp = 0.4;
  }
  let gm = b.y - mean_line(clamp(b.x / C, 0.0, 1.0), mm, pp).x * C;
  var xj = -1.0;
  var gj = 1.0;
  if (shape == JIB_MAIN) {
    let jf = jib_frame(b.x, b.y, jtx, jty, jang);
    xj = jf.x;
    gj = jf.y - mean_line(clamp(xj / (JIB_CHORD * C), 0.0, 1.0), m, p).x * JIB_CHORD * C;
  }
  return vec4f(b.x, gm, xj, gj);
}

// does the straight segment between two points cross the membrane? g = signed offset from it,
// x = position along it
fn crosses(g0: f32, g1: f32, x0: f32, x1: f32, span: f32) -> bool {
  var hit = false;
  if ((g0 <= 0.0) != (g1 <= 0.0)) {
    let xc = x0 + g0 / (g0 - g1) * (x1 - x0);
    if (xc >= 0.0 && xc <= span) {
      hit = true;
    }
  }
  return hit;
}

// which part covers the point (x, y), in cells: 0 air, 1 main sail / body, 2 jib, 3 mast.
// Sails and plate are zero-thickness membranes in the simulation (they block lattice links, see
// build_links); only for drawing do they get the thickness th (th = 0: they are not solid).
fn body_at(x: f32, y: f32, shape: i32, a: f32, m: f32, p: f32, mast_r: f32, jtx: f32, jty: f32, jang: f32,
           th: f32) -> i32 {
  let bd = to_body(x, y, a);
  let xb = bd.x;
  let yb = bd.y;
  var s = 0;
  if (shape == SAIL || shape == MAST_SAIL || shape == JIB_MAIN) {
    if (th > 0.0) {
      if (in_membrane(xb, yb, C, m, p, th)) {
        s = 1;
      }
      if (shape == JIB_MAIN) {
        let jf = jib_frame(xb, yb, jtx, jty, jang);
        if (in_membrane(jf.x, jf.y, JIB_CHORD * C, m, p, th)) {
          s = 2;
        }
      }
    }
    if (shape != SAIL && xb * xb + yb * yb <= mast_r * mast_r) {
      s = 3;
    }
  } else if (shape == PLATE) {
    if (th > 0.0 && in_membrane(xb, yb, C, 0.0, 0.5, th)) {
      s = 1;
    }
  } else if (shape == CYLINDER) {
    let r = 0.5 * CYLINDER_DIAMETER * C;
    if ((xb - 0.5 * C) * (xb - 0.5 * C) + yb * yb <= r * r) {
      s = 1;
    }
  } else if (shape == NACA0012) {
    if (in_naca(xb, yb, C, 0.0, 0.5, 0.12)) {
      s = 1;
    }
  } else if (shape == NACA2412) {
    if (in_naca(xb, yb, C, 0.02, 0.4, 0.12)) {
      s = 1;
    }
  }
  return s;
}

// world position and unit tangent (towards the leech) at fraction xn along a sail (1 main, 2 jib)
fn sail_frame(xn: f32, sail: i32, a: f32, m: f32, p: f32, jtx: f32, jty: f32, jang: f32) -> vec4f {
  let ml = mean_line(xn, m, p);
  var L = C;
  if (sail == 2) {
    L = JIB_CHORD * C;
  }
  var q = vec2f(xn * L, ml.x * L);                // in the sail's own frame
  var t = normalize(vec2f(1.0, ml.y));
  if (sail == 2) {                                // jib frame -> main body frame
    let cj = cos(jang);
    let sj = sin(jang);
    q = vec2f(jtx, jty) + vec2f(q.x * cj + q.y * sj, -q.x * sj + q.y * cj);
    t = vec2f(t.x * cj + t.y * sj, -t.x * sj + t.y * cj);
  }
  let ca = cos(a);                                // main body frame -> world
  let sa = sin(a);
  let dx = q.x - 0.5 * C;
  return vec4f(PX + dx * ca + q.y * sa, PY - dx * sa + q.y * ca, t.x * ca + t.y * sa, -t.x * sa + t.y * ca);
}

// far field of a lifting body: free stream + point vortex (lift) + point source (drag / wake)
fn far_velocity(i: i32, j: i32, gam: f32, src: f32) -> vec2f {
  let dx = f32(i) - PX;
  let dy = f32(j) - PY;
  let r2 = max(dx * dx + dy * dy, 0.25 * C * C);
  let k = 1.0 / (2.0 * PI * r2);
  return vec2f(U_LAT + k * (-gam * dy + src * dx), k * (gam * dx + src * dy));
}

// Spalding's law of the wall, solved for u+ = s / u_tau from R = s d / nu (s: speed along the wall at
// distance d) by Newton iteration
fn wall_uplus(R: f32) -> f32 {
  var up = select(sqrt(R), log(R) / KAPPA + 1.0, R > 120.0);
  for (var it = 0; it < 8; it++) {
    let ku = KAPPA * up;
    let ek = exp(ku);
    let g = up + EKB * (ek - 1.0 - ku - 0.5 * ku * ku - ku * ku * ku / 6.0);
    let gp = 1.0 + EKB * KAPPA * (ek - 1.0 - ku - 0.5 * ku * ku);
    up = max(up - (g - R / up) / (gp + R / (up * up)), 1e-3);
  }
  return up;
}

// Full-size boundary layer at a wall cell with velocity u and wall normal n: the law of the wall gives the
// friction velocity u_tau; the wall then moves with the slip velocity s - u_tau / kappa and the cell gets the
// eddy viscosity kappa u_tau d, so the bounce-back passes the turbulent wall stress u_tau^2 to the air.
// Returns (slip velocity, eddy viscosity).
fn wall_slip(u: vec2f, n: vec2f) -> vec3f {
  var r = vec3f(0.0);
  if (dot(n, n) > 0.5) {
    let ut = u - dot(u, n) * n;
    let s = length(ut);
    if (s > 1e-9) {
      let utau = s / wall_uplus(s * WALL_D / P.nu);
      r = vec3f(ut * (max(s - utau / KAPPA, 0.0) / s), KAPPA * utau * WALL_D);
    }
  }
  return r;
}

fn lut_index(which: i32, t: f32) -> i32 {
  return which * 256 + i32(clamp(t, 0.0, 1.0) * 255.0);
}
`;

// helpers that read a buffer: only included by the kernels that declare it
const LOAD_FA = /* wgsl */ `
fn load_fa(c: i32) -> array<f32, 9> {
  var f: array<f32, 9>;
${each(K9, (k) => `  f[${k}] = fa[${k} * N + c];`)}
  return f;
}`;
const LOAD_STORE_FB = /* wgsl */ `
fn load_fb(c: i32) -> array<f32, 9> {
  var f: array<f32, 9>;
${each(K9, (k) => `  f[${k}] = fb[${k} * N + c];`)}
  return f;
}

// fb[c] = a + b - d (non-equilibrium extrapolation of the open boundaries)
fn store_fb_sum(c: i32, a: array<f32, 9>, b: array<f32, 9>, d: array<f32, 9>) {
${each(K9, (k) => `  fb[${k} * N + c] = a[${k}] + b[${k}] - d[${k}];`)}
}`;
const SAMPLE_VEL = /* wgsl */ `
fn sample_vel(pos: vec2f) -> vec2f {
  let x = clamp(pos.x, 0.0, f32(NX) - 1.001);
  let y = clamp(pos.y, 0.0, f32(NY) - 1.001);
  let i = i32(x);
  let j = i32(y);
  let fx = x - f32(i);
  let fy = y - f32(j);
  return (1.0 - fx) * (1.0 - fy) * vel[ix(i, j)] + fx * (1.0 - fy) * vel[ix(i + 1, j)]
       + (1.0 - fx) * fy * vel[ix(i, j + 1)] + fx * fy * vel[ix(i + 1, j + 1)];
}`;

const SAMPLE_RHO = /* wgsl */ `
fn sample_rho(pos: vec2f) -> f32 {
  let x = clamp(pos.x, 0.0, f32(NX) - 1.001);
  let y = clamp(pos.y, 0.0, f32(NY) - 1.001);
  let i = i32(x);
  let j = i32(y);
  let fx = x - f32(i);
  let fy = y - f32(j);
  return (1.0 - fx) * (1.0 - fy) * rho[ix(i, j)] + fx * (1.0 - fy) * rho[ix(i + 1, j)]
       + (1.0 - fx) * fy * rho[ix(i, j + 1)] + fx * fy * rho[ix(i + 1, j + 1)];
}`;

const GRID = "@compute @workgroup_size(16, 16)\nfn main(@builtin(global_invocation_id) gid: vec3u)";
const LINE = "@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3u)";
const GRID_IJ = `let i = i32(gid.x);
  let j = i32(gid.y);
  if (i >= NX || j >= NY) {
    return;
  }
  let c = ix(i, j);`;

/** Compute kernels: name -> {uses: buffer names, size: "grid", "nx", "ny", "tracers", "workgroup" (one
 *  group of 256) or a number of threads, code}. */
function computeKernels() {
  return {
    init_flow: {
      uses: ["fa", "fb", "mask", "rho", "vel"], size: "grid", code: /* wgsl */ `
${GRID} {
  ${GRID_IJ}
  // uniform flow plus an up-draft blob behind the body. Without it a symmetric body (plate across
  // the flow) keeps a symmetric wake for 20-40 chords before the vortex street starts
  let r2 = ((f32(i) - PX - C) * (f32(i) - PX - C) + (f32(j) - PY) * (f32(j) - PY)) / ((0.5 * C) * (0.5 * C));
  var u = vec2f(U_LAT, START_KICK * U_LAT * exp(-r2));
  if (mask[c] != 0) {
    u = vec2f(0.0, 0.0);
  }
  let f = feq(1.0, u);
${each(K9, (k) => `  fa[${k} * N + c] = f[${k}];\n  fb[${k} * N + c] = f[${k}];`)}
  rho[c] = 1.0;
  vel[c] = u;
}`,
    },

    build_mask: {
      uses: ["mask_new"], size: "grid", code: /* wgsl */ `
${GRID} {
  ${GRID_IJ}
  var s = body_at(f32(i), f32(j), P.shape, P.a, P.m, P.p, P.mast_r, P.jtx, P.jty, P.jang, 0.0);
  if (i == 0 || j == 0 || i == NX - 1 || j == NY - 1) {
    s = 0;
  }
  mask_new[c] = s;
}`,
    },

    apply_mask: {
      uses: ["fa", "mask", "mask_new"], size: "grid", code: /* wgsl */ `
${GRID} {
  ${GRID_IJ}
  if (mask[c] != 0 && mask_new[c] == 0) {
    let f = feq(1.0, vec2f(0.0, 0.0));            // uncovered cell: fluid at rest
${each(K9, (k) => `    fa[${k} * N + c] = f[${k}];`)}
  }
  mask[c] = mask_new[c];
}`,
    },

    // mark the lattice links that cross a membrane (sail, jib, plate): populations bounce back there
    build_links: {
      uses: ["link"], size: "grid", code: /* wgsl */ `
${GRID} {
  ${GRID_IJ}
  let shape = P.shape;
  var bits = 0u;
  // NACA profiles are solid cells, plus a membrane along their camber line: it seals the thin
  // trailing edge, which is thinner than a cell over its last few percent of chord
  if (shape == SAIL || shape == MAST_SAIL || shape == JIB_MAIN || shape == PLATE
      || shape == NACA0012 || shape == NACA2412) {
    let q0 = membranes(f32(i), f32(j), shape, P.a, P.m, P.p, P.jtx, P.jty, P.jang);
${each(K9.slice(1), (k) => `    {
      let q1 = membranes(f32(i + ${E[k][0]}), f32(j + ${E[k][1]}), shape, P.a, P.m, P.p, P.jtx, P.jty, P.jang);
      if (crosses(q0.y, q1.y, q0.x, q1.x, C)) {
        bits |= ${1 << k}u;
      }
      if (shape == JIB_MAIN && crosses(q0.w, q1.w, q0.z, q1.z, JIB_CHORD * C)) {
        bits |= ${1 << (8 + k)}u;
      }
    }`)}
  }
  link[c] = bits;
}`,
    },

    reset_bbox: {
      uses: ["bbox_atomic"], size: 1, code: /* wgsl */ `
${LINE} {
  if (gid.x == 0u) {
    atomicStore(&bbox_atomic[0], NX);
    atomicStore(&bbox_atomic[1], NY);
    atomicStore(&bbox_atomic[2], -1);
    atomicStore(&bbox_atomic[3], -1);
  }
}`,
    },

    find_bbox: {
      uses: ["mask", "link", "bbox_atomic"], size: "grid", code: /* wgsl */ `
${GRID} {
  ${GRID_IJ}
  if (mask[c] != 0 || link[c] != 0u) {
    atomicMin(&bbox_atomic[0], i);
    atomicMin(&bbox_atomic[1], j);
    atomicMax(&bbox_atomic[2], i);
    atomicMax(&bbox_atomic[3], j);
  }
}`,
    },

    // Smagorinsky constant fades to zero at walls (van Driest style) so the wall friction stays low. Full-size
    // boundary layer: no damping, and near the walls the mixing length FS_MIXING * kappa * wall distance
    wall_damping: {
      uses: ["mask", "link", "cs", "bbox"], size: "grid", code: /* wgsl */ `
${GRID} {
  ${GRID_IJ}
  let r = select(${Math.trunc(4 * WALL_DAMPING)}, DAMP_R, P.fs != 0u);
  var d2 = f32(r * r);
  if (i >= bbox[0] - r && i <= bbox[2] + r && j >= bbox[1] - r && j <= bbox[3] + r) {
    for (var di = -r; di <= r; di++) {
      for (var dj = -r; dj <= r; dj++) {
        let cc = ix(clamp(i + di, 0, NX - 1), clamp(j + dj, 0, NY - 1));
        if (mask[cc] != 0 || link[cc] != 0u) {
          d2 = min(d2, f32(di * di + dj * dj));
        }
      }
    }
  }
  let g = 1.0 - exp(-sqrt(d2) / WALL_DAMPING);
  cs[c] = SMAGORINSKY * g * g;
  if (P.fs != 0u) {
    let dd = sqrt(d2);
    cs[c] = select(SMAGORINSKY, max(SMAGORINSKY, FS_MIXING * KAPPA * min(dd, FS_MIX_MAX)), dd <= FS_SLEEVE_CELLS);
  }
}`,
    },

    // unit normal (pointing into the air) of every air cell next to a wall, for the law of the wall: from the
    // camber line for the sail, plate and jib membranes, from the blocked neighbours for solid bodies
    build_wall: {
      uses: ["mask", "link", "wn"], size: "grid", code: /* wgsl */ `
${GRID} {
  ${GRID_IJ}
  var w = vec2f(0.0);
  if (mask[c] == 0 && i > 0 && j > 0 && i < NX - 1 && j < NY - 1) {
    let lk = link[c];
    let b = to_body(f32(i), f32(j), P.a);
    let ca = cos(P.a);
    let sa = sin(P.a);
    if ((lk & 0x1FEu) != 0u) {                     // main sail / plate / camber line
      var mm = P.m;
      var pp = P.p;
      if (P.shape == PLATE || P.shape == NACA0012) {
        mm = 0.0;
      } else if (P.shape == NACA2412) {
        mm = 0.02;
        pp = 0.4;
      }
      let ml = mean_line(clamp(b.x / C, 0.0, 1.0), mm, pp);
      var nb = normalize(vec2f(-ml.y, 1.0));
      if (b.y - ml.x * C < 0.0) {
        nb = -nb;
      }
      w = vec2f(nb.x * ca + nb.y * sa, -nb.x * sa + nb.y * ca);
    } else if (((lk >> 8u) & 0x1FEu) != 0u) {      // jib
      let jf = jib_frame(b.x, b.y, P.jtx, P.jty, P.jang);
      let ml = mean_line(clamp(jf.x / (JIB_CHORD * C), 0.0, 1.0), P.m, P.p);
      var nj = normalize(vec2f(-ml.y, 1.0));
      if (jf.y - ml.x * JIB_CHORD * C < 0.0) {
        nj = -nj;
      }
      let cj = cos(P.jang);
      let sj = sin(P.jang);
      let nb = vec2f(nj.x * cj + nj.y * sj, -nj.x * sj + nj.y * cj);
      w = vec2f(nb.x * ca + nb.y * sa, -nb.x * sa + nb.y * ca);
    } else {
      var ns = vec2f(0.0);
${each(K9.slice(1), (k) => `      if (mask[ix(i + ${E[k][0]}, j + ${E[k][1]})] != 0) {
        ns -= ${fl(W[k])} * vec2f(${fl(E[k][0])}, ${fl(E[k][1])});
      }`)}
      if (length(ns) > 1e-9) {
        w = normalize(ns);
      }
    }
  }
  wn[c] = w;
}`,
    },

    // fa holds post-collision populations: pull-stream them (bounce-back at walls), collide, store in fb
    step: {
      uses: ["fa", "fb", "mask", "link", "cs", "wn"], size: "grid", code: /* wgsl */ `
${LOAD_FA}

${COLLIDE}

${GRID} {
  let i = i32(gid.x);
  let j = i32(gid.y);
  if (i < 1 || i >= NX - 1 || j < 1 || j >= NY - 1) {
    return;
  }
  let c = ix(i, j);
  if (mask[c] != 0) {
    return;
  }
  var lk = link[c];
  lk = (lk | (lk >> 8u)) & 511u;                   // links blocked by any membrane (bits 1..8)
  var fin: array<f32, 9>;
  fin[0] = fa[c];
  var walls = 0u;
${each(K9.slice(1), (k) => `  {
    let s = c - (${E[k][0]}) - (${E[k][1]}) * NX;
    let wall = mask[s] != 0 || ((lk >> ${OPP[k]}u) & 1u) != 0u;   // solid cell, or the link crosses a sail
    fin[${k}] = select(fa[${k} * N + s], fa[${OPP[k]} * N + c], wall);   // halfway bounce-back
    walls |= select(0u, ${1 << k}u, wall);
  }`)}
  // full-size boundary layer: the wall moves with the slip velocity of the law of the wall
  var nut = 0.0;
  if (P.fs != 0u && walls != 0u) {
    let mo = moments(load_fa(c));
    let sl = wall_slip(mo.yz, wn[c]);
    nut = sl.z;
${each(K9.slice(1), (k) => `    if ((walls & ${1 << k}u) != 0u) {
      fin[${k}] += ${fl(6 * W[k])} * mo.x * dot(vec2f(${fl(E[k][0])}, ${fl(E[k][1])}), sl.xy);
    }`)}
  }
  let fo = collide(fin, i, j, c, nut);
${each(K9, (k) => `  fb[${k} * N + c] = fo[${k}];`)}
}`,
    },

    // top and bottom: open far field. The cross-flow comes from the far field (the wake pushes air out
    // sideways as in open air, but the whole wake cannot drift sideways); the along-flow velocity and
    // the density come from the neighbouring cell. Closed (mirror) walls made the domain a wind tunnel
    // that has to carry the drag with a pressure drop, which inflated separated-flow forces.
    boundaries_tb: {
      uses: ["fb"], size: "nx", code: /* wgsl */ `
${LOAD_STORE_FB}

${LINE} {
  let i = i32(gid.x);
  if (i < 1 || i >= NX - 1) {
    return;
  }
  for (var jj = 0; jj < 2; jj++) {
    let jb = select(NY - 1, 0, jj == 0);
    let jn = select(NY - 2, 1, jj == 0);
    let fnb = load_fb(ix(i, jn));
    let mn = moments(fnb);
    let un = mn.yz;
    let ub = vec2f(un.x, far_velocity(i, jb, P.gam, P.src).y);
    store_fb_sum(ix(i, jb), feq(mn.x, ub), fnb, feq(mn.x, un));
  }
}`,
    },

    boundaries_io: {
      uses: ["fb"], size: "ny", code: /* wgsl */ `
${LOAD_STORE_FB}

${LINE} {
  let j = i32(gid.x);
  if (j >= NY) {
    return;
  }
  // inlet: uniform velocity, density extrapolated, non-equilibrium part copied (Guo et al.)
  let f1 = load_fb(ix(1, j));
  let m1 = moments(f1);
  store_fb_sum(ix(0, j), feq(m1.x, far_velocity(0, j, P.gam, P.src)), f1, feq(m1.x, m1.yz));
  // outlet: pressure of the far field (Bernoulli), velocity extrapolated. A uniform pressure here
  // would ignore the vortex far field of a lifting body and add ~10 % lift in this short domain
  let f2 = load_fb(ix(NX - 2, j));
  let m2 = moments(f2);
  let uf = far_velocity(NX - 1, j, P.gam, P.src);
  let rf = 1.0 + 1.5 * (U_LAT * U_LAT - dot(uf, uf));
  store_fb_sum(ix(NX - 1, j), feq(rf, m2.yz), f2, feq(m2.x, m2.yz));
}`,
    },

    // momentum exchange on the wall links around the body, summed by one workgroup. With P.favg != 0 it
    // is the mean of the last two steps (fa and the previous populations still in fb): close to tau = 0.5
    // the populations alternate from one step to the next, and so does the force of a single step
    forces: {
      uses: ["fa", "fb", "mask", "link", "bbox", "out", "wn"], size: "workgroup", code: /* wgsl */ `
${LOAD_FA}

var<workgroup> red: array<vec2f, 768>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  // only the cells around the body
  let x0 = max(1, bbox[0] - 1);
  let y0 = max(1, bbox[1] - 1);
  let x1 = min(NX - 1, bbox[2] + 2);
  let y1 = min(NY - 1, bbox[3] + 2);
  let w = x1 - x0;
  let h = y1 - y0;
  var f1 = vec2f(0.0);
  var f2 = vec2f(0.0);
  var f3 = vec2f(0.0);
  if (w > 0 && h > 0) {
    for (var t = i32(li); t < w * h; t += 256) {
      let c = ix(x0 + t % w, y0 + t / w);
      if (mask[c] == 0) {
        let lk = link[c];
        var us = vec2f(0.0);                        // slip velocity of the wall (full-size boundary layer)
        var rw = 1.0;
        if (P.fs != 0u) {
          let n = wn[c];
          if (dot(n, n) > 0.5) {
            let mo = moments(load_fa(c));
            us = wall_slip(mo.yz, n).xy;
            rw = mo.x;
          }
        }
${each(K9.slice(1), (k) => `        {
          var s = mask[c + (${E[k][0]}) + (${E[k][1]}) * NX];
          if (s == 0) {
            if (((lk >> ${k}u) & 1u) != 0u) {
              s = 1;
            } else if (((lk >> ${8 + k}u) & 1u) != 0u) {
              s = 2;
            }
          }
          if (s != 0) {
            // population heading into the wall bounces back: momentum 2 f e_k. The rest-state part
            // (w_k) is removed, so parts touching each other (mast/main) do not pick up the static
            // pressure on their hidden contact faces; the total force is unchanged.
            let f2k = select(2.0 * fa[${k} * N + c], fa[${k} * N + c] + fb[${k} * N + c], P.favg != 0u);
            let ek = vec2f(${fl(E[k][0])}, ${fl(E[k][1])});
            let d = (f2k - ${fl(2 * W[k])} - ${fl(6 * W[k])} * rw * dot(ek, us)) * ek;
            if (s == 1) {
              f1 += d;
            } else if (s == 2) {
              f2 += d;
            } else {
              f3 += d;
            }
          }
        }`)}
      }
    }
  }
  red[li] = f1;
  red[256u + li] = f2;
  red[512u + li] = f3;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (li < st) {
      red[li] += red[li + st];
      red[256u + li] += red[256u + li + st];
      red[512u + li] += red[512u + li + st];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    out[0] = 0.0;
    out[1] = 0.0;
    out[2] = red[0].x;
    out[3] = red[0].y;
    out[4] = red[256].x;
    out[5] = red[256].y;
    out[6] = red[512].x;
    out[7] = red[512].y;
  }
}`,
    },

    macro: {
      uses: ["fa", "mask", "rho", "vel"], size: "grid", code: /* wgsl */ `
${LOAD_FA}

${GRID} {
  ${GRID_IJ}
  if (mask[c] == 0) {
    let mo = moments(load_fa(c));
    rho[c] = mo.x;
    vel[c] = mo.yz;
  } else {
    // inside a body: velocity 0, pressure of the neighbouring air (smooth colours at the edge)
    var r = 0.0;
    var cnt = 0;
${each([1, 2, 3, 4], (k) => `    {
      let cc = ix(clamp(i + ${E[k][0]}, 0, NX - 1), clamp(j + ${E[k][1]}, 0, NY - 1));
      if (mask[cc] == 0) {
        r += moments(load_fa(cc)).x;
        cnt += 1;
      }
    }`)}
    rho[c] = 1.0;
    if (cnt > 0) {
      rho[c] = r / f32(cnt);
    }
    vel[c] = vec2f(0.0, 0.0);
  }
}`,
    },

    render: {
      uses: ["vel", "rho", "lut", "img"], size: "grid", code: /* wgsl */ `
${GRID} {
  ${GRID_IJ}
  let view = P.view;
  var col = vec3f(0.08, 0.09, 0.11);              // smoke-only view: dark background
  if (view == 0) {
    col = lut[lut_index(0, length(vel[c]) / (2.0 * U_LAT))].xyz;
  } else if (view == 1) {
    let ip = min(i + 1, NX - 1);
    let im = max(i - 1, 0);
    let jp = min(j + 1, NY - 1);
    let jm = max(j - 1, 0);
    let w = 0.5 * ((vel[ix(ip, j)].y - vel[ix(im, j)].y) - (vel[ix(i, jp)].x - vel[ix(i, jm)].x));
    col = lut[lut_index(1, 0.5 + 0.5 * w * C / (U_LAT * VORTICITY_RANGE))].xyz;
  } else if (view == 2) {
    let cp = (rho[c] - 1.0) / 3.0 / (0.5 * U_LAT * U_LAT);
    var t = 0.5;
    if (cp < 0.0) {
      t = 0.5 - 0.5 * cp / CP_MIN;
    } else {
      t = 0.5 + 0.5 * cp / CP_MAX;
    }
    col = lut[lut_index(2, t)].xyz;
  }
  img[c] = vec4f(col, 1.0);
}`,
    },

    // flow speed along the lee side of the main, 2 cells off the cloth (negative = reversed = separated), and
    // the density (pressure) 1.5 cells off the cloth on both sides, for the full-size drag estimate
    leeward_profile: {
      uses: ["vel", "rho", "out"], size: LEE_SAMPLES, code: /* wgsl */ `
${SAMPLE_VEL}

${SAMPLE_RHO}

${LINE} {
  let k = i32(gid.x);
  if (k >= LEE_SAMPLES) {
    return;
  }
  let f = sail_frame((f32(k) + 0.5) / f32(LEE_SAMPLES), 1, P.a, P.m, P.p, 0.0, 0.0, 0.0);
  let t = f.zw;
  let nrm = vec2f(-t.y, t.x);
  let u = sample_vel(f.xy + 2.0 * nrm);
  out[OUT_LEE + k] = dot(u, t) / U_LAT;
  out[OUT_RHO + k] = sample_rho(f.xy + 1.5 * nrm);
  out[OUT_RHO + LEE_SAMPLES + k] = sample_rho(f.xy - 1.5 * nrm);
}`,
    },

    // Telltale ribbons on both sides of the sails, and one on each leech. Each one follows the local air
    // flow from where it is tied on, so it streams aft in attached flow and lifts, points forward or wanders
    // where the flow has separated - like the yarns on a real sail. A leech telltale streams straight aft
    // when the flow leaves the leech cleanly and curls behind the sail when the leech stalls.
    // Here only the ribbon points are computed (in cells); the page draws them.
    telltales: {
      uses: ["vel", "mask", "out"], size: TELLTALES.length, code: /* wgsl */ `
${SAMPLE_VEL}

${LINE} {
  let k = i32(gid.x);
  if (k >= N_TELLTALES) {
    return;
  }
  let shape = P.shape;
  let base = OUT_TT + k * TT_STRIDE;
  let seg = ${fl(TELLTALE_LENGTH)} * C / f32(TELLTALE_SEGMENTS);
  let sail = TT_SAIL[k];
  let side = TT_SIDE[k];
  var count = 0;
  if (sail == 1 || shape == JIB_MAIN) {
    let f = sail_frame(TT_POS[k], sail, P.a, P.m, P.p, P.jtx, P.jty, P.jang);
    let t = f.zw;
    let root = f.xy;
    var pos = root + 1.0 * side * vec2f(-t.y, t.x);    // tied on just off the cloth
    if (abs(side) < 0.5) {
      pos = root + 1.0 * t;                           // leech: just behind the trailing edge
    }
    out[base + 1] = pos.x;
    out[base + 2] = pos.y;
    count = 1;
    var d = t;
    var alive = true;
    for (var s = 0; s < TELLTALE_SEGMENTS; s++) {
      if (alive) {
        let u = sample_vel(pos);
        if (length(u) > 1e-6) {
          d = normalize(u);
        }
        let q = pos + seg * d;
        // the ribbon cannot pass through the cloth or the mast
        let q0 = membranes(pos.x, pos.y, shape, P.a, P.m, P.p, P.jtx, P.jty, P.jang);
        let q1 = membranes(q.x, q.y, shape, P.a, P.m, P.p, P.jtx, P.jty, P.jang);
        if (crosses(q0.y, q1.y, q0.x, q1.x, C)) {
          alive = false;
        }
        if (shape == JIB_MAIN && crosses(q0.w, q1.w, q0.z, q1.z, JIB_CHORD * C)) {
          alive = false;
        }
        let qi = i32(clamp(q.x, 0.0, f32(NX) - 1.0));
        let qj = i32(clamp(q.y, 0.0, f32(NY) - 1.0));
        if (mask[ix(qi, qj)] != 0) {
          alive = false;
        }
        if (alive) {
          out[base + 1 + 2 * count] = q.x;
          out[base + 2 + 2 * count] = q.y;
          count += 1;
          pos = q;
        }
      }
    }
  }
  out[base] = f32(count);
}`,
    },

    reset_tracers: {
      uses: ["tracers"], size: "tracers", code: /* wgsl */ `
${LINE} {
  let p = i32(gid.x);
  if (p >= N_LINES * PER_LINE) {
    return;
  }
  let line = p / PER_LINE;
  let k = p % PER_LINE;
  let y0 = PY + (f32(line) + 0.5 - 0.5 * f32(N_LINES)) * (2.2 * C / f32(N_LINES));
  // as if the rake had been releasing smoke for a long time: particle k left it PER_LINE - k releases ago
  // (the next release is particle 0); the ones that would be past the outlet wait out of view
  var x = RAKE_X + f32(PER_LINE - k) / SMOKE_PER_CELL;
  if (x >= f32(NX - 2)) {
    x = f32(NX) - 1.0;
  }
  tracers[p].pos = vec2f(x, y0);
}`,
    },

    advect: {
      uses: ["tracers", "vel", "mask"], size: "tracers", code: /* wgsl */ `
${SAMPLE_VEL}

${LINE} {
  let p = i32(gid.x);
  if (p >= N_LINES * PER_LINE) {
    return;
  }
  // Each rake line releases smoke at a steady rate (SMOKE_PER_CELL particles per cell of free stream),
  // re-using its particles in turn, the oldest first. This frame releases P.emit_n of them, starting with
  // number P.emit0: each starts at the rake at its own moment within the frame and moves for the rest of
  // the frame (same number of sub-steps, shorter ones). (Sending particles back to the rake when they left
  // the domain made the streaks dashed: they arrived in clumps from the wake vortices, and a fast GPU
  // dropped a whole frame's returns on one point.)
  let shape = P.shape;
  let line = p / PER_LINE;
  let y0 = PY + (f32(line) + 0.5 - 0.5 * f32(N_LINES)) * (2.2 * C / f32(N_LINES));
  let slot = (p % PER_LINE - i32(P.emit0) + PER_LINE) % PER_LINE;
  var dt = P.dt;
  var pos = tracers[p].pos;
  if (slot < P.emit_n) {
    let frame_steps = P.dt * f32(P.nsub);
    let released = (f32(slot) + 0.5) / f32(P.emit_n) * frame_steps;   // steps after the frame started
    pos = vec2f(RAKE_X, y0);
    dt = (frame_steps - released) / f32(P.nsub);
  }
  let waiting = pos.x >= f32(NX - 2);              // parked past the outlet until its next release
  for (var s = 0; s < select(P.nsub, 0, waiting); s++) {
    let u1 = sample_vel(pos);
    let u2 = sample_vel(pos + 0.5 * dt * u1);
    let nw = pos + dt * u2;
    // smoke cannot pass through a sail
    let q0 = membranes(pos.x, pos.y, shape, P.a, P.m, P.p, P.jtx, P.jty, P.jang);
    let q1 = membranes(nw.x, nw.y, shape, P.a, P.m, P.p, P.jtx, P.jty, P.jang);
    var blocked = false;
    if (shape == SAIL || shape == MAST_SAIL || shape == JIB_MAIN || shape == PLATE
        || shape == NACA0012 || shape == NACA2412) {
      if (crosses(q0.y, q1.y, q0.x, q1.x, C)) {
        blocked = true;
      }
    }
    if (shape == JIB_MAIN && crosses(q0.w, q1.w, q0.z, q1.z, JIB_CHORD * C)) {
      blocked = true;
    }
    if (!blocked) {
      pos = nw;
    }
  }
  let i = i32(clamp(pos.x, 0.0, f32(NX) - 1.0));
  let j = i32(clamp(pos.y, 0.0, f32(NY) - 1.0));
  if (pos.x >= f32(NX - 2) || pos.y < 1.0 || pos.y > f32(NY - 2) || mask[ix(i, j)] != 0) {
    pos = vec2f(f32(NX) - 1.0, y0);                // out of view until its next release
  }
  tracers[p].pos = pos;
  tracers[p].norm = vec2f((pos.x - VX0) / VW, (pos.y - VY0) / VH);
}`,
    },
  };
}

// KBC collision with the Smagorinsky sub-grid model and the absorbing layers (Flowniac.Solver._collide)
const COLLIDE = /* wgsl */ `
fn collide(fin: array<f32, 9>, i: i32, j: i32, c: i32, nut: f32) -> array<f32, 9> {
  let mo = moments(fin);
  let rho = mo.x;
  let u = mo.yz;
  let fe = feq(rho, u);
  var df: array<f32, 9>;
${each(K9, (k) => `  df[${k}] = fin[${k}] - fe[${k}];`)}
  let pxx = df[1] + df[3] + df[5] + df[6] + df[7] + df[8];
  let pyy = df[2] + df[4] + df[5] + df[6] + df[7] + df[8];
  let pxy = df[5] - df[6] + df[7] - df[8];
  // effective relaxation time: molecular + Smagorinsky sub-grid viscosity (or, in a wall cell of the full-size
  // boundary layer, the eddy viscosity nut of the law of the wall) + outlet sponge
  var tau = P.tau0;
  let csv = cs[c];
  if (nut > 0.0) {
    tau = P.tau0 + 3.0 * nut;
  } else if (csv > 0.0) {
    let qn = sqrt(pxx * pxx + pyy * pyy + 2.0 * pxy * pxy);
    tau = 0.5 * (tau + sqrt(tau * tau + 18.0 * 1.41421356 * csv * csv * qn / rho));
  }
  let x0 = f32(NX - 1) - SPONGE;
  var so = 0.0;                                     // outlet sponge ramp 0..1
  if (f32(i) > x0) {
    so = (f32(i) - x0) / SPONGE;
  }
  tau += (TAU_SPONGE - tau) * so * so;
  // KBC entropic collision (Karlin, Boesch, Chikatamarla 2014): the shear part ds relaxes with 1/tau,
  // the higher-order part dh with a rate chosen to keep the entropy balance -> stable as tau -> 0.5
  let beta = 0.5 / tau;
  let nn = pxx - pyy;
  let ds = array<f32, 9>(0.0, 0.25 * nn, -0.25 * nn, 0.25 * nn, -0.25 * nn,
                         0.25 * pxy, -0.25 * pxy, 0.25 * pxy, -0.25 * pxy);
  var dh: array<f32, 9>;
  var sh = 0.0;
  var hh = 0.0;
${each(K9, (k) => `  dh[${k}] = df[${k}] - ds[${k}];
  sh += ds[${k}] * dh[${k}] / max(fe[${k}], 1e-12);
  hh += dh[${k}] * dh[${k}] / max(fe[${k}], 1e-12);`)}
  var gamma = 2.0;
  if (hh > 1e-14) {
    gamma = 1.0 / beta - (2.0 - 1.0 / beta) * sh / hh;
  }
  var fp: array<f32, 9>;
${each(K9, (k) => `  fp[${k}] = fin[${k}] - beta * (2.0 * ds[${k}] + gamma * dh[${k}]);`)}
  // absorbing layers at inlet, outlet, top and bottom: relax towards the undisturbed stream so that
  // sound waves and vortices leave the domain instead of echoing back onto the sail. The outlet layer
  // relaxes only the pressure: pulling the wake back to free-stream speed 2.5 chords behind the body
  // pushed it like a fan and nearly doubled the drag of fully separated flow (plate across the flow)
  let ss = max(0.0, max((SIDE_SPONGE - f32(min(j, NY - 1 - j))) / SIDE_SPONGE, (SIDE_SPONGE - f32(i)) / SIDE_SPONGE));
  let sig_out = SIGMA_SPONGE * so * so;
  let sig_side = SIGMA_SPONGE * ss * ss;
  if (sig_out + sig_side > 0.0) {
    let uf = far_velocity(i, j, P.gam, P.src);
    let rf = 1.0 + 1.5 * (U_LAT * U_LAT - dot(uf, uf));    // Bernoulli
    if (sig_out > sig_side) {
      let m2 = moments(fp);
      let fe2 = feq(rf, m2.yz);
${each(K9, (k) => `      fp[${k}] += sig_out * (fe2[${k}] - fp[${k}]);`)}
    } else {
      let fe2 = feq(rf, uf);
${each(K9, (k) => `      fp[${k}] += sig_side * (fe2[${k}] - fp[${k}]);`)}
    }
  }
  return fp;
}`;

/** Draw passes: the flow picture (compose) and the smoke particles (splat_tracers). */
function renderKernels() {
  return {
    // Draw the flow at screen resolution: colours interpolated between cells, the dinghy outline
    // (P.hull != 0), and the bodies evaluated from their exact shape with 3x3 sub-samples (smooth,
    // anti-aliased edges). Pixel (0, 0) is the top left corner of the picture.
    compose: {
      uses: ["img", "bbox", "hull"], code: /* wgsl */ `
fn img_bilinear(xi: f32, yi: f32) -> vec3f {
  let x = clamp(xi, 0.0, f32(NX) - 1.001);
  let y = clamp(yi, 0.0, f32(NY) - 1.001);
  let i = i32(x);
  let j = i32(y);
  let fx = x - f32(i);
  let fy = y - f32(j);
  return ((1.0 - fx) * (1.0 - fy) * img[ix(i, j)] + fx * (1.0 - fy) * img[ix(i + 1, j)]
        + (1.0 - fx) * fy * img[ix(i, j + 1)] + fx * fy * img[ix(i + 1, j + 1)]).xyz;
}

fn body_color(s: i32, view: i32) -> vec3f {
  var col = vec3f(0.94, 0.94, 0.94);                // (main) sail / body
  if (s == 2) {
    col = vec3f(0.76, 0.85, 0.98);                  // jib
  } else if (s == 3) {
    col = vec3f(0.58, 0.58, 0.60);                  // mast
  }
  if (view == 2) {
    col *= 0.3;                                     // dark bodies on the light pressure view
  }
  return col;
}

// anti-aliased dinghy outline (half width r pixels, dark rim) over colour col at picture pixel q
fn hull_blend(col_in: vec3f, q: vec2f, r: f32, view: i32) -> vec3f {
  var col = col_in;
  var d = 1e9;
  for (var k = 0; k < HULL_SEGMENTS; k++) {
    let a = hull[k];
    let ab = hull[k + 1] - a;
    let t = clamp(dot(q - a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    d = min(d, length(q - a - t * ab));
  }
  var core = vec3f(0.98, 0.80, 0.42);               // light wood
  if (view == 2) {
    core = vec3f(0.32, 0.20, 0.07);                 // dark on the light pressure view
  } else {
    col = mix(col, vec3f(0.03, 0.03, 0.03), 0.7 * clamp(r + 1.5 - d, 0.0, 1.0));
  }
  return mix(col, core, clamp(r + 0.5 - d, 0.0, 1.0));
}

@vertex
fn vs(@builtin(vertex_index) v: u32) -> @builtin(position) vec4f {
  let xy = vec2f(f32((v << 1u) & 2u), f32(v & 2u));   // full-screen triangle
  return vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let fw = P.fw;
  let fh = P.fh;
  var sx = VW / fw;                                 // cells per pixel (the same in both directions)
  var u = frag.x / fw;                              // along the wind
  var v = 1.0 - frag.y / fh;                        // across the wind
  if (ROTATED) {                                    // wind from the top of the picture
    sx = VW / fh;
    u = frag.y / fh;
    v = frag.x / fw;
  }
  let x = VX0 + u * VW;
  let y = VY0 + v * VH;
  var col = img_bilinear(x, y);
  if (P.hull != 0) {
    let q = floor(frag.xy);
    if (q.x >= P.hb0 && q.y >= P.hb1 && q.x <= P.hb2 && q.y <= P.hb3) {
      col = hull_blend(col, frag.xy, P.hull_r, P.view);
    }
  }
  let th = max(2.5 * sx, DRAW_THICKNESS * C);       // drawn sail thickness: >= 2.5 pixels
  if (x > f32(bbox[0]) - 3.0 && x < f32(bbox[2]) + 3.0 && y > f32(bbox[1]) - 3.0 && y < f32(bbox[3]) + 3.0) {
    var cover = 0.0;
    var sid = 0;
    for (var su = 0; su < 3; su++) {
      for (var sv = 0; sv < 3; sv++) {
        let s = body_at(x + f32(su - 1) * sx / 3.0, y + f32(sv - 1) * sx / 3.0,
                        P.shape, P.a, P.m, P.p, P.mast_r, P.jtx, P.jty, P.jang, th);
        if (s != 0) {
          cover += 1.0 / 9.0;
          sid = s;
        }
      }
    }
    if (sid != 0) {
      col = col * (1.0 - cover) + body_color(sid, P.view) * cover;
    }
  }
  return vec4f(col, 1.0);
}`,
    },

    // smoke particles as round dots of P.dot pixels with an anti-aliased edge (blended), so the streaks
    // look smooth instead of stair-stepped; larger on big, high-resolution pictures
    splat_tracers: {
      uses: ["tracers"], code: /* wgsl */ `
struct DotOut {
  @builtin(position) pos: vec4f,
  @location(0) off: vec2f,          // pixels from the dot centre
}

@vertex
fn vs(@builtin(vertex_index) v: u32, @builtin(instance_index) p: u32) -> DotOut {
  let q = tracers[p].norm;
  var pix = vec2f(q.x * P.fw, (1.0 - q.y) * P.fh);
  if (ROTATED) {
    pix = vec2f(q.y * P.fw, q.x * P.fh);
  }
  let h = 0.5 * P.dot + 1.0;        // half size of the quad: the dot plus a pixel for the soft edge
  let corner = vec2f(select(-h, h, v == 1u || v == 4u || v == 5u), select(-h, h, v == 2u || v == 3u || v == 5u));
  let xy = pix + corner;
  var o: DotOut;
  o.pos = vec4f(xy.x / P.fw * 2.0 - 1.0, 1.0 - xy.y / P.fh * 2.0, 0.0, 1.0);
  o.off = corner;
  return o;
}

@fragment
fn fs(d: DotOut) -> @location(0) vec4f {
  let cover = clamp(0.5 * P.dot + 0.5 - length(d.off), 0.0, 1.0);
  return vec4f(P.smoke_r, P.smoke_g, P.smoke_b, cover);
}`,
    },
  };
}

/** Full WGSL source of one kernel: constants, parameter block, its buffers, shared functions, code. */
function source(sim, kernel, access) {
  const bindings = kernel.uses.map((name, k) =>
    `@group(0) @binding(${k + 1}) var<storage, ${name === "bbox_atomic" ? "read_write" : access}> ` +
    `${name}: ${BUFFER_TYPES[name]};`).join("\n");
  return `${header(sim)}
@group(0) @binding(0) var<uniform> P: Params;
${bindings}
${COMMON}
${kernel.code}
`;
}

/** All kernels of a grid, with their WGSL source. */
export function buildKernels(sim) {
  const compute = computeKernels();
  for (const k of Object.values(compute)) k.wgsl = source(sim, k, "read_write");
  const render = renderKernels();
  for (const k of Object.values(render)) k.wgsl = source(sim, k, "read");
  return { compute, render };
}
