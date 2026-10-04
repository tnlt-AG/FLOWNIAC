// Settings and user state. Started as a copy of the "Settings" section and class State of Flowniac.py
// (same names in UPPER_CASE, same values); only the web version is developed further.

export const VERSION = "v1.1";          // shown next to the title; raise it with each release

export const RHO_AIR = 1.225;           // kg/m^3, sea level, 15 degC
export const NU_AIR = 1.46e-5;          // m^2/s, kinematic viscosity of air at 15 degC
export const SPAN_EFFICIENCY = 0.9;     // Oswald factor e for the whole-sail induced-drag estimate

export const U_LAT = 0.05;              // inflow speed in lattice units (Mach 0.09): suction peaks reach 3-4x this
export const RE_PER_CELL2 = 2.0;        // simulated Re = this * (cells per chord)^2 -> boundary layer ~3.5 cells thick
export const SMAGORINSKY = 0.15;        // sub-grid model constant away from walls, keeps shed vortices stable
export const WALL_DAMPING = 4.0;        // sub-grid model fades out within this many cells of a wall
export const DOMAIN_CHORDS = [6.0, 4.0];    // domain length and height, in chords
export const PIVOT_CHORDS = [2.5, 2.0];     // rotation point (mid-chord of the main sail), from inlet and bottom
export const SPONGE_CHORDS = 1.0;           // absorbing layer in front of the outlet
export const SIDE_SPONGE_CHORDS = 0.3;      // absorbing layer along the top and bottom walls and the inlet
export const TAU_SPONGE = 0.8;              // extra viscosity reached at the outlet (damps the wake)
export const SIGMA_SPONGE = 0.05;           // per-step pull towards undisturbed flow at the edge (kills sound echoes)
export const START_KICK = 0.5;              // up-draft behind the body at the start (fraction of U): starts the vortex street

// Boundary layer "full size (approx.)": the viscosity of the real Reynolds number (wind x sail width), the
// turbulent law of the wall at the sails (slip velocity and wall-cell eddy viscosity from Spalding's law) and
// strong mixing near the walls. The grid cannot resolve the thin turbulent boundary layer of a real sail; this
// mixing stands in for it so the flow stays attached about as long as on a real sail, but it makes the layer
// far too thick, so the simulated drag is too high: the drag shown is estimated instead (results.js).
// Tuned for the single sail on the High and Ultra grids.
export const BOUNDARY_LAYERS = ["Model size", "Full size (approx.)"];
export const KAPPA = 0.41;              // von Karman constant
export const B_LOG = 5.2;               // log-law intercept
export const FS_MIXING = 1.2;           // near-wall mixing length = this * KAPPA * wall distance ...
export const FS_MIXING_MAX = 0.03;      // ... up to this fraction of the chord
export const FS_SLEEVE = 0.08;          // the extra mixing acts within this fraction of the chord of a wall
export const FS_FORM_FACTOR = 2.0;      // drag estimate: form drag of attached flow = this x skin friction

export const QUALITY_CELLS = { low: 64, medium: 100, high: 160, ultra: 256 };   // cells per chord
export const AUTO_SECONDS_PER_CHORD = 3.0;  // "auto" picks the finest grid that moves the flow one chord in this time
export const TARGET_FPS = 20;
export const SLOW_MOTION = [0, 1, 2, 3, 5, 10, 20, 50];   // playback slower than the real wind; 0 = as fast as the GPU allows
export const UI_SCALE = 2.0;            // line widths in the picture (the control column scales with the browser zoom)
export const WIND_FROM = "top";         // "top": wind blows from the top of the picture (sailors' view); "left": landscape
export const SOLVER_SHARE = 0.6;        // at least this fraction of each frame goes to the flow solver

export const SETTLE_CHORDS = 2.0;       // flow passes (in chords) ignored after a change before averaging forces
export const AVERAGE_CHORDS = 5.0;      // averaging length before a point is added to the polar plot (shedding is slow)

export const JIB_CHORD = 0.7;           // jib chord as a fraction of the main chord
export const JIB_AOA_DEFAULT = 5.0;     // jib angle of attack at the start (best L/D with the main at 12 deg, heading 30)
export const JIB_MIN_SLOT = 0.02;       // the jib angle of attack stops where this slot (fraction of main chord) remains
export const SMOKE_OPACITY = 0.6;      // smoke dots are partly transparent: flow colours and sails show through
export const DRAW_THICKNESS = 0.02;     // drawn thickness of sails and plate (fraction of chord); they are simulated as zero
export const CYLINDER_DIAMETER = 0.4;   // cylinder diameter as a fraction of the chord
export const HULL_LENGTH = 1.5;         // dinghy outline (drawing only): length / main chord, about Laser proportions
export const HULL_BEAM = 0.32;          // beam / hull length
export const HULL_MAST = 1.0 / 3.0;     // mast (leading edge of the object) position from the bow, fraction of the length
export const HULL_WIDEST = 0.6;         // widest point from the bow, fraction of the length
export const HULL_TRANSOM = 0.72;       // transom width / beam
export const HULL_POINTS = 40;          // outline points per side
export const VORTICITY_RANGE = 25.0;    // colour scale for vorticity, in units of U / chord
export const CP_RANGE = [-3.0, 1.0];    // colour scale for the pressure coefficient

export const SHAPES = [  // [key, display name]
  ["sail", "Sail"],
  ["mast_sail", "Mast + sail"],
  ["jib_main", "Jib + main (with mast)"],
  ["plate", "Flat plate"],
  ["cylinder", "Cylinder"],
  ["naca0012", "NACA 0012 (wing sail)"],
  ["naca2412", "NACA 2412 (cambered)"],
];
export const [SAIL, MAST_SAIL, JIB_MAIN, PLATE, CYLINDER, NACA0012, NACA2412] = SHAPES.map((_, k) => k);
export const SAIL_SHAPES = [SAIL, MAST_SAIL, JIB_MAIN];
export const TELLTALE_LENGTH = 0.07;    // telltale ribbon length, fraction of the chord
export const TELLTALE_SEGMENTS = 6;
export const TELLTALES = [  // [sail: 1 main / 2 jib, position along its chord,
                            //  side: +1 leeward (red) / -1 windward (green) / 0 leech telltale on the trailing edge (yellow)]
  [1, 0.15, 1], [1, 0.15, -1], [1, 0.5, 1], [1, 0.5, -1], [1, 1.0, 0],
  [2, 0.15, 1], [2, 0.15, -1], [2, 0.5, 1], [2, 0.5, -1], [2, 1.0, 0],
];
export const LEE_SAMPLES = 40;          // points along the lee side of the main used for 'leeward flow attached'
export const VIEWS = ["Speed", "Vorticity", "Pressure (Cp)", "Smoke only"];
export const FORCE_AXES = ["Lift / drag", "Drive / side"];   // how forces are split: wind axes, or boat axes (whole sail)
export const ELEMENT_NAMES = { 1: "Main", 2: "Jib", 3: "Mast" };

// D2Q9 lattice: rest, 4 axis directions, 4 diagonals
export const E = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]];
export const W = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
export const OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6];

export const HELP_LINES = [
  "Keys (Shift = increase / finer):",
  "  Up/Down  angle of attack",
  "  Left/Right  camber    d/D  draft",
  "  1-7  shape    v  view    t  smoke",
  "  f  arrows   l  telltales   p  polar",
  "  b/B  boat heading    n  boat",
  "  x  forces: lift/drag or drive/side",
  "  w/W  wind    c/C  width    h/H  height",
  "  m/M  mast    j/J  jib angle of attack",
  "  r  reset flow    Backspace  clear polar",
  "  s/S  slow motion    Space  pause",
  "  i  help",
];

const clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const round = (v, digits) => Number(v.toFixed(digits));
const radians = (deg) => deg * Math.PI / 180;

/** NACA 4-digit mean line: [y/c, dy/dx] at x/c = x for camber m at position p (as mean_line in shaders.js). */
export function meanLine(x, m, p) {
  if (m <= 0.0) return [0.0, 0.0];
  if (x < p) return [m / (p * p) * (2 * p * x - x * x), 2 * m / (p * p) * (p - x)];
  const q = (1 - p) ** 2;
  return [m / q * (1 - 2 * p + 2 * p * x - x * x), 2 * m / q * (p - x)];
}

/**
 * Where the jib sits, in the frame of the main sail (luff at 0, chord along +x, lee side +y; in main chords):
 * its tack at the bow of the boat (HULL_MAST * HULL_LENGTH ahead of the mast on the centreline, which is the
 * boom angle st.heading - st.aoa off the main chord), and its chord at st.jib_aoa to the wind.
 * Returns {tx, ty, ang}: tack, and jib chord angle relative to the main chord (rad, + = towards windward).
 */
export function jibPlacement(st) {
  const boom = radians(st.heading - st.aoa);
  const d = HULL_MAST * HULL_LENGTH;
  return { tx: -d * Math.cos(boom), ty: d * Math.sin(boom), ang: radians(st.jib_aoa - st.aoa) };
}

/** Points along the jib's camber line in the main frame (main chords). */
function jibPoints(st, n = 60) {
  const { tx, ty, ang } = jibPlacement(st);
  const cj = Math.cos(ang);
  const sj = Math.sin(ang);
  const pts = [];
  for (let k = 0; k <= n; k++) {
    const x = k / n;
    const qx = x * JIB_CHORD;
    const qy = meanLine(x, st.camber, st.draft)[0] * JIB_CHORD;
    pts.push([tx + qx * cj + qy * sj, ty - qx * sj + qy * cj]);
  }
  return pts;
}

/**
 * Slot and overlap of the jib: the smallest distance from the jib to the main (lee side of its camber line, or
 * the mast; negative if the jib is to windward of the main or touches it), and how far the jib leech reaches
 * behind the main luff. Both in main chords.
 */
export function jibSlot(st) {
  const r = 0.5 * st.mast;
  let slot = Infinity;
  for (const [x, y] of jibPoints(st)) {
    slot = Math.min(slot, Math.hypot(x, y) - r);
    if (x >= 0.0 && x <= 1.0) {
      const [yc, dy] = meanLine(x, st.camber, st.draft);
      slot = Math.min(slot, (y - yc) / Math.sqrt(1 + dy * dy));
    }
  }
  const pts = jibPoints(st, 1);
  return { slot, overlap: pts[1][0] };
}

/** Largest jib angle of attack (deg) up to which, sheeting in from fully eased, the jib keeps JIB_MIN_SLOT. */
export function jibAoaLimit(st) {
  const s = Object.assign(Object.create(Object.getPrototypeOf(st)), st);
  let last = null;
  for (let a = -30.0; a <= 90.0; a += 0.25) {
    s.jib_aoa = a;
    if (jibSlot(s).slot < JIB_MIN_SLOT) return last === null ? 90.0 : last;
    last = a;
  }
  return 90.0;
}

/** Everything the user can change. */
export class State {
  constructor() {
    this.shape = SAIL;
    this.aoa = 12.0;          // deg, between wind and (main) chord line
    this.camber = 0.10;       // sail depth, fraction of chord
    this.draft = 0.45;        // position of max depth, fraction of chord from the luff
    this.mast = 0.05;         // mast diameter / chord
    this.jib_aoa = JIB_AOA_DEFAULT;   // deg, between wind and jib chord; the jib tack is at the bow
    this.wind = 8.0;          // m/s
    this.width = 3.0;         // m, main chord ("sail width")
    this.height = 9.0;        // m, sail height (luff)
    this.heading = 30.0;      // deg, boat centreline to the wind (to the right); drawing and drive/side only
    this.quality = "medium";  // grid preset (key of QUALITY_CELLS); main() sets the one it starts with
    this.boundary = 0;        // index into BOUNDARY_LAYERS: 0 model size, 1 full size (approx.)
    this.slowmo = 0;          // index into SLOW_MOTION
    this.view = 0;
    this.axes = 0;            // index into FORCE_AXES
    this.tracers = true;
    this.arrows = true;
    this.telltales = true;
    this.boat = true;
    this.polar = false;
    this.paused = false;
    this.help = false;
    this.reset_flow = false;
    this.clear_polar = false;
  }

  geometry() {
    return [this.shape, round(this.aoa, 3), round(this.camber, 4), round(this.draft, 4),
            round(this.mast, 4), round(this.jib_aoa, 3), this.shape === JIB_MAIN ? round(this.heading, 3) : 0].join();
  }

  configLabel() {
    let name = SHAPES[this.shape][1];
    if (SAIL_SHAPES.includes(this.shape)) {
      name += ` ${(this.camber * 100).toFixed(0)}%/${(this.draft * 100).toFixed(0)}%`;
    }
    if (this.shape === MAST_SAIL || this.shape === JIB_MAIN) name += ` mast ${(this.mast * 100).toFixed(0)}%`;
    if (this.shape === JIB_MAIN) {
      name += ` jib ${this.jib_aoa.toFixed(1)}deg heading ${this.heading.toFixed(0)}deg`;
    }
    return name;
  }

  /** Reference length (for coefficients and area) as a fraction of the main chord. */
  refFraction() {
    if (this.shape === CYLINDER) return CYLINDER_DIAMETER;
    if (this.shape === JIB_MAIN) return 1.0 + JIB_CHORD;
    return 1.0;
  }

  clamp() {
    this.shape = clip(Math.trunc(this.shape), 0, SHAPES.length - 1);
    this.aoa = clip(this.aoa, -30.0, 90.0);
    this.camber = clip(this.camber, 0.0, 0.20);
    this.draft = clip(this.draft, 0.25, 0.65);
    this.mast = clip(this.mast, 0.01, 0.12);
    this.wind = clip(this.wind, 1.0, 20.0);
    this.width = clip(this.width, 0.5, 10.0);
    this.height = clip(this.height, 1.0, 40.0);
    this.heading = clip(this.heading, 0.0, 180.0);
    this.jib_aoa = clip(this.jib_aoa, -30.0, 90.0);
    if (this.shape === JIB_MAIN) this.jib_aoa = Math.min(this.jib_aoa, jibAoaLimit(this));
    if (!(this.quality in QUALITY_CELLS)) this.quality = "medium";
    this.boundary = clip(Math.round(this.boundary), 0, BOUNDARY_LAYERS.length - 1);
    this.slowmo = clip(Math.round(this.slowmo), 0, SLOW_MOTION.length - 1);
    this.view = ((Math.trunc(this.view) % VIEWS.length) + VIEWS.length) % VIEWS.length;
    this.axes = ((Math.trunc(this.axes) % FORCE_AXES.length) + FORCE_AXES.length) % FORCE_AXES.length;
  }
}
