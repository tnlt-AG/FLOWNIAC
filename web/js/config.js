// Settings and user state. Started as a copy of the "Settings" section and class State of Flowniac.py
// (same names in UPPER_CASE, same values); only the web version is developed further.

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
export const DRAW_THICKNESS = 0.01;     // drawn thickness of sails and plate (fraction of chord); they are simulated as zero
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
  "  m/M  mast    g/G  jib gap",
  "  j/J  jib angle    o/O  jib overlap",
  "  r  reset flow    Backspace  clear polar",
  "  s/S  slow motion    Space  pause",
  "  i  help",
];

const clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const round = (v, digits) => Number(v.toFixed(digits));

/** Everything the user can change. */
export class State {
  constructor() {
    this.shape = SAIL;
    this.aoa = 12.0;          // deg, between wind and (main) chord line
    this.camber = 0.10;       // sail depth, fraction of chord
    this.draft = 0.45;        // position of max depth, fraction of chord from the luff
    this.mast = 0.05;         // mast diameter / chord
    this.jib_gap = 0.05;      // slot width between jib leech and main / main chord
    this.jib_overlap = 0.10;  // jib leech behind the main luff / main chord
    this.jib_angle = -15.0;   // jib chord angle relative to the main chord, deg (best L/D at main AoA 12)
    this.wind = 8.0;          // m/s
    this.width = 3.0;         // m, main chord ("sail width")
    this.height = 9.0;        // m, sail height (luff)
    this.heading = 30.0;      // deg, boat centreline to the wind (to the right); drawing and drive/side only
    this.quality = "medium";  // grid preset (key of QUALITY_CELLS); main() sets the one it starts with
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
            round(this.mast, 4), round(this.jib_gap, 4), round(this.jib_overlap, 4), round(this.jib_angle, 3)].join();
  }

  configLabel() {
    let name = SHAPES[this.shape][1];
    if (SAIL_SHAPES.includes(this.shape)) {
      name += ` ${(this.camber * 100).toFixed(0)}%/${(this.draft * 100).toFixed(0)}%`;
    }
    if (this.shape === MAST_SAIL || this.shape === JIB_MAIN) name += ` mast ${(this.mast * 100).toFixed(0)}%`;
    if (this.shape === JIB_MAIN) {
      const ja = this.jib_angle.toFixed(0);
      name += ` gap ${(this.jib_gap * 100).toFixed(0)}% ov ${(this.jib_overlap * 100).toFixed(0)}% ` +
              `jib ${this.jib_angle >= 0 ? "+" : ""}${ja}deg`;
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
    this.jib_gap = clip(this.jib_gap, 0.02, 0.25);
    this.jib_overlap = clip(this.jib_overlap, -0.2, 0.4);
    this.jib_angle = clip(this.jib_angle, -20.0, 15.0);
    this.wind = clip(this.wind, 1.0, 20.0);
    this.width = clip(this.width, 0.5, 10.0);
    this.height = clip(this.height, 1.0, 40.0);
    this.heading = clip(this.heading, 0.0, 180.0);
    if (!(this.quality in QUALITY_CELLS)) this.quality = "medium";
    this.slowmo = clip(Math.round(this.slowmo), 0, SLOW_MOTION.length - 1);
    this.view = ((Math.trunc(this.view) % VIEWS.length) + VIEWS.length) % VIEWS.length;
    this.axes = ((Math.trunc(this.axes) % FORCE_AXES.length) + FORCE_AXES.length) % FORCE_AXES.length;
  }
}
