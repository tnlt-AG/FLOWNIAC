"""
FLOWNIAC - educational 2D airflow simulator for sails and other shapes
======================================================================

Lattice-Boltzmann (D2Q9) flow solver running on the GPU with Taichi.
Originally based on LBM_Taichi by Wang (hietwll@gmail.com), https://github.com/hietwll/LBM_Taichi
Copyright (C) 2026 tnlt AG. Free software under the GNU General Public License v3.0, see LICENSE.

Physics model (read this before trusting the numbers)
-----------------------------------------------------
* Real sails (chord 2-5 m, wind 2-15 m/s) run at Reynolds numbers of 0.3-5 million. A grid that fits a
  laptop cannot resolve the boundary layer at that Re (it would be thinner than one cell and the result
  would depend on numerical details). So each quality preset simulates the highest Re its grid resolves:
  Re_sim = RE_PER_CELL2 * (cells per chord)^2, i.e. ~8'000 (low) ... ~130'000 (ultra). The collision
  operator is entropic (KBC) with a Smagorinsky sub-grid model in the wake for stability.
* Consequence: lift is close to full-size values, but the 2D section drag is higher than on a real sail
  (thicker, laminar boundary layers separate earlier). Higher presets get closer to full-size drag.
  Wind speed and sail size scale the forces in Newton:  F = C * 1/2 * rho * V^2 * A.
* The simulation is 2D. It gives the sail SECTION coefficients (no tip vortex). The whole-sail estimate
  adds induced drag  Cdi = Cl^2 / (pi * e * AR)  with aspect ratio AR = sail height / chord.
* Known 2D limitation: in fully separated flow (stall, AoA > ~20 deg, downwind, flat plate across the
  flow) the shed vortices stay too coherent without the third dimension, so forces there come out
  too high: the flat plate across the flow gives Cd ~3.1, like published 2D simulations (3-3.5), while
  experiments with very long plates give ~2.0 and a real sail (height ~3x chord) across the wind ~1.2.
  Attached and mildly separated flow (the useful upwind range) is fine.
* Sails and the flat plate have zero thickness, like real sail cloth: the membrane blocks every lattice
  link that crosses it (air bounces back on both sides). Mast, cylinder and NACA profiles are solid cells.
  Walls: halfway bounce-back. Forces: momentum exchange on the wall links.
  Inlet: far-field velocity. Outlet: far-field pressure. Top/bottom: open, cross-flow from the far field.
  Absorbing layers along all edges pull the flow towards the far field of a lifting body (free stream
  plus a vortex for the measured lift and a source for the measured drag), so sound waves and vortices
  leave the domain and the small domain behaves like open air instead of a narrow wind tunnel. The
  outlet layer only evens out the pressure: forcing the wake back to free-stream speed there, and closed
  top/bottom walls, had nearly doubled the drag of fully separated flow in this short domain.
* The dinghy outline is only drawn: the hull does not take part in the flow (the simulated section is at
  sail height, above the deck). Boat heading h = angle between the centreline and the wind the sail
  sees (the apparent wind). "Drive / side" splits the whole-sail force (with induced drag) along and
  across the boat:  drive = L sin(h) - D cos(h),  side = L cos(h) + D sin(h).

Validation (Medium preset = 100 cells per chord, Re_sim 20'000, unless noted; domain 6 x 4 chords;
averages over ~20 chords of flow, which still scatter by ~3-6 % because the flow sheds vortices)
------------------------------------------------------------------------------------------------
  Cylinder, Re 100:              Cd 1.40   (literature 1.33-1.35 unbounded)
  Sail 10 % camber (zero thickness), AoA 5/10/15/30:  L/D 9.3 / 10.9 / 5.8 / 1.9   (Cl 1.01 / 1.75 / 1.91 / 2.09)
  Mast + sail, AoA 10:           Cl 1.77  Cd 0.24  L/D 7.5   (the mast wake costs ~30 % of L/D)
  Jib + main, AoA 12:            Cl 0.97  Cd 0.12  L/D 8.3
  Sail AoA 10, High preset (Re_sim 51'000):  Cl 1.72  Cd 0.11  L/D 15.9
  Flat plate, AoA 5:             Cl 0.54   (thin-airfoil theory 0.55)
  Flat plate across the flow:    Cd 3.1    (2D simulations 3-3.5; experiments ~2.0, see above)
  Sail across the flow (AoA 90): Cd 3.1
  Same solver in a 20 x 16-chord domain (open air): flat plate AoA 5 Cl 0.54 Cd 0.056, jib + main AoA 12
  Cl 0.95 Cd 0.104. So lift is right; upwind drag here reads ~7-12 % high (L/D ~6-10 % low).
  Sails at 30 / 60 / 90 deg (50 cells per chord): forces within ~15 % of a 30 x 32-chord domain.
  References: thin cambered plates at Re ~1e4 reach L/D ~5-10 in wind tunnels; published 2D CFD of a
  full-size mast + sail gives Cl 1.6, Cd 0.19 (L/D 8.4) at 19 deg.
  Thick wing sections (NACA) suffer most from the low simulated Re: NACA 2412 at AoA 6 gives L/D 7.0 on
  Medium and only ~2.7 on Low (the rounded nose is ~1 cell there), but ~80-100 at full size. The flat plate
  at AoA 6 gives L/D 7.6 even on Low.

Usage
-----
    python Flowniac.py                         # auto-select quality from a quick GPU speed test
    python Flowniac.py --quality high          # low | medium | high | ultra
    python Flowniac.py --ui gui                # basic window (works without Vulkan)
    python Flowniac.py --ui-scale 2.5          # larger control column (default 2)
    python Flowniac.py --wind-from left        # wind from the left instead of from the top
    python Flowniac.py --shape jib_main --aoa 15
    python Flowniac.py --heading 45 --forces drive  # boat 45 deg off the wind, show drive and side force
    python Flowniac.py --snapshot out.png --chords 8  # run off-screen, save a picture, print results

Use the sliders in the left column with the mouse, or the keyboard (press "i" for the key list).

The same simulator runs in the browser (WebGPU): see web/README.md. Keep the two in step: a change to
the solver here needs the same change in web/js and a new parity reference (python web/tools/parity.py).
"""

import argparse
import math
import sys
import time

import numpy as np
import taichi as ti
import taichi.math as tm

# --------------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------------
RHO_AIR = 1.225           # kg/m^3, sea level, 15 degC
NU_AIR = 1.46e-5          # m^2/s, kinematic viscosity of air at 15 degC
SPAN_EFFICIENCY = 0.9     # Oswald factor e for the whole-sail induced-drag estimate

U_LAT = 0.05              # inflow speed in lattice units (Mach 0.09): suction peaks reach 3-4x this
RE_PER_CELL2 = 2.0        # simulated Re = this * (cells per chord)^2 -> boundary layer ~3.5 cells thick
SMAGORINSKY = 0.15        # sub-grid model constant away from walls, keeps shed vortices stable
WALL_DAMPING = 4.0        # sub-grid model fades out within this many cells of a wall
DOMAIN_CHORDS = (6.0, 4.0)    # domain length and height, in chords (8 long: same forces within the scatter)
PIVOT_CHORDS = (2.5, 2.0)     # rotation point (mid-chord of the main sail), from inlet and bottom
SPONGE_CHORDS = 1.0           # absorbing layer in front of the outlet
SIDE_SPONGE_CHORDS = 0.3      # absorbing layer along the top and bottom walls and the inlet
TAU_SPONGE = 0.8              # extra viscosity reached at the outlet (damps the wake)
SIGMA_SPONGE = 0.05           # per-step pull towards undisturbed flow at the edge (kills sound echoes)
START_KICK = 0.5              # up-draft behind the body at the start (fraction of U): starts the vortex street

QUALITY_CELLS = {"low": 64, "medium": 100, "high": 160, "ultra": 256}   # cells per chord
AUTO_SECONDS_PER_CHORD = 3.0  # "auto" picks the finest grid that moves the flow one chord in this time
TARGET_FPS = 20
UI_SCALE = 2.0            # size of the control column text and sliders (1 = small, 2 = large)
WINDOW_FRACTION = 0.75    # window width as a fraction of the screen width
WIND_FROM = "top"         # "top": wind blows from the top of the picture (sailors' view); "left": landscape
SOLVER_SHARE = 0.6       # at least this fraction of each frame goes to the flow solver

SETTLE_CHORDS = 2.0       # flow passes (in chords) ignored after a change before averaging forces
AVERAGE_CHORDS = 5.0      # averaging length before a point is added to the polar plot (shedding is slow)

JIB_CHORD = 0.7           # jib chord as a fraction of the main chord
DRAW_THICKNESS = 0.01     # drawn thickness of sails and plate (fraction of chord); they are simulated as zero
CYLINDER_DIAMETER = 0.4   # cylinder diameter as a fraction of the chord
HULL_LENGTH = 1.5         # dinghy outline (drawing only): length / main chord, about Laser proportions
HULL_BEAM = 0.32          # beam / hull length
HULL_MAST = 1.0 / 3.0     # mast (leading edge of the object) position from the bow, fraction of the length
HULL_WIDEST = 0.6         # widest point from the bow, fraction of the length
HULL_TRANSOM = 0.72       # transom width / beam
HULL_POINTS = 40          # outline points per side
VORTICITY_RANGE = 25.0    # colour scale for vorticity, in units of U / chord
CP_RANGE = (-3.0, 1.0)    # colour scale for the pressure coefficient

SHAPES = [  # (key, display name)
    ("sail", "Sail"),
    ("mast_sail", "Mast + sail"),
    ("jib_main", "Jib + main (with mast)"),
    ("plate", "Flat plate"),
    ("cylinder", "Cylinder"),
    ("naca0012", "NACA 0012 (wing sail)"),
    ("naca2412", "NACA 2412 (cambered)"),
]
SAIL, MAST_SAIL, JIB_MAIN, PLATE, CYLINDER, NACA0012, NACA2412 = range(len(SHAPES))
SAIL_SHAPES = (SAIL, MAST_SAIL, JIB_MAIN)
TELLTALE_LENGTH = 0.07    # telltale ribbon length, fraction of the chord
TELLTALE_SEGMENTS = 6
TELLTALES = [  # (sail: 1 main / 2 jib, position along its chord,
               #  side: +1 leeward (red) / -1 windward (green) / 0 leech telltale on the trailing edge (yellow))
    (1, 0.15, 1), (1, 0.15, -1), (1, 0.5, 1), (1, 0.5, -1), (1, 1.0, 0),
    (2, 0.15, 1), (2, 0.15, -1), (2, 0.5, 1), (2, 0.5, -1), (2, 1.0, 0),
]
LEE_SAMPLES = 40          # points along the lee side of the main used for 'leeward flow attached'
VIEWS = ["Speed", "Vorticity", "Pressure (Cp)", "Smoke only"]
FORCE_AXES = ["Lift / drag", "Drive / side"]   # how forces are split: wind axes, or boat axes (whole sail)
ELEMENT_NAMES = {1: "Main", 2: "Jib", 3: "Mast"}

# D2Q9 lattice: rest, 4 axis directions, 4 diagonals
E = ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))
W = (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)
OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)

HELP_LINES = [
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
    "  Space  pause    i  help    Esc  quit",
]


class State:
    """Everything the user can change."""

    def __init__(self):
        self.shape = SAIL
        self.aoa = 12.0          # deg, between wind and (main) chord line
        self.camber = 0.10       # sail depth, fraction of chord
        self.draft = 0.45        # position of max depth, fraction of chord from the luff
        self.mast = 0.05         # mast diameter / chord
        self.jib_gap = 0.05      # slot width between jib leech and main / main chord
        self.jib_overlap = 0.10  # jib leech behind the main luff / main chord
        self.jib_angle = -15.0   # jib chord angle relative to the main chord, deg (best L/D at main AoA 12)
        self.wind = 8.0          # m/s
        self.width = 3.0         # m, main chord ("sail width")
        self.height = 9.0        # m, sail height (luff)
        self.heading = 30.0      # deg, boat centreline to the wind (to the right); drawing and drive/side only
        self.view = 0
        self.axes = 0            # index into FORCE_AXES
        self.tracers = True
        self.arrows = True
        self.telltales = True
        self.boat = True
        self.polar = False
        self.paused = False
        self.help = False
        self.reset_flow = False
        self.clear_polar = False

    def geometry(self):
        return (self.shape, round(self.aoa, 3), round(self.camber, 4), round(self.draft, 4),
                round(self.mast, 4), round(self.jib_gap, 4), round(self.jib_overlap, 4), round(self.jib_angle, 3))

    def config_label(self):
        name = SHAPES[self.shape][1]
        if self.shape in (SAIL, MAST_SAIL, JIB_MAIN):
            name += f" {self.camber * 100:.0f}%/{self.draft * 100:.0f}%"
        if self.shape in (MAST_SAIL, JIB_MAIN):
            name += f" mast {self.mast * 100:.0f}%"
        if self.shape == JIB_MAIN:
            name += f" gap {self.jib_gap * 100:.0f}% ov {self.jib_overlap * 100:.0f}% jib {self.jib_angle:+.0f}deg"
        return name

    def ref_fraction(self):
        """Reference length (for coefficients and area) as a fraction of the main chord."""
        if self.shape == CYLINDER:
            return CYLINDER_DIAMETER
        if self.shape == JIB_MAIN:
            return 1.0 + JIB_CHORD
        return 1.0

    def clamp(self):
        self.shape = int(np.clip(self.shape, 0, len(SHAPES) - 1))
        self.aoa = float(np.clip(self.aoa, -30.0, 90.0))
        self.camber = float(np.clip(self.camber, 0.0, 0.20))
        self.draft = float(np.clip(self.draft, 0.25, 0.65))
        self.mast = float(np.clip(self.mast, 0.01, 0.12))
        self.jib_gap = float(np.clip(self.jib_gap, 0.02, 0.25))
        self.jib_overlap = float(np.clip(self.jib_overlap, -0.2, 0.4))
        self.jib_angle = float(np.clip(self.jib_angle, -20.0, 15.0))
        self.wind = float(np.clip(self.wind, 1.0, 20.0))
        self.width = float(np.clip(self.width, 0.5, 10.0))
        self.height = float(np.clip(self.height, 1.0, 40.0))
        self.heading = float(np.clip(self.heading, 0.0, 180.0))
        self.view = int(self.view) % len(VIEWS)
        self.axes = int(self.axes) % len(FORCE_AXES)


# --------------------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------------------
@ti.func
def feq_k(k: ti.template(), rho, u):
    eu = E[k][0] * u[0] + E[k][1] * u[1]
    return W[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * u.dot(u))


@ti.func
def feq_vec(rho, u):
    f = ti.Vector([0.0] * 9)
    for k in ti.static(range(9)):
        f[k] = feq_k(k, rho, u)
    return f


@ti.func
def moments(f):
    """Density and velocity of a population vector, packed as (rho, ux, uy)."""
    rho = f.sum()
    u = ti.Vector([0.0, 0.0])
    for k in ti.static(range(9)):
        u += f[k] * ti.Vector([E[k][0], E[k][1]])
    return tm.vec3(rho, u[0] / rho, u[1] / rho)


@ti.func
def rgb8(col):
    return ti.cast(tm.clamp(col, 0.0, 1.0) * 255.0 + 0.5, ti.u8)


@ti.func
def mean_line(xn, m, p):
    """NACA 4-digit mean line: returns (y/c, dy/dx) at x/c = xn for camber m at position p."""
    yc = 0.0
    dyc = 0.0
    if m > 0.0:
        if xn < p:
            yc = m / (p * p) * (2.0 * p * xn - xn * xn)
            dyc = 2.0 * m / (p * p) * (p - xn)
        else:
            yc = m / ((1.0 - p) ** 2) * (1.0 - 2.0 * p + 2.0 * p * xn - xn * xn)
            dyc = 2.0 * m / ((1.0 - p) ** 2) * (p - xn)
    return tm.vec2(yc, dyc)


@ti.func
def in_membrane(x, y, c, m, p, th):
    """Thin cambered membrane from (0, 0) to (c, 0) with thickness th (all in cells)."""
    inside = False
    if x >= -0.5 * th and x <= c + 0.5 * th:
        ml = mean_line(tm.clamp(x / c, 0.0, 1.0), m, p)
        inside = ti.abs(y - ml[0] * c) / ti.sqrt(1.0 + ml[1] * ml[1]) <= 0.5 * th
    return inside


@ti.func
def in_naca(x, y, c, m, p, t):
    inside = False
    if x >= 0.0 and x <= c:
        xn = x / c
        yt = 5.0 * t * c * (0.2969 * ti.sqrt(xn) - 0.1260 * xn - 0.3516 * xn ** 2 + 0.2843 * xn ** 3
                            - 0.1036 * xn ** 4)
        yc = mean_line(xn, m, p)[0] * c
        inside = ti.abs(y - yc) <= yt
    return inside


@ti.data_oriented
class Solver:
    def __init__(self, cells_per_chord, n_tracer_lines=24, re_sim=None):
        n = int(cells_per_chord)
        self.n = n
        self.re_sim = re_sim if re_sim else RE_PER_CELL2 * n * n
        self.nx = int(DOMAIN_CHORDS[0] * n)
        self.ny = int(DOMAIN_CHORDS[1] * n)
        # off-cell offsets: the half cell in y breaks the up/down symmetry; with the quarter cell in x a
        # straight plate at 0, 45, 90 or 135 deg never runs exactly through cell centres (there the last
        # bit of cos/sin would decide which lattice links it blocks)
        self.px = PIVOT_CHORDS[0] * n + 0.25
        self.py = PIVOT_CHORDS[1] * n + 0.5
        self.tau0 = 3.0 * U_LAT * n / self.re_sim + 0.5
        self.sponge = SPONGE_CHORDS * n
        self.side_sponge = SIDE_SPONGE_CHORDS * n
        # part of the domain that is shown: everything except the absorbing layers
        self.vx0, self.vy0 = self.side_sponge, self.side_sponge
        self.vx1, self.vy1 = self.nx - 1 - self.sponge, self.ny - 1 - self.side_sponge
        self.vw, self.vh = self.vx1 - self.vx0, self.vy1 - self.vy0
        self.rotated = WIND_FROM == "top"            # picture turned 90 deg clockwise: wind from the top
        self.steps_per_chord = n / U_LAT
        nx, ny = self.nx, self.ny

        self.f = [ti.Vector.field(9, ti.f32, shape=(nx, ny)) for _ in range(2)]
        self.cur = 0
        self.mask = ti.field(ti.i32, shape=(nx, ny))       # 0 fluid, 1 main/body, 2 jib, 3 mast
        self.mask_new = ti.field(ti.i32, shape=(nx, ny))
        self.link = ti.field(ti.i32, shape=(nx, ny))       # links crossing a membrane: bit k main/plate, 8+k jib (k=1..8)
        self.cs = ti.field(ti.f32, shape=(nx, ny))         # Smagorinsky constant per cell
        self.rho = ti.field(ti.f32, shape=(nx, ny))
        self.vel = ti.Vector.field(2, ti.f32, shape=(nx, ny))
        self.force = ti.Vector.field(2, ti.f32, shape=4)
        self.bbox = ti.field(ti.i32, shape=4)
        self.img = ti.Vector.field(3, ti.f32, shape=(nx, ny))
        self.lut = ti.Vector.field(3, ti.f32, shape=(3, 256))
        self.lut.from_numpy(build_luts())

        # smoke tracers: n_lines rake lines, 3 particles per cell so each line reads as a smoke streak
        self.n_lines = n_tracer_lines
        self.per_line = 3 * nx
        nt = self.n_lines * self.per_line
        self.tp = ti.Vector.field(2, ti.f32, shape=nt)
        self.tp_norm = ti.Vector.field(2, ti.f32, shape=nt)     # 0..1 across the flow picture
        self.tage = ti.field(ti.f32, shape=nt)
        self.rake_x = 0.5 * n
        self.max_age = 1.5 * nx / U_LAT
        self.tt_def = ti.Vector.field(3, ti.f32, shape=len(TELLTALES))
        self.tt_def.from_numpy(np.array(TELLTALES, dtype=np.float32))
        self.lee = ti.field(ti.f32, shape=LEE_SAMPLES)     # tangential speed along the lee side of the main
        self.arrow_vtx = ti.Vector.field(2, ti.f32, shape=18)
        self.arrow_col = ti.Vector.field(3, ti.f32, shape=18)
        self.hull_pix = ti.Vector.field(2, ti.f32, shape=2 * HULL_POINTS + 1)   # closed outline, picture pixels
        self.hull_box = ti.field(ti.i32, shape=4)                               # its pixel bounding box
        self.hull_sig = None

        self.bb = [0, 0, -1, -1]
        self.geo = (SAIL, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0)
        self.gamma = 0.0      # far-field circulation (lattice units), from the measured lift
        self.source = 0.0     # far-field source strength, from the measured drag
        self.mask.fill(0)
        self.reset_flow()
        self.reset_tracers()

    # ---------------------------------------------------------------- initialisation / geometry
    @ti.kernel
    def _init_flow(self, fa: ti.template(), fb: ti.template()):
        c = ti.static(float(self.n))
        for i, j in fa:
            # uniform flow plus an up-draft blob behind the body. Without it a symmetric body (plate across
            # the flow) keeps a symmetric wake for 20-40 chords before the vortex street starts
            r2 = ((i - self.px - c) ** 2 + (j - self.py) ** 2) / (0.5 * c) ** 2
            u = ti.Vector([U_LAT, START_KICK * U_LAT * ti.exp(-r2)])
            if self.mask[i, j] != 0:
                u = ti.Vector([0.0, 0.0])
            fa[i, j] = feq_vec(1.0, u)
            fb[i, j] = fa[i, j]
            self.rho[i, j] = 1.0
            self.vel[i, j] = u

    def reset_flow(self):
        self._init_flow(self.f[0], self.f[1])
        self.cur = 0

    @ti.func
    def _to_body(self, x, y, a):
        """Point in the body frame: luff of the main at (0, 0), chord along +x (all in cells)."""
        c = ti.static(float(self.n))
        dx = x - self.px
        dy = y - self.py
        ca, sa = ti.cos(a), ti.sin(a)
        return tm.vec2(dx * ca - dy * sa + 0.5 * c, dx * sa + dy * ca)

    @ti.func
    def _jib_frame(self, xb, yb, m, p, jgap, jover, jang):
        """Body-frame point in the jib's own frame (luff at 0, chord along +x). The jib leech sits jover
        behind the main luff, with a slot of width jgap to the lee side of the main."""
        c = ti.static(float(self.n))
        xo = tm.clamp(jover / c, 0.0, 1.0)
        te = tm.vec2(jover, mean_line(xo, m, p)[0] * c + jgap)
        cj, sj = ti.cos(jang), ti.sin(jang)
        le = te - JIB_CHORD * c * tm.vec2(cj, -sj)
        return tm.vec2((xb - le[0]) * cj - (yb - le[1]) * sj, (xb - le[0]) * sj + (yb - le[1]) * cj)

    @ti.func
    def _membranes(self, x, y, shape, a, m, p, jgap, jover, jang):
        """Position of the point along, and signed offset from, each zero-thickness membrane (cells):
        (x along main / plate, offset, x along jib, offset)."""
        c = ti.static(float(self.n))
        b = self._to_body(x, y, a)
        mm = m
        pp = p
        if shape == PLATE or shape == NACA0012:
            mm = 0.0
        elif shape == NACA2412:
            mm = 0.02
            pp = 0.4
        gm = b[1] - mean_line(tm.clamp(b[0] / c, 0.0, 1.0), mm, pp)[0] * c
        xj = -1.0
        gj = 1.0
        if shape == JIB_MAIN:
            jf = self._jib_frame(b[0], b[1], m, p, jgap, jover, jang)
            xj = jf[0]
            gj = jf[1] - mean_line(tm.clamp(xj / (JIB_CHORD * c), 0.0, 1.0), m, p)[0] * JIB_CHORD * c
        return tm.vec4(b[0], gm, xj, gj)

    @ti.func
    def _cross(self, g0, g1, x0, x1, length):
        """Does the straight segment between two points cross the membrane? g = signed offset from it,
        x = position along it."""
        hit = False
        if (g0 <= 0.0) != (g1 <= 0.0):
            xc = x0 + g0 / (g0 - g1) * (x1 - x0)
            if xc >= 0.0:
                if xc <= length:
                    hit = True
        return hit

    @ti.func
    def _body_at(self, x, y, shape, a, m, p, mast_r, jgap, jover, jang, th):
        """Which part covers the point (x, y), in cells: 0 air, 1 main sail / body, 2 jib, 3 mast.

        Sails and plate are zero-thickness membranes in the simulation (they block lattice links, see
        _build_links); only for drawing do they get the thickness th (th = 0: they are not solid)."""
        c = ti.static(float(self.n))
        bd = self._to_body(x, y, a)
        xb, yb = bd[0], bd[1]
        s = 0
        if shape == SAIL or shape == MAST_SAIL or shape == JIB_MAIN:
            if th > 0.0:
                if in_membrane(xb, yb, c, m, p, th):
                    s = 1
                if shape == JIB_MAIN:
                    jf = self._jib_frame(xb, yb, m, p, jgap, jover, jang)
                    if in_membrane(jf[0], jf[1], JIB_CHORD * c, m, p, th):
                        s = 2
            if shape != SAIL and xb * xb + yb * yb <= mast_r * mast_r:
                s = 3
        elif shape == PLATE:
            if th > 0.0:
                if in_membrane(xb, yb, c, 0.0, 0.5, th):
                    s = 1
        elif shape == CYLINDER:
            r = 0.5 * CYLINDER_DIAMETER * c
            if (xb - 0.5 * c) ** 2 + yb * yb <= r * r:
                s = 1
        elif shape == NACA0012:
            if in_naca(xb, yb, c, 0.0, 0.5, 0.12):
                s = 1
        elif shape == NACA2412:
            if in_naca(xb, yb, c, 0.02, 0.4, 0.12):
                s = 1
        return s

    @ti.kernel
    def _build_links(self, shape: int, a: float, m: float, p: float, jgap: float, jover: float, jang: float):
        """Mark the lattice links that cross a membrane (sail, jib, plate): populations bounce back there."""
        c = ti.static(float(self.n))
        for i, j in self.link:
            bits = 0
            # NACA profiles are solid cells, plus a membrane along their camber line: it seals the thin
            # trailing edge, which is thinner than a cell over its last few percent of chord
            if (shape == SAIL or shape == MAST_SAIL or shape == JIB_MAIN or shape == PLATE
                    or shape == NACA0012 or shape == NACA2412):
                q0 = self._membranes(float(i), float(j), shape, a, m, p, jgap, jover, jang)
                for k in ti.static(range(1, 9)):
                    q1 = self._membranes(float(i + E[k][0]), float(j + E[k][1]), shape, a, m, p, jgap, jover, jang)
                    if self._cross(q0[1], q1[1], q0[0], q1[0], c):
                        bits |= 1 << k
                    if shape == JIB_MAIN:
                        if self._cross(q0[3], q1[3], q0[2], q1[2], JIB_CHORD * c):
                            bits |= 1 << (8 + k)
            self.link[i, j] = bits

    @ti.kernel
    def _build_mask(self, shape: int, a: float, m: float, p: float, mast_r: float,
                    jgap: float, jover: float, jang: float):
        for i, j in self.mask_new:
            s = self._body_at(float(i), float(j), shape, a, m, p, mast_r, jgap, jover, jang, 0.0)
            if i == 0 or j == 0 or i == self.nx - 1 or j == self.ny - 1:
                s = 0
            self.mask_new[i, j] = s

    @ti.kernel
    def _apply_mask(self, fa: ti.template()):
        for i, j in self.mask:
            if self.mask[i, j] != 0 and self.mask_new[i, j] == 0:
                fa[i, j] = feq_vec(1.0, ti.Vector([0.0, 0.0]))   # uncovered cell: fluid at rest
            self.mask[i, j] = self.mask_new[i, j]

    @ti.kernel
    def _find_bbox(self):
        self.bbox[0] = self.nx
        self.bbox[1] = self.ny
        self.bbox[2] = -1
        self.bbox[3] = -1
        for i, j in self.mask:
            if (self.mask[i, j] != 0) | (self.link[i, j] != 0):
                ti.atomic_min(self.bbox[0], i)
                ti.atomic_min(self.bbox[1], j)
                ti.atomic_max(self.bbox[2], i)
                ti.atomic_max(self.bbox[3], j)

    @ti.kernel
    def _wall_damping(self, x0: int, y0: int, x1: int, y1: int, r: int):
        # Smagorinsky constant fades to zero at walls (van Driest style) so the wall friction stays low
        for i, j in self.cs:
            d2 = float(r * r)
            if i >= x0 - r and i <= x1 + r and j >= y0 - r and j <= y1 + r:
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        ii = tm.clamp(i + di, 0, self.nx - 1)
                        jj = tm.clamp(j + dj, 0, self.ny - 1)
                        if (self.mask[ii, jj] != 0) | (self.link[ii, jj] != 0):
                            d2 = ti.min(d2, float(di * di + dj * dj))
            self.cs[i, j] = SMAGORINSKY * (1.0 - ti.exp(-ti.sqrt(d2) / WALL_DAMPING)) ** 2

    def set_geometry(self, st):
        c = float(self.n)
        self.geo = (st.shape, math.radians(st.aoa), st.camber, st.draft, 0.5 * st.mast * c,
                    st.jib_gap * c, st.jib_overlap * c, math.radians(st.jib_angle))
        self._build_mask(*self.geo)
        self._apply_mask(self.f[self.cur])
        g = self.geo
        self._build_links(g[0], g[1], g[2], g[3], g[5], g[6], g[7])
        self._find_bbox()
        b = self.bbox.to_numpy()
        self.bb = [int(v) for v in b]
        self._wall_damping(int(b[0]), int(b[1]), int(b[2]), int(b[3]), int(4 * WALL_DAMPING))

    # ---------------------------------------------------------------- time stepping
    @ti.kernel
    def _step(self, fa: ti.template(), fb: ti.template(), gam: float, src: float):
        # fa holds post-collision populations: pull-stream them (bounce-back at walls), collide, store in fb
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            if self.mask[i, j] == 0:
                lk = self.link[i, j]
                lk = (lk | (lk >> 8)) & 511                 # links blocked by any membrane (bits 1..8)
                fin = ti.Vector([0.0] * 9)
                for k in ti.static(range(9)):
                    wall = self.mask[i - E[k][0], j - E[k][1]] != 0
                    if ti.static(k > 0):
                        if (lk >> OPP[k]) & 1:               # the link to the source crosses a sail
                            wall = True
                    if wall:
                        fin[k] = fa[i, j][OPP[k]]        # halfway bounce-back
                    else:
                        fin[k] = fa[i - E[k][0], j - E[k][1]][k]
                fb[i, j] = self._collide(fin, i, j, gam, src)

    @ti.func
    def _collide(self, fin, i, j, gam, src):
        mo = moments(fin)
        rho = mo[0]
        u = tm.vec2(mo[1], mo[2])
        feq = feq_vec(rho, u)
        df = fin - feq
        pxx = 0.0
        pyy = 0.0
        pxy = 0.0
        for k in ti.static(range(9)):
            pxx += E[k][0] * E[k][0] * df[k]
            pyy += E[k][1] * E[k][1] * df[k]
            pxy += E[k][0] * E[k][1] * df[k]
        # effective relaxation time: molecular + Smagorinsky sub-grid viscosity + outlet sponge
        tau = self.tau0 + 0.0
        cs = self.cs[i, j]
        if cs > 0.0:
            qn = ti.sqrt(pxx * pxx + pyy * pyy + 2.0 * pxy * pxy)
            tau = 0.5 * (tau + ti.sqrt(tau * tau + 18.0 * 1.41421356 * cs * cs * qn / rho))
        x0 = ti.static(self.nx - 1 - self.sponge)
        so = 0.0                                     # outlet sponge ramp 0..1
        if i > x0:
            so = (i - x0) / ti.static(self.sponge)
        tau += (TAU_SPONGE - tau) * so * so
        # KBC entropic collision (Karlin, Boesch, Chikatamarla 2014): the shear part ds relaxes with 1/tau,
        # the higher-order part dh with a rate chosen to keep the entropy balance -> stable as tau -> 0.5
        beta = 0.5 / tau
        nn = pxx - pyy
        ds = ti.Vector([0.0, 0.25 * nn, -0.25 * nn, 0.25 * nn, -0.25 * nn,
                        0.25 * pxy, -0.25 * pxy, 0.25 * pxy, -0.25 * pxy])
        dh = df - ds
        sh = 0.0
        hh = 0.0
        for k in ti.static(range(9)):
            inv = 1.0 / ti.max(feq[k], 1e-12)
            sh += ds[k] * dh[k] * inv
            hh += dh[k] * dh[k] * inv
        gamma = 2.0
        if hh > 1e-14:
            gamma = 1.0 / beta - (2.0 - 1.0 / beta) * sh / hh
        fpost = fin - beta * (2.0 * ds + gamma * dh)
        # absorbing layers at inlet, outlet, top and bottom: relax towards the undisturbed stream so that
        # sound waves and vortices leave the domain instead of echoing back onto the sail. The outlet layer
        # relaxes only the pressure: pulling the wake back to free-stream speed 2.5 chords behind the body
        # pushed it like a fan and nearly doubled the drag of fully separated flow (plate across the flow)
        side = ti.static(self.side_sponge)
        ss = ti.max(0.0, ti.max((side - ti.min(j, self.ny - 1 - j)) / side, (side - i) / side))
        sig_out = SIGMA_SPONGE * so * so
        sig_side = SIGMA_SPONGE * ss * ss
        if sig_out + sig_side > 0.0:
            uf = self._far_velocity(i, j, gam, src)
            rf = 1.0 + 1.5 * (U_LAT * U_LAT - uf.dot(uf))        # Bernoulli
            if sig_out > sig_side:
                mo = moments(fpost)
                fpost += sig_out * (feq_vec(rf, tm.vec2(mo[1], mo[2])) - fpost)
            else:
                fpost += sig_side * (feq_vec(rf, uf) - fpost)
        return fpost

    @ti.kernel
    def _boundaries(self, fb: ti.template(), gam: float, src: float):
        nx, ny = self.nx, self.ny
        for i in range(1, nx - 1):
            # top and bottom: open far field. The cross-flow comes from the far field (the wake pushes air out
            # sideways as in open air, but the whole wake cannot drift sideways); the along-flow velocity and
            # the density come from the neighbouring cell. Closed (mirror) walls made the domain a wind tunnel
            # that has to carry the drag with a pressure drop, which inflated separated-flow forces.
            for jj in ti.static(range(2)):
                jb = 0 if jj == 0 else ny - 1
                jn = 1 if jj == 0 else ny - 2
                mn = moments(fb[i, jn])
                un = tm.vec2(mn[1], mn[2])
                ub = tm.vec2(un[0], self._far_velocity(i, jb, gam, src)[1])
                fb[i, jb] = feq_vec(mn[0], ub) + fb[i, jn] - feq_vec(mn[0], un)
        for j in range(ny):
            # inlet: uniform velocity, density extrapolated, non-equilibrium part copied (Guo et al.)
            m1 = moments(fb[1, j])
            u1 = tm.vec2(m1[1], m1[2])
            fb[0, j] = feq_vec(m1[0], self._far_velocity(0, j, gam, src)) + fb[1, j] - feq_vec(m1[0], u1)
            # outlet: pressure of the far field (Bernoulli), velocity extrapolated. A uniform pressure here
            # would ignore the vortex far field of a lifting body and add ~10 % lift in this short domain
            m2 = moments(fb[nx - 2, j])
            u2 = tm.vec2(m2[1], m2[2])
            uf = self._far_velocity(nx - 1, j, gam, src)
            rf = 1.0 + 1.5 * (U_LAT * U_LAT - uf.dot(uf))
            fb[nx - 1, j] = feq_vec(rf, u2) + fb[nx - 2, j] - feq_vec(m2[0], u2)

    @ti.func
    def _far_velocity(self, i, j, gam, src):
        # far field of a lifting body: free stream + point vortex (lift) + point source (drag / wake)
        dx = i - self.px
        dy = j - self.py
        r2 = ti.max(dx * dx + dy * dy, 0.25 * self.n * self.n)
        k = 1.0 / (2.0 * math.pi * r2)
        return ti.Vector([U_LAT + k * (-gam * dy + src * dx), k * (gam * dx + src * dy)])

    def set_far_field(self, cl, cd, ref_fraction):
        """Circulation and source strength from the (averaged) force coefficients (Kutta-Joukowski).

        The inlet and the absorbing layers then carry the flow a lifting body induces far away, so a
        small domain behaves like an unbounded one instead of a narrow wind tunnel."""
        L = ref_fraction * self.n
        cl = float(np.clip(cl, -4.0, 4.0)) if np.isfinite(cl) else 0.0
        cd = float(np.clip(cd, 0.0, 4.0)) if np.isfinite(cd) else 0.0
        self.gamma = -0.5 * cl * U_LAT * L
        self.source = 0.5 * cd * U_LAT * L

    def advance(self, steps):
        for _ in range(int(steps)):
            a, b = self.f[self.cur], self.f[1 - self.cur]
            self._step(a, b, self.gamma, self.source)
            self._boundaries(b, self.gamma, self.source)
            self.cur = 1 - self.cur

    # ---------------------------------------------------------------- diagnostics
    @ti.kernel
    def _forces(self, fa: ti.template(), x0: int, y0: int, x1: int, y1: int):
        for e in range(4):
            self.force[e] = ti.Vector([0.0, 0.0])
        for i, j in ti.ndrange((x0, x1), (y0, y1)):   # only the cells around the body
            if self.mask[i, j] == 0:
                lk = self.link[i, j]
                for k in ti.static(range(1, 9)):
                    s = self.mask[i + E[k][0], j + E[k][1]]
                    if s == 0:
                        if (lk >> k) & 1:
                            s = 1
                        elif (lk >> (8 + k)) & 1:
                            s = 2
                    if s != 0:
                        # population heading into the wall bounces back: momentum 2 f e_k. The rest-state part
                        # (w_k) is removed, so parts touching each other (mast/main) do not pick up the static
                        # pressure on their hidden contact faces; the total force is unchanged.
                        self.force[s] += 2.0 * (fa[i, j][k] - W[k]) * ti.Vector([E[k][0], E[k][1]])

    def coefficients(self, ref_fraction):
        """Per element (index 1..3) and total (index 0): [Cd, Cl] of the 2D section."""
        b = self.bb
        if b[2] < b[0]:
            return np.zeros((4, 2))
        self._forces(self.f[self.cur], max(1, b[0] - 1), max(1, b[1] - 1),
                     min(self.nx - 1, b[2] + 2), min(self.ny - 1, b[3] + 2))
        f = self.force.to_numpy().astype(np.float64)
        f[0] = f[1:].sum(axis=0)
        q = 0.5 * U_LAT * U_LAT * ref_fraction * self.n
        return f / q

    @ti.kernel
    def _macro(self, fa: ti.template()):
        for i, j in self.rho:
            if self.mask[i, j] == 0:
                mo = moments(fa[i, j])
                self.rho[i, j] = mo[0]
                self.vel[i, j] = tm.vec2(mo[1], mo[2])
            else:
                # inside a body: velocity 0, pressure of the neighbouring air (smooth colours at the edge)
                r = 0.0
                cnt = 0
                for k in ti.static(range(1, 5)):
                    ii = tm.clamp(i + E[k][0], 0, self.nx - 1)
                    jj = tm.clamp(j + E[k][1], 0, self.ny - 1)
                    if self.mask[ii, jj] == 0:
                        r += moments(fa[ii, jj])[0]
                        cnt += 1
                self.rho[i, j] = 1.0
                if cnt > 0:
                    self.rho[i, j] = r / cnt
                self.vel[i, j] = ti.Vector([0.0, 0.0])

    def update_macro(self):
        self._macro(self.f[self.cur])

    # ---------------------------------------------------------------- rendering
    @ti.func
    def _lut(self, which, t):
        idx = ti.cast(tm.clamp(t, 0.0, 1.0) * 255.0, ti.i32)
        return self.lut[which, idx]

    @ti.kernel
    def render(self, view: int):
        c = ti.static(float(self.n))
        for i, j in self.img:
            col = tm.vec3(0.08, 0.09, 0.11)            # smoke-only view: dark background
            if view == 0:
                col = self._lut(0, self.vel[i, j].norm() / (2.0 * U_LAT))
            elif view == 1:
                ip = ti.min(i + 1, self.nx - 1)
                im = ti.max(i - 1, 0)
                jp = ti.min(j + 1, self.ny - 1)
                jm = ti.max(j - 1, 0)
                w = 0.5 * ((self.vel[ip, j][1] - self.vel[im, j][1]) - (self.vel[i, jp][0] - self.vel[i, jm][0]))
                col = self._lut(1, 0.5 + 0.5 * w * c / (U_LAT * VORTICITY_RANGE))
            elif view == 2:
                cp = (self.rho[i, j] - 1.0) / 3.0 / (0.5 * U_LAT * U_LAT)
                t = 0.5
                if cp < 0.0:
                    t = 0.5 - 0.5 * cp / CP_RANGE[0]
                else:
                    t = 0.5 + 0.5 * cp / CP_RANGE[1]
                col = self._lut(2, t)
            self.img[i, j] = col

    @ti.func
    def _img_bilinear(self, x, y):
        x = tm.clamp(x, 0.0, self.nx - 1.001)
        y = tm.clamp(y, 0.0, self.ny - 1.001)
        i = ti.cast(x, ti.i32)
        j = ti.cast(y, ti.i32)
        fx = x - i
        fy = y - j
        return ((1 - fx) * (1 - fy) * self.img[i, j] + fx * (1 - fy) * self.img[i + 1, j]
                + (1 - fx) * fy * self.img[i, j + 1] + fx * fy * self.img[i + 1, j + 1])

    @ti.func
    def _body_color(self, s, view):
        col = tm.vec3(0.94, 0.94, 0.94)                 # (main) sail / body
        if s == 2:
            col = tm.vec3(0.76, 0.85, 0.98)             # jib
        elif s == 3:
            col = tm.vec3(0.58, 0.58, 0.60)             # mast
        if view == 2:
            col *= 0.3                                  # dark bodies on the light pressure view
        return col

    @ti.func
    def _cell_to_pix(self, x, y, ox, oy, fw, fh):
        """Position in cells -> pixel in the flow picture (which may be turned so the wind comes from the top)."""
        u = (x - self.vx0) / self.vw                   # along the wind, 0..1
        v = (y - self.vy0) / self.vh                   # across the wind, 0..1
        pix = tm.vec2(ox + u * fw, oy + v * fh)
        if ti.static(self.rotated):
            pix = tm.vec2(ox + v * fw, oy + (1.0 - u) * fh)
        return pix

    def leading_edge(self, st):
        """Front of the object in cells: luff of the main / mast centre, or the upstream point of the cylinder."""
        c = float(self.n)
        if st.shape == CYLINDER:
            return np.array([self.px - 0.5 * CYLINDER_DIAMETER * c, self.py])
        a = math.radians(st.aoa)
        return np.array([self.px - 0.5 * c * math.cos(a), self.py + 0.5 * c * math.sin(a)])

    def hull_outline(self, st):
        """Dinghy outline in cells, as a closed polygon (bow, starboard side, transom, port side, bow).
        Drawing only: the hull is not part of the flow (on a real boat it sits below the sail section).
        It turns about the leading edge of the object, pointing st.heading to the right of the wind."""
        b = math.radians(st.heading)
        fwd = np.array([-math.cos(b), math.sin(b)])           # towards the bow (the wind blows along +x)
        across = np.array([math.sin(b), math.cos(b)])         # towards starboard
        length = HULL_LENGTH * self.n
        t = np.linspace(0.0, 1.0, HULL_POINTS)                # from the bow to the transom
        w = np.where(t < HULL_WIDEST, 1.0 - (1.0 - np.minimum(t / HULL_WIDEST, 1.0)) ** 2.5,
                     1.0 - (1.0 - HULL_TRANSOM) * ((t - HULL_WIDEST) / (1.0 - HULL_WIDEST)) ** 2)
        w = 0.5 * HULL_BEAM * length * w
        mid = self.leading_edge(st) + ((HULL_MAST - t) * length)[:, None] * fwd
        stb = mid + w[:, None] * across
        port = mid - w[:, None] * across
        return np.concatenate([stb, port[::-1], stb[:1]])

    def cells_to_pix(self, pts, fw, fh):
        """numpy twin of _cell_to_pix, relative to the corner of the flow picture."""
        u = (pts[:, 0] - self.vx0) / self.vw
        v = (pts[:, 1] - self.vy0) / self.vh
        if self.rotated:
            return np.stack([v * fw, (1.0 - u) * fh], axis=1)
        return np.stack([u * fw, v * fh], axis=1)

    def set_hull(self, st, fw, fh, r):
        sig = (st.shape, st.aoa, st.heading, fw, fh, r)
        if sig == self.hull_sig:
            return
        self.hull_sig = sig
        pix = self.cells_to_pix(self.hull_outline(st), fw, fh)
        lo = np.floor(pix.min(axis=0) - r - 3.0)
        hi = np.ceil(pix.max(axis=0) + r + 3.0)
        box = [max(lo[0], 0), max(lo[1], 0), min(hi[0], fw - 1), min(hi[1], fh - 1)]
        self.hull_pix.from_numpy(pix.astype(np.float32))
        self.hull_box.from_numpy(np.array(box, dtype=np.int32))

    @ti.func
    def _hull_blend(self, col, q, r, view):
        """Anti-aliased dinghy outline (half width r pixels, dark rim) over colour col at picture pixel q."""
        d = 1e9
        for k in range(2 * HULL_POINTS):
            a = self.hull_pix[k]
            ab = self.hull_pix[k + 1] - a
            t = tm.clamp((q - a).dot(ab) / ti.max(ab.dot(ab), 1e-6), 0.0, 1.0)
            d = ti.min(d, (q - a - t * ab).norm())
        core = tm.vec3(0.98, 0.80, 0.42)                       # light wood
        if view == 2:
            core = tm.vec3(0.32, 0.20, 0.07)                   # dark on the light pressure view
        else:
            col = tm.mix(col, tm.vec3(0.03, 0.03, 0.03), 0.7 * tm.clamp(r + 1.5 - d, 0.0, 1.0))
        return tm.mix(col, core, tm.clamp(r + 0.5 - d, 0.0, 1.0))

    @ti.kernel
    def compose(self, dst: ti.template(), ox: int, oy: int, fw: int, fh: int, view: int, hull: int, hull_r: float,
                shape: int, a: float, m: float, p: float, mast_r: float, jgap: float, jover: float, jang: float):
        """Draw the flow into dst[ox:ox+fw, oy:oy+fh] at window resolution: colours interpolated between
        cells, the dinghy outline (hull != 0), and the bodies evaluated from their exact shape with 3x3
        sub-samples (smooth, anti-aliased edges)."""
        sx = self.vw / fw                   # cells per pixel (the same in both directions)
        if ti.static(self.rotated):
            sx = self.vw / fh
        x0 = self.bbox[0] - 3.0
        y0 = self.bbox[1] - 3.0
        x1 = self.bbox[2] + 3.0
        y1 = self.bbox[3] + 3.0
        th = ti.max(2.5 * sx, DRAW_THICKNESS * self.n)        # drawn sail thickness: >= 2.5 pixels
        for i, j in ti.ndrange(fw, fh):
            u = (i + 0.5) / fw                                 # along the wind
            v = (j + 0.5) / fh                                 # across the wind
            if ti.static(self.rotated):                        # wind from the top of the picture
                u = 1.0 - (j + 0.5) / fh
                v = (i + 0.5) / fw
            x = self.vx0 + u * self.vw
            y = self.vy0 + v * self.vh
            col = self._img_bilinear(x, y)
            if hull != 0:
                if (i >= self.hull_box[0] and j >= self.hull_box[1] and i <= self.hull_box[2]
                        and j <= self.hull_box[3]):
                    col = self._hull_blend(col, tm.vec2(i + 0.5, j + 0.5), hull_r, view)
            if x > x0 and x < x1 and y > y0 and y < y1:
                cover = 0.0
                sid = 0
                for su, sv in ti.static(ti.ndrange(3, 3)):
                    s = self._body_at(x + (su - 1) * sx / 3.0, y + (sv - 1) * sx / 3.0,
                                      shape, a, m, p, mast_r, jgap, jover, jang, th)
                    if s != 0:
                        cover += 1.0 / 9.0
                        sid = s
                if sid != 0:
                    col = col * (1.0 - cover) + self._body_color(sid, view) * cover
            dst[ox + i, oy + j] = rgb8(col)

    @ti.kernel
    def splat_tracers(self, dst: ti.template(), ox: int, oy: int, fw: int, fh: int, col: tm.vec3):
        # smoke particles as 2x2 pixel dots, painted straight into the frame (much faster than circles)
        c8 = rgb8(col)
        for p in self.tp:
            q = self.tp_norm[p]
            pix = tm.vec2(q[0] * fw, q[1] * fh)
            if ti.static(self.rotated):
                pix = tm.vec2(q[1] * fw, (1.0 - q[0]) * fh)
            x = ox + ti.cast(ti.floor(pix[0]), ti.i32)
            y = oy + ti.cast(ti.floor(pix[1]), ti.i32)
            for di, dj in ti.static(ti.ndrange(2, 2)):
                if x + di >= ox and x + di < ox + fw and y + dj >= oy and y + dj < oy + fh:
                    dst[x + di, y + dj] = c8

    @ti.kernel
    def draw_line(self, dst: ti.template(), x0: float, y0: float, x1: float, y1: float, col: tm.vec3):
        W, H = dst.shape
        n = ti.cast(ti.max(ti.abs(x1 - x0) * W, ti.abs(y1 - y0) * H), ti.i32) + 1
        for k in range(n + 1):
            t = k / n
            x = ti.cast((x0 + t * (x1 - x0)) * W, ti.i32)
            y = ti.cast((y0 + t * (y1 - y0)) * H, ti.i32)
            for di, dj in ti.ndrange((-1, 2), (-1, 2)):          # 3 px wide
                if x + di >= 0 and x + di < W and y + dj >= 0 and y + dj < H:
                    dst[x + di, y + dj] = rgb8(col)

    @ti.kernel
    def blit(self, dst: ti.template(), src: ti.template(), x0: int, y0: int):
        for i, j in src:
            dst[x0 + i, y0 + j] = src[i, j]

    # ---------------------------------------------------------------- telltales and lee-side probe
    @ti.func
    def _sail_frame(self, xn, sail, a, m, p, jgap, jover, jang):
        """World position and unit tangent (towards the leech) at fraction xn along a sail (1 main, 2 jib)."""
        c = ti.static(float(self.n))
        ml = mean_line(xn, m, p)
        L = c
        if sail == 2:
            L = JIB_CHORD * c
        q = tm.vec2(xn * L, ml[0] * L)                  # in the sail's own frame
        t = tm.vec2(1.0, ml[1]).normalized()
        if sail == 2:                                   # jib frame -> main body frame
            xo = tm.clamp(jover / c, 0.0, 1.0)
            te = tm.vec2(jover, mean_line(xo, m, p)[0] * c + jgap)
            cj, sj = ti.cos(jang), ti.sin(jang)
            le = te - JIB_CHORD * c * tm.vec2(cj, -sj)
            q = le + tm.vec2(q[0] * cj + q[1] * sj, -q[0] * sj + q[1] * cj)
            t = tm.vec2(t[0] * cj + t[1] * sj, -t[0] * sj + t[1] * cj)
        ca, sa = ti.cos(a), ti.sin(a)                   # main body frame -> world
        dx = q[0] - 0.5 * c
        return tm.vec4(self.px + dx * ca + q[1] * sa, self.py - dx * sa + q[1] * ca,
                       t[0] * ca + t[1] * sa, -t[0] * sa + t[1] * ca)

    @ti.kernel
    def leeward_profile(self, a: float, m: float, p: float):
        # flow speed along the lee side of the main, 2 cells off the cloth (negative = reversed = separated)
        for k in self.lee:
            f = self._sail_frame((k + 0.5) / LEE_SAMPLES, 1, a, m, p, 0.0, 0.0, 0.0)
            t = tm.vec2(f[2], f[3])
            u = self._sample_vel(tm.vec2(f[0], f[1]) + 2.0 * tm.vec2(-t[1], t[0]))
            self.lee[k] = u.dot(t) / U_LAT

    @ti.func
    def _paint_segment(self, dst: ti.template(), p0, p1, ox, oy, fw, fh, r, col):
        # thick line between two points given in cells, into the flow picture part of the frame
        a0 = self._cell_to_pix(p0[0], p0[1], ox, oy, fw, fh)
        a1 = self._cell_to_pix(p1[0], p1[1], ox, oy, fw, fh)
        n = ti.cast(ti.max(ti.abs(a1[0] - a0[0]), ti.abs(a1[1] - a0[1])), ti.i32) + 1
        for s in range(n + 1):
            q = a0 + (a1 - a0) * (s / n)
            for di, dj in ti.ndrange((-4, 5), (-4, 5)):
                if di * di + dj * dj <= r * r:
                    x = ti.cast(q[0], ti.i32) + di
                    y = ti.cast(q[1], ti.i32) + dj
                    if x >= ox and x < ox + fw and y >= oy and y < oy + fh:
                        dst[x, y] = col

    @ti.kernel
    def draw_telltales(self, dst: ti.template(), ox: int, oy: int, fw: int, fh: int, shape: int, a: float,
                       m: float, p: float, jgap: float, jover: float, jang: float, scale: float):
        """Telltale ribbons on both sides of the sails, and one on each leech. Each one follows the local air
        flow from where it is tied on, so it streams aft in attached flow and lifts, points forward or wanders
        where the flow has separated - like the yarns on a real sail. A leech telltale streams straight aft
        when the flow leaves the leech cleanly and curls behind the sail when the leech stalls."""
        c = ti.static(float(self.n))
        seg = TELLTALE_LENGTH * c / TELLTALE_SEGMENTS
        for layer in ti.static(range(2)):              # dark outline first, then the coloured ribbon
            for k in self.tt_def:
                sail = ti.cast(self.tt_def[k][0], ti.i32)
                side = self.tt_def[k][2]
                if sail == 1 or shape == JIB_MAIN:
                    f = self._sail_frame(self.tt_def[k][1], sail, a, m, p, jgap, jover, jang)
                    t = tm.vec2(f[2], f[3])
                    root = tm.vec2(f[0], f[1])
                    pos = root + 1.0 * side * tm.vec2(-t[1], t[0])  # tied on just off the cloth
                    if ti.abs(side) < 0.5:
                        pos = root + 1.0 * t                         # leech: just behind the trailing edge
                    col = rgb8(tm.vec3(0.05, 0.05, 0.05))
                    r = 0.8 * scale + 1.2
                    if ti.static(layer == 1):
                        r = 0.8 * scale
                        col = rgb8(tm.vec3(1.0, 0.25, 0.2))
                        if side < -0.5:
                            col = rgb8(tm.vec3(0.25, 0.95, 0.35))
                        elif side < 0.5:
                            col = rgb8(tm.vec3(1.0, 0.86, 0.1))
                    d = t
                    alive = True
                    for _ in range(TELLTALE_SEGMENTS):
                        if alive:
                            u = self._sample_vel(pos)
                            if u.norm() > 1e-6:
                                d = u.normalized()
                            q = pos + seg * d
                            # the ribbon cannot pass through the cloth or the mast
                            q0 = self._membranes(pos[0], pos[1], shape, a, m, p, jgap, jover, jang)
                            q1 = self._membranes(q[0], q[1], shape, a, m, p, jgap, jover, jang)
                            if self._cross(q0[1], q1[1], q0[0], q1[0], c):
                                alive = False
                            if shape == JIB_MAIN:
                                if self._cross(q0[3], q1[3], q0[2], q1[2], JIB_CHORD * c):
                                    alive = False
                            qi = ti.cast(tm.clamp(q[0], 0.0, self.nx - 1.0), ti.i32)
                            qj = ti.cast(tm.clamp(q[1], 0.0, self.ny - 1.0), ti.i32)
                            if self.mask[qi, qj] != 0:
                                alive = False
                            if alive:
                                self._paint_segment(dst, pos, q, ox, oy, fw, fh, r, col)
                                pos = q

    # ---------------------------------------------------------------- smoke tracers
    @ti.kernel
    def reset_tracers(self):
        for p in self.tp:
            line = p // self.per_line
            k = p % self.per_line
            y0 = self.py + (line + 0.5 - 0.5 * self.n_lines) * (2.2 * self.n / self.n_lines)
            x = self.rake_x + (self.nx - 2 - self.rake_x) * (k + ti.random()) / self.per_line
            self.tp[p] = tm.vec2(x, y0)
            self.tage[p] = ti.random() * self.max_age

    @ti.func
    def _sample_vel(self, pos):
        x = tm.clamp(pos[0], 0.0, self.nx - 1.001)
        y = tm.clamp(pos[1], 0.0, self.ny - 1.001)
        i = ti.cast(x, ti.i32)
        j = ti.cast(y, ti.i32)
        fx = x - i
        fy = y - j
        return ((1 - fx) * (1 - fy) * self.vel[i, j] + fx * (1 - fy) * self.vel[i + 1, j]
                + (1 - fx) * fy * self.vel[i, j + 1] + fx * fy * self.vel[i + 1, j + 1])

    @ti.kernel
    def _advect(self, dt: float, nsub: int, shape: int, a: float, m: float, dp: float,
                jgap: float, jover: float, jang: float):
        c = ti.static(float(self.n))
        for p in self.tp:
            pos = self.tp[p]
            for _ in range(nsub):
                u1 = self._sample_vel(pos)
                u2 = self._sample_vel(pos + 0.5 * dt * u1)
                new = pos + dt * u2
                # smoke cannot pass through a sail. (The shape test must stay inside the loop: computed once
                # before it, Taichi 1.7 miscompiled this block and pinned particles on the chord line.)
                q0 = self._membranes(pos[0], pos[1], shape, a, m, dp, jgap, jover, jang)
                q1 = self._membranes(new[0], new[1], shape, a, m, dp, jgap, jover, jang)
                blocked = False
                if (shape == SAIL or shape == MAST_SAIL or shape == JIB_MAIN or shape == PLATE
                        or shape == NACA0012 or shape == NACA2412):
                    if self._cross(q0[1], q1[1], q0[0], q1[0], c):
                        blocked = True
                if shape == JIB_MAIN:
                    if self._cross(q0[3], q1[3], q0[2], q1[2], JIB_CHORD * c):
                        blocked = True
                if not blocked:
                    pos = new
            self.tage[p] += dt * nsub
            line = p // self.per_line
            y0 = self.py + (line + 0.5 - 0.5 * self.n_lines) * (2.2 * self.n / self.n_lines)
            i = ti.cast(tm.clamp(pos[0], 0.0, self.nx - 1.0), ti.i32)
            j = ti.cast(tm.clamp(pos[1], 0.0, self.ny - 1.0), ti.i32)
            if pos[0] >= self.nx - 2:
                pos = tm.vec2(self.rake_x + (pos[0] - (self.nx - 2)), y0)   # keeps the spacing of the streak
                self.tage[p] = 0.0
            elif pos[1] < 1 or pos[1] > self.ny - 2 or self.mask[i, j] != 0 or self.tage[p] > self.max_age:
                pos = tm.vec2(self.rake_x, y0)
                self.tage[p] = 0.0
            self.tp[p] = pos
            self.tp_norm[p] = tm.vec2((pos[0] - self.vx0) / self.vw, (pos[1] - self.vy0) / self.vh)

    def advect_tracers(self, steps):
        if steps <= 0:
            return
        nsub = max(1, int(math.ceil(steps * U_LAT * 3.0)))   # at most ~1/3 cell per sub-step at 1 U
        g = self.geo
        self._advect(steps / nsub, nsub, g[0], g[1], g[2], g[3], g[5], g[6], g[7])


def build_luts():
    """Colour tables: 0 speed (plasma), 1 vorticity (yellow-orange-black-green-cyan), 2 Cp (blue-white-red)."""
    t = np.linspace(0.0, 1.0, 256)
    lut = np.zeros((3, 256, 3), dtype=np.float32)
    try:
        from matplotlib import colormaps
        lut[0] = colormaps["plasma"](t)[:, :3]
        lut[2] = colormaps["RdBu_r"](t)[:, :3]
    except Exception:
        lut[0] = np.stack([t, 0.2 + 0.6 * t * (1 - t), 1 - t], axis=1)
        lut[2] = np.stack([np.interp(t, [0, .5, 1], [.1, 1, .8]), np.interp(t, [0, .5, 1], [.3, 1, .1]),
                           np.interp(t, [0, .5, 1], [.8, 1, .1])], axis=1)
    stops = [0.0, 0.25, 0.5, 0.75, 1.0]
    cols = np.array([[1.0, 1.0, 0.0], [0.953, 0.490, 0.016], [0.0, 0.0, 0.0], [0.176, 0.976, 0.529], [0.0, 1.0, 1.0]])
    lut[1] = np.stack([np.interp(t, stops, cols[:, k]) for k in range(3)], axis=1)
    return lut


# --------------------------------------------------------------------------------------------------
# Force averaging, results and polar plot
# --------------------------------------------------------------------------------------------------
class Averager:
    """Instantaneous forces fluctuate with vortex shedding: settle, then take the long-time mean."""

    def __init__(self, steps_per_chord):
        self.spc = steps_per_chord
        self.reset()

    def reset(self):
        self.age = 0.0
        self.ema = None
        self.sum = None
        self.weight = 0.0

    def add(self, c, steps):
        if steps <= 0:
            if self.ema is None:
                self.ema = c.copy()
            return
        self.age += steps
        a = 1.0 - math.exp(-steps / (0.5 * self.spc))
        self.ema = c.copy() if self.ema is None else self.ema + a * (c - self.ema)
        if self.age > SETTLE_CHORDS * self.spc:
            self.sum = c * steps if self.sum is None else self.sum + c * steps
            self.weight += steps

    @property
    def value(self):
        return self.sum / self.weight if self.weight > 0 else self.ema

    @property
    def chords(self):
        return self.age / self.spc

    @property
    def ready(self):
        return self.weight >= AVERAGE_CHORDS * self.spc

    def status(self):
        if self.age <= SETTLE_CHORDS * self.spc:
            return f"flow settling: {self.chords:.1f} of {SETTLE_CHORDS:.0f} chords"
        avg = self.weight / self.spc
        return f"averaged over {avg:.1f} chords" + ("" if self.ready else " (settling)")


def results(st, coeffs, sim):
    """All numbers shown to the user, from time-averaged 2D coefficients."""
    cd, cl = coeffs[0]
    ref_m = st.ref_fraction() * st.width
    area = ref_m * st.height
    ar = st.height / ref_m
    cdi = cl * cl / (math.pi * SPAN_EFFICIENCY * ar)
    q = 0.5 * RHO_AIR * st.wind ** 2
    re = st.wind * (ref_m if st.shape == CYLINDER else st.width) / NU_AIR
    r = dict(cd=cd, cl=cl, ld=cl / cd if abs(cd) > 1e-9 else float("nan"), cdi=cdi, cd3=cd + cdi, ar=ar,
             ld3=cl / (cd + cdi) if abs(cd + cdi) > 1e-9 else float("nan"), q=q, area=area, re=re,
             lift=cl * q * area, drag2=cd * q * area, dragi=cdi * q * area, elements=[])
    r["drag3"] = r["drag2"] + r["dragi"]
    r["total"] = math.hypot(r["lift"], r["drag3"])
    # whole-sail force in boat axes, for a boat heading b to the (apparent) wind
    b = math.radians(st.heading)
    r["drive"] = r["lift"] * math.sin(b) - r["drag3"] * math.cos(b)
    r["side"] = r["lift"] * math.cos(b) + r["drag3"] * math.sin(b)
    for e in (1, 2, 3):
        if st.shape in (MAST_SAIL, JIB_MAIN) and (e != 2 or st.shape == JIB_MAIN):
            r["elements"].append((ELEMENT_NAMES[e], coeffs[e][1], coeffs[e][0]))
    return r


def fmt_num(v):
    """Short label for a scale: 118, 16, 2.5, 2."""
    if abs(v) >= 9.95:
        return f"{v:.0f}"
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def playback(st, sim, steps_per_second):
    """How the animation compares with reality: the simulated air moves steps_per_second * U_LAT / n chords
    per second on screen, the real wind st.wind / st.width chords per second."""
    if st.paused:
        return "paused"
    if steps_per_second <= 0.0:
        return "–"
    k = (st.wind / st.width) / (steps_per_second * U_LAT / sim.n)
    if k >= 1.1:
        return f"{k:.0f}x slow motion" if k >= 9.5 else f"{k:.1f}x slow motion"
    if k > 0.9:
        return "real time"
    return f"{1.0 / k:.1f}x faster than real"


def fmt_ld(cl, cd):
    return f"{cl / cd:.1f}" if cd > 0.005 else "–"      # drag ~0 only happens while the flow settles


def result_lines(st, r, avg, sim, info, lee=None):
    lines = [
        f"{SHAPES[st.shape][1]}, AoA {st.aoa:.1f}°",
        f"  {avg.status()}",
        f"2D section  Cl {r['cl']:.2f}  Cd {r['cd']:.3f}  L/D {fmt_ld(r['cl'], r['cd'])}",
    ]
    for name, cl, cd in r["elements"]:
        lines.append(f"  {name:4s}      Cl {cl:.2f}  Cd {cd:.3f}")
    if lee is not None:
        sep = lee < 0.0
        who = "Main leeward" if st.shape == JIB_MAIN else "Leeward"
        lines.append(f"{who} flow attached {100 * (1 - sep.mean()):.0f}% of chord")
        if sep.any():
            lines.append(f"  separated from {100 * (np.argmax(sep) + 0.5) / len(lee):.0f}% (full size: later)")
    lines.append(f"Whole sail  Cd {r['cd3']:.3f}  L/D {fmt_ld(r['cl'], r['cd3'])}  (AR {r['ar']:.1f})")
    if (st.boat or st.axes == 1) and st.shape != CYLINDER:
        boom = f"Boom {st.heading - st.aoa:.0f}° off centreline"
        if st.shape == JIB_MAIN:
            boom += f", jib {st.heading - st.aoa - st.jib_angle:.0f}°"
        lines.append(boom)
    if st.axes == 1:
        lines.append(f"Drive {r['drive']:.0f} N  Side {r['side']:.0f} N  ({r['area']:.1f} m²)")
        if r["drive"] > 0.0:
            lines.append(f"  side force = {r['side'] / r['drive']:.1f} x drive")
        elif r["drive"] < 0.0:
            lines.append("  no drive: " + ("too close to the wind" if st.heading < 60.0
                                           else "sail force points backwards"))
        else:
            lines.append("  –")
    else:
        lines.append(f"Lift {r['lift']:.0f} N  Drag {r['drag3']:.0f} N  ({r['area']:.1f} m²)")
        lines.append(f"  = {r['drag2']:.0f} N section + {r['dragi']:.0f} N induced")
    lines.append(f"Re {r['re']:.2g} real, {sim.re_sim:.2g} simulated")
    if st.shape == CYLINDER and r["re"] > 3e5:
        lines.append("  real cylinder: drag crisis > Re 3e5")
    if st.shape in (NACA0012, NACA2412):
        lines.append("Note: thick profiles suffer at low Re")
        lines.append("  at full size (Re ~1e6) L/D ~50-100")
        if sim.n < QUALITY_CELLS["medium"]:
            lines.append("  Low grid: nose under-resolved,")
            lines.append("  use quality Medium or higher")
    lines.append(info)
    return lines


class PolarPlot:
    """Separate matplotlib window: Cl, L/D and drag polar vs angle of attack, one curve per configuration."""

    def __init__(self):
        self.fig = None
        self.data = {}          # config label -> {aoa: (cl, cd)}
        self.dirty = False
        self.last_draw = 0.0
        self.last_flush = 0.0
        self.current = None

    def record(self, label, aoa, cl, cd):
        pts = self.data.setdefault(label, {})
        key = round(aoa * 2) / 2
        old = pts.get(key)
        pts[key] = (cl, cd)
        if old is None or abs(old[0] - cl) > 0.01 or abs(old[1] - cd) > 0.002:
            self.dirty = True

    def clear(self):
        self.data = {}
        self.dirty = True

    def open(self):
        if self.fig is not None:
            return True
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed - polar plot disabled")
            return False
        plt.ion()
        self.plt = plt
        self.fig, self.ax = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
        try:
            self.fig.canvas.manager.set_window_title("FLOWNIAC polar")
        except Exception:
            pass
        self.fig.show()
        self.dirty = True
        return True

    def close(self):
        if self.fig is not None:
            self.plt.close(self.fig)
            self.fig = None

    def update(self, ar, current):
        """current = (aoa, cl, cd) of the live (not yet converged) point. Returns False if the window was closed."""
        if self.fig is None:
            return True
        if not self.plt.fignum_exists(self.fig.number):
            self.fig = None
            return False
        now = time.time()
        if current != self.current and now - self.last_draw > 2.0:
            self.current = current
            self.dirty = True
        if self.dirty and now - self.last_draw > 0.5:
            self._draw(ar)
            self.dirty = False
            self.last_draw = now
        if now - self.last_flush > 0.1:
            self.fig.canvas.flush_events()
            self.last_flush = now
        return True

    def _draw(self, ar):
        a0, a1, a2 = self.ax
        for a in self.ax:
            a.clear()
            a.grid(True, alpha=0.3)
        k = 1.0 / (math.pi * SPAN_EFFICIENCY * ar)
        for i, (label, pts) in enumerate(sorted(self.data.items())):
            aoa = np.array(sorted(pts))
            cl = np.array([pts[a][0] for a in aoa])
            cd = np.array([pts[a][1] for a in aoa])
            col = f"C{i % 10}"
            a0.plot(aoa, cl, "o-", color=col, label=label)
            a1.plot(aoa, cl / cd, "o-", color=col)
            a1.plot(aoa, cl / (cd + k * cl * cl), "s--", color=col, alpha=0.6)
            a2.plot(cd, cl, "o-", color=col)
        if self.current is not None:
            aoa, cl, cd = self.current
            a0.plot([aoa], [cl], "kx", ms=9)
            a1.plot([aoa], [cl / cd], "kx", ms=9)
            a2.plot([cd], [cl], "kx", ms=9)
        a0.set(xlabel="AoA [deg]", ylabel="Cl (2D)", title="Lift coefficient")
        a1.set(xlabel="AoA [deg]", ylabel="L/D", title=f"L/D  (solid 2D, dashed whole sail AR {ar:.1f})")
        a2.set(xlabel="Cd (2D)", ylabel="Cl (2D)", title="Drag polar")
        if self.data:
            a0.legend(fontsize=7, loc="lower right")
        self.fig.canvas.draw_idle()


# --------------------------------------------------------------------------------------------------
# User interface: slider panel (ti.ui) and keyboard-only fallback (ti.GUI)
# --------------------------------------------------------------------------------------------------
class Keys:
    """Keyboard handling shared by both windows. Arrow keys auto-repeat while held."""

    STEP = {"aoa": 1.0, "camber": 0.01, "draft": 0.05, "wind": 1.0, "width": 0.25, "height": 1.0,
            "mast": 0.01, "jib_gap": 0.01, "jib_angle": 1.0, "jib_overlap": 0.05, "heading": 5.0}
    LETTER = {"d": "draft", "w": "wind", "c": "width", "h": "height", "m": "mast", "g": "jib_gap",
              "j": "jib_angle", "o": "jib_overlap", "b": "heading"}
    HELD = {"Up": ("aoa", 1), "Down": ("aoa", -1), "Right": ("camber", 1), "Left": ("camber", -1)}

    def __init__(self):
        self.held_since = {}
        self.last_repeat = {}

    def press(self, st, key, shift):
        if len(key) == 1 and key.isupper():
            key, shift = key.lower(), True
        if key in self.LETTER:
            name = self.LETTER[key]
            setattr(st, name, getattr(st, name) + (1 if shift else -1) * self.STEP[name])
        elif key in "1234567" and len(key) == 1:
            st.shape = int(key) - 1
        elif key == "v":
            st.view = (st.view + 1) % len(VIEWS)
        elif key == "t":
            st.tracers = not st.tracers
        elif key == "l":
            st.telltales = not st.telltales
        elif key == "f":
            st.arrows = not st.arrows
        elif key == "n":
            st.boat = not st.boat
        elif key == "x":
            st.axes = (st.axes + 1) % len(FORCE_AXES)
        elif key == "p":
            st.polar = not st.polar
        elif key == "r":
            st.reset_flow = True
        elif key == "BackSpace" or key == "Backspace":
            st.clear_polar = True
        elif key == " " or key == "Space":
            st.paused = not st.paused
        elif key == "i":
            st.help = not st.help
        st.clamp()

    def _step(self, st, key, fine):
        name, sign = self.HELD[key]
        setattr(st, name, getattr(st, name) + sign * self.STEP[name] * (0.2 if fine else 1.0))
        st.clamp()

    def press_held(self, st, key, fine):
        """First step of an arrow key at once, so a short tap between two frames still counts."""
        if key not in self.held_since:
            self.held_since[key] = self.last_repeat[key] = time.time()
            self._step(st, key, fine)

    def held(self, st, is_pressed, fine):
        """Called once per frame: arrow keys keep changing angle of attack and camber while held."""
        now = time.time()
        for key in self.HELD:
            if is_pressed(key):
                if key not in self.held_since:
                    self.press_held(st, key, fine)
                elif now - self.held_since[key] > 0.35 and now - self.last_repeat[key] > 0.06:
                    self.last_repeat[key] = now
                    self._step(st, key, fine)
            else:
                self.held_since.pop(key, None)


def _font(family, size, bold=False):
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        prop = font_manager.FontProperties(family=family, weight="bold" if bold else "normal")
        return ImageFont.truetype(font_manager.findfont(prop), size)
    except Exception:
        try:
            return ImageFont.load_default(size)
        except TypeError:
            return ImageFont.load_default()


class Panel:
    """Control column on the left of the window: selectors, sliders, check boxes, buttons and results.

    Drawn with Pillow because Taichi's own GUI cannot scale its font; clicks and drags are handled here."""

    BG, FG, DIM, WARN = (24, 26, 31), (232, 234, 238), (140, 146, 158), (255, 170, 60)
    TRACK, ACCENT, KNOB, BOX = (58, 62, 72), (70, 135, 215), (240, 240, 240), (44, 48, 56)
    W_UNITS = 290            # column width at UI scale 1
    H_UNITS = 662            # height of the tallest column content (jib + main with results) at UI scale 1

    # label, State attribute, min, max, step, display factor, format, shapes it applies to (None = all)
    SLIDERS = [
        ("Angle of attack", "aoa", -30.0, 90.0, 0.5, 1.0, "{:.1f}°", None),
        ("Camber", "camber", 0.0, 0.2, 0.005, 100.0, "{:.1f} %", (SAIL, MAST_SAIL, JIB_MAIN)),
        ("Draft position", "draft", 0.25, 0.65, 0.01, 100.0, "{:.0f} %", (SAIL, MAST_SAIL, JIB_MAIN)),
        ("Mast diameter", "mast", 0.01, 0.12, 0.005, 100.0, "{:.1f} %", (MAST_SAIL, JIB_MAIN)),
        ("Jib slot", "jib_gap", 0.02, 0.25, 0.005, 100.0, "{:.1f} %", (JIB_MAIN,)),
        ("Jib overlap", "jib_overlap", -0.2, 0.4, 0.01, 100.0, "{:.0f} %", (JIB_MAIN,)),
        ("Jib angle", "jib_angle", -20.0, 15.0, 0.5, 1.0, "{:+.1f}°", (JIB_MAIN,)),
        ("Wind", "wind", 1.0, 20.0, 0.5, 1.0, "{:.1f} m/s", None),
        ("Sail width", "width", 0.5, 10.0, 0.05, 1.0, "{:.2f} m", None),
        ("Sail height", "height", 1.0, 40.0, 0.5, 1.0, "{:.1f} m", None),
        ("Boat heading", "heading", 0.0, 180.0, 1.0, 1.0, "{:.0f}°", None),
    ]

    def __init__(self, width, height, scale):
        from PIL import Image, ImageDraw
        self.Image, self.ImageDraw = Image, ImageDraw
        self.w, self.h, self.scale = width, height, scale
        self.font = _font("DejaVu Sans", self.u(12))
        self.title = _font("DejaVu Sans", self.u(15), bold=True)
        self.mono = _font("DejaVu Sans Mono", self.u(11))
        self.small = _font("DejaVu Sans", self.u(10))
        self.lut = (build_luts() * 255.0 + 0.5).astype(np.uint8)     # same colours as the flow picture
        self.field = ti.Vector.field(3, ti.u8, shape=(width, height))
        self.hits = []           # (x0, y0, x1, y1, kind, data) of the clickable parts, from the last drawing
        self.active = None       # attribute of the slider being dragged
        self.sig = None
        self.lines = None
        self.t_text = 0.0

    def u(self, v):
        return int(round(v * self.scale))

    # ---------------------------------------------------------------- drawing
    def render(self, st, lines):
        """Redraw when a control changed, or every 0.25 s for the results."""
        now = time.time()
        sig = (st.shape, st.view, st.axes, st.tracers, st.arrows, st.telltales, st.boat, st.polar, st.paused,
               st.help, self.active, tuple(round(getattr(st, sl[1]), 5) for sl in self.SLIDERS))
        if sig == self.sig and (lines == self.lines or now - self.t_text < 0.4):
            return False
        self.sig, self.lines, self.t_text = sig, lines, now
        u = self.u
        img = self.Image.new("RGB", (self.w, self.h), self.BG)
        d = self.ImageDraw.Draw(img)
        self.hits = []
        pad, row = u(10), u(19)
        self.xt0, self.xt1, self.xv = u(112), self.w - u(66), self.w - pad   # slider track and value column
        y = pad
        d.text((pad, y), "FLOWNIAC", font=self.title, fill=self.FG)
        d.text((pad + d.textlength("FLOWNIAC", font=self.title) + u(8), y + u(4)), "sail aerodynamics",
               font=self.font, fill=self.DIM)
        y += u(26)
        y = self._select(d, y, row, "Shape", "shape", [name for _, name in SHAPES], st)
        for sl in self.SLIDERS:
            if sl[7] is not None and st.shape not in sl[7]:
                continue
            if sl[1] == "wind":
                y += u(8)
            y = self._slider(d, y, row, sl, st)
        y += u(8)
        y = self._select(d, y, row, "View", "view", VIEWS, st)
        y = self._colorbar(d, y, st)
        y = self._select(d, y, row, "Forces", "axes", FORCE_AXES, st)
        y = self._toggles(d, y, row, [("Smoke", "tracers"), ("Arrows", "arrows"), ("Telltales", "telltales")], st)
        y = self._toggles(d, y, row, [("Boat", "boat"), ("Polar plot", "polar")], st)
        y = self._buttons(d, y + u(4), row, [("Run" if st.paused else "Pause", "paused", "toggle"),
                                             ("Reset flow", "reset_flow", "button"),
                                             ("Clear polar", "clear_polar", "button")])
        y += u(8)
        d.line([(pad, y), (self.w - pad, y)], fill=self.TRACK, width=max(1, u(1)))
        y += u(8)
        d.text((pad, y), "Keyboard" if st.help else "Results", font=self.font, fill=self.DIM)
        y += u(19)
        for line in (HELP_LINES if st.help else lines):
            if y + u(14) > self.h - pad:
                break
            d.text((pad, y), line, font=self.mono, fill=self.WARN if line.startswith("!!") else self.FG)
            y += u(14)
        a = np.asarray(img, dtype=np.uint8)                      # rows top to bottom
        self.field.from_numpy(np.ascontiguousarray(a[::-1].transpose(1, 0, 2)))
        return True

    def _select(self, d, y, row, label, attr, names, st):
        u, cy = self.u, y + row // 2
        d.text((u(10), cy), label, font=self.font, fill=self.FG, anchor="lm")
        x0, x1 = self.xt0 - u(4), self.xv
        d.rounded_rectangle([x0, y + u(2), x1, y + row - u(2)], radius=u(4), fill=self.BOX)
        d.text((x0 + u(7), cy), "‹", font=self.font, fill=self.DIM, anchor="lm")
        d.text((x1 - u(7), cy), "›", font=self.font, fill=self.DIM, anchor="rm")
        name, font = names[getattr(st, attr)], self.font
        if d.textlength(name, font=font) > x1 - x0 - u(36):          # keep clear of the arrows
            font = _font("DejaVu Sans", int(self.u(12) * (x1 - x0 - u(36)) / d.textlength(name, font=font)))
        d.text(((x0 + x1) // 2, cy), name, font=font, fill=self.FG, anchor="mm")
        xm = (x0 + x1) // 2
        self.hits.append((x0, y, xm, y + row, "select", (attr, len(names), -1)))
        self.hits.append((xm, y, x1, y + row, "select", (attr, len(names), 1)))
        return y + row + u(3)

    def _colorbar(self, d, y, st):
        """Colour scale of the flow picture in real units, so the wind speed shows: m/s, 1/s or Pa."""
        u = self.u
        x0, x1 = self.xt0 - u(4), self.xv
        yb0, yb1 = y + u(2), y + u(10)
        d.text((u(10), (yb0 + yb1) // 2), "Scale", font=self.font, fill=self.DIM, anchor="lm")
        if st.view == 3:
            d.text((x0, (yb0 + yb1) // 2), "smoke only: no colours", font=self.small, fill=self.DIM, anchor="lm")
            return y + u(27)
        lut = self.lut[st.view]
        for x in range(x0, x1):
            d.line([(x, yb0), (x, yb1)], fill=tuple(int(c) for c in lut[(x - x0) * 255 // max(x1 - x0 - 1, 1)]))
        q = 0.5 * RHO_AIR * st.wind ** 2
        if st.view == 0:                                    # speed: 0 .. 2x the wind
            ticks = (fmt_num(0.0), f"{fmt_num(st.wind)} = wind", f"{fmt_num(2.0 * st.wind)} m/s")
        elif st.view == 1:                                  # vorticity: +-VORTICITY_RANGE wind / chord
            w = VORTICITY_RANGE * st.wind / st.width
            ticks = (fmt_num(-w), "0", f"+{fmt_num(w)} 1/s")
        else:                                               # pressure: Cp -3 .. +1 times the dynamic pressure
            ticks = (fmt_num(CP_RANGE[0] * q), "0 = ambient", f"+{fmt_num(CP_RANGE[1] * q)} Pa")
        ty = yb1 + u(2)
        d.text((x0, ty), ticks[0], font=self.small, fill=self.FG, anchor="la")
        d.text(((x0 + x1) // 2, ty), ticks[1], font=self.small, fill=self.FG, anchor="ma")
        d.text((x1, ty), ticks[2], font=self.small, fill=self.FG, anchor="ra")
        return y + u(27)

    def _slider(self, d, y, row, sl, st):
        label, attr, lo, hi, step, factor, fmt, _ = sl
        u, cy = self.u, y + row // 2
        v = getattr(st, attr)
        d.text((u(10), cy), label, font=self.font, fill=self.FG, anchor="lm")
        x0, x1, r = self.xt0, self.xt1, u(7)
        d.rounded_rectangle([x0, cy - u(3), x1, cy + u(3)], radius=u(3), fill=self.TRACK)
        kx = x0 + (x1 - x0) * min(max((v - lo) / (hi - lo), 0.0), 1.0)
        if kx - x0 > u(6):
            d.rounded_rectangle([x0, cy - u(3), kx, cy + u(3)], radius=u(3), fill=self.ACCENT)
        d.ellipse([kx - r, cy - r, kx + r, cy + r], fill=self.KNOB,
                  outline=self.ACCENT if self.active == attr else None, width=u(2))
        d.text((self.xv, cy), fmt.format(v * factor), font=self.font, fill=self.FG, anchor="rm")
        self.hits.append((x0 - r, y, x1 + r, y + row, "slider", sl))
        return y + row

    def _toggles(self, d, y, row, items, st, cols=3):
        u, cy = self.u, y + row // 2
        part = (self.w - 2 * u(10)) // cols
        for k, (label, attr) in enumerate(items):
            x0 = u(10) + k * part
            on = getattr(st, attr)
            d.rounded_rectangle([x0, cy - u(7), x0 + u(14), cy + u(7)], radius=u(3),
                                fill=self.ACCENT if on else self.BOX)
            if on:
                d.line([(x0 + u(3), cy), (x0 + u(6), cy + u(4)), (x0 + u(11), cy - u(4))],
                       fill=self.FG, width=max(1, u(2)))
            d.text((x0 + u(20), cy), label, font=self.font, fill=self.FG, anchor="lm")
            self.hits.append((x0, y, x0 + part, y + row, "toggle", attr))
        return y + row

    def _buttons(self, d, y, row, items):
        u = self.u
        part = (self.w - 2 * u(10)) // len(items)
        for k, (label, attr, kind) in enumerate(items):
            x0 = u(10) + k * part
            x1 = x0 + part - u(6)
            d.rounded_rectangle([x0, y, x1, y + row + u(2)], radius=u(4), fill=self.BOX)
            d.text(((x0 + x1) // 2, y + (row + u(2)) // 2), label, font=self.font, fill=self.FG, anchor="mm")
            self.hits.append((x0, y, x1, y + row + u(2), kind, attr))
        return y + row + u(2)

    # ---------------------------------------------------------------- mouse
    def mouse(self, st, x, y, pressed, down, released):
        """x, y in column pixels (y from the top)."""
        if pressed:
            for x0, y0, x1, y1, kind, data in self.hits:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    if kind == "slider":
                        self.active = data[1]
                        self._drag(st, data, x)
                    elif kind == "select":
                        attr, n, step = data
                        setattr(st, attr, (getattr(st, attr) + step) % n)
                    elif kind == "toggle":
                        setattr(st, data, not getattr(st, data))
                    elif kind == "button":
                        setattr(st, data, True)
                    break
        elif down and self.active is not None:
            for x0, y0, x1, y1, kind, data in self.hits:
                if kind == "slider" and data[1] == self.active:
                    self._drag(st, data, x)
        if released or not down:
            self.active = None
        st.clamp()

    def _drag(self, st, sl, x):
        _, attr, lo, hi, step, _, _, _ = sl
        t = min(max((x - self.xt0) / (self.xt1 - self.xt0), 0.0), 1.0)
        setattr(st, attr, round((lo + t * (hi - lo)) / step) * step)


class Layout:
    """Window = control column (left) + flow picture (right), in physical screen pixels."""

    def __init__(self, nx, ny, ui_scale, window_w=None):
        sw, sh = screen_size()
        max_h = int(0.85 * sh)
        self.scale = min(ui_scale, max_h / Panel.H_UNITS)
        if self.scale < ui_scale - 1e-6:
            print(f"UI scale reduced to {self.scale:.2f} so the control column fits the screen")
        self.col_w = int(Panel.W_UNITS * self.scale)
        self.fw = max(int(window_w or WINDOW_FRACTION * sw) - self.col_w, 320)
        self.fh = int(self.fw * ny / nx)
        if self.fh > max_h:
            self.fh = max_h
            self.fw = int(self.fh * nx / ny)
        self.W = self.col_w + self.fw
        self.H = max(self.fh, int(Panel.H_UNITS * self.scale))
        self.ox, self.oy = self.col_w, (self.H - self.fh) // 2


class Front:
    """Shared by both windows: the composed frame (control column + flow picture) and mouse handling."""

    def __init__(self, sim, layout):
        self.sim, self.L = sim, layout
        self.frame = ti.Vector.field(3, ti.u8, shape=(layout.W, layout.H))
        self.frame.from_numpy(np.full((layout.W, layout.H, 3), 12, dtype=np.uint8))
        self.panel = Panel(layout.col_w, layout.H, layout.scale)
        self.keys = Keys()
        self.offscreen = False

    def _mouse(self, st, pos, down, pressed, released):
        x = pos[0] * self.L.W
        y = (1.0 - pos[1]) * self.L.H                 # column coordinates: y from the top
        if pressed and x >= self.L.col_w:
            return                                     # clicks in the flow picture do nothing (yet)
        self.panel.mouse(st, x, y, pressed, down, released)

    def _compose(self, st, lines):
        sim, L = self.sim, self.L
        if self.panel.render(st, lines):
            sim.blit(self.frame, self.panel.field, 0, 0)          # the column part of the frame persists
        hull_r = 0.6 * L.scale + 0.4                              # half width of the dinghy outline, pixels
        if st.boat:
            sim.set_hull(st, L.fw, L.fh, hull_r)
        sim.compose(self.frame, L.ox, L.oy, L.fw, L.fh, st.view, int(st.boat), hull_r, *sim.geo)
        if st.tracers:
            sim.splat_tracers(self.frame, L.ox, L.oy, L.fw, L.fh, tracer_color(st))
        if st.telltales and st.shape in SAIL_SHAPES:
            g = sim.geo
            sim.draw_telltales(self.frame, L.ox, L.oy, L.fw, L.fh, g[0], g[1], g[2], g[3], g[5], g[6], g[7],
                               L.scale)

    def _to_window(self, arrows):
        v, c = arrows
        u, w = v[:, 0].copy(), v[:, 1].copy()          # along / across the wind, 0..1
        if self.sim.rotated:
            u, w = w, 1.0 - u
        v = np.stack([(self.L.ox + u * self.L.fw) / self.L.W, (self.L.oy + w * self.L.fh) / self.L.H], axis=1)
        return v.astype(np.float32), c


class GGUIFront(Front):
    """Main window (Taichi GGUI, needs Vulkan); smoke and force arrows are drawn as smooth vector shapes."""

    def __init__(self, sim, layout, offscreen=False):
        super().__init__(sim, layout)
        self.window = ti.ui.Window("FLOWNIAC - sail aerodynamics", (layout.W, layout.H), vsync=False,
                                   show_window=not offscreen)
        self.canvas = self.window.get_canvas()
        self.offscreen = offscreen

    @property
    def running(self):
        return self.window.running

    def poll(self, st):
        if self.offscreen:
            return
        w = self.window
        shift = w.is_pressed(ti.ui.SHIFT)
        pressed = released = False
        for e in w.get_events(ti.ui.PRESS):        # GGUI events only carry .key; the filter gives the type
            if e.key == ti.ui.LMB:
                pressed = True
            elif e.key == ti.ui.ESCAPE:
                w.running = False
            elif e.key in Keys.HELD:
                self.keys.press_held(st, e.key, shift)
            else:
                self.keys.press(st, e.key, shift)
        for e in w.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                released = True
        self.keys.held(st, w.is_pressed, shift)
        self._mouse(st, w.get_cursor_pos(), w.is_pressed(ti.ui.LMB), pressed, released)

    def draw(self, st, lines, arrows):
        sim, L = self.sim, self.L
        self._compose(st, lines)
        self.canvas.set_image(self.frame)
        if st.arrows and arrows is not None:
            v, c = self._to_window(arrows)
            sim.arrow_vtx.from_numpy(v)
            sim.arrow_col.from_numpy(c)
            self.canvas.lines(sim.arrow_vtx, width=0.003, per_vertex_color=sim.arrow_col)

    def show(self):
        self.window.show()

    def save(self, path):
        self.window.save_image(path)


class ClassicFront(Front):
    """Fallback window (ti.GUI, works without Vulkan). The frame is composed completely on the GPU."""

    def __init__(self, sim, layout):
        super().__init__(sim, layout)
        self.gui = ti.GUI("FLOWNIAC - sail aerodynamics", res=(layout.W, layout.H), fast_gui=True)

    @property
    def running(self):
        return self.gui.running

    def poll(self, st):
        g = self.gui
        shift = g.is_pressed(ti.GUI.SHIFT)
        pressed = released = False
        for e in g.get_events():
            if e.key == ti.GUI.LMB:
                pressed |= e.type == ti.GUI.PRESS
                released |= e.type == ti.GUI.RELEASE
            elif e.type == ti.GUI.PRESS:
                if e.key in (ti.GUI.ESCAPE, ti.GUI.EXIT):
                    g.running = False
                elif e.key in Keys.HELD:
                    self.keys.press_held(st, e.key, shift)
                else:
                    self.keys.press(st, e.key, shift)
        self.keys.held(st, g.is_pressed, shift)
        self._mouse(st, g.get_cursor_pos(), g.is_pressed(ti.GUI.LMB), pressed, released)

    def draw(self, st, lines, arrows):
        sim, L = self.sim, self.L
        self._compose(st, lines)
        if st.arrows and arrows is not None:
            v, c = self._to_window(arrows)
            for k in range(0, len(v), 2):
                sim.draw_line(self.frame, v[k][0], v[k][1], v[k + 1][0], v[k + 1][1], tm.vec3(*c[k]))
        self.gui.set_image(self.frame)

    def show(self):
        self.gui.show()

    def save(self, path):
        ti.tools.imwrite(self.frame.to_numpy(), path)


def tracer_color(st):
    return (0.12, 0.12, 0.12) if st.view == 2 else (0.92, 0.92, 0.92)   # dark smoke on the light Cp view


def arrow_geometry(sim, st, r):
    """Line segments (normalized picture coords) for the total force (white) and its two parts:
    lift (green) and drag (red) of the 2D section, or drive (blue) and side force (orange) of the whole sail."""
    o = np.array([(sim.px - sim.vx0) / sim.vw, (sim.py - sim.vy0) / sim.vh])
    scale = 0.45 * st.ref_fraction() * sim.n       # arrow length per unit coefficient, in cells
    cl = r["cl"]
    if st.axes == 1:
        cd = r["cd3"]                               # boat axes: whole sail, including induced drag
        b = math.radians(st.heading)
        fwd, across = np.array([-math.cos(b), math.sin(b)]), np.array([math.sin(b), math.cos(b)])
        drive = fwd * (cl * math.sin(b) - cd * math.cos(b))
        side = across * (cl * math.cos(b) + cd * math.sin(b))
        parts = ((tuple(drive), (0.3, 0.75, 1.0)), (tuple(side), (1.0, 0.6, 0.15)))
    else:
        cd = r["cd"]
        parts = (((0.0, cl), (0.2, 0.9, 0.3)), ((cd, 0.0), (1.0, 0.3, 0.2)))
    verts, cols = [], []
    for vec, col in parts + (((cd, cl), (1.0, 1.0, 1.0)),):
        d = np.array([vec[0] * scale / sim.vw, vec[1] * scale / sim.vh])
        tip = o + d
        n = math.hypot(d[0] * sim.vw, d[1] * sim.vh)
        if n < 1e-6:
            head = [tip, tip, tip, tip]
        else:
            ux, uy = d[0] * sim.vw / n, d[1] * sim.vh / n
            h = min(0.25 * n, 0.08 * sim.n)
            left = tip + np.array([(-ux * 0.9 - uy * 0.5) * h / sim.vw, (-uy * 0.9 + ux * 0.5) * h / sim.vh])
            right = tip + np.array([(-ux * 0.9 + uy * 0.5) * h / sim.vw, (-uy * 0.9 - ux * 0.5) * h / sim.vh])
            head = [tip, left, tip, right]
        verts += [o, tip] + head
        cols += [col] * 6
    return np.array(verts, dtype=np.float32), np.array(cols, dtype=np.float32)


# --------------------------------------------------------------------------------------------------
# Hardware: quality presets and auto-selection
# --------------------------------------------------------------------------------------------------
def init_taichi(arch):
    archs = {"gpu": ti.gpu, "cuda": ti.cuda, "vulkan": ti.vulkan, "metal": ti.metal, "cpu": ti.cpu}
    ti.init(arch=archs[arch], default_fp=ti.f32, offline_cache=True, log_level=ti.WARN)


def seconds_per_chord(n, mlups):
    cells = DOMAIN_CHORDS[0] * DOMAIN_CHORDS[1] * n * n
    return (n / U_LAT) * cells / (mlups * 1e6) * 1.2     # +20 % for rendering and UI


def auto_quality(arch):
    """Measure the solver speed on the 'medium' grid and pick the finest preset that is fast enough."""
    print("Measuring GPU speed for quality selection ...")
    sim = Solver(QUALITY_CELLS["medium"], n_tracer_lines=1)
    sim.set_geometry(State())
    sim.advance(10)
    ti.sync()
    t0 = time.perf_counter()
    steps = 0
    while time.perf_counter() - t0 < 1.0:
        sim.advance(20)
        ti.sync()
        steps += 20
    mlups = sim.nx * sim.ny * steps / (time.perf_counter() - t0) / 1e6
    del sim
    ti.reset()
    init_taichi(arch)
    choice = "low"
    for name in ("low", "medium", "high", "ultra"):
        if seconds_per_chord(QUALITY_CELLS[name], mlups) <= AUTO_SECONDS_PER_CHORD:
            choice = name
    print(f"  {mlups:.0f} million cell updates/s -> quality '{choice}'"
          f" ({seconds_per_chord(QUALITY_CELLS[choice], mlups):.1f} s per chord of flow)")
    return choice


def screen_size():
    """Screen size in physical pixels (Taichi windows are not enlarged by the Windows display scaling)."""
    if sys.platform == "win32":
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass                                   # already set (Taichi does it too)
            u32 = ctypes.windll.user32
            return u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)
        except Exception:
            pass
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return size
    except Exception:
        return 1920, 1080


# --------------------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Educational 2D airflow around sails (lattice Boltzmann, Taichi).")
    ap.add_argument("--quality", default="auto", choices=["auto"] + list(QUALITY_CELLS))
    ap.add_argument("--arch", default="gpu", choices=["gpu", "cuda", "vulkan", "metal", "cpu"])
    ap.add_argument("--ui", default="auto", choices=["auto", "ggui", "gui"], help="ggui = slider panel, gui = keyboard")
    ap.add_argument("--shape", default="sail", choices=[k for k, _ in SHAPES])
    ap.add_argument("--aoa", type=float, default=12.0, help="angle of attack [deg]")
    ap.add_argument("--camber", type=float, default=10.0, help="sail camber [%% of chord]")
    ap.add_argument("--draft", type=float, default=45.0, help="position of max camber [%% of chord]")
    ap.add_argument("--wind", type=float, default=8.0, help="wind speed [m/s]")
    ap.add_argument("--width", type=float, default=3.0, help="sail width (chord) [m]")
    ap.add_argument("--height", type=float, default=9.0, help="sail height [m]")
    ap.add_argument("--heading", type=float, default=30.0, help="boat heading to the wind [deg]")
    ap.add_argument("--forces", default="lift", choices=["lift", "drive"], help="lift/drag or drive/side force")
    ap.add_argument("--no-boat", action="store_true", help="hide the dinghy outline")
    ap.add_argument("--view", default="speed", choices=["speed", "vorticity", "pressure", "smoke"])
    ap.add_argument("--polar", action="store_true", help="open the polar plot window at start")
    ap.add_argument("--snapshot", metavar="PNG", help="run off-screen, save an image and print the results")
    ap.add_argument("--chords", type=float, default=8.0, help="flow passes (in chords) before --snapshot")
    ap.add_argument("--visible", action="store_true", help="show the window while running --snapshot")
    ap.add_argument("--ui-scale", type=float, default=UI_SCALE, help="size of the control column (default 2)")
    ap.add_argument("--window", type=int, help="window width in pixels (default: 3/4 of the screen)")
    ap.add_argument("--wind-from", default=WIND_FROM, choices=["top", "left"],
                    help="wind direction in the picture (default: from the top, as sailors draw it)")
    return ap.parse_args()


def main():
    args = parse_args()
    init_taichi(args.arch)
    quality = auto_quality(args.arch) if args.quality == "auto" else args.quality
    n = QUALITY_CELLS[quality]

    st = State()
    st.shape = [k for k, _ in SHAPES].index(args.shape)
    st.aoa, st.camber, st.draft = args.aoa, args.camber / 100.0, args.draft / 100.0
    st.wind, st.width, st.height = args.wind, args.width, args.height
    st.view = ["speed", "vorticity", "pressure", "smoke"].index(args.view)
    st.heading, st.axes, st.boat = args.heading, ["lift", "drive"].index(args.forces), not args.no_boat
    st.polar = args.polar
    st.clamp()

    sim = Solver(n)
    sim.rotated = args.wind_from == "top"
    if sim.rotated:
        layout = Layout(sim.vh, sim.vw, args.ui_scale, args.window)     # portrait flow picture
    else:
        layout = Layout(sim.vw, sim.vh, args.ui_scale, args.window)
    front = None
    if args.ui in ("auto", "ggui"):
        try:
            front = GGUIFront(sim, layout, offscreen=args.snapshot is not None and not args.visible)
        except Exception as ex:
            if args.ui == "ggui":
                raise
            print(f"GGUI window unavailable ({ex}); using the basic window.")
    if front is None:
        front = ClassicFront(sim, layout)

    print(f"Grid {sim.nx} x {sim.ny} ({n} cells per chord), simulated Re {sim.re_sim:.3g}, "
          f"tau0 = {sim.tau0:.5f}, quality '{quality}'")
    avg = Averager(sim.steps_per_chord)
    polar = PolarPlot()
    geometry = None
    steps_per_frame = 10
    fps = 0.0
    sps = 0.0                                       # solver steps per second, smoothed
    total_steps = 0
    t_last = time.perf_counter()
    t_step = 0.0
    overhead = 0.02
    unstable_until = 0.0
    lee = None                                      # time-averaged lee-side flow of the main sail

    while front.running:
        front.poll(st)
        if st.geometry() != geometry:
            geometry = st.geometry()
            sim.set_geometry(st)
            avg.reset()
            lee = None
        if st.reset_flow:
            st.reset_flow = False
            sim.reset_flow()
            sim.reset_tracers()
            avg.reset()
        if st.clear_polar:
            st.clear_polar = False
            polar.clear()

        steps = 0 if st.paused else steps_per_frame
        t_solver = time.perf_counter()
        sim.advance(steps)
        total_steps += steps
        coeffs = sim.coefficients(st.ref_fraction())       # reads back from the GPU, so it also waits for it
        t_solver = time.perf_counter() - t_solver
        sim.update_macro()
        if not np.isfinite(coeffs).all():
            # the flow blew up (extreme case for this grid): restart it instead of showing garbage
            print("Flow became unstable and was reset.")
            sim.reset_flow()
            sim.reset_tracers()
            avg.reset()
            unstable_until = time.time() + 4.0
            coeffs = np.zeros((4, 2))
        avg.add(coeffs, steps)
        if st.tracers:
            sim.advect_tracers(steps)
        sim.render(st.view)
        if st.shape in SAIL_SHAPES:
            sim.leeward_profile(sim.geo[1], sim.geo[2], sim.geo[3])
            prof = sim.lee.to_numpy()
            k = 1.0 - math.exp(-steps / (0.5 * sim.steps_per_chord)) if steps > 0 else 0.0
            lee = prof if lee is None else lee + k * (prof - lee)
        else:
            lee = None

        c = avg.value
        sim.set_far_field(c[0][1], c[0][0], st.ref_fraction())
        r = results(st, c, sim)
        if avg.ready:
            polar.record(st.config_label(), st.aoa, r["cl"], r["cd"])
        if st.polar and polar.open():
            if not polar.update(r["ar"], (st.aoa, r["cl"], r["cd"])):
                st.polar = False
        elif not st.polar:
            polar.close()

        info = f"{quality}, {fps:.0f} fps, {playback(st, sim, sps)}"
        lines = result_lines(st, r, avg, sim, info, lee)
        if time.time() < unstable_until:
            lines.insert(0, "!! flow unstable: restarted")
        front.draw(st, lines, arrow_geometry(sim, st, r))

        if args.snapshot and total_steps >= args.chords * sim.steps_per_chord:
            front.save(args.snapshot)
            print("\n".join(lines).replace("²", "2").replace("°", " deg").replace("–", "-"))
            break
        if args.snapshot and front.offscreen:
            steps_per_frame = 200
            continue
        front.show()

        # steps per frame: aim for TARGET_FPS, but never let drawing eat more than 1/3 of the time
        now = time.perf_counter()
        dt = max(now - t_last, 1e-4)
        t_last = now
        fps = 0.9 * fps + 0.1 / dt if fps > 0 else 1.0 / dt
        if steps > 0:
            sps = 0.9 * sps + 0.1 * steps / dt if sps > 0 else steps / dt
        if steps > 0:
            # smoothed timings; single slow frames (plot redraw, geometry change) are clipped
            t_step = 0.8 * t_step + 0.2 * (t_solver / steps) if t_step else t_solver / steps
            overhead = 0.8 * overhead + 0.2 * min(max(dt - t_solver, 0.0), 0.1)
            want = max((1.0 / TARGET_FPS - overhead) / t_step, SOLVER_SHARE / (1.0 - SOLVER_SHARE) * overhead / t_step)
            want = min(want, 0.2 / t_step)       # keep at least ~5 frames per second
            steps_per_frame = int(np.clip(0.7 * steps_per_frame + 0.3 * want, 2, 2000))


if __name__ == "__main__":
    main()
