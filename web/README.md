# FLOWNIAC web version

The browser version of FLOWNIAC and the one that is developed further. It started as a port of
`Flowniac.py` (the desktop version, no longer developed): the same lattice-Boltzmann solver, written as
WebGPU compute shaders (WGSL), with the same controls, keys, results and polar plot. Since then it got a
Quality selector (switches the grid live), a Slow motion slider (keys `s`/`S`), continuous smoke
streaks and a Boundary layer selector: "model size" (the simulation at the grid's Reynolds number, as in
`Flowniac.py`) or "full size (approx.)" (law of the wall and near-wall mixing, drag estimated; see the
header of `js/config.js`). Plain HTML, CSS and JavaScript modules: no build step, no dependencies.

## Run it

The page has to be served over HTTP (browsers do not load JavaScript modules from `file://`):

```bash
python -m http.server 8000 --directory web
```

Then open <http://localhost:8000>. Any static web host (e.g. GitHub Pages) works the same way.

Needs a browser with WebGPU: current Chrome or Edge, Safari 26 or newer, Firefox 141 or newer on Windows.

The URL sets the start values, like the command line of `Flowniac.py`, for example
`index.html?quality=high&shape=jib_main&aoa=15&wind-from=left&forces=drive&boundary=full&polar`:
`quality` (auto, low, medium, high, ultra; auto measures the GPU and picks the finest grid that still
moves the flow one chord in 3 s), `shape`, `aoa`, `camber`, `draft` (%), `wind`, `width`,
`height`, `heading`, `forces` (lift, drive), `boundary` (model, full), `view` (speed, vorticity,
pressure, smoke), `no-boat`,
`polar`, `wind-from` (top, left). The control column scales with the browser zoom (Ctrl +/-).

## Files

| File | Ported from Flowniac.py |
|---|---|
| `js/config.js` | Settings section, `State` |
| `js/shaders.js` | the Taichi kernels and `@ti.func`s of `Solver` |
| `js/solver.js` | `Solver` (buffers, geometry, stepping, forces, hull outline) |
| `js/results.js` | `Averager`, `results`, `result_lines`, `playback`, `arrow_geometry` |
| `js/ui.js` | `Keys`, `Panel` |
| `js/polar.js` | `PolarPlot` |
| `js/main.js` | `main`, `Front`, `auto_quality` |
| `js/luts.js` | generated from `build_luts` by `tools/make_luts.py` |

## Solver parity with Flowniac.py

The flow solver (constants, kernels, order of operations) still matches `Flowniac.py` with the "model
size" boundary layer; the smoke, the quality selection, the playback speed and the "full size (approx.)"
boundary layer are web-only. The parity test checks the solver (model size, forces of a single step):

```bash
python web/tools/parity.py
```

This runs a few validation cases with `Flowniac.py` and writes `parity_reference.json`. Then open
<http://localhost:8000/parity.html>: it runs the same cases on the WebGPU solver and compares the forces.
Early values agree to about 0.01 %; after a few chords the vortex shedding amplifies rounding
differences, so the long averages agree only within the usual scatter of a few percent.

`python -m http.server` sends no cache headers, so after editing a file the browser may keep running the
old module: reload with Ctrl+Shift+R.
