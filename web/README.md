# FLOWNIAC web version

The browser version of `Flowniac.py`: the same lattice-Boltzmann solver, written as WebGPU compute
shaders (WGSL), with the same controls, keys, results and polar plot. Plain HTML, CSS and JavaScript
modules: no build step, no dependencies.

## Run it

The page has to be served over HTTP (browsers do not load JavaScript modules from `file://`):

```bash
python -m http.server 8000 --directory web
```

Then open <http://localhost:8000>. Any static web host (e.g. GitHub Pages) works the same way.

Needs a browser with WebGPU: current Chrome or Edge, Safari 26 or newer, Firefox 141 or newer on Windows.

The URL takes the same options as the command line of `Flowniac.py`, for example
`index.html?quality=high&shape=jib_main&aoa=15&wind-from=left&forces=drive&polar`:
`quality` (auto, low, medium, high, ultra), `shape`, `aoa`, `camber`, `draft` (%), `wind`, `width`,
`height`, `heading`, `forces` (lift, drive), `view` (speed, vorticity, pressure, smoke), `no-boat`,
`polar`, `wind-from` (top, left). The control column scales with the browser zoom (Ctrl +/-).

## Files

| File | Mirrors in Flowniac.py |
|---|---|
| `js/config.js` | Settings section, `State` |
| `js/shaders.js` | the Taichi kernels and `@ti.func`s of `Solver` |
| `js/solver.js` | `Solver` (buffers, geometry, stepping, forces, hull outline) |
| `js/results.js` | `Averager`, `results`, `result_lines`, `playback`, `arrow_geometry` |
| `js/ui.js` | `Keys`, `Panel` |
| `js/polar.js` | `PolarPlot` |
| `js/main.js` | `main`, `Front`, `auto_quality` |
| `js/luts.js` | generated from `build_luts` by `tools/make_luts.py` |

## Keeping both versions identical

The two versions are kept in step by hand: same constants, same kernels, same order of operations.
After any change to the solver, change both and run the parity test:

```bash
python web/tools/parity.py
```

This runs a few validation cases with `Flowniac.py` and writes `parity_reference.json`. Then open
<http://localhost:8000/parity.html>: it runs the same cases on the WebGPU solver and compares the forces.
Early values agree to about 0.01 %; after a few chords the vortex shedding amplifies rounding
differences, so the long averages agree only within the usual scatter of a few percent.

`python -m http.server` sends no cache headers, so after editing a file the browser may keep running the
old module: reload with Ctrl+Shift+R.
