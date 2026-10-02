"""Parity reference: run validation cases with Flowniac.py and store the forces in web/parity_reference.json.
web/parity.html runs the same cases with the WebGPU solver and compares. Re-run this after every
change to the solver (in both versions).

    python web/tools/parity.py

Both sides use the same procedure as the main loop: CHUNK steps, read the forces, average, update the
far field. Early instantaneous values must agree closely (same kernels, both f32); later the vortex
shedding amplifies rounding differences, so the long averages only agree within the usual scatter.
"""

import datetime
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import Flowniac as fs  # noqa: E402

CHUNK = 64                         # steps between force readings and far-field updates
CHECKPOINTS = [0.25, 1.0, 3.0]     # chords of flow: instantaneous total Cd, Cl
# symmetric=True: the lift is ~0 and its sign random (symmetry breaking), so parity.html compares it
# relative to the drag instead of relative to itself
CASES = [
    # re_sim is based on the chord; the cylinder diameter is 0.4 chord, so Re_D = 0.4 * 250 = 100
    dict(name="Cylinder Re_D 100", quality="low", re_sim=250, shape="cylinder", aoa=0.0, chords=8.0,
         symmetric=True),
    dict(name="Sail AoA 10", quality="low", shape="sail", aoa=10.0, chords=8.0),
    dict(name="Mast + sail AoA 15", quality="low", shape="mast_sail", aoa=15.0, chords=8.0),
    dict(name="Jib + main AoA 12", quality="low", shape="jib_main", aoa=12.0, chords=8.0),
    dict(name="NACA 2412 AoA 6", quality="low", shape="naca2412", aoa=6.0, chords=8.0),
    dict(name="Flat plate AoA 90", quality="low", shape="plate", aoa=90.0, chords=8.0, symmetric=True),
    dict(name="Sail AoA 10, medium", quality="medium", shape="sail", aoa=10.0, chords=8.0),
]


def run_case(case):
    st = fs.State()
    st.shape = [k for k, _ in fs.SHAPES].index(case["shape"])
    st.aoa = case["aoa"]
    st.clamp()
    sim = fs.Solver(fs.QUALITY_CELLS[case["quality"]], n_tracer_lines=1, re_sim=case.get("re_sim"))
    sim.set_geometry(st)
    avg = fs.Averager(sim.steps_per_chord)
    ref = st.ref_fraction()
    total = round(case["chords"] * sim.steps_per_chord / CHUNK) * CHUNK
    checks = {round(c * sim.steps_per_chord / CHUNK) * CHUNK: c for c in CHECKPOINTS}
    steps = 0
    points = []
    t0 = time.perf_counter()
    while steps < total:
        sim.advance(CHUNK)
        steps += CHUNK
        coeffs = sim.coefficients(ref)
        avg.add(coeffs, CHUNK)
        c = avg.value
        sim.set_far_field(c[0][1], c[0][0], ref)
        if steps in checks:
            points.append(dict(chords=checks[steps], cd=float(coeffs[0][0]), cl=float(coeffs[0][1])))
    c = avg.value
    seconds = time.perf_counter() - t0
    print(f"  {case['name']}: Cd {c[0][0]:.4f}  Cl {c[0][1]:.4f}  ({seconds:.1f} s)")
    return dict(case, steps=steps, checkpoints=points, average=dict(cd=float(c[0][0]), cl=float(c[0][1])))


def main():
    fs.init_taichi("gpu")
    print("Flowniac.py parity reference:")
    out = dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), chunk=CHUNK,
               cases=[run_case(case) for case in CASES])
    path = ROOT / "web" / "parity_reference.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
