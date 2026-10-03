// Parity test: runs the cases of parity_reference.json (written by web/tools/parity.py with Flowniac.py)
// on the WebGPU solver, with the same procedure, and compares the forces.

import { QUALITY_CELLS, SHAPES, State } from "./config.js";
import { initGPU } from "./gpu.js";
import { Averager } from "./results.js";
import { Solver } from "./solver.js";

// allowed relative difference: early values must match closely, later the vortex shedding amplifies
// rounding differences (f32 on both sides, but different summation order and compilers). The impulsive
// start of the plate at high AoA is the most sensitive moment: Flowniac.py alone scatters 0.5 % there.
const TOLERANCE = { 0.25: 0.03, 1: 0.02, 3: 0.1, average: 0.05 };

async function runCase(device, c, chunk) {
  const st = new State();
  st.shape = SHAPES.findIndex(([key]) => key === c.shape);
  st.aoa = c.aoa;
  st.clamp();
  const sim = await Solver.create(device, QUALITY_CELLS[c.quality], { nTracerLines: 1, reSim: c.re_sim ?? null });
  sim.setParam("favg", 0);                        // the reference forces are those of a single step
  let enc = device.createCommandEncoder();
  sim.setGeometry(enc, st);
  sim.submit(enc);
  const avg = new Averager(sim.stepsPerChord);
  const ref = st.refFraction();
  const total = Math.round(c.chords * sim.stepsPerChord / chunk) * chunk;
  const checks = new Map(c.checkpoints.map((p) => [Math.round(p.chords * sim.stepsPerChord / chunk) * chunk, p.chords]));
  const points = [];
  const t0 = performance.now();
  for (let steps = chunk; steps <= total; steps += chunk) {
    enc = device.createCommandEncoder();
    sim.advance(enc, chunk);
    sim.probe(enc);
    sim.submit(enc);
    const coeffs = sim.coefficients(await sim.readOut(), ref);
    avg.add(coeffs, chunk);
    const v = avg.value;
    sim.setFarField(v[0][1], v[0][0], ref);
    if (checks.has(steps)) points.push({ chords: checks.get(steps), cd: coeffs[0][0], cl: coeffs[0][1] });
  }
  const seconds = (performance.now() - t0) / 1000;
  const v = avg.value;
  const mlups = sim.nx * sim.ny * total / seconds / 1e6;
  sim.destroy();
  return { points, average: { cd: v[0][0], cl: v[0][1] }, seconds, mlups };
}

function row(cells, cls = "") {
  const tr = document.createElement("tr");
  if (cls) tr.className = cls;
  for (const c of cells) {
    const td = document.createElement("td");
    td.textContent = c;
    tr.append(td);
  }
  return tr;
}

async function main() {
  const status = document.getElementById("status");
  const table = document.querySelector("#parity tbody");
  const ref = await (await fetch("parity_reference.json", { cache: "no-store" })).json();
  const { device } = await initGPU();
  const report = [];
  let failures = 0;
  status.textContent = `Reference from ${ref.generated}. Running ${ref.cases.length} cases ...`;
  for (const c of ref.cases) {
    status.textContent = `Running ${c.name} ...`;
    const web = await runCase(device, c, ref.chunk);
    const compare = (label, tol, py, js) => {
      for (const q of ["cd", "cl"]) {
        // the lift of a symmetric body is ~0 with a random sign: compare it with the size of the force
        const scale = q === "cl" && c.symmetric ? Math.abs(py.cd) : Math.max(Math.abs(py[q]), 0.05);
        const diff = Math.abs(js[q] - py[q]) / scale;
        const ok = diff <= tol;
        failures += ok ? 0 : 1;
        report.push({ case: c.name, at: label, q, python: py[q], web: js[q], diff, ok });
        table.append(row([c.name, label, q === "cd" ? "Cd" : "Cl", py[q].toFixed(4), js[q].toFixed(4),
                          `${(100 * diff).toFixed(2)} %`, ok ? "ok" : `> ${100 * tol} %`], ok ? "" : "fail"));
      }
    };
    c.checkpoints.forEach((p, k) => compare(`${p.chords} chords`, TOLERANCE[p.chords], p, web.points[k]));
    compare("average", TOLERANCE.average, c.average, web.average);
    table.append(row([`${web.seconds.toFixed(1)} s, ${web.mlups.toFixed(0)} million cell updates/s`], "note"));
  }
  status.textContent = failures === 0 ? "All values agree with Flowniac.py." : `${failures} values differ more than allowed.`;
  status.className = failures === 0 ? "ok" : "fail";
  window.parityReport = { failures, report };
}

main().catch((e) => {
  document.getElementById("status").textContent = e.message;
  window.parityReport = { error: e.message };
});
