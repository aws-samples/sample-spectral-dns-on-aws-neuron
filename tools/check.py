# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Validation check of a run CSV (step,t,E,Omega,eps_enstrophy,eps_dEdt,step_wall_ms) against the shipped references.

    check.py <run.csv> --grid N --ranks R [--ref <csv>] [--tol X] [--repeat <other run.csv>]

Every run must satisfy: no NaN; the t = 0 state has the analytic Taylor-Green energy E = 0.125 and enstrophy
Omega = 0.375 within 1e-6 relative; on every step the energy budget closes, |-dE/dt - 2 nu Omega| <= 5 percent of the
peak dissipation. With a reference (chosen from the grid and rank count unless --ref is given) the run must agree
with it row by row: max |E - E_ref| / max |E_ref| and the same for eps within the tolerance, the same number of rows
and time axis, and the dissipation peak at the reference's time. --repeat compares step,t,E,Omega of two runs
bitwise. Exit status 0 when every criterion passes, 1 otherwise; every criterion prints one CHECK line.

References (in references/): the fp64 NumPy oracle of the single-core path for 64^3 and 128^3 (same fp32 constants,
tolerance 1e-6), and the Trainium1 32-rank runs for 128^3, 256^3 (4e-9) and 512^3 (2e-7) for the multi-core path.
"""
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = os.path.join(HERE, "..", "references")
E0, OMEGA0 = 0.125, 0.375           # analytic t = 0 kinetic energy and enstrophy of the Taylor-Green field (u0 = 1)


def reference_for(grid, ranks):
    """(reference csv, tolerance) for the contract row, or (None, None) when the grid has no reference."""
    if grid == 64:
        return os.path.join(REFS, "oracle_64.csv"), 1e-6
    if grid == 128 and ranks == 1:
        return os.path.join(REFS, "oracle_128.csv"), 1e-6
    if grid == 128:
        return os.path.join(REFS, "tgv_dist2_128_32.csv"), 4e-9
    if grid == 256:
        return os.path.join(REFS, "tgv_dist2_256_32.csv"), 4e-9
    if grid == 512:
        return os.path.join(REFS, "tgv_dist2_512_32.csv"), 2e-7
    return None, None


def load(path):
    rows = [l.split(",") for l in open(path, encoding="utf-8").read().splitlines()[1:] if l.strip()]
    col = lambda i: np.array([float(r[i]) if r[i] != "" else np.nan for r in rows])
    return dict(step=col(0), t=col(1), E=col(2), Omega=col(3), eps=col(4), dEdt=col(5), ms=col(6), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--grid", type=int, required=True); ap.add_argument("--ranks", type=int, default=1)
    ap.add_argument("--nu", type=float, default=1.0 / 1600); ap.add_argument("--ref"); ap.add_argument("--tol", type=float)
    ap.add_argument("--repeat")
    a = ap.parse_args()
    ok = True

    def crit(name, passed, detail):
        nonlocal ok
        ok = ok and passed
        print(f"CHECK {name}: {'PASS' if passed else 'FAIL'} ({detail})")

    r = load(a.run)
    n = len(r["E"])
    crit("rows", n > 1, f"{n} rows, t_end={r['t'][-1]:.3f}")
    nan = int(np.isnan(r["E"]).sum() + np.isnan(r["Omega"]).sum())
    crit("no_nan", nan == 0, f"{nan} NaN in E or Omega")
    crit("t0_analytic", abs(r["E"][0] - E0) <= 1e-6 * E0 and abs(r["Omega"][0] - OMEGA0) <= 1e-6 * OMEGA0,
         f"E0={r['E'][0]:.9g} Omega0={r['Omega'][0]:.9g} vs analytic {E0} and {OMEGA0}")
    ip = int(np.nanargmax(r["eps"])); peak = r["eps"][ip]
    budget = np.abs(r["dEdt"][1:] - r["eps"][1:]) <= 0.05 * peak
    crit("energy_budget", bool(np.all(budget)), f"-dE/dt within 5% of peak eps on {int(budget.sum())}/{len(budget)} steps")
    w = r["ms"][2:] if n > 2 else r["ms"]
    print(f"CHECK timing: median_ms={np.median(w):.2f} p10_ms={np.percentile(w, 10):.2f} p90_ms={np.percentile(w, 90):.2f} "
          f"peak_eps={peak:.6g} at t={r['t'][ip]:.3f}")

    ref_path, tol = reference_for(a.grid, a.ranks)
    if a.ref: ref_path = a.ref
    if a.tol: tol = a.tol
    if ref_path:
        ref = load(ref_path)
        m = min(n, len(ref["E"]))
        crit("ref_rows", len(ref["E"]) >= n, f"reference {os.path.basename(ref_path)} has {len(ref['E'])} rows, run has {n}")
        crit("ref_time_axis", bool(np.allclose(ref["t"][:m], r["t"][:m], atol=1e-5)), "time axes agree" if m else "no rows")
        eE = np.abs(r["E"][:m] - ref["E"][:m]).max() / np.abs(ref["E"][:m]).max()
        eeps = np.abs(r["eps"][:m] - ref["eps"][:m]).max() / np.abs(ref["eps"][:m]).max()
        crit("ref_E", eE <= tol, f"max rel err E {eE:.2e} <= {tol:.0e}")
        crit("ref_eps", eeps <= tol, f"max rel err eps {eeps:.2e} <= {tol:.0e}")
        ir = int(np.nanargmax(ref["eps"][:m]))
        crit("ref_peak_time", ir == ip, f"run peak {peak:.6g} at t={r['t'][ip]:.3f}, reference {ref['eps'][ir]:.6g} at t={ref['t'][ir]:.3f}")
    else:
        print(f"CHECK reference: none for {a.grid}^3 (no NaN, analytic t0 and energy budget only)")

    if a.repeat:
        b = load(a.repeat)
        same = len(r["rows"]) == len(b["rows"]) and all(x[:4] == y[:4] for x, y in zip(r["rows"], b["rows"]))
        nd = sum(1 for x, y in zip(r["rows"], b["rows"]) if x[:4] != y[:4]) + abs(len(r["rows"]) - len(b["rows"]))
        crit("repeat_bitwise", same, f"step,t,E,Omega {'identical' if same else f'differ in {nd} rows'} vs {os.path.basename(a.repeat)}")

    print(f"CHECK {a.grid}^3 R={a.ranks} {os.path.basename(a.run)}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
