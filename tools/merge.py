# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Sum the per-rank partial CSVs of tgv_fused into one run CSV (same columns as tgv_run.c) and,
if a reference CSV is given, compare E and eps_enstrophy row by row (row counts must match).
Usage: merge.py <prefix> <R> <nu> <out.csv> [ref.csv]"""
import sys, numpy as np
prefix, R, nu, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
parts = [np.loadtxt(f"{prefix}_r{r}.csv", delimiter=",", skiprows=1, ndmin=2) for r in range(R)]
n = {p.shape[0] for p in parts}
assert len(n) == 1 and parts[0].shape[0] > 1, f"rank CSVs disagree on row count or are empty: {n}"
step, t = parts[0][:, 0], parts[0][:, 1]
E = sum(p[:, 2] for p in parts); Om = sum(p[:, 3] for p in parts); wall = np.max([p[:, 4] for p in parts], axis=0)
eps = 2 * nu * Om
dEdt = np.full_like(E, np.nan); dEdt[1:] = -(E[1:] - E[:-1]) / (t[1:] - t[:-1])
with open(out, "w", encoding="utf-8") as f:
    f.write("step,t,E,Omega,eps_enstrophy,eps_dEdt,step_wall_ms\n")
    for i in range(len(E)):
        f.write(f"{int(step[i])},{t[i]:.6f},{E[i]:.9g},{Om[i]:.9g},{eps[i]:.9g},{'' if np.isnan(dEdt[i]) else f'{dEdt[i]:.9g}'},{wall[i]:.3f}\n")
ip = int(np.argmax(eps)); print(f"TGVDIST-MERGE rows={len(E)} t_end={t[-1]:.3f} E_end={E[-1]:.6g} peak_eps={eps[ip]:.5g} at t={t[ip]:.3f} median_step_ms={np.median(wall[1:]):.2f}")
if len(sys.argv) > 5:
    ref = np.loadtxt(sys.argv[5], delimiter=",", skiprows=1, ndmin=2, usecols=(0, 1, 2, 3, 4))
    assert ref.shape[0] == len(E) > 1, f"reference has {ref.shape[0]} rows, run has {len(E)}"
    assert np.allclose(ref[:, 1], t, atol=1e-5), "time axes differ"
    rE = np.abs(E - ref[:, 2]).max() / np.abs(ref[:, 2]).max(); reps = np.abs(eps - ref[:, 4]).max() / np.abs(ref[:, 4]).max()
    ir = int(np.argmax(ref[:, 4]))
    print(f"TGVDIST-CHECK vs {sys.argv[5]}: max_rel_err E {rE:.2e} eps {reps:.2e}; ref peak_eps={ref[ir,4]:.5g} at t={ref[ir,1]:.3f}")
