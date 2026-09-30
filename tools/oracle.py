# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""fp64 NumPy oracle for the same Re=1600 run: E(t), Omega(t), eps(t) CSV. Usage: oracle.py <N> <steps> <out.csv>"""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tgv_reference import tgv_init, wavenumbers, rhs, kinetic_energy
N, steps, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
nu = 1.0 / 1600; dt = 0.5 * (2 * np.pi / N)
u, v, w = tgv_init(N); KX, KY, KZ, K2, deal = wavenumbers(N)
uh, vh, wh = np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)
a = [1/3, 15/16, 8/15]; b = [0, -5/9, -153/128]
decay = np.exp(-nu * K2 * dt); decay[0, 0, 0] = 1.0
K2e = K2.copy(); K2e[0, 0, 0] = 0.0
def stats(uh, vh, wh):
    e = np.abs(uh)**2 + np.abs(vh)**2 + np.abs(wh)**2
    return 0.5 * e.sum() / N**6, 0.5 * (K2e * e).sum() / N**6
with open(out, "w", encoding="utf-8") as f:
    f.write("step,t,E,Omega,eps_enstrophy,eps_dEdt,step_wall_ms\n")
    E, Om = stats(uh, vh, wh); f.write(f"0,0,{E:.9g},{Om:.9g},{2*nu*Om:.9g},,\n"); Eprev = E; t = 0.0
    for step in range(1, steps + 1):
        t0 = time.perf_counter(); du = dv = dw = 0.0
        for s in range(3):
            fx, fy, fz = rhs(uh, vh, wh, KX, KY, KZ, K2, deal)
            du = b[s] * du + dt * fx; dv = b[s] * dv + dt * fy; dw = b[s] * dw + dt * fz
            uh = uh + a[s] * du; vh = vh + a[s] * dv; wh = wh + a[s] * dw
        uh *= decay; vh *= decay; wh *= decay; t += dt
        E, Om = stats(uh, vh, wh); ms = (time.perf_counter() - t0) * 1e3
        f.write(f"{step},{t:.6f},{E:.9g},{Om:.9g},{2*nu*Om:.9g},{-(E-Eprev)/dt:.9g},{ms:.3f}\n"); Eprev = E
        if step % 20 == 0: f.flush(); print(f"oracle N={N} step {step} t={t:.3f} E={E:.6f} eps={2*nu*Om:.6f} {ms:.0f} ms/step", flush=True)
print(f"ORACLE N={N} steps={steps} done", flush=True)
