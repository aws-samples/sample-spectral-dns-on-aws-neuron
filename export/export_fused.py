# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Export the multi-core NEFF set for grid N and R ranks into the current directory: dist_fwd3_<N>_<R>.neff (initial
distributed transform), tgv_fused_<N>_<R>.neff (one NEFF for the three RK3 stages; the RK constants are inputs) and the
DFT matrices dist_{C,S,nS}_<N>.bin that driver/tgv_fused.c loads.
    export_fused.py <N> <R>
Kernel knobs from the environment, defaults are the validated configuration: TGV_WIDE_EL (2048), TGV_PLANE_EL (1024),
TGV_DIRECT (1), TGV_ELEM_EL (512), TGV_WIDE_ROWS/X/COPY/CROSS (1). R must divide N and N*N; N <= 512."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("kernels", "export"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import numpy as np
import dft3d_dist_nki as D, dft3d_dist3_nki as D3, tgv_dist_nki as T, tgv_fused_wide_nki as WK
from export_neff import export
N, R = int(sys.argv[1]), int(sys.argv[2]); tag = ""
assert N % R == 0 and (N * N) % R == 0 and N <= 512, "R must divide N and N*N; N <= 512"
for m in (D, D3, T, WK): m.N_GLOBAL, m.R_RANKS = N, R
for k, dflt in (("WIDE_EL", 2048), ("PLANE_EL", 1024), ("DIRECT", 1), ("ELEM_EL", 512), ("WIDE_ROWS", 1), ("WIDE_X", 1), ("WIDE_COPY", 1), ("WIDE_CROSS", 1)):
    setattr(WK, k, int(os.environ.get("TGV_" + k, dflt)))
print(f"knobs N={N} R={R} WIDE_EL={WK.WIDE_EL} PLANE_EL={WK.PLANE_EL} DIRECT={WK.DIRECT} ELEM_EL={WK.ELEM_EL} "
      f"switches rows={WK.WIDE_ROWS} x={WK.WIDE_X} copy={WK.WIDE_COPY} cross={WK.WIDE_CROSS} tag={tag!r}", flush=True)
L, C2 = N // R, N * N // R
C, S, nS = D.dft_matrices(N)
for n, a in (("C", C), ("S", S), ("nS", nS)):
    if not os.path.exists(f"dist_{n}_{N}.bin"): a.tofile(f"dist_{n}_{N}.bin")
z2 = np.zeros((N, C2), np.float32); z3 = np.zeros((L, N, N), np.float32); zk = np.zeros((128, 512), np.float32)
ok = True
def do(kernel, args, name):
    global ok
    path = os.path.abspath(f"{name}_{N}_{R}.neff")
    if os.path.exists(path): print("NEFF", path, "exists", flush=True); return
    t0 = time.time(); p = export(kernel, args, path); ok = ok and bool(p)
    print("NEFF", path, os.path.getsize(path) if p else "FAILED", f"{time.time() - t0:.0f}s", flush=True)
do(D3.dist_fwd3, (z3, z3, z3, C, S, nS), "dist_fwd3")
do(WK.tgv_fused, (z2,) * 12 + (C, S, nS) + (z2,) * 6 + (zk,) * 3, "tgv_fused")
sys.exit(0 if ok else 2)
