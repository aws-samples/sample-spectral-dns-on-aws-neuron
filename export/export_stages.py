# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Export the three RK3 stage NEFFs of the single-core solver for grid N into the current directory and write the
input .bin files (initial spectral state, zero accumulators, constants) that driver/tgv_run.c loads.
Usage: export_stages.py <N>   (N in 32..128, a multiple of 32; run from the build directory)"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("kernels", "tools", "export"):
    sys.path.insert(0, os.path.join(HERE, "..", d))
import numpy as np
import tgv_stage_nki as K
from tgv_reference import tgv_init
from dft3d_perm_nki import perm
from export_neff import export
N = int(sys.argv[1]); c = K.constants(N)
u, v, w = tgv_init(N); uh, vh, wh = np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)
def sp(a): return np.ascontiguousarray(perm(a.real), dtype=np.float32), np.ascontiguousarray(perm(a.imag), dtype=np.float32)
st = list(sp(uh)) + list(sp(vh)) + list(sp(wh)); acc = [np.zeros((N, N, N), np.float32) for _ in range(6)]
consts = [c["C"], c["S"], c["nS"], c["KX"], c["KY"], c["KZ"], c["IK2"], c["DEAL"], c["DECAY"]]
names = ["uh_re", "uh_im", "vh_re", "vh_im", "wh_re", "wh_im", "du_re", "du_im", "dv_re", "dv_im", "dw_re", "dw_im", "C", "S", "nS", "KX", "KY", "KZ", "IK2", "DEAL", "DECAY"]
for n, a in zip(names, st + acc + consts): a.tofile(f"stage_{n}_{N}.bin")
out = {}
for s in range(3):
    K.STAGE_A, K.STAGE_B, K.STAGE_DT, K.STAGE_DECAY = K.RK_A[s], K.RK_B[s], c["dt"], (s == 2)
    p = export(K.tgv_stage, tuple(np.zeros_like(a) for a in st + acc + consts), os.path.abspath(f"tgv_stage{s}_{N}.neff"))
    out[f"stage{s}"] = (p, os.path.getsize(p) if p else 0)
print(json.dumps({"stage": "stage_export", "N": N, "neffs": out}), flush=True)
