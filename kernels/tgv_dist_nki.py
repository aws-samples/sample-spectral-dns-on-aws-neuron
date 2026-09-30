# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Elementwise tile helpers of the distributed Taylor-Green stage: load and store a (rows, w) fp32 tile of a 2-D HBM
view, tensor-tensor and tensor-scalar operations on SBUF tiles. The spectral state lives in the rank layout of
dft3d_dist_nki: rank s holds an (N, C2) block, row = kx index, column n -> global (ky, kz) = divmod(s*C2 + n, N).
Elementwise work runs on 128-row x 512-column SBUF tiles. N_GLOBAL and R_RANKS are set by the export script.
"""
import nki, nki.language as nl, nki.isa as nisa

P = 128
CH = 512
N_GLOBAL = 128
R_RANKS = 2


def _ld(view, r0, rs, c0, w):
    t = nl.ndarray((rs, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=view[r0:r0 + rs, c0:c0 + w])
    return t


def _st(view, r0, rs, c0, w, t):
    nisa.dma_copy(dst=view[r0:r0 + rs, c0:c0 + w], src=t)


def _tt(a, b, op, rs, w):
    o = nl.ndarray((rs, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o, data1=a, data2=b, op=op)
    return o


def _ts(a, c, rs, w):
    o = nl.ndarray((rs, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o, data=a, op0=nl.multiply, operand0=c)
    return o


def _outs(shape, k):
    outs = []
    for i in range(k):
        outs.append(nl.ndarray(shape, dtype=nl.float32, buffer=nl.shared_hbm))
    return outs
