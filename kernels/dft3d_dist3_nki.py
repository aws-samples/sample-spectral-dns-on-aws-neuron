# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Fused three-field distributed 3-D DFT (item 2, step D): one all_to_all per phase instead of six.
dist_fwd3 transforms three real slabs (u, v, w) to spectra; dist_inv3 transforms three spectra to
slabs. All six (re, im) blocks of the three fields are packed into one (R, 6, L, C2) buffer so a
single nki.collectives.all_to_all moves them; the collective count per RK3 stage drops from 18 to 3.
Data movement and passes are those of dft3d_dist_nki (imported); layouts unchanged.
"""
import nki, nki.language as nl, nki.isa as nisa
import nki.collectives as ncc
from nki.collectives import ReplicaGroup
import dft3d_dist_nki as D
from dft3d_dist_nki import _rows_pass, _dist_pass, _copy_block, _load_mats, _hbm, P

N_GLOBAL = 128
R_RANKS = 2


def _pack4(dst4, fi, src2, L, R, C2):
    """(L, R*C2) -> dst4[s][fi] (L, C2) for every destination rank s."""
    for s in range(R):
        for r0 in range(0, L, P):
            rs = min(P, L - r0)
            _copy_block(dst4[s][fi][r0:r0 + rs], src2[r0:r0 + rs, s * C2:(s + 1) * C2], rs, C2)


def _unpack4(dst2, src4, fi, L, R, C2):
    for s in range(R):
        for r0 in range(0, L, P):
            rs = min(P, L - r0)
            _copy_block(dst2[r0:r0 + rs, s * C2:(s + 1) * C2], src4[s][fi][r0:r0 + rs], rs, C2)


def _gather_rows(dst2, src4, fi, L, R, C2):
    """recv4[src][fi] (L, C2) blocks -> dst2 (N, C2) rows src*L.. (the x index)."""
    for r in range(R):
        for r0 in range(0, L, P):
            rs = min(P, L - r0)
            _copy_block(dst2[r * L + r0:r * L + r0 + rs], src4[r][fi][r0:r0 + rs], rs, C2)


def _scatter_rows(dst4, fi, src2, L, R, C2):
    for r in range(R):
        for r0 in range(0, L, P):
            rs = min(P, L - r0)
            _copy_block(dst4[r][fi][r0:r0 + rs], src2[r * L + r0:r * L + r0 + rs], rs, C2)


@nki.jit
def dist_fwd3(a0: nl.ndarray, a1: nl.ndarray, a2: nl.ndarray, C: nl.ndarray, S: nl.ndarray, nS: nl.ndarray):
    N = N_GLOBAL
    R = R_RANKS
    L = N // R
    C2 = N * N // R
    cs, sps, sns = _load_mats(C, S, nS, N)
    s4 = _hbm((R, 6, L, C2))
    srcs = [a0, a1, a2]
    for f in range(3):
        b_re = _hbm((L, N, N)); b_im = _hbm((L, N, N)); t_re = _hbm((N, N)); t_im = _hbm((N, N))
        for p in range(L):
            _rows_pass(t_re, t_im, srcs[f][p], None, cs, sps, sns, N, N, True, False, 1.0)
            _rows_pass(b_re[p], b_im[p], t_re, t_im, cs, sps, sns, N, N, False, False, 1.0)
        _pack4(s4, 2 * f, b_re.reshape((L, N * N)), L, R, C2)
        _pack4(s4, 2 * f + 1, b_im.reshape((L, N * N)), L, R, C2)
    r4 = _hbm((R, 6, L, C2))
    ncc.all_to_all([s4], [r4], ReplicaGroup((tuple(range(R)),)), 0)
    outs = []
    for f in range(3):
        g_re = _hbm((N, C2)); g_im = _hbm((N, C2))
        _gather_rows(g_re, r4, 2 * f, L, R, C2); _gather_rows(g_im, r4, 2 * f + 1, L, R, C2)
        o_re = nl.ndarray((N, C2), dtype=nl.float32, buffer=nl.shared_hbm)
        o_im = nl.ndarray((N, C2), dtype=nl.float32, buffer=nl.shared_hbm)
        _dist_pass(o_re, o_im, g_re, g_im, cs, sps, sns, N, C2, False, 1.0)
        outs.append(o_re); outs.append(o_im)
    return (outs[0], outs[1], outs[2], outs[3], outs[4], outs[5])


@nki.jit
def dist_inv3(F0_re: nl.ndarray, F0_im: nl.ndarray, F1_re: nl.ndarray, F1_im: nl.ndarray, F2_re: nl.ndarray, F2_im: nl.ndarray,
              C: nl.ndarray, S: nl.ndarray, nS: nl.ndarray):
    N = N_GLOBAL
    R = R_RANKS
    L = N // R
    C2 = N * N // R
    cs, sps, sns = _load_mats(C, nS, S, N)      # inverse signs
    s4 = _hbm((R, 6, L, C2))
    F = [[F0_re, F0_im], [F1_re, F1_im], [F2_re, F2_im]]
    for f in range(3):
        r_re = _hbm((N, C2)); r_im = _hbm((N, C2))
        _dist_pass(r_re, r_im, F[f][0], F[f][1], cs, sps, sns, N, C2, False, 1.0)
        _scatter_rows(s4, 2 * f, r_re, L, R, C2); _scatter_rows(s4, 2 * f + 1, r_im, L, R, C2)
    r4 = _hbm((R, 6, L, C2))
    ncc.all_to_all([s4], [r4], ReplicaGroup((tuple(range(R)),)), 0)
    scale = 1.0 / float(N ** 3)
    outs = []
    for f in range(3):
        b_re = _hbm((L, N, N)); b_im = _hbm((L, N, N))
        _unpack4(b_re.reshape((L, N * N)), r4, 2 * f, L, R, C2); _unpack4(b_im.reshape((L, N * N)), r4, 2 * f + 1, L, R, C2)
        out = nl.ndarray((L, N, N), dtype=nl.float32, buffer=nl.shared_hbm)
        t_re = _hbm((N, N)); t_im = _hbm((N, N))
        for p in range(L):
            _rows_pass(t_re, t_im, b_re[p], b_im[p], cs, sps, sns, N, N, False, False, 1.0)
            _rows_pass(out[p], None, t_re, t_im, cs, sps, sns, N, N, False, True, scale)
        outs.append(out)
    return (outs[0], outs[1], outs[2])
