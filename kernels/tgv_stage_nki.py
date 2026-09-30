# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""One Williamson RK3 stage of the pseudo-spectral Taylor-Green solver on the permuting
3-D DFT (step 7). Spectral state and constants live in the kernel layout [kx, kz, ky]
(dft3d_perm_nki.perm of the standard arrays); physical fields are standard x, y, z.

Per stage: vorticity in spectral space (elementwise), 6 inverse transforms to physical,
cross product u x omega (elementwise), 3 forward transforms, dealias, Leray projection,
    du = b du + dt f ;  uh = uh + a du ;  [stage 3: uh *= decay]
Stage constants are module globals read at trace time (the frontend rejects closures):
    STAGE_A, STAGE_B, STAGE_DT, STAGE_DECAY (bool)
Inputs: uh_re, uh_im, vh_re, vh_im, wh_re, wh_im, du_re, du_im, dv_re, dv_im, dw_re, dw_im,
        C, S, nS (N x N), KX, KY, KZ, IK2 (1/K2), DEAL (0/1 mask), DECAY -- all (N,N,N) fp32
        in the kernel layout. Outputs: 12 arrays (new state, new accumulators).
Elementwise work runs on (N, N^2) views in 512-column chunks (N partitions).
"""
import numpy as np
import nki, nki.language as nl, nki.isa as nisa
from dft3d_perm_nki import dft3d_perm

STAGE_A = 1.0 / 3.0
STAGE_B = 0.0
STAGE_DT = 0.05
STAGE_DECAY = False
CH = 512


def _ld(view, c0, w, N):
    t = nl.ndarray((N, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=view[0:N, c0:c0 + w])
    return t


def _st(view, c0, w, N, t):
    nisa.dma_copy(dst=view[0:N, c0:c0 + w], src=t)


def _tt(a, b, op, N, w):
    o = nl.ndarray((N, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o, data1=a, data2=b, op=op)
    return o


def _ts(a, c, N, w):
    o = nl.ndarray((N, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o, data=a, op0=nl.multiply, operand0=c)
    return o


def _hbm(N):
    return nl.ndarray((N, N, N), dtype=nl.float32, buffer=nl.hbm)


def _out(N):
    return nl.ndarray((N, N, N), dtype=nl.float32, buffer=nl.shared_hbm)


def _v(a, N):
    return a.reshape((N, N * N))


def _vorticity(uh, vh, wh, KX, KY, KZ, N):
    """uh etc are (re, im) pairs of HBM arrays. Returns 3 (re, im) pairs in HBM."""
    ox_re, ox_im, oy_re, oy_im, oz_re, oz_im = _hbm(N), _hbm(N), _hbm(N), _hbm(N), _hbm(N), _hbm(N)
    NN = N * N
    for c0 in range(0, NN, CH):
        w = min(CH, NN - c0)
        kx = _ld(_v(KX, N), c0, w, N); ky = _ld(_v(KY, N), c0, w, N); kz = _ld(_v(KZ, N), c0, w, N)
        ur = _ld(_v(uh[0], N), c0, w, N); ui = _ld(_v(uh[1], N), c0, w, N)
        vr = _ld(_v(vh[0], N), c0, w, N); vi = _ld(_v(vh[1], N), c0, w, N)
        wr = _ld(_v(wh[0], N), c0, w, N); wi = _ld(_v(wh[1], N), c0, w, N)
        # ox = i(KY wh - KZ vh): re = KZ vi - KY wi ; im = KY wr - KZ vr
        _st(_v(ox_re, N), c0, w, N, _tt(_tt(kz, vi, nl.multiply, N, w), _tt(ky, wi, nl.multiply, N, w), nl.subtract, N, w))
        _st(_v(ox_im, N), c0, w, N, _tt(_tt(ky, wr, nl.multiply, N, w), _tt(kz, vr, nl.multiply, N, w), nl.subtract, N, w))
        # oy = i(KZ uh - KX wh): re = KX wi - KZ ui ; im = KZ ur - KX wr
        _st(_v(oy_re, N), c0, w, N, _tt(_tt(kx, wi, nl.multiply, N, w), _tt(kz, ui, nl.multiply, N, w), nl.subtract, N, w))
        _st(_v(oy_im, N), c0, w, N, _tt(_tt(kz, ur, nl.multiply, N, w), _tt(kx, wr, nl.multiply, N, w), nl.subtract, N, w))
        # oz = i(KX vh - KY uh): re = KY ui - KX vi ; im = KX vr - KY ur
        _st(_v(oz_re, N), c0, w, N, _tt(_tt(ky, ui, nl.multiply, N, w), _tt(kx, vi, nl.multiply, N, w), nl.subtract, N, w))
        _st(_v(oz_im, N), c0, w, N, _tt(_tt(kx, vr, nl.multiply, N, w), _tt(ky, ur, nl.multiply, N, w), nl.subtract, N, w))
    return (ox_re, ox_im), (oy_re, oy_im), (oz_re, oz_im)


def _cross(u, v, w_, ox, oy, oz, N):
    """Physical real fields (standard layout): cx = v oz - w oy ; cy = w ox - u oz ; cz = u oy - v ox."""
    cx, cy, cz = _hbm(N), _hbm(N), _hbm(N)
    NN = N * N
    for c0 in range(0, NN, CH):
        w = min(CH, NN - c0)
        tu = _ld(_v(u, N), c0, w, N); tv = _ld(_v(v, N), c0, w, N); tw = _ld(_v(w_, N), c0, w, N)
        tx = _ld(_v(ox, N), c0, w, N); ty = _ld(_v(oy, N), c0, w, N); tz = _ld(_v(oz, N), c0, w, N)
        _st(_v(cx, N), c0, w, N, _tt(_tt(tv, tz, nl.multiply, N, w), _tt(tw, ty, nl.multiply, N, w), nl.subtract, N, w))
        _st(_v(cy, N), c0, w, N, _tt(_tt(tw, tx, nl.multiply, N, w), _tt(tu, tz, nl.multiply, N, w), nl.subtract, N, w))
        _st(_v(cz, N), c0, w, N, _tt(_tt(tu, ty, nl.multiply, N, w), _tt(tv, tx, nl.multiply, N, w), nl.subtract, N, w))
    return cx, cy, cz


def _update(fx, fy, fz, uh, vh, wh, du, dv, dw, KX, KY, KZ, IK2, DEAL, DECAY, N):
    """Dealias, project, RK update (and decay). Returns 12 output arrays."""
    outs = []
    for i in range(12):
        outs.append(_out(N))
    NN = N * N
    a, b, dt = float(STAGE_A), float(STAGE_B), float(STAGE_DT)
    for c0 in range(0, NN, CH):
        w = min(CH, NN - c0)
        kx = _ld(_v(KX, N), c0, w, N); ky = _ld(_v(KY, N), c0, w, N); kz = _ld(_v(KZ, N), c0, w, N)
        ik2 = _ld(_v(IK2, N), c0, w, N); deal = _ld(_v(DEAL, N), c0, w, N)
        fs = [fx, fy, fz]; ks = [kx, ky, kz]; st = [uh, vh, wh]; acc = [du, dv, dw]
        for part in range(2):   # 0 = re, 1 = im
            f = []
            for c in range(3):
                f.append(_tt(_ld(_v(fs[c][part], N), c0, w, N), deal, nl.multiply, N, w))
            div = _tt(_tt(kx, f[0], nl.multiply, N, w), _tt(ky, f[1], nl.multiply, N, w), nl.add, N, w)
            div = _tt(div, _tt(kz, f[2], nl.multiply, N, w), nl.add, N, w)
            div = _tt(div, ik2, nl.multiply, N, w)
            for c in range(3):
                fp = _tt(f[c], _tt(ks[c], div, nl.multiply, N, w), nl.subtract, N, w)       # projected force
                d_old = _ld(_v(acc[c][part], N), c0, w, N)
                d_new = _tt(_ts(d_old, b, N, w), _ts(fp, dt, N, w), nl.add, N, w)          # du = b du + dt f
                s_old = _ld(_v(st[c][part], N), c0, w, N)
                s_new = _tt(s_old, _ts(d_new, a, N, w), nl.add, N, w)                      # uh += a du
                if STAGE_DECAY:
                    s_new = _tt(s_new, _ld(_v(DECAY, N), c0, w, N), nl.multiply, N, w)
                _st(_v(outs[2 * c + part], N), c0, w, N, s_new)          # outs 0..5: uh_re, uh_im, vh_re, vh_im, wh_re, wh_im
                _st(_v(outs[6 + 2 * c + part], N), c0, w, N, d_new)      # outs 6..11: du_re, du_im, dv_re, dv_im, dw_re, dw_im
    return outs


@nki.jit
def tgv_stage(uh_re: nl.ndarray, uh_im: nl.ndarray, vh_re: nl.ndarray, vh_im: nl.ndarray, wh_re: nl.ndarray, wh_im: nl.ndarray,
              du_re: nl.ndarray, du_im: nl.ndarray, dv_re: nl.ndarray, dv_im: nl.ndarray, dw_re: nl.ndarray, dw_im: nl.ndarray,
              C: nl.ndarray, S: nl.ndarray, nS: nl.ndarray, KX: nl.ndarray, KY: nl.ndarray, KZ: nl.ndarray,
              IK2: nl.ndarray, DEAL: nl.ndarray, DECAY: nl.ndarray):
    N = C.shape[0]
    inv_scale = 1.0 / float(N ** 3)
    uh = (uh_re, uh_im); vh = (vh_re, vh_im); wh = (wh_re, wh_im)
    ox, oy, oz = _vorticity(uh, vh, wh, KX, KY, KZ, N)
    # inverse transforms to physical (real): Sp = nS, Sn = S
    u = dft3d_perm(uh_re, uh_im, C, nS, S, False, True, inv_scale)[0]
    v = dft3d_perm(vh_re, vh_im, C, nS, S, False, True, inv_scale)[0]
    w = dft3d_perm(wh_re, wh_im, C, nS, S, False, True, inv_scale)[0]
    px = dft3d_perm(ox[0], ox[1], C, nS, S, False, True, inv_scale)[0]
    py = dft3d_perm(oy[0], oy[1], C, nS, S, False, True, inv_scale)[0]
    pz = dft3d_perm(oz[0], oz[1], C, nS, S, False, True, inv_scale)[0]
    cx, cy, cz = _cross(u, v, w, px, py, pz, N)
    # forward transforms (real in): Sp = S, Sn = nS
    fx = dft3d_perm(cx, None, C, S, nS, True, False, 1.0)
    fy = dft3d_perm(cy, None, C, S, nS, True, False, 1.0)
    fz = dft3d_perm(cz, None, C, S, nS, True, False, 1.0)
    outs = _update(fx, fy, fz, uh, vh, wh, (du_re, du_im), (dv_re, dv_im), (dw_re, dw_im), KX, KY, KZ, IK2, DEAL, DECAY, N)
    return (outs[0], outs[1], outs[2], outs[3], outs[4], outs[5], outs[6], outs[7], outs[8], outs[9], outs[10], outs[11])


def constants(N, nu=1.0 / 1600, dt=None):
    """Kernel-layout constants from the oracle's definitions."""
    from tgv_reference import wavenumbers
    from dft3d_perm_nki import dft_matrices, perm
    KX, KY, KZ, K2, deal = wavenumbers(N)
    dt = dt or 0.5 * (2 * np.pi / N)
    decay = np.exp(-nu * K2 * dt).astype(np.float32); decay[0, 0, 0] = 1.0
    C, S, nS = dft_matrices(N)
    f32 = lambda a: np.ascontiguousarray(perm(a), dtype=np.float32)
    return dict(C=C, S=S, nS=nS, KX=f32(KX), KY=f32(KY), KZ=f32(KZ), IK2=f32(1.0 / K2), DEAL=f32(deal.astype(np.float32)), DECAY=f32(decay), dt=dt)


RK_A = [1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0]
RK_B = [0.0, -5.0 / 9.0, -153.0 / 128.0]
