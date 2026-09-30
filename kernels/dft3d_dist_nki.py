# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Distributed 3-D DFT over R NeuronCores by x-slab decomposition (item 2, step A).

Rank r holds the slab a[r*L:(r+1)*L, :, :] of an N^3 field (L = N/R planes). Forward:
  1. local 2-D transform of every plane with two permuting passes (data stationary):
     plane [j, k] -> [k, j'] -> [j', k']                                (O2, no transposes)
  2. pack the slab (L, N*N) into (R, L, C2) blocks by destination rank (C2 = N*N/R) and
     nki.collectives.all_to_all over collective_dim 0, so rank s receives rows i of every
     rank for its column chunk s: recv (R, L, C2) == [i, n] (N x C2)
  3. transform along i with the layout-preserving orientation (matrix stationary, data
     moving): out[i', n] = sum_i W[i, i'] recv[i, n]                     (O1)
Result per rank: spectrum [kx (all), n], n = local index into the (ky, kz) chunk s, i.e.
global (ky, kz) = divmod(s*C2 + n, N). Inverse runs the three steps backwards with the
conjugate matrices and 1/N^3. Contractions over N > 128 rows are chunked into 128-row
K-tiles accumulated in PSUM, so N = 256 and 512 work; N <= 512 (moving free size).
Complex data is (re, im) fp32; W = C - i*sign*S: re' = C re + Sp im, im' = C im + Sn re
with Sp = sign*S, Sn = -sign*S (forward sign = +1, inverse -1). Constants passed: C, Sp, Sn.
Module globals R_RANKS and N_GLOBAL are read at trace time (frontend rejects closures).
A NumPy mirror (numpy_forward / numpy_inverse) reproduces the same data movement rank by
rank for validation without hardware (the simulator has no collectives).
"""
import numpy as np
import nki, nki.language as nl, nki.isa as nisa
import nki.collectives as ncc
from nki.collectives import ReplicaGroup

P = 128
DMA_W = 512
R_RANKS = 2
N_GLOBAL = 128


def _rows_pass(dst_re, dst_im, src_re, src_im, cs, sps, sns, N, M, real_in, real_out, scale):
    """O2: dst[M, N] = transform along the rows of src[N, M]; rows staged and contracted in
    128-row K-chunks (SBUF has 128 partitions). cs/sps/sns are lists of (ks, N) SBUF tiles."""
    w = min(DMA_W, M)
    nk = (N + P - 1) // P
    for c0 in range(0, M, w):
        re_k = []
        im_k = []
        for kt in range(nk):
            k0 = kt * P
            ks = min(P, N - k0)
            t = nl.ndarray((ks, w), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=t, src=src_re[k0:k0 + ks, c0:c0 + w])
            re_k.append(t)
            if not real_in:
                t2 = nl.ndarray((ks, w), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=t2, src=src_im[k0:k0 + ks, c0:c0 + w])
                im_k.append(t2)
        for s0 in range(0, w, P):
            ms = min(P, w - s0)
            p_re = nl.ndarray((ms, N), dtype=nl.float32, buffer=nl.psum)
            for kt in range(nk):
                ks = min(P, N - kt * P)
                nisa.nc_matmul(dst=p_re, stationary=re_k[kt][0:ks, s0:s0 + ms], moving=cs[kt], accumulate=(kt > 0))
                if not real_in:
                    nisa.nc_matmul(dst=p_re, stationary=im_k[kt][0:ks, s0:s0 + ms], moving=sps[kt], accumulate=True)
            o_re = nl.ndarray((ms, N), dtype=nl.float32, buffer=nl.sbuf)
            if scale == 1.0:
                nisa.tensor_copy(dst=o_re, src=p_re)
            else:
                nisa.tensor_scalar(dst=o_re, data=p_re, op0=nl.multiply, operand0=scale)
            nisa.dma_copy(dst=dst_re[c0 + s0:c0 + s0 + ms, 0:N], src=o_re)
            if not real_out:
                p_im = nl.ndarray((ms, N), dtype=nl.float32, buffer=nl.psum)
                for kt in range(nk):
                    ks = min(P, N - kt * P)
                    nisa.nc_matmul(dst=p_im, stationary=re_k[kt][0:ks, s0:s0 + ms], moving=sns[kt], accumulate=(kt > 0))
                    if not real_in:
                        nisa.nc_matmul(dst=p_im, stationary=im_k[kt][0:ks, s0:s0 + ms], moving=cs[kt], accumulate=True)
                o_im = nl.ndarray((ms, N), dtype=nl.float32, buffer=nl.sbuf)
                if scale == 1.0:
                    nisa.tensor_copy(dst=o_im, src=p_im)
                else:
                    nisa.tensor_scalar(dst=o_im, data=p_im, op0=nl.multiply, operand0=scale)
                nisa.dma_copy(dst=dst_im[c0 + s0:c0 + s0 + ms, 0:N], src=o_im)


def _dist_pass(dst_re, dst_im, src_re, src_im, cs, sps, sns, N, M, real_out, scale):
    """O1: dst[N, M] = W (N x N) . src[N, M] along the rows, layout preserved. Complex in."""
    nk = (N + P - 1) // P
    for m0 in range(0, N, P):            # output rows i' (stationary free <= 128)
        ms = min(P, N - m0)
        for c0 in range(0, M, DMA_W):    # columns n (moving free <= 512)
            w = min(DMA_W, M - c0)
            p_re = nl.ndarray((ms, w), dtype=nl.float32, buffer=nl.psum)
            p_im = None
            if not real_out:
                p_im = nl.ndarray((ms, w), dtype=nl.float32, buffer=nl.psum)
            for kt in range(nk):
                k0 = kt * P
                ks = min(P, N - k0)
                re_sb = nl.ndarray((ks, w), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=re_sb, src=src_re[k0:k0 + ks, c0:c0 + w])
                im_sb = nl.ndarray((ks, w), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=im_sb, src=src_im[k0:k0 + ks, c0:c0 + w])
                nisa.nc_matmul(dst=p_re, stationary=cs[kt][0:ks, m0:m0 + ms], moving=re_sb, accumulate=(kt > 0))
                nisa.nc_matmul(dst=p_re, stationary=sps[kt][0:ks, m0:m0 + ms], moving=im_sb, accumulate=True)
                if not real_out:
                    nisa.nc_matmul(dst=p_im, stationary=cs[kt][0:ks, m0:m0 + ms], moving=im_sb, accumulate=(kt > 0))
                    nisa.nc_matmul(dst=p_im, stationary=sns[kt][0:ks, m0:m0 + ms], moving=re_sb, accumulate=True)
            o_re = nl.ndarray((ms, w), dtype=nl.float32, buffer=nl.sbuf)
            if scale == 1.0:
                nisa.tensor_copy(dst=o_re, src=p_re)
            else:
                nisa.tensor_scalar(dst=o_re, data=p_re, op0=nl.multiply, operand0=scale)
            nisa.dma_copy(dst=dst_re[m0:m0 + ms, c0:c0 + w], src=o_re)
            if not real_out:
                o_im = nl.ndarray((ms, w), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_im, src=p_im)
                nisa.dma_copy(dst=dst_im[m0:m0 + ms, c0:c0 + w], src=o_im)


def _copy_block(dst_view, src_view, rows, cols):
    """HBM->SBUF->HBM copy of a (rows, cols) block, rows <= 128, in 512-column pieces."""
    for c0 in range(0, cols, DMA_W):
        w = min(DMA_W, cols - c0)
        t = nl.ndarray((rows, w), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=src_view[0:rows, c0:c0 + w])
        nisa.dma_copy(dst=dst_view[0:rows, c0:c0 + w], src=t)


def _pack(dst3, src2, L, R, C2):
    """(L, R*C2) -> (R, L, C2): block s = columns [s*C2, (s+1)*C2)."""
    for s in range(R):
        for r0 in range(0, L, P):
            rs = min(P, L - r0)
            _copy_block(dst3[s][r0:r0 + rs], src2[r0:r0 + rs, s * C2:(s + 1) * C2], rs, C2)


def _unpack(dst2, src3, L, R, C2):
    """(R, L, C2) -> (L, R*C2)."""
    for s in range(R):
        for r0 in range(0, L, P):
            rs = min(P, L - r0)
            _copy_block(dst2[r0:r0 + rs, s * C2:(s + 1) * C2], src3[s][r0:r0 + rs], rs, C2)


def _load_mats(C, Sp, Sn, N):
    """DFT matrices as lists of (ks, N) SBUF tiles, one per 128-row K-chunk."""
    cs = []
    sps = []
    sns = []
    for k0 in range(0, N, P):
        ks = min(P, N - k0)
        c = nl.ndarray((ks, N), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=c, src=C[k0:k0 + ks, 0:N]); cs.append(c)
        sp = nl.ndarray((ks, N), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=sp, src=Sp[k0:k0 + ks, 0:N]); sps.append(sp)
        sn = nl.ndarray((ks, N), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=sn, src=Sn[k0:k0 + ks, 0:N]); sns.append(sn)
    return cs, sps, sns


def _hbm(shape):
    return nl.ndarray(shape, dtype=nl.float32, buffer=nl.hbm)


def dist_forward(a, C, Sp, Sn, N, R):
    """Real slab (L, N, N) -> spectrum (N, C2) as (re, im) in HBM (internal)."""
    L = N // R
    C2 = N * N // R
    cs, sps, sns = _load_mats(C, Sp, Sn, N)
    b_re = _hbm((L, N, N)); b_im = _hbm((L, N, N))
    t_re = _hbm((N, N)); t_im = _hbm((N, N))
    for p in range(L):
        _rows_pass(t_re, t_im, a[p], None, cs, sps, sns, N, N, True, False, 1.0)         # [j,k] -> [k,j']
        _rows_pass(b_re[p], b_im[p], t_re, t_im, cs, sps, sns, N, N, False, False, 1.0)  # [k,j'] -> [j',k']
    s_re = _hbm((R, L, C2)); s_im = _hbm((R, L, C2))
    _pack(s_re, b_re.reshape((L, N * N)), L, R, C2); _pack(s_im, b_im.reshape((L, N * N)), L, R, C2)
    r_re = _hbm((R, L, C2)); r_im = _hbm((R, L, C2))
    grp = ReplicaGroup((tuple(range(R)),))
    ncc.all_to_all([s_re], [r_re], grp, 0)
    ncc.all_to_all([s_im], [r_im], grp, 0)
    f_re = _hbm((N, C2)); f_im = _hbm((N, C2))
    _dist_pass(f_re, f_im, r_re.reshape((N, C2)), r_im.reshape((N, C2)), cs, sps, sns, N, C2, False, 1.0)
    return f_re, f_im


def dist_inverse(f_re, f_im, C, Sp, Sn, N, R, out):
    """Spectrum (N, C2) -> real slab written to out (L, N, N). Sp/Sn already inverse-signed."""
    L = N // R
    C2 = N * N // R
    cs, sps, sns = _load_mats(C, Sp, Sn, N)
    r_re = _hbm((N, C2)); r_im = _hbm((N, C2))
    _dist_pass(r_re, r_im, f_re, f_im, cs, sps, sns, N, C2, False, 1.0)   # along i', layout kept: [i, n]
    s_re = _hbm((R, L, C2)); s_im = _hbm((R, L, C2))
    grp = ReplicaGroup((tuple(range(R)),))
    ncc.all_to_all([r_re.reshape((R, L, C2))], [s_re], grp, 0)
    ncc.all_to_all([r_im.reshape((R, L, C2))], [s_im], grp, 0)
    b_re = _hbm((L, N, N)); b_im = _hbm((L, N, N))
    _unpack(b_re.reshape((L, N * N)), s_re, L, R, C2); _unpack(b_im.reshape((L, N * N)), s_im, L, R, C2)
    t_re = _hbm((N, N)); t_im = _hbm((N, N))
    scale = 1.0 / float(N ** 3)
    for p in range(L):
        _rows_pass(t_re, t_im, b_re[p], b_im[p], cs, sps, sns, N, N, False, False, 1.0)   # [j',k'] -> [k',j]
        _rows_pass(out[p], None, t_re, t_im, cs, sps, sns, N, N, False, True, scale)      # [k',j] -> [j,k] real
    return out


@nki.jit
def dist_fwd_real(a: nl.ndarray, C: nl.ndarray, S: nl.ndarray, nS: nl.ndarray):
    N = N_GLOBAL
    R = R_RANKS
    f_re, f_im = dist_forward(a, C, S, nS, N, R)
    C2 = N * N // R
    o_re = nl.ndarray((N, C2), dtype=nl.float32, buffer=nl.shared_hbm)
    o_im = nl.ndarray((N, C2), dtype=nl.float32, buffer=nl.shared_hbm)
    for r0 in range(0, N, P):
        rs = min(P, N - r0)
        _copy_block(o_re[r0:r0 + rs], f_re[r0:r0 + rs], rs, C2)
        _copy_block(o_im[r0:r0 + rs], f_im[r0:r0 + rs], rs, C2)
    return o_re, o_im


@nki.jit
def dist_inv_real(F_re: nl.ndarray, F_im: nl.ndarray, C: nl.ndarray, S: nl.ndarray, nS: nl.ndarray):
    N = N_GLOBAL
    R = R_RANKS
    L = N // R
    C2 = N * N // R
    # collectives cannot read IO tensors: stage the inputs into internal HBM first
    f_re = _hbm((N, C2)); f_im = _hbm((N, C2))
    for r0 in range(0, N, P):
        rs = min(P, N - r0)
        _copy_block(f_re[r0:r0 + rs], F_re[r0:r0 + rs], rs, C2)
        _copy_block(f_im[r0:r0 + rs], F_im[r0:r0 + rs], rs, C2)
    out = nl.ndarray((L, N, N), dtype=nl.float32, buffer=nl.shared_hbm)
    dist_inverse(f_re, f_im, C, nS, S, N, R, out)
    return out


# ---------------------------------------------------------------- NumPy mirror (same data movement)
def dft_matrices(N):
    k = np.arange(N)[:, None]; n = np.arange(N)[None, :]; ang = 2.0 * np.pi * k * n / N
    C = np.cos(ang).astype(np.float32); S = np.sin(ang).astype(np.float32)
    return C, S, (-S).astype(np.float32)


def _np_rows(re, im, C, Sp, Sn, real_in=False, real_out=False, scale=1.0):
    dre = re.T @ C + (0 if real_in else im.T @ Sp)
    if real_out: return dre * scale, None
    dim = (0 if real_in else im.T @ C) + re.T @ Sn
    return dre * scale, dim * scale


def numpy_forward(a, R):
    """a: full (N,N,N) real field. Returns per-rank spectra list of (re, im) each (N, C2)."""
    N = a.shape[0]; L = N // R; C2 = N * N // R
    C, S, nS = dft_matrices(N); C, S, nS = C.astype(np.float64), S.astype(np.float64), nS.astype(np.float64)
    packed = []
    for r in range(R):
        slab = a[r * L:(r + 1) * L]
        b_re = np.zeros((L, N, N)); b_im = np.zeros((L, N, N))
        for p in range(L):
            t_re, t_im = _np_rows(slab[p], None, C, S, nS, real_in=True)
            b_re[p], b_im[p] = _np_rows(t_re, t_im, C, S, nS)
        packed.append((b_re.reshape(L, R, C2).transpose(1, 0, 2), b_im.reshape(L, R, C2).transpose(1, 0, 2)))  # (R, L, C2) by dest rank
    outs = []
    for s in range(R):
        r_re = np.concatenate([packed[r][0][s] for r in range(R)], axis=0)  # rows i, from each source rank r
        r_im = np.concatenate([packed[r][1][s] for r in range(R)], axis=0)
        outs.append((C @ r_re + S @ r_im, C @ r_im + nS @ r_re))          # O1 along i
    return outs


def numpy_inverse(spectra, R):
    N = spectra[0][0].shape[0]; L = N // R; C2 = N * N // R
    C, S, nS = dft_matrices(N); C, S, nS = C.astype(np.float64), S.astype(np.float64), nS.astype(np.float64)
    Sp, Sn = nS, S   # inverse signs
    back = []
    for s in range(R):
        f_re, f_im = spectra[s]
        r_re = C @ f_re + Sp @ f_im; r_im = C @ f_im + Sn @ f_re
        back.append((r_re.reshape(R, L, C2), r_im.reshape(R, L, C2)))
    a = np.zeros((N, N, N))
    for r in range(R):
        s_re = np.stack([back[s][0][r] for s in range(R)]); s_im = np.stack([back[s][1][r] for s in range(R)])  # (R, L, C2) by source rank
        b_re = s_re.transpose(1, 0, 2).reshape(L, N, N); b_im = s_im.transpose(1, 0, 2).reshape(L, N, N)
        for p in range(L):
            t_re, t_im = _np_rows(b_re[p], b_im[p], C, Sp, Sn)
            a[r * L + p], _ = _np_rows(t_re, t_im, C, Sp, Sn, real_out=True, scale=1.0 / N ** 3)
    return a


def rank_layout_indices(N, R, s):
    """Global (kx, ky, kz) index arrays for rank s's spectrum (N, C2)."""
    C2 = N * N // R
    n = np.arange(C2) + s * C2
    kx = np.arange(N)[:, None] * np.ones((1, C2), dtype=int)
    ky = (n // N)[None, :] * np.ones((N, 1), dtype=int)
    kz = (n % N)[None, :] * np.ones((N, 1), dtype=int)
    return kx, ky, kz


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 32; R = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    rng = np.random.default_rng(0); a = rng.standard_normal((N, N, N))
    F = np.fft.fftn(a); spectra = numpy_forward(a, R)
    err = 0.0
    for s, (re, im) in enumerate(spectra):
        kx, ky, kz = rank_layout_indices(N, R, s); ref = F[kx, ky, kz]
        err = max(err, np.abs(re - ref.real).max(), np.abs(im - ref.imag).max())
    back = numpy_inverse(spectra, R)
    print(f"numpy mirror N={N} R={R}: forward max_abs_err vs fftn {err/np.abs(F).max():.2e}, roundtrip {np.abs(back-a).max():.2e}")
