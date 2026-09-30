# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""3-D DFT of an N^3 fp32 field as three permuting matmul passes, no transposes (step 7).

Every pass transforms the row axis of a 2-D view with the DATA as the nc_matmul stationary
operand and the N x N DFT matrix as the moving operand, so the result lands transposed:
    dst[m, i'] = sum_i src[i, m] W[i, i']        (nc_matmul: dst = stationary.T @ moving)
Applied to a[i, j, k]:
    pass 1  view [i, (j,k)]      -> b[j][k][i']      (transform along i)
    pass 2  per j, block [k, i'] -> c[j][i'][k']     (transform along k)
    pass 3  view [j, (i',k')]    -> out[i'][k'][j']  (transform along j)
so a forward transform returns the spectrum with the last two axes swapped,
F_perm[kx, kz, ky], and the same three passes with the conjugate matrices and a 1/N^3
scale map that layout back to the standard x, y, z order. The pseudo-spectral solver keeps
its spectral state in the permuted layout and its wavenumber constants permuted to match.
Complex data travels as (re, im) fp32 pairs; with W = C - i*sign*S:
    re' = C re + Sp im,   im' = C im + Sn re,   Sp = sign*S, Sn = -sign*S
(C and S are symmetric). A real input needs 2 matmuls per tile in pass 1 instead of 4, and
a real output needs 2 in pass 3. Tiles: stationary = data [N rows, 128 cols], moving =
matrix [N, N], PSUM [128, N]; DMAs move 512 columns at a time (2 KiB per partition at N=64).
N <= 128, N multiple of 32, N^2 multiple of 512.
"""
import numpy as np
import nki, nki.language as nl, nki.isa as nisa

P = 128
DMA_W = 512


def _rows_pass(dst_re, dst_im, src_re, src_im, C, Sp, Sn, N, M, real_in, real_out, scale):
    """dst[M, N] = transform along rows of src[N, M]; dst/src are 2-D HBM views."""
    w = min(DMA_W, M)
    for c0 in range(0, M, w):
        re_sb = nl.ndarray((N, w), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=re_sb, src=src_re[0:N, c0:c0 + w])
        if not real_in:
            im_sb = nl.ndarray((N, w), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=im_sb, src=src_im[0:N, c0:c0 + w])
        for s0 in range(0, w, P):
            ms = min(P, w - s0)
            p_re = nl.ndarray((ms, N), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=p_re, stationary=re_sb[0:N, s0:s0 + ms], moving=C, accumulate=False)
            if not real_in:
                nisa.nc_matmul(dst=p_re, stationary=im_sb[0:N, s0:s0 + ms], moving=Sp, accumulate=True)
            o_re = nl.ndarray((ms, N), dtype=nl.float32, buffer=nl.sbuf)
            if scale == 1.0:
                nisa.tensor_copy(dst=o_re, src=p_re)
            else:
                nisa.tensor_scalar(dst=o_re, data=p_re, op0=nl.multiply, operand0=scale)
            nisa.dma_copy(dst=dst_re[c0 + s0:c0 + s0 + ms, 0:N], src=o_re)
            if not real_out:
                p_im = nl.ndarray((ms, N), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=p_im, stationary=re_sb[0:N, s0:s0 + ms], moving=Sn, accumulate=False)
                if not real_in:
                    nisa.nc_matmul(dst=p_im, stationary=im_sb[0:N, s0:s0 + ms], moving=C, accumulate=True)
                o_im = nl.ndarray((ms, N), dtype=nl.float32, buffer=nl.sbuf)
                if scale == 1.0:
                    nisa.tensor_copy(dst=o_im, src=p_im)
                else:
                    nisa.tensor_scalar(dst=o_im, data=p_im, op0=nl.multiply, operand0=scale)
                nisa.dma_copy(dst=dst_im[c0 + s0:c0 + s0 + ms, 0:N], src=o_im)


def _load_mats(C, Sp, Sn, N):
    c = nl.ndarray((N, N), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=c, src=C)
    sp = nl.ndarray((N, N), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=sp, src=Sp)
    sn = nl.ndarray((N, N), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=sn, src=Sn)
    return c, sp, sn


def dft3d_perm(f_re, f_im, C, Sp, Sn, real_in, real_out, scale):
    """Three passes. f_im may be None when real_in. Returns (out_re, out_im) or (out_re,)."""
    N = C.shape[0]
    NN = N * N
    c, sp, sn = _load_mats(C, Sp, Sn, N)
    b_re = nl.ndarray((N, N, N), dtype=nl.float32, buffer=nl.hbm)
    b_im = nl.ndarray((N, N, N), dtype=nl.float32, buffer=nl.hbm)
    _rows_pass(b_re.reshape((NN, N)), b_im.reshape((NN, N)), f_re.reshape((N, NN)),
               None if real_in else f_im.reshape((N, NN)), c, sp, sn, N, NN, real_in, False, 1.0)
    k_re = nl.ndarray((N, N, N), dtype=nl.float32, buffer=nl.hbm)
    k_im = nl.ndarray((N, N, N), dtype=nl.float32, buffer=nl.hbm)
    for j in range(N):
        _rows_pass(k_re[j], k_im[j], b_re[j], b_im[j], c, sp, sn, N, N, False, False, 1.0)
    out_re = nl.ndarray((N, N, N), dtype=nl.float32, buffer=nl.shared_hbm)
    if real_out:
        _rows_pass(out_re.reshape((NN, N)), None, k_re.reshape((N, NN)), k_im.reshape((N, NN)),
                   c, sp, sn, N, NN, False, True, scale)
        return (out_re,)
    out_im = nl.ndarray((N, N, N), dtype=nl.float32, buffer=nl.shared_hbm)
    _rows_pass(out_re.reshape((NN, N)), out_im.reshape((NN, N)), k_re.reshape((N, NN)), k_im.reshape((N, NN)),
               c, sp, sn, N, NN, False, False, scale)
    return out_re, out_im


@nki.jit
def dft3d_fwd_real(f: nl.ndarray, C: nl.ndarray, S: nl.ndarray, nS: nl.ndarray):
    """Real field -> permuted spectrum (re, im). Forward: Sp = S, Sn = -S."""
    return dft3d_perm(f, None, C, S, nS, True, False, 1.0)


@nki.jit
def dft3d_fwd_cplx(f_re: nl.ndarray, f_im: nl.ndarray, C: nl.ndarray, S: nl.ndarray, nS: nl.ndarray):
    return dft3d_perm(f_re, f_im, C, S, nS, False, False, 1.0)


@nki.jit
def dft3d_inv_real(F_re: nl.ndarray, F_im: nl.ndarray, C: nl.ndarray, S: nl.ndarray, nS: nl.ndarray):
    """Permuted spectrum -> real field in standard layout. Inverse: Sp = -S, Sn = S, scale 1/N^3."""
    N = C.shape[0]
    return dft3d_perm(F_re, F_im, C, nS, S, False, True, 1.0 / float(N ** 3))[0]


@nki.jit
def dft3d_inv_cplx(F_re: nl.ndarray, F_im: nl.ndarray, C: nl.ndarray, S: nl.ndarray, nS: nl.ndarray):
    N = C.shape[0]
    return dft3d_perm(F_re, F_im, C, nS, S, False, False, 1.0 / float(N ** 3))


def dft_matrices(N):
    k = np.arange(N)[:, None]; n = np.arange(N)[None, :]
    ang = 2.0 * np.pi * k * n / N
    C = np.cos(ang).astype(np.float32); S = np.sin(ang).astype(np.float32)
    return C, S, (-S).astype(np.float32)


def perm(a):
    """Standard [kx, ky, kz] -> kernel layout [kx, kz, ky] (its own inverse)."""
    return np.ascontiguousarray(np.transpose(a, (0, 2, 1)))


if __name__ == "__main__":
    import sys, time
    ok = True
    for N in [int(a) for a in sys.argv[1:]] or (32, 64):
        rng = np.random.default_rng(3)
        f = rng.standard_normal((N, N, N), dtype=np.float32)
        C, S, nS = dft_matrices(N)
        t0 = time.perf_counter()
        F_re, F_im = nki.simulate(dft3d_fwd_real)(f, C, S, nS)
        ref = perm(np.fft.fftn(f.astype(np.float64)))
        e_fwd = max(np.abs(F_re - ref.real).max(), np.abs(F_im - ref.imag).max()) / np.abs(ref).max()
        g = nki.simulate(dft3d_inv_real)(np.asarray(F_re), np.asarray(F_im), C, S, nS)
        e_rt = np.abs(np.asarray(g) - f).max() / np.abs(f).max()
        G_re, G_im = nki.simulate(dft3d_fwd_cplx)(f, np.zeros_like(f), C, S, nS)
        e_cx = max(np.abs(G_re - ref.real).max(), np.abs(G_im - ref.imag).max()) / np.abs(ref).max()
        print(f"N={N} fwd_real={e_fwd:.2e} roundtrip={e_rt:.2e} fwd_cplx={e_cx:.2e} simulate_s={time.perf_counter()-t0:.1f}", flush=True)
        ok &= e_fwd < 1e-4 and e_rt < 1e-4 and e_cx < 1e-4
    print("STEP7-DFT", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
