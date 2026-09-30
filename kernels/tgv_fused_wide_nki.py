# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# DIRECT defaults to 1, the validated configuration.
"""Fused RK3 stage of the distributed Taylor-Green solver with fewer, wider, loads-first DMAs in the transform passes.

The NEFF is the per-stage kernel that driver/tgv_fused.c executes. The transform passes and the copies use these DMA
shapes:
  - plane (rows) passes: G = PLANE_EL // N planes per DMA. Every K-chunk of the G planes (re and im) is loaded before
    the first matmul; the outputs of the G planes for one 128-column block form one wide SBUF tile stored by one DMA.
    The per-plane (N, N) transform scratch becomes a whole-slab (L, N, N) scratch so pass 2 reads it plane-batched too.
  - x (dist) passes: WIDE_EL columns per DMA (WIDE_EL // 512 of the former pieces); every K-chunk of a column block is
    loaded once and reused for every output row block (the baseline reloads the source per 128 output rows).
  - pack/unpack/scatter/gather copies: WIDE_EL columns per piece instead of 512.
  - cross product: G row blocks per DMA, elementwise on the wide tile.
  - DIRECT = 1: no copies at all. The phase-a x passes store straight into the all_to_all send buffer, the phase-a plane
    passes read straight from the receive buffer, the phase-b plane passes store into the send buffer and the phase-b x
    passes read from the receive buffer. A 128-row block of the logical (N, C2) or (N, N) plane view spans 128 // (N // R)
    rank chunks of the staging buffer, so those DMAs address a partition sub-range of the tile (one DMA per chunk).
  - ELEM_EL: columns per tile in the curl and dealias/project/update passes (512 keeps the baseline shape; 1024 halves
    their DMA count with six resident constants of 4 KB per partition; 2048 would put 48 KB of DMA-written constants
    next to the transients and approach the 64 KB budget).
  - WIDE_ROWS, WIDE_X, WIDE_COPY, WIDE_CROSS: per-family switches (0 = the baseline helper for that family), for
    attributing the step-time change to a pass family.
Module globals N_GLOBAL, R_RANKS, WIDE_EL, PLANE_EL, DIRECT, ELEM_EL and the switches are read at trace time. SBUF per partition at the widest
point: plane pass 2 * nk * PLANE_EL * 4 B of loads plus 2 * PLANE_EL * 4 B of output (nk = N // 128); x pass
2 * nk * WIDE_EL * 4 B plus 2 * WIDE_EL * 4 B. At 256^3 with the defaults: 24 KB and 48 KB.
"""
import nki, nki.language as nl, nki.isa as nisa
import nki.collectives as ncc
from nki.collectives import ReplicaGroup
from dft3d_dist_nki import _load_mats, _hbm, P, _rows_pass, _dist_pass
from dft3d_dist3_nki import _pack4, _unpack4, _gather_rows, _scatter_rows
from tgv_dist_nki import _ld, _st, _tt, CH

N_GLOBAL = 128
R_RANKS = 2
WIDE_EL = 2048     # columns per DMA in the x passes and the copies (8 KB per partition)
PLANE_EL = 1024    # free elements per DMA in the plane passes: PLANE_EL // N planes (4 KB per partition)
DIRECT = 1         # 1: transforms read and write the staging buffers directly, no pack/unpack copies (validated)
ELEM_EL = 512      # columns per tile in the curl and dealias/project/update passes (512 = the baseline shape)
WIDE_ROWS = 1      # per-family switches for attribution: 0 uses the baseline helper for that family
WIDE_X = 1
WIDE_COPY = 1
WIDE_CROSS = 1



def _pin(bufs, rows_list, C2):
    """One (128, 512) tile read from the head of every staging buffer, summed: a use of each buffer at this point of
    the program, which fixes the buffer's live range (entry and exit) without writing into it."""
    acc = _ld(bufs[0].reshape((rows_list[0], C2)), 0, P, 0, CH)
    for i in range(1, len(bufs)):
        acc = _tt(acc, _ld(bufs[i].reshape((rows_list[i], C2)), 0, P, 0, CH), nl.add, P, CH)
    return acc

def _sb(shape):
    return nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)


def _ps(shape):
    return nl.ndarray(shape, dtype=nl.float32, buffer=nl.psum)


def _out(dst, p, scale):
    if scale == 1.0:
        nisa.tensor_copy(dst=dst, src=p)
    else:
        nisa.tensor_scalar(dst=dst, data=p, op0=nl.multiply, operand0=scale)


def _pieces(r0, rs, Lc):
    """Rows [r0, r0+rs) of a logical block cut at the Lc-row chunk boundaries: (chunk, first row in chunk, count,
    offset of the piece inside the block)."""
    out = []
    r = r0
    while r < r0 + rs:
        c = r // Lc
        l0 = r - c * Lc
        n = min(Lc - l0, r0 + rs - r)
        out.append((c, l0, n, r - r0))
        r += n
    return out


# ---- accessors: io(tile, r0, rs, p0, g, load) moves rows [r0, r0+rs) of the g planes p0.. between HBM and a (rs, g*N) tile
def _planes_hbm(a, N, S):
    """(L, N, N) HBM tensor (or an (L*N, N) view seen as blocks of S rows: pass S = P)."""
    def io(t, r0, rs, p0, g, load):
        W = g * N
        sb = t.ap([[W, rs], [N, g], [1, N]])
        hb = a.ap([[N, rs], [S * N, g], [1, N]], offset=p0 * S * N + r0 * N)
        if load:
            nisa.dma_copy(dst=sb, src=hb)
        else:
            nisa.dma_copy(dst=hb, src=sb)
    return io


def _planes_staged(buf, nf, fi, L, C2, N, R):
    """Staging buffer (R, nf, L, C2) seen as the (L, N, N) slab of field fi: plane p, row j sits in rank chunk j // Lr at
    buf[j // Lr][fi][p, (j % Lr) * N : +N], Lr = N // R. One DMA per rank chunk into a partition sub-range of the tile."""
    Lr = N // R

    def io(t, r0, rs, p0, g, load):
        W = g * N
        for (c, l0, n, off) in _pieces(r0, rs, Lr):
            sb = t.ap([[W, n], [N, g], [1, N]], offset=off * W)
            hb = buf.ap([[N, n], [C2, g], [1, N]], offset=((c * nf + fi) * L + p0) * C2 + l0 * N)
            if load:
                nisa.dma_copy(dst=sb, src=hb)
            else:
                nisa.dma_copy(dst=hb, src=sb)
    return io


# ---- accessors: io(tile, r0, rs, c0, w, load) moves the (rs, w) block at rows r0, columns c0 of an (N, M) matrix
def _rows_hbm(a):
    def io(t, r0, rs, c0, w, load):
        if load:
            nisa.dma_copy(dst=t, src=a[r0:r0 + rs, c0:c0 + w])
        else:
            nisa.dma_copy(dst=a[r0:r0 + rs, c0:c0 + w], src=t)
    return io


def _rows_staged(buf, fi, L):
    """Staging buffer (R, nf, L, C2) seen as the (N, C2) block of field fi: row i sits at buf[i // L][fi][i % L]."""
    def io(t, r0, rs, c0, w, load):
        for (c, l0, n, off) in _pieces(r0, rs, L):
            v = buf[c][fi][l0:l0 + n, c0:c0 + w]
            if load:
                nisa.dma_copy(dst=t[off:off + n, 0:w], src=v)
            else:
                nisa.dma_copy(dst=v, src=t[off:off + n, 0:w])
    return io


def _rows_pass_wide(dst_re, dst_im, src_re, src_im, cs, sps, sns, N, L, G, real_in, real_out, scale):
    """dst[p][s, :] = transform along the rows of src[p] (N x N) for the L planes, G planes per DMA. src_*/dst_* are
    plane accessors (None where real)."""
    nk = (N + P - 1) // P
    for p0 in range(0, L, G):
        g = min(G, L - p0)
        W = g * N
        re_k = []
        im_k = []
        for kt in range(nk):
            k0 = kt * P
            ks = min(P, N - k0)
            t = _sb((ks, W)); src_re(t, k0, ks, p0, g, True); re_k.append(t)
            if not real_in:
                t2 = _sb((ks, W)); src_im(t2, k0, ks, p0, g, True); im_k.append(t2)
        for s0 in range(0, N, P):
            ms = min(P, N - s0)
            o_re = _sb((ms, W))
            o_im = None if real_out else _sb((ms, W))
            for gi in range(g):
                c = gi * N + s0
                p_re = _ps((ms, N))
                for kt in range(nk):
                    ks = min(P, N - kt * P)
                    nisa.nc_matmul(dst=p_re, stationary=re_k[kt][0:ks, c:c + ms], moving=cs[kt], accumulate=(kt > 0))
                    if not real_in:
                        nisa.nc_matmul(dst=p_re, stationary=im_k[kt][0:ks, c:c + ms], moving=sps[kt], accumulate=True)
                _out(o_re[0:ms, gi * N:(gi + 1) * N], p_re, scale)
                if not real_out:
                    p_im = _ps((ms, N))
                    for kt in range(nk):
                        ks = min(P, N - kt * P)
                        nisa.nc_matmul(dst=p_im, stationary=re_k[kt][0:ks, c:c + ms], moving=sns[kt], accumulate=(kt > 0))
                        if not real_in:
                            nisa.nc_matmul(dst=p_im, stationary=im_k[kt][0:ks, c:c + ms], moving=cs[kt], accumulate=True)
                    _out(o_im[0:ms, gi * N:(gi + 1) * N], p_im, scale)
            dst_re(o_re, s0, ms, p0, g, False)
            if not real_out:
                dst_im(o_im, s0, ms, p0, g, False)


def _dist_pass_wide(dst_re, dst_im, src_re, src_im, cs, sps, sns, N, M, W, real_out, scale):
    """dst[N, M] = W (N x N) . src[N, M] along the rows, layout preserved, complex in. Columns in blocks of W: every
    K-chunk of a block (re, im) is loaded first, then each output row block m0 contracts per 512-column slice from
    SBUF and stores one wide tile. src_*/dst_* are row-block accessors."""
    nk = (N + P - 1) // P
    for c0 in range(0, M, W):
        w = min(W, M - c0)
        re_k = []
        im_k = []
        for kt in range(nk):
            k0 = kt * P
            ks = min(P, N - k0)
            a = _sb((ks, w)); src_re(a, k0, ks, c0, w, True); re_k.append(a)
            b = _sb((ks, w)); src_im(b, k0, ks, c0, w, True); im_k.append(b)
        for m0 in range(0, N, P):
            ms = min(P, N - m0)
            o_re = _sb((ms, w))
            o_im = None if real_out else _sb((ms, w))
            for s0 in range(0, w, CH):
                sw = min(CH, w - s0)
                p_re = _ps((ms, sw))
                for kt in range(nk):
                    ks = min(P, N - kt * P)
                    nisa.nc_matmul(dst=p_re, stationary=cs[kt][0:ks, m0:m0 + ms], moving=re_k[kt][0:ks, s0:s0 + sw], accumulate=(kt > 0))
                    nisa.nc_matmul(dst=p_re, stationary=sps[kt][0:ks, m0:m0 + ms], moving=im_k[kt][0:ks, s0:s0 + sw], accumulate=True)
                _out(o_re[0:ms, s0:s0 + sw], p_re, scale)
                if not real_out:
                    p_im = _ps((ms, sw))
                    for kt in range(nk):
                        ks = min(P, N - kt * P)
                        nisa.nc_matmul(dst=p_im, stationary=cs[kt][0:ks, m0:m0 + ms], moving=im_k[kt][0:ks, s0:s0 + sw], accumulate=(kt > 0))
                        nisa.nc_matmul(dst=p_im, stationary=sns[kt][0:ks, m0:m0 + ms], moving=re_k[kt][0:ks, s0:s0 + sw], accumulate=True)
                    _out(o_im[0:ms, s0:s0 + sw], p_im, 1.0)
            dst_re(o_re, m0, ms, c0, w, False)
            if not real_out:
                dst_im(o_im, m0, ms, c0, w, False)


def _copy_block_wide(dst_view, src_view, rows, cols, W):
    for c0 in range(0, cols, W):
        w = min(W, cols - c0)
        t = _sb((rows, w))
        nisa.dma_copy(dst=t, src=src_view[0:rows, c0:c0 + w])
        nisa.dma_copy(dst=dst_view[0:rows, c0:c0 + w], src=t)


def _pack4w(dst4, fi, src2, L, R, C2, W):
    for s in range(R):
        for r0 in range(0, L, P):
            rs = min(P, L - r0)
            _copy_block_wide(dst4[s][fi][r0:r0 + rs], src2[r0:r0 + rs, s * C2:(s + 1) * C2], rs, C2, W)


def _unpack4w(dst2, src4, fi, L, R, C2, W):
    for s in range(R):
        for r0 in range(0, L, P):
            rs = min(P, L - r0)
            _copy_block_wide(dst2[r0:r0 + rs, s * C2:(s + 1) * C2], src4[s][fi][r0:r0 + rs], rs, C2, W)


def _gather_rowsw(dst2, src4, fi, L, R, C2, W):
    for r in range(R):
        for r0 in range(0, L, P):
            rs = min(P, L - r0)
            _copy_block_wide(dst2[r * L + r0:r * L + r0 + rs], src4[r][fi][r0:r0 + rs], rs, C2, W)


def _scatter_rowsw(dst4, fi, src2, L, R, C2, W):
    for r in range(R):
        for r0 in range(0, L, P):
            rs = min(P, L - r0)
            _copy_block_wide(dst4[r][fi][r0:r0 + rs], src2[r * L + r0:r * L + r0 + rs], rs, C2, W)


def _curl_w(uh_re, uh_im, vh_re, vh_im, wh_re, wh_im, KX, KY, KZ, N, C2, W):
    """tgv_dist2_nki._curl with W columns per tile."""
    o = []
    for i in range(6):
        o.append(_hbm((N, C2)))
    for r0 in range(0, N, P):
        rs = min(P, N - r0)
        for c0 in range(0, C2, W):
            w = min(W, C2 - c0)
            kx = _ld(KX, r0, rs, c0, w); ky = _ld(KY, r0, rs, c0, w); kz = _ld(KZ, r0, rs, c0, w)
            ur = _ld(uh_re, r0, rs, c0, w); ui = _ld(uh_im, r0, rs, c0, w)
            vr = _ld(vh_re, r0, rs, c0, w); vi = _ld(vh_im, r0, rs, c0, w)
            wr = _ld(wh_re, r0, rs, c0, w); wi = _ld(wh_im, r0, rs, c0, w)
            _st(o[0], r0, rs, c0, w, _tt(_tt(kz, vi, nl.multiply, rs, w), _tt(ky, wi, nl.multiply, rs, w), nl.subtract, rs, w))
            _st(o[1], r0, rs, c0, w, _tt(_tt(ky, wr, nl.multiply, rs, w), _tt(kz, vr, nl.multiply, rs, w), nl.subtract, rs, w))
            _st(o[2], r0, rs, c0, w, _tt(_tt(kx, wi, nl.multiply, rs, w), _tt(kz, ui, nl.multiply, rs, w), nl.subtract, rs, w))
            _st(o[3], r0, rs, c0, w, _tt(_tt(kz, ur, nl.multiply, rs, w), _tt(kx, wr, nl.multiply, rs, w), nl.subtract, rs, w))
            _st(o[4], r0, rs, c0, w, _tt(_tt(ky, ui, nl.multiply, rs, w), _tt(kx, vi, nl.multiply, rs, w), nl.subtract, rs, w))
            _st(o[5], r0, rs, c0, w, _tt(_tt(kx, vr, nl.multiply, rs, w), _tt(ky, ur, nl.multiply, rs, w), nl.subtract, rs, w))
    return o


def _rk_tile(src, W):
    """RK constant (128, 512) input widened to (128, W) by SBUF copies (compute-written, so outside the DMA-resident budget)."""
    t = _ld(src, 0, P, 0, CH)
    if W <= CH:
        return t
    o = _sb((P, W))
    for c0 in range(0, W, CH):
        nisa.tensor_copy(dst=o[0:P, c0:c0 + CH], src=t)
    return o


def _cross_base(ph, cr, N, L):
    """Baseline cross product (tgv_fused_nki): (128, min(512, N)) tiles."""
    rows = L * N
    vu = ph[0].reshape((rows, N)); vv = ph[1].reshape((rows, N)); vw = ph[2].reshape((rows, N))
    vx = ph[3].reshape((rows, N)); vy = ph[4].reshape((rows, N)); vz = ph[5].reshape((rows, N))
    c0v = cr[0].reshape((rows, N)); c1v = cr[1].reshape((rows, N)); c2v = cr[2].reshape((rows, N))
    for r0 in range(0, rows, P):
        rs = min(P, rows - r0)
        for c0 in range(0, N, CH):
            wd = min(CH, N - c0)
            tu = _ld(vu, r0, rs, c0, wd); tv = _ld(vv, r0, rs, c0, wd); tw = _ld(vw, r0, rs, c0, wd)
            tx = _ld(vx, r0, rs, c0, wd); ty = _ld(vy, r0, rs, c0, wd); tz = _ld(vz, r0, rs, c0, wd)
            _st(c0v, r0, rs, c0, wd, _tt(_tt(tv, tz, nl.multiply, rs, wd), _tt(tw, ty, nl.multiply, rs, wd), nl.subtract, rs, wd))
            _st(c1v, r0, rs, c0, wd, _tt(_tt(tw, tx, nl.multiply, rs, wd), _tt(tu, tz, nl.multiply, rs, wd), nl.subtract, rs, wd))
            _st(c2v, r0, rs, c0, wd, _tt(_tt(tu, ty, nl.multiply, rs, wd), _tt(tv, tx, nl.multiply, rs, wd), nl.subtract, rs, wd))


def _rows_any(dst_re, dst_im, src_re, src_im, cs, sps, sns, N, L, G, real_in, real_out, scale, wide):
    """Plane pass over L planes: the wide helper (accessors) or the baseline per-plane helper (HBM tensors)."""
    if wide:
        _rows_pass_wide(dst_re, dst_im, src_re, src_im, cs, sps, sns, N, L, G, real_in, real_out, scale)
    else:
        for p in range(L):
            _rows_pass(dst_re[p], None if dst_im is None else dst_im[p], src_re[p], None if src_im is None else src_im[p],
                       cs, sps, sns, N, N, real_in, real_out, scale)


def _cross_wide(ph, cr, N, L, G):
    """cr = u x omega on the slabs, G row blocks of 128 rows per DMA."""
    rows = L * N
    nb = rows // P
    acc = [_planes_hbm(a, N, P) for a in ph]
    out = [_planes_hbm(a, N, P) for a in cr]
    for b0 in range(0, nb, G):
        g = min(G, nb - b0)
        W = g * N
        t = []
        for i in range(6):
            x = _sb((P, W)); acc[i](x, 0, P, b0, g, True); t.append(x)
        tu, tv, tw, tx, ty, tz = t
        out[0](_tt(_tt(tv, tz, nl.multiply, P, W), _tt(tw, ty, nl.multiply, P, W), nl.subtract, P, W), 0, P, b0, g, False)
        out[1](_tt(_tt(tw, tx, nl.multiply, P, W), _tt(tu, tz, nl.multiply, P, W), nl.subtract, P, W), 0, P, b0, g, False)
        out[2](_tt(_tt(tu, ty, nl.multiply, P, W), _tt(tv, tx, nl.multiply, P, W), nl.subtract, P, W), 0, P, b0, g, False)


@nki.jit
def tgv_fused(uh_re: nl.ndarray, uh_im: nl.ndarray, vh_re: nl.ndarray, vh_im: nl.ndarray, wh_re: nl.ndarray, wh_im: nl.ndarray,
              du_re: nl.ndarray, du_im: nl.ndarray, dv_re: nl.ndarray, dv_im: nl.ndarray, dw_re: nl.ndarray, dw_im: nl.ndarray,
              C: nl.ndarray, S: nl.ndarray, nS: nl.ndarray,
              KX: nl.ndarray, KY: nl.ndarray, KZ: nl.ndarray, IK2: nl.ndarray, DEAL: nl.ndarray, DECAY: nl.ndarray,
              RKA: nl.ndarray, RKB: nl.ndarray, RKDT: nl.ndarray):
    N = N_GLOBAL
    R = R_RANKS
    L = N // R
    C2 = N * N // R
    W = WIDE_EL
    G = max(1, PLANE_EL // N)
    WE = ELEM_EL
    assert not DIRECT or (WIDE_ROWS and WIDE_X), "DIRECT needs the wide plane and x passes (the staged accessors)"
    pl = (lambda a: _planes_hbm(a, N, N)) if WIDE_ROWS else (lambda a: a)
    grp = ReplicaGroup((tuple(range(R)),))
    cs, fsp, fsn = _load_mats(C, S, nS, N)          # forward signs (cs, fsp, fsn); inverse signs (cs, fsn, fsp)
    s4a = _hbm((R, 12, L, C2)); r4a = _hbm((R, 12, L, C2))
    s4b = _hbm((R, 6, L, C2)); r4b = _hbm((R, 6, L, C2))
    bufs = [s4a, r4a, s4b, r4b]
    brows = [R * 12 * L, R * 12 * L, R * 6 * L, R * 6 * L]
    g_entry = _pin(bufs, brows, C2)
    # ---- phase a: curl, six inverse transforms, one all_to_all
    o = _curl_w(uh_re, uh_im, vh_re, vh_im, wh_re, wh_im, KX, KY, KZ, N, C2, WE)
    spectra = [[uh_re, uh_im], [vh_re, vh_im], [wh_re, wh_im], [o[0], o[1]], [o[2], o[3]], [o[4], o[5]]]
    for f in range(6):
        if DIRECT:
            _dist_pass_wide(_rows_staged(s4a, 2 * f, L), _rows_staged(s4a, 2 * f + 1, L), _rows_hbm(spectra[f][0]), _rows_hbm(spectra[f][1]),
                            cs, fsn, fsp, N, C2, W, False, 1.0)
        else:
            r_re = _hbm((N, C2)); r_im = _hbm((N, C2))
            if WIDE_X:
                _dist_pass_wide(_rows_hbm(r_re), _rows_hbm(r_im), _rows_hbm(spectra[f][0]), _rows_hbm(spectra[f][1]), cs, fsn, fsp, N, C2, W, False, 1.0)
            else:
                _dist_pass(r_re, r_im, spectra[f][0], spectra[f][1], cs, fsn, fsp, N, C2, False, 1.0)
            if WIDE_COPY:
                _scatter_rowsw(s4a, 2 * f, r_re, L, R, C2, W); _scatter_rowsw(s4a, 2 * f + 1, r_im, L, R, C2, W)
            else:
                _scatter_rows(s4a, 2 * f, r_re, L, R, C2); _scatter_rows(s4a, 2 * f + 1, r_im, L, R, C2)
    ncc.all_to_all([s4a], [r4a], grp, 0)
    scale = 1.0 / float(N ** 3)
    ph = []
    for f in range(6):
        if DIRECT:
            src_re = _planes_staged(r4a, 12, 2 * f, L, C2, N, R); src_im = _planes_staged(r4a, 12, 2 * f + 1, L, C2, N, R)
        else:
            b_re = _hbm((L, N, N)); b_im = _hbm((L, N, N))
            if WIDE_COPY:
                _unpack4w(b_re.reshape((L, N * N)), r4a, 2 * f, L, R, C2, W); _unpack4w(b_im.reshape((L, N * N)), r4a, 2 * f + 1, L, R, C2, W)
            else:
                _unpack4(b_re.reshape((L, N * N)), r4a, 2 * f, L, R, C2); _unpack4(b_im.reshape((L, N * N)), r4a, 2 * f + 1, L, R, C2)
            src_re = pl(b_re); src_im = pl(b_im)
        out = _hbm((L, N, N))
        t_re = _hbm((L, N, N)); t_im = _hbm((L, N, N))
        _rows_any(pl(t_re), pl(t_im), src_re, src_im, cs, fsn, fsp, N, L, G, False, False, 1.0, WIDE_ROWS)
        _rows_any(pl(out), None, pl(t_re), pl(t_im), cs, fsn, fsp, N, L, G, False, True, scale, WIDE_ROWS)
        ph.append(out)
    # ---- phase b: cross product on the slabs, three forward transforms, one all_to_all
    cr = []
    for i in range(3):
        cr.append(_hbm((L, N, N)))
    if WIDE_CROSS:
        _cross_wide(ph, cr, N, L, G)
    else:
        _cross_base(ph, cr, N, L)
    for f in range(3):
        t_re = _hbm((L, N, N)); t_im = _hbm((L, N, N))
        _rows_any(pl(t_re), pl(t_im), pl(cr[f]), None, cs, fsp, fsn, N, L, G, True, False, 1.0, WIDE_ROWS)
        if DIRECT:
            _rows_pass_wide(_planes_staged(s4b, 6, 2 * f, L, C2, N, R), _planes_staged(s4b, 6, 2 * f + 1, L, C2, N, R),
                            _planes_hbm(t_re, N, N), _planes_hbm(t_im, N, N), cs, fsp, fsn, N, L, G, False, False, 1.0)
        else:
            b_re = _hbm((L, N, N)); b_im = _hbm((L, N, N))
            _rows_any(pl(b_re), pl(b_im), pl(t_re), pl(t_im), cs, fsp, fsn, N, L, G, False, False, 1.0, WIDE_ROWS)
            if WIDE_COPY:
                _pack4w(s4b, 2 * f, b_re.reshape((L, N * N)), L, R, C2, W); _pack4w(s4b, 2 * f + 1, b_im.reshape((L, N * N)), L, R, C2, W)
            else:
                _pack4(s4b, 2 * f, b_re.reshape((L, N * N)), L, R, C2); _pack4(s4b, 2 * f + 1, b_im.reshape((L, N * N)), L, R, C2)
    ncc.all_to_all([s4b], [r4b], grp, 0)
    fs = []
    for f in range(3):
        if DIRECT:
            src_re = _rows_staged(r4b, 2 * f, L); src_im = _rows_staged(r4b, 2 * f + 1, L)
        else:
            g_re = _hbm((N, C2)); g_im = _hbm((N, C2))
            if WIDE_COPY:
                _gather_rowsw(g_re, r4b, 2 * f, L, R, C2, W); _gather_rowsw(g_im, r4b, 2 * f + 1, L, R, C2, W)
            else:
                _gather_rows(g_re, r4b, 2 * f, L, R, C2); _gather_rows(g_im, r4b, 2 * f + 1, L, R, C2)
            src_re = _rows_hbm(g_re); src_im = _rows_hbm(g_im)
        f_re = _hbm((N, C2)); f_im = _hbm((N, C2))
        if WIDE_X or DIRECT:
            _dist_pass_wide(_rows_hbm(f_re), _rows_hbm(f_im), src_re, src_im, cs, fsp, fsn, N, C2, W, False, 1.0)
        else:
            _dist_pass(f_re, f_im, g_re, g_im, cs, fsp, fsn, N, C2, False, 1.0)
        fs.append([f_re, f_im])
    # ---- dealias, project, RK update, decay (unchanged from tgv_fused_nki)
    outs = []
    for i in range(12):
        outs.append(nl.ndarray((N, C2), dtype=nl.float32, buffer=nl.shared_hbm))
    ra = _rk_tile(RKA, WE); rb = _rk_tile(RKB, WE); rdt = _rk_tile(RKDT, WE)
    st = [[uh_re, uh_im], [vh_re, vh_im], [wh_re, wh_im]]
    acc = [[du_re, du_im], [dv_re, dv_im], [dw_re, dw_im]]
    for r0 in range(0, N, P):
        rs = min(P, N - r0)
        for c0 in range(0, C2, WE):
            w_ = min(WE, C2 - c0)
            kx = _ld(KX, r0, rs, c0, w_); ky = _ld(KY, r0, rs, c0, w_); kz = _ld(KZ, r0, rs, c0, w_)
            ik2 = _ld(IK2, r0, rs, c0, w_); deal = _ld(DEAL, r0, rs, c0, w_); dec = _ld(DECAY, r0, rs, c0, w_)
            ks = [kx, ky, kz]
            for part in range(2):
                f = []
                for c in range(3):
                    f.append(_tt(_ld(fs[c][part], r0, rs, c0, w_), deal, nl.multiply, rs, w_))
                div = _tt(_tt(kx, f[0], nl.multiply, rs, w_), _tt(ky, f[1], nl.multiply, rs, w_), nl.add, rs, w_)
                div = _tt(div, _tt(kz, f[2], nl.multiply, rs, w_), nl.add, rs, w_)
                div = _tt(div, ik2, nl.multiply, rs, w_)
                for c in range(3):
                    fp = _tt(f[c], _tt(ks[c], div, nl.multiply, rs, w_), nl.subtract, rs, w_)
                    d_new = _tt(_tt(_ld(acc[c][part], r0, rs, c0, w_), rb[0:rs, 0:w_], nl.multiply, rs, w_),
                                _tt(fp, rdt[0:rs, 0:w_], nl.multiply, rs, w_), nl.add, rs, w_)
                    s_new = _tt(_ld(st[c][part], r0, rs, c0, w_), _tt(d_new, ra[0:rs, 0:w_], nl.multiply, rs, w_), nl.add, rs, w_)
                    s_new = _tt(s_new, dec, nl.multiply, rs, w_)
                    _st(outs[2 * c + part], r0, rs, c0, w_, s_new)
                    _st(outs[6 + 2 * c + part], r0, rs, c0, w_, d_new)
    guard = nl.ndarray((P, CH), dtype=nl.float32, buffer=nl.shared_hbm)
    _st(guard, 0, P, 0, CH, _tt(g_entry, _pin(bufs, brows, C2), nl.add, P, CH))
    return (outs[0], outs[1], outs[2], outs[3], outs[4], outs[5], outs[6], outs[7], outs[8], outs[9], outs[10], outs[11], guard)
