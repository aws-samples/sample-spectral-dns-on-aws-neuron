# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Spectral checkpoints of the multi-core driver -> VTK ImageData frames for ParaView.

    vtk_frames.py --prefix build/n256_r8/tgvfused_256_8 --N 256 --ranks 8 --out frames [--dt 0.01227] [--check-analytic]

Run the case with CKPT_EVERY=<steps> (scripts/run.sh) so that driver/tgv_fused dumps <prefix>_ckpt_<t:.4f>_r<rank>.bin
at t = 0 and every CKPT_EVERY steps. For every checkpoint time the [kx, ky, kz] spectrum is reassembled from the rank
blocks (rank r holds (6, N, C2) fp32 with column g = r C2 + n and global (ky, kz) = divmod(g, N)), inverse-transformed
onto the N^3 grid (the forward transform is unnormalised, so numpy.fft.ifftn is the inverse) and turned into three
point-data arrays:

    u                    (3 components)  velocity
    vorticity_magnitude                  |curl u|, from the spectral derivatives i k_j u_i
    q_criterion                          0.5 (|Omega|^2 - |S|^2), Omega / S the antisymmetric /
                                         symmetric parts of the velocity gradient (nine i k_j u_i)

Output: <out>/frame_<idx:04d>.vti (XML ImageData, appended raw binary, zlib, float32, origin 0,
spacing 2 pi / N), <out>/frames.pvd (the time series) and <out>/frames_stats.json (per frame: t,
step, ranges and percentiles of every array, and the field-space kinetic energy 0.5 <|u|^2> and
enstrophy 0.5 <|omega|^2>, which must equal the E and Omega columns of the solver's energy.csv by
Parseval; --energy-csv makes the script check that and report the maximum relative deviation).

Requires NumPy only; frames open in ParaView (File, Open, frames.pvd).
       [--dt 0.0121951] [--workers 3] [--zlib-level 4] [--energy-csv energy.csv] [--check-analytic]
"""
import argparse, glob, json, os, re, struct, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

BLOCK = 1 << 20  # zlib block size in bytes (any value is legal; one header entry per block)


def discover_times(prefix):
    ts = []
    for p in glob.glob(f"{prefix}_ckpt_*_r0.bin"):
        m = re.search(r"_ckpt_([0-9.]+)_r0\.bin$", p)
        ts.append(float(m.group(1)))
    return sorted(ts)


def load_ckpt_ranks(prefix, t, N, R):
    """Rank r holds (6, N, C2) fp32 blocks (uh_re, uh_im, vh_re, vh_im, wh_re, wh_im) with column g = r*C2 + n and
    global (ky, kz) = divmod(g, N); concatenating the ranks along the column axis gives [kx, ky, kz] order."""
    C2 = N * N // R
    parts = []
    for r in range(R):
        path = f"{prefix}_ckpt_{t:.4f}_r{r}.bin"
        a = np.fromfile(path, dtype=np.float32)
        if a.size != 6 * N * C2:
            raise SystemExit(f"{path}: {a.size} floats, expected {6 * N * C2}")
        parts.append(a.reshape(6, N, C2))
    a = np.concatenate(parts, axis=2).reshape(6, N, N, N).astype(np.float64)
    return tuple(np.ascontiguousarray(a[2 * i]) + 1j * np.ascontiguousarray(a[2 * i + 1]) for i in range(3))


def spectrum(prefix, t, N, R):
    uh, vh, wh = load_ckpt_ranks(prefix, t, N, R)
    return [np.ascontiguousarray(a.astype(np.complex64)) for a in (uh, vh, wh)]


def fields(uh, vh, wh, N):
    """Velocity, vorticity magnitude and Q on the N^3 grid, all float32 [x, y, z]."""
    k = np.fft.fftfreq(N, d=1.0 / N).astype(np.float32)
    K = (k[:, None, None], k[None, :, None], k[None, None, :])
    inv = lambda F: np.fft.ifftn(F).real.astype(np.float32, copy=False)
    U = [inv(uh), inv(vh), inv(wh)]
    G = [[inv(1j * K[j] * F) for j in range(3)] for F in (uh, vh, wh)]   # G[i][j] = d u_i / d x_j
    ox = G[2][1] - G[1][2]; oy = G[0][2] - G[2][0]; oz = G[1][0] - G[0][1]
    o2 = ox * ox + oy * oy + oz * oz
    del ox, oy, oz
    omega2 = 0.5 * o2                                        # |Omega|^2, Omega_ij = (G_ij - G_ji)/2
    s2 = G[0][0] ** 2 + G[1][1] ** 2 + G[2][2] ** 2
    for i in range(3):
        for j in range(i + 1, 3):
            s2 += 0.5 * (G[i][j] + G[j][i]) ** 2                     # |S|^2, S_ij = (G_ij + G_ji)/2
    del G
    q = (0.5 * (omega2 - s2)).astype(np.float32)
    vort = np.sqrt(o2).astype(np.float32)
    u = np.stack(U, axis=-1)                                 # (N, N, N, 3)
    return u, vort, q


def compress(raw, level):
    n = (len(raw) + BLOCK - 1) // BLOCK
    last = len(raw) - (n - 1) * BLOCK
    with ThreadPoolExecutor(max_workers=4) as ex:
        parts = list(ex.map(lambda i: zlib.compress(raw[i * BLOCK:(i + 1) * BLOCK], level), range(n)))
    head = struct.pack("<QQQ", n, BLOCK, last) + struct.pack(f"<{n}Q", *(len(p) for p in parts))
    return head + b"".join(parts)


def write_vti(path, N, arrays, level):
    """arrays: list of (name, ncomp, float32 array [x, y, z(, c)]). VTK point order is x fastest."""
    h = 2.0 * np.pi / N
    blobs, xml, off = [], [], 0
    for name, nc, a in arrays:
        flat = np.ascontiguousarray(a.transpose(2, 1, 0) if nc == 1 else a.transpose(2, 1, 0, 3))
        blob = compress(flat.astype("<f4", copy=False).tobytes(), level)
        xml.append(f'<DataArray type="Float32" Name="{name}" NumberOfComponents="{nc}" format="appended" offset="{off}"/>')
        blobs.append(blob); off += len(blob)
    ext = f"0 {N - 1} 0 {N - 1} 0 {N - 1}"
    head = ('<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" '
            'header_type="UInt64" compressor="vtkZLibDataCompressor">\n'
            f'<ImageData WholeExtent="{ext}" Origin="0 0 0" Spacing="{h!r} {h!r} {h!r}">\n<Piece Extent="{ext}">\n'
            '<PointData Scalars="vorticity_magnitude" Vectors="u">\n' + "\n".join(xml) +
            '\n</PointData>\n<CellData/>\n</Piece>\n</ImageData>\n<AppendedData encoding="raw">\n_')
    with open(path, "wb") as f:
        f.write(head.encode()); [f.write(b) for b in blobs]; f.write(b"\n</AppendedData>\n</VTKFile>\n")
    return os.path.getsize(path)


def stats_of(name, a):
    q = np.percentile(a, [50, 90, 95, 99]).tolist()
    return {"min": float(a.min()), "max": float(a.max()), "p50": q[0], "p90": q[1], "p95": q[2], "p99": q[3]}


def analytic_check(u, vort, q, N):
    x = np.arange(N, dtype=np.float64) * (2 * np.pi / N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    ua = np.sin(X) * np.cos(Y) * np.cos(Z); va = -np.cos(X) * np.sin(Y) * np.cos(Z)
    oa = np.sqrt((np.cos(X) * np.sin(Y) * np.sin(Z)) ** 2 + (np.sin(X) * np.cos(Y) * np.sin(Z)) ** 2 + (2 * np.sin(X) * np.sin(Y) * np.cos(Z)) ** 2)
    # Q = -0.5 G_ij G_ji for the divergence-free field, from the analytic gradient
    gxx = np.cos(X) * np.cos(Y) * np.cos(Z); gxy = -np.sin(X) * np.sin(Y) * np.cos(Z); gxz = -np.sin(X) * np.cos(Y) * np.sin(Z)
    gyx = np.sin(X) * np.sin(Y) * np.cos(Z); gyy = -np.cos(X) * np.cos(Y) * np.cos(Z); gyz = np.cos(X) * np.sin(Y) * np.sin(Z)
    qa = -0.5 * (gxx * gxx + gyy * gyy + 2 * gxy * gyx)   # gz* = 0 (w = 0) and third row zero
    return {"u": float(np.abs(u[..., 0] - ua).max()), "v": float(np.abs(u[..., 1] - va).max()), "w": float(np.abs(u[..., 2]).max()),
            "vorticity_magnitude": float(np.abs(vort - oa).max()), "q_criterion": float(np.abs(q - qa).max()), "q_scale": float(np.abs(qa).max())}


def do_frame(args):
    prefix, t, idx, N, R, out, level, dt, check, delete = args
    t0 = time.perf_counter()
    uh, vh, wh = spectrum(prefix, t, N, R)
    if delete:
        for r in range(R):
            os.remove(f"{prefix}_ckpt_{t:.4f}_r{r}.bin")
    t1 = time.perf_counter()
    u, vort, q = fields(uh, vh, wh, N)
    del uh, vh, wh
    t2 = time.perf_counter()
    speed2 = np.einsum("...i,...i->...", u, u, dtype=np.float64)
    row = {"index": idx, "t": t, "file": f"frame_{idx:04d}.vti",
           "step": int(round(t / dt)) if dt else None,
           "E_field": float(0.5 * speed2.mean()), "enstrophy_field": float(0.5 * np.mean(vort.astype(np.float64) ** 2)),
           "speed": stats_of("speed", np.sqrt(speed2, dtype=np.float32)),
           "vorticity_magnitude": stats_of("vorticity_magnitude", vort), "q_criterion": stats_of("q_criterion", q)}
    if check and t == 0.0:
        row["analytic_max_abs_err"] = analytic_check(u, vort, q, N)
    t3 = time.perf_counter()
    nbytes = write_vti(os.path.join(out, row["file"]), N, [("u", 3, u), ("vorticity_magnitude", 1, vort), ("q_criterion", 1, q)], level)
    t4 = time.perf_counter()
    row["bytes"] = nbytes
    row["seconds"] = {"load": round(t1 - t0, 2), "fields": round(t2 - t1, 2), "stats": round(t3 - t2, 2), "write": round(t4 - t3, 2), "total": round(t4 - t0, 2)}
    print(json.dumps({k: row[k] for k in ("index", "t", "bytes", "seconds", "E_field", "enstrophy_field")}), flush=True)
    return row


def write_pvd(path, rows):
    body = "\n".join(f'<DataSet timestep="{r["t"]!r}" group="" part="0" file="{r["file"]}"/>' for r in rows)
    open(path, "w", encoding="utf-8").write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
                          f'<Collection>\n{body}\n</Collection>\n</VTKFile>\n')


def parseval_check(rows, csv_path):
    import csv
    by_step = {int(r["step"]): r for r in csv.DictReader(open(csv_path, encoding="utf-8"))}
    worst = {"E": 0.0, "Omega": 0.0}
    for r in rows:
        ref = by_step.get(r["step"])
        if ref is None:
            continue
        E, Om = float(ref["E"]), float(ref["Omega"])
        r["E_csv"], r["Omega_csv"] = E, Om
        worst["E"] = max(worst["E"], abs(r["E_field"] - E) / E)
        worst["Omega"] = max(worst["Omega"], abs(r["enstrophy_field"] - Om) / Om)
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True, help="checkpoint prefix, e.g. build/n256_r8/tgvfused_256_8")
    ap.add_argument("--N", type=int, required=True); ap.add_argument("--ranks", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=None); ap.add_argument("--workers", type=int, default=3); ap.add_argument("--zlib-level", type=int, default=4)
    ap.add_argument("--energy-csv", default=None); ap.add_argument("--check-analytic", action="store_true")
    ap.add_argument("--times", nargs="*", type=float, default=None, help="subset of checkpoint times (default: all)")
    ap.add_argument("--delete-bins", action="store_true", help="remove each checkpoint's rank files once they are loaded")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    times = discover_times(a.prefix)
    if a.times is not None:
        times = [t for t in times if any(abs(t - s) < 1e-6 for s in a.times)]
    if not times:
        raise SystemExit(f"no {a.prefix}_ckpt_*_r0.bin")
    T0 = time.perf_counter()
    jobs = [(a.prefix, t, i, a.N, a.ranks, a.out, a.zlib_level, a.dt, a.check_analytic, a.delete_bins) for i, t in enumerate(times)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(do_frame, jobs))
    write_pvd(os.path.join(a.out, "frames.pvd"), rows)
    report = {"N": a.N, "ranks": a.ranks, "dt": a.dt, "spacing": 2 * np.pi / a.N, "frames": rows,
              "total_seconds": round(time.perf_counter() - T0, 1), "total_bytes": sum(r["bytes"] for r in rows), "workers": a.workers, "zlib_level": a.zlib_level}
    if a.energy_csv:
        report["parseval_max_rel_dev"] = parseval_check(rows, a.energy_csv)
    json.dump(report, open(os.path.join(a.out, "frames_stats.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({"FRAMES": len(rows), "seconds": report["total_seconds"], "bytes": report["total_bytes"], "parseval": report.get("parseval_max_rel_dev")}), flush=True)


if __name__ == "__main__":
    main()
