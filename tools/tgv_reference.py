# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""NumPy reference for the pseudo-spectral Taylor-Green vortex (the oracle for the NKI
solver). Incompressible Navier-Stokes in a 2 pi periodic box, Fourier collocation,
2/3-rule dealiasing, low-storage RK3, exact integration of the viscous term.
The same arithmetic order will be mirrored by the NKI kernels so that the two agree
to fp32 round-off step by step. Run: python tgv_reference.py 64 100
"""
import sys, time
import numpy as np

def make_grid(N):
    x = 2 * np.pi * np.arange(N) / N
    return np.meshgrid(x, x, x, indexing="ij")

def tgv_init(N, u0=1.0):
    X, Y, Z = make_grid(N)
    u = u0 * np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -u0 * np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)
    return u.astype(np.float32), v.astype(np.float32), w.astype(np.float32)

def wavenumbers(N):
    k = np.fft.fftfreq(N, d=1.0 / N).astype(np.float32)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0  # avoid division by zero for the mean mode
    dealias = (np.abs(KX) < N / 3) & (np.abs(KY) < N / 3) & (np.abs(KZ) < N / 3)
    return KX, KY, KZ, K2, dealias

def rhs(uh, vh, wh, KX, KY, KZ, K2, dealias):
    """Nonlinear term in rotational form, projected: dU/dt = P(u x omega)."""
    u = np.fft.ifftn(uh).real; v = np.fft.ifftn(vh).real; w = np.fft.ifftn(wh).real
    ox = np.fft.ifftn(1j * (KY * wh - KZ * vh)).real
    oy = np.fft.ifftn(1j * (KZ * uh - KX * wh)).real
    oz = np.fft.ifftn(1j * (KX * vh - KY * uh)).real
    fx = np.fft.fftn(v * oz - w * oy) * dealias
    fy = np.fft.fftn(w * ox - u * oz) * dealias
    fz = np.fft.fftn(u * oy - v * ox) * dealias
    div = (KX * fx + KY * fy + KZ * fz) / K2
    return fx - KX * div, fy - KY * div, fz - KZ * div

def kinetic_energy(uh, vh, wh, N):
    return float(0.5 * (np.sum(np.abs(uh)**2 + np.abs(vh)**2 + np.abs(wh)**2)) / N**6)

def run(N=64, steps=100, nu=1.0 / 1600, dt=None):
    u, v, w = tgv_init(N)
    KX, KY, KZ, K2, dealias = wavenumbers(N)
    uh, vh, wh = np.fft.fftn(u), np.fft.fftn(v), np.fft.fftn(w)
    dt = dt or 0.5 * (2 * np.pi / N)  # CFL ~0.5 at u0 = 1
    a = [1/3, 15/16, 8/15]; b = [0, -5/9, -153/128]  # Williamson low-storage RK3
    ke = [kinetic_energy(uh, vh, wh, N)]
    t0 = time.perf_counter()
    for _ in range(steps):
        du = dv = dw = 0.0
        for s in range(3):
            fx, fy, fz = rhs(uh, vh, wh, KX, KY, KZ, K2, dealias)
            du = b[s] * du + dt * fx; dv = b[s] * dv + dt * fy; dw = b[s] * dw + dt * fz
            uh = uh + a[s] * du; vh = vh + a[s] * dv; wh = wh + a[s] * dw
        decay = np.exp(-nu * K2 * dt).astype(np.float32); decay[0, 0, 0] = 1.0
        uh *= decay; vh *= decay; wh *= decay
        ke.append(kinetic_energy(uh, vh, wh, N))
    wall = time.perf_counter() - t0
    return np.array(ke), dt, wall

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    ke, dt, wall = run(N, steps)
    print(f"N={N} steps={steps} dt={dt:.4f} t_end={steps*dt:.3f} ke0={ke[0]:.6f} ke_end={ke[-1]:.6f} "
          f"ke_ratio={ke[-1]/ke[0]:.6f} ms_per_step={1000*wall/steps:.1f}")
    # early-time laminar check: for t << 1 the TGV kinetic energy follows exp(-2 nu * 3 t) at k^2 = 3
    lam = float(np.exp(-2 * (1.0/1600) * 3 * steps * dt))
    print(f"laminar_decay_reference={lam:.6f} (valid only while the flow stays laminar)")
