# Method

## The case

Incompressible Navier-Stokes in a periodic box of side 2 pi, Taylor-Green initial field u = sin x cos y cos z,
v = -cos x sin y cos z, w = 0 (u0 = 1), viscosity nu = 1/1600. The solver is Fourier collocation on N^3 points:
the spectral velocity is advanced in rotational form, du_hat/dt = P(u x omega)_hat - nu k^2 u_hat, with P the Leray
projection and the product formed in physical space and dealiased with the 2/3 rule. Time integration is the
Williamson low-storage third-order Runge-Kutta scheme (a = 1/3, 15/16, 8/15; b = 0, -5/9, -153/128), with the
viscous term integrated exactly by multiplying the spectrum with exp(-nu k^2 dt) once per step. The time step is
dt = 0.5 (2 pi / N). Every step the driver reads the spectral velocity back and writes kinetic energy
E = 0.5 sum |u_hat|^2 / N^6, enstrophy Omega = 0.5 sum k^2 |u_hat|^2 / N^6 and eps = 2 nu Omega; the check compares
these to the shipped references and closes the energy budget -dE/dt = eps on every step. `tools/tgv_reference.py`
is the NumPy statement of the same algorithm and `tools/oracle.py` produces the fp64 reference CSVs.

## Transforms as matmuls

Neuron has no FFT instruction and no fp64. A real DFT of length N along one axis is a pair of N x N matmuls with the
cosine and sine matrices (C, S), applied to the N^2 columns of a 2-D view of the field; complex data travels as
(re, im) fp32 pairs and a complex product is four matmuls accumulating pairwise in PSUM. The Tensor engine
contracts over the partition axis, so each pass uses the data as the stationary operand and the DFT matrix as the
moving operand: `dst[m, i'] = sum_i src[i, m] W[i, i']`. The result lands transposed, which is used rather than
undone: three passes cycle the layout through the axes and no transpose instruction is issued. A forward transform
returns the spectrum as [kx, kz, ky]; the same three passes with the conjugate matrices and a 1/N^3 scale return to
x, y, z. The solver keeps its spectral state and wavenumber constants in the permuted layout (`kernels/dft3d_perm_nki.py`).

At N = 128 a pass is one 128-partition stationary tile against 128 x 512 moving tiles, the shape the Tensor engine
is built for. For N > 128 the contraction over N rows is chunked into 128-row K-tiles accumulated in PSUM
(`kernels/dft3d_dist_nki.py`, `_rows_pass` and `_dist_pass`), which is what makes 256 and 512 work; the moving free
size caps N at 512.

## One RK3 stage on one NeuronCore

`kernels/tgv_stage_nki.py` is one stage as one kernel: spectral vorticity omega_hat = i k x u_hat (elementwise, the
multiply by i is a real/imaginary swap), six inverse transforms (velocity and vorticity to physical space), the
cross product u x omega, three forward transforms, dealiasing, projection, the low-storage update
du = b du + dt f, u_hat += a du, and on the third stage the viscous decay. The stage constants are module globals
read at trace time (the NKI frontend rejects closures), so `export/export_stages.py` compiles three NEFFs, one per
stage, and `driver/tgv_run.c` executes them in sequence with the twelve state and accumulator tensors resident on
the device in two ping-pong sets. Only the six spectral state tensors are read back per step, for the statistics.
Grids 32 to 128 (N a multiple of 32).

## R NeuronCores: slabs and one all-to-all per phase

For R ranks the box is split into x-slabs: rank r holds planes x in [rL, (r+1)L), L = N/R, and the spectrum in the
rank layout [kx (all), n] with n indexing the rank's chunk of C2 = N^2/R (ky, kz) pairs. A forward transform is two
local permuting passes per plane, a pack into (R, 6, L, C2) blocks by destination rank holding all six (re, im)
components of the three fields, one `nki.collectives.all_to_all`, and a layout-preserving pass along x; the inverse
mirrors it (`kernels/dft3d_dist3_nki.py`, `dist_fwd3`). Wavenumber constants are built per rank in the rank layout
by the driver.

The whole stage is one NEFF, `tgv_fused` in `kernels/tgv_fused_wide_nki.py`: phase a is the curl and the six
inverse transforms behind one all-to-all; phase b is the cross product, the three forward transforms behind a second
all-to-all, then dealias, project and update. The two collectives use four distinct staging buffers (send and receive
for each), and the RK constants are input tiles so that one NEFF serves the three stages. Two collective-bearing
NEFFs alternating on a core compute wrong physics at 8 ranks, so the stage is not split further; the reasons and
measurements are in [neuron-rules.md](neuron-rules.md). The kernel's DMA shapes are chosen for the hardware:
plane passes move 1024/N planes per DMA with every K-chunk loaded before the first matmul, x passes move 2048
columns per DMA, and the transform passes read and write the collective staging buffers directly with no pack or
unpack copies. SBUF per partition at the widest point (256^3) stays under the 64 KB budget for DMA-resident data.

`driver/tgv_fused.c` forks one process per rank pinned to its NeuronCore, loads the initial transform and the stage
NEFF with `nrt_load_collectives`, runs the initial transform from the analytic field on the device, barriers once,
and then executes the stage NEFF three times per step. Each rank writes partial sums of E and Omega for its block;
`tools/merge.py` sums them into the run CSV. The parent supervises the ranks: a rank exiting non-zero or no
progress within 240 s kills the group. Ranks leave through `_exit`, so the driver flushes its output first. The
collectives bootstrap address comes from `NEURON_RT_ROOT_COMM_ID`, which `scripts/run.sh` sets to a fresh port per
run because a killed run leaves its port in TIME_WAIT for 60 s.

Rank counts: R must divide N and N^2. On Inferentia2, collectives run inside a group of four chips (8 ranks on an
inf2.24xlarge or inf2.48xlarge, 2 on an inf2.xlarge or inf2.8xlarge); Trainium1 accepts 32 ranks on a trn1.32xlarge.
