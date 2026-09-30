# Platform rules the kernels obey

Rules of the Neuron hardware, runtime and NKI compiler that this solver depends on, each with the measurement that
established it and the place in this repository that follows it. Platform: NKI 0.6, neuronx-cc 2.27, Neuron SDK
2.32, fp32 with `--auto-cast=none`, backend optimisations disabled; NeuronCore-v2 on Inferentia2 (inf2.xlarge,
inf2.24xlarge) and Trainium1 (trn1.2xlarge, trn1.32xlarge). None of these rules is in the Neuron documentation;
each cost a measurement on a paid instance. Where a rule was established on a kernel that is not part of this
sample (the distributed statevector simulator built alongside it), the measurement is cited and the sample states
how it complies.

## 1. Every collective of a time step lives in one NEFF

**Rule.** Two NEFFs that each carry an `all_to_all` and alternate on the same NeuronCores compute wrong physics at
8 ranks over four Inferentia2 chips, nondeterministically. A host barrier between every execute does not cure it.
One NEFF with one `all_to_all` is bitwise reproducible; one NEFF holding two `all_to_all` calls with distinct
destination buffers is exact.

**Measurement.** inf2.24xlarge, 256^3, 8 ranks, checked row by row against the 32-rank Trainium1 run. Two-NEFF
stage (phase a with one all-to-all, phase b with one all-to-all): with a host barrier after every execute, peak
dissipation 0.015257 at t = 9.388 and errors of 9.0e-2 in E and 3.6e-1 in eps; without the barrier 0.017621 at
8.946 and 1.7e-1 / 6.3e-1; five runs gave five different wrong answers, and 384^3 and 512^3 reached NaN. Phase a
alone in a loop of 330 executions: every readback bitwise identical to iteration 0 on every rank (59 comparisons per
rank of 12,582,912 floats). The single-NEFF stage: 128^3 and 256^3 identical to the reference to 9 digits, three
256^3 runs bitwise identical, 512^3 within 1.75e-7 with the same peak, value and time, and the same E at t = 10.
The fault does not appear at 2 or 4 ranks on the same instance or at 32 ranks on trn1.32xlarge.

**In this repository.** `kernels/tgv_fused_wide_nki.py`, kernel `tgv_fused`: the whole RK3 stage with both
`ncc.all_to_all([s4a], [r4a], grp, 0)` and `ncc.all_to_all([s4b], [r4b], grp, 0)` in one NEFF;
`driver/tgv_fused.c` executes that one NEFF three times per step. The RK constants are inputs so that one NEFF
serves the three stages.

**Caveat.** What the runtime shares between two collective-bearing models on one core is not identified. The
compiler places pre-collective temporaries inside the receive buffer's extent by liveness (352 MiB scratch at 256^3,
8 ranks); this is harmless in every single-NEFF run measured.

## 2. No staging buffer is shared by two collectives

**Rule.** A rank's send lands in a peer's receive buffer as soon as the sender enters the collective, before the
peer has posted it: entering an `all_to_all` is not a rendezvous on the receive side. No two collectives of a step
share a destination, and no buffer a peer may still be reading is the destination of a collective a faster rank
can reach first. With distinct destinations, program order on each rank is the only ordering needed.

**Measurement.** Distributed statevector exchange with two all-to-all calls per chunk through the same send and
receive pair: exact at 2 and 4 ranks, wrong at 8 ranks (norms 0.9994 to 1.002, changing run to run) in every
variant tried. With four buffers (send1, recv1, send2, recv2) the same exchange is exact at 8 ranks with 16.8 and
33.6 MB per peer per call and 64 calls per layer. Payload size is refuted as the cause: the payloads that went
wrong with a shared pair (12.6 and 100 MB per peer) bracket payloads that are exact with distinct buffers.

**In this repository.** `tgv_fused` allocates `bufs = [s4a, r4a, s4b, r4b]`: (R, 12, L, C2) send and receive
buffers for the inverse transforms and (R, 6, L, C2) for the forward transforms, exact at 8 ranks up to 512^3
(100 MB per peer per call). `_pin` reads one tile of every staging buffer at kernel entry and exit and sums them
into the guard output, so the four buffers' live ranges span the whole kernel.

## 3. DMA-written resident SBUF state stays under 64 KB per partition

**Rule.** The compiler places every DMA destination in the top 96 KB of the 192 KB partition and carves the
rotating pool for streamed tiles out of what that half has left. When the DMA-written footprint (long-lived tiles
plus streaming slots plus DMA'd temporaries) crosses the 96 KB line, a streamed operand gets one or two SBUF slots
and every tile pays a full DMA round trip. Compute-written long-lived state (a `tensor_copy` destination) does not
count. The working budget is 64 KB per partition of DMA-written resident data.

**Measurement.** trn1.2xlarge, one core, an fp32 kernel streaming a matrix against resident state written by DMA,
microseconds per iteration against resident bytes per partition: 32 KB 2,752; 64 KB 2,798; 72 KB 2,809; 80 KB
2,804; 88 KB 2,823; 96 KB 11,201 (a 4.0x cliff); 128 KB 8,256. At 16,384 rows: 64 KB 10,982; 96 KB 40,608; 128 KB
42,436. In the profile of a slow variant the streamed operand's slot reuse distance is 1, DMA issue waits 2.3 to
2.5 us for the previous matmul to release the slot, and DMA engines idle 60 to 71 percent, with HBM bytes equal to
the algorithm's. Routing the DMA into a transient tile and `tensor_copy` into the long-lived tile restores 3,839 us
at 128 KB and 3,002 at 96 KB with bit-identical output.

**In this repository.** The sizing globals of `kernels/tgv_fused_wide_nki.py`: `WIDE_EL = 2048` (8 KB per
partition per x-pass DMA), `PLANE_EL = 1024` (4 KB per plane-pass DMA), `ELEM_EL = 512`. At 256^3 the widest
points are the plane pass at 24 KB, the x pass at 48 KB and the cross product at 36 KB per partition, with 6 KB of
resident DFT matrices; the RK constant tiles are widened by SBUF copies, so they are compute-written.

## 4. Inferentia2 collectives run inside a four-chip group; 12 ranks are refused

**Rule.** On Inferentia2 the collective library builds an `all_to_all` for groups of 2, 4 or 8 ranks inside one
four-chip group. A 12-rank group on the six-chip inf2.24xlarge chain and a 16-rank group on the inf2.48xlarge ring
are refused, and so is a strided group that spans four-chip groups.

**Measurement.** inf2.24xlarge (`neuron-ls` shows a chain: device 0 connects to 1 only, 5 to 4 only), 384^3 on
12 ranks: `alltoall cannot be supported without Mesh algorithm, check if Mesh algorithm is supported on the current
replica group size`, then `Failed to post op 0 for function 0` on every device. inf2.48xlarge (12 chips): 8 ranks on
cores 0 to 7 run; 16 ranks fail at execute with `Failed to build and load collectives resources`; a two-round
exchange of groups of 8 then pairs (r, r + 8) fails with `no_hier no_mesh replica-group [4,12]`. The 8-rank group
moves about 6.5 GB/s per direction per core.

**In this repository.** `scripts/run.sh` and the validation contract use 2 ranks on an inf2.xlarge and 8 on an
inf2.24xlarge (`ReplicaGroup((tuple(range(R)),))` in `tgv_fused`, one group of all ranks). 12 ranks on the
inf2.24xlarge are not a supported configuration of this sample.

## 5. Trainium1 accepts one group of 32 ranks; strided cross-chip groups are refused

**Rule.** On trn1.32xlarge (16 chips, 32 cores, 2-D torus) a single `all_to_all` replica group of all 32 ranks
builds and runs at any op count tried (128 collective ops in one NEFF). A group whose ranks are strided across chips
({r, r + 8, r + 16, r + 24}) is refused at load.

**Measurement.** Statevector exchange at 32 ranks: exact and bitwise repeatable at 33 to 35 qubits, 25.7 to
27.5 GB/s per direction per core. The strided second round is refused with `Failed to post op 64 for function 0`
and `Failed to build and load collectives resources`. This solver: 32-rank all-to-all at 128^3, 256^3 and 512^3
(the reference CSVs in `references/`), 512^3 at 241 ms per step.

**In this repository.** The same one-group `ReplicaGroup` serves R = 32 on Trainium1; `export/export_fused.py`
takes any R that divides N and N^2.

## 6. Cores ramp after the runtime's reset; keep the reset in a forked multi-rank driver

**Rule.** `nrt_init` resets the NeuronCores by default and for a few seconds afterwards the compute engines run
at a lower effective rate, about 2.2x slower per NEFF; the ramp is consumed by wall time, not by executions, and
does not restart on idle. A single-process driver may set `NEURON_RT_RESET_CORES=0` and skip it. A fork-per-rank
collective driver keeps the reset: with it skipped, a core still owned by a killed process fails `nrt_init` and the
other ranks wait forever in the collectives bootstrap.

**Measurement.** A 16.6 ms NEFF executed 300 to 400 times from a fresh process: with the default reset min /
median / max 16.659 / 17.498 / 37.016 ms on inf2 and 16.562 / 16.587 / 29.666 on trn1; with the reset skipped
16.589 / 16.609 / 16.647 and 16.566 / 16.584 / 16.612. Profiles along the recovery: 36.9 ms at execution 20, 26.4 at
60, 20.6 at 80, 17.7 at 100 with identical HBM bytes and a Tensor engine active time that falls from 31.8 to 10.9 ms.
In this solver the first steps of a run are slower for the same reason (64^3: first 200 steps 9.86 ms average
against a 5.4 ms median; 256^3 at 8 ranks: 183.8 ms over the first 20 steps against 121.1 steady), and a rank whose
core was owned by a killed process fails `nrt_init` with `cores busy, ret -16` when the reset is skipped.

**In this repository.** `scripts/env.sh` unsets `NEURON_RT_RESET_CORES`, so both drivers run with the default
reset; `tools/check.py` reports the median step time, which the ramp does not move, and the p90, which it does.

## 7. Wide, loads-first DMA bodies; no power-of-two partition strides

**Rule.** Two DMA facts of NeuronCore-v2 under NKI 0.6. (a) neuronx-cc ends every `nl.fori_loop` iteration with a
drain of all DMA queues and a barrier of all engines, so cross-iteration double buffering is impossible; a body
moves many tiles per DMA, holds several chunks, and issues all its loads before its first compute. Every DMA is
software-issued on the Pool queue with no queue or engine hint. (b) A power-of-two partition stride runs a DMA pass at
about half the rate of a contiguous one; staging rows are padded so the stride is not a power of two.

**Measurement.** Streaming matmul, microseconds per tile: unrolled tile by tile 3.18; rolled with one tile per body
11.94; rolled with 16 tiles per body 5.69; two chunks per body with loads first 3.32. In this solver the same
change (plane passes moving 1024/N planes per DMA with every K-chunk loaded before the first matmul, x passes moving
2048 columns per DMA, transforms reading and writing the collective buffers directly) is bitwise identical in its
results and 13.2 percent faster at 256^3 on 2 ranks (175.78 to 152.63 ms per step), 10.1 percent at 256^3 on 8 ranks
(82.05 to 73.73) and 21 percent at 128^3 on 2 ranks; the source-level DMA count per stage at 256^3 falls from 41,107
to 12,307 and the bytes from 7,972 to 5,092 MB. The stride effect: a statevector exchange pass over 8 GB fell from
1,708 to 743 ms with row padding of 512 elements and eight tiles per body.

**In this repository.** `kernels/tgv_fused_wide_nki.py`: the accessors `_planes_hbm`, `_planes_staged`,
`_rows_hbm`, `_rows_staged` feeding `_rows_pass_wide` and `_dist_pass_wide`, with `DIRECT = 1` (no pack or
unpack copies). The stage kernel is fully unrolled (Python `range`), so (a)'s loop boundary does not arise in it; the
kernels of this sample do not use power-of-two partition strides.

## 8. `neuronx-cc` on PATH, `NEURON_PLATFORM_TARGET_OVERRIDE=trn1`, `--auto-cast=none`

**Rule.** The NKI compile driver resolves its backend `neuronx-cc` from PATH; invoking the venv's Python by
absolute path is not enough. A standalone `@nki.jit` compiles for Trainium3 by default and the runtime refuses the
artifact ("NEFF arch: v4, instance arch: v2"); `NEURON_PLATFORM_TARGET_OVERRIDE=trn1` selects the NeuronCore-v2
target for both Trainium1 and Inferentia2. fp32 kernels compile with `--auto-cast=none`. `neuronx-cc` refuses a
non-empty artifacts directory (NCC_IDRV077).

**Measurement.** Without the venv's `bin` on PATH the driver invokes its backend as `python None compile ...` and
the failure surfaces as `NCCError: neuronx-cc compilation failed with exit code 2` with `can't open file
'<artifacts>/None'`; it reads as a kernel error and is not one (two instances were consumed before the argv named
the cause). With the target override, NKI 0.6 compiles and runs on NeuronCore-v2 and the 1-D DFT matches NumPy to
1.6e-7.

**In this repository.** `scripts/env.sh` prepends the venv's `bin` and `/opt/aws/neuron/bin` to PATH and exports
`NEURON_PLATFORM_TARGET_OVERRIDE=trn1` and `NEURON_CC_FLAGS=--auto-cast=none`; `export/export_neff.py` reads the
target from that variable, compiles with `TracerFrontend(enable_backend_opt=False)` and
`CompileOptions(...).disable_backend_optimizations()`, and removes the `art-<name>/` directory before each compile.

## 9. The NKI frontend accepts a subset of Python

**Rule.** The compiler's specialisation pass rejects tuple targets in `for` loops ("expecting simple variable"),
generator expressions and comprehensions ("unsupported expression"), and closures (grid size, rank count and stage
constants travel as module globals). The direct-call path (`kernel(arrays)` on the device) additionally rejects
inner functions. Dict and list subscript assignment, `if` on trace-time values and helper functions are accepted.
`nl.mod` is accepted by the CPU simulator and rejected by `neuronx-cc` at the ISA check (NCC_IXCG863); the same
applies to other operators absent from the ALU table.

**Measurement.** Kernels that passed the simulator failed at the hardware compile until rewritten; after the
rewrite the RK3 stage matched the fp64 oracle to 2.0e-9 at 64^3 after one step. The on-device NumPy unit test of the
wide DMA passes could not run under the direct-call frontend; the bitwise comparison of the NEFF against the baseline
NEFF is the exactness test.

**In this repository.** `N_GLOBAL` and `R_RANKS` in the distributed kernel modules, set by `export/export_fused.py`
before tracing; `STAGE_A`, `STAGE_B`, `STAGE_DT`, `STAGE_DECAY` in `kernels/tgv_stage_nki.py`, set per stage by
`export/export_stages.py`; every kernel is exported through `TracerFrontend.compile`, never called directly.

## 10. The rank count divides N and N^2; a tensor holds at most 2 GB

**Rule.** Each rank holds L = N / R planes and C2 = N^2 / R spectral columns; an R that does not divide N
truncates the layout and the NEFF fails to load its collectives resources. A single HBM tensor holds at most 2 GB
(2^29 fp32 elements): DMA byte offsets are 32-bit and a 4 GB tensor aborts the DMA.

**Measurement.** 12 ranks at 128^3 and 256^3 (L = 10, C2 = 1365 after truncation): collectives resources fail to
load. A 2^30-element fp32 tensor: `TX_DATA_AXI_RESPONSE_ERROR` on every queue and `NRT_EXEC_HW_ERR_DMA_ABORT` after
the 30 s timeout, while 2 GB and 3 GB tensors run.

**In this repository.** `export/export_fused.py` asserts `N % R == 0 and (N * N) % R == 0 and N <= 512`. The
largest tensor of this sample is the 512^3 8-rank receive buffer at 8 x 12 x 64 x 32768 fp32 = 805 MB, under the
limit.

## 11. Forked libnrt ranks: flush before `_exit`, watchdog, one bootstrap port per run

**Rule.** A multi-rank driver is one process per NeuronCore, forked from a parent, rank r pinned with
`NEURON_RT_VISIBLE_CORES=r`, collectives loaded with `nrt_load_collectives`. Ranks leave through `_exit`, which
does not flush stdio, so a rank flushes before `nrt_close`. The parent supervises: a rank exiting non-zero, or no
progress by step 20 within 240 s, kills every live rank. Each run uses its own collectives bootstrap address through
`NEURON_RT_ROOT_COMM_ID`: a killed run leaves its port in TIME_WAIT for 60 s and the next run's rank 0 cannot bind.

**Measurement.** Forked ranks that `_exit` lose their buffered stdout. A rank killed mid-initialisation leaves its
core busy (`Logical Neuron Core(s) not available ... cores busy`) and the next process recovers it only with the
core reset (rule 6). A run started within 60 s of a killed one fails with `Failed to bind(127.0.0.1<46820>), error:
Address already in use`, `ncclInitGlobalComm failed` and ranks `awaiting root parameters`, which reads like a
collective failure. An exporter process (`import nki`) also holds device handles while it compiles, so an export
overlapping a run gives the same "cores busy".

**In this repository.** `driver/tgv_fused.c`: `run_rank` ends with `fflush(stdout); nrt_close();` before the
child's `_exit`; `main` polls `waitpid(..., WNOHANG)` every 100 ms against a shared-memory progress counter and
kills every live rank with SIGKILL on `"a rank failed"` or `"watchdog: no step 20 within 240 s"`; the driver sets
`NEURON_RT_ROOT_COMM_ID` only when the caller has not, and `scripts/run.sh` sets it to `127.0.0.1:<46820 + random>`
for every run. `scripts/run.sh` exports the NEFFs before the run starts, never alongside one.
