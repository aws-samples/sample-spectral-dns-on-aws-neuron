# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Compile a bare @nki.jit kernel to a .neff file. Framework-free; the compile needs no
device (neuronx-cc cross-compiles), only execution does.

Working recipe (NKI 0.6.0, verified on trn1 2026-09-17):
    from nki._backends.context import nki_ir_context
    from nki.compiler.frontend import TracerFrontend
    from nki.compiler.ncc_driver import CompileOptions, CompiledKernel
    with nki_ir_context() as ctx:
        res = TracerFrontend(enable_backend_opt=False).compile(
            ctx, kernel, inputs=bound.arguments, target="trn1", lnc=1, artifacts_dir=art)
        CompiledKernel.from_frontend(
            res, CompileOptions(target="trn1", lnc=1, output_path=neff,
                                artifacts_dir=art).disable_backend_optimizations())

The kernel is an ordinary @nki.jit function: inputs annotated nl.ndarray, output allocated
with buffer=nl.shared_hbm and returned. The output tensor is named output_0 in the NEFF.
Two dead ends found and avoided: nki.compiler.parallel_compile / KernelSpec is a different
out-parameter batch API whose kernel convention is finicky; and the standalone kernel(np)
call runs on device but writes no NEFF to disk (it compiles in memory).
"""
import inspect, os, sys
import numpy as np
import nki, nki.language as nl, nki.isa as nisa


@nki.jit
def matmul_kernel(lhsT: nl.ndarray, rhs: nl.ndarray) -> nl.ndarray:
    # C[M,N] = lhsT[K,M].T @ rhs[K,N]; single 128-tile, fp32.
    K, M = lhsT.shape
    _, N = rhs.shape
    out = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
    a = nl.ndarray((K, M), dtype=nl.float32, buffer=nl.sbuf)
    b = nl.ndarray((K, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=lhsT)
    nisa.dma_copy(dst=b, src=rhs)
    p = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=p, stationary=a, moving=b)
    o = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=p)
    nisa.dma_copy(dst=out, src=o)
    return out


def export(kernel, sample_inputs, out_path, target=None, lnc=1):
    from nki._backends.context import nki_ir_context
    from nki.compiler.frontend import TracerFrontend
    from nki.compiler.ncc_driver import CompileOptions, CompiledKernel
    target = target or os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    import shutil
    art = os.path.join(os.path.dirname(out_path) or ".", "art-" + os.path.splitext(os.path.basename(out_path))[0])
    shutil.rmtree(art, ignore_errors=True)   # neuronx-cc refuses a non-empty artifacts dir (NCC_IDRV077)
    os.makedirs(art, exist_ok=True)
    with nki_ir_context() as ctx:
        fe = TracerFrontend(enable_backend_opt=False)
        bound = inspect.signature(kernel.func).bind(*sample_inputs)
        res = fe.compile(ctx, kernel, inputs=bound.arguments, target=target, lnc=lnc, artifacts_dir=art)
        opts = CompileOptions(target=target, lnc=lnc, output_path=out_path,
                              artifacts_dir=art).disable_backend_optimizations()
        CompiledKernel.from_frontend(res, opts)
    return out_path if os.path.exists(out_path) else None


if __name__ == "__main__":
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "matmul.neff")
    K, M, N = 128, 64, 128
    ins = (np.zeros((K, M), np.float32), np.zeros((K, N), np.float32))
    p = export(matmul_kernel, ins, out)
    print("WROTE" if p else "FAILED", p, (os.path.getsize(p) if p else 0), "bytes")
    sys.exit(0 if p else 2)
