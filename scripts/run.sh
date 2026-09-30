#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Run the Re = 1600 Taylor-Green vortex on grid N with R NeuronCores to t = T_END and write build/n<N>_r<R>/tgv_<N>_<R>.csv.
# Case: u0 = 1, box 2 pi, nu = 1/1600, 2/3-rule dealiasing, Williamson RK3, dt = 0.5 * 2 pi / N, statistics every step.
# R = 1 uses the three single-core stage NEFFs and driver/tgv_run; R > 1 the fused stage NEFF and driver/tgv_fused (one
# process per NeuronCore, all-to-all collectives inside the kernel) and merges the per-rank CSVs. NEFFs are exported on
# first use into the build directory.
#   scripts/run.sh <N> <R> [T_END] [TAG]      TAG is appended to the CSV name (a second run for the bitwise repeat)
# CKPT_EVERY=<steps> (R > 1) makes the driver dump the spectral state per rank at t = 0 and every CKPT_EVERY steps for
# tools/vtk_frames.py.
set -euo pipefail
N="${1:?grid}"; R="${2:?ranks}"; TEND="${3:-10}"; TAG="${4:-}"; NU=0.000625
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; BUILD="$ROOT/build/n${N}_r${R}"; mkdir -p "$BUILD"; cd "$BUILD"
command -v neuronx-cc >/dev/null || { echo "run.sh: neuronx-cc not on PATH; source scripts/env.sh first" >&2; exit 1; }
DT=$(python3 -c "import math; print(0.5 * 2 * math.pi / $N)")
STEPS=$(python3 -c "import math; print(math.ceil($TEND / (0.5 * 2 * math.pi / $N)))")
echo "run.sh: N=$N R=$R steps=$STEPS dt=$DT nu=$NU t_end=$TEND build=$BUILD"
if [ "$R" -eq 1 ]; then
  if [ ! -f "tgv_stage2_${N}.neff" ]; then
    echo "run.sh: exporting the three stage NEFFs for N=$N ($(date -u +%FT%TZ))"
    python3 "$ROOT/export/export_stages.py" "$N" 2> >(grep -viE "warn|deprecat" >&2 || true) | tail -1 | tee export.log
  fi
  export NEURON_RT_VISIBLE_CORES=0
  "$ROOT/driver/tgv_run" "$N" "$STEPS" "$DT" "$NU" "tgv_${N}_${R}${TAG}.csv" 2> "run${TAG}.err" | tee "run${TAG}.log"
  tail -2 run.err || true
else
  if [ ! -f "tgv_fused_${N}_${R}.neff" ]; then
    echo "run.sh: exporting dist_fwd3 and tgv_fused for N=$N R=$R ($(date -u +%FT%TZ))"
    python3 "$ROOT/export/export_fused.py" "$N" "$R" 2> >(grep -viE "warn|deprecat" >&2 || true) | tee export.log
  fi
  export NEURON_RT_ROOT_COMM_ID="127.0.0.1:$((46820 + RANDOM % 1000))"   # one collectives bootstrap port per run
  PFX="tgvfused_${N}_${R}${TAG}"; rm -f "${PFX}"_r*.csv
  "$ROOT/driver/tgv_fused" "$N" "$R" "$STEPS" "$DT" "$NU" 1 "$PFX" "${CKPT_EVERY:-0}" 2> "run${TAG}.err" | tee "run${TAG}.log"
  grep -vE "^step " "run${TAG}.err" | tail -3 || true
  python3 "$ROOT/tools/merge.py" "$PFX" "$R" "$NU" "tgv_${N}_${R}${TAG}.csv" | tee "merge${TAG}.log"
fi
