#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Solve with checkpoints, turn them into VTK frames and render an isosurface animation with OSMesa ParaView (pvbatch):
#   scripts/render_hero.sh <N> <R> <ckpt_every> <paraview_dir> [T_END] [ISO2]
# paraview_dir is an unpacked "ParaView-5.13.x-osmesa-MPI-Linux-Python3.x-x86_64" tree (the headless build from
# paraview.org). Output: build/n<N>_r<R>/frames/*.vti and build/n<N>_r<R>/png/frame_*.png; assemble a video on any machine
# with ffmpeg: tools/make_video.sh build/n<N>_r<R>/png <out_dir> 8 <name>.
# Disk: each checkpoint holds the six spectral blocks of every rank (6 N^3 fp32, 3.2 GB at 512^3); the rank files are
# deleted as soon as a frame is written. Memory per conversion worker is about 20 N^3 bytes (11 GB at 512^3).
set -euo pipefail
N="${1:?grid}"; R="${2:?ranks}"; EVERY="${3:?ckpt_every}"; PV="${4:?paraview_dir}"; TEND="${5:-10}"; ISO2="${6:-1.5}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; B="$ROOT/build/n${N}_r${R}"; PFX="$B/tgvfused_${N}_${R}anim"
DT=$(python3 -c "import math; print(0.5 * 2 * math.pi / $N)")
WORKERS=$(python3 -c "import os; print(max(1, min(6, (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')) // (24 * $N ** 3))))")
echo "render_hero: N=$N R=$R ckpt_every=$EVERY t_end=$TEND workers=$WORKERS iso2=$ISO2 ($(date -u +%FT%TZ))"
make -C "$ROOT/driver" -s
CKPT_EVERY="$EVERY" "$ROOT/scripts/run.sh" "$N" "$R" "$TEND" anim
echo "render_hero: frames ($(date -u +%FT%TZ))"
python3 "$ROOT/tools/vtk_frames.py" --prefix "$PFX" --N "$N" --ranks "$R" --out "$B/frames" --dt "$DT" --workers "$WORKERS" \
  --delete-bins --check-analytic --energy-csv "$B/tgv_${N}_${R}anim.csv" | tail -1
echo "render_hero: render ($(date -u +%FT%TZ))"
LD_LIBRARY_PATH="$PV/lib/mesa:$PV/lib" "$PV/bin/pvbatch" "$ROOT/tools/render_frames.py" "$B/frames/frames.pvd" "$B/frames/frames_stats.json" "$B/png" \
  --iso2 "$ISO2" --subtitle "Taylor-Green vortex, Re 1600, ${N}^3 pseudo-spectral DNS on $R AWS Neuron cores" 2>&1 | tail -2
echo "render_hero: done ($(date -u +%FT%TZ)); PNGs in $B/png"
