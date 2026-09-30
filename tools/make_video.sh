#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Laptop: PNG frames -> H.264 MP4 (8 fps, yuv420p, faststart) and an 8-frame contact sheet.
# Usage: make_video.sh <png_dir> <out_dir> [fps] [mp4_basename] [contact_sheet_name]
set -euo pipefail
PNG="$1"; OUT="$2"; FPS="${3:-8}"; NAME="${4:-tgv_re1600_256_neuron}"; SHEET="${5:-contact_sheet.png}"; mkdir -p "$OUT"
N=$(ls "$PNG"/frame_*.png | wc -l | tr -d ' ')
ffmpeg -y -loglevel error -framerate "$FPS" -i "$PNG/frame_%04d.png" -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart "$OUT/$NAME.mp4"
# contact sheet: 8 frames evenly spaced from the first to the last, 4 x 2, each 640 px wide (the time caption is burnt in by the render)
CS=$(mktemp -d); for i in 0 1 2 3 4 5 6 7; do IDX=$(( i * (N - 1) / 7 )); ln -s "$(cd "$PNG" && pwd)/$(printf 'frame_%04d.png' "$IDX")" "$CS/$(printf 'cs_%02d.png' "$i")"; done
ffmpeg -y -loglevel error -framerate 1 -i "$CS/cs_%02d.png" -vf "scale=640:-1,tile=4x2:padding=4:margin=4:color=0x0A0B12" -frames:v 1 -update 1 "$OUT/$SHEET"; rm -rf "$CS"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,nb_frames,r_frame_rate -show_entries format=duration,size -of default=noprint_wrappers=1 "$OUT/$NAME.mp4"
ls -la "$OUT"
