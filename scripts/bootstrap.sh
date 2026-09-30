#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# One command from a fresh Neuron Deep Learning AMI (Ubuntu 24.04, Neuron SDK with NKI 0.6) to a checked run:
#   ./scripts/bootstrap.sh            # 64^3 on one NeuronCore, about 3 minutes on an inf2.xlarge
#   GRID=128 RANKS=1 ./scripts/bootstrap.sh
# Builds the driver, exports the NEFFs on the box, runs the case to t = 10 and checks it against the shipped reference.
# Prints the CHECK lines and exits non-zero when any criterion fails. The log is kept in build/bootstrap.log.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; mkdir -p build
exec > >(tee -a build/bootstrap.log) 2>&1
echo "=== bootstrap start $(date -u +%FT%TZ) GRID=${GRID:-64} RANKS=${RANKS:-1}"
# shellcheck disable=SC1091
. scripts/env.sh || exit 1
neuron-ls 2>/dev/null | grep -E "^\|" || echo "neuron-ls unavailable"
make check GRID="${GRID:-64}" RANKS="${RANKS:-1}" T_END="${T_END:-10}"; rc=$?
echo "=== bootstrap end $(date -u +%FT%TZ) exit=$rc"
exit "$rc"
