# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# Source this once per shell on a Neuron Deep Learning AMI (Ubuntu): activates the Neuron Python venv that carries
# NKI and neuronx-cc, puts the compiler on PATH, and sets the compile target and flags the kernels are validated with.
#   NEURON_PLATFORM_TARGET_OVERRIDE=trn1   NKI compiles for Trainium3 by default; Inferentia2 and Trainium1 are NeuronCore-v2.
#   NEURON_CC_FLAGS=--auto-cast=none       keep every operation fp32; the checks assume no automatic downcast.
_venv=""
for _v in /opt/aws_neuronx_venv_jax_* /opt/aws_neuronx_venv_pytorch_*; do
  if [ -x "$_v/bin/python" ] && "$_v/bin/python" -c "import nki" 2>/dev/null; then _venv="$_v"; break; fi
done
if [ -z "$_venv" ]; then echo "env.sh: no /opt/aws_neuronx_venv_* with the nki package found; install the Neuron SDK first" >&2; return 1 2>/dev/null || exit 1; fi
# shellcheck disable=SC1091
. "$_venv/bin/activate"
export PATH="$_venv/bin:/opt/aws/neuron/bin:$PATH"
export LD_LIBRARY_PATH="/opt/aws/neuron/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NEURON_PLATFORM_TARGET_OVERRIDE=trn1
export NEURON_CC_FLAGS="--auto-cast=none"
unset NEURON_RT_VISIBLE_CORES NEURON_RT_RESET_CORES
echo "env.sh: venv $_venv, neuronx-cc $(command -v neuronx-cc || echo MISSING), nki $(python -c 'import nki; print(getattr(nki, "__version__", "?"))' 2>/dev/null)"
unset _venv _v
