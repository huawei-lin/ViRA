#!/usr/bin/env bash
# Install verl (commit 174888f) with FSDP + vLLM into $VERL_DIR/.venv (uv), then causal-conv1d (Qwen3.5) against that
# torch. Needs uv and a CUDA toolkit (CUDA_HOME). Training uses this environment (scripts/train.sh).
set -euo pipefail
VERL=${VERL_DIR:-${VIRA_HOME:-$HOME/vira}/verl}
CUDA_HOME=${CUDA_HOME:?CUDA_HOME = a CUDA toolkit matching the torch build}
[ -d "$VERL/.git" ] || git clone -q https://github.com/verl-project/verl.git "$VERL"
cd "$VERL"
git fetch -q origin 174888f
git checkout -q 174888f
uv python install 3.12
uv sync --frozen --python 3.12 --extra fsdp --extra vllm
source .venv/bin/activate
export PATH=$CUDA_HOME/bin:$PATH
uv pip install ninja packaging setuptools wheel
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=${MAX_JOBS:-32} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0} \
  uv pip install --no-build-isolation "causal-conv1d==1.6.1"
python -c "import torch, vllm, transformers; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__)"
