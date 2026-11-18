#!/usr/bin/env bash
# vLLM servers for vira.eval.run and vira.infer: one OpenAI-compatible server per GPU, port PORT0 + gpu index,
# 127.0.0.1 only.
#   serve.sh start 9b|27b [MODEL_DIR [NAME]]  serve MODEL_DIR (default: ViRA-9B / ViRA-27B under VIRA_MODELS_DIR) as
#                                             NAME (default: the directory name) on every GPU
#   serve.sh wait NAME                        block until every server answers as NAME (exit 1 after 30 min)
#   serve.sh stop | status | urls             urls: the --urls argument of vira.eval.run
# 9b | 27b is the size of the model, for the settings below; a checkpoint of a training run is served the same way,
# e.g. serve.sh start 9b /path/to/global_step_N/actor/huggingface ck.
# Settings: --generation-config vllm (plain greedy, without the models' own sampling defaults such as
# repetition_penalty 1.05); --allowed-local-media-path / (images come as file:// URLs from the data and view cache
# dirs); 128k context and up to 10 images per request. A 27B model (bf16, about 56 GB) on one 80 GB GPU gets more
# memory for weights + KV cache and at most 128 sequences (its linear-attention layers keep one cache block per
# running sequence).
# Env: VIRA_MODELS_DIR, VLLM_BIN (the vLLM env's bin), CUDA_HOME (nvcc for vLLM's JIT kernels), GPUS (default: all),
# PORT0 (8100), LOG_DIR (server logs vllm_<NAME>_<gpu>.log).
set -u
VIRA_HOME=${VIRA_HOME:-$HOME/vira}
VLLM_BIN=${VLLM_BIN:-$VIRA_HOME/env_vllm/bin}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$VLLM_BIN:$CUDA_HOME/bin:$PATH
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader)}
PORT0=${PORT0:-8100}
LOG_DIR=${LOG_DIR:-$VIRA_HOME/logs}

case "${1:-status}" in
start)
  case ${2:?9b | 27b} in
    9b)  default=ViRA-9B;  mem=0.90; seqs="" ;;
    27b) default=ViRA-27B; mem=0.94; seqs="--max-num-seqs 128" ;;
    *) echo "model size: 9b | 27b" >&2; exit 2 ;;
  esac
  dir=${3:-${VIRA_MODELS_DIR:-$VIRA_HOME/models}/$default}
  name=${4:-$(basename "$dir")}
  mkdir -p "$LOG_DIR"
  for g in $GPUS; do
    port=$((PORT0 + g))
    if curl -sf "localhost:$port/v1/models" >/dev/null; then echo "gpu $g :$port already up"; continue; fi
    CUDA_VISIBLE_DEVICES=$g nohup setsid vllm serve "$dir" --served-model-name "$name" --port $port --host 127.0.0.1 \
      --max-model-len 128000 --limit-mm-per-prompt '{"image": 10}' --allowed-local-media-path / \
      --gpu-memory-utilization $mem --generation-config vllm --seed 0 $seqs \
      > "$LOG_DIR/vllm_${name}_$g.log" 2>&1 < /dev/null &
    echo "gpu $g -> $name :$port ($dir)"
  done ;;
wait)
  name=${2:?served name}
  for _ in $(seq 120); do
    n=0
    for g in $GPUS; do curl -sf "localhost:$((PORT0 + g))/v1/models" | grep -q "\"$name\"" && n=$((n + 1)); done
    [ $n -eq $(echo $GPUS | wc -w) ] && exit 0
    sleep 15
  done
  exit 1 ;;
stop)  # a server started here leads its own process group (setsid): kill the group (API server + engine core)
  pids=""
  for g in $GPUS; do
    port=$((PORT0 + g))
    pid=$(ss -ltnp 2>/dev/null | grep ":$port " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    [ -n "$pid" ] || continue
    if [ "$(ps -o pgid= -p "$pid" | tr -d ' ')" = "$pid" ]; then kill -- -"$pid"; else kill "$pid"; fi
    pids="$pids $pid"; echo "stopped :$port (pid $pid)"
  done
  for _ in $(seq 60); do [ -n "$pids" ] && kill -0 $pids 2>/dev/null || break; sleep 2; done ;;
status)
  for g in $GPUS; do
    port=$((PORT0 + g))
    name=$(curl -sf "localhost:$port/v1/models" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    echo "gpu $g :$port ${name:-down}"
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader ;;
urls)
  u=""; for g in $GPUS; do u="$u,http://127.0.0.1:$((PORT0 + g))/v1"; done; echo "${u#,}" ;;
*) echo "usage: serve.sh start 9b|27b [MODEL_DIR [NAME]] | wait NAME | stop | status | urls" >&2; exit 2 ;;
esac
