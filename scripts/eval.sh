#!/usr/bin/env bash
# eval.sh QUEUE [WAIT_LOG] -- the offline evaluation of a queue of models, one after another, each on all GPUs.
#   QUEUE: lines "LABEL SIZE MODEL_DIR". SIZE is 9b or 27b (the serving settings of scripts/serve.sh); MODEL_DIR is
#   ViRA-9B, ViRA-27B or the HF export of a checkpoint (.../global_step_N/actor/huggingface), served as "ck" on every
#   GPU. Per model, fastest first:
#     V* + HR-Bench 4K/8K at 1 MP; the contrast sets of BENCH_DIR (ViRA-Bench or ViRA-Bench-Real, versions O/A/B) at
#     0.5, 1, 2 and 4 MP; the benchmarks of GEN_DIR (MME-RealWorld-Lite, TreeBench, O3-Bench, MMStar, HallusionBench)
#     at 1 MP; V*/HR-Bench at the maximum; GEN_DIR at the maximum; the contrast sets at the maximum.
#   WAIT_LOG: start only after that training log has its "exit" line (the run is done and its GPUs are free).
#   ONLY="ext_1mp bench_0.5mp ...": run only these parts; DONE is then not written (PART_DONE_<host> instead).
# Env: BENCH_DIR (items.jsonl + images), EXT_DIR (items_ext.jsonl + images, from vira.eval.ext_items; skipped if
#   absent), GEN_DIR (items_gen.jsonl + images, from vira.eval.gen_items; skipped if absent), EVAL_DIR, CACHE (resized
#   views), WORKERS (episodes in flight), and serve.sh's VIRA_MODELS_DIR, VLLM_BIN, GPUS, PORT0.
# Results EVAL_DIR (default OUT_DIR/eval)/<LABEL>/{bench,ext,gen}_<budget>.jsonl (budget: 0.5mp, 1mp, 2mp, 4mp, maxmp),
# run_*.log, vllm_ck_<gpu>.log, timing.tsv, DONE (FAILED if the model could not be served); EVAL_DIR/full.log.
# Resumable: finished episodes are skipped by vira.eval.run, finished parts by their .done_* marker, finished models
# by DONE.
# Tables: python -m vira.eval.summarize EVAL_DIR
set -u
shopt -s nullglob
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE")
VIRA_HOME=${VIRA_HOME:-$HOME/vira}
VLLM_BIN=${VLLM_BIN:-$VIRA_HOME/env_vllm/bin}; PY=$VLLM_BIN/python
MODELS=${VIRA_MODELS_DIR:-$VIRA_HOME/models}
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
BENCH_DIR=${BENCH_DIR:-$VIRA_HOME/data/vira_bench}
EXT_DIR=${EXT_DIR:-$VIRA_HOME/eval/ext}
GEN_DIR=${GEN_DIR:-$VIRA_HOME/eval/gen}
EVAL_DIR=${EVAL_DIR:-${OUT_DIR:-$REPO/outputs}/eval}
CACHE=${CACHE:-/dev/shm/vira_eval_cache}
WORKERS=${WORKERS:-192}
Q=$(readlink -f "${1:?queue}")
serve() { bash "$HERE/serve.sh" "$@"; }
log() { echo "$(date -u '+%F %T') $*" >> "$EVAL_DIR/full.log"; }
mkdir -p "$EVAL_DIR" "$CACHE"
cp -n "$BENCH_DIR/items.jsonl" "$EVAL_DIR/items_bench.jsonl"
[ -f "$EXT_DIR/items_ext.jsonl" ] && cp -n "$EXT_DIR/items_ext.jsonl" "$EVAL_DIR/items_ext.jsonl"
[ -f "$GEN_DIR/items_gen.jsonl" ] && cp -n "$GEN_DIR/items_gen.jsonl" "$EVAL_DIR/items_gen.jsonl"
if [ -n "${2:-}" ]; then
  while ! grep -q "^exit" "$2" 2>/dev/null; do sleep 60; done
  log "training done: $2"; sleep 60
fi
URLS=$(serve urls)

run() {  # bench|ext|gen BUDGET
  local items=$BENCH_DIR/items.jsonl root=$BENCH_DIR s tag=$1_${2}mp
  [ "$1" = ext ] && { items=$EXT_DIR/items_ext.jsonl; root=$EXT_DIR; }
  [ "$1" = gen ] && { items=$GEN_DIR/items_gen.jsonl; root=$GEN_DIR; }
  [ -f "$items" ] || return
  [ -f "$OUT/.done_$tag" ] && return
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $tag "* ]] && return
  s=$(date +%s)
  "$PY" -m vira.eval.run --model ck --budget "$2" --items "$items" --data-root "$root" --urls "$URLS" \
      --cache "$CACHE" --out "$OUT/$tag.jsonl" --workers "$WORKERS" >> "$OUT/run_$tag.log" 2>&1 && touch "$OUT/.done_$tag"
  printf "%s\t%s\t%s\t%s\n" "$1" "$2" $(( $(date +%s) - s )) "$(hostname)" >> "$OUT/timing.tsv"
  log "$LABEL $tag $(( $(date +%s) - s ))s"
}

while read -r LABEL SIZE M <&3; do
  [ -z "${LABEL:-}" ] && continue
  OUT=$EVAL_DIR/$LABEL; mkdir -p "$OUT"; [ -f "$OUT/DONE" ] && continue
  case $SIZE in 9b) B=$MODELS/Qwen3.5-9B ;; 27b) B=$MODELS/Qwen3.8-27B ;; *) log "$LABEL: size $SIZE is not 9b or 27b"; continue ;; esac
  for f in "$B"/*.json "$B"/*.jinja; do  # verl's HF export lacks the image preprocessor (and maybe the chat template):
    n=$(basename "$f")                   # they are taken from the base model
    case $n in config.json|model.safetensors.index.json) continue ;; esac
    [ -e "$M/$n" ] || cp "$f" "$M/"
  done
  serve stop; log "serving $LABEL ($M)"
  LOG_DIR=$OUT serve start "$SIZE" "$M" ck
  if ! serve wait ck; then log "FAILED to serve $LABEL"; touch "$OUT/FAILED"; serve stop; continue; fi
  run ext 1
  for b in 0.5 1 2 4; do run bench $b; done
  run gen 1
  run ext max
  run gen max
  run bench max
  if [ -n "${ONLY:-}" ]; then touch "$OUT/PART_DONE_$(hostname)"; log "$LABEL part done ($ONLY)"
  else touch "$OUT/DONE"; log "$LABEL done"; fi
done 3< "$Q"
serve stop; log "queue done"
