#!/usr/bin/env bash
# launch.sh RUN [hydra overrides] -- the training runs of the paper. A run makes one pass over its training data
# (16 items per step, 12 rollouts per item, seed 42), with the reward, prompts and hyperparameters of scripts/train.sh,
# and is validated inside training (step 0, every EVAL_STEPS steps, last step).
#   vira_9b    ViRA-9B:  CrossRPO (lambda = 0.5) on Qwen3.5-9B
#   vira_27b   ViRA-27B: CrossRPO (lambda = 0.5) on Qwen3.8-27B (vLLM tensor parallel 2, sequence parallel 2)
# CROSSRPO_LAMBDA=<value> trains with another weight of the joint term (0: no joint term, every version is trained as
# a separate single image); the run name then ends in -lam<value>.
# Data in DATA_DIR, as `python -m vira.hf train` writes them from ViRA-Train: the training parquet (train_9b.parquet /
# train_27b.parquet; one row per contrast set or single image), the validation items dev.jsonl (contrast sets, images
# under EVAL_IMAGE_ROOT) and dev_single.parquet (the single images of the validation set).
# A run started again resumes from its latest checkpoint and keeps logging to the same run (id next to the
# checkpoints). Other variables are passed through to train.sh, e.g. SAVE_OPTIMIZER=0.
CODE=$(cd "$(dirname "$0")/.." && pwd)
VIRA_HOME=${VIRA_HOME:-$HOME/vira}
DATA=${DATA_DIR:-$VIRA_HOME/data}
export OUT_DIR=${OUT_DIR:-$CODE/outputs}
RUN=${1:?vira_9b | vira_27b}
PY=${VERL_DIR:-$VIRA_HOME/verl}/.venv/bin/python
case $RUN in
  vira_9b)  BASE=Qwen3.5-9B;  SIZE=9b;  export GPU_UTIL=${GPU_UTIL:-0.55} ;;
  vira_27b) BASE=Qwen3.8-27B; SIZE=27b; export GPU_UTIL=${GPU_UTIL:-0.5} ROLLOUT_TP=${ROLLOUT_TP:-2} SP=${SP:-2} ;;
  *) echo "unknown run $RUN" >&2; exit 1 ;;
esac
export MODEL=${MODEL:-${VIRA_MODELS_DIR:-$VIRA_HOME/models}/$BASE}
export CROSSRPO_LAMBDA=${CROSSRPO_LAMBDA:-0.5}
METHOD=CrossRPO; [ "$CROSSRPO_LAMBDA" = 0.5 ] || METHOD=CrossRPO-lam$CROSSRPO_LAMBDA
export TRAIN=${TRAIN:-$DATA/train_$SIZE.parquet} BATCH=16
DATA_TAG=${DATA_TAG:-train-$SIZE}
# every row once: steps = rows // batch
export STEPS=${STEPS:-$("$PY" -c "import pyarrow.parquet as pq; print(pq.read_metadata('$TRAIN').num_rows // $BATCH)")}
export EXP=${BASE}__${METHOD}__${DATA_TAG}__seed42${EXP_SUFFIX:-}
export SAVE=${SAVE:-20} NGPU=${NGPU:-8} SEED=42 WANDB_JOB_TYPE=train
export WANDB_TAGS="$BASE,$METHOD,data=$DATA_TAG,steps=$STEPS,seed=42"
export EVAL_DATA=${EVAL_DATA:-$DATA/dev.jsonl} EVAL_IMAGE_ROOT=${EVAL_IMAGE_ROOT:-$DATA/vira_train} \
       EVAL_EXTRA_PARQUET=${EVAL_EXTRA_PARQUET:-$DATA/dev_single.parquet} \
       EVAL_STEPS=${EVAL_STEPS:-20} EVAL_BUDGETS="${EVAL_BUDGETS:-0.5 4}"
mkdir -p "$OUT_DIR/ckpt/$EXP" "$OUT_DIR/logs"
ID=$OUT_DIR/ckpt/$EXP/wandb_run_id
[ -s "$ID" ] || python3 -c "import secrets; print(secrets.token_hex(4))" > "$ID"
export WANDB_RUN_ID=$(cat "$ID") WANDB_RESUME=allow
LOG=$OUT_DIR/logs/$EXP.log
[ -f "$LOG" ] && mv "$LOG" "$OUT_DIR/logs/$EXP.$(date -u +%m%d-%H%M).log"
bash "$CODE/scripts/train.sh" "${@:2}" > "$LOG" 2>&1      # extra args = Hydra overrides, e.g. trainer.val_only=True
echo "exit $?" >> "$LOG"
