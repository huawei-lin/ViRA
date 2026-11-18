#!/usr/bin/env bash
# CrossRPO training of the ViRA zoom agent on verl (v1 trainer, commit 174888f), one node, for Qwen3.5-9B and
# Qwen3.8-27B.
#   EXP=<run name> TRAIN=<training parquet> STEPS=<steps> bash scripts/train.sh [hydra overrides]
# scripts/launch.sh sets these variables for the runs of the paper. The reward is the correctness of the answer.
# Smoke test: STEPS=2 SAVE=-1 X_LO=4 X_HI=4 (a fixed pixel budget).
# rollout.load_format=auto is required: with verl's default "dummy" start, vLLM 0.32's FusedMMInputNorm buffers
# (visual.input_norm.weight/bias = 1/std, -mean/std) are randomized and the param-only weight sync never restores
# them, so vLLM sees broken pixels.
set -euo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
VIRA_HOME=${VIRA_HOME:-$HOME/vira}
OUT=${OUT_DIR:-$CODE/outputs}                # checkpoints, logs, dumps, eval_runs, run_cfg
VERL=${VERL_DIR:-$VIRA_HOME/verl}
PATCH=16; XMAX=16.777216                     # Qwen3.5 / Qwen3.8: 16-px patches (x 2 merge), at most 16.78 MP per image
MODEL=${MODEL:-${VIRA_MODELS_DIR:-$VIRA_HOME/models}/Qwen3.5-9B}
EXP=${EXP:?EXP = run name, <base>__<method>__<data>__seed<k>, e.g. Qwen3.5-9B__CrossRPO__train-9b__seed42}
PROJECT=${PROJECT:-ViRA}                 # project name of the logger (WANDB_ENTITY from the environment)
SEED=${SEED:-42}                         # budget sampling, rollout and data seeds (the item order is fixed in the parquet)
SHUFFLE=${SHUFFLE:-False}                # the training parquets are pre-shuffled and used once in order
CROSSRPO_LAMBDA=${CROSSRPO_LAMBDA:-0.5}  # lambda, the weight of the joint term
JUDGE_KEY_FILE=${JUDGE_KEY_FILE:-}       # file with the OpenRouter key of the GPT-4o judge of open answers (mode 600);
                                         # empty: the judge reads OPENROUTER_API_KEY from its environment
TRAIN_TURN_CAP=${TRAIN_TURN_CAP:-4096}   # token cap of each reply in training
EVAL_TURN_CAP=${EVAL_TURN_CAP:-4096}     # token cap of each reply in the validation inside training
TRAIN=${TRAIN:?TRAIN = training parquet (python -m vira.hf train)}
# validation inside training (vira/eval/args.py EvalArguments): on when EVAL_DATA (contrast-set items, jsonl) is set;
# step 0 (EVAL_ON_START), every EVAL_STEPS steps and the last step, greedy, budgets EVAL_BUDGETS (MP)
EVAL_DATA=${EVAL_DATA:-}
EVAL_IMAGE_ROOT=${EVAL_IMAGE_ROOT:-$(dirname "${EVAL_DATA:-.}")}
EVAL_EXTRA_PARQUET=${EVAL_EXTRA_PARQUET:-}   # the single images of the validation set (a pre-built parquet), optional
EVAL_STEPS=${EVAL_STEPS:-20}
EVAL_ON_START=${EVAL_ON_START:-true}
EVAL_BUDGETS=${EVAL_BUDGETS:-"0.5 4"}
EVAL_BATCH=${EVAL_BATCH:-256}
STEPS=${STEPS:?STEPS = number of training steps (launch.sh: rows of TRAIN // BATCH, one pass over the data)}
SAVE=${SAVE:-20}
BATCH=${BATCH:-16}               # items (contrast sets or single images) per step; x 12 rollouts = 192 rollouts
N=${N:-12}                       # rollouts per item, split evenly over its K versions (n = 12 / K per version)
PL=${PL:-20480}                  # prompt: first view <= 16,384 tokens + text
RL=${RL:-40960}                  # response: text + up to 6 crops
MAXTOK=${MAXTOK:-$(( (PL + RL) / ${SP:-1} ))}   # dynamic-bsz token budget per GPU; verl multiplies it by SP, and
                                  # budget x SP must hold the longest sequence
X_LO=${X_LO:-0.5}                # pixel budget x in MP: log-uniform on [X_LO, X_HI], one draw per item
X_HI=${X_HI:-$XMAX}
GPU_UTIL=${GPU_UTIL:-0.55}
NGPU=${NGPU:-8}                  # with CUDA_VISIBLE_DEVICES for part of a node
# large models (Qwen3.8-27B): ROLLOUT_TP (vLLM tensor parallel), PARAM_OFFLOAD / OPTIM_OFFLOAD (FSDP CPU offload),
# SP (Ulysses sequence parallel); defaults keep the 9B setting
LOGGER=${LOGGER:-"['console','wandb']"}
# checkpoints: model + optimizer + extra state (exact resume); SAVE_OPTIMIZER=0 leaves the optimizer out (smaller; a
# resume then restarts Adam's moments)
CKPT_SAVE="['model','optimizer','extra','hf_model']"; CKPT_LOAD="['model','optimizer','extra']"
[ "${SAVE_OPTIMIZER:-1}" = 0 ] && { CKPT_SAVE="['model','extra','hf_model']"; CKPT_LOAD="['model','extra']"; }
CKPT=${CKPT:-$OUT/ckpt/$EXP}
DUMP=${DUMP:-$OUT/dumps/$EXP}
EVAL_OUT=${EVAL_OUT:-$OUT/eval_runs/$EXP}   # eval_dataset.parquet, eval/step_N.jsonl, eval_history.jsonl
mkdir -p "$OUT/logs" "$OUT/tmp" "$CKPT" "$DUMP" "$OUT/run_cfg/$EXP"

# per-run copy of the agent loop config with this run's budget range, seed, reply cap and judge key file
sed -e "s/^  x_lo: .*/  x_lo: $X_LO/" -e "s/^  x_hi: .*/  x_hi: $X_HI/" -e "s/^  x_seed: .*/  x_seed: $SEED/" \
    -e "s/^  train_turn_cap: .*/  train_turn_cap: $TRAIN_TURN_CAP/" \
    -e "s|^  judge_key_file: .*|  judge_key_file: \"$JUDGE_KEY_FILE\"|" \
    "$CODE/vira/train/config/agent_loop.yaml" > "$OUT/run_cfg/$EXP/agent_loop.yaml"

export TMPDIR=$OUT/tmp RAY_TMPDIR=/tmp/ray_vira PYTHONPATH=$CODE:${PYTHONPATH:-}
export VLLM_USE_V1=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-${EXP%%__*}}   # group = base model
[ -n "${WANDB_TAGS:-}" ] && export WANDB_TAGS
cd "$VERL"
source .venv/bin/activate

if [ -n "$EVAL_DATA" ]; then
  EVAL_TXT=$(python -m vira.eval.args --eval_dataset "$EVAL_DATA" --eval_image_root "$EVAL_IMAGE_ROOT" \
    --eval_output_dir "$EVAL_OUT" --eval_steps "$EVAL_STEPS" --eval_on_start "$EVAL_ON_START" \
    --eval_batch_size "$EVAL_BATCH" --eval_budgets_mp $EVAL_BUDGETS --eval_max_new_tokens "$EVAL_TURN_CAP" \
    ${EVAL_EXTRA_PARQUET:+--eval_extra_parquet "$EVAL_EXTRA_PARQUET"})
  mapfile -t EVAL_OVR <<< "$EVAL_TXT"
else
  EVAL_OVR=(data.val_files="$TRAIN" trainer.val_before_train=False trainer.test_freq=-1)
fi

# adv_estimator=grpo: verl then computes the advantage per set of rollouts with one call, which the trainer replaces
# with the CrossRPO advantage (vira.train.trainer). rollout_correction.bypass_mode: the log-probabilities of the
# rollout (vLLM) are the old log-probabilities of the clipped loss (no recompute pass)
python -m vira.train.main \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.rollout_correction.bypass_mode=true \
    data.train_files="$TRAIN" \
    data.train_batch_size=$BATCH \
    data.max_prompt_length=$PL \
    data.max_response_length=$RL \
    data.filter_overlong_prompts=False \
    data.image_patch_size=$PATCH \
    data.shuffle=$SHUFFLE \
    data.seed=$SEED \
    data.dataloader_num_workers=4 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAXTOK \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-False} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIM_OFFLOAD:-False} \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP:-1} \
    actor_rollout_ref.actor.use_torch_compile=False \
    "actor_rollout_ref.actor.checkpoint.save_contents=$CKPT_SAVE" \
    "actor_rollout_ref.actor.checkpoint.load_contents=$CKPT_LOAD" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-1} \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.seed=$SEED \
    actor_rollout_ref.actor.data_loader_seed=$SEED \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_model_len=$((PL + RL)) \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.limit_images=8 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAXTOK \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=vira \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=6 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=7 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CODE/vira/train/config/tool_config.yaml" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$OUT/run_cfg/$EXP/agent_loop.yaml" \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    actor_rollout_ref.rollout.agent.default_agent_loop=vira_agent \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=vira_sync \
    trainer.critic_warmup=0 \
    trainer.logger="$LOGGER" \
    trainer.project_name=$PROJECT \
    trainer.experiment_name="$EXP" \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.balance_batch=True \
    "${EVAL_OVR[@]}" \
    trainer.save_freq=$SAVE \
    trainer.total_training_steps=$STEPS \
    trainer.total_epochs=100 \
    trainer.default_local_dir="$CKPT" \
    trainer.resume_mode=auto \
    +vira.crossrpo_lambda=$CROSSRPO_LAMBDA \
    +vira.dump_dir="$DUMP" \
    +vira.prune_old_ckpt=True \
    +vira.keep_hf_last=${KEEP_HF_LAST:-0} \
    +vira.keep_full_last=${KEEP_FULL_LAST:-2} \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$CODE" \
    "$@"
