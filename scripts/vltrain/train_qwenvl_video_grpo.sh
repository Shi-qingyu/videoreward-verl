#!/usr/bin/env bash
set -euxo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
verl_root="${repo_root}/verl"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-${repo_root}/data/train_region.json}"
EVAL_FILE="${EVAL_FILE:-${repo_root}/data/train_region.json}"

PROJECT_NAME="${PROJECT_NAME:-qwenvl-video-verl}"
EXP_NAME="${EXP_NAME:-grpo-local}"
CKPT_DIR="${CKPT_DIR:-${repo_root}/ckpts/${PROJECT_NAME}/${EXP_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${repo_root}/.cache/${PROJECT_NAME}/${EXP_NAME}/rollout}"
VAL_DIR="${VAL_DIR:-${repo_root}/.cache/${PROJECT_NAME}/${EXP_NAME}/val}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-4}"

TRAIN_GPUS="${TRAIN_GPUS:-1}"
NNODES="${NNODES:-1}"

ACC_REWARD_WEIGHT="${ACC_REWARD_WEIGHT:-1.0}"
FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-1.0}"
IOU_REWARD_WEIGHT="${IOU_REWARD_WEIGHT:-1.0}"

CUSTOM_DATASET_PATH="${repo_root}/verl/verl/utils/dataset/qwenvl_video_rl_dataset.py"
CUSTOM_REWARD_PATH="${repo_root}/verl/verl/utils/reward_score/qwenvl_video_grpo_reward.py"

cd "${verl_root}"

python3 -u -m verl.trainer.main_ppo \
    data.train_files="['${TRAIN_FILE}']" \
    data.val_files="['${EVAL_FILE}']" \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.custom_cls.path="${CUSTOM_DATASET_PATH}" \
    data.custom_cls.name=QwenVLVideoRLDataset \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size=null \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.grad_clip=5.0 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    reward_model.reward_manager=naive \
    reward_model.launch_reward_fn_async=False \
    custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    custom_reward_function.name=compute_score \
    custom_reward_function.reward_kwargs.acc_weight=${ACC_REWARD_WEIGHT} \
    custom_reward_function.reward_kwargs.format_weight=${FORMAT_REWARD_WEIGHT} \
    custom_reward_function.reward_kwargs.iou_weight=${IOU_REWARD_WEIGHT} \
    trainer.logger="['console']" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${TRAIN_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    trainer.validation_data_dir="${VAL_DIR}" \
    trainer.resume_mode=auto
