#!/usr/bin/env bash
ray stop
set -euxo pipefail
export TORCH_CUDA_ARCH_LIST="9.0"  # 根据你的GPU改，A100是8.0，H100是9.0

project_name='open-o3-video-verl-bzy'
exp_name='grpo_openo3_reproduce_q3reward_new'

adv_estimator=grpo
use_kl_in_reward=True
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 16))
max_response_length=$((1024 * 1))
# 影响rollout prefill，对速度影响不大，会影响kv cache大小
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 1))

sp_size=1
# actor_ppo_max_token_len*sp_size作为max_token_len
# / sp_size 能影响update和backward的memory
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))
# 作为计算log_prob的max_token_len，和sp_size无关 影响forward update_actor, 长prompt时只能用1
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 4))

## Reinforce-Ada setting
n_resp_per_prompt=4
loss_agg_mode="seq-mean-token-mean"


train_prompt_mini_bsz=4
train_prompt_bsz=$((train_prompt_mini_bsz * 4))


# Paths
base_path="$(pwd)"
model_path=/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/Open-o3-Video-verl/ckpts/open-o3-video-verl-bzy/grpo_openo3_reproduce_q3reward/global_step_2326/actor_hf
ckpts_dir=/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/Open-o3-Video-verl/ckpts/${project_name}/${exp_name}
rollout_dir=${base_path}/.cache/${project_name}/${exp_name}/rollout
val_dir=${base_path}/.cache/${project_name}/${exp_name}/val


export DATA_ROOT="/mnt/bn/strategy-mllm-train/common/datasets/Open-o3-Video-data/videos"
train=/mnt/bn/strategy-mllm-train/common/datasets/Open-o3-Video-data/json_data/STGR-RL.json
# train=/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/datasets/open-o3-video-data/sample_val.json

eval=/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/datasets/open-o3-video-data/sample_val.json

train_files="['$train']"
eval_files="['$eval']"

custom_reward_function_path=/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/Open-o3-Video-verl/verl/verl/utils/reward_score/openo3_video_reward.py
custom_reward_function_name=compute_score

# Algorithm
temperature=1.0
top_p=0.95
top_k=20 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.8
val_temperature=0.1

# Mathematically equivalent 
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null

# performance - offload 分开控制
# 4B模型8卡FSDP分片后每卡参数约1GB，optimizer states约2GB，完全放得下
# 如果OOM，按顺序加回: optimizer_offload -> activation_offload -> param_offload
param_offload=False
optimizer_offload=True
activation_offload=True

strategy=fsdp2
# vllm自动计算rollout_batch, 设置超过该值时会造成构造rollout llm时使用default max_num_seqs=1024时OOM
gpu_memory_utilization=0.75
# vllm的max_num_seqs
max_num_seqs=512

export VLLM_ATTENTION_BACKEND=FLASHINFER
export PYTHONWARNINGS="ignore::FutureWarning"

cd /mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/Open-o3-Video-verl/verl

python3 -u -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$eval_files" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.image_patch_size=16 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_batch_normalize=True \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.norm_adv_by_std_in_grpo=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=${activation_offload} \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=3.0 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
    actor_rollout_ref.actor.strategy=${strategy} \
    actor_rollout_ref.actor.fsdp_config.offload_policy=${param_offload} \
    actor_rollout_ref.ref.strategy=${strategy} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=5.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=naive \
    reward_model.launch_reward_fn_async=False \
    custom_reward_function.path=${custom_reward_function_path} \
    custom_reward_function.name=${custom_reward_function_name} \
    trainer.logger=['wandb','console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=2000 \
    trainer.save_freq=400 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$ckpts_dir" \
    trainer.rollout_data_dir="$rollout_dir" \
    trainer.validation_data_dir="$val_dir" \
    trainer.balance_batch=True \
    trainer.resume_mode=auto