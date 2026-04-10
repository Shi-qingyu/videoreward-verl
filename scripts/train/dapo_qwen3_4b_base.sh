#!/usr/bin/env bash
set -euxo pipefail

project_name='verl_math_qwen3_test'
exp_name='qwen3_14b_base_32k'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.01
kl_loss_type=mse

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 32))
# 影响rollout prefill，对速度影响不大，会影响kv cache大小
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 1))
sp_size=8

# actor_ppo_max_token_len*sp_size作为max_token_len
# / sp_size 能影响updae和backward的memory
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
# 作为计算log_prob的max_token_len，和sp_size无关 影响forward update_actor, 3x容易oom
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=0.5

# dr.grpo loss
loss_agg_mode="seq-mean-token-sum-norm"

# filter_groups
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
truncate_sample=False
train_prompt_mini_bsz=16
# orz_grpo=128
train_prompt_bsz=$((train_prompt_mini_bsz * 4))
# gen_prompt_bsz=$(awk "BEGIN { printf \"%.0f\", $train_prompt_bsz * 1.2 }")
gen_prompt_bsz=320
# orz_grpo=64 
n_resp_per_prompt=8

# Paths
base_path=/opt/tiger/live_strategy_posttrain
train_file=${base_path}/data/dapo-math-17k.parquet
test_file=${base_path}/data/aime-2024.parquet
model_path=/opt/tiger/Qwen3-4B-Instruct-2507
ckpts_dir=${base_path}/checkpoints/${project_name}/${exp_name}
rollout_dir=${base_path}/.cache/${project_name}/${exp_name}/rollout
val_dir=${base_path}/.cache/${project_name}/${exp_name}/val

reasoning_train_path=${base_path}/data/rl_reasoning_data_hard.parquet
aime_test_path=${base_path}/data/rl_eval_aime24.parquet
test_files="['$aime_test_path']"
custom_reward_function_path=${base_path}/reward_system/reward.py
custom_reward_function_name=reward_fn

# Algorithm
temperature=1.0
top_p=0.8
top_k=20 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.8
val_temperature=0.7

# Mathematically equivalent
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null

# performance
offload=True
strategy=fsdp2
# vllm自动计算rollout_batch, 设置超过该值时会造成构造rollout llm时使用default max_num_seqs=1024时OOM
gpu_memory_utilization=0.8
max_num_seqs=128

# data.filter_overlong_prompts_workers=10 \

cd verl
python3 -u -m recipe.dapo.main_dapo \
    data.train_files="$reasoning_train_path" \
    data.val_files="$aime_test_path" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.truncate_sample=${truncate_sample} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=${offload} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.strategy=${strategy} \
    actor_rollout_ref.actor.fsdp_config.offload_policy=${offload} \
    actor_rollout_ref.ref.strategy=${strategy} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
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
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    custom_reward_function.path=${custom_reward_function_path} \
    custom_reward_function.name=${custom_reward_function_name} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=1 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$ckpts_dir" \
    trainer.rollout_data_dir="$rollout_dir" \
    trainer.validation_data_dir="$val_dir" \
    trainer.balance_batch=False \
    trainer.resume_mode=disable
