#!/usr/bin/env bash
set -euxo pipefail

project_name='react_exp'
exp_name='v1_8_recipe_b200_gpt'

adv_estimator=rloo_vectorized

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.001
kl_loss_type=mse

clip_ratio_low=0.2
clip_ratio_high=0.285

max_prompt_length=$((1024 * 80))
max_response_length=$((1024 * 16))
# 影响rollout prefill，对速度影响不大，会影响kv cache大小
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 1))
sp_size=4

# actor_ppo_max_token_len*sp_size作为max_token_len
# / sp_size 能影响updae和backward的memory，会影响update速度
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / 2))
# 作为计算log_prob的max_token_len，和sp_size无关 影响forward update_actor, 长prompt时只能用1
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / 2))

## Reinforce-Ada setting
n_resp_per_prompt=16
loss_agg_mode="token-mean"
multiround_adaptive_downsampling=False
reinforce_ada_choice="balanced" # "positive_focused" or "balanced"
global_stat_est=True
positive_threshold=0.5
round_repeat=$((n_resp_per_prompt * 2))
max_rounds=2


train_prompt_mini_bsz=24
train_prompt_bsz=$((train_prompt_mini_bsz * 4))

# Paths
base_path=/opt/tiger/live_strategy_posttrain
model_path=/opt/tiger/gpt-oss-20b-bf16
ckpts_dir=${base_path}/checkpoints/${project_name}/${exp_name}
rollout_dir=${base_path}/.cache/${project_name}/${exp_name}/rollout
val_dir=${base_path}/.cache/${project_name}/${exp_name}/val

summary_bench_2507=${base_path}/data/summary_bench_2507_eval_v2.parquet
analysis_bench_2507=${base_path}/data/analysis_bench_2507_eval_v2.parquet
react_claude_bench_2507=${base_path}/data/react_claude_bench_2507_react_eval.parquet
eval_files="['$summary_bench_2507', '$analysis_bench_2507', '$react_claude_bench_2507']"

summary_train_2506=${base_path}/data/summary_train_2506_train.parquet
summary_bench_2504=${base_path}/data/summary_bench_2504_train.parquet
summary_user_2507=${base_path}/data/summary_train_2507_user_train_v2.parquet
summary_user_2508=${base_path}/data/summary_train_2508_user_train_v2.parquet
single_turn_gemini_0813=${base_path}/data/single_turn_gemini_0813_train_v2.parquet
# react_0818=${base_path}/data/react_gemini_0818_train.parquet
react_0819=${base_path}/data/react_gpt_0819_react_train.parquet
train_files="['$summary_train_2506', '$summary_bench_2504', '$summary_user_2507', '$summary_user_2508', '$single_turn_gemini_0813', '$react_0819']"

custom_reward_function_path=${base_path}/reward_system/reward.py
custom_reward_function_name=batch_analysis_reward_fn

# Algorithm
temperature=0.8
top_p=0.95
val_top_p=0.95
val_temperature=0.8

# Mathematically equivalent
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null

# performance
offload=True
strategy=fsdp2
# vllm自动计算rollout_batch, 设置超过该值时会造成构造rollout llm时使用default max_num_seqs=1024时OOM
gpu_memory_utilization=0.8
max_num_seqs=384

export MODEL_TEMPLATE=gpt-oss
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

PYTHONUNBUFFERED=1 python3 -u -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$eval_files" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    +data.apply_chat_template_kwargs.reasoning_effort=medium \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=3.0 \
    actor_rollout_ref.actor.router_aux_loss_coef=0.0 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.multiround_adaptive_downsampling=${multiround_adaptive_downsampling} \
    algorithm.reinforce_ada_choice=${reinforce_ada_choice} \
    algorithm.global_stat_est=${global_stat_est} \
    algorithm.round_repeat=${round_repeat} \
    algorithm.max_rounds=${max_rounds} \
    algorithm.positive_threshold=${positive_threshold} \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.actor.policy_loss.loss_mode="archer" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.rollout_is_threshold=2.0 \
    algorithm.rollout_is=True \
    algorithm.rollout_is_mode=mask \
    algorithm.rollout_is_level=token \
    actor_rollout_ref.nccl_timeout=1800 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.watchdog_timeout=1800 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=${offload} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=40 \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=batch \
    reward_model.launch_reward_fn_async=True \
    custom_reward_function.path=${custom_reward_function_path} \
    custom_reward_function.name=${custom_reward_function_name} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=8 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="$ckpts_dir" \
    trainer.rollout_data_dir="$rollout_dir" \
    trainer.validation_data_dir="$val_dir" \
    trainer.balance_batch=True \
    trainer.resume_mode=auto
