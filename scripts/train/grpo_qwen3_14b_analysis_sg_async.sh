#!/usr/bin/env bash
set -x

project_name='react_exp'
exp_name='react_sglang_debug'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.001
kl_loss_type=mse

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 96))
max_response_length=$((1024 * 32))
max_model_len=$((1024 * 128))
# 影响rollout prefill，对速度影响不大，会影响kv cache大小
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 1))
sp_size=8

# actor_ppo_max_token_len*sp_size作为max_token_len
# / sp_size 能影响updae和backward的memory
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
# 作为计算log_prob的max_token_len，和sp_size无关 影响forward update_actor, 长prompt时只能用1
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))

# dr.grpo loss
loss_agg_mode="seq-mean-token-sum-norm"

train_prompt_mini_bsz=24
train_prompt_bsz=$((train_prompt_mini_bsz * 4))
gen_prompt_bsz=$((train_prompt_mini_bsz * 4))
n_resp_per_prompt=8

# Paths
base_path=/opt/tiger/live_strategy_posttrain
model_path=/opt/tiger/Qwen3-14B
ckpts_dir=${base_path}/checkpoints/${project_name}/${exp_name}
rollout_dir=${base_path}/.cache/${project_name}/${exp_name}/rollout
val_dir=${base_path}/.cache/${project_name}/${exp_name}/val

# reasoning_train_path=${base_path}/data/debug_single_turn_19_0613.parquet
# reasoning_test_path=${base_path}/data/debug_single_turn_19_0613_eval.parquet

single_turn_gemini_0613=${base_path}/data/single_turn_gemini_bench_0613_eval.parquet
single_turn_0620=${base_path}/data/single_turn_claude_0620_train.parquet
single_turn_0621=${base_path}/data/single_turn_claude_qwen_0621_train.parquet
# single_turn_0622=${base_path}/data/single_turn_claude_qwen_0622_train.parquet
summary_train_2506=${base_path}/data/summary_train_2506_train.parquet
summary_bench_2504=${base_path}/data/summary_bench_2504_train.parquet
summary_user_2507=${base_path}/data/summary_train_2507_user_train.parquet
train_files="['$summary_train_2506', '$summary_bench_2504', '$summary_user_2507', '$single_turn_0620', '$single_turn_0621']"

custom_reward_function_path=${base_path}/reward_system/reward.py
custom_reward_function_name=batch_analysis_reward_fn

# Algorithm
temperature=1.0
top_p=0.98
top_k=20 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.95
val_temperature=0.6

# Mathematically equivalent
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null

# performance
offload=True
strategy=fsdp2
# vllm自动计算rollout_batch, 设置超过该值时会造成构造rollout llm时使用default max_num_seqs=1024时OOM
gpu_memory_utilization=0.6
max_num_seqs=128

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
# 可以和实际dp_size保持一致，每个dp_size起一个AsyncSglangServer，worker通过load_balancing选择空闲的server进行rollout
agent_workers=2
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# 清除代理，以通过http方式请求RM
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY all_proxy ALL_PROXY
# SUPPORT AGENT
export PYTHONPATH="${base_path}:${base_path}/verl:${base_path}/ttlive_strategy_agent:$PYTHONPATH"
export DEBUG=True
export RL_TRAIN=True
# YARN配置 - 扩展context length
export YARN_JSON_CONFIG='{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'

# 无论配置为不同的key还是str，都无法正常解析
# +actor_rollout_ref.model.override_config.rope_scaling='{type:yarn,factor:4.0,original_max_position_embeddings:32768}' \

python3 -u -m verl.trainer.main_ppo \
    data.train_files="$summary_user_2507" \
    data.val_files="$single_turn_gemini_0613" \
    data.return_raw_chat=$return_raw_chat \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
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
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=${offload} \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
    actor_rollout_ref.model.trust_remote_code=False \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.agent.num_workers=${agent_workers} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
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
    reward_model.reward_manager=batch \
    custom_reward_function.path=${custom_reward_function_path} \
    custom_reward_function.name=${custom_reward_function_name} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=1 \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="$ckpts_dir" \
    trainer.rollout_data_dir="$rollout_dir" \
    trainer.validation_data_dir="$val_dir" \
    trainer.balance_batch=False \
    trainer.resume_mode=disable
