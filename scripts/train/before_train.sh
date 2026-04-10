#!/usr/bin/env bash

# 清除代理，以通过http方式请求RM
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY all_proxy ALL_PROXY

# SUPPORT AGENT
base_path="$(pwd)"
# 当前版本的放在前面，避免引入verl时有误
# base_path已经由框架层引入
export PYTHONPATH="${base_path}:${base_path}/verl:${base_path}/ttlive_strategy_agent:$PYTHONPATH"
echo "$PYTHONPATH"

# export DEBUG=True
export RL_TRAIN=True
export TOKENIZERS_PARALLELISM=true
export VLLM_USE_V1=1
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 # for vllm0.11.0 with TP
export WANDB_API_KEY="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"

## use when ray debugging
# export RAY_DEBUG_POST_MORTEM=1

