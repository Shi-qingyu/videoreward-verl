#!/bin/bash

pip install -U uv
# create venv
# imp从python 3.12开始移除，但byted-wandb安装时依赖imp
uv venv .venv --python=3.11 --seed
source .venv/bin/activate
# install custom packages
uv pip install -r requirements.txt

# torch需要强制cu128 
# 依赖torch的都不放在requirements中，避免重复安装
# 新版vllm/transformers在b系列上会造成flash-attn报varlen错误
uv pip install "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0"
uv pip install torchdata==0.10.0
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match
# flash-attn / flashinfer-python version 主要考虑vllm兼容
wget -c -O flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
uv pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
# 如果网络问题可换成pip install
uv pip install flashinfer-python
uv pip install numpy==1.26.4

# 安装agent相关依赖
uv pip install -r requirements_agent.txt
pip install -r requirements_agent_byted.txt