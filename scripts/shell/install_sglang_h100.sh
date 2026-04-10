#!/bin/bash

# install uv
pip install -U uv
uv venv sg_venv --python=3.11 --seed
source sg_venv/bin/activate

uv pip install -r requirements_sglang.txt

# install custom packages
uv pip install torchdata==0.10.0
# 安装合适的torchvision
# uv pip install "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" --torch-backend=cu128 --reinstall --refresh
# try with cuda 12.4
uv pip install sglang --prerelease=allow
uv pip install numpy==1.26.4

# install flash-attn 需要正确的cuda版本
# uv pip install --no-build-isolation -v flash-attn==2.8.3 
wget -c -O flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
uv pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# 安装agent相关依赖
uv pip install -r requirements_agent.txt
pip install -r requirements_agent_byted.txt

