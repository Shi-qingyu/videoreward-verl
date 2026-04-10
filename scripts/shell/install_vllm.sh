#!/bin/bash

pip install -U uv

export PROJECT_NAME=$(basename "$(pwd)")
export UV_PROJECT_ENVIRONMENT=$HOME/venv/$PROJECT_NAME
if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
    uv venv --python 3.11 "$UV_PROJECT_ENVIRONMENT"
else
    echo "venv exists"
fi

source $UV_PROJECT_ENVIRONMENT/bin/activate

# install custom packages
uv pip install -r requirements.txt

# torch需要强制cu128 
# 依赖torch的都不放在requirements中，避免重复安装
# 新版vllm/transformers在b系列上会造成flash-attn报varlen错误
uv pip install "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0"
uv pip install torchdata==0.10.0

# flash-attn / flashinfer-python version 主要考虑vllm兼容
wget -c -O flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
uv pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
# 如果网络问题可换成pip install

uv pip install flashinfer-python

uv pip install "vllm[flashinfer]==0.11.0"
# uv pip install "vllm==0.11.0"

# 最新wheel 4.57.1仍有报错
# uv pip install --upgrade "git+https://github.com/huggingface/transformers"
# 这个可能存在版本错误，直接复制shihao 目录下的transformers进行安装
# git clone https://github.com/huggingface/transformers.git
# cd transformers
# uv pip install '.[torch]'
# cd ..


uv pip install transformers==4.57.1
uv pip install numpy==1.26.4
uv pip install debugpy==1.8.0
uv pip install rouge_score

uv pip uninstall wandb
uv pip install -U byted-wandb -i https://bytedpypi.byted.org/simple
uv pip install setuptools==75.6.0
