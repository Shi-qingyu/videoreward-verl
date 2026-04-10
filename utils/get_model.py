import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

src = "/opt/tiger/gpt-oss-20b"          # 原始 MXFP4 + BF16 混合权重
dst = "/opt/tiger/gpt-oss-20b-bf16"     # 目标纯 BF16 目录
# attn_implementation = "eager"
attn_implementation = "flash_attention_2"

model = AutoModelForCausalLM.from_pretrained(
    src,
    dtype=torch.bfloat16,
    quantization_config=Mxfp4Config(dequantize=True),  # 关键：解量化到 BF16
    attn_implementation=attn_implementation,           # 没装 FA2 就用 "sdpa"
    trust_remote_code=True,
    device_map="auto",
    use_cache=False,
)
# quantization_config前面dequantize=True后在config.json中不会出现
# model.config.quantization_config = None
# 前面指定后config.json已经是bf16
# model.config.dtype = "bfloat16"
# 前面指定后正确写入config.json
# model.config.attn_implementation = attn_implementation

model.save_pretrained(dst, safe_serialization=True)

tok = AutoTokenizer.from_pretrained(src, use_fast=True)
tok.save_pretrained(dst)