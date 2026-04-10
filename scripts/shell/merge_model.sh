#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/shell/export_openo3_fsdp_to_hf.sh <global_step_dir_or_actor_dir> [output_dir]

Examples:
  bash scripts/shell/export_openo3_fsdp_to_hf.sh \
    /mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/Open-o3-Video-verl/ckpts/open-o3-video-verl-bzy/grpo_openo3_reproduce-4h/global_step_2326

  bash scripts/shell/export_openo3_fsdp_to_hf.sh \
    /mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/Open-o3-Video-verl/ckpts/open-o3-video-verl-bzy/grpo_openo3_reproduce-4h/global_step_2326/actor \
    /mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/Open-o3-Video-verl/ckpts/open-o3-video-verl-bzy/grpo_openo3_reproduce-4h/global_step_2326/actor_hf

Environment variables:
  PYTHON_BIN   Python executable to use. Defaults to /home/tiger/venv/Open-o3-Video-verl/bin/python
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 1
fi

INPUT_PATH="$1"
PYTHON_BIN="${PYTHON_BIN:-/home/tiger/venv/Open-o3-Video-verl/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERL_DIR="${REPO_ROOT}/verl"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: Python executable not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -d "${INPUT_PATH}" ]]; then
    echo "Error: input path does not exist: ${INPUT_PATH}" >&2
    exit 1
fi

if [[ -d "${INPUT_PATH}/actor" ]]; then
    ACTOR_DIR="${INPUT_PATH}/actor"
    DEFAULT_OUTPUT_DIR="${INPUT_PATH}/actor_hf"
else
    ACTOR_DIR="${INPUT_PATH}"
    DEFAULT_OUTPUT_DIR="$(cd "${ACTOR_DIR}/.." && pwd)/actor_hf"
fi

OUTPUT_DIR="${2:-${DEFAULT_OUTPUT_DIR}}"

if [[ ! -f "${ACTOR_DIR}/fsdp_config.json" ]]; then
    echo "Error: ${ACTOR_DIR} does not look like an FSDP actor checkpoint (missing fsdp_config.json)." >&2
    exit 1
fi

if ! compgen -G "${ACTOR_DIR}/model_world_size_*_rank_*.pt" > /dev/null; then
    echo "Error: no FSDP model shards found under ${ACTOR_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Repo root      : ${REPO_ROOT}"
echo "Python         : ${PYTHON_BIN}"
echo "Actor checkpoint: ${ACTOR_DIR}"
echo "Output dir     : ${OUTPUT_DIR}"

cd "${VERL_DIR}"

echo
echo "[1/2] Merging FSDP checkpoint into Hugging Face format..."
"${PYTHON_BIN}" -m verl.model_merger merge \
    --backend fsdp \
    --trust-remote-code \
    --local_dir "${ACTOR_DIR}" \
    --target_dir "${OUTPUT_DIR}"

echo
echo "[2/2] Verifying safetensors output..."
if compgen -G "${OUTPUT_DIR}/*.safetensors" > /dev/null || [[ -f "${OUTPUT_DIR}/model.safetensors.index.json" ]]; then
    echo "Found safetensors artifacts directly in ${OUTPUT_DIR}."
else
    echo "No safetensors found after merge. Re-saving with safe_serialization=True..."
    SRC_DIR="${OUTPUT_DIR}" DST_DIR="${OUTPUT_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
import shutil
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

src = os.environ["SRC_DIR"]
dst = os.environ["DST_DIR"]
tmp = dst + ".tmp_safetensors"

if os.path.exists(tmp):
    shutil.rmtree(tmp)

config = AutoConfig.from_pretrained(src, trust_remote_code=True)
arch = (config.architectures or [""])[0]
model_type = getattr(config, "model_type", "").lower()

if "ForTokenClassification" in arch:
    model_cls = AutoModelForTokenClassification
elif "ForConditionalGeneration" in arch or "vision" in model_type or "vl" in model_type:
    model_cls = AutoModelForVision2Seq
else:
    model_cls = AutoModelForCausalLM

model = model_cls.from_pretrained(src, trust_remote_code=True, torch_dtype="auto")
model.save_pretrained(tmp, safe_serialization=True)

try:
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=True)
    processor.save_pretrained(tmp)
except Exception:
    pass

try:
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    tokenizer.save_pretrained(tmp)
except Exception:
    pass

for name in os.listdir(src):
    src_path = os.path.join(src, name)
    dst_path = os.path.join(tmp, name)
    if os.path.exists(dst_path):
        continue
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path)
    else:
        shutil.copy2(src_path, dst_path)

backup = dst + ".backup_bin"
if os.path.exists(backup):
    shutil.rmtree(backup)
os.rename(src, backup)
os.rename(tmp, dst)
print(f"Safetensors export completed. Backup of previous output: {backup}")
PY
fi

echo
echo "Done. Output files:"
ls -lh "${OUTPUT_DIR}" | sed -n '1,120p'
