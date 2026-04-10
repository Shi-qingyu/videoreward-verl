export TIKTOKEN_RS_CACHE_DIR=/mnt/bn/strategy-mllm-train/user/hjy/misc/harmony_vocab
mkdir -p /mnt/bn/strategy-mllm-train/user/hjy/misc/harmony_vocab
python -c 'from openai_harmony import load_harmony_encoding; load_harmony_encoding("HarmonyGptOss")'
ls -l /mnt/bn/strategy-mllm-train/user/hjy/misc/harmony_vocab