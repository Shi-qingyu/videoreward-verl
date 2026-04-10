git fetch
git branch --set-upstream-to=origin/feat/train_gpt
bash scripts/shell/pull.sh
bash scripts/shell/install.sh
source .venv/bin/activate
source scripts/train/before_train.sh
bash scripts/train/rloo_qwen3_4b_rfada_is_normal.sh