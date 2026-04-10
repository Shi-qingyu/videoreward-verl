#!/usr/bin/env bash
ray stop
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="9.0"  # 根据你的GPU改，A100是8.0，H100是9.0
# -------------------------
# 基础路径与项目
# -------------------------
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

VERTION_NAME="video-o3-reproduce"  # 需要手动设置
export RUN_NAME="GRPO-${VERTION_NAME}-NNODES${NNODES:-1}"
export DATA_MODE="video"
export VLLM_USE_V1=1
export WANDB_MODE=offline

export PRETRAINED_PATH="/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/models/Qwen2.5-VL-7B-Instruct"
export BASE_IMAGE_DIR="/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/data/Open-o3-Video/videos"
export CKPT_SAVE_DIR="/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/Open-o3-Video-verl/ckpts/${RUN_NAME}"
export LOG_SAVE_DIR="${repo_root}/logs/${RUN_NAME}/resume_$(date +"%Y%m%d-%H%M")"
export WANDB_DIR="/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/tmp/cache/wandb_dir"
export WANDB_ARTIFACT_DIR="/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/tmp/cache/artifacts_dir"
export TMPDIR=/tmp
export HYDRA_FULL_ERROR=1

mkdir -p "${CKPT_SAVE_DIR}" "${LOG_SAVE_DIR}" "${WANDB_DIR}" "${WANDB_ARTIFACT_DIR}"

# -------------------------
# 数据文件路径
# -------------------------
RL_ROOT="/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/data/"
ANNODATA_RL="${RL_ROOT}/Seeker-173K/RL_test"

# 这暂时没有
ANNODATA_TEST="${RL_ROOT}/annodata/test/subset"
SUBSET_CHARADES_TEST="${ANNODATA_TEST}/subset_charades_test_600.json"
SUBSET_MLVU_TEST="${ANNODATA_TEST}/subset_4fps_mlvu_val_400.json"
SUBSET_VIDEOMME_TEST="${ANNODATA_TEST}/subset_4fps_videomme_600.json"

# 训练文件
# TODO: 修改为小批量的 测试训练数据
# CHARADES="${ANNODATA_RL}/charades_grounding_12408.json"
# CGBENCH_WT="${ANNODATA_RL}/cgbench_correct_clue_single_w_tool_6764.json"
# LLaVid_M_WT="${ANNODATA_RL}/llava-video_youtube_qa_mc_2_3_m_clue_multi_w_tool_13900.json"
# LLaVid_M_WOT="${ANNODATA_RL}/llava-video_youtube_qa_mc_2_3_m_clue_multi_wo_tool_29523.json"
# LLaVid_S_WT="${ANNODATA_RL}/llava-video_youtube_qa_mc_2_3_m_clue_single_w_tool_79848.json"
# LLaVid_S_WOT="${ANNODATA_RL}/llava-video_youtube_qa_mc_2_3_m_clue_single_wo_tool_9946.json"
# LongVDB_WT="${ANNODATA_RL}/longvideodb_gemini_clue_single_w_tool_7000.json"
# LongVideoReason_FREE="${ANNODATA_RL}/longvideoreason_qa_from120to3600_freeform_9531.json"
# NEXTGQA_WT="${ANNODATA_RL}/nextgqa_val_w_tool_2365.json"
# NEXTGQA_WOT="${ANNODATA_RL}/nextgqa_val_wo_tool_702.json"
# # 没开源
# SELFBUILT_1_WT="${ANNODATA_RL}/selfbuilt_1_qa_f180to600_clue_single_w_tool_5796.json"
# SELFBUILT_2_WT="${ANNODATA_RL}/selfbuilt_2_qa_f180to600_clue_single_w_tool_7491.json"

CGBENCH_WT="${ANNODATA_RL}/cgbench_correct_clue_single_w_tool_sample_100.json"
LLaVid_M_WT="${ANNODATA_RL}/llava-video_youtube_qa_mc_2_3_m_clue_multi_w_tool_sample_100.json"
LLaVid_M_WOT="${ANNODATA_RL}/llava-video_youtube_qa_mc_2_3_m_clue_multi_wo_tool_sample_100.json"


# -------------------------
# PPO & RL 参数（保持原有参数不动）
# -------------------------
max_prompt_length=18432
max_response_length=8192
train_batch_size=16
ppo_mini_batch_size=16
rollout_n=4
val_n=1
max_generation_round=6
val_max_generation_round=6
limit_mm_per_prompt_video=12
val_limit_mm_per_prompt_video=12
gpt_threads=16
n_gpus_per_node=8
overview_fps=2.0
source_frames_fps=4.0
max_pixels=16384
min_pixels=512
gpu_memory_utilization=0.7
temperature=1.0
top_p=1
top_k=-1
multi_turn_prompt_type=v2
val_do_sample=false
filter_overlong_prompts=true
system_prompt="tool_crop"
use_3drope=true
rejection_sample=true
rejection_sample_multiplier=1
max_num_gen_batches=0
max_total_response_length=32786
vllm_infer_batch_size=16
ref_log_prob_micro_batch_size_per_gpu=4

custom_dataset_path="$repo_root/verl/verl/utils/dataset/video_o3_dataset.py"
custom_reward_path="$repo_root/verl/verl/utils/reward_score/video_o3_reward.py"
ckpts_dir="$repo_root/ckpts/${VERTION_NAME}/${RUN_NAME}"
rollout_dir="$repo_root/.cache/${VERTION_NAME}/${RUN_NAME}/rollout"
val_dir="$repo_root/.cache/${VERTION_NAME}/${RUN_NAME}/val"

mkdir -p "$ckpts_dir" "$rollout_dir" "$val_dir"

export DATA_MODE
export VLLM_USE_V1
# export PYTHONPATH="$repo_root/verl:$repo_root:${PYTHONPATH:-}"

cd "$repo_root/verl"

# -------------------------
# Python 训练命令
# -------------------------
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files=[${CGBENCH_WT},${LLaVid_M_WT},${LLaVid_M_WOT}] \
    data.val_files=[${LLaVid_M_WOT}] \
    data.train_batch_size="${train_batch_size}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.return_raw_chat=True \
    data.filter_overlong_prompts="${filter_overlong_prompts}" \
    data.system_prompt="${system_prompt}" \
    data.use_3drope="${use_3drope}" \
    data.base_media_dir="${BASE_IMAGE_DIR}" \
    data.answer_key=solution \
    data.video_key=video \
    data.overview_fps="${overview_fps}" \
    data.source_frames_fps="${source_frames_fps}" \
    data.max_pixels="${max_pixels}" \
    data.min_pixels="${min_pixels}" \
    data.tool_call=crop \
    data.acc_reward_weight=1.0 \
    data.format_reward_weight=1.0 \
    data.use_tool_reward_weight=0.0 \
    data.decay_penalty_weight=0.05 \
    data.gpt_extract_answer=True \
    data.extract_answer_tags=strict \
    data.custom_cls.path="${custom_dataset_path}" \
    data.custom_cls.name=VideoO3Dataset \
    custom_reward_function.path="${custom_reward_path}" \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=naive_multithreads_tool \
    +reward_model.reward_kwargs.gpt_threads="${gpt_threads}" \
    +reward_model.reward_kwargs.extra_info.acc_reward_weight=1.0 \
    +reward_model.reward_kwargs.extra_info.format_reward_weight=1.0 \
    +reward_model.reward_kwargs.extra_info.decay_penalty_weight=0.05 \
    +reward_model.reward_kwargs.extra_info.gpt_extract_answer=True \
    +reward_model.reward_kwargs.extra_info.extract_answer_tags=strict \
    +reward_model.reward_kwargs.extra_info.max_total_response_length="${max_total_response_length}" \
    actor_rollout_ref.model.path="${PRETRAINED_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm_multi_turn_tool_call \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="${rollout_n}" \
    actor_rollout_ref.rollout.val_n="${val_n}" \
    actor_rollout_ref.rollout.val_do_sample="${val_do_sample}" \
    actor_rollout_ref.rollout.temperature="${temperature}" \
    actor_rollout_ref.rollout.top_p="${top_p}" \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.response_length="${max_response_length}" \
    actor_rollout_ref.rollout.prompt_length="${max_prompt_length}" \
    actor_rollout_ref.rollout.max_total_response_length="${max_total_response_length}" \
    actor_rollout_ref.rollout.max_generation_round="${max_generation_round}" \
    actor_rollout_ref.rollout.val_max_generation_round="${val_max_generation_round}" \
    actor_rollout_ref.rollout.limit_mm_per_prompt.video="${limit_mm_per_prompt_video}" \
    actor_rollout_ref.rollout.val_limit_mm_per_prompt.video="${val_limit_mm_per_prompt_video}" \
    actor_rollout_ref.rollout.vllm_infer_batch_size="${vllm_infer_batch_size}" \
    actor_rollout_ref.rollout.multi_turn_prompt_type="${multi_turn_prompt_type}" \
    actor_rollout_ref.rollout.use_relative_coordinates=True \
    actor_rollout_ref.rollout.use_raw_image=True \
    actor_rollout_ref.rollout.max_pixels="${max_pixels}" \
    actor_rollout_ref.rollout.min_pixels="${min_pixels}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=34816 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    'actor_rollout_ref.rollout.stop=["</grounding>"]' \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ref_log_prob_micro_batch_size_per_gpu}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.max_num_gen_batches="${max_num_gen_batches}" \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${VERTION_NAME}" \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.test_freq=100 \
    trainer.save_freq=100 \
    trainer.use_3drope="${use_3drope}" \
    trainer.rejection_sample="${rejection_sample}" \
    trainer.rejection_sample_multiplier="${rejection_sample_multiplier}" \
    trainer.default_local_dir="${ckpts_dir}" \
    trainer.rollout_data_dir="${rollout_dir}" \
    trainer.validation_data_dir="${val_dir}" \
    2>&1 | tee "${LOG_SAVE_DIR}/train_log.txt"
