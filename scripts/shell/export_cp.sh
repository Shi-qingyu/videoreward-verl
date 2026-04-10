#!/bin/bash

# Script to export FSDP checkpoint to HuggingFace format
# Usage: ./export_cp.sh <checkpoint_path>
# Example: ./export_cp.sh react_exp/qwen3_baseline/global_step_1

if [ $# -ne 1 ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 react_exp/qwen3_baseline/global_step_1"
    exit 1
fi

CHECKPOINT_PATH=$1

# Construct the full paths
LOCAL_DIR="/opt/tiger/live_strategy_posttrain/checkpoints/${CHECKPOINT_PATH}/actor"
TARGET_DIR="/opt/tiger/live_strategy_posttrain/checkpoints/${CHECKPOINT_PATH}/output"


# Check if the actor directory exists
if [ ! -d "$LOCAL_DIR" ]; then
    echo "Error: Actor directory not found: $LOCAL_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Exporting model from: $LOCAL_DIR"
echo "Output directory: $TARGET_DIR"

# Run the model merger command
cd verl
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR"

if [ $? -eq 0 ]; then
    echo "Model export completed successfully!"
    echo "Merged model saved to: $TARGET_DIR"
else
    echo "Model export failed!"
    exit 1
fi

HDFS_BASE_DIR="hdfs://harunava/home/byte_ttlive_strategy/aigc/trained_models/strategy_agent/"

# Parse checkpoint path: prj_name/exp_name/global_step_n
PRJ_NAME=$(echo "$CHECKPOINT_PATH" | cut -d'/' -f1)
EXP_NAME=$(echo "$CHECKPOINT_PATH" | cut -d'/' -f2)
GLOBAL_STEP=$(echo "$CHECKPOINT_PATH" | cut -d'/' -f3)

# Validate checkpoint path format
if [ -z "$PRJ_NAME" ] || [ -z "$EXP_NAME" ] || [ -z "$GLOBAL_STEP" ]; then
    echo "Error: CHECKPOINT_PATH must be in the form prj_name/exp_name/global_step_n"
    exit 1
fi

HDFS_EXP_DIR="${HDFS_BASE_DIR}${PRJ_NAME}/${EXP_NAME}"
HDFS_SAVE_DIR="${HDFS_EXP_DIR}/${GLOBAL_STEP}_output"

echo "Creating HDFS directory: ${HDFS_EXP_DIR}"
hdfs dfs -mkdir -p "${HDFS_EXP_DIR}" || { echo "Error: failed to create ${HDFS_EXP_DIR}"; exit 1; }

echo "Uploading to HDFS: ${HDFS_SAVE_DIR}"
hdfs dfs -put -f "${TARGET_DIR}" "${HDFS_SAVE_DIR}" || { echo "Error: failed to upload to ${HDFS_SAVE_DIR}"; exit 1; }

echo "HDFS upload completed: ${HDFS_SAVE_DIR}"