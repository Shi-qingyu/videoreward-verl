#! /bin/bash

# mkdir -p data
mkdir -p log
mkdir -p checkpoints
mkdir -p .cache

###### model
hdfs dfs -get hdfs://harunava/home/byte_ttlive_strategy/aigc/user/huangjunyi/base_models/Qwen3-4B-Instruct-2507 /opt/tiger/Qwen3-4B-Instruct-2507
hdfs dfs -get hdfs://harunava/home/byte_ttlive_strategy/aigc/user/huangjunyi/base_models/gpt-oss-20b-bf16 /opt/tiger/gpt-oss-20b-bf16
hdfs dfs -get hdfs://harunava/home/byte_ttlive_strategy/aigc/user/huangjunyi/base_models/gpt-oss/gpt-oss-20b-eager /opt/tiger/gpt-oss-20b-eager
hdfs dfs -get hdfs://harunava/home/byte_ttlive_strategy/aigc/user/huangjunyi/base_models/Qwen3-14B-fixlen /opt/tiger/Qwen3-14B
hdfs dfs -get hdfs://harunava/home/byte_ttlive_strategy/aigc/user/huangjunyi/base_models/Qwen3-30B-A3B-Instruct-2507 /opt/tiger/Qwen3-30B-A3B-Instruct-2507

###### data
hdfs dfs -get hdfs://harunava/home/byte_ttlive_strategy/aigc/user/huangjunyi/analysis_data/data /opt/tiger/live_strategy_posttrain/data

# local_data_path="/opt/tiger/live_strategy_posttrain/data"

# wget http://tosv.byted.org/obj/bin-us/toscli -O toscli && chmod a+x toscli

# tos_path="llm_posttrain/reasoning_model_data"

# # 定义需要下载的数据集
# datasets=(
#     # "rl_reasoning_data_hard.parquet"
#     # "rl_eval_aime24.parquet"
#     # "debug_single_turn_19_0613.parquet"
#     # "debug_single_turn_19_0613_eval.parquet"
#     # "single_turn_gemini_bench_0613_eval.parquet"
#     "summary_bench_2507_eval.parquet"
#     "summary_bench_2507_eval_v2.parquet"
#     "analysis_bench_2507_eval.parquet"
#     "analysis_bench_2507_eval_v2.parquet"
#     "react_claude_bench_2507_react_eval.parquet"
#     # "single_turn_claude_0620_train.parquet"
#     # "single_turn_claude_qwen_0621_train.parquet"
#     # "single_turn_claude_qwen_0622_train.parquet"
#     "summary_train_2506_train.parquet"
#     "summary_bench_2504_train.parquet"
#     "summary_train_2507_user_train.parquet"
#     "summary_train_2507_user_train_v2.parquet"
#     "summary_train_2508_user_train.parquet"
#     "summary_train_2508_user_train_v2.parquet"
#     "single_turn_gemini_0813_train.parquet"
#     "single_turn_gemini_0813_train_v2.parquet"
#     "react_gemini_0818_train.parquet"
#     "react_gpt_0819_react_train.parquet"
# )

# # 循环下载数据集
# for filename in "${datasets[@]}"; do
#     local_file_path="${local_data_path}/${filename}"
#     tos_file_path="${tos_path}/${filename}"
    
#     echo "Downloading ${tos_file_path} to ${local_file_path}..."
#     ./toscli -timeout 10m -bucket live-strategy-mllm-us -accessKey 1C07S87RNA7MGPGTXBPM get -filename ${local_file_path} ${tos_file_path}
# done
