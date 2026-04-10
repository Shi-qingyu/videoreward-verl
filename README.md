# Open-o3 Video VERL

装环境 

```
cd /path/to/Open-o3-Video-verl

bash scripts/shell/install_vllm.sh

source /home/tiger/venv/Open-o3-Video-verl/bin/activate

source scripts/train/before_train.sh
```


# Open-o3 Video VERL merge  版本

兼容 `dev_bzy`  分支中的 Open-o3-video verl训练代码（不影响能跑，效果有待确认，应该不影响，受影响可以暂时切换 `dev_bzy` 训练，两个框架的 vision_process 逻辑不一样 我分别划分在了 `./verl/my_qwen_vl_utils/` 中）

```
# open-o3-video verl 训练
bash /mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/Open-o3-Video-verl/scripts/vltrain/train_openo3_qwen3vl_grpo.sh

# 原生 Video-o3 RL
bash /mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/Open-o3-Video-verl/scripts/vltrain/train_video_qwen25vl_dapo.sh

# 原生 Qwen3VL Video-o3 RL
bash /mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/Open-o3-Video-verl/scripts/vltrain/train_video_qwen3vl_dapo.sh
```

## 主要兼容的地方

### Video-o3 -> Open-o3-Video-verl 迁移说明

- `Video-o3` 里和视频多轮工具调用相关的能力，被迁到 VERL 标准训练链路中
- 原来一部分耦合在单体脚本/单体 worker 里的逻辑，在这里被拆到 `trainer / worker / rollout / reward / dataset` 几层
- 训练主流程尽量复用 VERL 自己的 PPO/GRPO/FSDP 框架，Video-o3 主要补视频数据、工具调用、reward 和 Qwen-VL 兼容

### 几个关键 Python 文件

1. **训练入口**

- 新仓：`verl/verl/trainer/main_ppo.py`
- 作用：训练主入口，负责加载配置、构造 trainer、reward manager、worker group

2. **训练主循环**

- 新仓：`verl/verl/trainer/ppo/ray_trainer.py`
- 老仓对应：`Video-o3/RL/verl/trainer/ppo/ray_trainer.py`
- 作用：这里串起了 rollout、打分、old log prob、advantage、actor update、validation
- 如果要查 “为什么老框架能跑、新框架跑不起来”，通常这里最先看

3. **Actor / Rollout 混部与 FSDP 切换**

- 新仓：`verl/verl/workers/fsdp_workers.py`
- 老仓对应：`Video-o3/RL/verl/workers/fsdp_workers.py`
- 作用：这是迁移里最关键的一层，负责：
  - actor 训练
  - old log prob 计算
  - rollout / trainer 模式切换
  - FSDP 参数/optimizer offload
  - vLLM 混部时的显存切换

4. **视频多轮工具调用 async rollout**

- 新仓：`verl/verl/workers/rollout/vllm_rollout/vllm_async_engine_video.py`
- 老仓对应：`Video-o3/RL/verl/workers/rollout/vllm_rollout/vllm_async_engine_video.py`
- 作用：承接 Video-o3 原来的视频工具调用、多轮采样、grounding 相关推理逻辑
- 像 `temporal_segment`、tool call、异步 reward 这类问题，优先看这里

5. **视频工具 schema / 参数校验**

- 新仓：`verl/verl/workers/rollout/vllm_rollout/function_tools_video.py`
- 作用：定义视频工具调用的参数格式和校验逻辑
- `temporal_segment must be a list or tuple with 2 elements` 这类报错，就在这层

6. **Video-o3 数据集接入**

- 新仓：`verl/verl/utils/dataset/video_o3_dataset.py`
- 作用：把 Video-o3 训练样本组织成 VERL 训练所需的 batch 格式

7. **Video-o3 reward 接入**

- 新仓：`verl/verl/utils/reward_score/video_o3_reward.py`
- 老仓可参考：`Video-o3/RL/verl/utils/reward_score/openo3_video_reward.py`
- 作用：把原来 Video-o3 的打分逻辑接成 VERL reward manager 可调用的接口


### 一句话理解迁移后的代码分工

如果只记一条，可以记成：

- `main_ppo.py`：入口
- `ray_trainer.py`：训练主流程
- `fsdp_workers.py`：actor/rollout 混部和显存切换
- `vllm_async_engine_video.py`：视频多轮工具调用推理
- `function_tools_video.py`：工具参数协议
- `video_o3_dataset.py`：数据
- `video_o3_reward.py`：reward

##  后续 todo

checking

<!-- tiou实验：

nohup bash scripts/vltrain/rloo_2b_only_tiou_1205.sh > ./rloo_2b_only_tiou_1205.out 2>&1 &


tiou+quality实验：

nohup bash scripts/vltrain/rloo_2b_tiou_quality_1205.sh > ./rloo_2b_tiou_quality_1205.out 2>&1 &


convert to huggingface model:

python -m verl.model_merger merge --backend fsdp --local_dir checkpoints/highlight_jiahao/2b_only_tiou/global_step_162/actor --target_dir checkpoints/highlight_jiahao/2b_only_tiou/global_step_162/hf_model

python -m verl.model_merger merge --backend fsdp --local_dir checkpoints/highlight_jiahao/2b_tiou_quality/global_step_162/actor --target_dir checkpoints/highlight_jiahao/2b_tiou_quality/global_step_162/hf_model -->


<!-- # 多机训练
- 分别在两台机器上完成环境的配置
- 假设两台机器分别为node0, node1
  1. 在两台机器起ray的shell中先执行如下指令来清除代理，让各机器都能连接到RM```unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY all_proxy ALL_PROXY```
  2. 在node0上起ray head  
  ```ray start --head --dashboard-host=0.0.0.0```  
  会看到类似"To add another node to this Ray cluster, run..."的输出
  1. 执行ray status，会看到ipv4:port，在node1上起ray worker
  ```ray start --address=ipv4:port```
  1. 此时使用ray status可查看当前可用GPU数
  2. 在任一node上执行训练脚本，注意nodes和tp等参数的设置 -->
   
<!-- # 多机agent RL训练
# 构建环境
bash scripts/shell/install.sh
source .venv/bin/activate
# 下载model data等
bash scripts/train/prepare_analysis.sh
# 修改各项环境变量
source scripts/train/before_train.sh
# head 启动ray
ray start --head --dashboard-host=0.0.0.0
# 其他机器
ray start --address=ipv4:port
# 启动训练脚本
bash scripts/train/grpo_qwen3_14b_analysis.sh > log/train.log 2>&1 -->
