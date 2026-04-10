# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
import traceback
from collections import defaultdict
from typing import Optional
import copy
from PIL import Image
import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from my_qwen_vl_utils.openo3_vision_process import process_vision_info


logger = logging.getLogger(__name__)

DATA_ROOT = os.getenv("DATA_ROOT", "/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/data/Open-o3-Video/videos")
ROOT = DATA_ROOT
GQA_ROOT = os.path.join(ROOT, "gqa")
TIMERFT_ROOT = os.path.join(ROOT, "timerft")
TVG_ROOT = os.path.join(ROOT, "tvg_r1")
VIDEO_ESPRESSO_KF_ROOT = os.path.join(ROOT, "videoespresso/kfs")
VIDEO_ESPRESSO_ROOT = os.path.join(ROOT, "videoespresso/videos")
STR_KF_ROOT = os.path.join(ROOT, "stgr/temporal_grounding/kfs")
STR_DATA = os.path.join(ROOT, "stgr/temporal_grounding/videos")
STR_PLM_KF_ROOT = os.path.join(ROOT, "stgr/plm/kfs")
STR_PLM_DATA = os.path.join(ROOT, "stgr/plm/videos")
GENERAL_VIDEO_ROOT = os.path.join(ROOT, "videor1")


SYSTEM_PROMPT = {
    "visual QA": "A conversation between user and assistant. The user provides an image and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When referring to particular objects in the reasoning process, the assistant must localize the object with bounding box coordinates between <box> and </box>. The answer must strictly follow the following format:`<obj>object_name</obj><box>bounding_box</box>'.",
    "temporal-spatial free-form QA": "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element in the reasoning process, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. The answer part only requires a text response; tags like <obj>, <box>, <t> are not needed.",
    "temporal QA": "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant determines the precise time period that answers the question. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When mentioning time during the reasoning process, the assistant must use the format: `<t>time_in_seconds</t>s'.The answer must strictly follow the following format: `From <t>start_time</t>s to <t>end_time</t>s'.",
    "temporal QA (MCQ)": "A conversation between user and assistant. The user provides a video and a multiple-choice question, and the Assistant determines the precise time period that answers the question and selects the correct option. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When mentioning time during the reasoning process, the assistant must use the format: `<t>time_in_seconds</t>s'. The answer must strictly follow the following format: `From <t>start_time</t>s to <t>end_time</t>s.\nCorrect Option: [ONLY THE LETTER]'.",
    "General video QA MCQ": "A conversation between user and assistant. The user provides a video and asks a multiple-choice question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element in the reasoning process, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. Only output the correct option in the <answer> </answer> section.",
    "General video QA Free-form": "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element in the reasoning process, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. The answer part only requires a text response; tags like <obj>, <box>, <t> are not needed."
}


def make_conversation_image_and_video(example):

    task = example.get('task')

    if task == 'visual QA':
        system_message = SYSTEM_PROMPT['visual QA']
        content_list = [{"type": "image"}, {"type": "text", "text": example['question']}]
    elif task in ['temporal-spatial free-form QA', 'temporal QA', 'temporal QA (MCQ)', 'General video QA MCQ', 'General video QA Free-form']:
        system_message = SYSTEM_PROMPT[task]
        content_list = [{"type": "video"}, {"type": "text", "text": example['question']}]
    else:
        raise ValueError(f"Unknown task: {task}")

    prompt_list = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": content_list}
    ]

    example['prompt'] = prompt_list   
    return example

    
def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    if not data_list:
        return {}

    tensors = {}
    non_tensors = {}

    all_keys = set()
    for data in data_list:
        all_keys.update(data.keys())

    for key in all_keys:
        values = [data.get(key, None) for data in data_list]
        non_none_values = [v for v in values if v is not None]

        if not non_none_values:
            non_tensors[key] = np.array(values, dtype=object)
            continue

        is_tensor_key = all(isinstance(v, torch.Tensor) for v in non_none_values)
        is_non_tensor_key = all(not isinstance(v, torch.Tensor) for v in non_none_values)

        if not (is_tensor_key or is_non_tensor_key):
            raise TypeError(f"Mixed tensor and non-tensor values found for key: {key}")

        if is_tensor_key:
            if len(non_none_values) != len(values):
                raise ValueError(f"Missing tensor value for key: {key}")
            tensors[key] = torch.stack(values, dim=0)
        else:
            non_tensors[key] = np.array(values, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get("image_patch_size", 14)
        self.max_prompt_length = config.get("max_prompt_length", 10000)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.tool_config_path = config.get("tool_config_path", None)
        self.tool_schemas = None
        if self.tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(self.tool_config_path)
                # match ToolAgentLoop behaviour: model_dump to plain dicts
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")

        # breakpoint()
        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file in self.data_files:
            file_ext = os.path.splitext(str(data_file))[1].lower()
            if file_ext in {".parquet", ".pq"}:
                dataset_format = "parquet"
            elif file_ext in {".json", ".jsonl"}:
                dataset_format = "json"
            else:
                raise ValueError(f"Unsupported data file format: {data_file}")

            dataframe = datasets.load_dataset(dataset_format, data_files=data_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")
        
        self.dataframe = self.dataframe.map(make_conversation_image_and_video)


    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        # breakpoint()
        row_dict: dict = self.dataframe[item]
        input_copy = copy.deepcopy(row_dict[self.prompt_key])
        resolved_video_path = None
        resolved_image_path = None

        source = row_dict['source']
        if source == 'videoespresso_train_video':
            if 'video_path' not in row_dict:
                raise ValueError(f"Missing video_path for source: {source}")
            video_root = VIDEO_ESPRESSO_ROOT
            resolved_video_path = os.path.join(video_root, row_dict['video_path'])
            input_copy[1]['content'][0]['video'] = resolved_video_path
        elif source == 'timerft':
            if 'video_path' not in row_dict:
                raise ValueError(f"Missing video_path for source: {source}")
            video_root = TIMERFT_ROOT
            resolved_video_path = os.path.join(video_root, row_dict['video_path'])
            input_copy[1]['content'][0]['video'] = resolved_video_path
        elif source == 'gqa':
            if 'image_path' not in row_dict:
                raise ValueError(f"Missing image_path for source: {source}")
            image_root = GQA_ROOT
            resolved_image_path = os.path.join(image_root, row_dict['image_path'])
            input_copy[1]['content'][0]['image'] = resolved_image_path
        elif "STR" in source:
            if 'video_path' not in row_dict:
                raise ValueError(f"Missing video_path for source: {source}")
            if "STR_plm" in source:
                video_root = STR_PLM_DATA
            else:
                video_root = STR_DATA
            resolved_video_path = os.path.join(video_root, row_dict['video_path'])
            input_copy[1]['content'][0]['video'] = resolved_video_path
        elif "TVG" in source:
            if 'video_path' not in row_dict:
                raise ValueError(f"Missing video_path for source: {source}")
            video_root = TVG_ROOT
            resolved_video_path = os.path.join(video_root, row_dict['video_path'])
            input_copy[1]['content'][0]['video'] = resolved_video_path
        elif "videor1" in source:
            if 'video_path' not in row_dict:
                raise ValueError(f"Missing video_path for source: {source}")
            video_root = GENERAL_VIDEO_ROOT
            resolved_video_path = os.path.join(video_root, row_dict['video_path'])
            input_copy[1]['content'][0]['video'] = resolved_video_path
        else:
            raise ValueError(f"Invalid source: {source}")
    
        # messages = self._build_messages(row_dict)
        messages = input_copy

        if 'key_items' in row_dict:
            keys_to_remove = []
            for key, itm in row_dict['key_items'].items():
                if itm is None:
                    keys_to_remove.append(key)
                elif isinstance(itm, dict):
                    sub_keys_to_remove = [k for k, v in itm.items() if v is None]
                    for k in sub_keys_to_remove:
                        del itm[k]            
            for key in keys_to_remove:
                del row_dict['key_items'][key]

        model_inputs = {}
        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )

            # breakpoint()
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            if image_inputs is None and video_inputs is None:
                raise ValueError(
                    f"Failed to parse media inputs for task={row_dict.get('task')} source={row_dict.get('source')} "
                    f"video_path={row_dict.get('video_path')} image_path={row_dict.get('image_path')}"
                )

            if image_inputs is not None:
                row_dict['image_size_refine'] = (image_inputs[0].size[0], image_inputs[0].size[1])  # W * H

            if video_inputs is not None:
                row_dict['video_sample_fps'] = video_kwargs['fps'][0]
                row_dict['video_duration'] = video_inputs[0].size(0) / video_kwargs['fps'][0]
                row_dict['image_size'] = (video_inputs[0].size(3), video_inputs[0].size(2))  # W * H
            
            if video_inputs is not None:
                if row_dict['task'] != "temporal-spatial free-form QA":
                    frame_prompt = ""
                    ori_idx = 0
                    while ori_idx < len(video_inputs[0]):
                        time_now = round(ori_idx / video_kwargs['fps'][0],1)
                        frame_prompt += f"Frame {ori_idx + 1} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                        ori_idx += 1
                    frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
                    raw_prompt = raw_prompt.replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)
                    row_dict['prompt_text_final'] = raw_prompt
                    # 将每帧作为独立图像传递给 VLLM
                    image_inputs = [video_inputs[0][i] for i in range(len(video_inputs[0]))]

                else:
                    width, height = video_inputs[0].size(3), video_inputs[0].size(2)
                    image_size = (width, height)
                    # Here, we need to add key frames.
                    if row_dict['source'] == 'videoespresso_train_video':
                        key_frame_root = VIDEO_ESPRESSO_KF_ROOT
                    elif 'STR_plm' in row_dict['source']:
                        key_frame_root = STR_PLM_KF_ROOT
                    else:
                        key_frame_root = STR_KF_ROOT

                    key_frames = []

                    for key_frame in row_dict["key_frames"]:
                        kf_path = os.path.join(key_frame_root, key_frame["path"])
                        kf = Image.open(kf_path)
                        kf = kf.convert('RGB')
                        resized_kf = kf.resize(image_size)
                        resized_kf = np.array(resized_kf)
                        resized_kf = np.transpose(resized_kf, (2, 0, 1))
                        resized_kf = torch.from_numpy(resized_kf)
                        key_frames.append((round(key_frame["time"]), resized_kf))

                    frame_prompt = ""
                    refined_image_inputs = []
                    kf_idx = 0
                    ori_idx = 0
                    frame_idx = 1
                    while ori_idx < len(video_inputs[0]):
                        time_now = int(ori_idx / video_kwargs['fps'][0])
                        if kf_idx < len(key_frames) and time_now >= key_frames[kf_idx][0]:
                            refined_image_inputs.append(key_frames[kf_idx][1])
                            time_now = round(key_frames[kf_idx][0],1)
                            frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                            kf_idx += 1
                        else:
                            refined_image_inputs.append(video_inputs[0][ori_idx])
                            time_now = round(ori_idx / video_kwargs['fps'][0],1)
                            frame_prompt += f"Frame {frame_idx} at {time_now}s: <|vision_start|><|image_pad|><|vision_end|>\n"
                            ori_idx += 1
                        frame_idx += 1
                    frame_prompt += f"The video is in total {int(video_inputs[0].size(0) / video_kwargs['fps'][0])} seconds.\n"
                    # 将每帧作为独立图像传递给 VLLM
                    image_inputs = refined_image_inputs
                    raw_prompt = raw_prompt.replace("<|vision_start|><|video_pad|><|vision_end|>", frame_prompt)
                    row_dict['prompt_text_final'] = raw_prompt

            multi_modal_data = {}
            multi_modal_data["image"] = image_inputs

            model_inputs = self.processor(
                text=[raw_prompt], images=image_inputs, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            row_dict["multi_modal_data"] = multi_modal_data

            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raise RuntimeError("Processor is required for RLHF dataset to process images/videos.")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        
        # 保留原有的 extra_info 字段
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        
        # 添加 reward 计算需要的 extra_info 字段
        extra_info_dict = {
            "index": index,
            "tools_kwargs": tools_kwargs,
            "interaction_kwargs": interaction_kwargs,
        }
        
        # 添加图像/视频尺寸信息（用于 spatial reward）
        if "image_size" in row_dict:
            extra_info_dict["image_size"] = row_dict["image_size"]
        if "image_size_refine" in row_dict:
            extra_info_dict["image_size_refine"] = row_dict["image_size_refine"]
        
        # 添加视频相关信息（用于 temporal reward）
        if "video_sample_fps" in row_dict:
            extra_info_dict["video_sample_fps"] = row_dict["video_sample_fps"]
        if "video_duration" in row_dict:
            extra_info_dict["video_duration"] = row_dict["video_duration"]
        if resolved_video_path is not None:
            extra_info_dict["video_path"] = resolved_video_path
         
        # 添加关键帧信息（用于 temporal-spatial reward）
        if "key_frames" in row_dict:
            extra_info_dict["key_frames"] = row_dict["key_frames"]
        if "key_items" in row_dict:
            extra_info_dict["key_items"] = row_dict["key_items"]
        if resolved_image_path is not None:
            extra_info_dict["image_path"] = resolved_image_path
            extra_info_dict["image_paths"] = [resolved_image_path]
         
        row_dict["extra_info"] = extra_info_dict
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        row_dict["data_source"] = row_dict["task"]

        # ground_truth is used for reward model
        row_dict["reward_model"] = {    
            "ground_truth": row_dict["answer"]
        }

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

if __name__ == '__main__':
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_tokenizer, hf_processor
    local_path = "/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/models/open-o3-video/sft_qwen3vl_4b_31k_1123"
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none
    train_files = ['/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/datasets/open-o3-video-data/samples_cleaned.parquet']
    config = {
        "debug": "666"
    }
    dataset = RLHFDataset(
        data_files=train_files,
        tokenizer=tokenizer,
        processor=processor,
        config=config
    )

    for i in range(len(dataset)):
        example = dataset[i]
        print(i)
    # example = dataset[1]
    # print("example.keys:", example.keys())
    # print("example:", example)
