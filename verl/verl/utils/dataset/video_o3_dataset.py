from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
import verl.utils.torch_functional as verl_F
from omegaconf import ListConfig
from torch.utils.data import Dataset

from verl.utils.dataset.task_prompt import get_system_prompt, process_problem_with_data_source
from verl.utils.model import compute_position_id_with_mask


def _ensure_repo_root_on_path(package_dir_name: str) -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / package_dir_name).is_dir():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return
    raise ModuleNotFoundError(f"Could not locate package directory '{package_dir_name}' from {__file__}")


try:
    from my_qwen_vl_utils import fetch_video_raw, fetch_video_raw_frame, resample_video_from_raw
    from my_qwen_vl_utils.video_o3_vision_process import IMAGE_FACTOR
except ModuleNotFoundError as exc:
    if exc.name != "my_qwen_vl_utils":
        raise
    _ensure_repo_root_on_path("my_qwen_vl_utils")
    from my_qwen_vl_utils import fetch_video_raw, fetch_video_raw_frame, resample_video_from_raw
    from my_qwen_vl_utils.video_o3_vision_process import IMAGE_FACTOR


def _ensure_list(data_files):
    if isinstance(data_files, (list, ListConfig)):
        return list(data_files)
    return [data_files]


def _resolve_video_path(row: dict[str, Any], base_dir: str) -> str:
    video_path = row.get("video") or row.get("video_path")
    if video_path is None:
        raise ValueError(f"Missing video path in row keys={list(row.keys())}")
    if os.path.isabs(video_path):
        return video_path
    return os.path.join(base_dir, video_path)


def _build_messages(row: dict[str, Any], system_prompt_override: Optional[str] = None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt_override or get_system_prompt(row)},
        {"role": "user", "content": process_problem_with_data_source(row)},
    ]


def _make_conversation_multimodal_video(
    dataframe: pd.DataFrame,
    base_media_dir: str,
    system_prompt_override: Optional[str] = None,
) -> pd.DataFrame:
    def make_conv(row: pd.Series) -> dict[str, Any]:
        item = row.to_dict()
        item.setdefault("data_source", item.get("source", "default"))
        item["prompt"] = _build_messages(item, system_prompt_override=system_prompt_override)

        video_path = item.get("video")
        if video_path is not None and not os.path.isabs(video_path):
            item["video"] = os.path.join(base_media_dir, video_path)
        return item

    return pd.DataFrame(dataframe.apply(make_conv, axis=1).tolist())


def _process_video_adaptive_token_num(
    video_path: str,
    max_tokens: int,
    min_tokens: int,
    fps: float,
    source_frames_fps: float,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path {video_path} not exists.")

    video_ele = {
        "video": video_path,
        "fps": fps,
        "source_frames_fps": source_frames_fps,
    }
    if video_path.endswith(".mp4"):
        raw_video, raw_sample_fps = fetch_video_raw(video_ele)
    else:
        raw_video, raw_sample_fps = fetch_video_raw_frame(video_ele)

    frames_num = raw_video.shape[0]
    max_tokens_per_frame = max_tokens // max(frames_num // 2, 1)
    min_tokens_per_frame = min_tokens // max(frames_num // 2, 1)
    resample_ele = {
        "max_pixels": max_tokens_per_frame * IMAGE_FACTOR * IMAGE_FACTOR,
        "min_pixels": min_tokens_per_frame * IMAGE_FACTOR * IMAGE_FACTOR,
    }
    video, sample_fps = resample_video_from_raw(
        raw_video,
        raw_sample_fps,
        resample_ele,
        return_video_sample_fps=True,
    )
    return video, sample_fps, raw_video, raw_sample_fps


class VideoO3Dataset(Dataset):
    def __init__(self, data_files, tokenizer, processor, config, max_samples: int = -1):
        self.data_files = _ensure_list(data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_samples = max_samples

        self.max_prompt_length = config.get("max_prompt_length", 18432)
        self.truncation = config.get("truncation", "error")
        self.max_pixels = int(config.get("max_pixels", 16384))
        self.min_pixels = int(config.get("min_pixels", 512))
        self.overview_fps = float(config.get("overview_fps", 2.0))
        self.source_frames_fps = float(config.get("source_frames_fps", 4.0))
        self.return_raw_chat = bool(config.get("return_raw_chat", True))
        self.filter_overlong_prompts = bool(config.get("filter_overlong_prompts", True))
        self.use_3drope = bool(config.get("use_3drope", True))
        self.base_media_dir = config.get("base_media_dir") or os.getenv("BASE_IMAGE_DIR", "./datasets")
        self.system_prompt_override = None
        self._read_files()

    def _read_files(self):
        frames = []
        for data_file in self.data_files:
            with open(data_file, "r", encoding="utf-8") as f:
                frames.append(pd.DataFrame(json.load(f)))
        self.dataframe = pd.concat(frames, ignore_index=True)
        self.dataframe = _make_conversation_multimodal_video(
            self.dataframe,
            base_media_dir=self.base_media_dir,
            system_prompt_override=self.system_prompt_override,
        )
        if self.filter_overlong_prompts:
            self.dataframe = self.dataframe[
                self.dataframe.apply(
                    lambda doc: len(self.tokenizer.apply_chat_template(doc["prompt"], add_generation_prompt=True))
                    <= self.max_prompt_length,
                    axis=1,
                )
            ].reset_index(drop=True)
        if self.max_samples > 0:
            self.dataframe = self.dataframe.iloc[: self.max_samples].reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row_dict = self.dataframe.iloc[item].to_dict()
        row = copy.deepcopy(row_dict)
        row.setdefault("data_source", row.get("source", "default"))
        resolved_video_path = _resolve_video_path(row, self.base_media_dir)
        messages = row.get("prompt") or _build_messages(row, system_prompt_override=self.system_prompt_override)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        video_tensor, sample_fps, raw_video_tensor, raw_sample_fps = _process_video_adaptive_token_num(
            resolved_video_path,
            self.max_pixels,
            self.min_pixels,
            fps=float(row.get("fps", self.overview_fps)),
            source_frames_fps=float(row.get("frame_fps", self.source_frames_fps)),
        )
        multi_modal_data = {"video": [video_tensor]}
        video_fps_used = {"fps": [sample_fps]}
        raw_multi_modal_metadata = {
            "video": {
                "path": resolved_video_path,
                "tensor": raw_video_tensor if resolved_video_path.endswith(".mp4") else torch.zeros(0, dtype=torch.uint8),
            },
            "fps": raw_sample_fps if resolved_video_path.endswith(".mp4") else float(row.get("frame_fps", self.source_frames_fps)),
            "length": raw_video_tensor.shape[0] / max(raw_sample_fps, 1e-6),
        }

        time_instruction = (
            f"This video is uniformly sampled at {sample_fps:.2f} fps, contains {video_tensor.shape[0]} frames "
            f"from 0 seconds to {(video_tensor.shape[0] / max(sample_fps, 1e-6) - 0.05):.1f} seconds."
        )
        raw_prompt = prompt_with_chat_template.replace(
            "<video>", "<|vision_start|><|video_pad|><|vision_end|>" + time_instruction
        )

        video_inputs = self.processor.video_processor(multi_modal_data["video"])
        video_grid_thw = video_inputs.get("video_grid_thw", None)
        merge_length = self.processor.video_processor.merge_size**2
        prompt_with_video_pad = prompt_with_chat_template
        video_idx = 0
        while "<video>" in prompt_with_video_pad:
            prompt_with_video_pad = prompt_with_video_pad.replace(
                "<video>",
                "<|vision_start|>"
                + "<|video_pad|>" * int(video_grid_thw[video_idx].prod().item() // merge_length)
                + "<|vision_end|>"
                + time_instruction,
                1,
            )
            video_idx += 1

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_video_pad,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if (
            self.use_3drope
            and self.processor is not None
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=None,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=video_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        ground_truth = row.get("solution")
        if ground_truth is None:
            if "answer" in row or "clue" in row:
                ground_truth = {
                    "answer": row.get("answer", ""),
                    "clue": row.get("clue", []),
                }
            else:
                raise ValueError(f"Missing ground truth fields in row keys={list(row.keys())}")

        uid = row.get("uid", row.get("id", row.get("doc_id", item)))
        extra_info = dict(row.get("extra_info", {}))
        extra_info.setdefault("resolved_video_path", resolved_video_path)
        extra_info.setdefault("video_sample_fps", sample_fps)

        result = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "multi_modal_data": multi_modal_data,
            "multi_modal_inputs": dict(video_inputs),
            "raw_prompt": messages[-1]["content"] if self.return_raw_chat else raw_prompt,
            "ground_truth": ground_truth,
            "data_source": row["data_source"],
            "uid": uid,
            "index": uid,
            "resolved_video_path": resolved_video_path,
            "video_sample_fps": sample_fps,
            "video_fps_used": video_fps_used,
            "raw_multi_modal_metadata": raw_multi_modal_metadata,
            "extra_info": extra_info,
        }
        if "key_frames" in row:
            result["extra_info"] = dict(result["extra_info"])
            result["extra_info"]["key_frames"] = row["key_frames"]
        return result
