from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
import verl.utils.torch_functional as verl_F
from omegaconf import ListConfig
from torch.utils.data import Dataset

from verl.utils.dataset.video_o3_dataset import _process_video_adaptive_token_num
from verl.utils.model import compute_position_id_with_mask


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}

VISION_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


def _ensure_list(data_files: str | list[str] | ListConfig) -> list[str]:
    if isinstance(data_files, (list, ListConfig)):
        return list(data_files)
    return [data_files]


def _resolve_media_path(base_dir: Path, media_path: str, override_base_dir: str | None = None) -> str:
    path = Path(media_path)
    if path.is_absolute():
        return str(path)
    if override_base_dir:
        return str((Path(override_base_dir) / media_path).resolve())
    return str((base_dir / media_path).resolve())


def _normalize_text_content(text: str) -> str:
    return "" if text is None else str(text)


def _build_messages(
    item: dict[str, Any],
    *,
    file_dir: Path,
    base_media_dir: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    images = item.get("images") or []
    if isinstance(images, str):
        images = [images]
    if images:
        raise NotImplementedError("QwenVLVideoRLDataset currently supports video samples only.")

    videos = item.get("videos") or []
    if isinstance(videos, str):
        videos = [videos]
    resolved_videos = [_resolve_media_path(file_dir, video, base_media_dir) for video in videos]

    video_pool = [{"type": "video", "video": video_path} for video_path in resolved_videos]
    messages: list[dict[str, Any]] = []

    for turn in item.get("conversations", []):
        role = ROLE_MAP.get(turn.get("from", ""), turn.get("from", ""))
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported conversation role: {turn.get('from')}")

        text = _normalize_text_content(turn.get("value"))
        if role == "assistant":
            messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            continue

        content: list[dict[str, Any]] = []
        for segment in re.split(r"(<video>)", text):
            if segment == "<video>":
                if not video_pool:
                    raise ValueError("The number of <video> placeholders exceeds the number of provided videos.")
                content.append(video_pool.pop(0))
            elif segment:
                content.append({"type": "text", "text": segment})

        messages.append({"role": "user", "content": content})

    if video_pool:
        raise ValueError(f"{len(video_pool)} video(s) remain unused in sample: {item}")

    return messages, resolved_videos


def _extract_prompt_and_ground_truth(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if not messages:
        raise ValueError("Empty conversations are not supported.")
    if messages[-1]["role"] != "assistant":
        raise ValueError("The last conversation turn must be from the assistant for GRPO training.")

    ground_truth_blocks = messages[-1]["content"]
    if len(ground_truth_blocks) != 1 or ground_truth_blocks[0].get("type") != "text":
        raise ValueError("The final assistant turn must contain a single text block.")

    prompt_messages = messages[:-1]
    if not prompt_messages:
        raise ValueError("At least one prompt turn is required before the final assistant answer.")

    return prompt_messages, str(ground_truth_blocks[0]["text"])


def _build_time_instruction(video_tensor: torch.Tensor, sample_fps: float) -> str:
    duration = video_tensor.shape[0] / max(sample_fps, 1e-6)
    return (
        f"This video is uniformly sampled at {sample_fps:.2f} fps, contains {video_tensor.shape[0]} frames "
        f"from 0 seconds to {max(duration - 0.05, 0.0):.1f} seconds."
    )


class QwenVLVideoRLDataset(Dataset):
    def __init__(self, data_files, tokenizer, processor, config, max_samples: int = -1):
        self.data_files = _ensure_list(data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_samples = max_samples

        self.max_prompt_length = int(config.get("max_prompt_length", 16384))
        self.truncation = config.get("truncation", "error")
        self.max_pixels = int(config.get("max_pixels", 16384))
        self.min_pixels = int(config.get("min_pixels", 512))
        self.overview_fps = float(config.get("overview_fps", 2.0))
        self.source_frames_fps = float(config.get("source_frames_fps", 4.0))
        self.return_raw_chat = bool(config.get("return_raw_chat", True))
        self.filter_overlong_prompts = bool(config.get("filter_overlong_prompts", False))
        self.use_3drope = bool(config.get("use_3drope", True))
        self.base_media_dir = config.get("base_media_dir")

        if self.processor is None:
            raise ValueError("QwenVLVideoRLDataset requires a processor.")

        self._read_files()

    def _read_files(self) -> None:
        rows: list[dict[str, Any]] = []
        for data_file in self.data_files:
            file_path = Path(data_file).resolve()
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, list):
                raise ValueError(f"Expected a JSON list in {data_file}, but got {type(loaded)}.")

            for local_idx, item in enumerate(loaded):
                sample = copy.deepcopy(item)
                sample["_data_file"] = str(file_path)
                sample["_data_dir"] = str(file_path.parent)
                sample["_local_idx"] = local_idx
                messages, resolved_videos = _build_messages(
                    sample,
                    file_dir=file_path.parent,
                    base_media_dir=self.base_media_dir,
                )
                prompt_messages, ground_truth = _extract_prompt_and_ground_truth(messages)
                sample["messages"] = messages
                sample["prompt"] = prompt_messages
                sample["ground_truth"] = ground_truth
                sample["resolved_videos"] = resolved_videos
                rows.append(sample)

        if self.filter_overlong_prompts:
            filtered_rows = []
            for row in rows:
                rendered = self.processor.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=False)
                if len(self.tokenizer.encode(rendered, add_special_tokens=False)) <= self.max_prompt_length:
                    filtered_rows.append(row)
            rows = filtered_rows

        if self.max_samples > 0:
            rows = rows[: self.max_samples]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = copy.deepcopy(self.rows[index])
        prompt_messages = row["prompt"]
        resolved_videos = row["resolved_videos"]

        prompt_with_chat_template = self.processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_with_video_pad = prompt_with_chat_template
        raw_prompt = prompt_with_chat_template

        processed_videos: list[torch.Tensor] = []
        sample_fps_list: list[float] = []
        raw_video_tensors: list[torch.Tensor] = []
        raw_sample_fps_list: list[float] = []

        for video_path in resolved_videos:
            video_tensor, sample_fps, raw_video_tensor, raw_sample_fps = _process_video_adaptive_token_num(
                video_path,
                self.max_pixels,
                self.min_pixels,
                fps=float(row.get("fps", self.overview_fps)),
                source_frames_fps=float(row.get("frame_fps", self.source_frames_fps)),
            )
            processed_videos.append(video_tensor)
            sample_fps_list.append(float(sample_fps))
            raw_video_tensors.append(raw_video_tensor)
            raw_sample_fps_list.append(float(raw_sample_fps))

        multi_modal_data = {"video": processed_videos}
        video_fps_used = {"fps": sample_fps_list}
        video_inputs = self.processor.video_processor(multi_modal_data["video"])
        video_grid_thw = video_inputs.get("video_grid_thw")
        merge_length = self.processor.video_processor.merge_size**2

        for video_idx, (video_tensor, sample_fps) in enumerate(zip(processed_videos, sample_fps_list, strict=False)):
            time_instruction = _build_time_instruction(video_tensor, sample_fps)
            prompt_with_video_pad = prompt_with_video_pad.replace(
                VISION_PLACEHOLDER,
                "<|vision_start|>"
                + "<|video_pad|>" * int(video_grid_thw[video_idx].prod().item() // merge_length)
                + "<|vision_end|>"
                + time_instruction,
                1,
            )
            raw_prompt = raw_prompt.replace(
                VISION_PLACEHOLDER,
                VISION_PLACEHOLDER + time_instruction,
                1,
            )

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
        uid = row.get("uid", row.get("id", f"{Path(row['_data_file']).stem}_{row['_local_idx']}"))

        raw_multi_modal_metadata = {}
        if raw_video_tensors:
            raw_multi_modal_metadata = {
                "video": raw_video_tensors[0],
                "fps": raw_sample_fps_list[0],
                "length": raw_video_tensors[0].shape[0] / max(raw_sample_fps_list[0], 1e-6),
            }

        extra_info = dict(row.get("extra_info", {}))
        extra_info.setdefault("resolved_video_path", resolved_videos[0] if resolved_videos else None)
        extra_info.setdefault("resolved_video_paths", resolved_videos)
        extra_info.setdefault("video_sample_fps", sample_fps_list[0] if sample_fps_list else None)
        extra_info.setdefault("video_sample_fps_list", sample_fps_list)

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "multi_modal_data": multi_modal_data,
            "multi_modal_inputs": dict(video_inputs),
            "raw_prompt": prompt_messages if self.return_raw_chat else raw_prompt,
            "ground_truth": row["ground_truth"],
            "data_source": row.get("source", "qwenvl_video_grpo"),
            "uid": uid,
            "index": uid,
            "resolved_video_path": resolved_videos[0] if resolved_videos else None,
            "video_sample_fps": sample_fps_list[0] if sample_fps_list else None,
            "video_fps_used": video_fps_used,
            "raw_multi_modal_metadata": raw_multi_modal_metadata,
            "extra_info": extra_info,
        }
