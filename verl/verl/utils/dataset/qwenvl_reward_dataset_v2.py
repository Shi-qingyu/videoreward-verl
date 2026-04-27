from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset

from verl.utils.model import compute_position_id_with_mask


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


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


def _normalize_text_content(text: str | None) -> str:
    return "" if text is None else str(text)


def _get_batch_value(batch: Any, key: str, default: Any = None) -> Any:
    if isinstance(batch, dict):
        return batch.get(key, default)
    return getattr(batch, key, default)


def _build_time_instruction(video_metadata: dict[str, Any], video_grid_thw: torch.Tensor, sample_fps: float, temporal_patch_size: int) -> str:
    """Build the same style of time instruction as the SFT dataset.

    In Qwen-VL processors, video_grid_thw[:, 0] is the temporal grid count after
    temporal patching, so the displayed frame count should be T * temporal_patch_size.
    """
    total_frames = int(video_grid_thw[0].item() * temporal_patch_size)
    duration = float(video_metadata.get("duration", total_frames / max(sample_fps, 1e-6)))
    return (
        f"This video is uniformly sampled at {sample_fps:.2f} fps, contains {total_frames} frames "
        f"from 0 seconds to {duration:.1f} seconds."
    )


def _build_messages(
    item: dict[str, Any],
    *,
    file_dir: Path,
    base_media_dir: str | None = None,
    time_instruction: str = "",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build Qwen-VL chat messages.

    Video placeholders are kept as structured video blocks and are intentionally
    not expanded into repeated <|video_pad|> tokens here. Expansion and actual
    video processing are delegated to processor.apply_chat_template(...,
    tokenize=True), matching the SFT path.
    """
    images = item.get("images") or []
    if isinstance(images, str):
        images = [images]
    if images:
        raise NotImplementedError("QwenVLRewardDataset currently supports video samples only.")

    videos = item.get("videos") or []
    if isinstance(videos, str):
        videos = [videos]
    resolved_videos = [_resolve_media_path(file_dir, video, base_media_dir) for video in videos]

    video_pool = [{"type": "video", "video": video_path} for video_path in resolved_videos]
    messages: list[dict[str, Any]] = []
    injected_time_instruction = False

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
            elif segment.strip():
                segment_text = segment.strip()
                if time_instruction and not injected_time_instruction:
                    segment_text = f"{time_instruction}\n{segment_text}"
                    injected_time_instruction = True
                content.append({"type": "text", "text": segment_text})

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


def _to_2d_tensor(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected {name} to be a 1D or 2D tensor, got shape {tuple(tensor.shape)}")
    return tensor


def _truncate_or_error(input_ids: torch.Tensor, attention_mask: torch.Tensor, max_length: int, truncation: str) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = input_ids.shape[-1]
    if seq_len <= max_length:
        return input_ids, attention_mask

    if truncation in {"error", "raise", None}:
        raise ValueError(f"Prompt length {seq_len} exceeds max_prompt_length={max_length}.")
    if truncation in {"left", "left_pad"}:
        return input_ids[:, -max_length:], attention_mask[:, -max_length:]
    if truncation in {"right", "right_pad"}:
        return input_ids[:, :max_length], attention_mask[:, :max_length]
    raise ValueError(f"Unsupported truncation mode: {truncation}")


class QwenVLRewardDataset(Dataset):
    def __init__(self, data_files, tokenizer, processor, config, max_samples: int = -1):
        self.data_files = _ensure_list(data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_samples = max_samples

        self.max_prompt_length = int(config.get("max_prompt_length", 16384))
        self.truncation = config.get("truncation", "error")
        self.return_raw_chat = bool(config.get("return_raw_chat", True))
        self.filter_overlong_prompts = bool(config.get("filter_overlong_prompts", False))
        self.use_3drope = bool(config.get("use_3drope", True))
        self.base_media_dir = config.get("base_media_dir")

        if self.processor is None:
            raise ValueError("QwenVLRewardDataset requires a processor.")
        if not hasattr(self.processor, "video_processor") or self.processor.video_processor is None:
            raise ValueError("QwenVLRewardDataset requires processor.video_processor.")

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

                # Build once without the dynamic time instruction so we can extract
                # static ground truth and paths during initialization. The final
                # prompt is rebuilt in __getitem__ after processor.video_processor
                # returns metadata, matching the SFT flow.
                messages, resolved_videos = _build_messages(
                    sample,
                    file_dir=file_path.parent,
                    base_media_dir=self.base_media_dir,
                    time_instruction="",
                )
                _, ground_truth = _extract_prompt_and_ground_truth(messages)
                sample["ground_truth"] = ground_truth
                sample["resolved_videos"] = resolved_videos
                rows.append(sample)

        if self.filter_overlong_prompts:
            filtered_rows = []
            for row in rows:
                messages, _ = _build_messages(
                    row,
                    file_dir=Path(row["_data_dir"]),
                    base_media_dir=self.base_media_dir,
                    time_instruction="",
                )
                prompt_messages, _ = _extract_prompt_and_ground_truth(messages)
                rendered = self.processor.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
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
        resolved_videos = row["resolved_videos"]
        if not resolved_videos:
            raise ValueError("QwenVLRewardDataset expects at least one video per sample.")

        sample_fps = float(getattr(self.processor.video_processor, "fps", row.get("fps", 2.0)))
        temporal_patch_size = int(getattr(self.processor.video_processor, "temporal_patch_size", 2))

        # SFT-style video handling: let the processor load/sample/resize video and
        # return both metadata and grid information. We do not call any custom video
        # decoding or adaptive token function here.
        vp_output = self.processor.video_processor(videos=resolved_videos, return_metadata=True)
        video_metadata_list = _get_batch_value(vp_output, "video_metadata")
        video_grid_thw = _get_batch_value(vp_output, "video_grid_thw")
        if video_metadata_list is None or video_grid_thw is None:
            raise ValueError("processor.video_processor(..., return_metadata=True) must return video_metadata and video_grid_thw.")
        if not isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = torch.tensor(video_grid_thw)

        time_instruction = _build_time_instruction(
            video_metadata=video_metadata_list[0],
            video_grid_thw=video_grid_thw[0],
            sample_fps=sample_fps,
            temporal_patch_size=temporal_patch_size,
        )

        messages, _ = _build_messages(
            row,
            file_dir=Path(row["_data_dir"]),
            base_media_dir=self.base_media_dir,
            time_instruction=time_instruction,
        )
        prompt_messages, ground_truth = _extract_prompt_and_ground_truth(messages)

        raw_prompt = self.processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        full_result = self.processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = _to_2d_tensor(full_result["input_ids"], name="input_ids")
        if "attention_mask" in full_result:
            attention_mask = _to_2d_tensor(full_result["attention_mask"], name="attention_mask")
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        input_ids, attention_mask = _truncate_or_error(
            input_ids,
            attention_mask,
            max_length=self.max_prompt_length,
            truncation=self.truncation,
        )

        full_video_grid_thw = full_result.get("video_grid_thw", video_grid_thw)
        if not isinstance(full_video_grid_thw, torch.Tensor):
            full_video_grid_thw = torch.tensor(full_video_grid_thw)

        second_per_grid_ts = full_result.get("second_per_grid_ts")
        if second_per_grid_ts is None:
            second_per_grid_ts = [temporal_patch_size / max(sample_fps, 1e-6)] * int(full_video_grid_thw.shape[0])

        if self.use_3drope and self.processor is not None:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=full_result.get("image_grid_thw"),
                    video_grid_thw=full_video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        uid = row.get("uid", row.get("id", f"{Path(row['_data_file']).stem}_{row['_local_idx']}"))

        multi_modal_inputs = {
            key: value
            for key, value in dict(full_result).items()
            if key not in {"input_ids", "attention_mask"}
        }
        multi_modal_inputs.setdefault("video_grid_thw", full_video_grid_thw)
        multi_modal_inputs.setdefault("second_per_grid_ts", second_per_grid_ts)

        video_sample_fps_list = [sample_fps] * len(resolved_videos)
        extra_info = dict(row.get("extra_info", {}))
        extra_info.setdefault("resolved_video_path", resolved_videos[0])
        extra_info.setdefault("resolved_video_paths", resolved_videos)
        extra_info.setdefault("video_sample_fps", video_sample_fps_list[0])
        extra_info.setdefault("video_sample_fps_list", video_sample_fps_list)
        extra_info.setdefault("time_instruction", time_instruction)

        return {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "multi_modal_data": {"video": resolved_videos},
            "multi_modal_inputs": multi_modal_inputs,
            "raw_prompt": prompt_messages if self.return_raw_chat else raw_prompt,
            "ground_truth": ground_truth,
            "reward_model": {
                "ground_truth": ground_truth,
            },
            "data_source": row.get("source", "qwenvl_reward_grpo"),
            "uid": uid,
            "index": uid,
            "resolved_video_path": resolved_videos[0],
            "video_sample_fps": video_sample_fps_list[0],
            "video_fps_used": {"fps": video_sample_fps_list},
            "raw_multi_modal_metadata": {
                "video": resolved_videos[0],
                "fps": video_sample_fps_list[0],
                "length": float(video_metadata_list[0].get("duration", 0.0)),
            },
            "extra_info": extra_info,
        }
