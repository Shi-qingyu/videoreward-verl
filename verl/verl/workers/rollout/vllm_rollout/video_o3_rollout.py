from __future__ import annotations

import copy
import json
import re
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from vllm.lora.request import LoRARequest

from verl import DataProto
from verl.utils.dataset.task_prompt import ERROR_INFO_MULTI_TURN_PROMPT
from my_qwen_vl_utils.video_o3_vision_process import fetch_video
from verl.utils.torch_functional import get_final_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import _pre_process_inputs, vLLMRollout


class vLLMVideoO3Rollout(vLLMRollout):
    def __init__(self, config, model_config, device_mesh):
        super().__init__(config=config, model_config=model_config, device_mesh=device_mesh)
        self.tokenizer = model_config.tokenizer
        self.max_generation_round = int(self.config.get("max_generation_round", 4))
        self.vllm_infer_batch_size = int(self.config.get("vllm_infer_batch_size", 8))
        self.max_pixels = int(self.config.get("max_pixels", 16384 * 28 * 28))
        self.min_pixels = int(self.config.get("min_pixels", 512 * 28 * 28))
        self.max_observation_frames = int(self.config.get("max_observation_frames", 12))
        self.strategy_fps = {"coarse": 1.0, "medium": 2.0, "fine": 4.0}
        im_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        self.im_end_token_ids = im_end_ids if im_end_ids else [self.tokenizer.eos_token_id]
        self.im_end_token_id = self.im_end_token_ids[0]

    def _build_lora_requests(self, batch_size: int):
        if not self.lora_kwargs:
            return None
        lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
        if not lora_int_ids:
            return None
        lora_int_id = lora_int_ids[0]
        return [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

    def _generate_once(self, vllm_inputs: list[dict[str, Any]], sampling_overrides: dict[str, Any], lora_requests):
        responses = []
        with self.update_sampling_params(**sampling_overrides):
            for start in range(0, len(vllm_inputs), self.vllm_infer_batch_size):
                cur_inputs = vllm_inputs[start : start + self.vllm_infer_batch_size]
                cur_lora = None if lora_requests is None else lora_requests[start : start + self.vllm_infer_batch_size]
                outputs = self.inference_engine.generate(
                    prompts=cur_inputs,
                    sampling_params=self.sampling_params,
                    lora_request=cur_lora,
                    use_tqdm=False,
                )
                for output in outputs:
                    response_ids = list(output.outputs[0].token_ids)
                    if self.im_end_token_id not in response_ids:
                        response_ids.append(self.im_end_token_id)
                    responses.append(response_ids)
        return responses

    def _parse_grounding(self, text: str):
        matches = re.findall(r"<grounding>(.*?)</grounding>", text, re.DOTALL)
        if not matches:
            return None
        try:
            payload = json.loads(matches[-1])
        except Exception as exc:
            return str(exc)
        segment = payload.get("temporal_segment")
        strategy = payload.get("sampling_strategy", "medium")
        if not isinstance(segment, list) or len(segment) != 2:
            return "temporal_segment must be a list of length 2"
        if strategy not in self.strategy_fps:
            return f"invalid sampling_strategy: {strategy}"
        start, end = float(segment[0]), float(segment[1])
        if end <= start:
            return "temporal_segment must satisfy end > start"
        return {"start": start, "end": end, "sampling_strategy": strategy}

    def _sample_clip_frames(self, video_path: str, start: float, end: float, strategy: str):
        video_tensor, sample_fps = fetch_video(
            {
                "video": video_path,
                "video_start": start,
                "video_end": end,
                "fps": self.strategy_fps[strategy],
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            },
            return_video_sample_fps=True,
        )
        total = video_tensor.shape[0]
        if total > self.max_observation_frames:
            indices = torch.linspace(0, total - 1, self.max_observation_frames).round().long().tolist()
            frames = [video_tensor[i] for i in indices]
        else:
            frames = [video_tensor[i] for i in range(total)]
        return frames, sample_fps

    def _build_observation_prompt(self, turn_idx: int, clip_start: float, sample_fps: float, nframes: int):
        lines = [f"After the above Action {turn_idx}, here is the refined video clip (Observation {turn_idx + 1}):"]
        for i in range(nframes):
            timestamp = clip_start + i / max(sample_fps, 1e-6)
            lines.append(f"Frame {i + 1} at {timestamp:.1f}s: <|vision_start|><|image_pad|><|vision_end|>")
        lines.append(
            "Continue your reasoning process inside <think> and </think>. If needed, you can keep selecting temporal "
            "segments from the original video by outputting <grounding> and </grounding> as before. Once you are ready "
            "to provide the final answer, put it inside <answer> and </answer>."
        )
        return "<|im_start|>user\n" + "\n".join(lines) + "\n<|im_end|>\n<|im_start|>assistant\n"

    def _build_error_prompt(self, error_message: str):
        return (
            "<|im_start|>user\n"
            f"ERROR occurs during grounding. Error Information: {error_message}.\n{ERROR_INFO_MULTI_TURN_PROMPT}"
            "<|im_end|>\n<|im_start|>assistant\n"
        )

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.size(0)
        non_tensor_batch = prompts.non_tensor_batch

        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object,
            )

        vllm_inputs = []
        prefix_prompt_lengths = []
        response_segments = []
        response_masks = []
        active_indices = list(range(batch_size))
        resolved_video_paths = list(non_tensor_batch["resolved_video_path"])
        raw_prompt_ids_list = list(non_tensor_batch["raw_prompt_ids"])
        multi_modal_data_list = list(non_tensor_batch["multi_modal_data"])
        for raw_prompt_ids, multi_modal_data in zip(raw_prompt_ids_list, multi_modal_data_list, strict=True):
            prompt_ids = list(raw_prompt_ids)
            vllm_inputs.append({"prompt_token_ids": prompt_ids, "multi_modal_data": copy.deepcopy(multi_modal_data)})
            prefix_prompt_lengths.append(len(prompt_ids))
            response_segments.append([])
            response_masks.append([])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        sampling_overrides = {}
        if not do_sample:
            sampling_overrides = {"best_of": 1, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "temperature": 0, "n": 1}
        elif is_validate:
            sampling_overrides = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,
            }
        lora_requests = self._build_lora_requests(batch_size)

        for turn_idx in range(self.max_generation_round):
            if not active_indices:
                break
            gen_inputs = [vllm_inputs[i] for i in active_indices]
            gen_lora_requests = None if lora_requests is None else [lora_requests[i] for i in active_indices]
            responses = self._generate_once(gen_inputs, sampling_overrides, gen_lora_requests)
            next_active_indices = []
            for active_idx, response_ids in zip(active_indices, responses, strict=True):
                vllm_inputs[active_idx]["prompt_token_ids"].extend(response_ids)
                response_segments[active_idx].extend(response_ids)
                response_masks[active_idx].extend([1] * len(response_ids))
                decoded = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                if "<answer>" in decoded and "</answer>" in decoded:
                    continue
                grounding_payload = self._parse_grounding(decoded)
                if grounding_payload is None:
                    continue
                if isinstance(grounding_payload, str):
                    error_prompt = self._build_error_prompt(grounding_payload)
                    error_ids = self.tokenizer.encode(error_prompt, add_special_tokens=False)
                    vllm_inputs[active_idx]["prompt_token_ids"].extend(error_ids)
                    response_segments[active_idx].extend(error_ids)
                    response_masks[active_idx].extend([0] * len(error_ids))
                    next_active_indices.append(active_idx)
                    continue
                try:
                    frames, sample_fps = self._sample_clip_frames(
                        resolved_video_paths[active_idx],
                        grounding_payload["start"],
                        grounding_payload["end"],
                        grounding_payload["sampling_strategy"],
                    )
                    observation_prompt = self._build_observation_prompt(
                        turn_idx,
                        grounding_payload["start"],
                        sample_fps,
                        len(frames),
                    )
                    observation_ids = self.tokenizer.encode(observation_prompt, add_special_tokens=False)
                    vllm_inputs[active_idx]["prompt_token_ids"].extend(observation_ids)
                    vllm_inputs[active_idx]["multi_modal_data"]["image"].extend(frames)
                    response_segments[active_idx].extend(observation_ids)
                    response_masks[active_idx].extend([0] * len(observation_ids))
                    next_active_indices.append(active_idx)
                except Exception as exc:
                    error_prompt = self._build_error_prompt(str(exc))
                    error_ids = self.tokenizer.encode(error_prompt, add_special_tokens=False)
                    vllm_inputs[active_idx]["prompt_token_ids"].extend(error_ids)
                    response_segments[active_idx].extend(error_ids)
                    response_masks[active_idx].extend([0] * len(error_ids))
                    next_active_indices.append(active_idx)
            active_indices = next_active_indices

        truncated_responses = [seq[: self.config.response_length] for seq in response_segments]
        truncated_masks = [seq[: self.config.response_length] for seq in response_masks]
        response = pad_2d_list_to_length(truncated_responses, self.pad_token_id, max_length=self.config.response_length).to(
            idx.device
        )
        response_generation_mask = pad_2d_list_to_length(truncated_masks, 0, max_length=self.config.response_length).to(
            idx.device
        )
        seq = torch.cat([idx, response], dim=-1)
        response_attention_mask = get_final_eos_mask(
            response,
            eos_token=self.im_end_token_ids,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        delta_position_id = torch.arange(1, response.size(1) + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        input_prompt_generation_mask = torch.zeros_like(idx, dtype=attention_mask.dtype, device=attention_mask.device)
        multi_turn_response_mask = torch.cat([input_prompt_generation_mask, response_generation_mask], dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "multi_turn_response_mask": multi_turn_response_mask,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
