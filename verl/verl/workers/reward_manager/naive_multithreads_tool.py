from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive_multithreads_tool")
class NaiveMultiThreadsToolRewardManager(AbstractRewardManager):
    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.extra_info = dict(kwargs.get("extra_info", {}))
        self.gpt_threads = int(kwargs.get("gpt_threads", 16))
        self.overlong_buffer_len = int(self.extra_info.get("overlong_buffer_len", 0))
        self.max_total_response_length = self.extra_info.get("max_total_response_length")
        self.default_num_examine = int(self.extra_info.get("log_samples_per_data_source", 5))

    def extract_responses_list(self, input_ids: torch.Tensor, multi_turn_response_mask: torch.Tensor) -> list[str]:
        diff = torch.diff(multi_turn_response_mask, prepend=torch.tensor([0], device=multi_turn_response_mask.device))
        starts = torch.where(diff == 1)[0]
        mask_appended = torch.cat([multi_turn_response_mask, torch.tensor([0], device=multi_turn_response_mask.device)], dim=0)
        diff_end = torch.diff(mask_appended)
        ends = torch.where(diff_end == -1)[0]
        segments = [input_ids[s : e + 1].tolist() for s, e in zip(starts, ends, strict=False)]
        return self.tokenizer.batch_decode(segments, skip_special_tokens=True)

    def _process_single(self, index: int, data_item):
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        if "multi_turn_response_mask" in data_item.batch:
            response_str = self.extract_responses_list(
                data_item.batch["input_ids"],
                data_item.batch["multi_turn_response_mask"],
            )
        else:
            response_str = [self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)]

        ground_truth = data_item.non_tensor_batch["ground_truth"]
        data_source = data_item.non_tensor_batch[self.reward_fn_key]
        sample_extra_info = dict(self.extra_info)
        sample_extra_info.update(data_item.non_tensor_batch.get("extra_info", {}))
        question = data_item.non_tensor_batch.get("raw_prompt", prompt_str)
        result = self.compute_score(
            prompt=question,
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=sample_extra_info,
        )
        if isinstance(result, dict):
            return index, 0.0, 0.0, 0.0, 0.0, bool(result.get("is_filter", False)), prompt_str, response_str, ground_truth

        score, acc_score, format_score = result
        overlong_reward = 0.0
        if self.overlong_buffer_len > 0 and self.max_total_response_length is not None:
            expected_len = self.max_total_response_length - self.overlong_buffer_len
            exceed_len = int(valid_response_length) - expected_len
            overlong_reward = min(-exceed_len / max(self.overlong_buffer_len, 1), 0.0)
            score += overlong_reward
        return index, float(score), float(acc_score), float(format_score), float(overlong_reward), False, prompt_str, response_str, ground_truth

    def __call__(self, data: DataProto, return_dict: bool = False, iteration: int | None = None):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": {}}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        acc_scores = [0.0] * len(data)
        format_scores = [0.0] * len(data)
        overlong_scores = [0.0] * len(data)
        valid_mask = [True] * len(data)
        already_print_data_sources: dict[str, int] = {}
        print_limit = self.num_examine if self.num_examine > 0 else self.default_num_examine

        with ThreadPoolExecutor(max_workers=self.gpt_threads) as executor:
            futures = [executor.submit(self._process_single, i, data[i]) for i in range(len(data))]
            results = [future.result() for future in as_completed(futures)]

        for index, score, acc_score, format_score, overlong_score, is_invalid, prompt_str, response_str, ground_truth in sorted(results):
            valid_response_length = int(data[index].batch["attention_mask"][data[index].batch["prompts"].shape[-1] :].sum().item())
            if valid_response_length > 0:
                reward_tensor[index, valid_response_length - 1] = score
            acc_scores[index] = acc_score
            format_scores[index] = format_score
            overlong_scores[index] = overlong_score
            valid_mask[index] = not is_invalid
            data_source = data[index].non_tensor_batch[self.reward_fn_key]
            already_print_data_sources.setdefault(data_source, 0)
            if not is_invalid and already_print_data_sources[data_source] < print_limit:
                already_print_data_sources[data_source] += 1
                response_text = " ".join(response_str) if isinstance(response_str, list) else response_str
                current_pid = os.getpid()
                base_dir = os.getenv("LOG_SAVE_DIR", "./ckpt/RL")
                log_path = os.path.join(base_dir, f"reward_manager/{current_pid}.txt")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"[prompt]: {prompt_str}\n")
                    f.write(f"[response]: {response_text}\n")
                    f.write(f"[ground_truth]: {ground_truth}\n")
                    f.write(f"[score]: {(score, acc_score, format_score)}\n")
                    f.write(f"[iteration]: {iteration}\n")
                    f.write(f"[data_source]: {data_source}\n")
                    f.write("----------------------------------------------------------------\n\n")
                print(
                    f"<<< [score]: {score} [acc_score]: {acc_score} [format_score]: {format_score} "
                    f"[iteration]: {iteration} [data_source]: {data_source} >>>"
                )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "acc_scores": acc_scores,
                    "format_scores": format_scores,
                    "overlong_scores": overlong_scores,
                    "valid_mask": valid_mask,
                },
            }
        return reward_tensor
