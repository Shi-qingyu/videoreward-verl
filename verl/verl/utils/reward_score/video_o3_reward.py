from __future__ import annotations

from verl.utils.reward_score import compute_video_score


def compute_score(prompt: str, data_source: str, solution_str, ground_truth, extra_info: dict | None = None, **kwargs):
    return compute_video_score(
        prompt=prompt,
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
