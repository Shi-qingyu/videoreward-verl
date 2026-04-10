# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def compute_video_score(prompt, data_source, solution_str, ground_truth, extra_info=None):
    if data_source in ["vstar_bench", "visual_probe_easy", "visual_probe_medium", "visual_probe_hard", "visual_probe_train", "deepeyes_train"]:
        reward_fn = extra_info["general_qa_reward_fn"]
        if reward_fn == "general_qa_tool":
            from . import general_qa_tool

            return general_qa_tool.compute_score(prompt, solution_str, ground_truth, extra_info)
        if reward_fn == "general_qa_tool_qwen":
            from . import general_qa_tool_qwen

            return general_qa_tool_qwen.compute_score(prompt, solution_str, ground_truth, extra_info)
        if reward_fn == "general_qa_tool_mc":
            from . import general_qa_tool_mc

            return general_qa_tool_mc.compute_score(prompt, solution_str, ground_truth, extra_info)
        raise NotImplementedError
    if data_source in ["tool_reward_without_round_penalty", "clue_multi_w_tool"]:
        from . import tool_reward_without_round_penalty

        return tool_reward_without_round_penalty.compute_score(prompt, solution_str, ground_truth, extra_info)
    if data_source in ["tool_free_form", "clue_multi_wo_tool"]:
        from . import tool_free_form

        return tool_free_form.compute_score(prompt, solution_str, ground_truth, extra_info)
    if data_source in ["tool_reward_with_round_penalty", "clue_single_w_tool"]:
        from . import tool_reward_with_round_penalty

        return tool_reward_with_round_penalty.compute_score(prompt, solution_str, ground_truth, extra_info)
    if data_source in ["tool_penalty_single_turn", "clue_single_wo_tool"]:
        from . import tool_penalty_single_turn

        return tool_penalty_single_turn.compute_score(prompt, solution_str, ground_truth, extra_info)
    if data_source in ["temporal_grounded_qa"]:
        from . import temporal_grounded_qa

        return temporal_grounded_qa.compute_score(prompt, solution_str, ground_truth, extra_info)
    if data_source in ["temporal_grounded_qa_iou"]:
        from . import temporal_grounded_qa_iou

        return temporal_grounded_qa_iou.compute_score(prompt, solution_str, ground_truth, extra_info)
    if data_source in [
        "test_temporal_grounded_qa",
        "nextgqa",
        "mlvu",
        "videomme",
        "cgbench",
        "vrbench",
        "videommmu",
        "lvbench",
        "longvideobench",
        "mmvu",
        "videoholmes",
        "longvideoreason",
    ]:
        from . import test_temporal_grounded_qa

        return test_temporal_grounded_qa.compute_score(prompt, solution_str, ground_truth, extra_info)
    if data_source in ["temporal_grounding"]:
        from . import temporal_grounding

        return temporal_grounding.compute_score(prompt, solution_str, ground_truth, extra_info)
    if data_source in ["test_temporal_grounding", "activitynet", "charades"]:
        from . import test_temporal_grounding

        return test_temporal_grounding.compute_score(prompt, solution_str, ground_truth, extra_info)
    raise NotImplementedError


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"): 
        from . import math_dapo 

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "vstar_bench",
        "visual_probe_easy",
        "visual_probe_medium",
        "visual_probe_hard",
        "visual_probe_train",
        "deepeyes_train",
        "tool_reward_without_round_penalty",
        "clue_multi_w_tool",
        "tool_free_form",
        "clue_multi_wo_tool",
        "tool_reward_with_round_penalty",
        "clue_single_w_tool",
        "tool_penalty_single_turn",
        "clue_single_wo_tool",
        "temporal_grounded_qa",
        "temporal_grounded_qa_iou",
        "test_temporal_grounded_qa",
        "nextgqa",
        "mlvu",
        "videomme",
        "cgbench",
        "vrbench",
        "videommmu",
        "lvbench",
        "longvideobench",
        "mmvu",
        "videoholmes",
        "longvideoreason",
        "temporal_grounding",
        "test_temporal_grounding",
        "activitynet",
        "charades",
    ]:
        res = compute_video_score(kwargs["prompt"], data_source, solution_str, ground_truth, extra_info)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["compute_video_score", "default_compute_score"]
