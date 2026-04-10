from pathlib import Path
import sys
import unittest


REPO_VERL_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_VERL_ROOT))

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.video_o3_reward import compute_score


class VideoO3RewardTest(unittest.TestCase):
    def test_tool_reward_without_round_penalty_matches_upstream_bonus_logic(self):
        prompt = "Question"
        responses = [
            '<think>inspect</think><grounding>{"temporal_segment": [0, 10], "sampling_strategy": "medium"}</grounding>',
            "<think>done</think><answer>A</answer>",
        ]
        ground_truth = {"answer": "A", "clue": [{"timestamp": [0, 10], "text": ""}]}
        extra_info = {"acc_reward_weight": 1.0, "format_reward_weight": 1.0}

        score, acc_score, format_score = compute_score(
            prompt=prompt,
            data_source="tool_reward_without_round_penalty",
            solution_str=responses,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        self.assertAlmostEqual(score, 3.0)
        self.assertAlmostEqual(acc_score, 1.0)
        self.assertAlmostEqual(format_score, 1.0)

        default_score = default_compute_score(
            "tool_reward_without_round_penalty",
            responses,
            ground_truth,
            extra_info=extra_info,
            prompt=prompt,
        )
        self.assertAlmostEqual(default_score, 3.0)

    def test_cgbench_route_matches_upstream_test_temporal_grounded_qa(self):
        prompt = "Question"
        responses = ["<think>done</think><answer>A</answer>"]
        ground_truth = {"answer": "A", "clue": [{"timestamp": [0, 10], "text": ""}]}
        extra_info = {"acc_reward_weight": 1.0, "format_reward_weight": 1.0, "gpt_extract_answer": True}

        score, acc_score, format_score = compute_score(
            prompt=prompt,
            data_source="cgbench",
            solution_str=responses,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        self.assertAlmostEqual(score, 1.0)
        self.assertAlmostEqual(acc_score, 1.0)
        self.assertAlmostEqual(format_score, 1e-9)

        default_score = default_compute_score(
            "cgbench",
            responses,
            ground_truth,
            extra_info=extra_info,
            prompt=prompt,
        )
        self.assertAlmostEqual(default_score, 1.0)


if __name__ == "__main__":
    unittest.main()
