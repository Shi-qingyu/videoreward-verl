import importlib.util
from pathlib import Path
import sys
import types
import unittest

rouge_score_module = types.ModuleType("rouge_score")
rouge_scorer_module = types.ModuleType("rouge_scorer")


class _DummyRougeScorer:
    def __init__(self, *args, **kwargs):
        pass

    def score(self, *args, **kwargs):
        raise NotImplementedError("ROUGE scoring is not needed in these tests")


rouge_scorer_module.RougeScorer = _DummyRougeScorer
rouge_score_module.rouge_scorer = rouge_scorer_module
sys.modules.setdefault("rouge_score", rouge_score_module)

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "verl"
    / "utils"
    / "reward_score"
    / "openo3_video_reward.py"
)
SPEC = importlib.util.spec_from_file_location("openo3_video_reward", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

_ans_viou_reward = MODULE._ans_viou_reward
_thk_spatial_reward = MODULE._thk_spatial_reward


class OpenO3VideoRewardTest(unittest.TestCase):
    def test_ans_viou_reward_resizes_visual_gt_to_qwen3_space(self):
        solution = "<think></think><answer><box>[100, 100, 500, 500]</box></answer>"
        ground_truth = "<box>[20, 10, 100, 50]</box>"

        score = _ans_viou_reward(
            solution,
            ground_truth,
            "visual QA",
            {
                "image_size": (200, 100),
                "image_size_refine": (448, 448),
            },
        )

        self.assertAlmostEqual(score, 1.0)

    def test_thk_spatial_reward_resizes_visual_gt_to_qwen3_space(self):
        solution = "<think><obj>cat</obj><box>[100, 100, 500, 500]</box></think><answer><box>[100, 100, 500, 500]</box></answer>"
        ground_truth = "<box>[20, 10, 100, 50]</box>"

        score = _thk_spatial_reward(
            solution,
            ground_truth,
            "visual QA",
            {
                "image_size": (200, 100),
                "image_size_refine": (448, 448),
            },
        )

        self.assertAlmostEqual(score, 1.0)

    def test_thk_spatial_reward_resizes_temporal_spatial_gt_to_qwen3_space(self):
        solution = (
            "<think><obj>person</obj><box>[100, 100, 500, 500]</box>at<t>2.5</t>s</think>"
            "<answer>person</answer>"
        )

        score = _thk_spatial_reward(
            solution,
            "unused",
            "temporal-spatial free-form QA",
            {
                "image_size": (200, 100),
                "key_frames": [{"time": 2.5, "idx": 0}],
                "key_items": {"0": {"person": [[0.1, 0.1, 0.5, 0.5]]}},
            },
        )

        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
