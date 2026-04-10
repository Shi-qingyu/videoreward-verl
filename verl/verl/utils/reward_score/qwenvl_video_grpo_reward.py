from __future__ import annotations

import json
import re
from typing import Any

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError("qwenvl_video_grpo_reward.py requires scipy to compute matched IoU.") from exc


EXPECTED_KEYS = [
    "Video Quality",
    "Subject Movement",
    "Physical Interaction",
    "Cause-Effect",
    "Subject Existence",
    "Object Existence",
    "Subject-Object Interaction",
]


def normalize_label(value: str | None) -> str:
    if value is None:
        return "fail"

    text = str(value).strip().lower().rstrip("。").rstrip(".")
    text = re.split(r"[\s,;:]+", text)[0] if text else "fail"

    yes_set = {
        "yes",
        "good",
        "true",
        "correct",
        "present",
        "exists",
        "aligned",
        "match",
        "matched",
        "plausible",
    }
    no_set = {
        "no",
        "bad",
        "false",
        "incorrect",
        "absent",
        "missing",
        "misaligned",
        "mismatch",
        "implausible",
    }

    if text in yes_set:
        return "yes"
    if text in no_set:
        return "no"
    return text if text else "fail"


def extract_tag_content(text: str, tag: str) -> list[str]:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    matches = re.findall(pattern, "" if text is None else str(text), re.S | re.I)
    return [match.strip() for match in matches] if matches else [""]


def parse_box(box_str: str) -> list[int]:
    return json.loads(box_str)


def compute_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def mean_matched_iou(gt_boxes: list[list[int]], pred_boxes: list[list[int]]) -> float:
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0

    iou_matrix = [[compute_iou(gt_box, pred_box) for pred_box in pred_boxes] for gt_box in gt_boxes]
    cost_matrix = [[-iou for iou in row] for row in iou_matrix]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_ious = [iou_matrix[r][c] for r, c in zip(row_ind, col_ind, strict=False)]
    return float(sum(matched_ious) / max(len(gt_boxes), len(pred_boxes)))


def parse_answer_block(answer_text: str) -> dict[str, str]:
    answer_dict: dict[str, str] = {}
    for key in EXPECTED_KEYS:
        pattern = rf"(?:^|\n)\s*{re.escape(key)}\s*:\s*([^\n]+)"
        match = re.search(pattern, answer_text, re.I)
        if match:
            answer_dict[key] = match.group(1).strip().rstrip(".").strip()
        else:
            answer_dict[key] = "Fail"
    return answer_dict


def parse_output(text: str) -> tuple[str, dict[str, str]]:
    normalized = "" if text is None else str(text)
    think_content = extract_tag_content(normalized, "think")[0]
    answer_text = extract_tag_content(normalized, "answer")[0]
    answer_dict = parse_answer_block(answer_text)
    return think_content, answer_dict


def acc_reward(solution_str: str, ground_truth: str) -> float:
    _, pred_dict = parse_output(solution_str)
    _, gt_dict = parse_output(ground_truth)

    matched = 0
    for key in EXPECTED_KEYS:
        if normalize_label(pred_dict[key]) == normalize_label(gt_dict[key]):
            matched += 1
    return matched / len(EXPECTED_KEYS)


def format_reward(solution_str: str) -> float:
    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    return 1.0 if re.match(pattern, "" if solution_str is None else str(solution_str), re.S | re.I) else 0.0


def iou_reward(solution_str: str, ground_truth: str) -> float:
    gt_boxes_raw = extract_tag_content(ground_truth, "region")
    if gt_boxes_raw[0] == "":
        return 1.0

    pred_boxes_raw = extract_tag_content(solution_str, "region")
    if pred_boxes_raw[0] == "":
        return 0.0

    try:
        gt_boxes = [parse_box(box_str) for box_str in gt_boxes_raw]
        pred_boxes = [parse_box(box_str) for box_str in pred_boxes_raw]
    except Exception:
        return 0.0
    return mean_matched_iou(gt_boxes, pred_boxes)


def compute_score(
    solution_str: str,
    ground_truth: str,
    data_source: str = "",
    extra_info: dict[str, Any] | None = None,
    acc_weight: float = 1.0,
    format_weight: float = 1.0,
    iou_weight: float = 1.0,
):
    del data_source, extra_info

    acc = acc_reward(solution_str, ground_truth)
    fmt = format_reward(solution_str)
    iou = iou_reward(solution_str, ground_truth)
    score = acc_weight * acc + format_weight * fmt + iou_weight * iou

    return {
        "score": float(score),
        "acc_reward": float(acc),
        "format_reward": float(fmt),
        "iou_reward": float(iou),
    }
