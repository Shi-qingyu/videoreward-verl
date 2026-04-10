import re
import os
import json
import ast
import numpy as np
from datetime import datetime
from rouge_score import rouge_scorer

QWEN3_COORD_SPACE = (1000, 1000)


def compute_score(solution_str, ground_truth, data_source="", extra_info=None):
    """
    Follows the same input/output convention as default_compute_score in __init__.py.

    Args:
        solution_str (str): The model's generated output string.
        ground_truth (str): The ground truth answer string.
        data_source (str): Task type, e.g. "temporal QA", "visual QA",
                           "temporal QA (MCQ)", "General video QA MCQ",
                           "General video QA Free-form",
                           "temporal-spatial free-form QA", etc.
        extra_info (dict, optional): Additional info needed by some reward functions,

    Returns:
        float: The sum of all sub-reward scores.
    """
    # breakpoint() 
    if extra_info is None:
        extra_info = {}

    reward_fns = {
        "ans_acc": _ans_acc_reward,
        "ans_tiou": _ans_tiou_reward,
        "ans_viou": _ans_viou_reward,
        "thk_temporal_point": _thk_temporal_point_reward,
        "thk_temporal_segment": _thk_temporal_segment_reward,
        "thk_spatial": _thk_spatial_reward,
        "format": _format_reward,
    }

    total_score = 0.0
    for name, fn in reward_fns.items():
        try:
            score = fn(solution_str, ground_truth, data_source, extra_info)
            total_score += score
        except Exception as e:
            print(f"Error computing {name} reward: {e}")

    return total_score


def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def compute_rouge_score(reference, hypothesis, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
    scores = scorer.score(reference, hypothesis)
    average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
    return average_fmeasure


def parse_temporal_spatial_reasoning_process(think_content: str):
    pattern = r"<obj>(.*?)</obj>((?:<box>\[.*?\]</box>)+)at<t>(.*?)</t>s"
    parsed_claims = []
    count = 0

    for match in re.finditer(pattern, think_content, re.DOTALL):
        try:
            object_name = match.group(1).strip()
            all_boxes_str = match.group(2)
            timestamp_str = match.group(3).strip()
            timestamp = float(timestamp_str)

            individual_box_strs = re.findall(r'\[.*?\]', all_boxes_str)
            bboxes = [json.loads(b_str) for b_str in individual_box_strs]

            parsed_claims.append({
                "id": count,
                "object_name": object_name,
                "timestamp": timestamp,
                "bboxes": bboxes
            })
            count += 1
        except (json.JSONDecodeError, ValueError, IndexError):
            continue

    return parsed_claims


def convert_coord_format(bbox, image_size):
    nx_min, ny_min, nx_max, ny_max = bbox
    width, height = image_size
    x_min = nx_min * width
    y_min = ny_min * height
    x_max = nx_max * width
    y_max = ny_max * height
    return [x_min, y_min, x_max, y_max]


def convert_coord_format_gqa(bbox, image_size, image_size_refine):
    bbox = list(bbox)  # avoid mutating original
    bbox[0] = bbox[0] * image_size_refine[0] / image_size[0]
    bbox[1] = bbox[1] * image_size_refine[1] / image_size[1]
    bbox[2] = bbox[2] * image_size_refine[0] / image_size[0]
    bbox[3] = bbox[3] * image_size_refine[1] / image_size[1]
    return bbox


def resize_absolute_gt_box_to_qwen3_space(bbox, image_size):
    if image_size is None:
        return list(bbox)
    return convert_coord_format_gqa(bbox, image_size, QWEN3_COORD_SPACE)


def resize_normalized_gt_box_to_qwen3_space(bbox):
    return convert_coord_format(bbox, QWEN3_COORD_SPACE)


def calculate_iou(boxA, boxB):
    try:
        if not (isinstance(boxB, list) and len(boxB) == 4):
            return 0.0
        boxA_corners = np.array(boxA, dtype=float)
        boxB_corners = np.array(boxB, dtype=float)
    except (ValueError, TypeError, IndexError):
        return 0.0

    xA = max(boxA_corners[0], boxB_corners[0])
    yA = max(boxA_corners[1], boxB_corners[1])
    xB = min(boxA_corners[2], boxB_corners[2])
    yB = min(boxA_corners[3], boxB_corners[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA_corners[2] - boxA_corners[0]) * (boxA_corners[3] - boxA_corners[1])
    boxB_area = (boxB_corners[2] - boxB_corners[0]) * (boxB_corners[3] - boxB_corners[1])
    union_area = boxA_area + boxB_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ==================== Single-sample reward functions ====================
# Adapted from r1v_reward_func.py batch functions.
# Each function takes (solution_str, ground_truth, data_source, extra_info)
# and returns a single float reward.

def _ans_acc_reward(solution_str, ground_truth, data_source, extra_info):
    """Accuracy reward based on question type routing via data_source."""
    sol = f'<answer>{ground_truth}</answer>'

    question_type = "free-form"
    if data_source == "temporal QA (MCQ)":
        question_type = "TG_MCQ"
    elif data_source == "General video QA MCQ":
        question_type = "MCQ"
    elif data_source in ("visual QA", "temporal QA"):
        question_type = "none"

    try:
        output_ans = extract_answer(solution_str)
        gt_ans = extract_answer(sol)

        if question_type == "TG_MCQ":
            gt_ans = ground_truth.split("\n")[0]
            try:
                choice = output_ans.split("Correct Option:")[1]
                gt_ans = gt_ans.strip()
                gt_list = [gt_ans, gt_ans + '.', '(' + gt_ans + ')', '[' + gt_ans + ']']
                return 1.0 if choice.strip() in gt_list else 0.0
            except Exception:
                return 0.0
        elif question_type == "free-form":
            score = compute_rouge_score(gt_ans, output_ans)
            return max(0.0, min(1.0, score))
        elif question_type == "MCQ":
            choice = output_ans
            gt_ans = gt_ans.strip()
            gt_list = [gt_ans, gt_ans + '.', '(' + gt_ans + ')', '[' + gt_ans + ']']
            return 1.0 if choice.strip() in gt_list else 0.0
        else:
            return 0.0
    except Exception as e:
        print(f"Error in _ans_acc_reward for data_source '{data_source}': {e}")
        return 0.0


def _ans_tiou_reward(solution_str, ground_truth, data_source, extra_info):
    """Temporal IoU reward for temporal QA tasks."""
    question_type = "none"
    if data_source == "temporal QA":
        question_type = "TG"
    elif data_source == "temporal QA (MCQ)":
        question_type = "TG_MCQ"

    try:
        output_ans = extract_answer(solution_str)

        if question_type == "TG":
            gt_ans = ast.literal_eval(ground_truth)
            pattern = r"<t>(\d+\.?\d*)</t>s to <t>(\d+\.?\d*)</t>s"
            match = re.search(pattern, output_ans)
            times = []
            if match:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                if end_time >= start_time:
                    times = [start_time, end_time]

            if len(times) == 2:
                start1, end1 = times
                start2, end2 = gt_ans
                intersection_start = max(start1, start2)
                intersection_end = min(end1, end2)
                intersection_length = max(0, intersection_end - intersection_start)
                union_length = max(end1, end2) - min(start1, start2)
                return intersection_length / union_length if union_length != 0 else 0.0
            return 0.0

        elif question_type == "TG_MCQ":
            gt_ans = ground_truth.split("\n")[1]
            gt_ans = ast.literal_eval(gt_ans)
            pattern = r"<t>(\d+\.?\d*)</t>s to <t>(\d+\.?\d*)</t>s"
            match = re.search(pattern, output_ans)
            times = []
            if match:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                if end_time >= start_time:
                    times = [start_time, end_time]

            if len(times) == 2:
                start1, end1 = times
                start2, end2 = gt_ans
                intersection_start = max(start1, start2)
                intersection_end = min(end1, end2)
                intersection_length = max(0, intersection_end - intersection_start)
                union_length = max(end1, end2) - min(start1, start2)
                return intersection_length / union_length if union_length != 0 else 0.0
            return 0.0
        else:
            return 0.0
    except Exception as e:
        print(f"Error in _ans_tiou_reward for data_source '{data_source}': {e}")
        return 0.0


def _ans_viou_reward(solution_str, ground_truth, data_source, extra_info):
    """Visual IoU reward for visual QA (bounding box) tasks."""
    if data_source != "visual QA":
        return 0.0

    sol = f'<answer>{ground_truth}</answer>'
    try:
        output_ans = extract_answer(solution_str)
        pattern = r"<box>(\[.*?\])</box>"

        match_gt = re.search(pattern, sol)
        bbox_gt = None
        if match_gt:
            bbox_gt = json.loads(match_gt.group(1))

        match_pred = re.search(pattern, output_ans)
        if match_pred and bbox_gt is not None:
            bbox_pred = json.loads(match_pred.group(1))
            image_size = extra_info.get("image_size")
            bbox_gt = resize_absolute_gt_box_to_qwen3_space(bbox_gt, image_size)
            return calculate_iou(bbox_gt, bbox_pred)
        return 0.0
    except Exception as e:
        print(f"Error in _ans_viou_reward for data_source '{data_source}': {e}")
        return 0.0


def _format_reward(solution_str, ground_truth, data_source, extra_info):
    """Format reward: checks <think>/<answer> structure and spatial-temporal tags."""
    content = solution_str

    think_pattern = r"<think>(.*?)</think>"
    answer_pattern = r"<answer>.*?</answer>"

    think_match = re.search(think_pattern, content, re.DOTALL)
    answer_match = re.search(answer_pattern, content, re.DOTALL)

    if not (think_match and answer_match):
        return 0.0

    if content.count("<think>") != content.count("</think>"):
        return 0.0
    if content.count("<answer>") != content.count("</answer>"):
        return 0.0

    think_content = think_match.group(1)

    count_obj_start = think_content.count('<obj>')
    count_obj_end = think_content.count('</obj>')
    count_time_start = think_content.count('<t>')
    count_time_end = think_content.count('</t>')
    count_box_start = think_content.count('<box>')
    count_box_end = think_content.count('</box>')

    if not (count_obj_start == count_obj_end and count_time_start == count_time_end and count_box_start == count_box_end):
        return 0.0

    has_st_reasoning = (count_obj_start > 0 and count_time_start > 0 and count_box_start > 0)

    if data_source in ("temporal QA", "temporal QA (MCQ)"):
        has_st_reasoning = count_time_start >= 2

    if data_source == "visual QA":
        pattern = r"<obj>(\w+)</obj><box>(\[.*?\])</box>"
        if re.search(pattern, content):
            has_st_reasoning = True

    if has_st_reasoning or "General video QA" in data_source:
        return 1.0
    else:
        return 0.5


def _thk_temporal_segment_reward(solution_str, ground_truth, data_source, extra_info):
    """Temporal segment reward: checks if predicted time points fall within GT interval."""
    if data_source in ("visual QA", "temporal-spatial free-form QA") or "General video QA" in data_source:
        return 0.0

    think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    if not think_match:
        return 0.0

    think_content = think_match.group(1)
    pattern = r'<t>([\d.]+)</t>s'

    gt_ans = ground_truth
    if data_source == "temporal QA (MCQ)":
        gt_ans = gt_ans.split("\n")[1]

    try:
        gt_ans = ast.literal_eval(gt_ans)
    except Exception:
        return 0.0

    try:
        times = [float(m) for m in re.findall(pattern, think_content)]
    except Exception:
        return 0.0

    if len(times) > 0:
        reward = sum(1.0 for t in times if gt_ans[0] <= t <= gt_ans[1]) / len(times)
        return reward
    return 0.0


def _thk_temporal_point_reward(solution_str, ground_truth, data_source, extra_info):
    """Temporal point reward: Gaussian proximity to GT key-frame timestamps."""
    if data_source in ("visual QA", "temporal QA", "temporal QA (MCQ)") or "General video QA" in data_source:
        return 0.0

    think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    if not think_match:
        return 0.0

    think_content = think_match.group(1)
    pattern = r'<t>([\d.]+)</t>s'

    try:
        pred_times = [float(m) for m in re.findall(pattern, think_content)]
    except Exception:
        pred_times = []

    if len(pred_times) == 0:
        return 0.0

    step_percent = extra_info.get("step_percent", 0.0) 
    # print(step_percent)
    key_frames = extra_info.get("key_frames", [])
    gt_times = [frame["time"] for frame in key_frames]
    # breakpoint()
    if not gt_times:
        return 0.0

    total_proximity_score = 0.0
    for time in pred_times:
        time_diff = min(abs(time - gt_time) for gt_time in gt_times)
        if step_percent < 3 / 4:
            sigma = 4 * (1 - step_percent)
        else:
            sigma = 1
        proximity_score = np.exp(-(time_diff ** 2) / (2 * sigma ** 2))
        total_proximity_score += proximity_score

    return total_proximity_score / len(pred_times)


def _thk_spatial_reward(solution_str, ground_truth, data_source, extra_info):
    """Spatial reward: IoU of predicted bounding boxes vs GT."""
    think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)

    if not think_match or not answer_match:
        return 0.0

    # visual QA - spatial only
    if data_source == "visual QA":
        pattern = r"<box>(\[.*?\])</box>"
        bbox_gt = None
        match_gt = re.search(pattern, ground_truth)
        if match_gt:
            try:
                bbox_gt = json.loads(match_gt.group(1))
            except Exception:
                bbox_gt = None

        output_think = think_match.group(1)
        match_pred = re.findall(pattern, output_think)
        bboxes_pred = []
        for bbox_item in match_pred:
            try:
                bboxes_pred.append(json.loads(bbox_item))
            except Exception:
                pass

        if len(bboxes_pred) > 0 and bbox_gt is not None:
            image_size = extra_info.get("image_size")
            bbox_gt = resize_absolute_gt_box_to_qwen3_space(bbox_gt, image_size)
            max_iou = max(calculate_iou(bbox_gt, bp) for bp in bboxes_pred)
            return max_iou
        return 0.0

    # temporal QA / General video QA - no spatial
    if data_source in ("temporal QA", "temporal QA (MCQ)") or "General video QA" in data_source:
        return 0.0

    # temporal + spatial (temporal-spatial free-form QA, etc.)
    think_content = think_match.group(1)
    parsed_claims = parse_temporal_spatial_reasoning_process(think_content)

    if not parsed_claims:
        return 0.0

    key_items = extra_info.get("key_items", {})
    key_frames = extra_info.get("key_frames", [])
    gt_times = [frame["time"] for frame in key_frames]

    total_iou_score = 0.0

    for claim in parsed_claims:
        pred_time = claim['timestamp']
        closest_time = -1
        min_time_diff = float('inf')
        threshold = 1.0

        for ii in range(len(gt_times)):
            if gt_times[ii] - pred_time < threshold:
                time_diff = abs(gt_times[ii] - pred_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_time = gt_times[ii]

        if closest_time == -1:
            continue

        key_frame = None
        for frame in key_frames:
            if frame["time"] == closest_time:
                key_frame = frame
                break

        if claim['bboxes'] is not None and isinstance(claim['bboxes'], list) and key_frame is not None:
            objects = key_items.get(str(key_frame["idx"]), {})
            max_iou = 0.0

            for obj in objects.keys():
                claim_boxes = claim['bboxes']
                gt_boxes = objects[obj]
                try:
                    is_claim_originally_multiple = isinstance(claim_boxes[0], list)
                except Exception:
                    print("Error:", claim_boxes)
                    continue

                if not is_claim_originally_multiple:
                    claim_boxes = [claim_boxes]

                list_of_max_ious = []
                for gt_box in gt_boxes:
                    gt_box = resize_normalized_gt_box_to_qwen3_space(gt_box)
                    ious_for_current_gt = [calculate_iou(gt_box, c_box) for c_box in claim_boxes]
                    iou_for_gt = max(ious_for_current_gt) if ious_for_current_gt else 0.0
                    list_of_max_ious.append(iou_for_gt)

                if list_of_max_ious:
                    iou = sum(list_of_max_ious) / len(list_of_max_ious)
                    if iou > max_iou:
                        max_iou = iou

            total_iou_score += max_iou

    return total_iou_score / len(parsed_claims)


# ==================== Test Examples ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Open-o3-Video Reward Functions")
    print("=" * 60)

    # Test 1: Temporal QA (MCQ)
    print("\n[Test 1] Temporal QA (MCQ)")
    solution_1 = """<think>
At <t>5.2</t>s, the person picks up the object. At <t>7.8</t>s, they put it down.
</think>
<answer>Correct Option: A</answer>"""
    
    ground_truth_1 = "A\n[5.0, 8.0]"
    data_source_1 = "temporal QA (MCQ)"
    
    score_1 = compute_score(solution_1, ground_truth_1, data_source_1)
    print(f"Solution: {solution_1[:80]}...")
    print(f"Ground Truth: {ground_truth_1}")
    print(f"Total Score: {score_1:.4f}")

    # Test 2: Visual QA
    print("\n[Test 2] Visual QA")
    solution_2 = """<think>
I can see <obj>cat</obj><box>[0.3, 0.2, 0.7, 0.8]</box> in the image.
</think>
<answer><box>[0.3, 0.2, 0.7, 0.8]</box></answer>"""
    
    ground_truth_2 = "[0.3, 0.2, 0.7, 0.8]"
    data_source_2 = "visual QA"
    extra_info_2 = {
        "image_size": (1920, 1080),
        "image_size_refine": (1920, 1080)
    }
    
    score_2 = compute_score(solution_2, ground_truth_2, data_source_2, extra_info_2)
    print(f"Solution: {solution_2[:80]}...")
    print(f"Ground Truth: {ground_truth_2}")
    print(f"Total Score: {score_2:.4f}")

    # Test 3: Temporal QA
    print("\n[Test 3] Temporal QA")
    solution_3 = """<think>
The event starts at <t>3.5</t>s and ends at <t>6.2</t>s.
</think>
<answer><t>3.5</t>s to <t>6.2</t>s</answer>"""
    
    ground_truth_3 = "[3.0, 6.5]"
    data_source_3 = "temporal QA"
    
    score_3 = compute_score(solution_3, ground_truth_3, data_source_3)
    print(f"Solution: {solution_3[:80]}...")
    print(f"Ground Truth: {ground_truth_3}")
    print(f"Total Score: {score_3:.4f}")

    # Test 4: General video QA Free-form
    print("\n[Test 4] General video QA Free-form")
    solution_4 = """<think>
The video shows a person walking in the park, feeding birds.
</think>
<answer>A person is walking in the park and feeding the birds.</answer>"""
    
    ground_truth_4 = "A person is walking in the park and feeding the birds."
    data_source_4 = "General video QA Free-form"
    
    score_4 = compute_score(solution_4, ground_truth_4, data_source_4)
    print(f"Solution: {solution_4[:80]}...")
    print(f"Ground Truth: {ground_truth_4}")
    print(f"Total Score: {score_4:.4f}")

    # Test 5: Temporal-spatial free-form QA
    print("\n[Test 5] Temporal-spatial free-form QA")
    solution_5 = """<think>
<obj>person</obj><box>[0.4, 0.3, 0.8, 0.9]</box>at<t>2.5</t>s
<obj>ball</obj><box>[0.1, 0.5, 0.3, 0.7]</box>at<t>5.0</t>s
</think>
<answer>The person throws the ball at around 2.5 seconds.</answer>"""
    
    ground_truth_5 = "The person throws the ball."
    data_source_5 = "temporal-spatial free-form QA"
    extra_info_5 = {
        "key_frames": [
            {"time": 2.5, "idx": 0},
            {"time": 5.0, "idx": 1}
        ],
        "key_items": {
            "0": {"person": [[[0.4, 0.3, 0.8, 0.9]]]},
            "1": {"ball": [[[0.1, 0.5, 0.3, 0.7]]]}
        },
        "image_size": (1920, 1080),
        "step_percent": 0.5
    }
    
    score_5 = compute_score(solution_5, ground_truth_5, data_source_5, extra_info_5)
    print(f"Solution: {solution_5[:80]}...")
    print(f"Ground Truth: {ground_truth_5}")
    print(f"Total Score: {score_5:.4f}")

    # Test 6: Format validation (bad format)
    print("\n[Test 6] Format validation - bad format")
    solution_6 = """<think>
This is incomplete
<answer>No closing tag"""
    
    ground_truth_6 = "Test"
    data_source_6 = "General video QA Free-form"
    
    score_6 = compute_score(solution_6, ground_truth_6, data_source_6)
    print(f"Solution: {solution_6[:80]}...")
    print(f"Ground Truth: {ground_truth_6}")
    print(f"Total Score: {score_6:.4f}")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
