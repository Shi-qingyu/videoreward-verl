#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".cache" / "video_o3_smoke_data"
DEFAULT_SAMPLE_COUNT = 100

LLAVA_VIDEO_ROOT = Path("/mnt/bn/strategy-mllm-train/common/datasets/LLaVA-Video-178K")
CGBENCH_VIDEO_ROOT = Path("/mnt/bn/strategy-mllm-train/common/datasets/CG-Bench/videos")

SOURCE_CONFIGS = [
    {
        "name": "llava_multi_w_tool",
        "kind": "llava",
        "source_path": Path(
            "/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/data/Seeker-173K/RL/"
            "llava-video_youtube_qa_mc_2_3_m_clue_multi_w_tool_13900.json"
        ),
        "output_prefix": "llava-video_youtube_qa_mc_2_3_m_clue_multi_w_tool_sample",
    },
    {
        "name": "llava_multi_wo_tool",
        "kind": "llava",
        "source_path": Path(
            "/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/data/Seeker-173K/RL/"
            "llava-video_youtube_qa_mc_2_3_m_clue_multi_wo_tool_29523.json"
        ),
        "output_prefix": "llava-video_youtube_qa_mc_2_3_m_clue_multi_wo_tool_sample",
    },
    {
        "name": "cgbench_single_w_tool",
        "kind": "cgbench",
        "source_path": Path(
            "/mnt/bn/strategy-mllm-train/intern/users/baozhongyuan/projects/data/Seeker-173K/RL/"
            "cgbench_correct_clue_single_w_tool_6764.json"
        ),
        "output_prefix": "cgbench_correct_clue_single_w_tool_sample",
    },
]


def _add_default_suffix(path_like: str, suffix: str = ".mp4") -> str:
    return path_like if Path(path_like).suffix else f"{path_like}{suffix}"


def _llava_candidates(raw_video: str) -> list[Path]:
    raw_video = raw_video.rstrip("/")
    trimmed = raw_video
    prefix = "pnorm2/llava_video/"
    if trimmed.startswith(prefix):
        trimmed = trimmed[len(prefix) :]
        parts = trimmed.split("/", 1)
        if len(parts) == 2:
            trimmed = parts[1]

    candidates = [
        LLAVA_VIDEO_ROOT / trimmed,
        LLAVA_VIDEO_ROOT / _add_default_suffix(trimmed),
        LLAVA_VIDEO_ROOT / "videos" / trimmed,
        LLAVA_VIDEO_ROOT / "videos" / _add_default_suffix(trimmed),
    ]
    return candidates


def _cgbench_candidates(raw_video: str) -> list[Path]:
    raw_video = raw_video.rstrip("/")
    basename = Path(raw_video).name
    candidates = [
        CGBENCH_VIDEO_ROOT / raw_video,
        CGBENCH_VIDEO_ROOT / _add_default_suffix(raw_video),
        CGBENCH_VIDEO_ROOT / basename,
        CGBENCH_VIDEO_ROOT / _add_default_suffix(basename),
    ]
    return candidates


def resolve_video_path(kind: str, raw_video: str) -> Path | None:
    raw_path = Path(raw_video)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    if kind == "llava":
        candidates = _llava_candidates(raw_video)
    elif kind == "cgbench":
        candidates = _cgbench_candidates(raw_video)
    else:
        raise KeyError(f"Unsupported source kind: {kind}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_sample(
    source_name: str,
    source_path: Path,
    kind: str,
    output_path: Path,
    count: int,
    seed: int,
) -> dict[str, Any]:
    with source_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    sampled_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    missing_examples: list[dict[str, str]] = []

    for index in indices:
        row = deepcopy(records[index])
        raw_video = row["video"]
        resolved_video = resolve_video_path(kind, raw_video)
        if resolved_video is None:
            if len(missing_examples) < 5:
                missing_examples.append({"doc_id": str(row.get("doc_id", index)), "video": raw_video})
            continue

        row_id = str(row.get("doc_id", row.get("id", row.get("uid", index))))
        if row_id in seen_ids:
            continue

        row["video"] = str(resolved_video)
        extra_info = dict(row.get("extra_info", {}))
        extra_info.setdefault("source_video", raw_video)
        extra_info.setdefault("sample_source", source_name)
        row["extra_info"] = extra_info

        sampled_records.append(row)
        seen_ids.add(row_id)
        if len(sampled_records) >= count:
            break

    if len(sampled_records) < count:
        raise RuntimeError(
            f"Only found {len(sampled_records)} valid records for {source_name}, fewer than requested {count}. "
            f"Missing examples: {missing_examples}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sampled_records, f, ensure_ascii=False, indent=2)

    return {
        "source_name": source_name,
        "source_path": str(source_path),
        "source_count": len(records),
        "sample_count": len(sampled_records),
        "output_path": str(output_path),
        "missing_examples": missing_examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small valid RL samples for video_o3 smoke training.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=20260326)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for offset, config in enumerate(SOURCE_CONFIGS):
        output_path = args.output_dir / f"{config['output_prefix']}_{args.count}.json"
        result = build_sample(
            source_name=config["name"],
            source_path=config["source_path"],
            kind=config["kind"],
            output_path=output_path,
            count=args.count,
            seed=args.seed + offset,
        )
        summary.append(result)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for item in summary:
        print(
            f"{item['source_name']}: source={item['source_count']} sample={item['sample_count']} "
            f"output={item['output_path']}"
        )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
