"""Run LifeMate inference and write a submission JSONL file."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)

import torch

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from configs.paths import get_inference_model_path, get_split_dir
from src.agent.zaic_planner import ZAICAgentPlanner
from src.inference.asr_engine import ASREngine
from src.inference.engine import HuggingFaceEngine
from src.tools.catalog_manager import CatalogManager


def _read_items(input_path: Path) -> list[dict]:
    items = []
    seen_ids = set()
    with input_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = item.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{input_path}:{line_number} has no valid sample id.")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id {sample_id!r} in {input_path}.")
            if not isinstance(item.get("audio"), str) or not item["audio"]:
                raise ValueError(f"{input_path}:{line_number} has no audio path.")
            seen_ids.add(sample_id)
            items.append(item)
    if not items:
        raise ValueError(f"No samples found in {input_path}.")
    return items


def _load_completed_ids(
    checkpoint_path: Path, expected_ids: set[str]
) -> set[str]:
    completed_ids: set[str] = set()
    with checkpoint_path.open(encoding="utf-8") as checkpoint_file:
        for line_number, line in enumerate(checkpoint_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in checkpoint {checkpoint_path}:{line_number}."
                ) from error
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or sample_id not in expected_ids:
                raise ValueError(
                    f"Unexpected or missing sample id in "
                    f"{checkpoint_path}:{line_number}: {sample_id!r}"
                )
            if sample_id in completed_ids:
                raise ValueError(
                    f"Duplicate sample id {sample_id!r} in checkpoint {checkpoint_path}."
                )
            completed_ids.add(sample_id)
    return completed_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a LifeMate submission.")
    parser.add_argument(
        "--split",
        choices=("public_test", "train"),
        default="public_test",
        help="Dataset split to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "submission.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--performance-log",
        type=Path,
        default=None,
        help=(
            "Per-sample timing and token metrics JSONL path. "
            "Defaults beside the submission output."
        ),
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=2,
        help="Maximum total LLM generate calls for each sample.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing samples to an existing output file.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=(
            Path(os.environ["LIFEMATE_RESUME_FROM"])
            if os.environ.get("LIFEMATE_RESUME_FROM")
            else None
        ),
        help="Optional input checkpoint to copy before resuming.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_llm_calls < 1:
        raise ValueError("--max-llm-calls must be greater than zero.")
    performance_log = args.performance_log or args.output.with_name(
        f"{args.output.stem}.performance.jsonl"
    )
    if performance_log.resolve() == args.output.resolve():
        raise ValueError("Performance log path must differ from the submission output.")
    split_dir = get_split_dir(args.split)
    input_path = split_dir / f"{args.split}.jsonl"
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")
    items = _read_items(input_path)
    expected_ids = {item["id"] for item in items}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume_from is not None:
        if not args.resume_from.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_from}")
        if not args.output.exists():
            shutil.copy2(args.resume_from, args.output)

    if (args.resume or args.resume_from is not None) and args.output.is_file():
        completed_ids = _load_completed_ids(args.output, expected_ids)
        output_mode = "a"
    else:
        completed_ids = set()
        output_mode = "w"

    remaining = len(items) - len(completed_ids)
    if remaining == 0:
        performance_log.parent.mkdir(parents=True, exist_ok=True)
        performance_log.touch(exist_ok=True)
        print(
            f"Submission already complete: {args.output} ({len(completed_ids)} samples).",
            flush=True,
        )
        print(f"Performance log: {performance_log}", flush=True)
        return

    print("Initializing ZipFormer ASR and LLM engines...", flush=True)
    asr_load_started = time.perf_counter()
    asr_engine = ASREngine()
    asr_load_seconds = time.perf_counter() - asr_load_started
    model_path = get_inference_model_path(require_sft=True)
    print(f"Loading LLM from: {model_path}", flush=True)
    llm_load_started = time.perf_counter()
    llm_engine = HuggingFaceEngine(model_path=model_path)
    llm_load_seconds = time.perf_counter() - llm_load_started
    print(
        f"[PERF_SETUP] asr_load_seconds={asr_load_seconds:.3f}; "
        f"llm_load_seconds={llm_load_seconds:.3f}; "
        f"asr_device={asr_engine.device}; llm_device={llm_engine.device}; "
        f"llm_dtype={llm_engine.model_dtype}; "
        f"attention={llm_engine.attention_implementation}; "
        f"max_input_tokens={llm_engine.max_input_tokens}; "
        f"max_new_tokens={llm_engine.max_new_tokens}",
        flush=True,
    )
    catalog_manager = CatalogManager(
        split_dir / f"{args.split}_tools_catalog.json"
    )
    planner = ZAICAgentPlanner(
        llm_engine,
        catalog_manager,
        asr_engine=asr_engine,
        max_llm_calls=args.max_llm_calls,
    )

    tools_directory = str(split_dir)
    if tools_directory not in sys.path:
        sys.path.insert(0, tools_directory)
    from zaic_tools import ToolEnv

    print(
        f"Processing {args.split}: {len(items)} samples; "
        f"already completed: {len(completed_ids)}; remaining: {remaining}.",
        flush=True,
    )
    performance_log.parent.mkdir(parents=True, exist_ok=True)
    performance_mode = "a" if output_mode == "a" else "w"
    print(f"Per-sample performance log: {performance_log}", flush=True)
    with args.output.open(output_mode, encoding="utf-8") as output_file:
        with performance_log.open(performance_mode, encoding="utf-8") as performance_file:
            processed_count = 0
            for item in items:
                sample_id = item["id"]
                if sample_id in completed_ids:
                    continue

                sample_started = time.perf_counter()
                audio_path = split_dir / item["audio"]
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"Audio file for {sample_id} not found: {audio_path}"
                    )

                print(f"{sample_id}: starting ASR", flush=True)
                asr_started = time.perf_counter()
                item["transcribed_text"] = asr_engine.transcribe(str(audio_path))
                asr_seconds = time.perf_counter() - asr_started
                print(
                    f"{sample_id}: ASR completed in "
                    f"{asr_seconds:.2f}s; starting LLM",
                    flush=True,
                )

                environment = ToolEnv(
                    args.split, sample_id, root=split_dir.parent
                )
                tool_calls, state_hash = planner.process_sample(
                    item, environment, split_dir
                )
                sample_seconds = time.perf_counter() - sample_started
                record = {
                    "id": sample_id,
                    "tool_calls": tool_calls,
                    "state_hash": state_hash,
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()

                performance_record = {
                    "id": sample_id,
                    "asr_seconds": asr_seconds,
                    "sample_seconds": sample_seconds,
                    "planner": planner.last_performance_stats,
                }
                performance_file.write(
                    json.dumps(performance_record, ensure_ascii=False) + "\n"
                )
                performance_file.flush()
                processed_count += 1
                print(
                    f"[PERF_SAMPLE] id={sample_id}; "
                    f"seconds={sample_seconds:.3f}; "
                    f"llm_calls={planner.last_performance_stats['llm_call_count']}; "
                    f"llm_seconds={planner.last_performance_stats['llm_seconds']:.3f}; "
                    f"tool_seconds={planner.last_performance_stats['tool_validation_seconds'] + planner.last_performance_stats['tool_execution_seconds']:.3f}; "
                    f"({processed_count}/{remaining} this run)",
                    flush=True,
                )

                del environment, tool_calls, state_hash, record, item
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    completed_ids = _load_completed_ids(args.output, expected_ids)
    missing_ids = expected_ids - completed_ids
    if missing_ids:
        raise RuntimeError(
            f"Submission incomplete: {len(missing_ids)} samples remain; "
            f"first missing IDs: {sorted(missing_ids)[:10]}"
        )
    print(
        f"Submission complete: {args.output} ({len(completed_ids)} samples).",
        flush=True,
    )
    print(f"Performance log saved: {performance_log}", flush=True)


if __name__ == "__main__":
    main()
