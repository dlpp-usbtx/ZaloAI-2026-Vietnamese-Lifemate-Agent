"""Evaluate the agent on the fixed validation split.

Run from the agent directory:
    python testing/evaluate.py --smoke
    python testing/evaluate.py --require-checkpoint
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import sys
from pathlib import Path
from typing import Any

AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from configs.paths import (
    BASE_DIR,
    BASE_MODEL_ID,
    get_inference_model_path,
    get_split_dir,
)
from src.agent.zaic_planner import ZAICAgentPlanner
from src.inference.asr_engine import ASREngine
from src.inference.engine import HuggingFaceEngine
from src.tools.catalog_manager import CatalogManager

SPLIT = "train"
SPLIT_DIR = get_split_dir(SPLIT)
if str(SPLIT_DIR) not in sys.path:
    sys.path.insert(0, str(SPLIT_DIR))

from zaic_tools import ToolEnv, canon, state_hash


def _call_key(call: dict[str, Any]) -> str:
    return json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def calculate_tool_f1(
    predicted: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> tuple[float, float, float]:
    """Return exact-call precision, recall and F1, respecting duplicate calls."""
    predicted_calls = Counter(_call_key(call) for call in predicted)
    expected_calls = Counter(_call_key(call) for call in expected)
    true_positives = sum((predicted_calls & expected_calls).values())
    false_positives = sum((predicted_calls - expected_calls).values())
    false_negatives = sum((expected_calls - predicted_calls).values())

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 1.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 1.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _has_user_history_audio(item: dict[str, Any]) -> bool:
    return any(
        entry.get("role") == "user" and entry.get("audio")
        for entry in item.get("history", [])
    )


def _smoke_cases(items: list[dict[str, Any]]) -> list[tuple[dict[str, Any], bool]]:
    """Pick a history-audio case, a multi-call case, and a no-call case."""
    selected: list[tuple[dict[str, Any], bool]] = []
    selected_ids: set[str] = set()

    history_audio = next(
        (item for item in items if _has_user_history_audio(item)), None
    )
    if history_audio is not None:
        selected.append((history_audio, True))
        selected_ids.add(history_audio["id"])

    multi_call = next(
        (
            item
            for item in items
            if item["id"] not in selected_ids
            and len(item.get("label", {}).get("tool_calls", [])) > 1
        ),
        None,
    )
    if multi_call is not None:
        selected.append((multi_call, False))
        selected_ids.add(multi_call["id"])

    no_call = next(
        (
            item
            for item in items
            if item["id"] not in selected_ids
            and not item.get("label", {}).get("tool_calls", [])
        ),
        None,
    )
    if no_call is not None:
        selected.append((no_call, False))
    return selected


def _remove_history_transcripts(item: dict[str, Any]) -> dict[str, Any]:
    """Make a smoke case exercise ASR on user history turns that have audio."""
    result = copy.deepcopy(item)
    for entry in result.get("history", []):
        if entry.get("role") == "user" and entry.get("audio"):
            entry.pop("audio_text", None)
            entry.pop("text", None)
            entry.pop("content", None)
    return result


def _load_validation_items() -> list[dict[str, Any]]:
    processed_dir = BASE_DIR / "data" / "processed"
    ids_path = processed_dir / "val_ids.json"
    raw_data_path = SPLIT_DIR / "train.jsonl"

    if not ids_path.is_file():
        raise FileNotFoundError(
            f"Validation IDs file not found: {ids_path}. "
            "Run training/prepare_train_data.py first."
        )
    if not raw_data_path.is_file():
        raise FileNotFoundError(f"Training data file not found: {raw_data_path}")

    validation_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if not isinstance(validation_ids, list) or not validation_ids:
        raise ValueError(f"{ids_path} must contain a non-empty list of IDs.")
    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError(f"{ids_path} contains duplicate IDs.")

    wanted_ids = set(validation_ids)
    items_by_id: dict[str, dict[str, Any]] = {}
    with raw_data_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("id") in wanted_ids:
                items_by_id[item["id"]] = item

    missing_ids = sorted(wanted_ids - items_by_id.keys())
    if missing_ids:
        raise ValueError(
            f"Validation IDs missing from train.jsonl: {missing_ids[:10]}"
        )
    return [items_by_id[sample_id] for sample_id in validation_ids]


def _select_items(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    validation_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    force_asr_ids: set[str] = set()

    if args.smoke and (args.limit is not None or args.ids):
        parser.error("--smoke cannot be combined with --limit or --ids.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero.")

    if args.smoke:
        smoke_cases = _smoke_cases(validation_items)
        if not smoke_cases:
            raise ValueError("No suitable validation examples found for smoke evaluation.")
        selected = []
        for item, force_asr in smoke_cases:
            selected.append(item)
            if force_asr:
                force_asr_ids.add(item["id"])
        return selected, force_asr_ids

    if args.ids:
        items_by_id = {item["id"]: item for item in validation_items}
        unknown_ids = [sample_id for sample_id in args.ids if sample_id not in items_by_id]
        if unknown_ids:
            raise ValueError(f"IDs are not in the validation split: {unknown_ids}")
        return [items_by_id[sample_id] for sample_id in args.ids], force_asr_ids

    selected = validation_items
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected, force_asr_ids


def run_evaluation() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the validation split generated by prepare_train_data.py."
    )
    parser.add_argument("--smoke", action="store_true", help="Run a few representative cases.")
    parser.add_argument("--limit", type=int, help="Limit evaluation to the first N validation examples.")
    parser.add_argument(
        "--ids", nargs="+", help="Evaluate only IDs listed in val_ids.json."
    )
    parser.add_argument(
        "--require-checkpoint",
        action="store_true",
        help="Fail if the SFT checkpoint is missing; do not fall back to the base model.",
    )
    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Evaluate the base model instead of the configured SFT adapter.",
    )
    args = parser.parse_args()
    if args.base_model and args.require_checkpoint:
        parser.error("--base-model cannot be combined with --require-checkpoint.")

    validation_items = _load_validation_items()
    selected_items, force_asr_ids = _select_items(
        parser, args, validation_items
    )
    if not selected_items:
        raise ValueError("No examples selected for evaluation.")

    model_path = (
        BASE_MODEL_ID
        if args.base_model
        else get_inference_model_path(require_sft=args.require_checkpoint)
    )
    if args.base_model or model_path == BASE_MODEL_ID:
        print(f"Evaluating base model: {model_path}")
    else:
        print(f"Evaluating SFT adapter/checkpoint: {model_path}")

    print(f"Evaluating {len(selected_items)} / {len(validation_items)} examples.")
    asr_engine = ASREngine() if force_asr_ids else None
    llm_engine = HuggingFaceEngine(model_path=model_path)
    catalog_manager = CatalogManager(SPLIT_DIR / "train_tools_catalog.json")
    planner = ZAICAgentPlanner(
        llm_engine, catalog_manager, asr_engine=asr_engine
    )

    exact_matches = 0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    state_matches = 0
    invalid_schema_calls = 0

    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "validation_predictions.jsonl"

    with predictions_path.open("w", encoding="utf-8") as prediction_file:
        for index, original_item in enumerate(selected_items, start=1):
            item = (
                _remove_history_transcripts(original_item)
                if original_item["id"] in force_asr_ids
                else copy.deepcopy(original_item)
            )
            # Use the gold final-turn transcript so this evaluation isolates
            # planning/tool use; the smoke history case separately tests ASR.
            item["transcribed_text"] = item.get("audio_text", "")

            env = ToolEnv(SPLIT, item["id"], root=SPLIT_DIR.parent)
            initial_state = env.get_state()
            predicted_calls, planner_hash = planner.process_sample(
                item, env, SPLIT_DIR
            )
            expected_calls = item.get("label", {}).get("tool_calls", [])
            expected_changed_state = item.get("label", {}).get("final_state", {})

            expected_state = copy.deepcopy(initial_state)
            expected_state.update(expected_changed_state)
            actual_state = env.get_state()
            actual_hash = env.get_state_hash()
            expected_hash = state_hash(expected_state)
            matches_state = canon(expected_state) == canon(actual_state)

            precision, recall, f1 = calculate_tool_f1(
                predicted_calls, expected_calls
            )
            exact = predicted_calls == expected_calls
            schema_errors = list(planner.last_schema_errors)

            exact_matches += exact
            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
            state_matches += matches_state
            invalid_schema_calls += len(schema_errors)

            prediction = {
                "id": item["id"],
                "predicted_tool_calls": predicted_calls,
                "expected_tool_calls": expected_calls,
                "ordered_exact_match": exact,
                "tool_call_precision": precision,
                "tool_call_recall": recall,
                "tool_call_f1": f1,
                "schema_errors": schema_errors,
                "state_match": matches_state,
                "expected_state_hash": expected_hash,
                "actual_state_hash": actual_hash,
                "planner_state_hash": planner_hash,
            }
            prediction_file.write(
                json.dumps(prediction, ensure_ascii=False) + "\n"
            )
            print(
                f"[{index}/{len(selected_items)}] {item['id']}: "
                f"exact={exact}, f1={f1:.3f}, "
                f"schema_errors={len(schema_errors)}, state_match={matches_state}"
            )

    count = len(selected_items)
    print("\nValidation results:")
    print(f"  Ordered tool-call exact match: {exact_matches / count:.2%}")
    print(f"  Macro precision:        {precision_sum / count:.2%}")
    print(f"  Macro recall:           {recall_sum / count:.2%}")
    print(f"  Macro F1:               {f1_sum / count:.2%}")
    print(f"  Schema-invalid attempts: {invalid_schema_calls}")
    print(f"  State match:            {state_matches / count:.2%}")
    print(f"  Per-example results:    {predictions_path}")


if __name__ == "__main__":
    run_evaluation()
