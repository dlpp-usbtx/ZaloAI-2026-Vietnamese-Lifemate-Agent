import json
import sys
import types

import pytest

import run_submission
from run_submission import _load_completed_ids, _read_items


def test_read_items_rejects_duplicate_sample_ids(tmp_path):
    input_path = tmp_path / "train.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "sample", "audio": "a.wav"}),
                json.dumps({"id": "sample", "audio": "b.wav"}),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate sample id"):
        _read_items(input_path)


def test_load_completed_ids_validates_resume_jsonl(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        json.dumps({"id": "sample-1"}) + "\n",
        encoding="utf-8",
    )

    assert _load_completed_ids(checkpoint, {"sample-1", "sample-2"}) == {
        "sample-1"
    }


def test_load_completed_ids_rejects_corrupted_lines(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text('{"id":"sample-1"}\n{broken\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        _load_completed_ids(checkpoint, {"sample-1"})


def test_load_completed_ids_rejects_unknown_and_duplicate_ids(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        "\n".join(
            [
                json.dumps({"id": "sample-1"}),
                json.dumps({"id": "sample-1"}),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate sample id"):
        _load_completed_ids(checkpoint, {"sample-1"})

    checkpoint.write_text(
        json.dumps({"id": "outside"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unexpected or missing"):
        _load_completed_ids(checkpoint, {"sample-1"})


def test_main_writes_performance_log_separately_from_submission(
    tmp_path, monkeypatch
):
    split_dir = tmp_path / "public_test"
    split_dir.mkdir()
    (split_dir / "public_test.jsonl").write_text(
        json.dumps({"id": "sample-1", "audio": "sample.wav"}) + "\n",
        encoding="utf-8",
    )
    (split_dir / "sample.wav").write_bytes(b"audio")
    (split_dir / "public_test_tools_catalog.json").write_text(
        "{}", encoding="utf-8"
    )

    class FakeASREngine:
        device = "cpu"

        def transcribe(self, audio_path):
            return "recognized request"

    class FakeLLMEngine:
        device = "cpu"
        model_dtype = "torch.float32"
        attention_implementation = "sdpa"
        max_input_tokens = 8192
        max_new_tokens = 256

        def __init__(self, model_path):
            self.model_path = model_path

    class FakePlanner:
        def __init__(self, *args, **kwargs):
            self.last_performance_stats = {
                "planner_seconds": 1.5,
                "llm_call_count": 1,
                "llm_seconds": 1.0,
                "tool_validation_seconds": 0.1,
                "tool_execution_seconds": 0.2,
                "llm_calls": [
                    {
                        "turn": 1,
                        "retry": 1,
                        "call_seconds": 1.0,
                        "engine": {
                            "input_tokens": 20,
                            "output_tokens": 5,
                            "generation_seconds": 0.8,
                        },
                    }
                ],
            }

        def process_sample(self, item, env, sample_dir):
            return [{"name": "tool"}], "state-hash"

    fake_tools = types.ModuleType("zaic_tools")
    fake_tools.ToolEnv = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "zaic_tools", fake_tools)
    monkeypatch.setattr(run_submission, "get_split_dir", lambda split: split_dir)
    monkeypatch.setattr(
        run_submission, "get_inference_model_path", lambda require_sft: "adapter"
    )
    monkeypatch.setattr(run_submission, "ASREngine", FakeASREngine)
    monkeypatch.setattr(run_submission, "HuggingFaceEngine", FakeLLMEngine)
    monkeypatch.setattr(run_submission, "ZAICAgentPlanner", FakePlanner)
    monkeypatch.setattr(run_submission, "CatalogManager", lambda path: object())

    submission_path = tmp_path / "submission.jsonl"
    performance_path = tmp_path / "performance.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_submission.py",
            "--split",
            "public_test",
            "--output",
            str(submission_path),
            "--performance-log",
            str(performance_path),
        ],
    )

    run_submission.main()

    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    assert set(submission) == {"id", "tool_calls", "state_hash"}
    assert submission["id"] == "sample-1"
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    assert performance["id"] == "sample-1"
    assert performance["planner"]["llm_calls"][0]["engine"]["output_tokens"] == 5
