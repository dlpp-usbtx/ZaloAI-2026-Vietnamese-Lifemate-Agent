import json

import pytest

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
