import json

import pytest

from testing.benchmark_asr import (
    error_counts,
    load_items,
    normalize_text,
    summarize_results,
)


def test_normalize_text_preserves_vietnamese_letters_and_removes_punctuation():
    assert normalize_text("  Mai, MƯA!  Đà Nẵng. ") == "mai mưa đà nẵng"


def test_error_counts_returns_word_and_character_edits():
    assert error_counts("mai mưa", "mai nắng") == {
        "word_errors": 1,
        "reference_words": 2,
        "char_errors": 4,
        "reference_chars": 7,
    }


def test_load_items_reads_audio_text_and_selected_ids(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "audio_text": "một"}),
                json.dumps({"id": "b", "audio_text": "hai"}),
            ]
        ),
        encoding="utf-8",
    )

    assert [item["id"] for item in load_items(data_path, None, ["b", "a"])] == [
        "b",
        "a",
    ]


def test_load_items_fails_when_reference_text_is_missing(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(json.dumps({"id": "a"}), encoding="utf-8")

    with pytest.raises(ValueError, match="audio_text"):
        load_items(data_path, None, None)


def test_summary_aggregates_error_rates_and_latency():
    records = [
        {
            "context_city": "Hà Nội",
            "word_errors": 1,
            "reference_words": 2,
            "char_errors": 1,
            "reference_chars": 4,
            "asr_seconds": 0.2,
            "real_time_factor": 0.5,
        },
        {
            "context_city": "Hà Nội",
            "word_errors": 0,
            "reference_words": 2,
            "char_errors": 0,
            "reference_chars": 4,
            "asr_seconds": 0.4,
            "real_time_factor": 1.0,
        },
    ]

    summary = summarize_results(records)

    assert summary["wer"] == 0.25
    assert summary["cer"] == 0.125
    assert summary["latency_seconds"]["p50"] == pytest.approx(0.3)
    assert summary["real_time_factor"]["mean"] == 0.75
    assert summary["by_context_city"]["Hà Nội"]["samples"] == 2
