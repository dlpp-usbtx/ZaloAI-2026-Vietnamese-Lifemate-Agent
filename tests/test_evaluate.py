from testing.evaluate import _smoke_cases, calculate_tool_f1
from configs import paths


def test_tool_f1_counts_duplicate_calls_correctly():
    call = {"name": "alarm.list", "arguments": {}}
    precision, recall, f1 = calculate_tool_f1([call, call], [call])

    assert precision == 0.5
    assert recall == 1.0
    assert round(f1, 6) == round(2 / 3, 6)


def test_tool_f1_is_one_when_both_sides_have_no_calls():
    assert calculate_tool_f1([], []) == (1.0, 1.0, 1.0)


def test_smoke_cases_include_history_audio_multistep_and_noop():
    items = [
        {
            "id": "history_multi",
            "history": [{"role": "user", "audio": "audio/u1.wav"}],
            "label": {"tool_calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]},
        },
        {
            "id": "multi",
            "history": [],
            "label": {"tool_calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]},
        },
        {"id": "noop", "history": [], "label": {"tool_calls": []}},
    ]

    selected = _smoke_cases(items)

    assert [(item["id"], force_asr) for item, force_asr in selected] == [
        ("history_multi", True),
        ("multi", False),
        ("noop", False),
    ]


def test_inference_model_path_prefers_exported_adapter(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(paths, "SFT_ADAPTER_DIR", adapter_dir)
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path / "checkpoint")

    assert paths.get_inference_model_path(require_sft=True) == str(adapter_dir)


def test_inference_model_path_uses_full_checkpoint_when_adapter_missing(
    tmp_path, monkeypatch
):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(paths, "SFT_ADAPTER_DIR", tmp_path / "missing_adapter")
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", checkpoint_dir)

    assert paths.get_inference_model_path(require_sft=True) == str(checkpoint_dir)


def test_inference_model_path_requires_sft_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SFT_ADAPTER_DIR", tmp_path / "missing_adapter")
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path / "missing_checkpoint")

    try:
        paths.get_inference_model_path(require_sft=True)
    except FileNotFoundError as exc:
        assert "No trained adapter or full-model checkpoint" in str(exc)
    else:
        raise AssertionError("Expected a missing SFT model to raise FileNotFoundError.")


def test_inference_model_path_falls_back_to_base_model(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SFT_ADAPTER_DIR", tmp_path / "missing_adapter")
    monkeypatch.setattr(paths, "CHECKPOINT_DIR", tmp_path / "missing_checkpoint")

    assert paths.get_inference_model_path() == paths.BASE_MODEL_ID
