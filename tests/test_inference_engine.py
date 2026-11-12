import json
from pathlib import Path

from src.inference import engine


class MockModel:
    def __init__(self):
        self.evaluated = False

    def eval(self):
        self.evaluated = True
        return self


class MockInputs(dict):
    def to(self, device):
        return self

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class MockTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.last_prompt = None

    def apply_chat_template(self, messages, **kwargs):
        return " ".join(message["content"] for message in messages)

    def __call__(self, prompts, **kwargs):
        self.last_prompt = prompts[0]
        tokens = prompts[0].split()
        return MockInputs(input_ids=engine.torch.arange(len(tokens)).unsqueeze(0))

    def batch_decode(self, generated_ids, **kwargs):
        return [" {\"action\": \"final_response\"} "]


class GeneratingMockModel(MockModel):
    def __init__(self):
        super().__init__()
        self.generation_kwargs = None
        self.input_ids = None

    def generate(self, input_ids, **kwargs):
        self.generation_kwargs = kwargs
        self.input_ids = input_ids
        return engine.torch.cat(
            [input_ids, engine.torch.tensor([[999]])],
            dim=1,
        )


def test_engine_loads_adapter_on_its_configured_base(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct"}),
        encoding="utf-8",
    )
    tokenizer = object()
    base_model = MockModel()
    adapter_model = MockModel()
    tokenizer_calls = []
    base_model_calls = []
    adapter_calls = []

    monkeypatch.setattr(engine, "set_deterministic_seed", lambda seed: None)
    monkeypatch.setattr(
        engine.AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: tokenizer_calls.append((path, kwargs)) or tokenizer,
    )
    monkeypatch.setattr(
        engine.AutoModelForCausalLM,
        "from_pretrained",
        lambda path, **kwargs: base_model_calls.append((path, kwargs)) or base_model,
    )
    monkeypatch.setattr(
        engine.PeftModel,
        "from_pretrained",
        lambda model, path, **kwargs: adapter_calls.append((model, path, kwargs))
        or adapter_model,
    )

    loaded = engine.HuggingFaceEngine(str(adapter_dir))

    assert loaded.tokenizer is tokenizer
    assert loaded.model is adapter_model
    assert adapter_model.evaluated
    assert tokenizer_calls[0][0] == str(adapter_dir)
    assert base_model_calls[0][0] == "Qwen/Qwen2.5-3B-Instruct"
    assert adapter_calls == [
        (base_model, str(adapter_dir), {"is_trainable": False})
    ]
    assert loaded.max_input_tokens == 8192
    assert loaded.max_new_tokens == 256


def test_tokenizer_metadata_is_patched_in_temporary_copy(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "extra_special_tokens": ["<|im_start|>", "<|im_end|>"],
    }
    original_config = json.dumps(config)
    (adapter_dir / "tokenizer_config.json").write_text(
        original_config, encoding="utf-8"
    )
    (adapter_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")

    hf_engine = engine.HuggingFaceEngine.__new__(engine.HuggingFaceEngine)
    tokenizer_path = Path(hf_engine._prepare_tokenizer_path(adapter_dir))

    patched_config = json.loads(
        (tokenizer_path / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert patched_config["extra_special_tokens"] == {
        "<|im_start|>": "<|im_start|>",
        "<|im_end|>": "<|im_end|>",
    }
    assert (tokenizer_path / "tokenizer.json").is_file()
    assert not (tokenizer_path / "adapter_model.safetensors").exists()
    assert (adapter_dir / "tokenizer_config.json").read_text(
        encoding="utf-8"
    ) == original_config


def test_generate_trims_old_messages_and_keeps_system_and_user_requests():
    llm_engine = engine.HuggingFaceEngine.__new__(engine.HuggingFaceEngine)
    llm_engine.device = "cpu"
    llm_engine.max_input_tokens = 8
    llm_engine.max_new_tokens = 256
    llm_engine.tokenizer = MockTokenizer()
    llm_engine.model = GeneratingMockModel()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "old history request"},
        {"role": "assistant", "content": "old assistant response"},
        {
            "role": "user",
            "content": "current user request",
            "_preserve_for_generation": True,
        },
        {"role": "user", "content": "recent observation"},
    ]

    output = llm_engine.generate(messages)

    assert output == "{\"action\": \"final_response\"}"
    assert llm_engine.model.generation_kwargs["max_new_tokens"] == 256
    assert llm_engine.model.input_ids.shape[-1] <= llm_engine.max_input_tokens
    assert "system prompt" in llm_engine.tokenizer.last_prompt
    assert "current user request" in llm_engine.tokenizer.last_prompt
    assert "recent observation" in llm_engine.tokenizer.last_prompt
    assert "old history request" not in llm_engine.tokenizer.last_prompt
    assert "old assistant response" not in llm_engine.tokenizer.last_prompt
    assert llm_engine.last_generation_stats["input_tokens"] == (
        llm_engine.model.input_ids.shape[-1]
    )
    assert llm_engine.last_generation_stats["output_tokens"] == 1
    assert llm_engine.last_generation_stats["generation_seconds"] >= 0
    assert llm_engine.last_generation_stats["prepare_seconds"] >= 0
    assert llm_engine.last_generation_stats["decode_seconds"] >= 0
    assert llm_engine.last_generation_stats["device"] == "cpu"


def test_generate_fails_when_required_messages_exceed_prompt_limit():
    llm_engine = engine.HuggingFaceEngine.__new__(engine.HuggingFaceEngine)
    llm_engine.device = "cpu"
    llm_engine.max_input_tokens = 2
    llm_engine.max_new_tokens = 256
    llm_engine.tokenizer = MockTokenizer()
    llm_engine.model = GeneratingMockModel()

    try:
        llm_engine.generate(
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "required user request"},
            ]
        )
    except ValueError as error:
        assert "Required system and user messages exceed" in str(error)
    else:
        raise AssertionError("Expected required messages to exceed prompt limit.")
