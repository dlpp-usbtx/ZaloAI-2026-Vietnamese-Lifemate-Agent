"""
src/inference/engine.py
Inference Engine bằng Hugging Face Transformers.
"""

import torch
import random
import numpy as np
import json
from pathlib import Path
import shutil
import tempfile
import time
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.inference.device import select_compute_device

def set_deterministic_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class HuggingFaceEngine:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-3B-Instruct",
        max_input_tokens: int = 8192,
        max_new_tokens: int = 256,
    ):
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be greater than zero.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")

        set_deterministic_seed(42)
        self.device, device_name = select_compute_device(torch)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.last_generation_stats: dict[str, float | int | str] = {}
        is_cuda = self.device.startswith("cuda")

        model_directory = Path(model_path)
        adapter_config_path = model_directory / "adapter_config.json"
        model_dtype = (
            torch.bfloat16
            if is_cuda and torch.cuda.is_bf16_supported()
            else torch.float16
            if is_cuda
            else torch.float32
        )
        self.model_dtype = str(model_dtype)
        device_map = {"": int(self.device.split(":")[1])} if is_cuda else None

        if adapter_config_path.is_file():
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            base_model_path = adapter_config.get("base_model_name_or_path")
            if not base_model_path:
                raise ValueError(
                    f"Adapter config does not specify base_model_name_or_path: "
                    f"{adapter_config_path}"
                )
            tokenizer_path = self._prepare_tokenizer_path(model_directory)
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=model_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(
                base_model, model_path, is_trainable=False
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=model_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        self.model.eval()
        print(
            f"[LLM] device={self.device} ({device_name}); dtype={self.model_dtype}",
            flush=True,
        )

    def _prepare_tokenizer_path(self, model_directory: Path) -> str:
        """Normalize exported tokenizer metadata without writing to read-only inputs."""
        tokenizer_config_path = model_directory / "tokenizer_config.json"
        if not tokenizer_config_path.is_file():
            return str(model_directory)

        tokenizer_config = json.loads(
            tokenizer_config_path.read_text(encoding="utf-8")
        )
        extra_special_tokens = tokenizer_config.get("extra_special_tokens")
        if not isinstance(extra_special_tokens, list):
            return str(model_directory)

        self._tokenizer_temp_dir = tempfile.TemporaryDirectory(
            prefix="lifemate-tokenizer-"
        )
        tokenizer_directory = Path(self._tokenizer_temp_dir.name)
        tokenizer_files = {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "vocab.txt",
            "tokenizer.model",
            "spiece.model",
            "chat_template.jinja",
        }
        for source in model_directory.iterdir():
            if source.is_file() and source.name in tokenizer_files:
                shutil.copy2(source, tokenizer_directory / source.name)

        tokenizer_config["extra_special_tokens"] = {
            token: token for token in extra_special_tokens
        }
        (tokenizer_directory / "tokenizer_config.json").write_text(
            json.dumps(tokenizer_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            "[LLM] Normalized extra_special_tokens in a temporary tokenizer copy.",
            flush=True,
        )
        return str(tokenizer_directory)

    def _prepare_inputs(self, messages: list):
        messages_to_encode = list(messages)
        protected_message_ids = set()

        if messages_to_encode and messages_to_encode[0].get("role") == "system":
            protected_message_ids.add(id(messages_to_encode[0]))

        explicitly_protected_messages = [
            message
            for message in messages_to_encode
            if message.get("_preserve_for_generation")
        ]
        if explicitly_protected_messages:
            protected_message_ids.update(
                id(message) for message in explicitly_protected_messages
            )
        else:
            first_user_message = next(
                (
                    message
                    for message in messages_to_encode
                    if message.get("role") == "user"
                ),
                None,
            )
            if first_user_message is not None:
                protected_message_ids.add(id(first_user_message))

        latest_user_message = next(
            (
                message
                for message in reversed(messages_to_encode)
                if message.get("role") == "user"
            ),
            None,
        )
        if latest_user_message is not None:
            protected_message_ids.add(id(latest_user_message))

        dropped_messages = 0
        while True:
            text_prompt = self.tokenizer.apply_chat_template(
                [
                    {
                        key: value
                        for key, value in message.items()
                        if key != "_preserve_for_generation"
                    }
                    for message in messages_to_encode
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer([text_prompt], return_tensors="pt")
            input_token_count = inputs["input_ids"].shape[-1]
            if input_token_count <= self.max_input_tokens:
                return inputs.to(self.device), input_token_count, dropped_messages

            removable_index = next(
                (
                    index
                    for index, message in enumerate(messages_to_encode)
                    if id(message) not in protected_message_ids
                ),
                None,
            )
            if removable_index is None:
                raise ValueError(
                    "Required system and user messages exceed the prompt limit "
                    f"of {self.max_input_tokens} tokens "
                    f"({input_token_count} tokens). Shorten the system prompt "
                    "or the required user content."
                )

            del messages_to_encode[removable_index]
            dropped_messages += 1

    def generate(self, messages: list) -> str:
        total_started = time.perf_counter()
        prepare_started = time.perf_counter()
        inputs, input_token_count, dropped_messages = self._prepare_inputs(messages)
        is_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        if is_cuda:
            torch.cuda.synchronize(self.device)
        prepare_seconds = time.perf_counter() - prepare_started

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

        generation_started = time.perf_counter()
        try:
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        except Exception:
            generation_seconds = time.perf_counter() - generation_started
            print(
                f"[LLM_PERF] status=error; input_tokens={input_token_count}; "
                f"prepare_seconds={prepare_seconds:.3f}; "
                f"generation_seconds={generation_seconds:.3f}",
                flush=True,
            )
            raise

        if is_cuda:
            torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - generation_started

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        decode_started = time.perf_counter()
        output_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        decode_seconds = time.perf_counter() - decode_started
        output_token_count = sum(output_ids.numel() for output_ids in generated_ids)
        total_seconds = time.perf_counter() - total_started
        self.last_generation_stats = {
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "dropped_messages": dropped_messages,
            "prepare_seconds": prepare_seconds,
            "generation_seconds": generation_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": total_seconds,
            "output_tokens_per_second": (
                output_token_count / generation_seconds
                if generation_seconds > 0
                else 0.0
            ),
        }

        if is_cuda:
            memory_report = {
                "cuda_allocated_gib": torch.cuda.memory_allocated(self.device)
                / 1024**3,
                "cuda_reserved_gib": torch.cuda.memory_reserved(self.device)
                / 1024**3,
                "cuda_peak_gib": torch.cuda.max_memory_allocated(self.device)
                / 1024**3,
            }
        else:
            memory_report = {"device": self.device}
        self.last_generation_stats.update(memory_report)
        print(
            "[LLM_PERF] "
            + json.dumps(self.last_generation_stats, ensure_ascii=False),
            flush=True,
        )
        return output_text