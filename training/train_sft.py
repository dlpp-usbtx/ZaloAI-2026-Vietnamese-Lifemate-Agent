"""
Fine-tune the configured base model with LoRA.

The trainer arguments are adapted to the installed TRL/Transformers API so this
script works with both the pinned legacy stack and newer SFTTrainer releases.
"""

import argparse
import inspect
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None

from configs.paths import BASE_DIR, CHECKPOINT_DIR


def build_training_arguments(config: dict, output_dir: Path):
    training = config["training"]
    values = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "learning_rate": training["learning_rate"],
        "num_train_epochs": training["num_train_epochs"],
        "warmup_ratio": training["warmup_ratio"],
        "logging_steps": training["logging_steps"],
        "save_strategy": training["save_strategy"],
        "save_total_limit": training.get("save_total_limit", 2),
        "report_to": "none",
    }

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    values["bf16"] = use_bf16
    values["fp16"] = use_fp16

    if SFTConfig is not None:
        parameters = inspect.signature(SFTConfig).parameters
        evaluation_key = (
            "eval_strategy" if "eval_strategy" in parameters
            else "evaluation_strategy"
        )
        values[evaluation_key] = training.get("evaluation_strategy", "epoch")
        values["max_length"] = config["max_seq_length"]
        supported_values = {
            key: value for key, value in values.items() if key in parameters
        }
        return SFTConfig(**supported_values), use_bf16, use_fp16

    parameters = inspect.signature(TrainingArguments).parameters
    evaluation_key = (
        "eval_strategy" if "eval_strategy" in parameters
        else "evaluation_strategy"
    )
    values[evaluation_key] = training.get("evaluation_strategy", "epoch")
    supported_values = {
        key: value for key, value in values.items() if key in parameters
    }
    return TrainingArguments(**supported_values), use_bf16, use_fp16


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LifeMate with SFT.")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a very slow CPU-only training run (not recommended for 3B models).",
    )
    args = parser.parse_args()

    config_path = BASE_DIR / "configs" / "sft_config.yaml"
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    train_path = BASE_DIR / "data" / "processed" / "sft_train.json"
    validation_path = BASE_DIR / "data" / "processed" / "sft_val.json"
    for data_path in (train_path, validation_path):
        if not data_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy {data_path}; chạy training/prepare_train_data.py trước."
            )

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "PyTorch hiện không thấy CUDA/GPU. Huấn luyện Qwen 3B trên CPU sẽ rất "
            "chậm; hãy dùng môi trường có GPU hoặc thêm --allow-cpu nếu bạn chủ "
            "động chấp nhận thời gian chạy dài."
        )

    model_id = config["model_name_or_path"]
    output_dir = CHECKPOINT_DIR
    print(f"Base model: {model_id}")
    print(
        "Nếu model chưa có trong Hugging Face cache, from_pretrained sẽ tải "
        "weights từ Hugging Face; cần kết nối mạng và dung lượng đĩa trống."
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    model_dtype = (
        torch.bfloat16 if use_bf16
        else torch.float16 if use_fp16
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    lora = config["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        target_modules=lora["target_modules"],
        lora_dropout=lora["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(validation_path),
        },
    )
    if not len(dataset["train"]) or not len(dataset["validation"]):
        raise ValueError("Tập train và validation phải đều có ít nhất một mẫu.")

    training_args, _, _ = build_training_arguments(config, output_dir)
    trainer_parameters = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "peft_config": peft_config,
        "args": training_args,
    }

    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    if "max_seq_length" in trainer_parameters:
        trainer_kwargs["max_seq_length"] = config["max_seq_length"]
    elif "max_length" in trainer_parameters and SFTConfig is None:
        trainer_kwargs["max_length"] = config["max_seq_length"]

    if "dataset_text_field" in trainer_parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    elif SFTConfig is not None:
        sft_config_parameters = inspect.signature(SFTConfig).parameters
        if "dataset_text_field" in sft_config_parameters:
            training_args.dataset_text_field = "text"

    if "formatting_func" in trainer_parameters and SFTConfig is None:
        trainer_kwargs["formatting_func"] = lambda row: tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    trainer = SFTTrainer(**trainer_kwargs)
    print(f"Huấn luyện {len(dataset['train'])} mẫu; validation {len(dataset['validation'])} mẫu.")
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Đang merge LoRA và lưu checkpoint tại: {output_dir}")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Huấn luyện hoàn tất.")


if __name__ == "__main__":
    main()
