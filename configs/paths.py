"""
configs/paths.py
Quản lý tập trung toàn bộ đường dẫn trong hệ thống.
"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn dữ liệu Zalo AI
DATA_DIR = Path(os.getenv("ZALO_DATA_DIR", BASE_DIR / ".." / "train+public_test")).resolve()

# Thư mục lưu checkpoint SFT dạng full model hoặc PEFT adapter.
CHECKPOINT_DIR = (BASE_DIR / "checkpoint_sft_final").resolve()
SFT_ADAPTER_DIR = Path(
    os.getenv(
        "LIFEMATE_SFT_ADAPTER_DIR",
        BASE_DIR / "lifemate_sft_export" / "adapter",
    )
).resolve()
BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def get_inference_model_path(require_sft: bool = False) -> str:
    """Prefer the exported LoRA adapter, then a local full-model checkpoint."""
    if (SFT_ADAPTER_DIR / "adapter_config.json").is_file():
        return str(SFT_ADAPTER_DIR)
    if (CHECKPOINT_DIR / "adapter_config.json").is_file():
        return str(CHECKPOINT_DIR)
    if (CHECKPOINT_DIR / "config.json").is_file():
        return str(CHECKPOINT_DIR)
    if require_sft:
        raise FileNotFoundError(
            "No trained adapter or full-model checkpoint found. "
            f"Adapter: {SFT_ADAPTER_DIR}; checkpoint: {CHECKPOINT_DIR}."
        )
    return BASE_MODEL_ID

def get_split_dir(split_name: str) -> Path:
    return DATA_DIR / split_name