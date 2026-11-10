"""Hardware selection shared by ASR and LLM inference."""

from __future__ import annotations

from typing import Any


PREFERRED_GPU_MODELS = ("A30", "T4")


def select_compute_device(torch_module: Any) -> tuple[str, str]:
    """Prefer a visible A30, then T4; use CPU when neither is available."""
    cuda = torch_module.cuda
    if not cuda.is_available():
        return "cpu", "CPU"

    candidates: list[tuple[int, int, str]] = []
    for index in range(cuda.device_count()):
        name = cuda.get_device_name(index)
        normalized_name = name.upper()
        for priority, model in enumerate(PREFERRED_GPU_MODELS):
            if model in normalized_name:
                candidates.append((priority, index, name))
                break

    if not candidates:
        names = ", ".join(cuda.get_device_name(index) for index in range(cuda.device_count()))
        return "cpu", f"CPU (no NVIDIA A30/T4 found; visible GPUs: {names})"

    _, index, name = min(candidates)
    cuda.set_device(index)
    return f"cuda:{index}", name
