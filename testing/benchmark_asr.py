"""Benchmark Vietnamese ASR models against train.jsonl audio_text labels.

Run from the agent directory:
    python testing/benchmark_asr.py --models phowhisper zipformer --limit 20
    python testing/benchmark_asr.py --models phowhisper zipformer

ZipFormer requires the CUDA-enabled sherpa-onnx wheel and a working CUDA
execution provider. Its model files are downloaded from Hugging Face on demand.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import statistics
import sys
import time
import unicodedata
import wave
from pathlib import Path
from typing import Any, Sequence

AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from configs.paths import get_split_dir
from src.inference.asr_engine import (
    ASREngine,
    ZIPFORMER_MODEL_ID,
    ZIPFORMER_REVISION,
)
from src.inference.device import select_compute_device


PHOWHISPER_MODEL_ID = "vinai/PhoWhisper-small"


def normalize_text(text: str) -> str:
    """Lowercase, normalize Unicode, remove punctuation, and collapse spaces."""
    normalized = unicodedata.normalize("NFC", text).lower()
    return " ".join(
        "".join(
            char if char.isalnum() or char.isspace() else " "
            for char in normalized
        ).split()
    )


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous_row = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current_row = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[hyp_index] + 1,
                    previous_row[hyp_index - 1] + (ref_token != hyp_token),
                )
            )
        previous_row = current_row
    return previous_row[-1]


def error_counts(reference: str, hypothesis: str) -> dict[str, int]:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    return {
        "word_errors": edit_distance(reference_words, hypothesis_words),
        "reference_words": len(reference_words),
        "char_errors": edit_distance(list(reference), list(hypothesis)),
        "reference_chars": len(reference),
    }


def load_items(
    data_path: Path, limit: int | None, requested_ids: list[str] | None
) -> list[dict[str, Any]]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be greater than zero.")
    if requested_ids and len(requested_ids) != len(set(requested_ids)):
        raise ValueError("--ids must not contain duplicates.")

    wanted_ids = set(requested_ids or [])
    items_by_id: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    with data_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item.get("audio_text"), str) or not item["audio_text"].strip():
                raise ValueError(
                    f"{data_path}:{line_number} is missing a non-empty audio_text label."
                )
            if requested_ids:
                if item.get("id") in wanted_ids:
                    items_by_id[item["id"]] = item
            else:
                items.append(item)
                if limit is not None and len(items) >= limit:
                    break

    if requested_ids:
        missing_ids = [sample_id for sample_id in requested_ids if sample_id not in items_by_id]
        if missing_ids:
            raise ValueError(f"IDs not found in {data_path}: {missing_ids[:10]}")
        items = [items_by_id[sample_id] for sample_id in requested_ids]
        if limit is not None:
            items = items[:limit]
    if not items:
        raise ValueError(f"No benchmark samples found in {data_path}.")
    return items


def _load_audio_16k(audio_path: Path):
    import torchaudio

    from src.inference.asr_engine import _load_audio

    waveform, sample_rate = _load_audio(str(audio_path))
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )(waveform.unsqueeze(0)).squeeze(0)
    return waveform.numpy()


class PhoWhisperTranscriber:
    def __init__(self, dtype_name: str):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        device_name, hardware_name = select_compute_device(torch)
        if not device_name.startswith("cuda"):
            raise RuntimeError(
                "PhoWhisper benchmarking requires an NVIDIA A30 or T4; "
                f"selected device: {hardware_name}"
            )
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
        self.device = torch.device(device_name)
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(PHOWHISPER_MODEL_ID)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            PHOWHISPER_MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()

    def transcribe(self, audio):
        import torch

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(
            device=self.device, dtype=self.dtype
        )
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_features,
                language="vi",
                task="transcribe",
                num_beams=1,
            )
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class ZipformerTranscriber:
    def __init__(self, model_revision: str, num_threads: int):
        self.engine = ASREngine(
            model_id=ZIPFORMER_MODEL_ID,
            revision=model_revision,
            num_threads=num_threads,
        )
        self.device = self.engine.device

    def transcribe(self, audio_path: str) -> str:
        return self.engine.transcribe(audio_path)


def _gpu_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _sample_path(item: dict[str, Any], split_dir: Path) -> Path:
    audio_path = (split_dir / item["audio"]).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file for {item['id']} was not found: {audio_path}")
    return audio_path


def run_model(
    model_name: str,
    items: list[dict[str, Any]],
    split_dir: Path,
    output_dir: Path,
    warmup_count: int,
    dtype_name: str,
    zipformer_revision: str,
    num_threads: int,
) -> dict[str, Any]:
    import torch

    if model_name == "phowhisper":
        model = PhoWhisperTranscriber(dtype_name)
        model_id = PHOWHISPER_MODEL_ID
    else:
        model = ZipformerTranscriber(zipformer_revision, num_threads)
        model_id = ZIPFORMER_MODEL_ID

    warmup_items = items[: min(warmup_count, len(items))]
    for item in warmup_items:
        audio_path = _sample_path(item, split_dir)
        if model_name == "zipformer":
            model.transcribe(str(audio_path))
        else:
            model.transcribe(_load_audio_16k(audio_path))
    _gpu_sync()

    output_path = output_dir / f"{model_name}.jsonl"
    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, item in enumerate(items, start=1):
            audio_path = _sample_path(item, split_dir)
            with wave.open(str(audio_path), "rb") as audio_file:
                audio_seconds = audio_file.getnframes() / audio_file.getframerate()
            _gpu_sync()
            started = time.perf_counter()
            if model_name == "zipformer":
                prediction = model.transcribe(str(audio_path))
            else:
                prediction = model.transcribe(_load_audio_16k(audio_path))
            _gpu_sync()
            elapsed = time.perf_counter() - started

            counts = error_counts(item["audio_text"], prediction)
            city = (
                item.get("context", {})
                .get("user_profile", {})
                .get("city")
            )
            record = {
                "id": item["id"],
                "context_city": city,
                "audio_seconds": audio_seconds,
                "asr_seconds": elapsed,
                "real_time_factor": elapsed / audio_seconds if audio_seconds else 0.0,
                "reference": item["audio_text"],
                "hypothesis": prediction,
                **counts,
            }
            records.append(record)
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            print(
                f"[{model_name}] {index}/{len(items)} {item['id']} "
                f"ASR={elapsed:.2f}s",
                flush=True,
            )

    summary = summarize_results(records)
    summary.update(
        {
            "model": model_id,
            "model_revision": (
                zipformer_revision if model_name == "zipformer" else None
            ),
            "samples": len(records),
            "warmup_samples_excluded": len(warmup_items),
            "precision": (
                dtype_name if model_name == "phowhisper" else "onnx model default"
            ),
            "provider": model.device if model_name == "zipformer" else "cuda",
            "results_file": str(output_path),
        }
    )
    (output_dir / f"{model_name}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_word_errors = sum(record["word_errors"] for record in records)
    total_reference_words = sum(record["reference_words"] for record in records)
    total_char_errors = sum(record["char_errors"] for record in records)
    total_reference_chars = sum(record["reference_chars"] for record in records)
    latencies = [record["asr_seconds"] for record in records]
    real_time_factors = [record["real_time_factor"] for record in records]
    by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_city[record["context_city"] or "unknown"].append(record)

    return {
        "wer": total_word_errors / total_reference_words if total_reference_words else 0.0,
        "cer": total_char_errors / total_reference_chars if total_reference_chars else 0.0,
        "total_word_errors": total_word_errors,
        "total_reference_words": total_reference_words,
        "total_char_errors": total_char_errors,
        "total_reference_chars": total_reference_chars,
        "latency_seconds": {
            "mean": statistics.mean(latencies),
            "p50": statistics.median(latencies),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "real_time_factor": {
            "mean": statistics.mean(real_time_factors),
            "p95": _percentile(real_time_factors, 0.95),
        },
        "by_context_city": {
            city: {
                "samples": len(city_records),
                "wer": (
                    sum(record["word_errors"] for record in city_records)
                    / sum(record["reference_words"] for record in city_records)
                    if sum(record["reference_words"] for record in city_records)
                    else 0.0
                ),
                "cer": (
                    sum(record["char_errors"] for record in city_records)
                    / sum(record["reference_chars"] for record in city_records)
                    if sum(record["reference_chars"] for record in city_records)
                    else 0.0
                ),
            }
            for city, city_records in sorted(by_city.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GPU ASR models against audio_text in train.jsonl."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["phowhisper", "zipformer"],
        default=["phowhisper", "zipformer"],
        help="ASR models to run.",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N train samples.")
    parser.add_argument("--ids", nargs="+", help="Run selected sample IDs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AGENT_DIR / "outputs" / "asr_benchmark",
        help="Directory for per-sample JSONL and summary JSON outputs.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of initial samples to warm up the model before timing.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Precision for PhoWhisper on CUDA.",
    )
    parser.add_argument(
        "--zipformer-revision",
        default=ZIPFORMER_REVISION,
        help="Hugging Face revision for ZipFormer model files.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="CPU threads used by sherpa-onnx in addition to its CUDA provider.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero.")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative.")
    if args.num_threads < 1:
        parser.error("--num-threads must be greater than zero.")
    if args.ids and args.limit is not None:
        parser.error("--ids cannot be combined with --limit.")

    import torch

    split_dir = get_split_dir("train")
    data_path = split_dir / "train.jsonl"
    items = load_items(data_path, args.limit, args.ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"CUDA available: {torch.cuda.is_available()}; "
        f"samples: {len(items)}; "
        f"outputs: {args.output_dir}",
        flush=True,
    )

    summaries = {}
    for model_name in args.models:
        print(f"Loading {model_name}...", flush=True)
        summaries[model_name] = run_model(
            model_name=model_name,
            items=items,
            split_dir=split_dir,
            output_dir=args.output_dir,
            warmup_count=args.warmup,
            dtype_name=args.precision,
            zipformer_revision=args.zipformer_revision,
            num_threads=args.num_threads,
        )

    combined_summary = {
        "dataset": str(data_path),
        "sample_count": len(items),
        "gpu": torch.cuda.get_device_name(0),
        "metric_normalization": (
            "Unicode NFC, lowercase, punctuation removed, whitespace collapsed; "
            "WER tokenizes on whitespace and CER counts characters including spaces."
        ),
        "models": summaries,
    }
    summary_path = args.output_dir / "comparison.summary.json"
    summary_path.write_text(
        json.dumps(combined_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}", flush=True)
    for name, summary in summaries.items():
        latency = summary["latency_seconds"]
        print(
            f"{name}: WER={summary['wer']:.2%}, CER={summary['cer']:.2%}, "
            f"ASR p50={latency['p50']:.2f}s, p95={latency['p95']:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
