"""
src/inference/asr_engine.py
Hệ thống ASR chuyển đổi Audio (.wav) sang Text tiếng Việt.
"""

import wave

import numpy as np
import torch
import torchaudio

from src.inference.device import select_compute_device


ZIPFORMER_MODEL_ID = "hynt/Zipformer-30M-RNNT-6000h"
ZIPFORMER_REVISION = "ad6f873b5f5b6ed083821253a88d81afed2505c4"
ZIPFORMER_FILES = {
    "encoder": "encoder-epoch-20-avg-10.onnx",
    "decoder": "decoder-epoch-20-avg-10.onnx",
    "joiner": "joiner-epoch-20-avg-10.onnx",
    "tokens": "config.json",
}


def _load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
    with wave.open(audio_path, "rb") as audio_file:
        if audio_file.getcomptype() != "NONE":
            raise ValueError(f"Unsupported WAV compression: {audio_file.getcomptype()}")

        channels = audio_file.getnchannels()
        sample_width = audio_file.getsampwidth()
        sample_rate = audio_file.getframerate()
        audio_bytes = audio_file.readframes(audio_file.getnframes())

    if sample_width == 1:
        samples = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128) / 128
    elif sample_width == 2:
        samples = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768
    elif sample_width == 3:
        bytes_per_sample = np.frombuffer(audio_bytes, dtype=np.uint8).reshape(-1, 3)
        values = (
            bytes_per_sample[:, 0].astype(np.int32)
            | (bytes_per_sample[:, 1].astype(np.int32) << 8)
            | (bytes_per_sample[:, 2].astype(np.int32) << 16)
        )
        samples = ((values ^ 0x800000) - 0x800000).astype(np.float32) / 8388608
    elif sample_width == 4:
        samples = np.frombuffer(audio_bytes, dtype="<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    samples = samples.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(samples), sample_rate


class ASREngine:
    def __init__(
        self,
        model_id: str = ZIPFORMER_MODEL_ID,
        revision: str = ZIPFORMER_REVISION,
        num_threads: int = 1,
    ):
        if model_id != ZIPFORMER_MODEL_ID:
            raise ValueError(
                f"Only the configured Vietnamese ZipFormer model is supported: {model_id}"
            )
        if num_threads < 1:
            raise ValueError("num_threads must be greater than zero.")

        try:
            import sherpa_onnx
        except ModuleNotFoundError as error:
            if error.name != "sherpa_onnx":
                raise
            raise RuntimeError(
                "ZipFormer requires sherpa-onnx. Install a CUDA-enabled wheel for "
                "GPU inference, or the standard sherpa-onnx wheel for CPU fallback."
            ) from error

        from huggingface_hub import hf_hub_download

        self.model_id = model_id
        self.revision = revision
        self.num_threads = num_threads
        self.device, device_name = select_compute_device(torch)
        model_files = {
            key: hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
            )
            for key, filename in ZIPFORMER_FILES.items()
        }

        provider = "cuda" if self.device.startswith("cuda") else "cpu"
        try:
            self.recognizer = self._create_recognizer(
                sherpa_onnx, model_files, provider
            )
        except (RuntimeError, ValueError, OSError) as error:
            if provider != "cuda":
                raise
            print(
                f"[ASR] CUDA provider unavailable on {device_name}; "
                f"falling back to CPU: {error}",
                flush=True,
            )
            self.device = "cpu"
            self.recognizer = self._create_recognizer(
                sherpa_onnx, model_files, "cpu"
            )
            provider = "cpu"
        print(
            f"[ASR] model={model_id}@{revision}; "
            f"device={self.device}; provider={provider}",
            flush=True,
        )

    def _create_recognizer(
        self, sherpa_onnx, model_files: dict[str, str], provider: str
    ):
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=model_files["encoder"],
            decoder=model_files["decoder"],
            joiner=model_files["joiner"],
            tokens=model_files["tokens"],
            num_threads=self.num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=provider,
        )

    def transcribe(self, audio_path: str) -> str:
        """Transcribe file âm thanh .wav sang văn bản."""
        waveform, sample_rate = _load_audio(audio_path)
        
        # Resample về 16kHz nếu cần
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, waveform.numpy())
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()