import wave

import numpy as np
import torch

from src.inference import asr_engine
from src.inference.asr_engine import ASREngine, _load_audio


def _write_wav(path, samples, sample_width, sample_rate=16000, channels=1):
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(channels)
        audio_file.setsampwidth(sample_width)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(samples)


def test_load_audio_reads_pcm_16bit_without_torchcodec(tmp_path):
    audio_path = tmp_path / "sample.wav"
    expected = np.array([-32768, 0, 16384], dtype="<i2")
    _write_wav(audio_path, expected.tobytes(), sample_width=2)

    waveform, sample_rate = _load_audio(str(audio_path))

    assert sample_rate == 16000
    assert torch.allclose(
        waveform, torch.tensor([-1.0, 0.0, 0.5]), atol=1e-6
    )


def test_load_audio_downmixes_stereo_to_mono(tmp_path):
    audio_path = tmp_path / "stereo.wav"
    samples = np.array([[32767, -32768], [16384, 16384]], dtype="<i2")
    _write_wav(audio_path, samples.tobytes(), sample_width=2, channels=2)

    waveform, _ = _load_audio(str(audio_path))

    expected = torch.tensor(
        [(32767 / 32768 - 1.0) / 2, 0.5], dtype=torch.float32
    )
    assert torch.allclose(waveform, expected, atol=1e-6)


def test_load_audio_decodes_24bit_pcm(tmp_path):
    audio_path = tmp_path / "24bit.wav"
    samples = bytes([0, 0, 128, 0, 0, 64])
    _write_wav(audio_path, samples, sample_width=3)

    waveform, _ = _load_audio(str(audio_path))

    assert torch.allclose(waveform, torch.tensor([-1.0, 0.5]), atol=1e-6)


def test_select_compute_device_prefers_a30_then_t4_then_cpu(monkeypatch):
    class MockCuda:
        available = True
        names = ["Tesla T4", "NVIDIA A30"]
        selected = None

        def is_available(self):
            return self.available

        def device_count(self):
            return len(self.names)

        def get_device_name(self, index):
            return self.names[index]

        def set_device(self, index):
            self.selected = index

    cuda = MockCuda()
    monkeypatch.setattr(asr_engine, "torch", type("Torch", (), {"cuda": cuda}))

    device, name = asr_engine.select_compute_device(asr_engine.torch)

    assert device == "cuda:1"
    assert name == "NVIDIA A30"
    assert cuda.selected == 1


def test_select_compute_device_uses_t4_if_a30_is_not_present(monkeypatch):
    class MockCuda:
        def is_available(self):
            return True

        def device_count(self):
            return 1

        def get_device_name(self, index):
            return "Tesla T4"

        def set_device(self, index):
            self.selected = index

    cuda = MockCuda()
    device, name = asr_engine.select_compute_device(
        type("Torch", (), {"cuda": cuda})
    )

    assert (device, name) == ("cuda:0", "Tesla T4")


def test_select_compute_device_falls_back_to_cpu_without_supported_gpu():
    class MockCuda:
        def is_available(self):
            return True

        def device_count(self):
            return 1

        def get_device_name(self, index):
            return "NVIDIA L4"

    assert asr_engine.select_compute_device(
        type("Torch", (), {"cuda": MockCuda()})
    ) == ("cpu", "CPU (no NVIDIA A30/T4 found; visible GPUs: NVIDIA L4)")


def test_asr_engine_falls_back_to_cpu_when_cuda_provider_fails(monkeypatch):
    import sys
    import types

    providers = []

    class MockRecognizer:
        @classmethod
        def from_transducer(cls, **kwargs):
            providers.append(kwargs["provider"])
            if kwargs["provider"] == "cuda":
                raise RuntimeError("CUDA provider not available")
            return cls()

    sherpa = types.SimpleNamespace(
        OfflineRecognizer=MockRecognizer,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", sherpa)
    monkeypatch.setattr(
        asr_engine,
        "select_compute_device",
        lambda torch_module: ("cuda:0", "NVIDIA A30"),
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, filename, revision: filename,
    )

    engine = ASREngine()

    assert providers == ["cuda", "cpu"]
    assert engine.device == "cpu"
