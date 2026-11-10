# LifeMate

LifeMate maps Vietnamese speech requests to tool calls using an ASR stage and a
fine-tuned Qwen language model.

## Repository layout

- `training/`: raw-data preparation and LoRA SFT training.
- `testing/`: model benchmarking and agent evaluation.
- `src/inference/`: ZipFormer ASR, compute-device selection, and the Qwen engine.
- `src/agent/`, `src/tools/`, `src/utils/`: planning and application logic.
- `tests/`: unit tests.
- `configs/`: model and data paths and training configuration.

The disconnected demo router/tools, placeholder DPO/synthetic-data commands,
duplicate evaluation entrypoint, and broken demo `main.py` were removed. The
ZAIC submission runner remains the single inference entrypoint.

## Inference hardware and models

The runner uses Vietnamese ZipFormer-30M-RNNT-6000h through `sherpa-onnx` and
selects a visible NVIDIA A30 first, then a T4, then CPU. The Qwen LLM uses the
same preference and loads in BF16 when supported, otherwise FP16 on the selected
GPU. If ZipFormer cannot initialize the CUDA execution provider, it reports the
reason and retries on CPU.

The ZipFormer model files are downloaded from the pinned Hugging Face revision
`ad6f873b5f5b6ed083821253a88d81afed2505c4`. The model card lists license
`CC-BY-NC-ND-4.0`; confirm that these terms permit the intended competition
submission and redistribution before use.

## Local setup

Install the project's dependencies in an environment whose PyTorch and
torchaudio versions match:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-asr-cpu.txt
```

For GPU inference, install the `sherpa-onnx` CUDA wheel compatible with the
machine's CUDA/cuDNN runtime instead of the CPU wheel. Consult the official
[sherpa-onnx installation guide](https://k2-fsa.github.io/sherpa/onnx/python/install.html)
and verify CUDA execution provider availability on the target machine.

Set `ZALO_DATA_DIR` to the extracted `train+public_test` directory and
`LIFEMATE_SFT_ADAPTER_DIR` to the exported adapter directory. The default paths
expect those directories beside `agent/`.

## Kaggle inference

Use [kaggle_run_submission.ipynb](./kaggle_run_submission.ipynb) to clone the
current source from GitHub and run `run_submission.py`. Push the repository to
GitHub first, then set `GITHUB_REPO_URL` in the notebook's configuration cell.
Attach the challenge dataset and exported SFT adapter as Kaggle inputs; the
notebook discovers their paths automatically or accepts explicit path
overrides. ZIP inputs are safely extracted when needed. The dataset input must
include `train+public_test/public_test/`, and
the adapter must include `adapter_config.json`, `adapter_model.safetensors`,
tokenizer files, and `chat_template.jinja`.

Enable Internet to install dependencies and download the base Qwen model and
pinned ZipFormer files. Select an A30 accelerator when available; otherwise use
a T4. The notebook tries compatible CUDA-enabled sherpa-onnx wheels and falls
back to the standard CPU wheel if unavailable. It preserves Kaggle's installed
PyTorch/CUDA/torchaudio versions.

The notebook runs:

```bash
python -u run_submission.py \
  --split public_test \
  --output /kaggle/working/submission.jsonl \
  --resume
```

The runner validates the dataset paths, writes each result immediately, checks
resume JSONL files for malformed or duplicate IDs, skips completed samples, and
verifies that the final output contains every input ID. To resume from a
checkpoint attached as a Kaggle input, set `LIFEMATE_RESUME_FROM` to its
`submission.jsonl` path in the notebook's configuration cell. The tokenizer
compatibility fix is applied to a temporary tokenizer-only copy; the read-only
adapter input is never modified.

## Training and evaluation

Run commands from the `agent/` directory:

```bash
python training/prepare_train_data.py
python training/train_sft.py
python testing/evaluate.py --smoke
python testing/evaluate.py --require-checkpoint
python testing/benchmark_asr.py --models zipformer --limit 20
```

The ASR benchmark compares model transcripts with the `audio_text` references
in `train.jsonl`; it reports WER, CER, latency, and real-time factor. ZipFormer
can also benchmark on CPU. PhoWhisper benchmarking requires an A30 or T4.

Run unit tests with:

```bash
python -m pytest tests
```
