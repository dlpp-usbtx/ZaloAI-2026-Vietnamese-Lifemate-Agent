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
PyTorch/CUDA/torchaudio versions. Its Kaggle requirements constrain
`tokenizers` to `>=0.20,<0.22`, which is compatible with the selected
Transformers versions and replaces incompatible newer Kaggle preinstalls
without changing the PyTorch/CUDA stack.

The notebook runs:

```bash
python -u run_submission.py \
  --split public_test \
  --output /kaggle/working/submission.jsonl \
  --performance-log /kaggle/working/lifemate_performance.jsonl \
  --max-llm-calls 2 \
  --planner-mode react \
  --max-available-tools 0 \
  --resume
```

`--planner-mode react` retains tool feedback and iterative planning.
`--planner-mode batch` asks for an ordered tool-call list and executes it after
one generation; this can reduce latency for multi-tool requests, but its
accuracy must be checked against validation because the exported adapter was
trained on single-call turns.

Compare both modes on the same validation IDs before using batch for a
submission:

```bash
python testing/evaluate.py --planner-mode react --max-llm-calls 2
python testing/evaluate.py --planner-mode batch --max-llm-calls 1
python testing/evaluate.py --planner-mode batch --max-llm-calls 1 --max-available-tools 8
```

Use `--limit 8` for an initial quick comparison, then run without `--limit` to
compare all 154 validation examples. Results are saved under distinct filenames
and report tool-call F1/exact match, state match, planning latency, LLM call
count, and budget exhaustion.

The runner validates the dataset paths, writes each result immediately, checks
resume JSONL files for malformed or duplicate IDs, skips completed samples, and
verifies that the final output contains every input ID. To resume from a
checkpoint attached as a Kaggle input, set `LIFEMATE_RESUME_FROM` to its
`submission.jsonl` path in the notebook's configuration cell. The tokenizer
compatibility fix is applied to a temporary tokenizer-only copy; the read-only
adapter input is never modified.

The runner writes a separate performance JSONL without changing the submission
schema. It records per-sample ASR and planner durations, per-call prompt/output
token counts, input preparation, generation and decoding times, tool
validation/execution times, and GPU memory. Startup model-loading times are
printed separately. The Kaggle notebook reports sample mean/p95/max, average
LLM calls per sample, generation latency, and output-token counts after
inference. The sidecar also records the ASR transcript (and a reference when
the input provides one), sample time context, selected model/device/dtype,
available tool names, parser decisions, ordered proposed tool calls, and
tool-result status keys for diagnosing ASR, planning, and execution issues.
`--max-available-tools 0` preserves the full installed tool list; positive
values enable a lightweight Vietnamese/English lexical relevance shortlist.
Keep this disabled unless a validation comparison shows that the shortlist
preserves tool-call accuracy.

The default two-call budget bounds retries and multi-step tool planning; raise
`MAX_LLM_CALLS_PER_SAMPLE` in the notebook if validation shows that complex
multi-tool samples lose accuracy. The LLM uses SDPA attention and constrained
JSON generation (including Vietnamese text characters) to avoid spending
extra inference calls repairing invalid output formats.

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
