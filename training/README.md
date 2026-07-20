# Metric-embedding trainer (GPU)

Offline training subsystem for the few-shot waveform embedding. Trains on the
Apple GPU (PyTorch MPS), then exports the compact, browser-native inference
assets consumed by `src/embedding/`. See [`../DESIGN.md`](../DESIGN.md) for the
architecture and rationale.

## Setup (one time)

```bash
python3 -m venv .venv-training
.venv-training/bin/pip install numpy torch onnx
```

`.venv-training/` is git-ignored. Requires an Apple-Silicon Mac for MPS; falls
back to CPU automatically if MPS is unavailable.

## Train + export

```bash
.venv-training/bin/python training/train.py
```

Runs ~8k prototypical episodes (~2 min on an M-series GPU), restores the
best-validation checkpoint, calibrates the open-set threshold and fusion
temperature, folds BatchNorm into the convolutions, verifies a pure-numpy
forward pass matches the torch model bit-for-bit, and writes:

- `src/embedding/assets/embedding-weights.json` — folded conv/linear weights +
  feature standardisation + preprocessing params (committed, ~0.8 MB).
- `src/embedding/assets/prototypes.json` — class prototypes, unknown threshold,
  fusion temperature (committed).
- `src/embedding/assets/parity-fixture.json` — inputs + reference outputs the TS
  parity tests assert against (committed).
- `src/embedding/assets/model-manifest.json` — seed, class lists, headline metrics.
- `training/artifacts/embedding.onnx` — alternative onnxruntime-web path (ignored).

The pipeline is seeded (`SEED` in `train.py`); a given seed reproduces a given
model. The TS runtime is a faithful port, so retraining requires no TS changes —
only the regenerated assets.

## Evaluate (honest held-out report + regression gate)

```bash
.venv-training/bin/python training/evaluate.py
```

Reports closed-set accuracy (fine + family + per-class + per-SNR + per-bandwidth),
few-shot re-enrollment (leave-one-known-class-out), open-set AUROC (per novel
class), and exits non-zero if any headline metric falls below its evidence-based
floor. Writes `training/artifacts/eval-report.json`.

## Files

| file | role |
|------|------|
| `rfgen.py` | modulator bank + randomised impairment channel |
| `preprocess.py` | detect → downconvert → resample → normalise + cumulant features |
| `model.py` | 1D-CNN metric embedding (PyTorch), BN-fold export |
| `dataset.py` | impaired-realisation pools + episodic sampler |
| `train.py` | GPU training, calibration, parity check, asset export |
| `evaluate.py` | held-out evaluation + regression gate |
| `recover.py` | blind CMA equalize + sync + SNR/quality (order push-through) |
| `order_refine.py` | quality-gated 16/64-QAM order refiner + calibration |

## Order push-through (ingestion-side)

The embedding resolves modulation *family* but not linear-digital *order* (16- vs
64-QAM) — an information limit at its front-end fidelity. `recover.py` blindly
equalizes and synchronizes a capture (a fractionally-spaced CMA equalizer that
inverts the channel and absorbs timing), and `order_refine.py` classifies the
order from the recovered constellation **only when a blind SNR estimate and the
post-equalization quality support it**, deferring otherwise. This is feedback DSP
that runs at capture ingestion (the SDR path), not in the browser.

```bash
.venv-training/bin/python training/order_refine.py   # fit + evaluate, ~3 min
```

Fits the refiner, writes `src/embedding/assets/order-refiner.json`, and reports
decided-accuracy vs defer-rate across SNR/channel. The TS side
(`src/embedding/order-refinement.ts`) consumes the order result and combines it
with the embedding's family call.
