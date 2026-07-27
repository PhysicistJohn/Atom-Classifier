# ComplexConv1d execution benchmark — 2026-07-27

## Decision

Use the one-call `groups=2` formulation through `execution="auto"`:

- CPU: fused for training and inference.
- MPS: fused while autograd is enabled; legacy four-call execution under `no_grad` /
  inference mode.
- Unbenchmarked device types: legacy.

`execution="legacy"` remains an unconditional runtime fallback. Execution mode is plain
module metadata and does not appear in `state_dict`.

## Candidate 1: dense 2×2 block convolution — rejected

The initially proposed real block weight was:

```text
[[Wr, -Wi],
 [Wi,  Wr]]
```

It is mathematically correct, but it changes float32 reduction order by summing across
`2 * in_channels` in one reduction. Across the deployed TransferUNet layer shapes:

- CPU worst forward error: `3.534e-6`
- CPU worst input-gradient error: `3.881e-6`
- MPS worst forward error in representative deployed shapes: `7.525e-6`
- MPS `TransferUNet.forward_embedding` speed: `0.728×` legacy

This fails the required `1e-6` primitive forward/backward contract and is slower on MPS.

## Candidate 2: grouped four-product convolution — accepted

One `groups=2` Conv1d receives `[Xr, Xi]`:

- group 0 emits `Wr*Xr, Wi*Xr`;
- group 1 emits `Wr*Xi, Wi*Xi`;
- the output combines those products with the original add/subtract order.

This removes three convolution launches without changing each convolution reduction.
`conv_re.weight` and `conv_im.weight` remain the only parameters, under their original
keys. Repeated functional weight views accumulate gradients into those original tensors.
Because the functional call bypasses forward hooks attached directly to the two child
Conv1d modules, hook-based external instrumentation should select `execution="legacy"`.
The repository's training and evaluation paths do not register such hooks.

Focused CPU and MPS tests passed with maximum forward/backward differences `<= 1e-6`.
The corrected trained checkpoint loaded with `strict=True`, paired-real export parity
continued to pass, and the U-Net phase-equivariance self-test passed.

## End-to-end timings

Machine: Apple arm64, PyTorch 2.8.0, Python 3.9.6. Model: corrected TransferUNet,
`forward_embedding`, batch 70, two I/Q channels, length 1024. Medians include device
synchronization. CPU used four threads.

| Device | Work | Legacy | Fused grouped | Legacy / fused |
|---|---:|---:|---:|---:|
| CPU | forward | 715.94 ms | 616.09 ms | **1.162×** |
| CPU | forward + backward | 947.09 ms | 765.67 ms | **1.237×** |
| MPS | no-grad forward | 12.61 ms | 14.00 ms | 0.901× |
| MPS | forward + backward | 105.60 ms | 54.56 ms | **1.936×** |

The MPS forward/backward result reproduced in a prior 25-repeat run at `2.047×`.
The no-grad regression is why `auto` retains legacy execution for MPS inference.

## Reproduction

```bash
.venv-training/bin/python -m unittest -v \
  training/zplane_ab/v2_full_variation/test_complex_conv_execution.py \
  training/zplane_ab/v2_full_variation/test_transfer_pipeline.py \
  training/zplane_ab/v2_full_variation/test_paired_real_inference.py

.venv-training/bin/python -u \
  training/zplane_ab/v2_full_variation/benchmark_complex_conv.py \
  --device cpu --threads 4 --warmup 1 --repeats 3 --with-backward

.venv-training/bin/python -u \
  training/zplane_ab/v2_full_variation/benchmark_complex_conv.py \
  --device mps --warmup 10 --repeats 50 --with-backward
```
