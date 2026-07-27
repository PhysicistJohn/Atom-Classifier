"""Benchmark legacy versus fused-grouped ComplexConv1d execution.

This benchmark intentionally measures ``TransferUNet.forward_embedding`` at the episodic
training shape (batch 70, input length 1024).  It never loads a corpus or writes model
assets.

Examples::

    .venv-training/bin/python -u \
      training/zplane_ab/v2_full_variation/benchmark_complex_conv.py \
      --device cpu --warmup 2 --repeats 5

    .venv-training/bin/python -u \
      training/zplane_ab/v2_full_variation/benchmark_complex_conv.py \
      --device mps --warmup 10 --repeats 30 --with-backward
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from complex_multiscale_backbone import set_complex_conv_execution  # noqa: E402
from unet_transfer import TransferUNet  # noqa: E402


ARCHITECTURE = {
    "base": 8,
    "depth": 4,
    "taps": 7,
    "embed_dim": 32,
    "n_features": 12,
    "n_profiles": 37,
    "hidden": 128,
    "magnorm": True,
    "modrelu_init": -2.0,
    "skip_dropout": 0.5,
    "feat_dropout": 0.5,
    "antialias": False,
    "head_pool": "meanstd",
    "residual_recon": True,
}


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(seconds: list[float]) -> dict[str, float]:
    ordered = sorted(seconds)
    return {
        "median_ms": statistics.median(ordered) * 1000.0,
        "mean_ms": statistics.mean(ordered) * 1000.0,
        "min_ms": min(ordered) * 1000.0,
        "max_ms": max(ordered) * 1000.0,
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))] * 1000.0,
    }


def _time_forward(
    net: TransferUNet,
    iq: torch.Tensor,
    features: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    net.eval()
    with torch.no_grad():
        for _ in range(warmup):
            net.forward_embedding(iq, features)
        _synchronize(device)
        rows = []
        for _ in range(repeats):
            start = time.perf_counter()
            net.forward_embedding(iq, features)
            _synchronize(device)
            rows.append(time.perf_counter() - start)
    return rows


def _time_forward_backward(
    net: TransferUNet,
    iq: torch.Tensor,
    features: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    # eval() removes dropout variance while retaining the exact encoder autograd graph.
    net.eval()

    def step() -> None:
        net.zero_grad(set_to_none=True)
        embedding = net.forward_embedding(iq, features)
        (embedding * target).mean().backward()

    for _ in range(warmup):
        step()
    _synchronize(device)
    rows = []
    for _ in range(repeats):
        start = time.perf_counter()
        step()
        _synchronize(device)
        rows.append(time.perf_counter() - start)
    return rows


def benchmark(args: argparse.Namespace) -> dict:
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "mps":
        torch.mps.manual_seed(args.seed)
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    net = TransferUNet(**ARCHITECTURE).to(device)
    iq = torch.randn(args.batch, 2, args.length, device=device)
    features = torch.randn(args.batch, 12, device=device)
    target = torch.randn(args.batch, 32, device=device)

    set_complex_conv_execution(net, "legacy")
    net.eval()
    with torch.no_grad():
        expected = net.forward_embedding(iq, features)
    set_complex_conv_execution(net, "fused_grouped")
    with torch.no_grad():
        actual = net.forward_embedding(iq, features)
    _synchronize(device)
    parity = float((actual - expected).abs().max().cpu())
    if parity > 1e-6:
        raise AssertionError(f"embedding parity {parity:.3e} exceeds 1e-6")

    timings = {}
    for execution in ("legacy", "fused_grouped"):
        set_complex_conv_execution(net, execution)
        timings[execution] = {
            "forward": _summary(
                _time_forward(
                    net,
                    iq,
                    features,
                    device,
                    args.warmup,
                    args.repeats,
                )
            )
        }
        if args.with_backward:
            timings[execution]["forward_backward"] = _summary(
                _time_forward_backward(
                    net,
                    iq,
                    features,
                    target,
                    device,
                    args.warmup,
                    args.repeats,
                )
            )

    speedups = {
        name: (
            timings["legacy"][name]["median_ms"]
            / timings["fused_grouped"][name]["median_ms"]
        )
        for name in timings["legacy"]
    }
    return {
        "device": str(device),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "seed": args.seed,
        "shape": {
            "batch": args.batch,
            "channels": 2,
            "length": args.length,
            "features": 12,
        },
        "architecture": ARCHITECTURE,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "embedding_max_abs_error": parity,
        "timings": timings,
        "legacy_over_fused_speedup": speedups,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--batch", type=int, default=70)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--with-backward", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    for name in ("batch", "length", "warmup", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.threads is not None and args.threads <= 0:
        parser.error("--threads must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = benchmark(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
