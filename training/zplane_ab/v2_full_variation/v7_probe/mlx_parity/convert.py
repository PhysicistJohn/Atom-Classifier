"""Bidirectional checkpoint converter: torch v7 trainer .pt <-> MLX safetensors.

Torch schema (v7_trainer.py):
    {"state_dict": OrderedDict[str, Tensor], "log_scale": float,
     "config": dict, "episode": int (optional -- absent in the final save)}

MLX artifact: safetensors written by mx.save_safetensors with metadata
    {"log_scale": str(float), "config": json string, "episode": str (optional)}

Layout mapping (converter owns ALL permutes, per docs/mlx-port-plan.md):
    Conv2d weight  torch OIHW (O, I, kH, kW)  ->  MLX (O, kH, kW, I)
    Linear / GroupNorm weights and all biases: 1:1, untouched.

No import of any MLX model module -- key names and permutes are derived from
the frozen v7_trainer.py architecture, hardcoded below and asserted 1:1
against every checkpoint touched (no silent key drops in either direction).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Exact state_dict key inventory of V7Net (v7_trainer.py, frozen reference).
# Blocks stem/d1..d5 (Block) and u1..u3 (UpBlock) each contribute
# {m}.conv.{weight,bias} + {m}.norm.{weight,bias}.
# ---------------------------------------------------------------------------
_BLOCKS = ("stem", "d1", "d2", "d3", "d4", "d5", "u1", "u2", "u3")

CONV_WEIGHT_KEYS = frozenset(
    [f"{m}.conv.weight" for m in _BLOCKS] + ["out.weight"]
)

EXPECTED_KEYS = frozenset(
    [f"{m}.{part}.{p}" for m in _BLOCKS
     for part in ("conv", "norm") for p in ("weight", "bias")]
    + ["embed.weight", "embed.bias",
       "conf.0.weight", "conf.0.bias",
       "conf.2.weight", "conf.2.bias",
       "out.weight", "out.bias"]
)
assert len(EXPECTED_KEYS) == 44  # 9 blocks x 4 + embed 2 + conf 4 + out 2


def _assert_keys(keys, where: str) -> None:
    keys = set(keys)
    if keys != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - keys)
        extra = sorted(keys - EXPECTED_KEYS)
        raise AssertionError(
            f"{where}: state_dict key set does not match V7Net 1:1.\n"
            f"  missing ({len(missing)}): {missing}\n"
            f"  extra   ({len(extra)}): {extra}"
        )


# ---------------------------------------------------------------------------
# torch -> mlx
# ---------------------------------------------------------------------------
def torch2mlx(in_path: str, out_path: str) -> None:
    import torch
    import mlx.core as mx

    ck = torch.load(in_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    _assert_keys(sd.keys(), f"torch2mlx({in_path})")

    arrays: dict[str, "mx.array"] = {}
    for k, t in sd.items():
        a = t.detach().cpu().numpy()
        if k in CONV_WEIGHT_KEYS:
            assert a.ndim == 4, f"{k}: expected 4-D conv weight, got {a.shape}"
            a = np.ascontiguousarray(a.transpose(0, 2, 3, 1))  # OIHW->OHWI
        arrays[k] = mx.array(a)
    _assert_keys(arrays.keys(), f"torch2mlx({in_path}) post-map")

    metadata = {
        "log_scale": str(float(ck["log_scale"])),
        "config": json.dumps(ck["config"]),
    }
    if "episode" in ck:
        metadata["episode"] = str(ck["episode"])

    mx.save_safetensors(out_path, arrays, metadata=metadata)
    print(f"torch2mlx: wrote {out_path} ({len(arrays)} tensors, "
          f"metadata keys {sorted(metadata)})")


# ---------------------------------------------------------------------------
# mlx -> torch
# ---------------------------------------------------------------------------
def mlx2torch(in_path: str, out_path: str) -> None:
    import torch
    import mlx.core as mx

    arrays, metadata = mx.load(in_path, return_metadata=True)
    _assert_keys(arrays.keys(), f"mlx2torch({in_path})")

    sd = {}
    for k in sorted(arrays):
        a = np.array(arrays[k])
        if k in CONV_WEIGHT_KEYS:
            assert a.ndim == 4, f"{k}: expected 4-D conv weight, got {a.shape}"
            a = np.ascontiguousarray(a.transpose(0, 3, 1, 2))  # OHWI->OIHW
        sd[k] = torch.from_numpy(a)
    _assert_keys(sd.keys(), f"mlx2torch({in_path}) post-map")

    ck = {
        "state_dict": sd,
        "log_scale": float(metadata["log_scale"]),
        "config": json.loads(metadata["config"]),
    }
    if "episode" in metadata:
        ck["episode"] = int(metadata["episode"])

    torch.save(ck, out_path)
    print(f"mlx2torch: wrote {out_path} ({len(sd)} tensors)")


# ---------------------------------------------------------------------------
# verify: torch -> mlx -> torch, bitwise
# ---------------------------------------------------------------------------
def verify(in_path: str, mlx_out: str) -> None:
    import torch

    torch2mlx(in_path, mlx_out)

    fd, tmp_pt = tempfile.mkstemp(suffix=".pt",
                                  dir=os.path.dirname(os.path.abspath(mlx_out)))
    os.close(fd)
    try:
        mlx2torch(mlx_out, tmp_pt)

        orig = torch.load(in_path, map_location="cpu", weights_only=False)
        back = torch.load(tmp_pt, map_location="cpu", weights_only=False)

        _assert_keys(orig["state_dict"].keys(), "verify(original)")
        _assert_keys(back["state_dict"].keys(), "verify(roundtrip)")

        n_ok = 0
        for k in sorted(orig["state_dict"]):
            a = orig["state_dict"][k].detach().cpu().numpy()
            b = back["state_dict"][k].detach().cpu().numpy()
            assert a.dtype == b.dtype, f"{k}: dtype {a.dtype} != {b.dtype}"
            assert a.shape == b.shape, f"{k}: shape {a.shape} != {b.shape}"
            assert np.array_equal(a, b), f"{k}: NOT bitwise equal"
            n_ok += 1

        assert float(orig["log_scale"]) == float(back["log_scale"]), (
            f"log_scale mismatch: {orig['log_scale']!r} vs {back['log_scale']!r}")
        assert orig["config"] == back["config"], "config mismatch"
        if "episode" in orig:
            assert "episode" in back and int(orig["episode"]) == int(back["episode"]), (
                f"episode mismatch: {orig.get('episode')!r} vs {back.get('episode')!r}")
        else:
            assert "episode" not in back, "spurious episode key in roundtrip"

        print(f"VERIFY PASS: {n_ok}/{len(EXPECTED_KEYS)} tensors bitwise equal; "
              f"log_scale={float(orig['log_scale'])!r}, "
              f"episode={orig.get('episode', '<absent>')}, config preserved.")
    finally:
        if os.path.exists(tmp_pt):
            os.unlink(tmp_pt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("torch2mlx", "mlx2torch", "verify"))
    ap.add_argument("infile")
    ap.add_argument("outfile")
    cli = ap.parse_args()

    if cli.mode == "torch2mlx":
        torch2mlx(cli.infile, cli.outfile)
    elif cli.mode == "mlx2torch":
        mlx2torch(cli.infile, cli.outfile)
    else:
        verify(cli.infile, cli.outfile)


if __name__ == "__main__":
    main()
