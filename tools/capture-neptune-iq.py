#!/usr/bin/env python
"""Capture complex I/Q from the NeptuneSDR (AD9361) into a corpus-shaped file.

Configures the receiver over libiio (network backend), lets AGC settle,
discards the stream head, and writes exactly --seconds of complex64 baseband
plus a JSON sidecar. Defaults produce 20 ms at 20 Msps = 400,000 samples,
one production-corpus row, the DACS runtime contract input.

Usage:
  .venv-training/bin/python tools/capture-neptune-iq.py \
    --center-mhz 98.7 --name fm-broadcast --out-dir training/artifacts/ota-20260802
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ADC_FULL_SCALE = 2047.0  # signed 12-bit in 16-bit words
BYTES_PER_SAMPLE = 4     # int16 I + int16 Q


def find_tool(name: str) -> str:
    p = shutil.which(name) or str(Path.home() / ".local/bin" / name)
    if not Path(p).exists():
        raise SystemExit(f"{name} not found on PATH or in ~/.local/bin")
    return p


def set_attr(iio_attr: str, uri: str, device: str, channel: str,
             attribute: str, value) -> None:
    r = subprocess.run(
        [iio_attr, "-u", uri, "-c", device, channel, attribute, str(value)],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"set {device}/{channel}/{attribute}={value}: "
                         f"{r.stderr.strip() or r.stdout.strip()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ip:10.0.0.250")
    ap.add_argument("--center-mhz", type=float, required=True)
    ap.add_argument("--rate", type=float, default=20e6)
    ap.add_argument("--seconds", type=float, default=0.02)
    ap.add_argument("--settle-seconds", type=float, default=0.75,
                    help="AGC/PLL settle time after retune before capture")
    ap.add_argument("--discard-seconds", type=float, default=0.05,
                    help="stream head to discard (buffered stale samples)")
    ap.add_argument("--gain-mode", default="slow_attack",
                    choices=("slow_attack", "fast_attack", "manual"))
    ap.add_argument("--rf-bandwidth-hz", type=float, default=None,
                    help="override analog RF bandwidth (default: min(rate, "
                         "56 MHz)); narrow this to isolate one channel")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    iio_attr = find_tool("iio_attr")
    iio_readdev = find_tool("iio_readdev")
    center_hz = int(round(args.center_mhz * 1e6))
    rate_hz = int(round(args.rate))
    bw_hz = int(args.rf_bandwidth_hz) if args.rf_bandwidth_hz \
        else min(rate_hz, 56_000_000)

    print(f"tune {center_hz/1e6:.3f} MHz @ {rate_hz/1e6:.3f} Msps "
          f"(bw {bw_hz/1e6:.1f} MHz, {args.gain_mode})", flush=True)
    set_attr(iio_attr, args.uri, "ad9361-phy", "altvoltage0", "frequency",
             center_hz)
    set_attr(iio_attr, args.uri, "ad9361-phy", "voltage0",
             "sampling_frequency", rate_hz)
    set_attr(iio_attr, args.uri, "ad9361-phy", "voltage0", "rf_bandwidth",
             bw_hz)
    set_attr(iio_attr, args.uri, "ad9361-phy", "voltage0",
             "gain_control_mode", args.gain_mode)
    time.sleep(args.settle_seconds)

    n_keep = int(round(args.seconds * rate_hz))
    n_discard = int(round(args.discard_seconds * rate_hz))
    n_total = n_keep + n_discard
    want_bytes = n_total * BYTES_PER_SAMPLE

    proc = subprocess.Popen(
        [iio_readdev, "-u", args.uri, "-b", "65536",
         "cf-ad9361-lpc", "voltage0", "voltage1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    buf = bytearray()
    t0 = time.time()
    while len(buf) < want_bytes:
        chunk = proc.stdout.read(want_bytes - len(buf))
        if not chunk:
            err = proc.stderr.read().decode(errors="replace")
            raise SystemExit(f"stream ended early ({len(buf)} bytes): {err}")
        buf += chunk
        if time.time() - t0 > 30:
            raise SystemExit("capture timed out")
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)

    raw = np.frombuffer(bytes(buf[: want_bytes]), dtype="<i2")
    iq = (raw[0::2].astype(np.float32)
          + 1j * raw[1::2].astype(np.float32)) / ADC_FULL_SCALE
    iq = iq[n_discard:n_discard + n_keep].astype(np.complex64)

    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    peak = float(np.max(np.abs(iq)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / f"{args.name}.iq.npy"
    np.save(npy, iq)
    sidecar = {
        "name": args.name,
        "uri": args.uri,
        "centerHz": center_hz,
        "sampleRateHz": rate_hz,
        "rfBandwidthHz": bw_hz,
        "gainMode": args.gain_mode,
        "samples": int(len(iq)),
        "seconds": len(iq) / rate_hz,
        "rms": rms,
        "peak": peak,
        "clipped": bool(peak >= 0.999),
        "capturedAt": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / f"{args.name}.json").write_text(json.dumps(sidecar, indent=1))
    print(f"wrote {npy}  n={len(iq)}  rms={rms:.4f}  peak={peak:.4f}"
          + ("  ⚠ CLIPPED" if sidecar["clipped"] else ""), flush=True)


if __name__ == "__main__":
    main()
