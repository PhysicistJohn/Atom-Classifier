"""Measure per-profile row diversity of a stage-2 long-dwell corpus.

Four measurements, deliberately reported side by side because they answer
different questions and the cheap one over-reports:

  distinctSha256      bytes on disk. A global phase rotation changes every
                      byte, so this counts DISTINCT ROWS, never distinct
                      content. It is the number the 2026-07-31 hash audit
                      quoted, kept here for comparability.

  distinctOnsets      distinct burst-onset patterns (rising edges of a 1 us
                      smoothed power envelope, quantized to 1 us). Invariant
                      to carrier phase, sensitive to time origin and to burst
                      scheduling. 1 is correct and expected for a
                      continuous-emission profile; for a bursty one it is the
                      number of distinct TDMA/PPDU/slot alignments realized.

  meanAbsRho          mean over row pairs of |<a,b>| / (|a| |b|). This is the
                      phase-invariant similarity the memorization audit used
                      to show that FM's 256 bit-distinct rows were one
                      waveform times a scalar. 1.0 means the rows are the same
                      waveform up to global phase.

  phaseInvClusters    single-link clusters at |rho| >= RHO_THRESHOLD. THE
                      honest realization count: 1 means the profile
                      contributes one waveform no matter how many hashes
                      differ.

  relRmsDiff / phaseInvRelRmsDiff
                      mean pairwise |a-b| and min-over-phase |a - e^(jt) b|,
                      each normalized by the rows' RMS energy. The gap between
                      the two is exactly the part of the difference that is
                      global carrier phase.

Usage:
  CORPUS_DIR=<dir> [ROWS=8] [ROW_BASE=0] [OUT_JSON=<path>] \
    .venv-training/bin/python tools/measure-corpus-diversity.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

CORPUS = Path(os.environ["CORPUS_DIR"])
ROWS = int(os.environ.get("ROWS", 8))
ROW_BASE = int(os.environ.get("ROW_BASE", 0))
RHO_THRESHOLD = float(os.environ.get("RHO_THRESHOLD", 0.999))
# 1 us at the 20 Msps common rate: the smoothing window for envelope onset
# detection and the quantization of the reported onset coordinates.
SMOOTH = int(os.environ.get("SMOOTH_SAMPLES", 20))
ONSET_FRACTION = float(os.environ.get("ONSET_FRACTION", 0.1))


def onset_pattern(row: np.ndarray) -> tuple[int, ...]:
    """Rising edges of the smoothed power envelope, quantized to SMOOTH."""
    power = (row.real.astype(np.float64) ** 2 + row.imag.astype(np.float64) ** 2)
    if power.max() <= 0:
        return ()
    kernel = np.ones(SMOOTH) / SMOOTH
    envelope = np.convolve(power, kernel, mode="same")
    active = envelope > ONSET_FRACTION * envelope.max()
    edges = np.flatnonzero(np.diff(active.astype(np.int8)) == 1) + 1
    if active[0]:
        edges = np.concatenate(([0], edges))
    return tuple((edges // SMOOTH).tolist())


def cluster(abs_rho: np.ndarray, threshold: float) -> int:
    """Single-link cluster count of the |rho| similarity matrix."""
    count = abs_rho.shape[0]
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left in range(count):
        for right in range(left + 1, count):
            if abs_rho[left, right] >= threshold:
                a, b = find(left), find(right)
                if a != b:
                    parent[a] = b
    return len({find(node) for node in range(count)})


def measure(rows: np.ndarray) -> dict:
    count = rows.shape[0]
    wide = rows.astype(np.complex128)
    norms = np.sqrt(np.einsum("ij,ij->i", wide.conj(), wide).real)
    gram = wide.conj() @ wide.T
    safe = np.where(norms > 0, norms, 1.0)
    abs_rho = np.abs(gram) / np.outer(safe, safe)
    np.fill_diagonal(abs_rho, 1.0)

    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    rel_raw, rel_phase_inv = [], []
    for i, j in pairs:
        scale = np.sqrt(0.5 * (norms[i] ** 2 + norms[j] ** 2))
        if scale == 0:
            rel_raw.append(0.0)
            rel_phase_inv.append(0.0)
            continue
        rel_raw.append(float(np.linalg.norm(wide[i] - wide[j]) / scale))
        # min over theta of |a - e^(j theta) b| = sqrt(|a|^2 + |b|^2 - 2|<a,b>|)
        residual = max(norms[i] ** 2 + norms[j] ** 2 - 2 * abs(gram[i, j]), 0.0)
        rel_phase_inv.append(float(np.sqrt(residual) / scale))

    off = [abs_rho[i, j] for i, j in pairs]
    return {
        "rows": count,
        "distinctSha256": len({hashlib.sha256(
            np.ascontiguousarray(row).tobytes()).hexdigest() for row in rows}),
        "distinctOnsetPatterns": len({onset_pattern(row) for row in rows}),
        "medianOnsetCount": float(np.median(
            [len(onset_pattern(row)) for row in rows])),
        "phaseInvClusters": cluster(abs_rho, RHO_THRESHOLD),
        "meanAbsRho": float(np.mean(off)) if off else 1.0,
        "maxAbsRho": float(np.max(off)) if off else 1.0,
        "relRmsDiff": float(np.mean(rel_raw)) if rel_raw else 0.0,
        "phaseInvRelRmsDiff": float(np.mean(rel_phase_inv)) if rel_phase_inv else 0.0,
        "zeroRows": int(np.sum(np.all(rows == 0, axis=1))),
        "medianRowRms": float(np.median(np.sqrt(
            np.mean(np.abs(rows.astype(np.complex128)) ** 2, axis=1)))),
    }


def main() -> None:
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    clean = np.load(CORPUS / "clean.npy", mmap_mode="r")
    order: list[str] = []
    index_of: dict[str, list[int]] = {}
    for entry in manifest["rows"]:
        profile = entry["profile"]
        if profile not in index_of:
            index_of[profile] = []
            order.append(profile)
        index_of[profile].append(entry["row"])

    diversity = manifest.get("diversity", {})
    report = {
        "corpusDir": str(CORPUS),
        "rowsMeasuredPerProfile": ROWS,
        "rowBase": ROW_BASE,
        "rhoThreshold": RHO_THRESHOLD,
        "profiles": {},
    }
    for profile in order:
        indices = index_of[profile][ROW_BASE:ROW_BASE + ROWS]
        rows = np.asarray(clean[indices])
        entry = measure(rows)
        entry["contentClass"] = diversity.get(profile, {}).get("contentClass")
        entry["contentRealizationsAvailable"] = (
            diversity.get(profile, {}).get("contentRealizationsAvailable"))
        report["profiles"][profile] = entry
        print(f"{profile:42s} sha={entry['distinctSha256']:>3d} "
              f"onset={entry['distinctOnsetPatterns']:>3d} "
              f"clusters={entry['phaseInvClusters']:>3d} "
              f"rho={entry['meanAbsRho']:.5f} "
              f"relRms={entry['relRmsDiff']:.4f} "
              f"piRelRms={entry['phaseInvRelRmsDiff']:.4f}", flush=True)

    if os.environ.get("OUT_JSON"):
        Path(os.environ["OUT_JSON"]).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
