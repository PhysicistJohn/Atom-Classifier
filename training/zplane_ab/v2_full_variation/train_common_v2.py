"""v2 data-preparation entry point: builds train/enroll/val pools from the
full-variation split (full_split.load_and_split_v2 / build_pool_v2), under
whichever preprocessing CONDITION is requested ("resampled" == training/
preprocess.py, unmodified; "native" == this directory's native_preprocess.py).

Everything else (the episodic training loop itself, timing probe, inference
latency measurement, parameter counting) is reused UNCHANGED from
../train_common.py -- only the data-preparation step differs for v2, because
only the split and preprocessing are new; the training mechanics are not.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import preprocess as pp  # noqa: E402 -- imported, never modified
import native_preprocess as npp  # noqa: E402
from full_split import load_and_split_v2, build_pool_v2  # noqa: E402
import dataset as ds  # noqa: E402
from train_common import train, timing_probe, infer_latency_ms_per_1k, count_params  # noqa: E402,F401 -- reused unchanged

CONDITIONS = {"resampled": pp, "native": npp}


def prepare_data_v2(condition: str):
    module = CONDITIONS[condition]
    iq, y, impaired, classes, tr_idx, en_idx, va_idx, rng = load_and_split_v2()
    n_classes = len(classes)
    print(f"[v2:{condition}] corpus {iq.shape} | impaired {int(impaired.sum())}/{len(impaired)} | classes {classes}")
    print(f"[v2:{condition}] split sizes: train {len(tr_idx)}  enroll {len(en_idx)}  val {len(va_idx)}")
    print("building pools ...")

    xtr, ftr, ytr, imp_tr = build_pool_v2(iq, y, impaired, tr_idx, rng, module, scale_jitter=0.08)
    fmean = ftr.mean(0)
    fstd = ftr.std(0) + 1e-6
    ftr = (ftr - fmean) / fstd
    idx_by_class = ds.class_indices(ytr, n_classes)

    xen, fen, yen, imp_en = build_pool_v2(iq, y, impaired, en_idx, rng, module, scale_jitter=0.0)
    fen = (fen - fmean) / fstd
    xva, fva, yva, imp_va = build_pool_v2(iq, y, impaired, va_idx, rng, module, scale_jitter=0.0)
    fva = (fva - fmean) / fstd

    print(f"  train {xtr.shape} (impaired {int(imp_tr.sum())}/{len(imp_tr)})"
          f"  enroll {xen.shape} (impaired {int(imp_en.sum())}/{len(imp_en)})"
          f"  val {xva.shape} (impaired {int(imp_va.sum())}/{len(imp_va)})")

    # The raw corpus (~1.8 GB as complex64) is only needed while the pools are being
    # built. Returning it pinned it for the entire training run and, together with the
    # clean-pair array, drove the machine deep into swap (43.9 of 45 GB) -- fatal for a
    # multi-day unattended campaign. Nothing downstream reads data["iq"].
    del iq
    return dict(
        classes=classes, n_classes=n_classes, condition=condition,
        input_length=int(xtr.shape[-1]),
        y=y, impaired=impaired,
        tr_idx=tr_idx, en_idx=en_idx, va_idx=va_idx, rng=rng,
        xtr=xtr, ftr=ftr, ytr=ytr, idx_by_class=idx_by_class, imp_tr=imp_tr,
        xen=xen, fen=fen, yen=yen, imp_en=imp_en,
        xva=xva, fva=fva, yva=yva, imp_va=imp_va,
        fmean=fmean, fstd=fstd,
    )
