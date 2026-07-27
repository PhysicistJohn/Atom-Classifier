"""Evaluator for the ES/RL-trained ZPlaneEmbedding: loads the saved
state_dict, calls evaluate_ab_v2.evaluate_v2 (imported, never modified) for
the exact same closed_overall/closed_clean/closed_impaired + per-class +
fewshot-LOO(K=1,5) protocol used by every other cell in this A/B, and writes
eval-report-zplane_es_{condition}.json in the same report schema -- directly
diffable against ../artifacts/eval-report-zplane_resampled.json (backprop) and
../artifacts/eval-report-cnn_resampled.json (CNN baseline).

Run:
  .venv-training/bin/python training/zplane_ab/v2_full_variation/es_rl/evaluate_es.py --condition resampled
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_ES_DIR = os.path.dirname(os.path.abspath(__file__))
_V2_DIR = os.path.dirname(_ES_DIR)
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR, _ES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from zplane_backbone import ZPlaneEmbedding  # noqa: E402 -- imported, never modified
import train_common_v2 as tcv2  # noqa: E402 -- imported, never modified
from evaluate_ab_v2 import evaluate_v2  # noqa: E402 -- imported, never modified

ARTIFACTS = os.path.join(_ES_DIR, "artifacts")


def load_es_net(manifest, state_dict_path):
    cfg = manifest["config"]
    net = ZPlaneEmbedding(width=cfg["width"], layers=cfg["layers"], sections=cfg["sections"],
                           warm_radius=cfg["warm_radius"])
    net.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["resampled", "native"], required=True)
    args = ap.parse_args()

    dev = torch.device("cpu")
    name = f"zplane_es_{args.condition}"
    out_dir = os.path.join(ARTIFACTS, name)
    manifest_path = os.path.join(out_dir, "manifest.json")
    state_path = os.path.join(out_dir, "state_dict.pt")
    if not (os.path.exists(manifest_path) and os.path.exists(state_path)):
        print(f"[{name}] no trained artifacts at {out_dir} -- run train_es.py first")
        return None

    manifest = json.load(open(manifest_path))
    net = load_es_net(manifest, state_path)
    data = tcv2.prepare_data_v2(condition=args.condition)

    report, protos = evaluate_v2(net, dev, data)
    report["backbone"] = "zplane_es"
    report["condition"] = args.condition
    report["algorithm"] = manifest.get("algorithm")
    report["input_length"] = manifest["input_length"]
    report["param_count"] = manifest["param_count"]
    report["wall_clock_train_s"] = manifest["wall_clock_train_s"]
    report["generations"] = manifest["generations"]
    report["popsize"] = manifest["popsize"]
    report["batch"] = manifest["batch"]
    report["infer_latency_ms_per_1k_cpu"] = manifest["infer_latency_ms_per_1k_cpu"]
    report["training_device"] = manifest.get("training_device")
    report["warm_start_ce"] = manifest["warm_start_ce"]
    report["warm_start_acc"] = manifest["warm_start_acc"]
    report["best_val_ce"] = manifest["best_val_ce"]
    report["classes"] = data["classes"]

    np.save(os.path.join(out_dir, "prototypes.npy"), protos)
    out_path = os.path.join(ARTIFACTS, f"eval-report-{name}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[{name}] closed_overall={report['closed_overall']:.3f}  "
          f"closed_clean={report['closed_clean']:.3f}  closed_impaired={report['closed_impaired']:.3f}  "
          f"loo_k1_overall={report['fewshot'].get('loo_k1_overall_mean')}  "
          f"loo_k5_overall={report['fewshot'].get('loo_k5_overall_mean')}")
    print(f"  wrote {out_path}")
    return report


if __name__ == "__main__":
    main()
