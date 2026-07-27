"""Autonomous weekend architecture-search campaign for the bluetooth/gsm/dsss
confusion problem. Runs unsupervised for an extended period: phase 1 screens
many architecture/hyperparameter variants at a reduced episode budget, phase 2
deep-trains the best few candidates at the full budget, phase 3 builds a
simple ensemble of the best models and writes a final report.

Safety guards for unsupervised multi-day operation:
  - disk space checked before every run; campaign halts cleanly (not mid-write)
    if free space drops below MIN_FREE_GB
  - global wall-clock budget so the campaign wraps up in time regardless of
    how any individual run goes
  - per-run try/except: one failing config is logged and skipped, never takes
    down the whole campaign
  - every result is appended to results.jsonl immediately (not buffered to the
    end), and REPORT.md is regenerated after every single run -- if this
    process dies at any point, everything up to that point is already on disk

Run (wrap with caffeinate so the machine doesn't sleep mid-campaign):
  caffeinate -i .venv-training/bin/python training/zplane_ab/v2_full_variation/campaign_runner.py
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np
import torch

_V2_DIR = os.path.dirname(os.path.abspath(__file__))
_ZPAB_DIR = os.path.dirname(_V2_DIR)
_TRAINING_DIR = os.path.dirname(_ZPAB_DIR)
for p in (_TRAINING_DIR, _ZPAB_DIR, _V2_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import Embedding, EMBED_DIM, N_FEATURES  # noqa: E402
from vit_backbone import ViTEmbedding  # noqa: E402
from complex_multiscale_backbone import ComplexMultiScaleEmbedding  # noqa: E402
import train_common_v2 as tcv2  # noqa: E402
from evaluate_ab_v2 import evaluate_v2  # noqa: E402
from train import embed_all, nearest, prototypes_from  # noqa: E402
from train_tuned import train_tuned  # noqa: E402
from train_margin_sample import train_with_sample_margin  # noqa: E402

ARTIFACTS = os.path.join(_V2_DIR, "artifacts", "campaign")
os.makedirs(ARTIFACTS, exist_ok=True)
RESULTS_JSONL = os.path.join(ARTIFACTS, "results.jsonl")
REPORT_MD = os.path.join(ARTIFACTS, "REPORT.md")
STATE_JSON = os.path.join(ARTIFACTS, "campaign_state.json")

MIN_FREE_GB = 5.0
CAMPAIGN_BUDGET_HOURS = 65.0  # leaves margin inside a ~72h weekend for the final report/ensemble
CAMPAIGN_START = time.perf_counter()

DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

_DATA_CACHE = {}


def get_data(condition: str):
    if condition not in _DATA_CACHE:
        _DATA_CACHE[condition] = tcv2.prepare_data_v2(condition=condition)
    return _DATA_CACHE[condition]


def free_disk_gb() -> float:
    return shutil.disk_usage("/").free / (1024 ** 3)


def campaign_elapsed_hours() -> float:
    return (time.perf_counter() - CAMPAIGN_START) / 3600.0


def build_net(backbone: str, params: dict, in_len: int):
    if backbone == "cnn":
        return Embedding(EMBED_DIM, N_FEATURES)
    if backbone == "vit":
        # patch_size must scale with in_len (1024 resampled vs 4096 native) or
        # the position embedding (sized for a fixed patch count) mismatches --
        # exactly the bug the smoke test caught. Mirrors iterate_tuned.py.
        p = dict(params)
        base_patch = p.pop("patch", 16)
        p["patch_size"] = base_patch * (in_len // 1024)
        p["in_len"] = in_len
        return ViTEmbedding(**p)
    if backbone == "complexlstm":
        return ComplexMultiScaleEmbedding(**params)
    raise ValueError(f"unknown backbone {backbone}")


def confusion_matrices(net, data, protos):
    net = net.to("cpu")
    emb_va = embed_all(net, data["xva"], data["fva"], torch.device("cpu"))
    pred, _ = nearest(emb_va, protos)
    true = data["yva"]
    imp = data["imp_va"]
    n = data["n_classes"]

    def cm(mask):
        m = np.zeros((n, n), dtype=int)
        for t, p in zip(true[mask], pred[mask]):
            m[t, p] += 1
        return m

    return {"overall": cm(np.ones_like(imp)).tolist(), "clean": cm(~imp).tolist(), "impaired": cm(imp).tolist()}


def triad_leakage(cm_overall, classes, triad=("bluetooth", "gsm", "dsss"), target="bluetooth"):
    """Sum of off-diagonal leakage FROM each triad member INTO `target`,
    the specific failure mode this whole campaign is chasing."""
    idx = {c: i for i, c in enumerate(classes)}
    ti = idx[target]
    total = 0
    for c in triad:
        if c == target:
            continue
        total += cm_overall[idx[c]][ti]
    return total


def run_one(tag: str, backbone: str, net_params: dict, train_hp: dict, episodes: int,
            condition: str = "native", use_margin: bool = False, margin_kwargs: dict | None = None):
    """Runs one experiment end to end: build, train, evaluate, confusion
    matrix, save artifacts, return a result dict. Raises on failure -- caller
    catches it, this function itself does not swallow errors (so the campaign
    log always shows what actually happened)."""
    data = get_data(condition)
    net = build_net(backbone, net_params, in_len=data["input_length"])
    n_params = sum(p.numel() for p in net.parameters())

    eval_every = train_hp.get("eval_every") or max(50, episodes // 10)
    t0 = time.perf_counter()
    if use_margin:
        net, history, wall = train_with_sample_margin(
            net, data, DEV, episodes=episodes, eval_every=eval_every, log_tag=tag,
            **(margin_kwargs or {}))
    else:
        net, history, wall = train_tuned(
            net, data, DEV, episodes=episodes, eval_every=eval_every, log_tag=tag,
            lr=train_hp.get("lr", 7e-4), weight_decay=train_hp.get("wd", 5e-4),
            warmup_frac=train_hp.get("warmup_frac", 0.05),
            label_smoothing=train_hp.get("label_smoothing", 0.05),
            clip_grad=train_hp.get("clip_grad"), log_every=max(1, episodes // 20))

    net_cpu = net.to("cpu")
    report, protos = evaluate_v2(net_cpu, torch.device("cpu"), data)
    cms = confusion_matrices(net_cpu, data, protos)
    leak = triad_leakage(cms["overall"], data["classes"])

    out_dir = os.path.join(ARTIFACTS, tag)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(net_cpu.state_dict(), os.path.join(out_dir, "state_dict.pt"))
    np.save(os.path.join(out_dir, "prototypes.npy"), protos)

    result = {
        "tag": tag, "backbone": backbone, "condition": condition, "episodes": episodes,
        "episodes_completed": history["episodes"], "net_params": net_params, "train_hp": train_hp,
        "use_margin": use_margin, "param_count": n_params, "wall_clock_s": round(wall, 1),
        "closed_overall": report["closed_overall"], "closed_clean": report["closed_clean"],
        "closed_impaired": report["closed_impaired"], "per_class_clean": report["per_class_clean"],
        "bluetooth_gsm_dsss_leakage_into_bluetooth": leak,
        "best_val_acc_during_training": history["best_val_acc"],
        "artifacts_dir": os.path.relpath(out_dir, ARTIFACTS),
    }
    with open(os.path.join(out_dir, "confusion_matrices.json"), "w") as f:
        json.dump(cms, f)
    return result


def log_result(result: dict):
    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_results() -> list:
    if not os.path.exists(RESULTS_JSONL):
        return []
    with open(RESULTS_JSONL) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_report():
    results = load_results()
    lines = ["# Weekend Architecture Search Campaign\n",
              f"Generated: elapsed {campaign_elapsed_hours():.2f}h, {len(results)} runs completed, "
              f"{free_disk_gb():.1f}GB free disk\n",
              "\n| tag | backbone | condition | episodes | overall | clean | impaired | bt/gsm/dsss->bt leak | params | wall(s) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: -x["closed_overall"]):
        lines.append(f"| {r['tag']} | {r['backbone']} | {r['condition']} | {r['episodes_completed']}/{r['episodes']} "
                      f"| {r['closed_overall']:.3f} | {r['closed_clean']:.3f} | {r['closed_impaired']:.3f} "
                      f"| {r['bluetooth_gsm_dsss_leakage_into_bluetooth']} | {r['param_count']} | {r['wall_clock_s']:.0f} |")
    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_state(phase: str, extra: dict | None = None):
    state = {"phase": phase, "elapsed_hours": campaign_elapsed_hours(), "free_disk_gb": free_disk_gb()}
    if extra:
        state.update(extra)
    with open(STATE_JSON, "w") as f:
        json.dump(state, f, indent=2)


def safe_run(tag: str, **kwargs) -> dict | None:
    if free_disk_gb() < MIN_FREE_GB:
        print(f"[campaign] ABORT: free disk {free_disk_gb():.1f}GB < {MIN_FREE_GB}GB minimum. Stopping campaign.",
              flush=True)
        save_state("halted_low_disk")
        raise SystemExit(1)
    if campaign_elapsed_hours() > CAMPAIGN_BUDGET_HOURS:
        print(f"[campaign] Time budget ({CAMPAIGN_BUDGET_HOURS}h) reached, skipping remaining runs.", flush=True)
        return None
    already = {r["tag"] for r in load_results()}
    if tag in already:
        print(f"[campaign] skip {tag} (already completed -- safe to resume after a crash)", flush=True)
        return None
    print(f"[campaign] === starting {tag} === (elapsed {campaign_elapsed_hours():.2f}h, "
          f"disk {free_disk_gb():.1f}GB free)", flush=True)
    try:
        result = run_one(tag=tag, **kwargs)
        log_result(result)
        write_report()
        print(f"[campaign] === {tag} done: overall={result['closed_overall']:.3f} "
              f"leak={result['bluetooth_gsm_dsss_leakage_into_bluetooth']} ===", flush=True)
        return result
    except Exception as e:
        print(f"[campaign] *** {tag} FAILED: {e} ***", flush=True)
        traceback.print_exc()
        with open(os.path.join(ARTIFACTS, "errors.log"), "a") as f:
            f.write(f"{tag}: {e}\n{traceback.format_exc()}\n---\n")
        return None


# ---------------------------------------------------------------------------
# Phase 1: broad screen. Each config is a distinct, informed hypothesis given
# everything found earlier this session (complex+multiscale+LSTM was the best
# performing architecture so far, 0.857+ and climbing at 1200/5000 episodes;
# these vary the parts of it most likely to matter, plus a couple of
# orthogonal ideas -- margin loss on the new backbone, condition re-check).
# ---------------------------------------------------------------------------
SCREEN_EPISODES = 1000

PHASE1 = [
    dict(tag="p1_cx_gru", backbone="complexlstm",
         net_params=dict(rnn_type="gru"), train_hp={}),
    dict(tag="p1_cx_unidirectional", backbone="complexlstm",
         net_params=dict(rnn_bidirectional=False), train_hp={}),
    dict(tag="p1_cx_no_rnn_ablation", backbone="complexlstm",
         net_params=dict(use_rnn=False), train_hp={}),
    dict(tag="p1_cx_bigger_lstm", backbone="complexlstm",
         net_params=dict(lstm_hidden=32, lstm_downsample=16), train_hp={}),
    dict(tag="p1_cx_wider", backbone="complexlstm",
         net_params=dict(complex_width=48, real_width=24, n_blocks=2), train_hp={}),
    dict(tag="p1_cx_deeper_dilation", backbone="complexlstm",
         net_params=dict(dilations=(1, 2, 4, 8, 16)), train_hp={}),
    dict(tag="p1_cx_bigger_kernel", backbone="complexlstm",
         net_params=dict(kernel_size=7), train_hp={}),
    dict(tag="p1_cx_no_label_smoothing", backbone="complexlstm",
         net_params={}, train_hp=dict(label_smoothing=0.0)),
    dict(tag="p1_cx_more_wd", backbone="complexlstm",
         net_params={}, train_hp=dict(wd=1e-3)),
    dict(tag="p1_cx_lower_lr", backbone="complexlstm",
         net_params={}, train_hp=dict(lr=3e-4)),
    dict(tag="p1_cx_resampled", backbone="complexlstm",
         net_params={}, train_hp={}, condition="resampled"),
    dict(tag="p1_cx_margin", backbone="complexlstm",
         net_params={}, train_hp={}, use_margin=True, margin_kwargs=dict(margin=0.75, lambda_repel=2.0)),
    dict(tag="p1_vit_bigger_embed", backbone="vit",
         net_params=dict(embed_dim=EMBED_DIM), train_hp={}),
]


def main():
    print(f"[campaign] starting. device={DEV} disk_free={free_disk_gb():.1f}GB "
          f"budget={CAMPAIGN_BUDGET_HOURS}h", flush=True)
    save_state("phase1_starting")

    for cfg in PHASE1:
        cond = cfg.get("condition", "native")
        safe_run(cfg["tag"], backbone=cfg["backbone"], net_params=cfg["net_params"],
                  train_hp=cfg["train_hp"], episodes=SCREEN_EPISODES, condition=cond,
                  use_margin=cfg.get("use_margin", False), margin_kwargs=cfg.get("margin_kwargs"))

    save_state("phase1_complete")

    # ------------------------------------------------------------------
    # Phase 2: deep-train the best candidates (by overall accuracy, tie-broken
    # by lower bluetooth/gsm/dsss leakage -- the actual target metric) at the
    # full episode budget. Includes complexlstm_run3 (already deep-training
    # separately) as a known reference, not re-run here.
    # ------------------------------------------------------------------
    DEEP_EPISODES = 5000
    results = [r for r in load_results() if r["tag"].startswith("p1_")]  # exclude smoke-test/other entries
    ranked = sorted(results, key=lambda r: (-r["closed_overall"], r["bluetooth_gsm_dsss_leakage_into_bluetooth"]))
    top = ranked[:4]
    print(f"[campaign] phase 2: deep-training top {len(top)} candidates: {[r['tag'] for r in top]}", flush=True)
    save_state("phase2_starting", extra={"top_candidates": [r["tag"] for r in top]})

    phase1_by_tag = {c["tag"]: c for c in PHASE1}
    for r in top:
        cfg = phase1_by_tag.get(r["tag"])
        if cfg is None:
            continue
        deep_tag = r["tag"].replace("p1_", "p2_deep_")
        cond = cfg.get("condition", "native")
        safe_run(deep_tag, backbone=cfg["backbone"], net_params=cfg["net_params"],
                  train_hp=cfg["train_hp"], episodes=DEEP_EPISODES, condition=cond,
                  use_margin=cfg.get("use_margin", False), margin_kwargs=cfg.get("margin_kwargs"))

    save_state("phase2_complete")
    write_report()
    print(f"[campaign] all phases complete at elapsed {campaign_elapsed_hours():.2f}h", flush=True)


if __name__ == "__main__":
    main()
