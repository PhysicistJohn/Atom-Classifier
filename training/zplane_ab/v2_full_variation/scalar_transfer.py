"""The literal transfer contract: freeze a U-Net and add one scalar output head.

The positive target is SNR on impaired rows only.  It was not an auxiliary training label,
survives input RMS normalization as a ratio, and is useful to a receiver.  Ridge strength is
chosen on the model-selection split; the held-out test split is touched once.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch


CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "artifacts",
    "signallab-corpus",
)
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _targets(corpus_indices, key="snrDb"):
    with open(os.path.join(CORPUS, "corpus.json")) as f:
        items = json.load(f)["items"]
    return np.asarray([float(items[int(i)].get(key, 0.0) or 0.0)
                       for i in corpus_indices], dtype=np.float64)


@torch.no_grad()
def extract_bottleneck(net, x, feat, dev, batch=128):
    net = net.to(dev).eval()
    out = []
    for start in range(0, len(x), batch):
        stop = min(start + batch, len(x))
        xb = torch.from_numpy(np.asarray(x[start:stop], dtype=np.float32)).to(dev)
        fb = torch.from_numpy(np.asarray(feat[start:stop], dtype=np.float32)).to(dev)
        net(xb, fb)
        out.append(net.last_bottleneck_pool.detach().float().cpu().numpy())
    return np.concatenate(out)


@dataclass
class RidgeHead:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    lam: float

    def predict(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        z = np.concatenate([z, np.ones((len(z), 1))], axis=1)
        # NumPy 2.0's Accelerate-backed ``matmul`` on this deployment spuriously raises
        # floating-point warnings for finite, well-scaled inputs. ``dot`` takes the same
        # BLAS path mathematically without contaminating strict numerical checks.
        return np.dot(z, self.weights)

    def serializable(self):
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "lambda": float(self.lam),
        }


def fit_ridge(x, y, lam):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    z = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    penalty = np.eye(z.shape[1])
    penalty[-1, -1] = 0.0
    gram = np.dot(z.T, z)
    rhs = np.dot(z.T, y)
    try:
        w = np.linalg.solve(gram + float(lam) * penalty, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(gram + float(lam) * penalty, rhs, rcond=None)[0]
    return RidgeHead(mean, scale, w, float(lam))


def metrics(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    residual = y - pred
    denom = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(residual ** 2) / max(denom, 1e-12)
    corr = np.corrcoef(y, pred)[0, 1] if np.std(pred) > 0 and np.std(y) > 0 else 0.0
    return {
        "r2": float(r2),
        "mae_db": float(np.mean(np.abs(residual))),
        "rmse_db": float(np.sqrt(np.mean(residual ** 2))),
        "pearson": float(corr),
    }


def tune_ridge(x_fit, y_fit, x_select, y_select):
    candidates = []
    for lam in LAMBDAS:
        head = fit_ridge(x_fit, y_fit, lam)
        score = metrics(y_select, head.predict(x_select))
        candidates.append((score["r2"], -score["mae_db"], head))
    return max(candidates, key=lambda row: (row[0], row[1]))[2]


def _bootstrap_delta(y, a, b, seed=0, draws=4000):
    """Paired improvement in absolute error: positive means predictor a is better."""
    y = np.asarray(y)
    delta = np.abs(y - b) - np.abs(y - a)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(1)
    return {
        "mean_mae_improvement_db": float(delta.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def scalar_transfer_report(trained_net, random_net, enroll, selection, test, dev,
                           target_key="snrDb"):
    """Fit one-output frozen heads and evaluate them on impaired-only untouched test rows."""
    splits = {}
    for name, data, suffix in (
        ("fit", enroll, "en"),
        ("selection", selection, "va"),
        ("test", test, "va"),
    ):
        mask = np.asarray(data[f"imp_{suffix}"], dtype=bool)
        x = np.asarray(data[f"x{suffix}"])[mask]
        f = np.asarray(data[f"f{suffix}"])[mask]
        idx = np.asarray(data[f"{suffix}_idx"])[mask]
        splits[name] = {
            "trained": extract_bottleneck(trained_net, x, f, dev),
            "random": extract_bottleneck(random_net, x, f, dev),
            "feat12": np.asarray(f, dtype=np.float64),
            "target": _targets(idx, target_key),
            "n": int(mask.sum()),
        }

    heads, preds, scores = {}, {}, {}
    for tap in ("trained", "random", "feat12"):
        head = tune_ridge(
            splits["fit"][tap], splits["fit"]["target"],
            splits["selection"][tap], splits["selection"]["target"],
        )
        pred = head.predict(splits["test"][tap])
        heads[tap] = head
        preds[tap] = pred
        scores[tap] = metrics(splits["test"]["target"], pred)

    y = splits["test"]["target"]
    const = np.full_like(y, splits["fit"]["target"].mean())
    scores["constant"] = metrics(y, const)
    vs_random = _bootstrap_delta(y, preds["trained"], preds["random"], seed=17)
    vs_feat = _bootstrap_delta(y, preds["trained"], preds["feat12"], seed=19)
    transfers = bool(
        scores["trained"]["r2"] > 0.0
        and vs_random["ci95"][0] > 0.0
    )
    return {
        "target": target_key,
        "contract": "frozen U-Net; add and train one scalar ridge output only",
        "rows": {name: split["n"] for name, split in splits.items()},
        "test_metrics": scores,
        "vs_random_encoder": vs_random,
        "vs_feat12": vs_feat,
        "transfers": transfers,
        "head": heads["trained"].serializable(),
        "controls": {
            "random_head": heads["random"].serializable(),
            "feat12_head": heads["feat12"].serializable(),
        },
    }
