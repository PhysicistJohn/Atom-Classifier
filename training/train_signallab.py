"""Retrain the embedding on Atom-SignalLab's OWN I/Q — the distribution the
Atomizer app actually feeds the classifier.

The rfgen-trained model did not transfer to SignalLab's I/Q (CW read as AM, FM as
BPSK). This retrains on a corpus whose fixed profiles begin with SignalLab's exact
native artifacts and then pass through the separately identified classifier receiver
transform (see tools/generate-signallab-iq-corpus.ts). The seven I/Q-separable classes
map to the app's protocol-leaf taxonomy:
    am · bluetooth · cw · dsss · fm · gsm · ofdm

Key alignment with the app: the app feeds CLEAN (noise-free, deterministic)
SignalLab I/Q, so prototypes and validation use clean preprocess, while the metric
is trained on lightly-augmented views (small AWGN/CFO/scale) for a robust space.

Run:  .venv-training/bin/python training/train_signallab.py
Exports src/embedding/assets/{embedding-weights,prototypes,parity-fixture,model-manifest}.json
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import preprocess as pp  # noqa: E402
import dataset as ds  # noqa: E402
from model import Embedding, INPUT_LEN, EMBED_DIM, N_FEATURES  # noqa: E402
from train import np_forward, embed_all, prototypes_from, nearest, sq_dist  # noqa: E402

SEED = 20260721
CORPUS = os.path.join(os.path.dirname(__file__), "artifacts", "signallab-corpus")
ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "embedding", "assets")
EPISODES = 6000
EVAL_EVERY = 1000
K_SHOT = 5
Q_QUERY = 5
AUG_PER_TRAIN = 4
RECEIVER_TRANSFORM_ID = "classifier-receiver-iq-transport-v1"
RECEIVER_TRANSFORM_ALGORITHM = (
    "deterministic-complex-rotator-blackman-windowed-sinc-resample-"
    "and-explicit-window-v1"
)


def load_corpus(*, include_provenance=False):
    manifest_path = os.path.join(CORPUS, "corpus.json")
    with open(manifest_path, "rb") as manifest_file:
        manifest_bytes = manifest_file.read()
    manifest = json.loads(manifest_bytes)
    _validate_corpus_manifest(manifest)
    n = manifest["count"]
    L = manifest["sampleCount"]
    raw = np.fromfile(os.path.join(CORPUS, "corpus.f32"), dtype="<f4").reshape(n, L, 2)
    for index, item in enumerate(manifest["items"]):
        observed = hashlib.sha256(raw[index].tobytes(order="C")).hexdigest()
        if observed != item["captureSha256"]:
            raise ValueError(
                f"corpus item {index} capture hash {observed} "
                f"does not match manifest {item['captureSha256']}"
            )
    iq = raw[:, :, 0].astype(np.float64) + 1j * raw[:, :, 1].astype(np.float64)
    classes = sorted(manifest["classes"])
    cindex = {c: i for i, c in enumerate(classes)}
    y = np.array([cindex[it["cls"]] for it in manifest["items"]], dtype=np.int64)
    impaired = np.array([bool(it.get("impaired", False)) for it in manifest["items"]])
    result = (iq, y, impaired, classes)
    if not include_provenance:
        return result
    provenance = {
        "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "lineageSchemaVersion": manifest["schemaVersion"],
        "signalLabSource": manifest["signalLabSource"],
        "sourcePolicy": manifest["sourcePolicy"],
    }
    return (*result, provenance)


def _validate_corpus_manifest(manifest):
    if manifest.get("schemaVersion") != 2:
        raise ValueError("SignalLab I/Q corpus must use source/receiver lineage schema 2")
    source = manifest.get("signalLabSource", {})
    if source.get("repository") != "Atom-SignalLab" or source.get("dirty") is not False:
        raise ValueError("SignalLab I/Q corpus must identify a clean Atom-SignalLab source")
    if not re.fullmatch(r"[a-f0-9]{40}", str(source.get("commit", ""))):
        raise ValueError("SignalLab I/Q corpus source commit must be a full Git object ID")
    policy = manifest.get("sourcePolicy", {})
    if (
        policy.get("fixedProfiles")
        != "exact-native-qualified-artifact-before-receiver-transform"
        or policy.get("flexibleProfiles")
        != "intrinsic-signal-bandwidth-continuous-source"
        or policy.get("transformedQualification")
        != "derived-from-independently-verified-digital-baseband"
        or policy.get("canonicalArtifactBytesRelabeled") is not False
        or policy.get("receiverCaptureBandwidthSeparate") is not True
        or policy.get("receiverTransform") != {
            "id": RECEIVER_TRANSFORM_ID,
            "algorithm": RECEIVER_TRANSFORM_ALGORITHM,
            "boundary": "zero",
        }
    ):
        raise ValueError("SignalLab I/Q corpus source/receiver policy is missing or unsafe")
    items = manifest.get("items")
    if not isinstance(items, list) or len(items) != manifest.get("count"):
        raise ValueError("SignalLab I/Q corpus item count does not match its manifest")
    sha256_pattern = re.compile(r"[a-f0-9]{64}")
    for index, item in enumerate(items):
        source_iq = item.get("sourceIq", {})
        transform = item.get("receiverTransform", {})
        for label, value in (
            ("source samples", source_iq.get("sourceSamplesSha256")),
            ("receiver output", item.get("receiverOutputSamplesSha256")),
            ("clean pair", item.get("cleanPairSha256")),
            ("capture", item.get("captureSha256")),
        ):
            if not sha256_pattern.fullmatch(str(value or "")):
                raise ValueError(f"corpus item {index} has invalid {label} SHA-256")
        if transform.get("sourceSamplesSha256") != source_iq.get("sourceSamplesSha256"):
            raise ValueError(f"corpus item {index} source hash breaks transform lineage")
        if transform.get("outputSamplesSha256") != item.get("receiverOutputSamplesSha256"):
            raise ValueError(f"corpus item {index} target hash breaks transform lineage")
        if (
            item.get("bandwidthSemantics") != "receiver-capture-bandwidth"
            or item.get("bandwidthHz") != item.get("captureBandwidthHz")
        ):
            raise ValueError(f"corpus item {index} conflates signal and capture bandwidth")
        signal_bandwidth = source_iq.get("signalBandwidthHz")
        capture_bandwidth = item.get("captureBandwidthHz")
        output_rate = item.get("sampleRateHz")
        if (
            not isinstance(signal_bandwidth, (int, float))
            or not isinstance(capture_bandwidth, (int, float))
            or not isinstance(output_rate, (int, float))
            or signal_bandwidth <= 0
            or capture_bandwidth < signal_bandwidth
            or capture_bandwidth > output_rate
        ):
            raise ValueError(f"corpus item {index} has invalid signal/capture geometry")
        if (
            transform.get("receiptVersion") != 1
            or transform.get("id") != RECEIVER_TRANSFORM_ID
            or transform.get("algorithm") != RECEIVER_TRANSFORM_ALGORITHM
            or transform.get("boundary") != "zero"
            or transform.get("sourceSampleRateHz") != source_iq.get("nativeSampleRateHz")
            or transform.get("sourceStartSample") != source_iq.get("startSampleIndex")
            or transform.get("sourceSampleCount") != source_iq.get("sampleCount")
            or transform.get("sourceCarrierOffsetHz")
            != source_iq.get("nativeCarrierOffsetHz")
            or transform.get("outputCarrierOffsetHz") != 0
            or transform.get("frequencyTranslationHz")
            != -source_iq.get("nativeCarrierOffsetHz", 0)
            or transform.get("outputSampleRateHz") != output_rate
            or transform.get("outputSampleCount") != manifest.get("sampleCount")
        ):
            raise ValueError(f"corpus item {index} output geometry breaks transform lineage")
        source_rate = transform["sourceSampleRateHz"]
        target_rate = transform["outputSampleRateHz"]
        rate_ratio = target_rate / source_rate
        normalized_cutoff = 0.5 if rate_ratio >= 1 else 0.5 * rate_ratio * 0.95
        expected_cutoff_hz = normalized_cutoff * source_rate
        expected_radius = math.ceil(16 / (2 * normalized_cutoff))
        if (
            not math.isclose(
                transform.get("antiAliasCutoffHz", -1),
                expected_cutoff_hz,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            or transform.get("kernelRadiusSourceSamples") != expected_radius
        ):
            raise ValueError(f"corpus item {index} has an invalid receiver filter receipt")
        expected_operations = []
        source_carrier_offset = source_iq.get("nativeCarrierOffsetHz", 0)
        if source_carrier_offset != 0:
            expected_operations.append({
                "kind": "frequency-translate",
                "algorithm": "complex-rotator-v1",
                "sourceCarrierOffsetHz": source_carrier_offset,
                "outputCarrierOffsetHz": 0,
            })
        if source_rate != target_rate:
            expected_operations.append({
                "kind": "resample",
                "algorithm": "blackman-windowed-sinc-v1",
                "sourceSampleRateHz": source_rate,
                "outputSampleRateHz": target_rate,
                "antiAliasCutoffHz": expected_cutoff_hz,
                "zeroCrossings": 16,
            })
        if transform.get("operations") != expected_operations:
            raise ValueError(f"corpus item {index} has invalid receiver operations")
        replay = source_iq.get("replay")
        qualification = source_iq.get("qualification")
        if qualification == "independently-verified-digital-baseband":
            if not sha256_pattern.fullmatch(str(source_iq.get("artifactSha256") or "")):
                raise ValueError(f"corpus item {index} fixed source lacks artifact identity")
            if transform.get("sourceArtifactSha256") != source_iq.get("artifactSha256"):
                raise ValueError(f"corpus item {index} artifact hash breaks transform lineage")
            if (
                transform.get("qualification")
                != "derived-from-independently-verified-digital-baseband"
            ):
                raise ValueError(f"corpus item {index} relabels transformed fixed I/Q")
            if replay == "cyclic":
                if (
                    not isinstance(source_iq.get("nativePeriodSamples"), int)
                    or source_iq["nativePeriodSamples"] <= 0
                    or "maxOneShotSamples" in source_iq
                ):
                    raise ValueError(f"corpus item {index} has invalid cyclic source bounds")
            elif replay == "one-shot":
                if (
                    source_iq.get("maxOneShotSamples") != source_iq.get("sampleCount")
                    or source_iq.get("sourceSamplesSha256") != source_iq.get("artifactSha256")
                    or "nativePeriodSamples" in source_iq
                ):
                    raise ValueError(f"corpus item {index} has invalid one-shot source bounds")
            else:
                raise ValueError(f"corpus item {index} fixed source has invalid replay semantics")
        elif qualification == "analytic-or-standards-derived-parameterized-source":
            if (
                replay != "continuous"
                or source_iq.get("artifactSha256") is not None
                or transform.get("sourceArtifactSha256") is not None
                or transform.get("qualification")
                != "receiver-transformed-parameterized-complex-baseband"
                or "nativePeriodSamples" in source_iq
                or "maxOneShotSamples" in source_iq
            ):
                raise ValueError(f"corpus item {index} has invalid flexible-source lineage")
        else:
            raise ValueError(f"corpus item {index} has unknown source qualification")


def build_pool(iq, y, idx, rng, scale_jitter=0.0):
    """Preprocess each corpus signal. Impairments are baked into the corpus
    (SignalLab's receiver model), so no extra augmentation here — only optional
    scale jitter for bandwidth-estimate robustness."""
    xs, fs, ys = [], [], []
    for i in idx:
        norm, _ = pp.preprocess(iq[i], scale_jitter=scale_jitter, rng=rng)
        xs.append(pp.to_channels(norm))
        fs.append(pp.iq_features(norm))
        ys.append(y[i])
    return (np.stack(xs).astype(np.float32), np.stack(fs).astype(np.float32),
            np.array(ys, dtype=np.int64))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    iq, y, impaired, classes, corpus_provenance = load_corpus(
        include_provenance=True
    )
    n_classes = len(classes)
    print(f"device: {dev} | corpus {iq.shape} | impaired {int(impaired.sum())}/{len(impaired)} | classes {classes}")

    # Clean realizations -> prototype enrollment + validation (the app feeds clean
    # I/Q, so the space must be anchored on clean). Impaired realizations (+ the
    # remaining clean) -> train, so the metric is robust to real receiver/channel
    # variation. Split by realization to avoid leakage.
    clean_pos = rng.permutation(np.where(~impaired)[0])
    imp_pos = np.where(impaired)[0]
    n_en = int(0.30 * len(clean_pos))
    n_va = int(0.30 * len(clean_pos))
    en_idx = clean_pos[:n_en]
    va_idx = clean_pos[n_en:n_en + n_va]
    tr_idx = np.concatenate([clean_pos[n_en + n_va:], imp_pos])

    print("building pools ...")
    xtr, ftr, ytr = build_pool(iq, y, tr_idx, rng, scale_jitter=0.08)
    fmean = ftr.mean(0); fstd = ftr.std(0) + 1e-6
    ftr = (ftr - fmean) / fstd
    idx_by_class = ds.class_indices(ytr, n_classes)
    xen, fen, yen = build_pool(iq, y, en_idx, rng, scale_jitter=0.0)
    fen = (fen - fmean) / fstd
    xva, fva, yva = build_pool(iq, y, va_idx, rng, scale_jitter=0.0)
    fva = (fva - fmean) / fstd
    print(f"  train {xtr.shape}  enroll {xen.shape}  val {xva.shape}")

    net = Embedding(EMBED_DIM, N_FEATURES).to(dev)
    scale = torch.nn.Parameter(torch.tensor(10.0, device=dev))
    opt = torch.optim.Adam(list(net.parameters()) + [scale], lr=1e-3, weight_decay=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPISODES)
    n_way = n_classes

    def eval_clean():
        protos = prototypes_from(embed_all(net, xen, fen, dev), yen, n_classes)
        pred, _ = nearest(embed_all(net, xva, fva, dev), protos)
        return float((pred == yva).mean())

    print(f"training {EPISODES} episodes ({n_way}-way {K_SHOT}-shot) ...")
    running, best_acc, best_state = 0.0, -1.0, None
    for ep in range(EPISODES):
        net.train()
        sx, sf, sl, qx, qf, ql = ds.sample_episode(xtr, ftr, idx_by_class, rng, n_way, K_SHOT, Q_QUERY)
        xb = torch.from_numpy(np.concatenate([sx, qx])).to(dev)
        fb = torch.from_numpy(np.concatenate([sf, qf])).to(dev)
        emb = net(xb, fb)
        se, qe = emb[: len(sx)], emb[len(sx):]
        protos = torch.stack([se[torch.from_numpy(sl).to(dev) == c].mean(0) for c in range(n_way)])
        logits = -sq_dist(qe, protos) * scale
        loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(ql).to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        running += loss.item()
        if (ep + 1) % 500 == 0:
            print(f"  ep {ep+1:5d}  loss {running/500:.4f}"); running = 0.0
        if (ep + 1) % EVAL_EVERY == 0:
            acc = eval_clean()
            tag = ""
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(net.state_dict()); tag = "  <- best"
            print(f"    [val(clean) {acc:.3f}]{tag}")
    if best_state is not None:
        net.load_state_dict(best_state)

    # enroll + validate on clean
    emb_en = embed_all(net, xen, fen, dev)
    protos = prototypes_from(emb_en, yen, n_classes)
    emb_va = embed_all(net, xva, fva, dev)
    pred, dists = nearest(emb_va, protos)
    closed_acc = float((pred == yva).mean())
    nn_known = dists.min(1)
    # generous open-set threshold: the app can feed geometry outside the training
    # val distribution, so known signals sit farther than the clean-val spread.
    # Keep the safety valve, but do not flag the 7 known classes as unknown.
    thr = float(max(np.quantile(nn_known, 0.99) * 1.6, float(nn_known.max()) * 1.25))
    print(f"clean closed-set accuracy: {closed_acc:.3f}")
    print("per class:", {classes[c]: round(float((pred[yva == c] == c).mean()), 3) for c in range(n_classes)})

    # temperature (min NLL on clean val)
    best_T, best_nll = 1.0, math.inf
    for T in np.linspace(0.02, 2.0, 100):
        lg = -dists / T; lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        nll = -np.log(p[np.arange(len(yva)), yva] + 1e-12).mean()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)

    weights = net.export_weights()
    weights["feat_mean"] = fmean.astype(float).tolist()
    weights["feat_std"] = fstd.astype(float).tolist()
    net.eval()
    with torch.no_grad():
        probe_x = xva[:12]
        torch_emb = net(torch.from_numpy(probe_x).to(dev), torch.from_numpy(fva[:12]).to(dev)).cpu().numpy()
    np_emb = np.stack([np_forward(probe_x[i], weights) for i in range(len(probe_x))])
    fold_err = float(np.abs(torch_emb - np_emb).max())
    print(f"BN-fold parity: max|dz| = {fold_err:.2e}")
    assert fold_err < 1e-4

    os.makedirs(ASSET_DIR, exist_ok=True)
    with open(os.path.join(ASSET_DIR, "embedding-weights.json"), "w") as f:
        json.dump({"input_len": INPUT_LEN, "embed_dim": EMBED_DIM, "n_features": N_FEATURES,
                   "preprocess": {"l_out": pp.L_OUT, "target_frac": pp.TARGET_FRAC, "nfft": pp.NFFT,
                                  "energy_edge": pp.ENERGY_EDGE, "noise_floor_scale": pp.NOISE_FLOOR_SCALE, "smooth": pp.SMOOTH},
                   **weights}, f)
    with open(os.path.join(ASSET_DIR, "prototypes.json"), "w") as f:
        json.dump({"classes": classes, "embed_dim": EMBED_DIM,
                   "prototypes": protos.astype(float).tolist(),
                   "unknown_threshold": thr, "temperature": best_T}, f)
    fwd_fixture = [{"input": xva[i].ravel().astype(float).tolist(), "embedding": np_emb[i].astype(float).tolist()}
                   for i in range(8)]
    # clean preprocess parity fixtures from raw corpus signals
    pre_fixture = []
    for c in [0, 1, 2]:
        gi = int(np.where(y == c)[0][0])
        norm, ctx = pp.preprocess(iq[gi], scale_jitter=0.0)
        pre_fixture.append({"class": classes[c],
                            "iq": np.stack([iq[gi].real, iq[gi].imag], 0).ravel().astype(float).tolist(),
                            "iq_len": int(len(iq[gi])),
                            "expected": pp.to_channels(norm).ravel().astype(float).tolist(),
                            "features": pp.iq_features(norm).astype(float).tolist(),
                            "center": ctx["center"], "bw": ctx["bw"]})
    with open(os.path.join(ASSET_DIR, "parity-fixture.json"), "w") as f:
        json.dump({"forward": fwd_fixture, "preprocess": pre_fixture}, f)
    with open(os.path.join(ASSET_DIR, "model-manifest.json"), "w") as f:
        json.dump({"seed": SEED, "source": "signallab-native-iq-plus-classifier-receiver-transform-v1", "classes": classes,
                   "episodes": EPISODES, "metrics": {"clean_closed_set_accuracy": closed_acc, "best_val_accuracy": best_acc},
                   "unknown_threshold": thr, "temperature": best_T,
                   "corpus": corpus_provenance}, f, indent=2)
    print("done. assets ->", os.path.relpath(ASSET_DIR))


if __name__ == "__main__":
    main()
