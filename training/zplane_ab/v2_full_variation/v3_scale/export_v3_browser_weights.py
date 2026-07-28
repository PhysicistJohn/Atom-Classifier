"""Convert a v3 runtime bundle into deterministic browser JSON weights.

Reads the ``atomos.v3.time-domain-invariant-fusion.runtime-bundle`` directory
written by ``export_v3_fusion_runtime.py`` and emits a JSON asset set that the
TypeScript runtime (``src/embedding/time-domain-encoder-v3.ts`` and
``src/embedding/time-domain-fusion-v3.ts``) can load without torch, numpy, or
any binary parsing:

* ``time-domain-fusion-weights-v3.json`` -- both branch encoders (real conv
  stacks with UNFOLDED BatchNorm statistics, paired-real complex stages),
  fusion centers and scalars taken from the state dict (float32-exact), the
  feature standardization moments, and the prototype classification head.
* ``time-domain-probe-fixture-v3.json`` -- a byte-identical copy of the
  bundle's ``probe_fixture.json``, so its SHA-256 is unchanged and the
  TypeScript parity suite anchors on exactly the Python-verified fixture.
* ``export-manifest.json`` -- SHA-256 of each emitted file plus the source
  bundle SHAs.  No timestamps anywhere; re-running the exporter on the same
  bundle produces byte-identical output.

Every bundle asset is SHA-verified against ``bundle_manifest.json`` before
conversion.  BatchNorm is exported unfolded (weight, bias, running_mean,
running_var, eps) rather than folded into the convolution, so the TypeScript
runtime can apply torch's exact evaluation-mode arithmetic; folding would
introduce an avoidable reduction-order difference.

This produces a development staging asset.  It is not release evidence, and it
refuses to write into ``src/embedding/assets`` (the live model), any release or
sealed path, or a non-empty output directory.  Both input and output are
required CLI arguments so an old default candidate cannot be exported by
accident.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ZPAB = V2.parent
TRAINING = ZPAB.parent
REPO = TRAINING.parent

BUNDLE_SCHEMA_ID = "atomos.v3.time-domain-invariant-fusion.runtime-bundle"
BROWSER_SCHEMA_ID = "atomos.v3.time-domain-invariant-fusion.browser-weights"
BROWSER_SCHEMA_VERSION = 1

MANIFEST_NAME = "bundle_manifest.json"
PROBE_NAME = "probe_fixture.json"
WEIGHTS_NAME = "time-domain-fusion-weights-v3.json"
FIXTURE_COPY_NAME = "time-domain-probe-fixture-v3.json"
EXPORT_MANIFEST_NAME = "export-manifest.json"

# torch defaults baked into the trained modules; recorded explicitly so the
# TypeScript runtime never has to guess a framework default.
BATCH_NORM_EPS = 1e-5  # nn.BatchNorm1d(eps=...) default used by _RealBlock
MAG_NORM_EPS = 1e-6  # unet_transfer.ComplexMagNorm default
MODRELU_MAGNITUDE_EPS = 1e-12  # invariant_patch_cnn.ComplexModReLU magnitude
EMBEDDING_NORMALIZE_EPS = 1e-12  # F.normalize default eps
LOWPASS = (0.25, 0.5, 0.25)  # _complex_lowpass_decimate kernel
COMPLEX_LAGS = (1, 2, 4)  # _ComplexPatchEncoder.lags

REAL_BLOCKS = ((2, 24, 7), (24, 48, 5), (48, 64, 3), (64, 64, 3))
COMPLEX_STAGES = ((1, 16, 7), (16, 32, 5), (32, 48, 3))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"refusing to serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON: sorted keys, no NaN, trailing newline."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def reject_unsafe_output(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    lowered = {part.lower() for part in resolved.parts}
    if "releases" in lowered or any("sealed" in part for part in lowered):
        raise ValueError(f"refusing release or sealed output path: {resolved}")
    live_assets = (REPO / "src" / "embedding" / "assets").resolve()
    if resolved == live_assets or live_assets in resolved.parents:
        raise ValueError(
            f"refusing to write into the live model asset directory: {resolved}"
        )
    return resolved


def validate_empty_output(path: Path) -> Path:
    """Return a safe fresh output path, refusing silent asset replacement."""
    output = reject_unsafe_output(path)
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"browser output is not a directory: {output}")
        contents = sorted(item.name for item in output.iterdir())
        if contents:
            raise FileExistsError(
                f"refusing to overwrite non-empty browser output {output} "
                f"(contains {', '.join(contents[:5])})"
            )
    return output


def load_verified_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Load the bundle manifest and SHA-verify every asset it names."""
    bundle_dir = Path(bundle_dir).resolve()
    manifest_path = bundle_dir / MANIFEST_NAME
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != BUNDLE_SCHEMA_ID:
        raise ValueError(f"unexpected bundle schema {manifest.get('schema')!r}")
    if manifest.get("schema_version") != 1:
        raise ValueError("unexpected bundle schema version")
    for name, recorded in manifest["assets"].items():
        asset_path = bundle_dir / name
        actual = _sha256(asset_path)
        if actual != recorded["sha256"]:
            raise ValueError(
                f"bundle asset {name} SHA mismatch: {actual} != "
                f"{recorded['sha256']}"
            )
    return {"dir": bundle_dir, "manifest": manifest}


def validate_external_staged_rejection(manifest: Mapping[str, Any]) -> None:
    """Require the classifier bundle's explicit external-policy separation.

    The browser fusion export contains no rejector.  Accepting a bundle with a
    fitted legacy additive slot, or an older "not refit yet" placeholder, would
    silently describe a different classifier from the staged release path.
    """
    rejection = manifest.get("rejection")
    if not isinstance(rejection, Mapping):
        raise ValueError("runtime bundle carries no rejection contract")
    if rejection.get("state") != "unset":
        raise ValueError(
            "browser fusion export requires an unset classifier rejection "
            "slot; staged rejection is exported separately"
        )
    if rejection.get("external_staged_policy_required_for_abstention") is not True:
        raise ValueError(
            "runtime bundle does not declare the external staged-policy "
            "requirement"
        )
    contract = rejection.get("required_contract")
    if not isinstance(contract, Mapping) or "stage-one survivors" not in str(
        contract.get("scope", "")
    ):
        raise ValueError(
            "runtime bundle rejection contract does not scope stage two to "
            "stage-one survivors"
        )
    blockers = manifest.get("release_blockers")
    if not isinstance(blockers, list) or not blockers:
        raise ValueError("runtime bundle carries no release requirements")
    serialized = " ".join(str(item) for item in blockers).lower()
    stale = ("not refit", "unmeasured", "unported", "no untouched release seed")
    found = [token for token in stale if token in serialized]
    if found:
        raise ValueError(
            f"runtime bundle carries stale release blockers: {found}"
        )


def _f32_vector(tensor: torch.Tensor, name: str, shape: tuple[int, ...]) -> list:
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {tensor.dtype}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor.reshape(-1).tolist()


def _linear(state: Mapping[str, torch.Tensor], prefix: str,
            in_dim: int, out_dim: int) -> dict[str, Any]:
    return {
        "in": in_dim,
        "out": out_dim,
        "weight": _f32_vector(
            state[f"{prefix}.weight"], f"{prefix}.weight", (out_dim, in_dim)
        ),
        "bias": _f32_vector(state[f"{prefix}.bias"], f"{prefix}.bias", (out_dim,)),
    }


def _real_branch(state: Mapping[str, torch.Tensor],
                 config: Mapping[str, Any]) -> dict[str, Any]:
    blocks = []
    for index, (cin, cout, kernel) in enumerate(REAL_BLOCKS):
        prefix = f"real_branch.patch_encoder.blocks.{index}"
        blocks.append({
            "in_channels": cin,
            "out_channels": cout,
            "kernel": kernel,
            "stride": 2,
            "padding": kernel // 2,
            "conv_weight": _f32_vector(
                state[f"{prefix}.conv.weight"],
                f"{prefix}.conv.weight",
                (cout, cin, kernel),
            ),
            "batch_norm": {
                "eps": BATCH_NORM_EPS,
                "weight": _f32_vector(
                    state[f"{prefix}.bn.weight"], f"{prefix}.bn.weight", (cout,)
                ),
                "bias": _f32_vector(
                    state[f"{prefix}.bn.bias"], f"{prefix}.bn.bias", (cout,)
                ),
                "running_mean": _f32_vector(
                    state[f"{prefix}.bn.running_mean"],
                    f"{prefix}.bn.running_mean",
                    (cout,),
                ),
                "running_var": _f32_vector(
                    state[f"{prefix}.bn.running_var"],
                    f"{prefix}.bn.running_var",
                    (cout,),
                ),
            },
        })
    stat_width = 2 * REAL_BLOCKS[-1][1]
    patch_dim = int(config["patch_dim"])
    hidden = int(config["hidden"])
    embed_dim = int(config["embed_dim"])
    n_features = int(config["n_features"])
    set_width = patch_dim * (2 if config["set_pool"] == "mean_std" else 1)
    return {
        "config": dict(config),
        "blocks": blocks,
        "patch_projection": _linear(
            state, "real_branch.patch_projection.0", stat_width, patch_dim
        ),
        "fc1": _linear(
            state, "real_branch.fc1", set_width + n_features, hidden
        ),
        "fc2": _linear(state, "real_branch.fc2", hidden, embed_dim),
    }


def _complex_branch(state: Mapping[str, torch.Tensor],
                    config: Mapping[str, Any]) -> dict[str, Any]:
    stages = []
    for index, (cin, cout, kernel) in enumerate(COMPLEX_STAGES):
        prefix = f"complex_branch.patch_encoder.stages.{index}"
        stages.append({
            "in_channels": cin,
            "out_channels": cout,
            "kernel": kernel,
            "padding": (kernel - 1) // 2,
            "weight_real": _f32_vector(
                state[f"{prefix}.conv.conv_re.weight"],
                f"{prefix}.conv.conv_re.weight",
                (cout, cin, kernel),
            ),
            "weight_imag": _f32_vector(
                state[f"{prefix}.conv.conv_im.weight"],
                f"{prefix}.conv.conv_im.weight",
                (cout, cin, kernel),
            ),
            "threshold_raw": _f32_vector(
                state[f"{prefix}.act.thresh_raw"],
                f"{prefix}.act.thresh_raw",
                (cout,),
            ),
        })
    channels = COMPLEX_STAGES[-1][1]
    stat_width = channels * (2 + 2 * len(COMPLEX_LAGS))
    patch_dim = int(config["patch_dim"])
    hidden = int(config["hidden"])
    embed_dim = int(config["embed_dim"])
    n_features = int(config["n_features"])
    set_width = patch_dim * (2 if config["set_pool"] == "mean_std" else 1)
    return {
        "config": dict(config),
        "stages": stages,
        "mag_norm_eps": MAG_NORM_EPS,
        "modrelu_magnitude_eps": MODRELU_MAGNITUDE_EPS,
        "lowpass": list(LOWPASS),
        "lags": list(COMPLEX_LAGS),
        "patch_projection": _linear(
            state, "complex_branch.patch_projection.0", stat_width, patch_dim
        ),
        "fc1": _linear(
            state, "complex_branch.fc1", set_width + n_features, hidden
        ),
        "fc2": _linear(state, "complex_branch.fc2", hidden, embed_dim),
    }


def _scalar(state: Mapping[str, torch.Tensor], name: str) -> float:
    tensor = state[name]
    if tensor.shape != () or tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be a float32 scalar")
    value = float(tensor)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def build_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    manifest = bundle["manifest"]
    validate_external_staged_rejection(manifest)
    bundle_dir: Path = bundle["dir"]
    architecture = manifest["architecture"]
    state = torch.load(
        bundle_dir / "fusion_state_dict.pt",
        map_location="cpu",
        weights_only=True,
    )

    embed_dim = int(architecture["real"]["embed_dim"])
    n_features = int(architecture["real"]["n_features"])
    prototypes = np.load(bundle_dir / "fusion_prototypes.npy")
    real_center = np.load(bundle_dir / "real_center.npy")
    complex_center = np.load(bundle_dir / "complex_center.npy")
    feature_mean = np.load(bundle_dir / "feature_mean.npy")
    feature_std = np.load(bundle_dir / "feature_std.npy")

    classes = list(manifest["classification"]["classes"])
    if prototypes.shape != (len(classes), 2 * embed_dim):
        raise ValueError(
            f"fusion_prototypes.npy has shape {prototypes.shape}, expected "
            f"({len(classes)}, {2 * embed_dim})"
        )
    for name, array, expected in (
        ("real_center.npy", real_center, (embed_dim,)),
        ("complex_center.npy", complex_center, (embed_dim,)),
        ("feature_mean.npy", feature_mean, (n_features,)),
        ("feature_std.npy", feature_std, (n_features,)),
    ):
        if array.shape != expected or array.dtype != np.float32:
            raise ValueError(f"{name} must be float32 {expected}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    if (feature_std <= 0).any():
        raise ValueError("feature_std.npy must be strictly positive")

    # Fusion scalars and centers come from the state dict, which carries the
    # exact float32 buffers the torch forward used (e.g. alpha 0.2 stored as
    # 0.20000000298023224), not the manifest's rounded decimals.
    state_real_center = np.asarray(
        state["real_center"].numpy(), dtype=np.float32
    )
    state_complex_center = np.asarray(
        state["complex_center"].numpy(), dtype=np.float32
    )
    if not np.array_equal(state_real_center, real_center):
        raise ValueError("state dict real_center disagrees with real_center.npy")
    if not np.array_equal(state_complex_center, complex_center):
        raise ValueError(
            "state dict complex_center disagrees with complex_center.npy"
        )

    payload = {
        "schema": BROWSER_SCHEMA_ID,
        "schema_version": BROWSER_SCHEMA_VERSION,
        "status": "staging_not_release",
        "development_only": True,
        "kind": manifest["kind"],
        "packed_length": int(manifest["frontend"]["packed_length"]),
        "parameter_count": int(manifest["parameter_count"]),
        "frontend": {
            "version": manifest["frontend"]["version"],
            "estimator_version": manifest["frontend"]["estimator_version"],
            "patch_length": int(manifest["frontend"]["patch_length"]),
            "patch_count": int(manifest["frontend"]["patch_count"]),
            "target_frac": float(manifest["frontend"]["target_frac"]),
            "feature_count": int(manifest["frontend"]["feature_count"]),
            "uses_frequency_transform": False,
            "source_sha256": dict(manifest["frontend"]["source_sha256"]),
        },
        "real": _real_branch(state, architecture["real"]),
        "complex": _complex_branch(state, architecture["complex"]),
        "fusion": {
            "real_center": _f32_vector(
                state["real_center"], "real_center", (embed_dim,)
            ),
            "complex_center": _f32_vector(
                state["complex_center"], "complex_center", (embed_dim,)
            ),
            "alpha_real": _scalar(state, "alpha_real"),
            "alpha_complex": _scalar(state, "alpha_complex"),
            "weight_real": _scalar(state, "weight_real"),
            "eps": _scalar(state, "eps"),
        },
        "embedding_normalize_eps": EMBEDDING_NORMALIZE_EPS,
        "feature_standardization": {
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
            "epsilon": float(
                manifest["feature_standardization"]["epsilon"]
            ),
            "rule": manifest["feature_standardization"]["rule"],
        },
        "classification": {
            "classes": classes,
            "prototypes": prototypes.tolist(),
            "distance": manifest["classification"]["distance"],
            "label": manifest["classification"]["label"],
            "embedding": manifest["classification"]["embedding"],
        },
        "rejection": {
            "state": manifest["rejection"]["state"],
            "runtime_behaviour": manifest["rejection"]["runtime_behaviour"],
            "external_staged_policy_required_for_abstention": True,
        },
        "not_compatible_with_schema_ids": list(
            manifest["not_compatible_with_schema_ids"]
        ),
        "provenance": {
            "source_bundle_schema": manifest["schema"],
            "source_bundle_assets_sha256": {
                name: record["sha256"]
                for name, record in manifest["assets"].items()
            },
            "assembly_seed": int(manifest["provenance"]["assembly_seed"]),
            "exporter": "export_v3_browser_weights.py",
            "batch_norm_folded": False,
        },
        "release_blockers": list(manifest["release_blockers"]),
        "release_evidence": False,
    }
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    bundle = load_verified_bundle(Path(args.bundle))
    output = validate_empty_output(Path(args.output))
    output.mkdir(parents=True, exist_ok=True)

    payload = build_payload(bundle)
    weights_path = output / WEIGHTS_NAME
    _write_json(weights_path, payload)

    # Byte-identical fixture copy: the parity anchor keeps its bundle SHA.
    fixture_source = bundle["dir"] / PROBE_NAME
    fixture_path = output / FIXTURE_COPY_NAME
    shutil.copyfile(fixture_source, fixture_path)
    fixture_sha = _sha256(fixture_path)
    recorded = bundle["manifest"]["assets"][PROBE_NAME]["sha256"]
    if fixture_sha != recorded:
        raise RuntimeError(
            f"probe fixture copy SHA {fixture_sha} != bundle {recorded}"
        )

    export_manifest = {
        "schema": f"{BROWSER_SCHEMA_ID}.export-manifest",
        "schema_version": BROWSER_SCHEMA_VERSION,
        "source_bundle": str(bundle["dir"].relative_to(REPO)),
        "source_bundle_manifest_sha256": _sha256(
            bundle["dir"] / MANIFEST_NAME
        ),
        "emitted": {
            WEIGHTS_NAME: {"sha256": _sha256(weights_path),
                           "bytes": weights_path.stat().st_size},
            FIXTURE_COPY_NAME: {"sha256": fixture_sha,
                                "bytes": fixture_path.stat().st_size},
        },
        "probe_fixture_is_byte_identical_to_bundle": True,
        "deterministic": {
            "json_sort_keys": True,
            "timestamps_recorded": False,
            "float_serialization": (
                "shortest round-trip decimal of the exact float32 value "
                "(Python repr of float(np.float32))"
            ),
        },
    }
    _write_json(output / EXPORT_MANIFEST_NAME, export_manifest)

    for name, record in export_manifest["emitted"].items():
        print(f"{record['sha256']}  {name}  ({record['bytes']} bytes)")
    print(f"wrote {output}")
    return export_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bundle",
        required=True,
        help="v3 runtime bundle directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="fresh browser staging asset directory",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
