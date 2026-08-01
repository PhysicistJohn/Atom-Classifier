"""Worst-length + profile-balanced auxiliary adaptation for the v5 branch.

Motivation
----------
The v5 checkpoint score is bound by ``historical_worst_length_balanced_accuracy``
-- the *minimum* balanced accuracy across the three runtime prefixes
{4096, 8192, 16384}.  Training, however, draws exactly one prefix view per base
identity uniformly at random (``_sample_episode`` ->
``rng.choice(base_view_positions[base])``), so the episodic loss optimises the
*mean* over lengths while the metric grades the *min*.  Nothing in the v5 loss
applies pressure to a model's weakest length.

The same mismatch was already solved on the *scale* axis: the supervised-pair
loss reduces with ``maximum_over_exactly_four_contiguous_scale_view_losses``,
and the scale-sensitive checkpoint terms sit at 0.97-1.00.  This module applies
the identical worst-case reduction to the length axis.

A second, independent defect compounds it.  Episodes are *class*-balanced, then
round-robin over profiles within the class.  Historical profile multiplicity is
wildly uneven -- am/cw/fm/dsss have 1 profile each, gsm has 7, ofdm has 24 --
so a rare mode receives a small fraction of the exposure a singleton class
profile gets.  The observed worst cell is exactly such a mode:
``historical:gsm-32qam-higher-symbol-rate-burst`` at length 4096 (85 base
identities, recall 0.2778).

This auxiliary therefore draws *profile-balanced* over the 37 historical
training profiles, so every mode receives equal mass regardless of its class's
profile count.

Loss
----
Per episode, for ``profiles_per_episode`` profiles drawn uniformly without
replacement from the 37 historical training profiles:

    draw one base identity uniformly from that profile
    emit all three ordered prefix views (4096, 8192, 16384)
    per-view public-class cross-entropy against DETACHED episode prototypes
    reduce: maximum over the three length views      (worst-case, per identity)
    reduce: arithmetic mean over the drawn profiles  (mirrors the pair loss)

Detachment discipline matches ``_supervised_pair_cross_entropy_for_pairs``
exactly: prototypes and the logit scale are detached, batch-norm statistics are
frozen via ``_auxiliary_batch_norm_eval``, so this auxiliary may update the
encoder but can never move episodic prototype construction or the shared logit
scale.

The term is *additive*.  It removes no data, reweights no existing loss, and
narrows no evaluation -- it only adds gradient pressure on the views that are
currently failing.

Installation is by monkey-patch of
``run_scale_orbit_dev._supervised_pair_cross_entropy_for_pairs``, which receives
every tensor this auxiliary needs (net, data, prototypes, log_scale, device) and
is called once per episode.  The v5 runner file and its test suite are left
untouched.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

import run_scale_orbit_dev as _runner
from run_scale_orbit_dev import _auxiliary_batch_norm_eval, sq_dist

# Ordered runtime prefixes; base_view_positions stores historical views in this
# exact order because _view_lengths -> training_view_lengths filters
# RUNTIME_INPUT_LENGTHS in ascending order.
EXPECTED_LENGTHS = (4096, 8192, 16384)
VIEWS_PER_IDENTITY = len(EXPECTED_LENGTHS)

WITHIN_IDENTITY_REDUCTION = (
    "maximum_over_exactly_three_contiguous_runtime_prefix_view_losses"
)
ACROSS_PROFILE_REDUCTION = (
    "arithmetic_mean_over_drawn_profile_balanced_hard_losses"
)
LOSS_CONTRACT = (
    "episodic_cross_entropy_plus_0.2_times_mean_over_31_literal_profiles_of_"
    "maximum_over_all_four_contiguous_scale_view_supervised_public_class_cross_"
    "entropy_plus_worst_length_weight_times_mean_over_profile_balanced_"
    "historical_draws_of_maximum_over_three_contiguous_runtime_prefix_view_"
    "public_class_cross_entropy_against_detached_current_episode_source_mixed_"
    "prototypes_and_detached_logit_scale"
)
SAMPLING_CONTRACT = (
    "each episode draws profiles_per_episode profiles uniformly without "
    "replacement from the 37 historical training profiles, then one base "
    "identity uniformly within each drawn profile, then emits that identity's "
    "three ordered runtime-prefix views"
)

WORST_LENGTH_RNG_XOR = 0xC0FFEE

# Weight of the supervised-pair term in the v5 loss.  The patch composes with
# it, so the auxiliary is scaled by worst_length_weight / this value.
_PAIR_WEIGHT = 0.2

_original_pair_cross_entropy = _runner._supervised_pair_cross_entropy_for_pairs

_STATE: dict[str, Any] = {
    "weight": 0.0,
    "profiles_per_episode": 12,
    "seed": 0,
    "ramp_episodes": 0,
    "pool": None,
    "pool_key": None,
    "rng": None,
    "calls": 0,
}


def configure(
    *,
    weight: float,
    profiles_per_episode: int,
    seed: int,
    ramp_episodes: int = 0,
) -> None:
    """Set auxiliary hyperparameters and reset per-run sampling state.

    ``ramp_episodes`` linearly ramps the auxiliary coefficient from 0 to
    ``weight`` over that many episodes.  A worst-case (maximum) reduction is
    high-variance while the encoder is still poor at every length, so applying
    it at full strength from episode 0 can dominate the episodic term and
    destabilise early training.  Ramping is standard practice for hard-example
    auxiliaries and costs nothing once the ramp completes.
    """
    weight_value = float(weight)
    if not np.isfinite(weight_value) or weight_value < 0.0:
        raise ValueError("worst_length_weight must be finite and non-negative")
    count = int(profiles_per_episode)
    if count <= 0:
        raise ValueError("worst_length_profiles must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    ramp = int(ramp_episodes)
    if ramp < 0:
        raise ValueError("ramp_episodes must be non-negative")
    _STATE.update(
        weight=weight_value,
        profiles_per_episode=count,
        seed=int(seed),
        ramp_episodes=ramp,
        pool=None,
        pool_key=None,
        rng=None,
        calls=0,
    )


def _current_weight() -> float:
    """Weight after applying the linear ramp for the current episode."""
    weight = float(_STATE["weight"])
    ramp = int(_STATE["ramp_episodes"])
    if ramp <= 0:
        return weight
    progress = min(1.0, float(_STATE["calls"]) / float(ramp))
    return weight * progress


def _build_pool(data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return (profile_view_table, profile_class_indices) for historical modes.

    ``profile_view_table`` is a ragged-free object array: one entry per
    historical profile holding an ``[n_bases, 3]`` int64 array of view
    positions.  ``profile_class_indices`` holds each profile's public class.
    """
    hierarchy = data["base_sampling_hierarchy"]
    base_view_positions = data["base_view_positions"]
    tables: list[np.ndarray] = []
    class_indices: list[int] = []
    for class_index, class_groups in enumerate(hierarchy):
        historical = class_groups.get("historical")
        if not historical:
            continue
        for _profile, bases in sorted(historical.items()):
            rows: list[list[int]] = []
            for base in np.asarray(bases, dtype=np.int64).tolist():
                views = np.asarray(base_view_positions[int(base)], dtype=np.int64)
                # Historical identities carry exactly one view per runtime
                # prefix.  Anything else is a repeated stored materialisation
                # and is skipped rather than silently mis-paired.
                if views.shape != (VIEWS_PER_IDENTITY,):
                    continue
                if len(set(views.tolist())) != VIEWS_PER_IDENTITY:
                    continue
                rows.append([int(value) for value in views.tolist()])
            if not rows:
                continue
            tables.append(np.asarray(rows, dtype=np.int64))
            class_indices.append(int(class_index))
    if not tables:
        raise ValueError(
            "worst-length auxiliary found no eligible historical profiles"
        )
    table_array = np.empty(len(tables), dtype=object)
    for index, value in enumerate(tables):
        table_array[index] = value
    return table_array, np.asarray(class_indices, dtype=np.int64)


def _ensure_pool(data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    key = id(data)
    if _STATE["pool"] is None or _STATE["pool_key"] != key:
        _STATE["pool"] = _build_pool(data)
        _STATE["pool_key"] = key
        _STATE["rng"] = np.random.default_rng(
            int(_STATE["seed"]) ^ WORST_LENGTH_RNG_XOR
        )
    return _STATE["pool"]


def worst_length_cross_entropy(
    net: Any,
    data: Mapping[str, Any],
    prototypes: torch.Tensor,
    log_scale: torch.Tensor,
    device: torch.device,
    *,
    phase_augmentation: bool,
) -> torch.Tensor:
    """Mean over profile-balanced draws of max CE across the three prefixes."""
    if phase_augmentation is not True:
        raise ValueError(
            "worst-length auxiliary requires phase augmentation enabled"
        )
    tables, class_indices = _ensure_pool(data)
    rng: np.random.Generator = _STATE["rng"]
    n_profiles = len(tables)
    draw = min(int(_STATE["profiles_per_episode"]), n_profiles)
    chosen = rng.choice(n_profiles, size=draw, replace=False)

    view_rows: list[np.ndarray] = []
    targets: list[int] = []
    for profile_index in chosen.tolist():
        table = tables[int(profile_index)]
        base_row = int(rng.integers(0, table.shape[0]))
        view_rows.append(table[base_row])
        targets.append(int(class_indices[int(profile_index)]))

    positions = np.stack(view_rows).reshape(-1)
    target_array = np.repeat(
        np.asarray(targets, dtype=np.int64), VIEWS_PER_IDENTITY
    )
    if positions.shape != (draw * VIEWS_PER_IDENTITY,):
        raise AssertionError("worst-length view flattening changed shape")
    if np.any(positions < 0) or np.any(positions >= len(data["xtr"])):
        raise ValueError("worst-length position outside aligned training views")

    x = torch.from_numpy(np.asarray(data["xtr"])[positions]).to(device)
    features = torch.from_numpy(np.asarray(data["ftr"])[positions]).to(device)
    x = _runner.v3_runner._phase_augment(x)
    with _auxiliary_batch_norm_eval(net):
        embeddings = net(x, features)
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != draw * VIEWS_PER_IDENTITY
        or not torch.isfinite(embeddings).all()
    ):
        raise ValueError("worst-length embeddings must be finite [views, dim]")

    # Same detachment discipline as the supervised-pair auxiliary: the encoder
    # may learn, prototypes and logit scale may not.
    logits = (
        -sq_dist(embeddings, prototypes.detach())
        * log_scale.detach().exp().clamp(1e-3, 100.0)
    )
    per_view = F.cross_entropy(
        logits,
        torch.from_numpy(target_array).to(device),
        label_smoothing=0.0,
        reduction="none",
    )
    per_identity = per_view.reshape(draw, VIEWS_PER_IDENTITY)
    hardest_length = per_identity.max(dim=1).values
    return hardest_length.mean()


def _patched_pair_cross_entropy(
    net: Any,
    data: Mapping[str, Any],
    pair_positions: np.ndarray,
    profile_class_indices: np.ndarray,
    prototypes: torch.Tensor,
    log_scale: torch.Tensor,
    device: torch.device,
    *,
    phase_augmentation: bool,
) -> torch.Tensor:
    """Original scale-axis pair loss plus the length-axis worst-case loss.

    The v5 trainer multiplies this return value by the frozen supervised-pair
    weight (0.2), so the auxiliary is pre-divided by that weight to make
    ``worst_length_weight`` an absolute coefficient on the total loss.
    """
    base = _original_pair_cross_entropy(
        net,
        data,
        pair_positions,
        profile_class_indices,
        prototypes,
        log_scale,
        device,
        phase_augmentation=phase_augmentation,
    )
    if float(_STATE["weight"]) <= 0.0:
        return base
    weight = _current_weight()
    auxiliary = worst_length_cross_entropy(
        net,
        data,
        prototypes,
        log_scale,
        device,
        phase_augmentation=phase_augmentation,
    )
    _STATE["calls"] += 1
    if weight <= 0.0:
        # Still keep the graph contribution at exactly zero during the ramp's
        # first episode rather than skipping the forward pass, so the sampling
        # RNG stream is identical regardless of ramp length.
        return base + 0.0 * auxiliary
    return base + (weight / _PAIR_WEIGHT) * auxiliary


def install() -> None:
    """Route the v5 trainer's auxiliary through the worst-length patch."""
    _runner._supervised_pair_cross_entropy_for_pairs = (
        _patched_pair_cross_entropy
    )


def uninstall() -> None:
    """Restore the unmodified v5 auxiliary."""
    _runner._supervised_pair_cross_entropy_for_pairs = (
        _original_pair_cross_entropy
    )


def metadata() -> dict[str, Any]:
    """Auditable description of the installed auxiliary."""
    pool = _STATE["pool"]
    return {
        "schema": "v5-worst-length-profile-balanced-auxiliary-v1",
        "weight": float(_STATE["weight"]),
        "ramp_episodes": int(_STATE["ramp_episodes"]),
        "final_applied_weight": _current_weight(),
        "profiles_per_episode": int(_STATE["profiles_per_episode"]),
        "historical_profiles_in_pool": (
            int(len(pool[0])) if pool is not None else None
        ),
        "views_per_identity": VIEWS_PER_IDENTITY,
        "expected_prefix_order": list(EXPECTED_LENGTHS),
        "within_identity_reduction": WITHIN_IDENTITY_REDUCTION,
        "across_profile_reduction": ACROSS_PROFILE_REDUCTION,
        "sampling_contract": SAMPLING_CONTRACT,
        "loss_contract": LOSS_CONTRACT,
        "rng_seed_rule": "seed xor 0xC0FFEE",
        "rng_separate_from_episodic_and_pair_rng": True,
        "prototypes": "detached_current_episode_source_mixed_public_class",
        "logit_scale": "detached_numeric_current_episode_logit_scale",
        "batch_norm_stat_firewall": True,
        "episodes_with_auxiliary_applied": int(_STATE["calls"]),
    }
