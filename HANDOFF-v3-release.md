# Atom-Classifier handoff

Written 2026-07-28 at commit `644218b`, branch `ship/corrected-corpus-classifier`
(pushed to `origin`). Repo: `/Users/johnelliott/PersonalGitHub/Atom-Classifier`.
Run `pwd` before doing anything — an environment banner has previously named the
wrong directory.

The full session-by-session record, including every correction and retraction,
is preserved in `HANDOFF-HISTORY.md` (1,751 lines) and in the git log. This
document is the current state only. Where they disagree, this document wins.

---

## 1. Where this stands, in five sentences

A v3 length- and scale-invariant RF classifier exists, is fully ported to the
browser with Python-anchored parity, and beats the frozen v2 model on every
identically-measured axis. Three one-shot sealed release runs have been spent
(seeds 20260731, 20260733, 20260734); each scored 21–22 of 23 gates, and every
axis that sank the v2 release has passed on all three independent draws. The
sole remaining failure is five-shot balanced accuracy at N4096, a high-variance
statistic whose three sealed draws span 0.823–0.850 around a 0.84 floor. The
defined next step (v3.3) swaps in the already-trained 8k-regularised fusion
weights, which raise five-shot genuinely, then spends seed 20260735. **The
owner has not yet authorized seed 20260735 — do not spend it without an
explicit go.**

## 2. Non-negotiable rules

1. **Fit on TRAINING only. Thresholds/ranks on ENROLLMENT only. Selection is
   scored, never fit.** The historical corrected test half is consumed.
2. **Sealed suites are one-shot.** Seeds 20260729 (v2), 20260731, 20260733,
   20260734 are consumed. Never re-run, tune against, or debug on them. Their
   `RELEASE_EVALUATION.json` files are immutable evidence.
3. **Do not weaken a gate after seeing a result.** One owner redeclaration
   exists (§5); it was made provably before generation. No second redeclaration
   of five-shot — that decision is recorded and stands.
4. **Novelty seed ledger** (all in code, enforced by refusals):
   spent: 20260938–41 (design), 20260942/3 (validated v3.0), 20260944/5
   (validated v3.1, a frozen FAIL), 20260946 (v3.2 design), 20260947/8 (v3.2
   validation). Clean: **20260949+**. Fitting-only band: 20261000–1999
   (20261001 used).
5. **Release seeds:** next unused is **20260735**. Note 20260730/20260732 are
   dev *model* seeds — never use them as release seeds (namespace collision).
6. Do not claim field/SOTA performance. Evidence is synthetic, internal, vs a
   fresh same-data incumbent. No real SDR captures exist in this repo.

## 3. The candidate (v3.2, exactly what sealed runs #2 and #3 measured)

All under `training/zplane_ab/v2_full_variation/` (`V2/`) and its
`artifacts/invariant_patch/v3_scale/` (`V3/`).

| component | location |
|---|---|
| fusion (4k multilength, both branches) | `V3/v3_fusion_multilength_seed20260730` |
| runtime bundle (closed-set, self-verified 0-error) | `V3/v3_runtime_bundle_seed20260730` — manifest sha `bfd972bf…` |
| stage-1 noise prefilter (0.01 budget, 7-coef logistic/length) | `V3/noise_prefilter_fit20261001_budget001/bundles` |
| stage-2 + composite policy (frozen, q95 on composite) | `V3/staged_validate_composite_budget001_seed20260730` |

Pipeline: raw I/Q → FFT-free autocorrelation pose estimate → dimensionless
resample to 16×64 patches → **stage-1 noise gate** (pose-degeneracy features;
captures longer than 16384 are gated on their first 16384 samples — the
causal-prefix rule) → real + complex patch CNNs (52,896 + 55,936 params) →
centered 0.5/0.5 fusion → nearest prototype (7 classes) → **stage-2**
composite open-set score `max(stage2_rank, stage1_score_rank)`, threshold q95
on enrollment survivors. Architecture contract: the rejector **gates before
classification** (`additive_only: false`) — recorded machine-readably in every
artifact and required in any release claim.

## 4. The three sealed runs

| seed | result | failing gate(s) | values |
|---|---|---|---|
| 20260731 (v3.0 policy) | 22/23 | known FUR | 0.1024 vs 0.10 |
| 20260733 (v3.2 composite) | 21/23 | FUR, five-shot | 0.1108 vs 0.10; 0.8449 vs 0.85 |
| 20260734 (v3.2, redeclared gates) | 22/23 | five-shot | 0.8228 vs 0.84 (FUR **passed**: 0.1070 vs 0.12) |

Established across all three draws: physical-scale balanced/cosine/agreement,
chirp AUROC + recall, noise AUROC + recall, N4096 clean, closed-set, length
invariance gates all PASS — i.e. everything that failed the v2 sealed release
is fixed and replicated. Also measured: enrollment-calibrated operating points
realise ~1.5–1.7× their dev rates on sealed populations (both policies, both
runs) — budget for this when setting any operating point.

Five-shot sealed draws: 0.8503, 0.8449, 0.8228 → mean 0.839, sd ≈ 0.014. It is
a high-variance **measurement** (5-row support draws at the shortest prefix),
not a stable model property; the same model scored 0.866–0.875 at the other
three lengths in run #3. Do not estimate its "true rate" from fewer than
several draws — that error was made and is documented in
`HANDOFF-HISTORY.md` §29.

## 5. The owner gate redeclaration (already in force)

Declared by the owner 2026-07-28, provably before seed 20260734 was generated
(the evaluator refuses an intent lacking the block):

- `open_known_false_unknown_worst_length`: ceiling **0.10 → 0.12**
- `five_shot_worst_length_balanced`: floor **0.85 → 0.84**

Implemented as `V3_GATE_REDECLARATION` in
`V2/v3_scale/evaluate_v3_release_suite.py`, layered on the still-imported v2
`GATE_FLOORS` so the other 15 gates cannot drift. Each re-levelled gate carries
the owner rationale verbatim. **Any release claim must state both levels and
the v2 levels they replace.** These levels stand; do not touch them.

## 6. Next step: the v3.3 cycle (defined, not started, awaiting owner go)

Goal: raise five-shot genuinely instead of moving the bar.

1. **Weights:** swap the fusion branches to the 8k-regularised runs
   (`V3/timecorr_{real,complex}_ml8000reg_seed20260730` — dropout 0.35, wd
   5e-4). Measured on dev: five-shot +0.012, closed +0.017 vs current weights;
   all six open-set gates passed dev under the old policy. Assemble with
   `V2/v3_scale/assemble_v3_fusion.py`, neutral `--branch-weight 0.5`.
2. **Re-validate the composite policy** on the new fusion:
   `V2/v3_scale/fit_v3_openset_staged.py --role design` on seed **20260949**,
   then a single `--role validate` on **20260950 20260951** (prefix lengths
   4096 8192 16384 32768). All gates must pass.
3. **Export a fresh runtime bundle** (`export_v3_fusion_runtime.py`) and
   re-run `measure_v3_remaining_gates.py` for the dev closed-set numbers.
4. **Refresh preflight pins** (`preflight_v3_release.py`): new candidate
   hashes, new staged artifact, launcher fixture for seed 20260735 via
   `evaluate_v3_release_suite.py --print-expected-protocol 20260735`, launcher
   consumed-seed map gains 20260734. Preflight must print GO.
5. **Ask the owner**, then spend seed 20260735: generation command is printed
   by preflight (env `RELEASE_EVALUATION_PROTOCOL=v3`,
   `RELEASE_TARGET_PER_CLASS=192`, isolated `SIGNALLAB_ROOT` below), then
   `evaluate_v3_release_suite.py --release-root … --device cpu`, once.

Expected margins if dev numbers transfer at the measured 1.5–1.7× shift:
five-shot mean ≈ 0.855–0.86 vs floor 0.84; FUR ≈ 0.10–0.11 vs ceiling 0.12.

## 7. After a passing sealed run: the ship path (all mapped, nothing hidden)

1. The full v3 TS runtime already exists and is green:
   `src/embedding/time-domain-{geometry,invariant-patch-preprocess,encoder,fusion,classifier,openset}-v3.ts`,
   216+ tests, parity vs Python fixtures at ≤1.8e-7 (tolerance 1e-6). Browser
   weights staging: `src/embedding/assets-v3-staging/` (+ the openset staging
   under `V2/artifacts/staging/time_domain_v3_openset/`). A v3.3 pass requires
   regenerating the weight JSONs and parity fixtures from the new bundle
   (exporters exist: `export_v3_browser_weights.py`,
   `export_v3_openset_browser_assets.py`).
2. **Deploy mechanism (verified via wrangler):** the live site is the
   `atomizer` Cloudflare Worker (`atomizer.radio-lab.app` /
   `signal.radio-lab.app`), account `0883c2d43b859db59430203a69b6707a`.
   Atomizer's renderer dynamic-imports this repo **by relative path** —
   `../Atom-Atomizer/apps/desktop/src/renderer/embedding-classifier-runtime.ts`
   lines ~118–136 import `Atom-Classifier/src/embedding/index.js` and
   `src/embedding/assets/*.json` — and the build bakes them into the worker
   bundle. Shipping = point that file at the v3 runtime + assets, rebuild
   Atomizer, `wrangler deploy`.
3. `src/embedding/assets/` is the LIVE v2 model
   (`embedding-weights.json` sha starts `4c566a17`) — untouched all session.
   **Never write there** until a sealed run passes and the owner says ship.

## 8. Footguns (each has already cost time or nearly cost data)

1. `tools/generate-signallab-iq-corpus.ts` **truncates 3.67 GB at import**
   (top-level `openSync(…, 'w')`, no main guard). The dev corpora are
   `chmod a-w` as protection. Release generation does NOT use it directly —
   use the launcher only.
2. `training/train_signallab.py` writes straight into the **live**
   `src/embedding/assets/`. Never run it directly.
3. MPS cannot backward through boolean-masked complex tensors; mask the real
   loss instead.
4. Apple Accelerate numpy raises **spurious IEEE flags on clean float64
   matmuls**; under the house `PYTHONWARNINGS=error` this aborts healthy code.
   Use the `noise_prefilter._dot` pattern (narrow `np.errstate` + explicit
   finiteness check) for any new linear algebra.
5. The isolated release source is a git **archive** (no `.git`) at
   `/private/tmp/atomos-release-source-final.RvbW7V/Atom-SignalLab` — a temp
   path; preflight verifies its digests. If it vanishes, rebuild from the
   pinned commits in `HANDOFF-HISTORY.md` §6.
6. Node pins: v22.23.1, npm/npx 10.9.8, PATH
   `/Users/johnelliott/.nvm/versions/node/v22.23.1/bin`. Python:
   `.venv-training/bin/python`, always `PYTHONWARNINGS=error`, PYTHONPATH
   `training:training/zplane_ab:training/zplane_ab/v2_full_variation:training/zplane_ab/v2_full_variation/v3_scale`.

## 9. Measurement traps that burned this project (condensed; full list in HISTORY)

- **A hard-coded constant smuggles the training distribution into an
  "independent" probe.** Sweep every constant an OOD test hard-codes.
- **An absent measurement that still writes a file looks finished.** Check the
  value exists, not just the file (an `isinstance` fallthrough once nulled the
  primary metric of an entire campaign while writing 17 KB reports).
- **Cross-budget and cross-protocol comparisons are confounds.** 700-vs-4000
  episode and dev-vs-sealed comparisons each produced a false conclusion here.
- **Never rank on a statistic that floors** (worst-of hid the best length
  response in the project).
- **Flat-at-chance is not invariance** — any invariance test needs a
  matched-condition validity floor.
- **n=2 is not an error bar** (see five-shot, §4).
- **Read the control before the headline**: absolute tap scores once pointed
  at exactly the wrong attachment layer because two taps were handed the
  answer as input.

## 10. Verification quick-reference

```bash
# Everything Python (currently 725/725):
cd /Users/johnelliott/PersonalGitHub/Atom-Classifier && \
PYTHONWARNINGS=error PYTHONPATH=training:training/zplane_ab:training/zplane_ab/v2_full_variation:training/zplane_ab/v2_full_variation/v3_scale \
.venv-training/bin/python -m unittest discover -s training/zplane_ab/v2_full_variation/v3_scale
```

```bash
# TypeScript (currently 216+ passing):
cd /Users/johnelliott/PersonalGitHub/Atom-Classifier && \
PATH=/Users/johnelliott/.nvm/versions/node/v22.23.1/bin:$PATH npx vitest run
```

```bash
# Preflight (GO/NO-GO, prints the generation command; never spends a seed):
cd /Users/johnelliott/PersonalGitHub/Atom-Classifier && \
PYTHONWARNINGS=error PYTHONPATH=training:training/zplane_ab:training/zplane_ab/v2_full_variation:training/zplane_ab/v2_full_variation/v3_scale \
.venv-training/bin/python training/zplane_ab/v2_full_variation/v3_scale/preflight_v3_release.py
```

Note: preflight currently pins the **v3.2** candidate and the seed-20260734
fixture; step 4 of §6 re-pins it for v3.3. It will correctly report NO-GO for
a v3.3 candidate until then.

## 11. One sentence of orientation

The science is done and replicated; what remains is one high-variance statistic
to clear with honestly better weights, one seed spend the owner must authorize,
and a mapped, mechanical ship path — resist any shortcut that trades the
evidence discipline for speed, because every failure in HISTORY came from
exactly that trade.
