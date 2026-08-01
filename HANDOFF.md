# Atom-Classifier handoff — v7 / DACS line

**Written:** 2026-07-31, late evening. Repo:
`/Users/johnelliott/PersonalGitHub/Atom-Classifier`, branch
`ship/corrected-corpus-classifier`. Run `pwd` before doing anything — an
environment banner has previously named the wrong directory.

## 2026-08-01 takeover addendum — diagnostic proof complete, not promotable

The original handoff below is historical context. Current evidence is:

- All 17 class-C profiles now have isolated seeded corpus-content paths. A
  real 34-profile × 8-row × 20 ms stage-1 diagnostic passed with 272 rows,
  936,678,400 bytes, no content deficits, and 8/8 seeds, fixed-phase probes,
  realizations, and row hashes for every class-C profile. It is preserved at
  `training/artifacts/longdwell-corpus-v2-diagnostic-34x8-20260801/`.
- The proof was generated twice from the same source state. The native capture
  is byte-identical both times (`c6d23de...`), and the manifests match after
  excluding only `generatedAt` and `elapsedSeconds`.
- `MIN_ACTIVE_SAMPLES=0` is deliberate: natural 20 ms Bluetooth LE silence is
  retained and reported rather than redrawn away. The proof has 4 zero LE
  windows and 5 distinct raw LE hashes; the verifier permits this one
  unconditioned case and rejects silent rows elsewhere or in a conditioned
  run.
- The proof records both repository revisions and `workingTreeClean: false`.
  It is diagnostic only: do not stage 2, train, or call it reproducible until
  both repositories have owner-authorized clean commits and a fresh proof has
  been generated from those commits.
- Full validation passed before this addendum: Atom-SignalLab 79 test files /
  614 tests (12 skipped), Atom-Classifier 38 files / 341 tests (8 skipped),
  both typechecks, the focused content suite, and the verifier's synthetic
  allowed/rejected-silence cases. The G4 checkpoint was re-evaluated safely
  into a separate artifact and matches the corrected five-seed metrics.
- `Atom-SignalLab` `stash@{0}` remains intact and its tracked patch matches
  the current restored tracked patch byte-for-byte. Do not reset, clean, or
  drop that stash.

No production regeneration, stage 2, or training was run during this
takeover. The associated source and documentation changes were later committed
only after owner authorization.

**Read this whole file before touching anything.** Several mistakes
documented here cost hours today and are easy to repeat.

---

## ⚠ FIRST FIVE MINUTES — do these before anything else

**1. NOTHING FROM THE v7 LINE IS COMMITTED.** As of this handoff,
`git status` shows the entire v7 workstream as *untracked*:
`training/zplane_ab/v2_full_variation/v7_probe/` (the MLX trainer, model,
parity harnesses, torch trainer edits), `docs/mlx-port-plan.md`,
`docs/paper/`, the `tools/` corpus scripts, and the v6 harnesses under
`v5_scale_orbit/`. Worse, **`training/artifacts/` is gitignored**
(`.gitignore:18`), so every checkpoint and all evidence in
`training/artifacts/mlx-g2/` — including `g4_bf16.safetensors`, the only
complete trained model — exists **only as working-tree files with no git
history**. A `git clean -fd` or a careless checkout erases a full day of
work and the only good checkpoint.
→ **Ask the owner whether to commit the code**, and until then do not run
any destructive git command. Consider copying `training/artifacts/mlx-g2/`
(≈40 MB excluding the corpus) somewhere safe first.

**2. Verify the machine state matches this document:**
```bash
cd /Users/johnelliott/PersonalGitHub/Atom-Classifier && pwd && \
  ls training/artifacts/mlx-g2/g4_bf16.safetensors && \
  du -sh training/artifacts/longdwell-production-corpus && \
  ps aux | grep "[P]ython" | grep -v Claude
```
Expect: the checkpoint present, an 80 GB corpus, and — depending on timing —
possibly a corpus-generation process still running (see §6).

**3. Confirm the honest numbers reproduce** (≈2 min, no training):
```bash
.venv-training/bin/python training/artifacts/mlx-g2/g5_offset_reeval.py
```
Should land on the §4 table. If it does not, stop and find out why before
building on anything here.

### Two workstreams exist in this repo — do not confuse them

| line | doc | state |
|---|---|---|
| **v3 release** (browser-shipped invariant-patch classifier, sealed release seeds) | **`HANDOFF-v3-release.md`** | paused awaiting an owner-authorized seed spend |
| **v7 / DACS** (this document) | this file | active |

**⚠ The v3 non-negotiables still bind.** In particular: sealed release suites
are one-shot; seeds 20260729/20260731/20260733/20260734 are consumed; the
next release seed 20260735 **must not be spent without explicit owner
authorization**; never write into `src/embedding/assets/` (that is the LIVE
v2 browser model). Read `HANDOFF-v3-release.md` §2 before any release-shaped
action. The v7 work below does not touch those artifacts.

Deep history for the v3 line: `HANDOFF-HISTORY.md` (1,751 lines).

---

## 0. TL;DR — v7 state in five lines

1. The v7 model (DACS) works. Best honest measurement: **balanced accuracy
   0.902 / 0.950 / 0.995** at 1 / 2.5 / 10 ms dwell; min-cell recall
   **0.223 / 0.583 / 0.979** (mean of 5 eval seeds, full 3,264-row split).
2. Training moved from **PyTorch-MPS to MLX**. Validated through gate G4
   (G0/G1/G2/G2b/G4 all pass). ~2.3× faster than fp32 torch, and unlike
   torch it neither deadlocks nor thrashes. **bf16 is the production config.**
3. A **measurement bug** was found and fixed: the eval protocol aliased the
   GSM burst raster. Every number produced before the fix — including all of
   the paper's — is optimistic.
4. A **corpus defect** was then found: 22 of 34 profiles contain exactly ONE
   bit-identical clean waveform across all rows. **This is the current top
   priority** and the reason a regeneration is planned.
5. The paper is typeset but its Experiments numbers are contaminated and
   unreproducible. The owner explicitly deprioritized it: *"Forget about the
   paper — do what is right to train the model."*

---

## 1. The v7 project

**Repos:** `Atom-Classifier` (model, training, corpora, paper) and
`Atom-SignalLab` (TypeScript standards-derived RF waveform synthesis).

**Task:** classify raw I/Q into 7 classes (`am, bluetooth, cw, dsss, fm,
gsm, ofdm`) spanning 34 waveform configurations ("profiles").

**The scientific core — don't lose it:** burst protocols are defined by
rhythms measured in *milliseconds* (GSM's 4.615 ms TDMA frame, Bluetooth's
625 µs slots, BLE's ~30 ms advertising grid), while classifiers are
conventionally trained on windows fixed in *samples*, spanning microseconds.
The window physically cannot contain the discriminating structure. Diagnosed
as an **evidence problem, not a modeling problem**: with the model frozen,
duration-complete data moved 7-class balanced accuracy 0.503 → 0.954 at 1 ms;
with the data frozen, no model-side lever (architecture swap, 3× budget,
worst-case reweighting) came close.

The owner's framing, which shaped the design: *"Time is just another axis —
empty slices are not a mistake."* Silence between bursts is class-conditional
evidence, not noise to discard. Hence the paper's line: **"sometimes the
signal is the noise."**

**DACS** (Dwell-Adaptive Classification from duration-complete Synthesis):
one weight-shared, fully-convolutional-in-time network reading a capture at
**any** dwell, with three heads:

- **prototype head** — knows the class (episodic; keeps enrollment/open-set
  viable)
- **masked denoising autoencoder** — knows the channel. Target is the *exact
  clean reference* from paired synthesis (not a corruption of the input), so
  silences are graded targets: the net must know where the rhythm's rests fall.
- **confidence head** — knows itself (BCE vs realized correctness), driving
  the dwell policy: answer at 1 ms if confident, else escalate 2.5 → 10 ms.

---

## 2. Where everything lives

### Code
| Path | What |
|---|---|
| `training/zplane_ab/v2_full_variation/v7_probe/v7_trainer.py` | torch-MPS trainer (reference for semantics) |
| `.../v7_probe/v7_trainer_mlx.py` | **MLX trainer — production** |
| `.../v7_probe/v7_model_mlx.py` | MLX model + primitives (1,573,028 params, matches torch) |
| `.../v7_probe/mlx_parity/convert.py` | torch `.pt` ↔ MLX safetensors, bitwise-verified both ways |
| `.../v7_probe/mlx_parity/test_unit_parity.py` | G0 harness (32 checks) |
| `.../v7_probe/mlx_parity/test_forward_parity.py` | G1 harness (13 checks) |
| `.../v7_probe/mlx_parity/gen_torch_fixtures.py` + `fixtures/` | torch ground truth for G0 |
| `.../v7_probe/eval_confusion.py` | torch-side confusion matrices (not yet ported; usable via convert.py) |
| `tools/generate-longdwell-probe-corpus.mjs` | corpus stage 1 (native clean synthesis via SignalLab) |
| `tools/longdwell_probe_stage2.py` | corpus stage 2 (resample to 20 Msps + seeded impairments) |
| `tools/production_corpus_plan.json` | the 34-profile plan |

### Data
`training/artifacts/longdwell-production-corpus/` — **80 GB**. 8,704 rows
(34 profiles × 256), each 400,000 samples = 20 ms at 20 Msps. `noisy.npy` +
`clean.npy` are row-aligned pairs; `manifest.json` records per-row role,
class, profile, `startSampleIndex`, and the impairments actually applied.
Split: 5,440 train / 3,264 eval (96 eval rows per profile).
**Known defect — see §6.**

### Checkpoints
- `training/artifacts/mlx-g2/g4_bf16.safetensors` — **the good one.** MLX
  bf16, episode 3000, complete run. Sidecar `.json` carries config, episode,
  both RNG `bit_generator` states, optimizer state, eval history, param hash.
- `training/artifacts/mlx-g2/step0.{safetensors,pt}` — the torch step-0 init,
  exported so MLX runs start from *identical* weights.
- `.../v7_probe/v7_ab_recon_on.ep{1000,2000}.pt` — torch, abandoned A/B run.

### Evidence
- `training/artifacts/mlx-g2/` — the whole night: G2 four-config matrix
  (`g2_mlx_{a,b,c,d}.json`), G4 result + log, confusion matrices
  (`g4_confusion.{json,png,py}`), the aliasing investigation
  (`subsample_bias_test.{py,json}`), the corrected re-eval
  (`g5_offset_reeval.{py,json,log}`), memory probes, all driver scripts.
- `docs/mlx-port-plan.md` — reviewed port plan, gates, and a **differences
  ledger** (every way MLX deliberately does not match torch).
- `docs/paper/` — paper, figures, versioned PDFs (v6…v12).

---

## 3. The MLX port

### Why we left PyTorch-MPS
Two reproducible failures, both diagnosed by stack-sampling:

1. **Deterministic deadlock.** 2 of 2 runs wedged in the *first training step
   after an eval*, 100% of samples in
   `MPSStream::copy_and_sync → waitUntilCompleted`. Mitigated (not cured) by
   `gc.collect()` + `torch.mps.synchronize()` + `torch.mps.empty_cache()`
   after each eval — that patch is in `v7_trainer.py` and did stop it.
2. **Throughput collapse.** The 11 GB training RAM cache kept being
   compressed out by macOS (episodes touch ~1% of it, so it "looks cold"),
   taking pace 0.6 → 12 s/ep. Mitigated with `mlock` (also in
   `v7_trainer.py`). Even then torch degraded, because it does synchronous
   host↔GPU copies per episode — **the copy layer MLX does not have.**

### Gate results
- **G0 — 32/32.** Spectrogram ~5e-6, GroupNorm 1.4e-6, cdist², losses, AdamW
  5-step trajectory 6e-8, full 3000-step realized-LR sequence to 3e-10
  (including across save/restore).
- **G1 — 13/13.** On the real ep-2000 checkpoint: **100.0000% argmax
  agreement** over all 3,264 eval rows, confusion matrices cell-for-cell
  identical, converter bitwise on all 44 tensors.
- **G2 — precision matrix** (300 episodes each, full corpus):

  | config | avg ms/ep | vs fp32 | accuracy drift | skipped steps |
  |---|---|---|---|---|
  | fp32-strict (`--no-tf32`) | 2,184 | 1.00× | anchor | 0 |
  | TF32 (default) | 1,726 | 1.27× | ≤0.007 | 0 |
  | **bf16 (`--dtype bf16`)** | **946** | **2.31×** | ≤0.011 | **0** |
  | fp16 (`--dtype fp16`) | 944 | — | **never trained** | **299/300** |

  fp16 produces non-finite losses immediately; the `mx.isfinite` guard
  skipped essentially every step (0.143 accuracy = 1/7 = untrained). bf16 is
  the same speed with fp32-range exponent. **Use bf16, never fp16.**
- **G2b — resume parity, byte-exact.** 200 eps + checkpoint + resume + 100
  yields a draw-tuple stream byte-identical to a straight 300-episode run.
- **G4 — full 3,000-episode replication.** 53.5 min, 1,032 ms/ep, 0 skipped
  steps, 30.8 GB peak. **The first complete v7 training run that has ever
  finished** (torch never got past ep 2000).
- **G5 — outstanding.** Port `eval_confusion.py` (or keep it on torch via
  `convert.py`) and wire the Optuna harness to the MLX trainer.

### MLX operational facts you must know
- **TF32 is ON by default on this M5 Max** and silently degrades fp32
  matmuls (2.7e-2 vs fp64 truth, versus 4.6e-6 with `MLX_ENABLE_TF32=0`).
  Production keeps it on **by explicit owner directive** (21% faster,
  validated at gate level). **Parity harnesses must set
  `MLX_ENABLE_TF32=0` before importing mlx**, or they cannot distinguish a
  porting bug from precision noise. `--no-tf32` exists for bisection.
- **Memory.** The T=6249 (10 ms) training signature originally peaked at
  58.8 GB and swapped; fixed to **25.06 GB** with `--ckpt-blocks 3
  --batch-chunks 5` (activation checkpointing on the first 3 encoder blocks
  + 5-way batch chunking). Both are mathematically exact — loss bit-identical
  before/after (2.7324 both). **Always pass these for full-corpus runs.**
- **Compile:** exactly **6** signatures (3 dwells × aux/no-aux). Warmup runs
  on a deep-copied throwaway state, then asserts the real params' hash and
  optimizer step counters are unchanged before episode 0. A 7th signature
  aborts the run.
- The corpus is held resident as MLX arrays, wired with
  `mx.set_wired_limit(~22 GB)` (corpus only, **not** activations) — this
  replaces `mlock`. `set_memory_limit` throttles, it does **not** raise; the
  leak guard is an explicit watchdog against the measured warmup peak.

---

## 4. THE HONEST NUMBERS (use these, not the paper's)

`g4_bf16.safetensors` (MLX bf16, ep 3000), full 3,264-row split, corrected
protocol, **mean of 5 offset seeds** (`mlx-g2/g5_offset_reeval.json`):

| dwell | balanced accuracy | min-cell recall | errors (of 3264) |
|---|---|---|---|
| 1 ms | **0.9020 ± 0.0026** | **0.2229 ± 0.0281** | ~494 |
| 2.5 ms | **0.9500 ± 0.0024** | **0.5833 ± 0.0338** | ~238 |
| 10 ms | **0.9947 ± 0.0005** | **0.9792 ± 0.0000** | ~11 |

Escalation at 1 ms: answers 100% of captures at accuracy **0.849**. The
confidence head is **uncalibrated** — a known, disclosed gap. Calibration on
held-out data is required before the escalation *curve* can be claimed; only
the *mechanism* is claimed.

**Per-class recall, 1 ms:** am 1.000, bluetooth 0.994, cw 0.988, dsss 0.990,
fm 0.983, **gsm 0.404**, ofdm 0.956. GSM is the whole story — at 1 ms the
burst is genuinely absent most of the time.

**Confusion structure** (`g4_confusion.png`): at 1 ms, **377 of 672 GSM rows
are called Bluetooth** — exactly the predicted failure (frame timing vs hop
timing, both invisible in a sub-millisecond window). At 10 ms that confusion
is **zero**.

**Error nesting — the cleanest escalation evidence:** errors 404 → 210 → 9
across dwells; 10 ms **fixes 396** of the 1 ms errors and **creates 1**.
Jaccard(E₁ₘₛ, E₁₀ₘₛ) = 0.0198. Listening longer is nearly pure gain.

---

## 5. The measurement bug (FIXED — understand it before trusting old numbers)

Eval rows inside a profile are a fixed-stride slide across one long capture
(GSM: `startSampleIndex = 26000·j`), and the GSM burst timeline has period
78,000 = 3 × 26,000. Therefore:

- `eval_rows[::3]` (the mid-run subsample) **locked onto a single burst
  phase** — the one where a burst always lands inside the first millisecond.
  GSM 1 ms recall: **0.9955 on `[::3]` vs 0.4390 on the full split.** The
  other two stride phases give min-cell 0.000.
- Worse, because every eval window started at capture **offset 0**, even the
  *full* split only ever sampled **three** discrete burst phases.

**Training was never affected** (it draws a fresh random offset per row per
episode). This was purely a measurement bug.

**The fix**, in *both* trainers inside a byte-identical shared block
(`# --- BEGIN/END SHARED EVAL PROTOCOL`, with a `diff` guard documented in
the source):

- mid-run subsample → seeded **random** subset (same size, drawn once at
  startup so all checkpoints of a run score the same rows)
- eval windows → per-row seeded **random** offsets for query **and**
  prototype rows, drawn in `[0, row_samples − max_dwell]` so one offset is
  valid at every dwell and **windows nest** (10 ms strictly extends 2.5 ms
  extends 1 ms — escalation semantics preserved exactly)
- flags `--eval-offset-mode {random,zero}`,
  `--eval-subsample-mode {random,stride}`, `--eval-offset-seed`; the plan is
  hashed into the result JSON

Burst-phase coverage went **3 → 668+**. Training streams re-verified
byte-exact against pre-fix references after the change.

---

## 6. TOP PRIORITY — corpus content diversity

**Defect:** 27 of 34 profiles realize fewer than 10 distinct phases across
their 96 eval rows; **22 realize exactly one** — all 96 clean waveforms
bit-identical (SHA-256 collapses to one hash; pairwise relative RMS
difference exactly 0.0).

**Root cause:** rows are placed by fixed stride into an index-pure generator
timeline, and the 20 ms row length is an exact integer multiple of the
LTE/NR 10 ms radio frame at every native rate used (30.72 / 15.36 / 1.92 /
122.88 Msps): stride = 2 × period → gcd = period → 1 phase.

**What it costs the model:** training already randomizes window offsets per
episode, so *phase* diversity within a row was never the training problem.
The problem is **content** — each profile contributes essentially one
payload/scheduling realization, varied only by per-row impairments. That is
a memorization risk and a generalization ceiling.

**What escaped, and why:** Bluetooth Classic and BLE have real diversity (26
and 66 distinct onsets) **because their generators already use randomness**
(keyed-hash slot utilization, the spec's random advDelay). The owner's rule
— *"Randomness everywhere is the rule of ML"* — is the principle to apply
everywhere else.

**Nuance:** LTE/NR ETM waveforms are *standards-defined*. Content repetition
is arguably correct there and random phase is the only meaningful variation.
The plan classifies each profile: (A) content is seed-variable, (B)
standards-fixed, (C) should vary but no knob exists.

**Also found:** `bluetooth-le-advertising-longdwell` has 29 of 96 eval rows
with no packet. Before calling it a bug, check the physics — a 20 ms window
on a ~30 ms advertising grid *should* be empty about a third of the time.
This may be correct, and is arguably the thesis in miniature.

### ⚠ ADDENDUM — the workflow finished after the handoff was written. Read this
### before acting on §6; it corrects numbers stated above.

**Corrected diversity measurement.** The "22 profiles bit-identical / GSM has
3 phases" figures above came from an eval-rows-only hash audit and a
mislabeled log line. Independently re-measured over all 8,704 rows, and more
importantly **phase-invariantly** (bit-hashing under-counts: FM's 256 rows
are all bit-distinct yet pairwise |ρ| = 1.00000 — the same waveform times a
global complex scalar, which the impairment stage randomizes anyway):

- **20 of 34 profiles have exactly ONE content realization** (all AM, CW, FM,
  LTE, NR, and wifi-ofdm-20m profiles).
- GSM has **12** realizations for 2 profiles, 3 for the other 5.
- The corpus holds **905 distinct 20 ms realizations across 8,704 rows
  (10.4%)**.
- **95.006% of eval rows duplicate a train row's clean waveform** in the same
  profile. Only **163 of 3,264 eval rows (5.0%) carry genuinely held-out
  content, and all 163 are bluetooth.**
- The split is **blocked, not interleaved** (rows 0–95 eval, 96–255 train per
  profile), which is why the only generator with real content variation ends
  up with fully disjoint train/eval content.

**Memorization result (the reassuring part):** there is **no train/eval gap**
— balanced train 0.9079/0.9528/0.9990 vs eval 0.9024/0.9501/0.9947 (gaps
−0.006/−0.003/−0.004), measured on the frozen G4 checkpoint with independent
offset plans so no prototype row is ever scored on the window it donated.
But the held-out-content test is at ceiling on both sides and covers one
class, so **the corpus cannot answer whether the model generalizes across
content for 6 of its 7 classes.** That is the finding that justifies
regeneration — not a demonstrated failure.

**Data-quality defect (new, real):** **80 of 256
`bluetooth-le-advertising-longdwell` rows are EXACTLY ZERO** in `clean.npy`
(51 train / 29 eval) — ~31% of the BLE class is pure silence labelled
`bluetooth`, and the model scores 1.000 on all of them at every dwell. It is
classifying noise-only rows as bluetooth. Decide deliberately whether that is
physics worth keeping (a 20 ms window on a 30 ms advertising grid *is* often
empty) or a labelling bug — but do not leave it undecided.

**Capability audit — this changes the plan.** `synthesizeAnalyticComplexIq`
has **no seed/payload/hop/scheduling parameter on its public surface**;
`startSampleIndex` is the only reachable degree of freedom. Classification:
- **(A) content variable — 2 profiles**: the two Bluetooth longdwell ones
  (keyed-hash occupancy/hop/phase, aperiodic timeline). Payload *bits* still
  never change — a qualified DH1 / ADV_NONCONN_IND vector is replayed.
- **(B) standards-fixed — 15 profiles**: LTE/NR test models (E-TM inputs are
  all-zero *by definition*; NR-TM data is spec-mandated PN23) plus cw/am/fm
  closed-form stimuli. **Repetition is correct here**; random phase is the
  only meaningful variation.
- **(C) should vary, no knob — 17 profiles**: all 7 GSM (the `seed` parameter
  is **unreachable dead code** — `isGeranFixedCatalogProfile` is tested first
  and the two profile sets are identical), plus wifi/DSSS.

**Toolchain is fixed and proven.** No tsx/ts-node needed: node v24.18.1
strips TS natively; the only gap was NodeNext `.js`→`.ts` specifier
resolution, solved by `tools/ts-source-resolve-hook.mjs` plus
`--experimental-transform-types`. Run it via
`npm run generate:longdwell-probe-corpus` (added to package.json). Verified
end-to-end producing real IQ.

**🔴 SignalLab work is at risk — act on this first.** The longdwell feature
was **never committed**: it lived only in `stash@{0}` of `Atom-SignalLab`
("On main: atomizer-check-sandbox-drift-1785548628"). The agent moved off
detached HEAD `02846e2` to `main` and used `git stash apply --index` (not
`pop`), so the stash still exists as a backup — but the repo now has 18
modified + 2 untracked files, uncommitted, holding
`src/bluetooth-long-dwell-iq.ts` (342 lines, 8/8 tests passing) and its test
file. **Commit that work in Atom-SignalLab.** A separate concurrent session
also landed untracked GERAN corpus generators there
(`src/geran-corpus-iq.ts`, `src/corpus-content-prng.ts`,
`src/geran-xcch-corpus-codec.ts`).

**Proof slice PASSED, and `docs/corpus-regen-runbook.md` now exists.** A real
34-profile × 8-row slice generated in 16 s (stage 1) + 6 s (stage 2). The
generator now makes three seeded per-row draws — time origin (stratified,
without replacement inside the profile's *measured* realization space),
carrier phase, and content where a knob exists — all recorded per row, with
the cyclic ceiling **asserted at runtime** (synthesize at k and k+period,
require identical hashes) rather than assumed. GSM went 3 → 8 distinct
realizations per 8 rows.

**Two unresolved decisions the next agent must make:**
1. `MIN_ACTIVE_SAMPLES` default. One agent set 1024 (rejects/redraws silent
   rows; BLE then 8/8 distinct), a concurrent one set 0 (keeps the natural
   silent rate; BLE 5/8 distinct, 3 silent). Both measurements are in the
   runbook. This is the same question as the BLE-silence defect above.
2. **Two sessions edited `tools/generate-longdwell-probe-corpus.mjs` and the
   runbook concurrently.** The edits were merged, not clobbered, but
   **re-read both files and re-run the proof slice before trusting them.**

Also note: the proof used 8 rows/profile, so every "distinct" count saturates
at 8. It cannot detect birthday collisions that only appear at 256 rows —
notably `wifi-ofdm-20m`, whose time-origin space is 1,000 samples and must
supply 256 distinct draws. The without-replacement drawer makes that exact by
construction and records `offsetSpaceExhausted`, but it is asserted, not
measured at production scale.

### In flight at handoff time
Workflow `w2k35buao`, three agents:
1. **toolchain-capability** — make the generator runnable (`Atom-SignalLab`
   is at detached HEAD `02846e2`, which predates the longdwell profile names;
   no tsx/ts-node installed) and audit per-profile content-variation knobs.
2. **memorization-audit** — quantify the cost: per-profile train-vs-eval
   accuracy, diversity-vs-recall correlation, and whether the corpus can
   answer the generalization question at all.
3. **regen-plan-proof** — update the generator for seeded random offsets
   **and** per-row content seeds, generate a real 34-profile × 8-row slice,
   and prove 8/8 distinct realizations before committing to a multi-hour
   rebuild. Writes `docs/corpus-regen-runbook.md`.

**Reading its results (it was launched in a session that has now ended).**
The workflow's own summary is gone with that session, but its journal
survives on disk — one `{"type":"result",...}` line per completed agent,
each containing that agent's full structured return value:

```bash
python3 -c "
import json
p='/Users/johnelliott/.claude/projects/-Users-johnelliott-PersonalGitHub/6179ba93-fb7a-4112-b70b-19540b0f4fce/subagents/workflows/wf_2f843779-d86/journal.jsonl'
for line in open(p):
    d=json.loads(line)
    if d.get('type')=='result': print(json.dumps(d, indent=1)[:4000])
"
```

**If the journal is incomplete or the workflow never finished**, the work is
not lost — it is fully specified in §6 above and its deliverables are
checkable on disk:
- `tools/generate-longdwell-probe-corpus.mjs` — does it now draw seeded
  random offsets *and* per-row content seeds? (`git diff`/read it; it was
  untracked at handoff, so compare against the behavior described in §6)
- `docs/corpus-regen-runbook.md` — **did not exist at handoff time.** If it
  is absent, the proof phase did not complete; redo it from §6's plan
  before regenerating anything.
- `Atom-SignalLab` git state — was it moved off detached HEAD `02846e2` to a
  ref containing the longdwell profiles, and was a TS loader installed? If
  `node tools/generate-longdwell-probe-corpus.mjs` still cannot import
  SignalLab, that is where to start.

### Plan after that
1. Regenerate into a **new directory** — never delete the existing corpus
   (80 GB, and it is the baseline everything so far was measured against).
2. Retrain v7 MLX bf16 on the new corpus, same seed, same protocol.
3. Compare against the §4 baseline. **Same architecture, same trainer,
   better data — that comparison is the experiment that says what the
   diversity was worth.**

---

## 7. The paper (deprioritized; here is its true state)

`docs/paper/v7_paper.tex`, 5 pages, compiles with `~/.local/bin/tectonic`.
Latest PDF `v7_paper_v12.pdf`. Author: John Elliott, Independent Researcher,
`JohnElliott@journalcorrespondence.com`. Figures: `fig_spectrograms.png`,
`fig_dae_breakdown.png` (**owner-approved: "Figure 3 is great"**),
`fig_architecture.png` (regenerate via `make_fig_architecture.py`), plus
inline pgfplots.

**The problem:** every quantity in the "Full system" paragraph traces to a
single mid-run log line at episode 1000 — `eval_rows[::3]`, 1,088 rows, the
luckiest of three burst phases. That log was **overwritten** by a `>`
redirect on restart, no result JSON was ever written (the run wedged), and
the identical-weights rerun disagreed (0.951/0.781 vs the paper's
0.909/0.448). The paper also argues its estimator-floor case on n=96 cells
while those min-cell figures were computed on n=32 (1088/34).

**If resumed:** replace Experiments with the §4 numbers (reproducible, with
error bars, and they tell the story *better* — GSM recall 0.404 → 0.701 →
0.997 across dwells is the empty-window thesis measured directly) and
disclose the aliasing as a methods finding. Do not try to reproduce the old
numbers; they are gone.

---

## 8. Landmines from today

- **CPU% ≠ progress.** The torch wedge spun at 90% CPU making zero progress.
  Use `sample <pid> 3 -file out.txt` plus CPU-time accrual over a window.
- **A monitor that greps only for success is silent through a crash.** The
  first stall monitor cried wolf because slow-but-alive looks like dead; the
  working version requires log silence **and** frozen CPU.
- **Workflow agents can die of context exhaustion at the reporting step.**
  The G2 agent did 113 log-heavy tool calls and had nothing left to file its
  structured report — the workflow "failed" though the work was on disk.
  Keep log-heavy agents narrow, or have them write files and report paths.
- **zsh does not word-split `$(...)`** — `kill $PIDS` fails. Use `| xargs kill`.
- **`pgrep -f` counts wrapper shells.** Use `ps aux | grep "[P]ython …"`.
- **`>` redirects destroy evidence** (that is how the paper's source log
  died). Use `>>` for anything that might be rerun.
- **/tmp scratchpads do not survive a reboot** — archive to
  `training/artifacts/` as you go.
- **Never regenerate 80 GB on an unverified fix.** The prior agent could not
  even run the generator. Prove on a small slice first.

---

## 9. Open tasks

| # | Task | State |
|---|---|---|
| 12 | v7 production Optuna study + held-out confirmation | pending (blocked on corpus) |
| 13 | MLX port G0–G5 | G0/G1/G2/G2b/G4 pass; **G5 remains** |
| 14 | **Corpus regeneration with content + phase diversity** | in flight — **top priority** |
| — | Paper rewrite with corrected numbers | deprioritized by owner |
| — | v3 release line (`HANDOFF-v3-release.md`) | paused; needs owner authorization to spend seed 20260735 |

### Upstream contributions (the owner explicitly wants these)
1. **PyTorch MPS**: the post-eval `copy_and_sync → waitUntilCompleted`
   deadlock — 2/2 reproductions with stack samples and a known mitigation.
   Needs a minimal reproducer to file.
2. **MLX**: TF32-on-by-default on M5 — not a bug, but we have measured error
   data (2.7e-2 vs 4.6e-6), a discoverability case, and a possible request
   for a programmatic toggle (env-var only today).
3. **MLX**: tall-skinny NHWC conv memory/perf findings from the per-block
   probes (`probe_final_{before,after}.log`); any complex64 or framing/STFT
   gaps hit during the port.

---

## 10. How to run things

```bash
# Production training (MLX, bf16) — ~53 min for 3000 episodes
.venv-training/bin/python \
  training/zplane_ab/v2_full_variation/v7_probe/v7_trainer_mlx.py \
  --episodes 3000 --eval-every 1000 --recon-weight 0.3 --seed 20260740 \
  --dtype bf16 --ckpt-blocks 3 --batch-chunks 5 --log-every 100 \
  --init-checkpoint training/artifacts/mlx-g2/step0.safetensors \
  --save-checkpoint <out>.safetensors --out <out>.json
```

```bash
# Parity harnesses (they set MLX_ENABLE_TF32=0 themselves — do not remove that)
.venv-training/bin/python training/zplane_ab/v2_full_variation/v7_probe/mlx_parity/test_unit_parity.py
```

```bash
# Re-evaluate any checkpoint under the corrected protocol, 5 seeds
.venv-training/bin/python training/artifacts/mlx-g2/g5_offset_reeval.py
```

Environment: `.venv-training` (Python 3.14), torch 2.13.0, mlx 0.32.0.
Node at `~/.local/node/bin`; tectonic at `~/.local/bin/tectonic`.
Machine: Apple M5 Max, 48 GB unified memory.

---

## 11. Working with this owner

- Wants **evidence, not reassurance** — every claim carries a number or a
  measurement. They caught the eval-aliasing bug themselves by noticing three
  dwell accuracies were suspiciously close. Take their skepticism seriously
  and go verify; it has been right.
- Prefers CLI over dashboards, and honest bad news early.
- Impatient with idle time, fine with long runs that are visibly progressing.
  Report status with real numbers when asked.
- Explicitly asked to be told when something is worth **contributing
  upstream**.
- Do not narrate every restart — but never silently paper over a failure
  either. Say what broke, what it cost, and what changed as a result.

---

## 11b. What I would do first, in order

1. **Protect the work** (§ FIRST FIVE MINUTES): ask about committing; back up
   `training/artifacts/mlx-g2/`.
2. **Recover the in-flight workflow's results** from the journal path in §6
   and determine whether the generator fix + proof slice actually landed.
3. **If the proof did not land:** redo it — generator gets seeded random
   offsets + per-row content seeds, then a real 34-profile × 8-row slice with
   the pass criterion "every seed-variable profile shows 8/8 distinct
   clean-waveform hashes." Do **not** regenerate 80 GB before that passes.
4. **If it did land:** run the full regeneration into a *new* directory
   (overnight; the runbook has the wall-time estimate), verify diversity with
   the runbook's script, then retrain MLX bf16 with the §10 command and
   compare to §4. That comparison is the point of the whole exercise.
5. **Then G5** (task 13): port `eval_confusion.py` and wire the Optuna
   harness, which unblocks task 12 (the production study).

Anything the owner asks for takes precedence over this list — especially if
they redirect away from the corpus. They have been consistently right about
priorities today.

## 12. One sentence of orientation

The architecture is validated and the training stack is finally fast and
reliable; what remains is to give the model data as diverse as the physics it
claims to model — fix the corpus, retrain, and measure against §4 — and to
resist the temptation to quote any number that a random re-measurement
would not reproduce.
