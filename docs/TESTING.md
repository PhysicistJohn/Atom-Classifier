# Testing

Three tiers, cheapest first. Every number quoted anywhere should be
reproducible by one of these commands.

## Tier 0 — unit / parity (seconds to minutes)

```bash
npm test                      # 38 files / 341 TS tests
.venv-training/bin/python training/zplane_ab/v2_full_variation/v7_probe/mlx_parity/test_unit_parity.py
.venv-training/bin/python training/zplane_ab/v2_full_variation/v7_probe/mlx_parity/test_forward_parity.py
```

## Tier 1 — standard eval split (~5 min)

Five-seed corrected-protocol eval of a checkpoint on the production corpus
eval split (3,264 rows; prototypes from the train split of the same corpus):

```bash
.venv-training/bin/python tools/eval-dacs-checkpoint.py \
  --checkpoint training/artifacts/sota-v2-2014019-12e7eef-20260801/content_v2_bf16.safetensors \
  --out <result.json>
```

`tools/eval-dacs-checkpoint.py` is the parameterized descendant of
`mlx-g2/g5_offset_reeval.py`; verified 2026-08-02 to reproduce the recorded
seed-20260740 reference exactly (bal 0.9043/0.9500/0.9927, errors
506/253/15). Caveat: the eval split shares clean content with the train
split for standards-fixed profiles — Tier 1 measures the deployed protocol,
not content generalization. That is what Tier 2 is for.

## Tier 2 — held-out corpus (SignalLab, fresh seeds)

A corpus the model has never seen in any form: same 34-profile plan, but
fresh offset, phase, content, and impairment seeds. Where a profile has a
content knob (GSM, Wi-Fi, LTE/NR PDSCH, Bluetooth) the payload/scheduling
content is genuinely new; standards-fixed profiles (E-TM, CW/AM/FM
analytic) differ in phase, offset, and impairments only — that is the
physics, not a shortcut.

Generate (~4 min for 34 × 64 rows, ~14 GB):

```bash
OUT_DIR=training/artifacts/holdout-corpus-seed20260802 \
PLAN_JSON=tools/production_corpus_plan.json \
ROWS_PER_PROFILE=64 DURATION_MS=20 OFFSET_SEED=20260802 \
npm run generate:longdwell-probe-corpus

CORPUS_DIR=training/artifacts/holdout-corpus-seed20260802 \
EVAL_PER_PROFILE=64 IMPAIR_SEED=20260802 \
.venv-training/bin/python tools/longdwell_probe_stage2.py
```

Evaluate — prototypes still come from the production train split
(enrollment never sees holdout rows):

```bash
.venv-training/bin/python tools/eval-dacs-checkpoint.py \
  --checkpoint <ckpt.safetensors> \
  --query-corpus training/artifacts/holdout-corpus-seed20260802 \
  --query-role all \
  --proto-corpus training/artifacts/longdwell-production-corpus-v2 \
  --out <result.json>
```

Pass expectation: within a few points of the Tier 1 numbers. A large gap
(especially on content-variable profiles) means the model memorized corpus
content — exactly the failure mode the corpus-v2 regeneration was built to
detect.

The seed is part of the artifact name. Never evaluate on a holdout seed
that has appeared in any training or tuning run; when a holdout seed gets
used for a decision (e.g. model selection), retire it and mint a new one.

## Tier 3 — over-the-air smoke test (NeptuneSDR)

The "simple duh" test: the classifier should recognize obvious live
signals. Hardware: NeptuneSDR (HAMGEEK P210, AD9361) at `ip:10.0.0.250`
via libiio, tuning range 70 MHz–6 GHz — note this excludes AM broadcast
(~0.5–1.7 MHz); aviation-band AM voice (118–137 MHz) is the reachable AM
representative and is intermittent by nature.

Capture 20 ms at 20 Msps (the runtime contract rate; 400,000 samples,
matching a corpus row) per target:

| target | tune | expected class |
|---|---|---|
| FM broadcast station | strongest carrier in 88–108 MHz | `fm` |
| aviation AM voice (opportunistic) | active channel in 118–137 MHz | `am` |
| LTE downlink | strongest carrier in 700–900 MHz / 1.9–2.1 GHz | `ofdm` |
| Wi-Fi 2.4 GHz | 2412/2437/2462 MHz | `ofdm` (HR-DSSS beacons → `dsss`) |
| Bluetooth | 2402–2480 MHz | `bluetooth` |
| GSM (if any survives the 2G sunset) | 850/1900 MHz GSM carriers | `gsm` |

```bash
# capture (writes <name>.iq.npy complex64 + sidecar json)
.venv-training/bin/python tools/capture-neptune-iq.py \
  --uri ip:10.0.0.250 --center-mhz 98.7 --name fm-broadcast

# classify all captures in a directory against a checkpoint
.venv-training/bin/python tools/classify-neptune-capture.py \
  --checkpoint <ckpt.safetensors> \
  --captures training/artifacts/ota-<date>/
```

Honest framing: training data is synthetic baseband at controlled SNR with
the signal occupying the row; a real capture has neighbor channels, AGC,
and front-end character the corpus never modeled. This tier is a sanity
check ("does it call an FM station fm?"), not a calibrated field
evaluation. Log every capture's result, including the wrong ones — the
wrong ones are the interesting data for the next corpus revision.

Results log: `docs/ota-results.md`. First run (2026-08-02): the Neptune's
LO is stuck at ≈98.283 MHz (vendor firmware bug — power-cycle and re-verify
tuning with the two-center shift test before trusting captures), and real
broadcast FM classifies as gsm/cw because the corpus `fm` is single-tone
lab FM. See `docs/better-test-plan.md` for the corpus-realism fix. The
frozen real FM-band capture lives at
`training/real-captures/neptune-fm-band-20260802.iq.npy` with labeled
stations in its sidecar; channelize with
`tools/classify-neptune-capture.py --shift-hz <offset> --lowpass-hz 150000`.
