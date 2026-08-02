# Better test + 24 h compute plan — drafted 2026-08-02

The OTA smoke test (`docs/ota-results.md`) showed the deployed model is
excellent on its synthetic distribution (holdout: 0.904/0.954/0.993 across
dwells — indistinguishable from the eval split) and blind to real broadcast
modulations, because the corpus `fm`/`am` are single-tone closed-form
stimuli. The better test, and the compute to back it, follow directly.

## The better test: real-capture benchmark + realism-hardened corpus

**Definition of success:** the classifier calls a real, channelized FM
broadcast station `fm` (and real AM voice `am` once hardware allows), while
Tier 1/Tier 2 numbers do not regress.

### Track 1 — realistic broadcast profiles in SignalLab (dev, ~2–3 h)

Add seeded corpus-only generators (same pattern as the GERAN/Wi-Fi
corpus-content paths, keeping the analytic catalog firewall intact):

- `fm-broadcast-mpx`: wideband FM of a stereo multiplex — seeded
  noise-like audio (pink-ish, preemphasized), 19 kHz pilot, 38 kHz DSB
  stereo subchannel, optional 57 kHz RDS BPSK, ±75 kHz deviation.
  Content seed → audio realization.
- `am-voice`: envelope statistics of voice (syllabic-rate seeded noise
  through a speech-band filter), full-carrier AM, ~52 kHz channel.

Both join the plan as additional profiles of the existing `fm`/`am`
classes (34 → 36 profiles). The closed-form profiles stay — they are the
lab stimuli and regression anchors.

### Track 2 — corpus v3 + retrain + evaluate (compute, ~2 h/run)

1. Regenerate the production corpus with the two new profiles (~15 min).
2. Retrain MLX bf16, standard config (~1 h).
3. Tier 1 (new eval split) + Tier 2 (fresh-seed holdout) + the **real
   capture benchmark**: the frozen 2026-08-02 stuck-band capture, every
   verified station channelized, expected class `fm`. This capture is the
   first entry of a growing labeled real-RF regression set; each future
   working-hardware session adds bands (airband AM, LTE, Wi-Fi, BT).

### Track 3 — fill the 24 h (queue behind Track 2, order by value)

1. **Five-seed training replicates** of the production config — training
   variance has never been measured (all published spread is eval-offset
   spread on one checkpoint). ~1 h each.
2. **Confidence-head calibration** on holdout data + measured escalation
   curve (the standing open item blocking any dwell-policy claim).
3. **Short-run Optuna study** (300-episode trials, ~5 min each) over lr /
   recon-weight / conf-weight / mask-frac, MLX trainer — the old G5/task-12
   item. Confirm winners with full runs.

## Hardware prerequisite for the OTA half

Power-cycle the Neptune, then verify tuning actually moves the spectrum
(capture at two centers 1 MHz apart must shift all peaks by exactly
1 MHz — `docs/ota-results.md` has the procedure). Until then, only the
frozen FM-band capture is trustworthy real data.

### Track 4 — propagation channel from Atom-DSP (added 2026-08-02)

The corpus's noisy chain had receiver impairments but a static channel —
no fading, no Doppler, no TDL structure — while Atom-DSP ships a
propagation model (TR 38.901 TDL-A/B/D, Rayleigh/Rician + Doppler,
counter-based and split-invariant) built for exactly this producer.
Adopted via a parity-gated numpy port:

- `tools/gen-dsp-channel-vectors.mjs` pins golden vectors from the
  TypeScript source of truth (Atom-DSP revision recorded in the JSON);
- `tools/dsp_channel.py` is the port; `tools/test_dsp_channel_parity.py`
  gates it (hash integer-exact, floats ≤1e-12, split-invariance, row-path
  vs scalar reference);
- stage 2 applies it per row behind `CHANNEL_MODE=on` (off reproduces the
  pre-channel chain byte-for-byte — verified), draws
  none/tdl-a/tdl-b/tdl-d at 25/30/25/20%, delay spread logU(30,500) ns,
  Rayleigh 70% / Rician K∈U(3,12) dB 30%, Doppler logU(1,300) Hz, fading
  on an exact 256-sample grid with linear interpolation, everything
  recorded per row in the manifest.

Corpus v3 = Track 1 profiles + `CHANNEL_MODE=on` + fresh seeds.

## Status log

- 2026-08-02: plan drafted. Five-seed replicates started (Track 3.1) while
  Track 1 awaits owner go-ahead (it adds generators to Atom-SignalLab,
  which has its own provenance discipline).
- 2026-08-02 (later): replicates superseded by the Optuna study
  (`training/artifacts/optuna-mlx-20260802/`, baseline re-enqueued every
  8th trial for variance). Track 4 implemented and verified; channel-on
  stage 2 costs ~30 s extra per 2,176 rows.
