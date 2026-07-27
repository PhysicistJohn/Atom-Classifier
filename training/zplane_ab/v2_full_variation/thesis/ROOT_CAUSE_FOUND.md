# Root cause of the bluetooth/gsm/dsss ceiling: a corpus generation bug

**Status: confirmed by direct measurement against the corpus, 2026-07-24.**

The adversarial review of THESIS.md ranked "audit the generator" (F5 / Step 0) as the one check
that could invalidate everything, and said it should gate all other work. It was run. It found the
problem, and the problem is not architectural.

---

## Bug 1: Bluetooth's sample rate is derived from the hop span, not the channel bandwidth

`Atom-SignalLab/src/waveforms.ts:91-92` defines both Bluetooth profiles with an ~79-84 MHz span
(the 2.4 GHz ISM hopping range) while carrying the true channel width in a sub-field:

```
'bluetooth-classic-connected': knownScenario(2_441_000_000, 79_000_000, 84_000_000, 'classic-hop',
    'classic-slots', { channelWidthHz: 1_000_000, slotSeconds: 0.000625, hopRateHz: 1_600 }),
```

`tools/generate-signallab-iq-corpus.ts:104-111` picks the sample rate from `d.occupiedBandwidthHz`:

```
const occ = d.occupiedBandwidthHz;               // ~79-80 MHz for bluetooth (the HOP SPAN)
... TARGET_FRACS.map((tf) => clampInt(occ / tf, SR_MIN, SR_MAX))
const bandwidthHz = clampInt(occ * 1.15, ...);   // -> 90.85 / 92.0 MHz recorded in the manifest
```

So the generator sizes the capture for a 90 MHz signal, while the synthesizer emits the real
~1 MHz GFSK channel. Measured against the actual waveform (`preprocess.estimate_band` over 12
samples/profile, median):

| profile | claimed fracBW | measured fracBW | ratio |
|---|---|---|---|
| bluetooth-classic-connected | 0.370 | **0.013** | **0.03** |
| bluetooth-le-advertising | 0.374 | **0.012** | **0.03** |
| *every other one of the 35 profiles* | -- | -- | 0.41 - 1.24 |

Bluetooth is the only profile whose metadata disagrees with its own waveform, and it disagrees by
**30x**. Consequences:

- **~180x oversampling.** fs is 175-245 MHz for a signal occupying ~2-3 MHz.
- **Window duration 16.7-23.3 us.** A Bluetooth Classic slot is 625 us; a BLE advertising packet is
  ~128 us. The 4096-sample window therefore spans **at most ~3.7% of one slot, and never a whole
  packet**. Burst duration, duty cycle, hop timing, packet structure -- the entire family of
  discriminants that separate a GFSK burst from a GMSK burst from a DSSS frame -- are *physically
  absent from the data*, not merely hard for the model to extract.

This is the single best explanation for why the triad confusion survived every one of the ~10
interventions: no loss function, normalization scheme, backbone, or training budget can recover
information the capture never contained.

## Bug 2: the resampled condition is ~90% zero padding for bluetooth and cw

`preprocess.preprocess()` resamples to a canonical fractional bandwidth of 0.5, then `center_fit`
pads or crops to 1024. When measured fracBW is tiny, the resample decimates hard and the result is
mostly padding. Measured (median real samples out of 1024, 40 samples/class):

| class | real samples / 1024 | % real |
|---|---|---|
| dsss | 1024 | 100.0% |
| ofdm | 1024 | 100.0% |
| gsm | 880 | 85.9% |
| fm | 824 | 80.5% |
| am | 480 | 46.9% |
| **bluetooth** | **96** | **9.4%** |
| **cw** | **96** | **9.4%** |

The "resampled vs native" comparison run across every architecture this session was therefore never
an apples-to-apples test of scale normalization. It was partly a test of *how much of each class
survived padding*. This is precisely the alternative explanation the adversarial review raised
against the thesis's top-ranked cause (P1), listed there as "effective-length collapse", and it is
confirmed.

---

## What this invalidates

- **The thesis's ranking.** P1 (absolute-scale deletion) was ranked first on the strength of the
  native-beats-resampled result (E6, +4.4 points). That result is now substantially explained by
  Bug 2 instead. The review predicted exactly this and asked for the check before any of Step 2.
- **Much of the architecture search.** Six architecture families and ~10 interventions were
  measured against a corpus in which one of the three confused classes cannot express its
  discriminative structure. Their relative ranking may survive; their *ceiling* is corpus-imposed.
- **The 1.5-3% "irreducible floor"** in the thesis is not measurable from this corpus.

## What to do instead, in order

1. **Fix the generator.** Use `channelWidthHz` (1 MHz classic / 2 MHz BLE), not the hop span, to
   size bluetooth captures. Audit every profile for the same span-vs-channel confusion.
2. **Capture long enough windows to contain burst structure.** For bluetooth that means >= 625 us
   (>= 1 slot), ideally several slots to expose the hop pattern; for GSM >= 577 us (>= 1 normal
   burst), ideally a 4.615 ms frame. At a correctly-chosen fs (a few MHz, not 245 MHz) this is a
   few thousand samples, not a petabyte corpus -- the thesis's Step 5 cost estimate was inflated
   because it assumed the broken sample rates.
3. **Re-measure the bluetooth/gsm/dsss confusion on the fixed corpus** before spending any further
   effort on architecture. The correct baseline for that re-measurement is the existing 38k CNN.
4. Only then revisit the thesis roadmap. Several of its steps (per-sub-profile diagnostics,
   profile-level prototypes, held-out-parameterization splits) remain sound and cheap.

## Status of running work

- The weekend architecture campaign was **stopped** -- it was searching architectures against a
  ceiling this bug imposes.
- `complexlstm_run3` (complex multi-scale + CLDNN, native, 5000 episodes) was **left running**. The
  review flagged it as the one unfinished experiment that directly tests the thesis's headline
  ceiling claim, and it is nearly complete. Its result is still informative as an
  architecture-vs-architecture datum on the *current* corpus; it is not informative about the true
  ceiling.
