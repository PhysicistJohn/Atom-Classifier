# Over-the-air results — 2026-08-02, NeptuneSDR

Checkpoint: `sota-v2-2014019-12e7eef-20260801/content_v2_bf16.safetensors`.
Tools: `tools/capture-neptune-iq.py`, `tools/classify-neptune-capture.py`.
Raw captures + classification JSON: `training/artifacts/ota-20260802/`
(git-ignored; the band snapshot `fm-stuck-band.iq.npy` is the keeper).

## Hardware finding first: the Neptune's tuning is broken

The RX LO attribute (`ad9361-phy/altvoltage0/frequency`) accepts writes and
reads them back, and RSSI responds — but the received spectrum **never
moves**. Two captures requested 1 MHz apart show identical spectra; an ENSM
alert/fdd toggle does not help. Fitting the received station lineup shows
the RF has been parked at **≈98.283 MHz** the whole time (every "airband",
"LTE", and "2.4 GHz" capture was actually the FM broadcast band). SSH reboot
was not possible (vendor changed the default password); the device needs a
**physical power-cycle**, after which `tools/capture-neptune-iq.py`'s
tune-a/tune-b shift test should be repeated before trusting any capture.
This is the class of vendor-firmware defect the Atom-NeptuneSDR-Firmware
project exists to fix.

Silver lining: the stuck band is the FM band, so real broadcast FM was
captured with known ground truth. Verified-real stations in the capture
(widths 70–190 kHz, FM-shaped): 93.7, 95.5, 100.1, 101.5, 101.7, 101.9,
103.1, 104.9 MHz. Narrow spurs at +2.2 MHz and others.

## Classification results on real FM

Full-band capture (14 stations at once) → `ofdm` at every dwell. Fair
enough: many narrowband carriers **is** structurally multicarrier.

Single stations channelized to the corpus condition (shifted to DC,
150 kHz lowpass, with and without corpus-matched AWGN):

| station | result at 1/2.5/10 ms | note |
|---|---|---|
| 101.7 MHz (HD Radio) | gsm / gsm / gsm | HD OFDM sidebands + dynamic audio |
| 104.9 MHz | cw / cw / cw | reads as bare carrier |
| 95.5 MHz | gsm / gsm / gsm | |
| 93.7 MHz | gsm(2)+cw(2)+fm(1) / gsm / gsm | one lone fm vote |

Harness sanity check: synthetic holdout `fm` and `am` rows pushed through
the **identical** capture-classification path score 5/5 votes for the right
class with large margins. The pipeline is correct; the failure is real.

## Root cause: the corpus fm/am/cw are closed-form lab stimuli

`Atom-SignalLab/src/complex-iq.ts` (`analyticLaboratorySample`):

- `cw` = constant `[1, 0]`
- `am` = **single-tone** DSB full-carrier AM
- `fm` = **single-tone** sinusoidal FM, β = 3 — a discrete Bessel-line
  spectrum

Real broadcast FM is noise-like wideband FM of a stereo multiplex (19 kHz
pilot, 38 kHz DSB stereo, 57 kHz RDS, ±75 kHz deviation, often HD Radio
OFDM sidebands). It shares only constant envelope with tone-FM-β3. The
model was never shown anything like it, so it lands on gsm/cw — and note
the model's spectrogram front end (Hann-64, 33 bins over 10 MHz ⇒ 312.5 kHz
bins) puts an entire FM channel inside one frequency bin, so the class
decision rides on time/phase structure, exactly where tone-FM and MPX-FM
differ most.

**This is a corpus-realism finding, not a model bug.** The classifier does
what it was taught. Teaching it real broadcast modulations is the next
corpus revision — see `docs/better-test-plan.md`.

## Scoreboard vs the original test intent

| intent | outcome |
|---|---|
| FM band → `fm` | ✗ — real WFM ∉ training distribution (finding above) |
| AM band → `am` | not reachable: AM broadcast < 70 MHz tuning floor; airband AM blocked by stuck LO |
| GSM/LTE | blocked by stuck LO (and GSM is largely sunset here) |
| bonus | full-band multicarrier → `ofdm` behaves sensibly; harness verified end-to-end |
