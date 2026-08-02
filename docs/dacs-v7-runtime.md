# DACS v7 runtime package

The trained v7 checkpoint is an episodic metric encoder, not a deployable
classifier by itself. It contains no fixed class prototypes and its confidence
head has not been calibrated for an open-set or dwell-escalation decision.
`tools/export-dacs-v7-runtime.py` closes the mechanical deployment gap without
silently widening the scientific claim.

The exporter:

1. requires a clean Atom-Classifier worktree and a corpus whose two recorded
   source worktrees were clean;
2. verifies the native MLX checkpoint against its sidecar parameter hash;
3. converts the BF16 encoder weights to a float32 ONNX inference graph while
   excluding the training-only decoder and optimizer state;
4. freezes one 64-row mean prototype per class and dwell from a single declared
   prototype-offset plan;
5. evaluates those unchanged prototypes over the full 3,264-row split under
   five independent query-offset plans; and
6. writes a canonical, hash-bound manifest covering the model, prototypes, and
   validation evidence.

Run from a clean source revision:

```bash
npm run package:dacs-v7-runtime -- \
  --output /path/to/empty/package-directory
```

The runtime contract is intentionally narrow:

- input is contiguous complex I/Q at exactly 20 Msps;
- supported prefixes are 20,000, 50,000, and 200,000 samples;
- browser preprocessing is RMS normalization followed by a periodic Hann-64,
  hop-32 complex FFT and the first 33 bins as real, imaginary, and
  `log1p(magnitude)` channels;
- the largest supported prefix present in the capture is used;
- classification is closed-set over `am`, `bluetooth`, `cw`, `dsss`, `fm`,
  `gsm`, and `ofdm`; and
- Atomizer must retain its released open-set gate ahead of DACS. The v7
  confidence logit is diagnostic telemetry only and cannot reject a capture or
  trigger an advertised dwell policy until it has independent calibration.

The package manifest pins both training provenance and export provenance. The
training source revision may be older than the exporter revision; that is
expected and both are recorded independently.
