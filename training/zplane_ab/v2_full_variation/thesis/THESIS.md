bandwidth normalization plus RMS normalization plus DC downconversion delete absolute bandwidth (Hz), absolute symbol/chip rate (Hz), and absolute power. These are the strongest available physical separators of the triad. | **PREPROCESSING** |
| **P2** | Fixed 4096-sample windows span a 30x range of absolute duration (136 us at 30 MHz, 4.096 ms at 1 MHz), so burst duration, duty cycle, and repetition period are visible for some samples and arithmetically absent for others. | **PREPROCESSING / ALIGNMENT** |
| **D1** | Class labels pool physically distinct sub-profiles. bluetooth pools GFSK-classic (h ~ 0.32, BT 0.5, 1 MHz) with BLE-advertising (h = 0.5, BT 0.5, 2 MHz). gsm pools GMSK normal bursts (h = 0.5, BT 0.3) with EDGE 8PSK/QPSK/16QAM/32QAM. ofdm pools ~20 LTE/NR/WiFi profiles. **BLE and GSM-GMSK are both Gaussian-filtered binary CPM at h = 0.5**, differing only in BT product (0.5 vs 0.3) and absolute rate, and absolute rate is what P1 deletes. | **DATA / LABEL-STRUCTURAL** |
| **A1** | Every backbone tried, and all 12 hand features, compute only alpha = 0 (time-averaged) statistics. Mean+std pooling, CLS-token attention pooling, and BiLSTM final-state pooling are all time averages. The scale-free discriminants that survive P1 (the dimensionless ratio 2h; the DSSS chip-to-symbol cycle-frequency ratio of 11:1; conjugate-SCF presence versus identical zero) live at alpha != 0 and require a quadratic lag-product beaten against a complex exponential, which a ReLU conv stack with global mean pooling cannot express. | **ARCHITECTURAL** |
| **A2** | Classification is nearest-prototype where the prototype is the *mean* of enrolled embeddings. A single mean is a misspecified estimator for a multimodal class: it lands in the gap between modes, which is territory that may belong to a neighbour. | **ARCHITECTURAL (inference rule)** |
| **A3** | 12 informative hand-feature dimensions are concatenated with a much higher-dimensional learned vector, linearly mixed, and L2-normalized to 32 dims. The prototypical loss has no mechanism that preserves a low-variance discriminative subspace against high-variance nuisance directions. | **ARCHITECTURAL (fusion)** |
| — | Optimization, capacity, class imbalance, loss shaping, training budget. | **Ruled out** (section 1.1) |

### 1.3 Evidence-to-cause matrix

Rows are the empirical facts that any diagnosis must explain. `+++` = uniquely and fully explained; `++` = well explained; `+` = consistent; `.` = not addressed.

| Evidence | P1 scale | P2 window | D1 pooled labels | A1 alpha=0 | A2 mean prototype | A3 fusion |
|---|---|---|---|---|---|---|
| E1. Robust median/MAD fixed a real 1.15M-norm outlier bug; leakage unchanged (215/239) | + | + | + | ++ | + | + |
| E2. STFT view: gsm->bt 236->143, but new cw absorption | ++ | ++ | . | ++ | . | . |
| E3. dsss well separated in hand features (z=2.956) yet leaks in learned space | . | . | . | ++ | + | **+++** |
| E4. 750k episodes flat from ep 20k | +++ | + | ++ | +++ | ++ | + |
| E5. lambda=8.0 repulsion: leakage 236->162 but bluetooth clean 0.968->0.279 | . | . | **+++** | . | **+++** | . |
| E6. Native (0.891) beats resampled (0.847) for **every** architecture | **+++** | + | . | . | . | . |
| E7. Oversampling 3x/5x: null | + | + | + | + | + | + |
| E8. Per-sample contrastive repulsion: null (215/237) | ++ | + | + | ++ | . | . |
| E9. Corpus regen 1811 -> 14k / 37 profiles: overall up, triad flat | ++ | + | ++ | ++ | + | . |
| E10. 2D and 1D have structurally different error sets | ++ | ++ | . | ++ | . | . |

### 1.4 Why each specific piece of evidence falls out

**Why robust feature normalization fixed a real bug but not the confusion (E1).** The bug was a *dynamic-range* defect in a channel that was not the binding constraint. The 12 hand features are all alpha = 0 phase-invariant statistics: c20, c40, c41, c42, c60, c63 are time-averaged cumulants, envelope moments are time-averaged, IF spread is a time-averaged second moment. For constant-envelope binary CPM, the alpha = 0 slice is *near-degenerate*: GFSK and GMSK are both constant-modulus Gaussian-filtered binary CPM, so their time-averaged cumulants nearly coincide by construction. Fixing the scaling of a near-degenerate feature restores its dynamic range without creating a distinction that was never in it. The fix also did nothing about P1, D1, or A2. It was a correct fix to a real defect that was simply not the bottleneck, which is the most common shape of a null result in a conjunctively-constrained system.

**Why the spectrogram traded one confusion for another (E2).** A representation change *moves* the kernel of the observation map; it does not shrink it. log|STFT| adds relative time-frequency energy structure (burst on/off, duty cycle, ridge geometry), which is exactly what the gsm/bluetooth pair needs, hence 236 -> 143. It simultaneously deletes the instantaneous-frequency trajectory, and once you delete IF, a GFSK burst is a narrow bright ridge, a low-index FM signal is a narrow bright ridge, a quiet-passage AM signal is a narrow bright ridge, and cw is a narrow bright ridge. The cw prototype becomes the attractor for everything whose log-magnitude spectrogram is a narrow ridge. This is not a bug in the ViT; it is the exact predicted consequence of magnitude-only representation, and the RF SSL literature independently reports the same mechanism (LatentWave's masking ablation: frequency-axis-destroying masks gain 11 points on channel tasks and lose 15 on classification, 66.1 vs 80.9).

**Why dsss and gsm are well separated in hand-feature space yet still confuse (E3).** Three mechanisms, ranked, and I want to flag a caveat on the diagnostic itself first.

*Caveat.* The reported 2.956 and 2.765 are **centroid** z-distances. With within-class robust spreads of 1.495 (bluetooth) and 2.053 (gsm), a centroid separation of 2.9 still implies substantial cloud overlap. Centroid distance is a Fisher-style summary; it is not classifier separability. The inference "dsss is well separated in hand-feature space" is weaker than it reads and should be replaced by an actual classifier measurement (falsification test F3, section 5).

Taking the diagnostic at face value, the ranked mechanisms are:

1. **A3, fusion dilution.** A 12-dimensional informative subspace concatenated with a higher-dimensional uninformative learned vector, linearly projected to 32 dims and L2-normalized, under a distance loss that has no term rewarding preservation of that subspace. Intervention 5 discovered empirically that the gsm cumulant outliers "were inflating the global feature std and compressing every other class's feature signal", which is direct evidence that the hand-feature channel's contribution to the embedding was being attenuated relative to the learned channel. Fixing the outliers fixed the *scale* imbalance but not the *dimensional* imbalance or the absence of any loss pressure to keep the informative direction.
2. **A1.** The learned path, being an alpha = 0 texture detector, sees a bandwidth-normalized DSSS waveform as generic wideband noise-like bursty digital, and a bandwidth-normalized GFSK burst as generic bursty digital. The chip-rate cyclic line at 0.5 x occupied bandwidth is the sharp separator, and it is precisely the object the architecture cannot compute.
3. **A2.** If bluetooth's prototype is the mean of two separated modes, it sits in the gap, which is a central, low-specificity region of the embedding. Anything under-determined lands nearest to it. "bluetooth" becomes the default attractor, which is exactly the observed asymmetry: both dsss and gsm leak *into* bluetooth, not into each other.

**Why 750k episodes changed nothing (E4).** Two independent reasons, either sufficient:
- The input map is not injective on the triad. No function of a non-injective map separates the collapsed classes. Training longer refines a function whose domain lacks the distinction.
- Even where the map is injective, the episodic prototypical loss (cross-entropy over negative squared distance to the class *mean*) instructs the model to map every sample of a class to a single point. For a genuinely bimodal class, the optimum of the stated objective is *not* the optimum of the task. Longer training makes the model better at the wrong objective. Flat-from-20,000 is the signature of arriving at a fixed point of a misspecified objective, not of failing to descend.

**Why strong prototype repulsion traded bluetooth accuracy for leakage reduction (E5).** This is the single most diagnostic result in the whole program and it is a clean, unambiguous fingerprint of D1 + A2. Bluetooth is bimodal (classic GFSK h~0.32; BLE h=0.5). One of those modes, BLE, is physically almost the same signal as gsm-normal-burst once absolute rate is removed, because both are Gaussian-filtered binary CPM at h = 0.5. Repulsion at lambda = 8.0 pushes the bluetooth *mean* away from the gsm mean. Since the mean sits between the two modes, pushing it away drags it out of *both* modes. Bluetooth samples are now far from their own prototype (0.968 -> 0.279 clean accuracy) and leakage improves only because bluetooth's prototype has stopped being a nearby attractor for anything at all. Overall accuracy fell to 0.839. You cannot repel a class mean away from a neighbour when one of that class's modes genuinely coincides with the neighbour. This experiment did not fail; it *measured* the multimodality.

**Why native beat resampled for every architecture, by 4.4 points on the CNN (E6).** Native retains samples-per-symbol, which is fs / R_s. Both fs and R_s are class-linked, so sps is a partial, entangled proxy for absolute rate. Resampling to a canonical fractional bandwidth of 0.5 forces sps to approximately 2 for every class, deleting the proxy. This is direct, measured evidence that absolute-scale information is worth several accuracy points to this problem, on your own corpus, already. It is also the reason the fix is *not* "go native": sps is entangled with fs across 63 values and the network is never told fs, so it cannot disentangle R_s in Hz from sps. Feeding the physical quantity explicitly is strictly better than leaking a correlate of it.

There is an alternative reading of E6 that must be disambiguated: resampling is a lossy interpolation that damages higher-order cumulants and IF estimates. Both readings predict native > resampled. They are separated by one experiment (section 5, F1).

**Why oversampling and per-sample contrastive repulsion were null (E7, E8).** Neither creates information. Oversampling changes episode frequency, not the estimator or the observation. Per-sample contrastive repulsion asks the network to separate points that are near-duplicates under the input map; there is no direction in which to push them. Additionally, the supervised-contrastive literature's effect sizes come from very large batches (SupCon ImageNet: batch 6144); a 7-way 5-shot episode is roughly 70 samples, so intervention 4 was a small-batch SupCon, which is not SupCon.

**Why corpus regeneration improved overall accuracy but not the triad (E9).** More profiles per class improves coverage and diversity, which helps every class whose label is coherent, and simultaneously *worsens* the pooling problem D1 by adding more heterogeneity inside the pooled labels. The two effects partially cancel on the triad and add on the rest. This is a supporting datum for D1, not a neutral one.

### 1.5 Ranking, and the conjunction argument

**Rank 1: P1 (preprocessing, absolute scale deleted).** Uniquely explains E6. Explains E4 fully, E2/E8/E9/E10 well. And it is the only single cause that is *sufficient on its own* to produce the observed pairwise confusions given the physics. The separators it deletes are not subtle: GSM 200 kHz / 270.833 ksym/s, Bluetooth 1 to 2 MHz / 1 Msym/s, 802.11b HR/DSSS 22 MHz / 11 Mchip/s. Those are factors of 3.7x and 11x in rate and 5x and 22x in bandwidth. Nothing else in this problem offers separations that large.

**Rank 2: D1 + A2 jointly (pooled labels and the mean-prototype estimator).** D1 is the data condition, A2 is the estimator misspecified for it; they are one cause with two halves and neither is meaningful without the other. Uniquely explains E5, which is the sharpest single result in the evidence set. Explains E4, E7, E9. Supported by the strongest external analogue available: Infinite Mixture Prototypes (ICML 2019) reports its largest gains precisely on superclass-level tasks (Omniglot alphabet: ProtoNet 65.6 -> IMP 92.0) and essentially *no* gain on unimodal classes (98.4 vs 98.2), and Deep Nearest Centroids (ICLR 2023) shows the multi-centroid gain scaling with class heterogeneity (ImageNet +0.5 top-1 at K=4, ADE20K +1.1 mIoU at K=10, reversing at K=20).

**Rank 3: A1 (alpha = 0 blindness).** Not the primary triad fix, and I want to be honest about that: once P1 is repaired, absolute bandwidth alone separates all three triad classes. A1's genuine, non-redundant contributions are three:
- **dsss versus ofdm**, which absolute bandwidth does *not* separate (20 MHz WiFi-OFDM versus 22 MHz HR/DSSS). Conjugate SCF is identically zero for QPSK/QAM/OFDM and non-zero for DSSS-BPSK. That is a hard physics-level separator for the one pair P1 cannot reach.
- **Robustness.** Cycle-frequency *locations* are invariant to multipath (well-replicated in the CSP literature), and non-conjugate cycle frequencies at k/T0 are immune to CFO. A bandwidth estimate under multipath and adjacent-channel energy is not robust; a cyclic line is.
- **Distribution-shift survival.** The Snoap/Latshaw/Popescu/Spooner result (Sensors 2023, 23(12):5735) is the most important external warning in this entire evidence base: an I/Q-trained capsule net at 97.5 percent in-distribution collapsed to 23.7 percent under a shift in *one* nuisance parameter's distribution (CFO), while a cyclic-cumulant front end went 92.3 -> 91.6. Your corpus is synthetic from a single generator with a fixed parameter distribution. Your 0.891 may be substantially optimistic in exactly this way.

**Rank 4: A3 (fusion dilution).** Uniquely explains E3, but rests on a centroid statistic that overstates separability. Cheap to test, high value if confirmed.

**Rank 5: P2 (window duration).** A *variance* cause rather than a bias cause. It explains why the within-class spreads (gsm 2.053, bluetooth 1.495) are large enough to swamp the 0.713 centroid distance, which is what makes the diagnostic *look* like irreducible overlap. At 4096 samples the arithmetic is unforgiving: at 30 MHz your window is 136 us, shorter than a single 577 us GSM burst; at 1 MHz it is 4.096 ms, spanning seven GSM bursts. So the burst-structure feature that best separates gsm from bluetooth is present in some samples of a class and physically absent from others, and which one you get is determined by the sample rate. Hop rate (1600 Hz, 625 us period) and GSM TDMA frame rate (216.7 Hz, 4.615 ms period) are simply unavailable at any sample rate above about 2 MHz. P2 is a multiplier on every other cause.

**The conjunction argument.** These constraints bind simultaneously and largely independently. In a conjunctively-constrained system, relieving one constraint yields approximately zero *measured* gain, because the others remain binding. That is a precise prediction of the observed phenomenology: nine interventions, nine null results, one intervention that reallocated error without reducing it (lambda = 8.0), and one representation change that changed the *kind* of error without reducing the total (STFT). Any diagnosis that names a single cause fails to explain why the null results were so uniformly null rather than partially helpful.

The corollary is uncomfortable and should be stated: **partial implementation of the recommendation may underdeliver relative to a linear extrapolation of its parts.** The exception is P1, which is close to individually sufficient for the triad, which is why it is ranked first and placed early in the roadmap.

---

## 2. The recommended architecture

### 2.1 Design principle

Each representation has a **kernel**: a set of distinctions it cannot make. The observed error structure of each model is the image of its representation's kernel.

- Bandwidth-normalized input: kernel contains absolute scale. Cannot separate signals differing mainly in rate.
- Conv/attention with time-averaged pooling: kernel contains cyclostationary structure at alpha != 0.
- log-magnitude STFT: kernel contains the instantaneous-frequency trajectory. Collapses narrowband-ridge classes onto cw.
- 4096 fixed samples: kernel contains burst repetition structure above roughly 2 MHz sample rate.

The design principle is therefore not "choose a better representation" but **choose a set of representations whose kernels intersect trivially, and fuse them so that a class distinguished by any one member is distinguished by the whole.** Fusion must be structured so a member cannot import its kernel's failures into regions where another member is already correct. That is the specific reason for distance-level gating and routed arbitration rather than embedding concatenation or probability averaging.

### 2.2 Prerequisite: corpus and evaluation changes

These are not optional and they gate everything downstream.

**C0. Generator audit (blocking, hours).** Read the Atom-SignalLab modulator. Verify it renders distinct modulation indices (h = 0.32 for bluetooth-classic, h = 0.5 for bluetooth-le, h = 0.5 for gsm-normal-burst) and distinct Gaussian BT products (0.5, 0.5, 0.3). **If one shared Gaussian-CPM modulator with fixed h and BT serves all three profiles, the discriminative information is absent from the corpus and no front end, cyclostationary or neural, can recover it.** That single fact would explain a plateau surviving nine interventions, three corpus scales, a generator upgrade, and 750,000 episodes better than any modeling hypothesis. This check dominates every other next step.

**C1. Absolute-time windows.** Replace the fixed 4096-sample window with a **fixed 10 ms observation** at whatever the native sample rate is. Sample counts then range from 10,000 (1 MHz) to 300,000 (30 MHz). 10 ms covers 2.2 GSM TDMA frames, 16 Bluetooth slots, 16 frequency hops, and thousands of DSSS chips. Every timing structure in the problem becomes observable for the first time. Variable-length input is a non-issue for a linear-time SSM trunk and impossible for quadratic attention, which is a load-bearing architectural consequence, not a convenience.

**C2. Decouple sample rate from class.** Draw fs independently of profile when regenerating. If fs is class-correlated by construction, any model handed fs (or handed native sps) is exploiting a receiver-side shortcut, and both your existing native-versus-resampled result and every scale-injection result below become uninterpretable.

**C3. Held-out-parameterization evaluation split.** Train on one distribution of CFO, multipath delay spread, and IQ imbalance; evaluate on a *shifted* one. This is the CSPB.ML.2018 -> 2022 protocol. Given the 97.5 -> 23.7 result cited above, reporting only in-distribution accuracy on a single-generator synthetic corpus is not a defensible measurement. Every number in the roadmap should be reported on both splits.

**C4. Measure the empty-window fraction.** For each bursty profile, measure what fraction of windows contain no signal energy at all (BLE advertising in particular has very low duty cycle). Windows containing only noise are label noise by construction and put a hard floor under the achievable accuracy. This is a corpus property, not a model property, and it is trivially measurable.

**C5. Corpus scale.** 200k to 500k samples across 37 profiles, at 10 ms each. Regeneration is free except for compute, which is unconstrained by the terms of this thesis.

### 2.3 Input views

Five views, chosen so their kernels intersect trivially.

**View A. Native complex I/Q, 10 ms, DC-centered, RMS-normalized, NOT bandwidth-resampled.** Frequency invariance is genuine nuisance removal and should be kept. Scale invariance should be achieved by *providing the scale as a feature*, not by destroying it.

**View B. Physical scalars (the direct antidote to P1).** Log-scaled, robust-normalized:
- log10(occupied bandwidth in Hz)
- log10(estimated symbol or chip rate in Hz), taken from the dominant non-conjugate cycle-frequency peak
- log10(estimated burst duration in seconds), duty cycle, log10(estimated burst repetition period)
- samples-per-symbol, and the decimation factor the front end applied
- **Not** fs itself, unless C2 is verified. fs is a receiver property; bandwidth and symbol rate are class properties.

**View C. Cyclic-domain surface (the direct antidote to A1).** Non-conjugate and conjugate spectral correlation over (alpha, f), computed by SSCA. Represented on **two alpha axes simultaneously**: one normalized by the estimated symbol rate (scale-free, yields the dimensionless ratios), one in absolute Hz (yields the 270.833k / 1M / 11M separation directly). Derived scalars appended:
- alpha-profile peaks and magnitudes, non-conjugate and conjugate
- chip-to-symbol cycle-frequency ratio (11:1 for Barker DSSS)
- **conjugate-SCF presence indicator** (identically zero for QPSK/8PSK/16QAM/OFDM, non-zero for BPSK/OOK/MSK/GMSK/GFSK). This is the separator for the dsss-versus-ofdm pair that absolute bandwidth cannot reach.
- the ratio **2h = (conjugate cycle-frequency doublet spacing) / (non-conjugate symbol-rate cycle frequency)**, predicting 1.00 for GSM-GMSK and BLE, and 0.56 to 0.70 for Bluetooth Classic.

*Honesty flag:* the general 2h formula is an extrapolation from CPM theory, verified only at the h = 0.5 special case in the source material and not found stated in that form in any paper. **Validate it numerically on one bluetooth-classic and one gsm sample before building on it.** Also note plainly: 2h does **not** separate BLE from GSM, because both are h = 0.5. That pair must be separated by absolute rate (View B) and BT product.

**View D. Higher-order cyclic cumulants, orders 2, 4, 6**, at alpha = (n - 2m) f0 +/- k/T0, k = 0..5, giving 165 estimates. Required because second-order SCF alone *collides* MSK/GMSK/OQPSK/SQPSK; order >= 4 is necessary to break that tie. This is the feature set that survived the CFO distribution shift (92.3 -> 91.6) that destroyed a raw-I/Q net (97.5 -> 23.7).

*Honesty flag:* Snoap et al. used 32,768-sample instances. At 4096 you have 8x less integration and correspondingly noisier estimates. At 10 ms native this concern largely evaporates (10 ms at 30 MHz is 300,000 samples), which is another reason C1 is a prerequisite rather than an enhancement.

**View E. Multi-resolution spectro-temporal tensor with an absolute-time axis (the antidote to E2 and P2).** Three resolutions with window and hop specified in **microseconds**, not samples, using the known fs: approximately (4 us / 1 us), (32 us / 8 us), (256 us / 64 us). Four channels per resolution:
- log magnitude
- **instantaneous frequency** (frame-to-frame phase difference, group-delay corrected). This is the direct antidote to cw absorption: cw has near-zero IF variance, GFSK has a two-level IF telegraph, FM has a continuously varying IF, AM has near-zero IF with amplitude variation. The one channel that disambiguates precisely the classes the log-magnitude view collapsed.
- envelope on/off at the long window (duty-cycle detector)
- conjugate-lag magnitude

Specifying the STFT in physical units fixes a defect in the existing 2D experiment that I believe is decisive: with a fixed 8-sample hop across 63 sample rates, the time axis of your spectrograms spans a 30x range, and fixed 8x8 ViT patches therefore cover physically incomparable regions across the corpus. You already enforce frequency, scale, and power invariance; time-axis invariance is the one you never enforced. Multi-resolution is supported by Schindler et al. 2017 (+3.56 percent absolute over the best single resolution) and required by the 3.7x and 11x symbol-rate span of your classes.

**View F. The existing 12 hand features**, retained, robust median/MAD normalized (keep intervention 5, it fixed a real bug), on a protected path (section 2.6). Retained because they demonstrably carry dsss-separating signal and because ablating a known-good channel for elegance is bad science.

### 2.4 Front end

**Differentiable, scale-emitting bandwidth matching, not a hard resample.** Keep DC downconversion. Replace the hard "resample to fractional bandwidth 0.5" with a differentiable bandwidth-matched decimation (the ZoomSpec AHLP construction) whose **applied scale factor is emitted as a feature into View B**. The network then sees both the normalized waveform and the normalization it received. This is precisely the "approximately equivariant" regime that Wang, Walters and Yu (ICML 2022) found beats both unconstrained and strictly-equivariant models, replicated across at least four follow-ups.

**No hard burst detector or segmenter.** Justification, and this is a correction to the framing in the brief:
- The two-stage detect-then-classify literature (WRIST, Vagollari, Xing) exists for **wideband multi-emitter scenes where you cannot classify what you have not localized in frequency**. Your windows are already single-signal and already frequency-centered. That justification does not transfer.
- No published paper runs the controlled aligned-versus-unaligned A/B. WRIST explicitly does not compare against fixed-window classification.
- **Cyclic feature magnitudes are time-shift invariant by construction**: delaying by t0 multiplies R_x^alpha(tau) by exp(-j 2 pi alpha t0), leaving |R| unchanged. The cyclic branch *dissolves* the alignment problem rather than requiring you to solve it first.
- Learned alignment is a documented weak result: Radio Transformer Networks (O'Shea et al., Asilomar 2016), the canonical paper, bought roughly 1 dB equivalent SNR, and its own authors declined to claim it achieves synchronization.

### 2.5 Per-view encoders

- **Branch A (waveform).** Complex-valued multi-scale dilated conv stem with **CReLU**, striding by 256 to token rate, then a **12-layer bidirectional Mamba-2 (selective SSM) trunk, d_model 512**. Approximately 40M parameters. At 300,000 samples the stem yields ~1,172 tokens.
  - *Why SSM and not attention:* MAMCA (IEEE Communications Letters 2024) reports Mamba *losing* to a CNN-LSTM at length 128 (60.79 vs 61.42) and *winning by 8.6 points* over MCLDNN at lengths up to 4096. Your length is 4096 today and 300,000 under C1, deep in the favorable regime. More decisively, quadratic attention over 300,000 samples is infeasible while linear-time SSM is trivial, so C1 and the backbone choice are coupled. Flag: MAMCA is a single-team claim on its own TorchSig-QAM construction; the bidirectional variant is established in audio and EEG but unpublished for RF.
  - *Why complex-valued but not phase-equivariant:* Kumar's Proposition 1 establishes that a complex linear layer is exactly the U(1)-equivariant subspace of a stacked-real two-channel layer, so strict equivariance *strictly removes capacity* from what you already have. SurReal (TNNLS) is the published precedent for exactly your FNO result: hard scaling-and-rotation invariance bought ~100x parameter efficiency and a materially lower ceiling (76.1 percent at 10 dB on RML2016.10a versus 92.4 percent for an unconstrained complex CNN), with the authors noting it "underperforms the baselines at lower SNRs."
  - *Why CReLU and not modReLU:* Cole et al. (arXiv:2004.01738) and Kumar 2026 independently find CReLU significantly better than modReLU and zReLU. Your FNO used modReLU, whose learnable dead-zone bias can kill all units. Choosing phase equivariance forced you into a bad activation.
  - *Calibrate expectations:* Kumar's controlled study shows the complex-versus-real gap collapsing from +22.94 points to +2.46 points under independent per-family tuning over the same search space, with the inflated gap attributable to dead seeds in the *real* baseline. **Tune the real and complex families independently, never at a shared trial index.** If you see a 20-point gap at matched hyperparameters, you have measured a dead seed.
- **Branch C/D (cyclic).** Small 2D ResNet over the (alpha, f) SCF surface, ~15M parameters, plus an MLP over the 165 cyclic-cumulant estimates, ~0.4M.
- **Branch E (spectro-temporal).** ConvNeXt-T or ViT-S applied across the three resolutions with shared weights, ~25M.
- **Branch B/F (scalars and hand features).** MLP, ~0.1M.

Note that this preserves the DNA of the best-trending result you have (the 98k complex CLDNN at 0.857 and still climbing at episode 1200: complex convolutions, envelope branch, IF branch, per-branch temporal pooling). It replaces BiLSTM with BiMamba-2 and adds the scale and cyclic branches. It is an extension of your strongest trajectory, not a replacement for it.

### 2.6 Fusion: two levels, deliberately not concatenation

**Level 1: per-branch prototypes and distance-level gated fusion.** Each branch b emits its own L2-normalized embedding e_b and maintains its own prototype set. The score is

    s(x, c) = sum_b w_b(x) * ( - d( e_b(x), P_{b,c} ) )

with w_b(x) a learned input-conditioned gate (softmax over branches from a small MLP on a shared summary vector).

*Why this and not embedding concatenation:* this is the direct structural fix for A3. Concatenating 12 informative dimensions into a high-dimensional vector and linearly mixing lets the loss trade the informative direction away; keeping per-branch distances means the dsss-separating hand-feature direction contributes its own additive term that cannot be diluted. The input-conditioned gate is the M-LSCANet-style conditional fusion that permits a branch to be *switched off* in the regime where it is actively harmful, which is exactly the cw-absorption case.

*Honesty flag:* **no published work does multi-representation fusion inside an episodic prototypical pipeline.** Two independent literature reviews flagged this as a gap. This construction is my own and is unguided by precedent. The intermediate-fusion preference it inherits (Lin 2026; Bai 2022 at +4.5/+6.8 absolute; Han 2021) is from standard-softmax settings.

**Level 2: routed triad arbitration (Hou et al., Scientific Reports 2023).** If the 7-way argmax lands in {bluetooth, gsm, dsss}, hand the sample to a **triad arbiter** whose output alphabet is *exactly* those three classes, and whose inputs are weighted toward Views B, C, D, E.

*Why routing and not ensembling:* your own arithmetic decides this. The STFT view eliminated 93 gsm->bluetooth errors and introduced at least 135 cw errors. Naive probability averaging or embedding concatenation imports the second number along with the first. Wood et al. (JMLR 2023) formalize the diversity/accuracy tradeoff and predict net-negative when the diverse member introduces a new error class. PANNs, the most-replicated waveform-plus-spectrogram fusion result in existence, gained only +0.008 mAP over the better branch (0.439 vs 0.431) despite the waveform branch being 4.2 mAP points worse alone. Hou's cascade, by contrast, achieved **+17 and +18.2 percent on the confused pair** and +2.67/+3.22 overall, precisely by giving the specialist a restricted output alphabet. Routing captures the 93 without the 135.

Note the important distinction from hierarchy-aware *losses*, which fail: Bertinetto et al. (CVPR 2020) show **every** hierarchy-aware method losing top-1 to flat cross-entropy on both tieredImageNet-H and iNaturalist-19, including the YOLO-v2-style top-down cascade (34.24 vs 31.58; 45.00 vs 44.04). The difference is that a hierarchy-aware loss re-weights the *same* model on the *same* features, whereas a routed specialist gets *different information* (different views, restricted alphabet, dedicated capacity). Only the latter wins. A naive coarse-then-fine cascade would be actively harmful here, because the natural coarse family ("bursty constant-envelope digital") contains all three triad members, so stage 1 would route all three into one bucket and hand stage 2 the identical hard decision plus an unrecoverable routing error.

### 2.7 Head and inference rule: profile-level multi-prototype

Maintain **one prototype per profile (37), not per class (7)**. Class score = soft-min over the profiles belonging to that class:

    s(x, c) = -tau * logsumexp_{p in c} ( -d(e(x), P_p) / tau )

Train episodically at the **profile** level (20-way 5-shot over the 37 profiles), with an auxiliary 7-way coarse loss computed through the soft-min pooling. This is the ISMIR-2021 hierarchical-prototype construction and the *supervised* version of Infinite Mixture Prototypes.

*Justification.* IMP's largest reported gain is on exactly the superclass condition (Omniglot alphabet: 65.6 -> 92.0, +26.4 absolute; alphabet-train to character-test 82.1 -> 95.4) and its gain on unimodal classes is nil (98.4 vs 98.2). Mensink's NCMC on ILSVRC'10 gives the same story on a different task family (NCM 39.0 -> NCMC k=10 34.6 top-5 error at 512 dims), and critically shows that **most of the gain is recoverable by switching to multi-centroid at test time only, with a metric trained for k=1** (39.0 -> 36.3). DNC (ICLR 2023) shows the gain scaling with heterogeneity and reversing when over-fragmented (ADE20K K=10 44.3, K=20 44.0), which is a warning against setting k too high.

Every one of IMP (DP-means), GEORGE (feature clustering), DNC (Sinkhorn), and PALM (reciprocal-neighbour soft assignment) exists to *infer* subclass structure that nobody labelled. **You already have it, as ground truth, as 37 named profiles.** You are in a strictly better position than any of those papers.

*A geometric point worth stating precisely, because it forecloses the obvious alternative:* a single-prototype nearest-centroid rule induces a Voronoi tessellation, so every class region is an intersection of half-spaces and therefore **convex**. A linear softmax head assigns argmax_c (w_c . x + b_c), so its class regions are *also* convex. **Switching to a learned linear head does not fix multimodality.** What the linear head buys over a centroid is that w_c is free rather than pinned to the class mean, which is a real but small gain, and it is why the measured prototype-versus-linear gap on matched embeddings is a few points rather than tens. Only a *union* of convex cells (multiple prototypes) or a nonlinear head produces non-convex class regions. This is the precise reason the recommendation is multi-prototype rather than "use a proper classifier."

### 2.8 Loss

1. **Profile-level prototypical cross-entropy** over negative squared Euclidean distance on L2-normalized embeddings, applied per branch and on the gated fusion.
2. **Coarse 7-class prototypical cross-entropy** on the soft-min-pooled distances (hierarchical multi-task).
3. **Norm-softmax auxiliary at the profile level. Not ArcFace, and never at the 7-class level.** The Fraunhofer study (arXiv:2510.23186), the closest published setup to yours, found norm-softmax *beating* ArcFace for 1D input and called the ArcFace underperformance surprising. Musgrave et al. (ECCV 2020) found ArcFace and CosFace *worse* than a well-tuned plain contrastive loss on CUB200, the most fine-grained benchmark in their controlled suite (67.50 and 67.32 versus 68.13 P@1). More fundamentally, all angular-margin and center losses **enforce unimodality per class by construction**. Applied to your 7 pooled labels they would optimize for the assumption that is false, and I predict they would reproduce your lambda = 8.0 collapse exactly. At the 37-profile level the unimodality assumption is approximately true and the margin is safe.
4. **Burst-phase invariance auxiliary (Mod-CL construction).** Positive pairs drawn from two disjoint random windows of the **same emission**, pulled together. This targets the arbitrary-burst-offset nuisance directly and requires no pretraining stage. Flag: Mod-CL is a May-2026 preprint with no replication; its own gain was +2.3 points at N=100 in a different setting.
5. **No class-level prototype repulsion.** You measured that it fails and the theory says why. If repulsion is wanted, use PALM-style *prototype-level* contrastive repulsion between **profile** prototypes, where the operation is well-posed.
6. **Damped branch-consistency term.** A Jensen-Shannon divergence penalty between branch predictive distributions (Bai et al. 2022 used exactly this and found unconstrained branch diversity needed damping), weight tuned low.

### 2.9 Training procedure

- Episodic, 20-way 5-shot over profiles, 200k to 500k episodes, AdamW, cosine schedule, per-branch independent hyperparameter search.
- Enrollment pool should preserve the impaired-to-clean ratio at inference, but **report accuracy stratified by impairment**, because a 3:1 impaired-to-clean enrollment pool biases every prototype toward the impaired manifold and that is a design decision that should be measured rather than inherited.
- **Stability by construction** for any pole/zero or SSM parameterization: diagonal, log-magnitude or negative-real-part parameterization (S4D/S5 lineage). Your 0.716 -> 0.29 collapse at episode 1666 with no recovery is the textbook signature of poles migrating to or outside the unit circle. Initialization inside the stable region provably does not keep you there during training.
- **Masked-reconstruction pretraining is optional and ranked last.** The SSL literature is unanimous and well-replicated across at least five independent groups: gains are +15 to +40 points at 1 to 20 labels per class and **+0 to +2 points at 100 or more per class** (Davaslioglu +1.2 at 90 percent labels; Kanu +0.17 at 100 percent and *minus* 0.08 at 1 percent; Mod-CL +2.3 at N=100 while still 1.3 below fully supervised; IQFM +0.7 on RML2016a at 500/class and *minus* 0.6 on fingerprinting). You are at 2000 per class and regenerable. If you pretrain, use masked reconstruction, not contrastive (reconstruction-only 68.21 vs contrastive-only 53.41 at 10 percent labels, and adding contrastive to reconstruction made it strictly worse at every label fraction). The real benefit is convergence speed (WavesFM reports up to 5x), which is a research-velocity argument, not an accuracy argument.
- Do **not** pretrain on your own 14k synthetic corpus. The published negative result (Aboulfotouh et al., arXiv:2411.09849) pretrained on 24 seconds of data and the pretrained model *lost* to from-scratch on both downstream tasks, with real-versus-simulated mismatch cited. Your corpus scale is closer to that failure than to EMind's 81M samples.

### 2.10 Parameter and compute budget

| Component | Params |
|---|---|
| Branch A: complex stem + 12-layer BiMamba-2, d=512 | ~40M |
| Branch C: SCF surface 2D encoder | ~15M |
| Branch D: cyclic-cumulant MLP | ~0.4M |
| Branch E: 3-resolution shared-weight ConvNeXt-T / ViT-S | ~25M |
| Branch B/F: scalar + hand-feature MLP | ~0.1M |
| Gate, fusion, heads | ~2M |
| Triad arbiter (reduced copies of C/D/E) | ~10M |
| **Total** | **~92M** |

This sits just under the 100M ceiling the wireless-foundation-model literature converges on (the WiFo-2 position paper argues explicitly that LLM-style scaling does not transfer and that wireless FMs stay below 100M).

Compute: SSCA over 300,000 samples dominates preprocessing at roughly a few hundred MFLOPs per sample, but it is a one-time, embarrassingly parallel, cacheable pass (order 1 to 2 days on 64 CPU cores for 500k samples). Training: 8 x A100/H100 for 3 to 7 days. Inference: the SCF surface is the bottleneck, and the shipping answer is a **reduced-alpha SSCA** evaluated only at the alpha values implied by the estimated symbol rate and its harmonics, which is 10 to 100x cheaper.

### 2.11 What is deliberately excluded, and why

| Excluded | Reason |
|---|---|
| A bigger or longer-trained transformer over raw I/Q | Already falsified in-house by the 750k-episode run. Externally, plain patchified ViT over raw I/Q loses to conv-recurrent hybrids (FEA-T 59.80 vs MCLDNN 60.99 on 2016.10a), consistent with your CNN 0.891 > ViT 0.876. |
| Strict phase-equivariant / complex neural operator | SurReal is the published precedent for the ceiling. Kumar's U(1)-subspace proposition proves the constraint strictly removes capacity from the stacked-real net you already have. |
| ArcFace / CosFace / center loss at the class level | Enforce unimodality; predicted to reproduce the lambda=8.0 collapse. |
| Naive ensembling of the 1D and 2D ViTs | Your own 93-versus-135 arithmetic, plus Wood et al.'s diversity/accuracy tradeoff, predicts net-negative. |
| Hard burst detector or segmenter as stage 1 | No published A/B supports it for single-emitter windows; cyclic magnitudes are shift-invariant anyway; RTN's learned alignment bought ~1 dB. |
| Wavelets / scalograms | The one clean head-to-head (Phan 2024) went to spectrograms at every SNR and on 3 of 4 signal types, at 1/37th the compute, with the wavelet winning only on the impulsive class by +0.003 AUC. Your discriminative structure (577 us bursts, 4.615 ms frames) is slow structure, not the sharp-transient regime where wavelets earn their keep. |
| Constellation-diagram branch | Structurally blind to your actual problem: GFSK and GMSK are both constant-envelope continuous-phase and map to the same ring. Also requires carrier and symbol timing recovery and enough symbols to populate the cloud, which your arbitrary-offset windows do not guarantee. |
| Hierarchy-aware loss (HXE, soft labels, top-down cascade) | Bertinetto et al.: every such method loses top-1 to flat CE. Routed arbitration is a different mechanism and is included. |
| RF foundation-model pretraining / downloaded checkpoints | Gains collapse at your label density; input-shape and domain mismatch; no public weights for any I/Q FM except EMind (checkpoint presence unverified). |
| Audio/speech SSL transfer (wav2vec, HuBERT) | A targeted literature search found **zero** published transfers to RF I/Q classification. The only quantified cross-modal-to-RF number is ImageNet initialization, which is a wash at high label counts (61.1 vs 61.2). |

---

## 3. Why this breaks the ceiling

### 3.1 Mechanism by mechanism, against the specific evidence

**M1. Absolute scale restored (View B) breaks the triad by brute physical separation.** GSM 200 kHz / 270.833 ksym/s, Bluetooth 1 to 2 MHz / 1 Msym/s, 802.11b HR/DSSS 22 MHz / 11 Mchip/s. Factors of 3.7x and 11x in rate, 5x and 22x in bandwidth. Every model in the program was denied these quantities. This is the axis none of the ten interventions touched, and it is the axis on which your one representation-level experiment already produced a 4.4-point effect (E6). Expected to be the single largest gain.

**M2. Cyclic branch (Views C and D) makes alpha != 0 computable, breaking dsss and providing shift-robust redundancy.** The chip-rate non-conjugate line sits at approximately 0.5 x occupied bandwidth, a ratio that survives bandwidth normalization; the chip-to-symbol ratio is 11:1 for Barker; the conjugate-SCF presence indicator is identically zero for OFDM/QAM and non-zero for DSSS-BPSK and all CPM. This explains E3 mechanistically: the confusion originates in the learned path because the learned path is architecturally incapable of computing a quadratic lag-product beaten against a complex exponential. It also addresses the pair that M1 cannot (20 MHz WiFi-OFDM versus 22 MHz HR/DSSS) and provides the distribution-shift robustness that a purely I/Q model demonstrably lacks (97.5 -> 23.7 versus 92.3 -> 91.6).

**M3. Profile-level multi-prototypes dissolve E5.** With one prototype per profile, "repel bluetooth from gsm" becomes "repel bluetooth-classic from gsm-normal-burst", which is a well-posed operation on two unimodal populations. The pathology where pushing a mean out of a gap drags it out of both modes simply cannot occur. Bluetooth is no longer required to be one point. This also removes the objective misspecification identified in E4: the loss no longer instructs the model to collapse a bimodal class.

**M4. Distance-level gating plus triad routing captures the STFT win without the STFT failure.** The gate can suppress the spectro-temporal branch where it is harmful; the arbiter cannot emit cw because cw is not in its alphabet. This converts your measured minus-42 into a plus-93.

**M5. The IF channel removes cw absorption at its source.** Log-magnitude discards the phase derivative; a static tone, a low-index FM signal, a quiet AM passage, and a GFSK burst all become narrow bright ridges. The IF channel restores the one quantity that tells them apart.

**M6. Absolute-time windows (C1) plus absolute-time STFT axes make burst structure a real feature for the first time.** At 10 ms, GSM TDMA framing (4.615 ms), Bluetooth hopping (625 us), slot structure, and duty cycle are all observable at every sample rate. At 4096 samples they are arithmetically unavailable above roughly 2 MHz. This is not an opinion, it is division. It also collapses the within-class spread that made the 0.713 centroid distance look like irreducibility, and it multiplies M2 by giving the cyclic estimators up to 70x more integration.

### 3.2 What is NOT breakable by architecture

I want to be plain about three things, because a thesis that promises architecture can fix everything is dishonest.

**(a) If the simulator does not render the distinction, nothing recovers it.** If Atom-SignalLab routes bluetooth-classic, bluetooth-le, and gsm through one Gaussian-CPM modulator with a single h and BT product, the discriminative information is absent from the corpus, the entire diagnosis above is superseded, and the fix is a **simulator change**, not a model change. This hypothesis explains the nine null results, the three corpus scales, the generator upgrade, and the 750k episodes more parsimoniously than any modeling hypothesis. It costs an afternoon to check and it must be checked first.

**(b) bluetooth-le versus gsm-normal-burst is irreducible *in the current representation* and only in it.** Both are Gaussian-filtered binary CPM at h = 0.5. After DC-centering, bandwidth normalization, and power normalization, the residual differences are the BT product (0.5 versus 0.3), a second-order pulse-shape effect, and burst structure, which the 4096-sample window frequently cannot see. This is a **preprocessing-conditional** irreducibility, not a physical one: with absolute rate restored, 1 Msym/s versus 270.833 ksym/s and 2 MHz versus 200 kHz make the pair trivial. The point is that you manufactured this irreducibility yourself, which is good news, because it means you can un-manufacture it.

**(c) Part of the residual is definitional and requires a taxonomy change, not a model change.** Your label set mixes levels of abstraction: bluetooth and gsm are *technologies*, dsss is a *spreading technique*, ofdm is a *modulation family*. 802.11b HR/DSSS and 802.11a/n/ac/ax OFDM are both WiFi, so a WiFi capture's label depends on which mode it used, while a Bluetooth capture's label does not depend on mode at all. An 802.11b long-preamble frame contains a DSSS preamble followed by a payload that may be CCK. A window landing in an inter-burst gap contains no signal and is label noise by construction. None of that is fixable by any architecture.

The non-architectural fix is a **product-level restructuring**: make the model's native output the 37 profiles, and make the shipped 7-label taxonomy a *view* over it. Then the taxonomy can be changed, split, merged, or re-leveled without retraining, and the ambiguity becomes a labeling policy decision rather than a modeling failure. This is, in my judgment, the highest-value non-architectural change available and it costs almost nothing given that you already have the profile labels.

---

## 4. Ablation-ordered roadmap

Ordered by expected gain per unit effort. Magnitudes are on overall accuracy on your current impairment-inclusive split unless noted. Confidence is my subjective probability that the direction and rough magnitude hold.

### Step 0. Generator audit and per-sub-profile diagnostic. Hours. Gating.
- Read the modulator; confirm distinct h (0.32 / 0.5 / 0.5) and BT (0.5 / 0.5 / 0.3).
- Recompute your robust z-distance diagnostic **per sub-profile pair** (2 bluetooth x 7 gsm = 14 pairs) rather than per class pair.
- Measure the fraction of windows per profile containing no signal energy (C4).
- **Predicted if the thesis holds:** bluetooth-classic versus gsm-normal-burst > 1.5; bluetooth-le versus gsm-normal-burst < 0.7; pooled pair 0.713 sits between them. That single table decides whether you have a label problem, a representation problem, or a corpus problem.
- Gain: none directly. Value: it can invalidate everything below. Confidence that it is informative: **very high (0.95)**.

### Step 1. Frozen-network multi-prototype at inference. Hours. No retraining.
Keep the 38k model exactly as it is. Replace the 7 class prototypes with 37 profile prototypes computed from the same enrollment pool (you have the labels; no clustering needed). Classify by min-over-sub-prototypes. Also run plain k-NN over the enrollment pool as a sensitivity probe.
- Precedent: Mensink's NCMC-test recovered most of the multi-centroid gain with a k=1-trained metric (39.0 -> 36.3 top-5 error).
- **Expected:** dsss->bluetooth down 20 to 40 percent relative; overall **+0.005 to +0.02**.
- **Confidence: medium-high (0.65).** Best expected-gain-per-unit-effort item in the entire document.

### Step 2. Inject absolute physical scalars into the existing model. One day.
Add log10(occupied BW in Hz), log10(estimated symbol/chip rate in Hz), log10(burst duration), duty cycle, samples-per-symbol, robust-normalized, to the hand-feature branch. Retrain the 38k model. **Verify C2 first** or the result is uninterpretable.
- **Expected: the largest single gain in the list.** gsm->bluetooth down by half or more; overall **+0.03 to +0.06** (0.891 -> 0.92 to 0.95).
- **Confidence: medium-high (0.7) on direction and rank; medium (0.5) on magnitude.**
- This is the decisive test of the primary diagnosis. If it does nothing, the thesis is substantially damaged (section 5, F1).

### Step 3. Train episodically over the 37 profile labels, soft-min pool to 7 at output. One day.
No architecture change, no API change, no UX change.
- **Expected:** overall **+0.01 to +0.03**; triad leakage down 30 to 50 percent relative.
- **Confidence: medium-high (0.65).** IMP's superclass result (+26.4 absolute) is the strongest analogue in the literature but is one paper in a different domain, and your classes are less extreme than Omniglot alphabets.

### Step 4. Minimal cyclic feature branch. Two to five days.
Do **not** start with the full SCF surface. Start with 32 to 64 scalars: non-conjugate and conjugate alpha-profile peak locations and magnitudes (both normalized by estimated R_s and in absolute Hz), the chip-to-symbol ratio, the conjugate-SCF presence indicator, and the 2h ratio. Validate the 2h formula numerically on two known samples before building on it.
- **Expected:** dsss->bluetooth toward near-zero; gsm versus bluetooth-classic materially improved; **no help for gsm versus bluetooth-LE**. Overall **+0.02 to +0.04**.
- **Confidence: medium (0.55).** The physics is well-replicated; the estimation from 4096 samples at 75 percent impairment is genuinely hard, and the reference result used 32,768-sample instances.

### Step 5. Absolute-time windows: regenerate at 10 ms per sample. One to two weeks.
- **Expected: +0.02 to +0.05**, concentrated in the triad, and it multiplies Step 4's gain by giving the cyclic estimators up to 70x more integration.
- **Confidence: high (0.85) on direction, medium (0.55) on magnitude.** The arithmetic (136 us < 577 us) is not contestable; how much the model exploits the newly-visible structure is.
- **This breaks the shipping constraint.** Everything in Steps 0 to 4 fits a 4096-sample, sub-200k-parameter shipped model. Step 5 does not. Treat this as the boundary between "improve the product" and "find the ceiling."

### Step 6. Absolute-time multi-resolution spectro-temporal branch with IF channels, distance-level gated fusion, triad routing. One to two weeks.
- **Expected: +0.01 to +0.03**, and specifically recovers the STFT view's 93-error gsm->bluetooth win without cw absorption.
- **Confidence: medium (0.5).** Hou's cascade precedent is strong but single-paper, and nobody has done multi-view fusion inside an episodic prototypical pipeline.

### Step 7. Backbone replacement: complex CReLU stem plus BiMamba-2 trunk, per-view encoders. Weeks.
- **Expected: +0.005 to +0.02.**
- **Confidence: medium-low (0.4).** Deliberately last despite being the most interesting part. MAMCA's crossover is a single-team claim; Kumar's controlled study puts the complex-versus-real gain at ~2.5 points, not 23; Musgrave's reality check shows metric-learning architecture and loss claims routinely evaporating under fair comparison; and ten interventions have already demonstrated that this problem is not backbone-limited.

### Step 8. Masked-reconstruction pretraining. Optional.
- **Expected: +0 to +0.02 accuracy; 3 to 5x convergence speedup.**
- **Confidence: high (0.85) that the accuracy gain is small.** This is the best-replicated finding in the entire SSL survey.

### Running throughout
Build the **held-out-parameterization split (C3) at Step 0** and report every number on both splits from then on. Steps 2 and 5 in particular are the ones most prone to measuring a shortcut rather than physics, and without C3 you cannot tell.

---

## 5. What would falsify this thesis

### F1 (primary). Absolute-scale scalar injection, Step 2.
- **Thesis right:** gsm->bluetooth drops from ~220/614 to under 100/614; overall gains at least +0.03.
- **Thesis wrong:** leakage lands within noise of 214 to 220. That would mean the model already possessed the scale information (leaking through sps in native mode) and P1 is not the binding constraint. The diagnosis would then have to fall back to D1+A2 and to simulator degeneracy.
- This experiment also disambiguates the two readings of E6. Run it on both native and resampled inputs. If **resampled + scalars >= native**, the native advantage was scale information (P1 confirmed). If **resampled + scalars < native** by a similar margin, the native advantage was resampling damage, P1 is weaker than claimed, and the recommendation to stop hard-resampling stands on a different basis.

### F2. Frozen-network multi-prototype and k-NN, Step 1.
- **Thesis right:** measurable drop in dsss->bluetooth, and plain k-NN over the enrollment pool materially outperforming nearest-prototype on the triad. That is the IMP alphabet signature (1-NN 92.4 versus ProtoNet 65.6).
- **Thesis wrong:** k-NN is within noise of nearest-prototype. That falsifies the multimodality cause (D1+A2) outright, cheaply, in an afternoon, and would mean the lambda=8.0 collapse needs a different explanation.

### F3. Hand-features-only classifier on the triad.
- **Thesis right (A3 confirmed):** a classifier on the 12 hand features alone achieves *better* dsss-versus-bluetooth separation than the full 32-dim embedding. That is a damning demonstration that early concatenation destroys information, and it justifies distance-level fusion directly.
- **Thesis wrong:** hand-features-only is worse than the embedding. Then the centroid z-distance was misleading (centroid distance is not separability, as flagged in 1.4), A3 is dead, and the dsss leak is attributable to A1 alone.

### F4. Cyclic-features-only classifier on the triad.
- **Thesis right:** at least 90 percent 3-way accuracy on {bluetooth, gsm, dsss} from cyclic features alone on clean samples.
- **Thesis wrong:** under 70 percent. Before concluding, re-run at 10 ms windows, because the 4096-sample estimation noise is a confound. If it is still under 70 percent at 10 ms, either the estimators are wrong or the generator does not render the physics (see F5), and M2 is dead.

### F5 (the one that can invalidate everything). Generator audit, Step 0.
- **Thesis right:** h and BT product are genuinely distinct across the three profiles.
- **Thesis wrong:** one shared modulator, fixed h, fixed BT. Then the corpus does not contain the distinction, no architecture can recover it, the ceiling is a data-generation ceiling, and the correct fix is a simulator change. I regard this as the single most likely way this whole thesis is wrong.

### F6 (cleanest quantitative falsifier). Cover-Hart bracket on the binary sub-problem.
Measure the asymptotic 1-NN error on bluetooth-versus-gsm, on a very large sample, (a) in the current preprocessed representation and (b) with absolute scalars restored. Cover-Hart gives Bayes_err <= NN_err <= 2 Bayes_err (1 - Bayes_err), so NN_err brackets the irreducible error from above and below.
- **Thesis right:** the bracket in (a) is high (NN error >= 0.25) and in (b) collapses toward zero. **The difference between those two numbers is a direct measurement of how much separability the preprocessing destroys**, which is the central quantitative claim of this thesis.
- **Thesis wrong:** the bracket stays high in both. Then the classes genuinely overlap even with absolute scale restored, the ceiling is definitional rather than architectural, and the correct response is the taxonomy restructuring in 3.2(c), not any of the architecture in section 2.
- This test needs no soft labels, which matters because the standard modern Bayes-error estimator (Ishida et al., ICLR 2023) requires multi-annotator soft labels that a generative corpus does not have.

### F7. Negative control on the backbone ranking.
If Steps 1 to 5 fix the triad and a subsequent BiMamba-2 swap adds no more than +0.02, the ordering in section 4 is vindicated. If instead a backbone swap **alone**, at 4096 samples, with no scalars and 7-class prototypes, breaks the triad, then my architectural ranking is wrong and the field's default answer ("use a better sequence model") is right. I consider this outcome unlikely given six architecture families already failed, but it is the cleanest test of the thesis's central ordering claim and it should be run.

---

## 6. Honest ceiling estimate

### 6.1 Where the current 10.9 percent error lives

At 0.891 overall on seven roughly-balanced classes, total error is about 10.9 percent. If dsss and gsm each run 25 to 30 percent error (the 1D ViT counts are 37.9 percent and 35.8 percent; the CNN at 0.891 is presumably somewhat better), those two classes alone contribute (0.27 + 0.27) / 7 = **7.7 points**. That leaves ~3.2 points spread across the other five, or ~4.5 percent each, which is consistent with bluetooth's reported 0.968 clean accuracy.

**The triad accounts for roughly 70 percent of all remaining error.** That is the single most important number in this analysis, because it means the ceiling and the triad are the same problem, and it means the payoff from fixing it is large rather than marginal.

### 6.2 Predicted accuracies

| Configuration | Current split | Held-out-parameterization split (C3) |
|---|---|---|
| Today | 0.891 | **unmeasured; I would predict 0.80 to 0.87** |
| Steps 1 to 3 (frozen multi-prototype, scalars, profile prototypes) | **0.93 to 0.95** | 0.89 to 0.93 |
| Plus Steps 4 and 5 (cyclic branch, 10 ms windows) | **0.96 to 0.975** | 0.93 to 0.955 |
| Full recommended architecture | **0.975 to 0.985** | **0.93 to 0.96** |
| Input map left as-is, any backbone, any budget | **0.90 to 0.92** | 0.80 to 0.88 |

The last row is the load-bearing prediction: **you are already within one to three points of the ceiling of your current observation map, and no amount of architecture, compute, or training will move you appreciably past it.** That is precisely what ten interventions and 750,000 episodes measured.

The held-out-parameterization column is a genuine extrapolation with wide error bars, but the direction is well grounded: an I/Q model on a single-generator synthetic corpus is at serious risk of a Snoap-style collapse (97.5 -> 23.7 under a single nuisance-parameter shift), while a cyclic-feature front end is not (92.3 -> 91.6). The recommended design should be *far* more shift-robust than the current one, and that is an argument for it that has nothing to do with accuracy on your current split. I would rate shift-robustness as the second most important reason to build it.

### 6.3 The irreducible floor

Given the taxonomy as it stands, and **assuming absolute scale is restored and windows are long enough**:

| Component | Estimated error contribution | Fixable by architecture? |
|---|---|---|
| Deep-impairment samples where multipath and AWGN destroy the features | 0.5 to 1.5 percent | No. Reduce by longer integration (Step 5) only. |
| bluetooth-LE versus gsm-normal-burst residual (both h = 0.5 Gaussian binary CPM) | under 0.3 percent *with* absolute scale; **large without it** | **Preprocessing-conditional**, not architectural. This is the whole thesis in one row. |
| Definitional ambiguity (dsss and ofdm both being WiFi modes; mixed levels of abstraction in the label set) | 0.5 to 1.5 percent | **No.** Requires taxonomy restructuring (3.2c). |
| Windows containing no burst at all (low-duty-cycle BLE advertising in particular) | unknown, **measure it** (C4); potentially large at 4096 samples, near-zero at 10 ms | Corpus/window problem, not a model problem. |

**Irreducible floor: roughly 1.5 to 3 percent error, i.e. a ceiling of 0.97 to 0.985**, of which the definitional component (0.5 to 1.5 percent) is permanently outside the reach of any model and can only be removed by changing what the labels mean.

If instead the window stays at 4096 samples and the bandwidth normalization stays as-is, **the floor is roughly 8 to 10 percent, i.e. a ceiling of 0.90 to 0.92.** Ten independent experiments have already established that number empirically. The purpose of this entire document is to argue that it is a property of the input map and not of the model, and to specify exactly which changes to the input map, the label granularity, and the inference rule move it.

### 6.4 The one-sentence summary

Stop searching the architecture space; the six families you tried all converged to the same answer because they were all asked the same, under-determined question. Restore absolute bandwidth and symbol rate, give the model a cyclostationary channel it can actually compute, stop averaging two physically distinct sub-profiles into one prototype, and lengthen the window until burst structure exists. The backbone is the last thing to change, not the first.