# SOTA AMC 2025-2026

## Findings

## SUMMARY UP FRONT

Yes, RF foundation models exist and are real published work (2024-2026), but the field is roughly where audio was in 2019, not where wav2vec 2.0 was in 2021. The models are small (0.3M-110M params), pretraining corpora are small-to-moderate (24 seconds to 81M short IQ snippets), almost none release usable weights, and -- critically for your situation -- **every measured gain is concentrated in the 1-100-labels-per-class regime and collapses to 0-2 accuracy points once labels are plentiful.** There are also published negative results, including from the most prolific group in the area.

---

# (1) PUBLISHED RF FOUNDATION MODELS FOR I/Q DATA

### EMind (the largest and most directly I/Q-native)
"EMind: A Foundation Model for Multi-task Electromagnetic Signals Understanding," Luo, Gui, Liu, Zhang, Zhang, Wang, Guo, Ma, Liu, He, Li, Qiu, Xie, Sun. arXiv:2508.18785 (Aug 2025); on OpenReview (forum v6byS5Dypp). Code: github.com/GabrielleTse/EMind (only model with a confirmed public repo).
- Architecture: masked autoencoder, transformer encoder-decoder, **110M params**, 12 encoder / 8 decoder layers, patch size 8, mask ratio 75%, reconstructing raw IQ.
- Pretraining: **81,117,525 samples** from 14 datasets (10 public, 4 self-collected): 75.8M "radio frequency", 3.7M communication, 1.5M radar, 120k interference. Signal lengths 128-4096 samples, sample rates kHz to hundreds of MHz. Length-adaptive multi-signal packing.
- Downstream (accuracy):
  - RML2016.10A AMC: EMind **62.51%** vs SpectrumFM 63.72% vs plain Transformer 59.27% vs ResNet 50.04%
  - RML2016.10B: 65.45% (SpectrumFM 65.35%, Transformer 63.10%)
  - RML2016.04C: 74.34% (Transformer 65.41%)
  - RML2018.01A: 63.83% (Transformer 59.45%)
  - ADS-B RF fingerprinting: 99.87% vs Transformer 78.77%, ResNet 84.51%
  - EM-AIS fingerprinting: 57.07% vs 39.12% / 42.02%
  - Interference ID: 81.70% / 79.19% vs ~77-80% Transformer
  - Few-shot RML2016.10A: 50 shots 48.38% (ResNet 38.97%), 100 shots 50.10% (ResNet 42.01%)
- **Honest reading: a 110M-param model pretrained on 81M samples LOSES to SpectrumFM on RML2016.10A and beats a from-scratch Transformer by only ~3 points on standard AMC.** The big wins are on fingerprinting (a task where pretraining on many emitters is obviously relevant) and in the sub-100-shot regime.

### SpectrumFM
"SpectrumFM: A Foundation Model for Intelligent Spectrum Management," Fuhui Zhou, Chunyu Liu, Hao Zhang, Wei Wu, Qihui Wu, Tony Q. S. Quek, Chan-Byoung Chae. arXiv:2505.06256, **accepted IEEE JSAC** (one of the few peer-reviewed entries here).
- Architecture: hybrid CNN + multi-head self-attention encoder (Conformer-like: FFN -> MHSA -> conv -> FFN with residuals). 16 encoder layers, latent dim 256, FFN 512, 4 heads, 128-symbol input, ~2.03e9 FLOPs/sample. Param count not stated.
- Pretraining: **~25 GB IQ**: RML2018.01A (2,555,904 samples), TechRec (9,747,800), self-collected (~9,000,000). Mask ratio 15%.
- Two SSL objectives: masked reconstruction (amplitude-phase domain) + **next-slot signal prediction** (predict symbol at slot N from slots 1..N-1, LSTM aggregation, MSE).
- Results: RML2016.10A recall 63.72% (ResNet 50.04, AMC_Net 59.10, Transformer 59.27); RML2016.10B 65.35; RML2016.04C 73.37 (+12.1% over prior best). Wireless Technology Classification on TechRec F1 **0.8226** vs best baseline MSNet 0.7481 (+9.3%). Spectrum sensing AUC 0.97 at -4 dB. Anomaly detection AUC 1.0.
- Ablation: masked reconstruction alone 0.6355 AMC / 0.7961 WTC; next-slot alone 0.6274 / 0.8076; combined best. Fine-tunes only ~2% of params.
- Few-shot: with 10% of data matches MSNet/AMC_Net at 100% of data on RML2016.10A.

### IQFM
"IQFM: A Wireless Foundational Model for I/Q Streams in AI-Native 6G," Omar Mashaal, Hatem Abou-Zeid (U. Calgary). arXiv:2506.06718 (Jun 2025). Preprint.
- **~0.3M parameters** (per the later Aboulfotouh et al. survey/multimodal paper). Contrastive SSL on over-the-air multi-antenna IQ, with "core" augmentations (cyclic time shift) plus task-specific augmentations.
- Headline: 99.67% modulation classification and 65.45% AoA with **1 labeled sample per class** ("7x better than supervised", "145x better than supervised").
- **The headline is from their own OTA testbed dataset. On the public benchmark it is nearly a wash:** RML2016a with 500 samples/class, IQFM **50.00% vs supervised 49.30%** (+0.7 pt). RF fingerprinting 96.05% vs supervised **96.64%** (SSL LOSES). Beam prediction 94.15% vs 89.53%.
- This is the single clearest illustration of the field's reporting pattern: enormous ratios in 1-shot on an in-house dataset, ~0 on a public benchmark at reasonable label counts.

### Foundation Model for Wireless Technology Recognition Using IQ Timeseries -- *closest in input shape to your setup*
Mohammad Cheraghinia, Eli De Poorter, Jaron Fontaine, Merouane Debbah, Adnan Shahid. arXiv:2505.19390 (May 2025).
- **860K params**, PatchTST-style patch transformer, input **(2, 4096)** stacked I/Q -- literally your tensor shape. Patch size and stride 128 -> ~32 tokens/channel, 4 encoder blocks, dim 128.
- 8 technology classes (Sigfox, LoRa, 802.15.4g, 802.11ah, LTE, 802.11ax, DVB-T, 5G-NR), sample rates 1/10/20 MHz, 482 MHz - 5.825 GHz, RTL-SDR / Anritsu / USRP B210 captures.
- Supervised pretraining, 8-class: **99.47%** vs CNN 54.89%, LSTM 69.01%, AE+CNN 78.01%, iTransformer 74.19%, vanilla Transformer 65.32%.
- **The interesting result for you: transfer to a held-out technology.** Fine-tuning on a technology excluded from pretraining -- supervised-pretrained variant 62.79% (LTE-excluded), self-supervised (masked patch reconstruction) variant **90.38%**. Zigbee-excluded: supervised 98.3%, SSL 94.64%. So masked-reconstruction SSL bought a large amount of generalization to an unseen class in one case and lost slightly in the other. Single paper, mixed direction.

### LWM-Spectro (spectrogram, not raw I/Q)
Namhyun Kim, Sadjad Alikhani, Ahmed Alkhateeb (ASU). arXiv:2601.08780 (Jan 2026). 12-layer transformer, d=128, 8 heads, MoE with 3 protocol experts (WiFi/LTE/5G) + router. Pretrained on **9.2M I/Q spectrograms** from DeepMIMO, 20 city scenarios. Masked spectrogram modeling at 70% mask + supervised contrastive at fine-tune. Joint SNR/Doppler classification: LWM fine-tuned 76.53% F1 at 5 samples/class -> 95.14% at 400; frozen encoder 47.41 -> 92.01; deep CNN 44.64 -> 90.54; ResNet-18 28.27 -> 50.13. Multi-protocol MoE 89.8% F1 at 100 samples/class, claims deep CNN needs 30x more data. **Again: the gap at 400 samples/class is 95.14 vs 90.54 -- 4.6 pts -- versus 32 pts at 5 samples/class.**

### WavesFM / 6G WavesFM
A. Aboulfotouh, E. Mohammed, H. Abou-Zeid. arXiv:2504.14100, published **IEEE (Xplore doc 11131142)**. Shared ViT backbone + task-specific MLP heads + LoRA. Handles spectrograms, CSI, and IQ arranged as OFDM resource grids. Four tasks: 5G NR positioning, MIMO-OFDM channel estimation, human activity sensing, RF signal classification. Claims superiority over individually-trained supervised baselines while sharing 80% of params, and **up to 5x faster convergence**. (Convergence speed is a more consistently reproduced benefit than final accuracy across this literature.)

### Others in the landscape (for completeness, less relevant)
- LWM (Alikhani, Alkhateeb et al., arXiv:2411.08872) -- masked channel modeling on CSI, not IQ classification.
- Multimodal Wireless Foundation Model, arXiv:2511.15162 -- ViT, **only ~7M params** (6.32M encoder), pretrained on **3,200 spectrograms + 3,200 IQ samples**. 20-class RF spectrogram classification: linear probe 63.59%, 2-block FT 84.14%, LoRA 89.41%; WavesFM 68.10 / 86.05. Note how tiny the pretraining corpus is -- this is the norm, not the exception.
- ReFormer (Yagna Kaasaragadda, Silvija Kokalj-Filipovic, Rowan U., arXiv:2501.00282, Dec 2024) -- **generative, not a representation-learning foundation model.** Autoregressive transformer over learned discrete RF tokens, used for data augmentation / "RF fakes". Often miscited as an RF foundation model; it is not one for classification.
- WiFo-2 / "Large Wireless Foundation Models: Stronger over Bigger," Xiang Cheng, Boxun Liu, Xuanyu Liu, Xuesong Cai (Peking U.), arXiv:2601.10963 (Jan 2026) -- position paper explicitly arguing **LLM-style scaling does not transfer to wireless**; wireless FMs stay under 100M params; "'large' refers to capability rather than sheer scale." Cites base-station compute (~100 TOPS) vs LLM (10,000+ TOPS) and the absence of large open RF corpora.

---

# (2) CONTRASTIVE / SSL PRETRAINING SPECIFICALLY FOR RF, WITH MEASURED NUMBERS

### Davaslioglu et al. 2022 -- the origin of the "order of magnitude" claim
"Self-Supervised RF Signal Representation Learning for NextG Signal Classification with Deep Learning," Kemal Davaslioglu, Serdar Boztas, Mehmet Can Ertem, Yalin E. Sagduyu, Ender Ayanoglu. **IEEE Wireless Communications Letters, 2022** (arXiv:2207.03046). Peer-reviewed.
- MoCo-v3. Five RF augmentations applied consecutively: DC shift (0-1e-4), time shift (+/-40 samples), amplitude scale (0.8-1.2), zero-masking (0-25 samples), AWGN.
- RML2016.10a, 11 classes. Accuracy vs labeled fraction:

| Labeled | MoCo-v3-512 | Xavier init (scratch) | ImageNet-pretrained |
|---|---|---|---|
| 0.5% (880) | **50.4%** | 9.1% | 14.9% |
| 1% (1,760) | **53.1%** | 11.4% | 34.5% |
| 5% (8.8K) | 55.2% | 49.2% | 53.1% |
| 10% (17.6K) | 54.6% | 53.6% | 54.4% |
| 90% (158.4K) | **62.4%** | 61.2% | 61.1% |

- **This one table is the most important thing in this entire report.** The 10x sample-efficiency claim is real and correctly stated -- but read the last row: at 90% labels the SSL advantage is **1.2 points**, and by 10% labels it is already **1.0 point**. This has been replicated in form by every subsequent paper that bothered to report the high-label end.

### Mod-CL (2026) -- the most rigorous SSL-for-AMC comparison I found
"Modulation Consistency-based Contrastive Learning for Self-Supervised Automatic Modulation Classification," Chenxu Wang, Shuang Wang, Lirong Han, Xinyu Hu, Hanlin Mo, Hantong Xing, Licheng Jiao. arXiv:2605.11875 (May 2026). Preprint, not peer reviewed.
- Key idea: build positive pairs from **different temporal segments of the same signal instance** rather than from augmented views, on the prior that modulation type is invariant across segments while symbol realization is not.
- Linear probing accuracy, RML2016.10A, M2SFE-tiny backbone:

| N/class | Mod-CL | SimCLR | MoCo | SemiAMC | MAC | Random init |
|---|---|---|---|---|---|---|
| 2 | 51.76 | 45.52 | 32.91 | 36.32 | 27.92 | 13.59 |
| 10 | 59.10 | 49.07 | 41.30 | 47.51 | 41.65 | 16.95 |
| 100 | 61.88 | 52.86 | 45.56 | 56.03 | 58.13 | 21.52 |

- **End-to-end fine-tuning** (the number that actually matters for you):

| Dataset | N | Mod-CL | SimCLR | Scratch | Fully supervised |
|---|---|---|---|---|---|
| 2016.10A | 2 | 55.91 | 49.25 | 35.75 | 63.07 |
| 2016.10A | 20 | 59.93 | 55.30 | 54.39 | -- |
| 2016.10A | 100 | 61.76 | 60.15 | 59.42 | -- |
| 2016.10B | 100 | 63.57 | 60.41 | 59.31 | 64.14 |

- At N=2 SSL is worth +20 points over scratch. **At N=100 it is worth +2.3 points, and full supervision (63.07) still beats the best SSL fine-tune (61.76).**
- Explicit negative finding about generic contrastive SSL for RF: SimCLR/MoCo objectives are "task-agnostic," and because augmented views preserve the symbol sequence, representations become entangled with nuisance factors (symbol realization, channel, noise) rather than isolating modulation-discriminative structure. MoCo is consistently ~10-15 points *worse* than SimCLR here.

### Masked reconstruction beats contrastive for RF -- replicated across three independent groups
"Enhancing Automatic Modulation Recognition With a Reconstruction-Driven Vision Transformer Under Limited Labels," Hossein Ahmadi, Banafsheh Saffari, Sajjad Emdadi Mahdimahalleh, Mohammad Esmaeil Safari, Aria Ahmadi. arXiv:2508.20193 (Aug/Sep 2025). RML2018.01A, 16 classes, -2 to +21 dB, 220k samples.

| Pretraining objective | 10% labels | 15% | 20% |
|---|---|---|---|
| Reconstruction only | **68.21%** | **71.01%** | **73.66%** |
| Reconstruction + contrastive | 66.40 | 70.24 | 71.41 |
| **Contrastive only** | **53.41** | 54.41 | 56.41 |

- Contrastive-only is ~15 points worse than reconstruction-only, and **adding contrastive to reconstruction made it worse at every label fraction.** Authors also note instability on high-order modulations with the joint objective.
- Supervised comparison: ResNet at 100% labels 78.50%, ViT at 100% labels 68.21%, semi-supervised ViT at 15% labels 71.01%. So the SSL ViT beats the supervised ViT but **never approaches the supervised ResNet** -- an architecture choice mattered more than the pretraining.
- Concordant: SpectrumFM's ablation (masked reconstruction 0.6355 AMC alone), EMind's pure-MAE design, LatentWave's latent-prediction framing all point the same way.

### LatentWave -- JEPA for RF, and an honest frozen-vs-supervised comparison
"LatentWave: JEPA Pretraining for Wireless Foundation Models," Ahmed Mohamed, Ahmed Aboulfotouh, Hatem Abou-Zeid. arXiv:2606.06373 (Jun 2026). ViT 8 layers/8 heads/256-dim, 16x16 patches, **~6.4M params** context+target encoders, ~490K predictor. Pretrained on only **~25,000 samples** across four datasets.
- Linear probing, frozen encoder:

| Task | LatentWave (region mask) | LatentWave (freq mask) | WavesFM | Supervised |
|---|---|---|---|---|
| RF signal classification | 80.9% | 66.1% | 80.3% | **86.5%** |
| Beam prediction | 51.6% | 63.1% | 51.2% | **88.9%** |
| LoS/NLoS | 92.9% | 93.4% | 93.4% | **95.9%** |
| 5G NR positioning (m) | 2.54 | 2.32 | 2.77 | **0.71** |

- **Frozen SSL representations lose to supervised on every task**, by 6 points on signal classification and by 3.5x on positioning. The paper is candid that masked-modeling approaches "bias representations toward low-level signal details" and require fine-tuning.
- Useful ablation for you if you ever revisit the STFT ViT: **masking geometry matters a lot and is task-dependent.** Region (2D block) and time masking preserve temporal discriminability needed for signal classification; frequency masking is +11 points on channel tasks but destroys classification accuracy (66.1 vs 80.9); **random masking consistently underperforms all structured strategies.**

### Kanu, Eshaghbeigi, Abou-Zeid 2025 -- the cleanest frozen/fine-tune decomposition
"Self-supervised Radio Representation Learning: Can we Learn Multiple Tasks?" arXiv:2509.03077 (Sep 2025). MoCo-v3, USRP X300 testbed, 68 GB / 4,609 files, 4 RX antennas at 5.88 GHz, 6 modulations, 1024-sample slices. Augmentations: antenna dropout and zero masking.

| Setting | Task | Scratch/supervised | SSL |
|---|---|---|---|
| **Frozen encoder** | AoA MAE | 8.93 deg | **4.28 deg** |
| **Frozen encoder** | AMC acc | 92.16% | **99.38%** |
| **Fine-tuned, 100% labels** | AoA MAE | 0.73 deg | 0.71 deg |
| **Fine-tuned, 100% labels** | AMC acc | 99.811% | 99.985% |
| Fine-tuned, 0.1% labels | AMC acc | 39.19% | **48.86%** |
| Fine-tuned, 1% labels | AMC acc | 98.03% | 97.95% (**SSL worse**) |
| Fine-tuned, 10% labels | AMC acc | 98.91% | 99.17% |

- At 1% labels SSL is already *slightly worse* than scratch. At 100% labels the gain is 0.17 points. This is a same-authors, same-lab confirmation of the Davaslioglu pattern.

### Earlier baselines cited in these comparisons
- SemiAMC: "Self-Contrastive Learning based Semi-Supervised Radio Modulation Classification," arXiv:2203.15932, IEEE MILCOM 2021 (Xplore 9652914), Illinois. SimCLR-style with rotation augmentation on I/Q. Appears as a baseline in Mod-CL's tables (56.03% at N=100 linear probe).
- "Residual Channel Boosts Contrastive Learning for Radio Frequency Fingerprint Identification," arXiv:2412.08885.
- "MCLRL: Multi-Domain Contrastive Learning with Reinforcement Learning for Few-Shot Modulation Recognition," arXiv:2502.19071.
- "Multi-representation domain attentive contrastive learning based unsupervised AMR," *Nature Communications* 2025 (s41467-025-60921-z) -- notable as a high-profile venue for unsupervised AMR.

---

# (3) DOES PRETRAIN-THEN-FINE-TUNE BEAT FROM-SCRATCH FOR MODULATION/PROTOCOL CLASSIFICATION?

**Well-replicated finding (a):** Yes, dramatically, when labels per class are in the single digits to low tens. Consistent across at least five independent groups (Davaslioglu/Sagduyu, Abou-Zeid/Calgary, Zhou/NUAA, Jiao/Xidian, Alkhateeb/ASU) with effect sizes of +15 to +40 accuracy points at 1-5 labels/class.

**Well-replicated finding (b), and the one that matters for you:** No, essentially not, once labels are abundant. Every paper that reports the full-label end shows the gain collapsing:
- Davaslioglu 2022: +1.2 pts at 90% labels (62.4 vs 61.2)
- Kanu 2025: +0.17 pts at 100% labels (99.985 vs 99.811); **-0.08 pts at 1% labels**
- Mod-CL 2026: +2.3 pts at 100/class (61.76 vs 59.42), and still 1.3 pts *below* the fully supervised model
- IQFM on RML2016a at 500/class: +0.7 pts (50.00 vs 49.30); **-0.6 pts on RF fingerprinting**
- Ahmadi 2025: SSL ViT at 15% labels (71.01%) beats supervised ViT at 100% (68.21%) but loses badly to supervised ResNet at 100% (78.50%)

**Published negative result (c), and it is a strong one:** "Self-Supervised Radio Pre-training: Toward Foundational Models for Spectrogram Learning," Ahmed Aboulfotouh, Ashkan Eshaghbeigi, Dimitrios Karslidis, Hatem Abou-Zeid. arXiv:2411.09849 (Nov 2024). ConvLSTM (5 ConvLSTM + 1 Conv3D, 64 kernels, 3x3), masked spectrogram modeling at 20% mask, pretrained on a Real-time Radio Dataset of 240 recordings totalling **~24 seconds** of over-the-air IQ (2.4-2.65 GHz, downtown Toronto).
- Spectrum forecasting: **the from-scratch baseline slightly outperformed the fine-tuned foundation model** at all time/frequency resolutions.
- NR-LTE segmentation: **the fine-tuned foundation model underperformed the from-scratch baseline**, particularly struggling to distinguish NR signals. Even the easier binary signal-vs-noise variant lost to the baseline.
- Authors' own diagnosis: "features provided by the pretrained backbone are not sufficiently discriminative," plus a real-vs-simulated distribution mismatch between pretraining and downstream data. This is the same group that later published WavesFM, IQFM, and LatentWave -- they published their failure first, which raises my confidence in their later positive numbers but also confirms that RF SSL fails quietly when the corpus is small or mismatched.

**Corroborating negative-ish signal:** EMind, at 110M params and 81M pretraining samples, loses to SpectrumFM (16 layers, ~2e9 FLOPs/sample) on RML2016.10A. And "Large Wireless Foundation Models: Stronger over Bigger" (Cheng et al., PKU, arXiv:2601.10963) is an explicit position paper arguing that LLM-style scaling laws do not carry over, that wireless FMs sit below 100M params, and that the field lacks the large open corpora that would be needed to test scaling. **There is no published RF scaling law analogous to Kaplan/Chinchilla, and no paper I found demonstrates monotone benefit from more RF pretraining data.**

**Convergence speed is the most consistently reproduced benefit**, more than final accuracy: WavesFM reports up to **5x** reduction in training time; Aboulfotouh 2411.09996 reports a pretrained ViT beating a **4x larger** from-scratch model on spectrogram segmentation "while requiring significantly less training time" (exact accuracy numbers not given in the abstract or metadata I could retrieve -- I could not verify the specific figures).

---

# (4) TRANSFER FROM AUDIO / SPEECH MODELS TO RF

**I could not find a single published paper transferring wav2vec 2.0, wav2vec, HuBERT, WavLM, or any speech SSL model to RF I/Q classification.** I searched specifically for this multiple ways. If such work exists it is obscure enough that I did not surface it. I am stating this as a negative search result, not as proof of non-existence.

What does exist:

1. **Vision-to-RF transfer, measured.** Davaslioglu et al. 2022 include ImageNet-pretrained initialization as a baseline: 14.9% at 0.5% labels, 34.5% at 1%, 53.1% at 5%, 61.1% at 90%. So ImageNet weights **do** transfer non-trivially to RF at very low label counts (34.5% vs 11.4% Xavier at 1% labels -- a 3x gap) but are beaten by in-domain SSL (53.1%) and are **worthless at high label counts** (61.1 vs 61.2 Xavier -- literally a wash). This is the closest thing to a quantified cross-modal-to-RF transfer result I found.

2. **Audio-to-mmWave knowledge distillation.** Radio2Text (arXiv:2308.08125) uses a trained audio network as teacher and an mmWave network as student for streaming speech recognition. This is cross-modal distillation for a speech task sensed via radio, **not** transfer of audio representations to RF modulation/protocol classification. Different problem.

3. **The architectural lineage is real even if the weight transfer is not.** AST (Gong et al., arXiv:2104.01778) itself works by ImageNet->audio-spectrogram cross-modal transfer, and SSAST extends it to self-supervised. Every RF spectrogram FM in this report (WavesFM, LWM-Spectro, LatentWave, the multimodal WFM) is a ViT-on-spectrogram design descended from that line. So the *method* transferred; the *weights* apparently have not been tried.

4. **Vision-language models on RF spectrograms** are an emerging thread (RF-GPT arXiv:2602.14833; "Seeing Radio: From Zero RF Priors to Explainable Modulation Recognition with VLMs" arXiv:2601.13157; "RF-Analyzer: Can Vision-Language Models Learn RF Understanding from Synthetic Data?" arXiv:2605.04676). All 2026 preprints, all render RF as images for an off-the-shelf VLM. I did not verify their numbers and would treat them as early-stage.

---

# WEIGHT AVAILABILITY (practical blocker)

- **EMind**: code at github.com/GabrielleTse/EMind -- the only confirmed public repo. I did not verify whether pretrained checkpoints are actually in it.
- **TorchSig** (Boegner et al., "Large Scale Radio Frequency Signal Classification"; "Large Scale RF Wideband Signal Detection & Recognition," arXiv:2211.10335): ships models pretrained on **Sig53** (5M synthetic samples, 53 signal classes with expert-chosen impairments) and WidebandSig53. These are *supervised* pretrained, not SSL, but they are the most practically downloadable RF pretrained weights that exist. Their finding that "Transformers outperform ConvNets without additional regularization" on Sig53 is worth noting given your ViT plateau.
- **SpectrumFM, IQFM, WavesFM, LWM-Spectro, LatentWave**: no public weights confirmed by my searches.

---

# CONFIDENCE TIERS, EXPLICITLY

**(a) Well-replicated (5+ independent groups, consistent direction):**
- SSL pretraining gives large gains at 1-20 labels/class for RF classification.
- Those gains shrink to 0-2 points at 100+ labels/class and can go slightly negative.
- Masked/reconstructive objectives beat instance-discrimination contrastive objectives for RF.
- Pretraining accelerates convergence more reliably than it raises the accuracy ceiling.

**(b) Single-paper claims (treat as unreplicated):**
- IQFM's 99.67% at 1 sample/class (own dataset, no independent reproduction).
- Mod-CL's specific numbers and the temporal-segment-positive-pair idea (May 2026, no reproduction yet).
- Cheraghinia's 90.38% vs 62.79% unseen-technology transfer result.
- EMind's 81M-sample pretraining corpus results.
- LWM-Spectro's 30x-data-efficiency claim.
- Aboulfotouh 2411.09849's negative result (though negative results are rarely contested).

**(c) My own extrapolation, flagged as such:**
- That no published RF SSL result addresses *structured confusion between specific classes* -- I checked and found only aggregate accuracy and per-SNR curves, never a confusion-matrix-targeted intervention. I am extrapolating from absence.
- That the field's pretraining corpora (24 seconds to 81M short snippets, mostly RadioML-family synthetic AMC data) are poorly matched to multi-protocol bursty-signal classification at 1-30 MHz varied sample rates.

**Venue caution:** SpectrumFM (IEEE JSAC), WavesFM (IEEE Xplore 11131142), Davaslioglu (IEEE WCL 2022), SemiAMC (MILCOM 2021), and the Nature Comms unsupervised-AMR paper are peer reviewed. **EMind, IQFM, LWM-Spectro, LatentWave, Mod-CL, Cheraghinia, Ahmadi, Aboulfotouh 2411.09849/2411.09996, Kanu, WiFo-2 are arXiv preprints.** The 2026-dated ones (2601.*, 2602.*, 2604.*, 2605.*, 2606.*) are all within the last seven months and have had no time for scrutiny.

## Applicability

## BOTTOM LINE: I do not think SSL pretraining is the right lever for your specific problem, and the literature I just surveyed is the reason, not a hunch.

### Why the core value proposition does not apply to you

Every measured gain in this literature is a **label-efficiency** gain. The mechanism is: unlabeled RF is abundant, labeled RF is expensive, SSL converts the former into a warm start. Your corpus is **synthetic and regenerable at arbitrary size from 37 parameterized profiles, with free ground-truth labels**. You are not label-limited. You are at 2000/class and could be at 20,000/class tomorrow at the cost of compute.

Line up your position against the tables:
- Davaslioglu at 90% labels: SSL +1.2 pts.
- Kanu at 100% labels: SSL +0.17 pts; at 1% labels, **-0.08 pts**.
- Mod-CL at 100/class: SSL +2.3 pts over scratch, still 1.3 pts *below* fully supervised.
- IQFM at 500/class on RML2016a: +0.7 pts; **-0.6 pts on fingerprinting**.

You are firmly in the regime where every one of those papers reports the gain has already collapsed. The honest expected effect of adding masked-IQ pretraining to your pipeline is **+0 to +2 points on overall accuracy**, i.e. 0.891 -> somewhere in 0.89-0.91. That is real but it is not what you are chasing, and your 1D CNN already sits at 0.891 while the field's best RML2016.10A number after 81M pretraining samples is 62.51%. Your problem is not the same problem these benchmarks measure.

### Why it will very probably not touch the bluetooth/gsm/dsss triad

This is the load-bearing point. **No paper I found reports SSL pretraining resolving a specific structured confusion between semantically-adjacent classes.** They report aggregate accuracy and per-SNR curves. Where per-class breakdowns appear, SSL shows the *same* failure structure as supervised training, just shifted: Ahmadi et al.'s semi-supervised ViT gets BPSK 99.91% and FM 100.0% but 64QAM 24.64% and 256QAM 40.20% -- exactly the confusion structure a from-scratch model has on RML2018, unchanged by pretraining.

Your diagnostic already localized the cause and it is not a representation-learning-capacity problem: bluetooth-vs-gsm centroid z-distance 0.713 is **smaller than either class's within-class spread** (gsm 2.053, bluetooth 1.495). Two classes whose centroids sit inside each other's spread are not separated by a better feature extractor; they are separated by (i) fixing the label geometry or (ii) finding a feature that actually distinguishes GFSK at h~0.32 from GMSK at h=0.5. Pretraining on unlabeled I/Q optimizes a reconstruction or instance-discrimination objective that is *blind to your label set entirely* -- it has no mechanism to allocate capacity to the one axis you need. Your own experiments already demonstrated this: 750,000 episodes of supervised metric learning with the labels in the loop did not move it. An objective without the labels is not more likely to.

I flag this as **(c) informed extrapolation from absence of evidence**, which is weaker than a measured negative. But the absence is broad -- I looked for it specifically.

### Where it WOULD help you, honestly

1. **The Mod-CL positive-pair construction is the one directly transferable idea, and it is cheap.** Positive pairs from *different temporal segments of the same signal instance* rather than from augmentations. Your windows are "fixed-length and arbitrary-phase relative to burst boundaries -- bursts land at random positions/offsets within the analysis window." That is precisely the nuisance factor Mod-CL's objective is designed to quotient out. You could add this as an **auxiliary loss on your existing episodic prototypical training** (two random 4096-windows from the same generated emission -> pull together), with no pretraining stage, no new corpus, no architecture change. It targets burst-phase invariance directly. Expected value: modest, but it is the highest-ratio-of-relevance-to-cost item in this entire report, and it composes with what you already ship. Caveat: Mod-CL is a May 2026 preprint with no independent replication, and its measured gain at N=100 was +2.3 pts on a different task.

2. **If you ever revisit the STFT 2D ViT, LatentWave's masking-geometry ablation is directly actionable:** region (2D block) and time masking preserve the temporal discriminability that signal classification needs; frequency masking costs 15 points on classification (66.1 vs 80.9); random masking consistently underperforms all structured strategies. Also relevant to your observed 2D-ViT cw-absorption failure: a frequency-axis-destroying representation is exactly what would collapse am/fm/bluetooth into a cw-like attractor.

3. **Masked reconstruction over contrastive, if you pretrain at all.** Three independent groups agree. Ahmadi's numbers are the sharpest: reconstruction-only 68.21% vs contrastive-only 53.41% at 10% labels, and **adding contrastive to reconstruction made it strictly worse at every label fraction**. Given you already tried per-sample contrastive repulsion (intervention #4, 215/237, no change), this is consistent with your own measurement -- contrastive objectives on RF do not do what you want.

4. **Convergence speed, not ceiling.** WavesFM's 5x training-time reduction and Aboulfotouh's "pretrained ViT beats a 4x-larger from-scratch model" are the most reproduced benefits. Your complex CLDNN hit 0.857 at episode 1200 of 5000 and was still climbing -- if you are compute-bound on exploring that architecture, a masked-IQ pretraining stage might get you to its ceiling faster. That is a **research-velocity** argument, not an accuracy argument.

### Two architectural data points worth more to you than the SSL findings

- **Cheraghinia et al. (arXiv:2505.19390) is your problem's nearest published neighbor**: 860K-param PatchTST-style patch transformer on **(2, 4096)** stacked I/Q -- your exact tensor -- classifying 8 *technologies* (not modulations) across 1/10/20 MHz sample rates, hitting **99.47%**. Patch size and stride 128, 4 encoder blocks, dim 128. That is a *supervised* result. It says a patch transformer at your input shape and roughly your parameter budget can do protocol recognition very well, which is a stronger argument for revisiting your 1D ViT's **patching/tokenization** (P=S=128, 32 tokens) than for pretraining it. Your 1D ViT plateaued at 0.876 with a confirmed hard ceiling -- worth checking whether their patch geometry differs from yours, because that is a free experiment.
- **TorchSig/Sig53 is the one downloadable pretrained RF checkpoint set** (5M synthetic samples, 53 classes, expert impairments -- structurally very like your generator). Supervised, not SSL. Their reported finding that Transformers beat ConvNets on Sig53 without extra regularization sits in tension with your measurement (CNN 0.891 > ViT 0.876), which is itself informative: your 7 classes are protocol/family labels pooling heterogeneous sub-profiles, whereas Sig53 classes are clean modulation labels. That difference is your problem in a nutshell.

### Where this research does NOT help you at all

- No public weights for any I/Q foundation model except EMind (repo exists; checkpoint presence unverified). So "download an RF wav2vec and fine-tune" is not currently an available move.
- Input-shape and domain mismatch: these models pretrain on 128-1024-sample snippets of RadioML-family data at fixed sample rates. You have 4096 complex samples across 63 sample rates from 1-30 MHz with a preprocessing chain that already imposes frequency/scale/power invariance. Loading someone else's checkpoint into that would be a substantial adaptation project for a +0-2 point expected return.
- **Audio/speech -> RF transfer: no published work found.** Do not budget time for it on the strength of an analogy. The only quantified cross-modal-to-RF number is ImageNet init, which is a wash (61.1 vs 61.2) at high label counts.
- The published negative result (Aboulfotouh et al. arXiv:2411.09849) is the cautionary case that most resembles a small-scale attempt: 24 seconds of pretraining data, ConvLSTM, and the pretrained model **lost to from-scratch** on both spectrum forecasting and NR-LTE segmentation, with real-vs-simulated distribution mismatch cited. If you pretrained on your own 14k synthetic samples, you would be operating at a corpus scale much closer to their failure than to EMind's 81M.

### What the literature implicitly points at instead

The structural observations you flagged but have not acted on -- bluetooth pooling two structurally different sub-profiles under one prototype, gsm pooling seven including both GMSK and 8PSK/QAM EDGE bursts, ofdm pooling ~20 -- are a **label-geometry** problem, and a single-prototype-per-class nearest-prototype classifier is the exact wrong inference rule for it. The AMC literature's standard two-stage burst-detect-then-classify pipeline that you noted is the other unexploited lever. Neither is an SSL problem. I did search for multi-prototype / subclass prototypical work for RF and found only adjacent material (backtracking contextual prototypical memory for radar mode recognition, ScienceDirect S1051200423002841; ProtoAoA arXiv:2604.14476) -- nothing that directly addresses heterogeneous-class multi-prototype for signal families. That is a gap in the literature, not a solved technique waiting to be applied.

## Confidence

Mixed, and deliberately tiered in the findings.

STRONG (multiple independent sources, consistent direction, at least some peer-reviewed): The core empirical pattern -- SSL gives 15-40 point gains at 1-20 labels/class and 0-2 points at 100+ labels/class -- is supported by at least five independent groups (Davaslioglu/Sagduyu IEEE WCL 2022; Kanu/Eshaghbeigi/Abou-Zeid 2025; Wang et al. Mod-CL 2026; Mashaal/Abou-Zeid IQFM 2025; Kim/Alikhani/Alkhateeb LWM-Spectro 2026) with numbers I retrieved directly from paper tables, not from abstracts. Same for masked-reconstruction beating contrastive (Ahmadi et al. ablation, SpectrumFM ablation, EMind and LatentWave design choices). Same for convergence-speed being the more reliable benefit than accuracy ceiling.

MODERATE (single source, verified numbers, no independent replication): EMind's 110M/81M-sample results, IQFM's 1-shot claims, Mod-CL's tables, Cheraghinia's unseen-technology transfer, LWM-Spectro's efficiency claim, LatentWave's masking-geometry ablation, and the Aboulfotouh 2411.09849 negative result. I pulled exact tables for most of these from arXiv HTML renderings, so the numbers are quoted accurately, but they are single-lab, mostly unrefereed preprints -- and roughly half are dated within the last seven months (2601-2606 arXiv IDs), so they have had essentially no time for scrutiny.

WEAK / EXPLICITLY FLAGGED AS MY OWN EXTRAPOLATION: The central applicability claim -- that SSL pretraining will not move your bluetooth/gsm/dsss triad confusion -- is an argument from absence of evidence. I searched for SSL work targeting structured inter-class confusion and found none; the closest supporting datum is Ahmadi et al.'s per-class breakdown showing the same confusion structure surviving pretraining. That is suggestive, not dispositive. Likewise my claim about domain/input-shape mismatch between published pretraining corpora and your setup is reasoning, not measurement.

THINGS I COULD NOT VERIFY, stated as such: exact parameter counts for SpectrumFM and LWM-Spectro; the numeric results behind Aboulfotouh 2411.09996's "outperforms a 4x larger from-scratch model" claim (abstract only, PDF extraction failed); whether EMind's GitHub repo actually contains pretrained checkpoints; whether "An Analysis of Radio Frequency Transfer Learning Behavior" (MAKE 2024, doi 10.3390/make6020057) contains negative transfer results -- MDPI returned 403 on both fetch attempts, so I excluded its content entirely rather than guess.

NEGATIVE SEARCH RESULT, stated as such: I found no published transfer of wav2vec/wav2vec2/HuBERT/WavLM to RF I/Q classification despite searching specifically for it several ways. I am reporting this as "I did not find it," not as "it does not exist."

Two PDF fetches (EMind, Kanu, Ahmadi via /pdf) returned unusable binary; I re-fetched via arXiv HTML renderings and got clean tables, so the numbers reported are from the HTML versions.

---

# RF Foundation Models

## Findings


# Multi-representation / multi-modal fusion for RF signal classification

Confidence tags used throughout: **[A]** = replicated across independent sources; **[B]** = single-paper claim, numbers verified from the source; **[C]** = single-paper claim I could NOT verify (paywall/403/PDF-parse failure), reported as claimed; **[D]** = my own extrapolation, not from any paper.

---

## (1) Papers that fuse multiple representations and quantify the gain

### 1a. The single most on-point paper for your situation

**Hou, Dong, et al. (2023), "Multi-domain-fusion deep learning for automatic modulation recognition in spatial cognitive radio," Scientific Reports 13, s41598-023-37165-2.** **[B]**

- Two branches: **MCLDNN on time-domain I/Q**, and **BiGRU3 on FFT (frequency-domain amplitude+phase)**.
- Crucially, they did **not** average or concatenate. They observed that the I/Q model had **one specific confusion it could not fix (AM-DSB vs WBFM)**, so they built a **cascade**: MCLDNN performs 10-way classification with the two confusable classes merged into a single meta-class "DW", then BiGRU3 does the **binary split of DW only**.
- Numbers: MCLDNN alone 92.27% (RML2016.10a) / 93.47% (10b). Cascade: **94.94% / 96.69%** (+2.67 / +3.22 absolute overall). The AM-DSB-vs-WBFM binary discrimination improved by **17% (10a) and 18.2% (10b)**.
- Interpretation: essentially **all** of the fusion gain came from one class pair. The second representation contributed nothing elsewhere and was structurally prevented from doing damage elsewhere.

This is the closest published analogue to what you measured (a second representation that fixes one confusion), and it is important that the authors' winning design **restricted the second model's authority to the confusable subset**.

### 1b. Dual-modal fusion with a clean single-vs-fused ablation

**Bai, Yao, Qi, Wang (2022), "Electromagnetic Modulation Signal Classification Using Dual-Modal Feature Fusion CNN" (MDPI, retrieved via PMC9142120; I did not verify the journal name).** **[B]**

- Modalities: **Gramian Angular Field images** (via ResNet50) + **raw I/Q** (via a complex-valued CNN). Fusion is feature-level concatenation, with a **Jensen–Shannon divergence penalty term** between the two branches' predictive distributions added to the loss (an explicit diversity/consistency regularizer).
- At −10 dB SNR: GAF-ResNet50 alone **87.6%**, CV-CNN alone **85.3%**, fused **92.1%** → **+4.5 / +6.8 absolute**.
- Dataset: 120k samples, 8 classes (2FSK, AM, DSB, FM, OFDM, QAM16, QPSK, SSB), −10 to +10 dB.

### 1c. Three-way feature fusion including hand features

**Han, Ren, Li, Zhu (2021), "Automatic Modulation Classification Based on Deep Feature Fusion for High Noise Level and Large Dynamic Input," Sensors 21(6):2117.** **[B for fused number, C for single-branch numbers]**

- Fuses **CNN features from FFT spectrum + SAE features from Welch PSD + statistical features (instantaneous amplitude/phase stats and higher-order cumulants)**, concatenated into a joint vector, classified by a probabilistic neural network. This is **late/feature-level fusion**.
- Fused reaches **99.8% at 0 dB**; single branches read approximately **CNN ~96%, SAE ~85%** at 0 dB (these single-branch values came from a figure-reading summary and should be treated as approximate). 10 classes, −20 to +20 dB.
- Relevant structural note: their statistical/HOC branch was the *best* branch at extreme low SNR and the worst at high SNR — i.e. **the branches' relative strength inverts with condition**, which is what makes fusion pay.

### 1d. Systematic early/intermediate/late fusion comparison for I/Q + constellation

**Lin, Guo, Yang, Wu, Peng (2026), "Multimodal Deep Learning based Automatic Modulation Recognition: Fusion of Signal Modalities," AI Engineering (sciltp.com/journals/aieng/articles/2606004161).** **[C — I could read the abstract/conclusions but could not extract the numeric tables]**

- Compares **early fusion** (concatenate representations before the network), **intermediate fusion** (merge inside the network), **late fusion** (combine per-modality outputs).
- Findings as stated: **intermediate fusion gives best accuracy**; **early fusion gives satisfactory accuracy at the lowest compute**; **late fusion gives low accuracy despite maximum flexibility**. Multimodal consistently beat unimodal across architectures, sample sizes, and channel models.
- Caveat: I could not verify the magnitudes, and "late fusion is worst" directly contradicts Hou 2023's cascade success — see §2 for reconciliation.

### 1e. Attention-based tri-modal fusion

**Han, Yu, Yang (2022), "Multimodal attention-based deep learning for automatic modulation classification," Frontiers in Energy Research 10:1041862.** **[C]**

- Three modalities: **constellation diagram**, **smoothed pseudo Wigner-Ville distribution** (time-frequency), and **spectral correlation function contour** (cyclostationary). Shared convolutional autoencoder per modality, then **bidirectional multi-head cross-modal attention**, then MLP. RML2016.10a.
- Claims large gains at −20 to 0 dB. **Important negative note: the paper provides no ablation isolating attention-fusion vs simple concatenation**, so the value of the attention mechanism specifically is unsupported.

### 1f. SNR-gated conditional fusion (precedent for "use modality 2 only where it helps")

**M-LSCANet, "A Multi-Modal Modulation Recognition Method with SNR Segmentation Based on Time Domain Signals and Constellation Diagrams," Electronics 12(14):3175 (2023).** **[C]**

- Uses **I/Q + constellation jointly at medium/high SNR, but I/Q ONLY at low SNR**, because the constellation representation becomes actively harmful when noise smears the cloud. CBAM channel+spatial reweighting after fusion. Reports 93.4% / 95.8% on RML2016.10b.
- This is direct published precedent for **conditionally disabling a modality in the regime where it introduces errors** — structurally the same move you'd want for your cw-absorption problem.

### 1g. MCANet (arXiv:2510.18336, Jiang et al., 2025) **[C, low confidence]**
Multimodal collaborative attention for AMR. The abstract contains **no numbers, no named datasets, no named modalities**. I do not consider this citable evidence.

### 1h. The most important NEGATIVE result on fusion

**Oladunni & Wong (2025), "Rethinking Multimodality: Optimizing Multimodal Deep Learning for Biomedical Signal Classification," arXiv:2508.00963.** **[B]** (biomedical signals, not RF — transfer is an assumption)

- Five models: three unimodal (1D-CNN time, 2D-CNN time-frequency, 1D-CNN-Transformer frequency) and two multimodal.
- **Hybrid 1 (1D-CNN + 2D-CNN)** beat baselines with statistical significance (p < 0.05, Bayesian posterior > 0.90).
- **Hybrid 2 (adding the frequency-domain Transformer as a third modality) gave NO further improvement and sometimes a marginal decline.**
- Stated conclusion: "optimal domain fusion isn't about the number of modalities, but the quality of their inherent complementarity." **Representational redundancy actively costs you.**

---

## (2) Which fusion strategy works best for signals

**Evidence is genuinely mixed and the apparent contradiction is informative.**

| Strategy | Evidence | Verdict |
|---|---|---|
| Early (concatenate raw representations) | Lin 2026: "satisfactory, cheapest" **[C]** | Works, weakest gains; requires representations to be dimensionally compatible |
| Intermediate / feature-level (separate encoders → concat embeddings → joint head) | Lin 2026 best **[C]**; Bai 2022 +4.5/+6.8 **[B]**; Han 2021 **[B]** | **Best supported overall.** This is the default in the AMC literature. |
| Late / probability averaging | Lin 2026 worst **[C]** | Weakest when done naively |
| **Cascade / routed decision fusion** (model 2 only arbitrates a specific confusable subset) | Hou 2023: +17/+18.2% on the confused pair, +2.67/+3.22 overall **[B]** | **Largest per-confusion gain of anything I found** |
| Gated / conditional (disable a modality in its bad regime) | M-LSCANet **[C]** | Under-studied, directly relevant to your problem |
| Cross-modal attention | Frontiers 2022 **[C]**; MCANet **[C]**; MFCA-Transformer (PMC12390484) **[C]** | Currently a fashion. **I could not find a single AMC paper with a clean ablation of cross-attention vs plain concatenation.** Treat the claimed benefit as unproven. |

**Reconciling "late fusion is worst" with "cascade fusion is best" [D]:** these are different operations that get lumped together. Naive late fusion averages probability vectors, which **dilutes** a specialist's correct minority opinion against a generalist's confident wrong one, and **imports the weaker model's error modes across all classes**. A cascade/gate does the opposite: it consults the second model only where the first is known to be unreliable, and grants it no authority elsewhere. Your specific situation (a second model that fixes one confusion and creates another) is precisely the case where naive averaging fails and routing wins.

---

## (3) Wavelets / multi-resolution vs STFT for burst-structured signals

**Honest summary: the theoretical argument is strong, the empirical record is weak and partly negative. I would not prioritize this.**

The cleanest head-to-head I could find:

**Phan, D.T. (2024), "Comparison Performance of Spectrogram and Scalogram as Input of Acoustic Recognition Task," arXiv:2403.03611.** **[B]** — CNN on MIMII machine-audio (18,019 files), AUC-ROC:

| Condition | Spectrogram | Scalogram |
|---|---|---|
| −6 dB | 0.981 | 0.921 |
| 0 dB | 0.992 | 0.964 |
| +6 dB | 0.997 | 0.988 |
| Fan (stationary) | 0.988 | 0.934 |
| Pump | 0.991 | 0.962 |
| Slider | 0.995 | 0.947 |
| **Valve (impulsive/transient)** | 0.984 | **0.987** |

Compute: **2.9 hours (spectrogram) vs 109 hours (scalogram)** for the same dataset — a **~37x** cost.

Stated conclusion: "spectrograms consistently outperform scalograms," except on the **valve** class, which is the impulsive/transient one, where the wavelet multiresolution property helps. **This is the one datapoint that supports "wavelets help for bursts" — and the margin is +0.003 AUC.**

Other wavelet-RF results, all without an STFT head-to-head:
- **Medaiyese, Ezuma, Lauf, Guvenç (2021), "Wavelet Transform Analytics for RF-Based UAV Detection and Identification System Using Machine Learning," arXiv:2102.11894 / Ad Hoc Networks.** **[B]** Wavelet scattering "scattergrams" + SqueezeNet, **98.9% at 10 dB SNR** for drone RF-controller ID under WiFi/Bluetooth interference. Notably they treat **transient** and **steady-state** signal segments as separate inputs. **No STFT/FFT baseline is reported.**
- **"Automatic Modulation Classification Based on Wavelet Analysis and CNN," Electronics 14(19):3801 (2025).** **[C, unverified — MDPI returned 403]** WS-CNN claimed **85.6% at −10 dB, 99.7% average above −2 dB**. I could not confirm whether an STFT baseline was run.
- **"Robust AMC Using CNN Based on Scalogram Information," Computers 11(11):162 (2022).** **[C, unverified — 403]**

**My assessment [D]:** the constant-Q property of wavelets buys you better time localization at high frequency, which is exactly what burst onsets need. But for your task the discriminative content isn't burst-*edge sharpness* — it's **burst duration, duty cycle, and repetition period** (GSM 577 µs bursts at 4.615 ms frame rate; Bluetooth classic 625 µs slots at 1600 hops/s; BLE advertising ~44–376 µs at very low duty cycle). Those are *slow* structures, well within STFT's reach with an appropriate hop, and a scalogram's advantage does not apply. I'd rate wavelets as low expected value here relative to cost.

---

## (4) Constellation-diagram-image classification: where it helps and where it hurts

**Where it demonstrably helps [A]:** resolving **dense linear digital constellations** — 16QAM vs 64QAM vs 256QAM, and QPSK vs 8PSK — which is the classic failure mode of raw-I/Q models. This is the near-universal motivation cited across the constellation-AMC literature.

**Reference point:** Peng, Jiang, et al. (2019), "Modulation Classification Based on Signal Constellation Diagrams and Deep Learning," IEEE TNNLS 30(3):718–727, DOI 10.1109/TNNLS.2018.2850703. Reported: at 2 dB SNR, accuracy improves from **82.8% → 97.1%** when a three-channel enhanced grid image is used instead of a plain constellation diagram. **[B for the number; note this is constellation-encoding engineering, not constellation-vs-I/Q.]**

**Where it hurts:**

1. **Constant-envelope / continuous-phase modulations carry almost no constellation information.** GFSK, GMSK, MSK, and CPFSK all map to (approximately) the *same ring* in the I/Q plane. **[A — this is textbook signal theory, not a contested claim.]** This is decisive for you: **Bluetooth (GFSK) and GSM (GMSK) are both on the ring.** A constellation branch is structurally near-blind to exactly the confusion you cannot fix. One study surfaced in search reported a CNN confusing **GFSK with QAM16** and degrading GFSK performance *at higher SNR* **[C, source not verified]**.
2. **Constellation requires carrier and symbol timing recovery, plus enough symbols to populate the cloud.** At unknown symbol rate, arbitrary window offset relative to a burst, and short observation, the cloud is either sparse or smeared. **[A]**
3. **Waqas, Ashraf, Zakwan (2023), "Modulation Classification Through Deep Learning Using Resolution Transformed Spectrograms," arXiv:2306.04655** — a summarization of this paper reports the observation that constellation plots "require additional processing overhead and have had comparable performance to raw I/Q signals," and that raw-I/Q models "could be more adept at handling varying-length signals than constellation plots because they are not limited by periodicity constraints for short-duration signals (i.e., burst transmissions)." **[C — I could not re-extract this passage from the PDF directly; treat the attribution as medium confidence, though the technical claim itself is uncontroversial.]** The paper's own headline result is 91.2% with resolution-transformed spectrograms and **99.61% computational-load reduction / 8× faster conversion**.
4. **M-LSCANet's design implicitly concedes it:** they *disable* the constellation modality at low SNR. **[C]**

**For your 7 classes specifically [D]:** a constellation branch would plausibly help separate **ofdm** (dense Gaussian cloud), **dsss** (real DBPSK/DQPSK/CCK constellations for 802.11b), **am/cw** (a line/point), and **fm** (a ring). It would contribute **essentially nothing to bluetooth-vs-gsm**, which is your actual problem. It might help pull **dsss out of the bluetooth basin**, which your diagnostic says originates in the learned waveform path rather than the hand features — so that is a real, if partial, target.

---

## (5) Your specific finding: fusing/ensembling models that make complementary errors

**Yes, there is published work, and the theory is well established. But the theory also predicts your naive-ensemble outcome would be poor, and tells you what to do instead.**

### Theory **[A]**
- **Krogh & Vedelsby (1995), "Neural Network Ensembles, Cross Validation, and Active Learning," NIPS 7 — the ambiguity decomposition:** ensemble error = (weighted average of individual errors) − (ensemble ambiguity/diversity). Diversity is subtracted directly from error.
- **Tumer & Ghosh (1996), "Error Correlation and Error Reduction in Ensemble Classifiers," Connection Science 8(3–4):385–404:** the added error of an averaging ensemble of N classifiers scales roughly as **(1 + δ(N−1))/N** where δ is the pairwise error correlation. At δ→0 you get the full 1/N reduction; at δ→1 you get nothing. **Complementary errors are the entire mechanism.**
- **Wood, Mu, Webb, Reeve, Luján, Brown (2023), "A Unified Theory of Diversity in Ensemble Learning," arXiv:2301.03962 (JMLR):** extends the decomposition to 0-1 loss and other losses, and — importantly for you — establishes that **diversity is not a free lunch**: there is an explicit **diversity/average-accuracy tradeoff**, and maximizing diversity is not the right objective. Adding a more-diverse but weaker member can make the ensemble worse.

### Empirical, in-domain and near-domain

- **Hou 2023 (§1a)** — the closest analogue. **+17/+18.2% on the confusable pair**, +2.67/+3.22 overall, achieved via **cascade routing, not averaging**. **[B]**
- **Kong, Cao, Iqbal, Wang, Wang, Plumbley (2020), "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition," IEEE/ACM TASLP (arXiv:1912.10211).** **[A — this is the most-replicated waveform-vs-spectrogram fusion result in existence.]** AudioSet tagging mAP:
  - CNN14 on log-mel spectrogram (2D): **0.431**
  - Wavegram-CNN (learned from raw waveform, 1D front end): **0.389**
  - **Wavegram-Logmel-CNN (fusing both): 0.439**

  **The gain over the better single branch is +0.008 mAP (~1.9% relative)**, despite the waveform branch being 4.2 mAP points worse alone. This is the calibration you should carry: **fusing a strong representation with a moderately weaker one typically buys 1–3% relative, not 5–10% absolute — unless the weaker one uniquely resolves a specific confusion (Hou's case), where the per-confusion gain can be 17%.**
- **Schindler, Lidy, Rauber (2017), "Multi-Temporal Resolution Convolutional Neural Networks for Acoustic Scene Classification," DCASE2017 Workshop.** **[B]** Multi-resolution parallel CNN gave **+3.56% absolute over the best single-resolution model** (and +12.49% over the DCASE baseline). Here the diversity axis is **time-frequency resolution**, not representation family.
- **Bai 2022 (§1b)** explicitly regularizes for cross-branch agreement with a **JS-divergence penalty** — evidence that unconstrained diversity between branches was found to need damping. **[B]**

### Direct read on your numbers **[D]**

Your 2D ViT's effect decomposes as:
- **Fixed:** gsm→bluetooth 236 → 143 = **93 errors eliminated**.
- **Introduced:** bluetooth→cw 59, am→cw 46, fm→cw 30 = **135 new errors** on those three axes alone.

Net on those axes: **−42**. And its overall accuracy is 0.850 vs the 1D ViT's 0.876. A naive probability-average or embedding-concat **will import the cw-absorption failure into classes where the time-domain model was already ~clean**, and the theory (Wood et al. 2023's diversity/accuracy tradeoff) says exactly this. **Do not expect a naive ensemble to beat 0.876 by much, and it may not beat it at all.**

The construction the literature supports is **Hou's**: let the time-domain model do 7-way with **{bluetooth, gsm, dsss} collapsed into one meta-class**, then let the spectrogram model do the **3-way split within the meta-class only**, with **cw not in its output alphabet**. That captures the 93 without any of the 135. Hou got +2.67 overall from a structurally identical move on a single pair; you have a triad that accounts for ~35–39% error on two of seven classes, so the headroom is larger.

**A gap I want to flag explicitly:** I found **no published work on multi-representation fusion inside an episodic prototypical / few-shot AMC pipeline**. All the fusion literature above uses standard softmax classifiers with fixed class sets. How to fuse two 32-dim L2-normalized embeddings (concatenate then re-normalize? per-branch learned temperature? separate prototypes per branch with distance-level fusion?) is **not addressed by any paper I found**. Treat any design there as unguided.

---

## (6) Optimal STFT parameters for bursty signal classification

**Direct answer: I could not find a single rigorous published parameter sweep (FFT size × hop × window) with accuracy numbers for AMC or RF burst classification. This is a real gap in the literature, not a gap in my search.** What exists:

**What the literature actually does [A]:** picks parameters by convention and moves on. Recurring defaults: Hann or Hamming window; **50–75% overlap**; FFT length 256 for audio-rate work; window length chosen so the event of interest spans several frames. Overlap beyond ~75% adds frames and compute but **does not improve the underlying time-frequency resolution**, which is set by window length alone.

**Datapoints I could verify:**
- **Waqas et al. 2023 (arXiv:2306.04655)** **[C]**: aggressively "resolution-transformed" (downsampled) spectrograms achieved **91.2% accuracy** with **up to 99.61% computational-load reduction and 8× faster conversion** from I/Q. Implication: **spectrogram resolution was massively over-provisioned for the task** — accuracy was insensitive to large reductions.
- **"Radar-Spectrogram-Based UAV Classification Using CNNs," Sensors 21(1):210 (2021)** **[C, unverified — 403]**: rather than selecting one setting, they generated **3 window sizes × 3 overlap ratios per signal as data augmentation**. Per-setting accuracies not verifiable.
- **Schindler et al. 2017** **[B]**: **+3.56% over the best single resolution** by running parallel resolutions. This is the strongest quantified evidence that **"don't tune, fuse" beats parameter search**.

### Analysis specific to your setup **[D] — arithmetic, not speculation**

Your configuration: 4096 complex samples, 64-bin FFT, hop 8 → 512 frames × 64 bins, **87.5% overlap** (unusually high; costs 4× the frames of hop 32 with no resolution benefit).

Two structural problems I did not see addressed anywhere in the literature, and which I think dominate any window-shape choice:

1. **Your analysis window spans a 30× range of absolute time.** With 4096 samples and sample rates from 1 to 30 MHz:
   - at 1 MHz: 4096 samples = **4.096 ms** → spans ~7 GSM bursts and ~6.5 Bluetooth slots. Duty-cycle and repetition structure fully visible.
   - at 30 MHz: 4096 samples = **136 µs** → **shorter than a single 577 µs GSM burst**. Burst periodicity is invisible in principle.

   So the very feature that most cleanly separates GSM from Bluetooth (burst duration and repetition period) is present in some of your samples and physically absent from others. A fixed hop of 8 *samples* corresponds to 8 µs at 1 MHz and 0.267 µs at 30 MHz — a 30× swing in the time axis of your spectrogram. **A 2D ViT with fixed 8×8 patches is therefore looking at patches spanning wildly different physical durations across the corpus.** My strong recommendation: **parameterize the STFT in absolute time (fixed hop in µs, fixed window in µs) using the known sample rate**, so that the time axis of every spectrogram means the same thing. You already do frequency and scale normalization; the time axis is the one invariance you have not enforced.

2. **64 bins is coarse for this discrimination.** After your preprocessing forces occupied fractional bandwidth to ~0.5, the signal occupies ~32 bins. GFSK (Bluetooth: 1 Msym/s, ~160–250 kHz deviation) vs GMSK (GSM: 270.833 ksym/s, 67.7 kHz deviation) differ in **instantaneous-frequency trajectory and symbol rate**, which a 32-bin-wide view resolves poorly. A longer FFT trades away the time resolution you need for burst structure — which is the argument for **multi-resolution parallel branches** (Schindler) rather than one setting.

3. **My honest ranking of what would most likely move your triad [D]:** the highest-value additional representation is probably **not** spectrogram, constellation, or wavelet — it is a **cyclostationary / cyclic-spectrum feature keyed to symbol rate**. GSM = 270.833 ksym/s, Bluetooth = 1 Msym/s, 802.11b DSSS = 11 Mchip/s. These are **exact, phase-invariant, burst-position-invariant, deterministic physical constants** that differ by 3.7× and 11×, and you know the absolute sample rate for every sample so you can convert cycle frequency to Hz. The SCF/cyclic-autocorrelation route is well established in AMC (used as one of three modalities in Han/Yu/Yang 2022; multiple SCF-CNN papers exist) **[A that the technique is standard; [D] that it targets your specific triad]**. The practical caveat is real: cyclic-spectrum estimation from only 4096 samples at arbitrary burst offset gives limited cyclic-frequency resolution and is noisy — expect this to need care, and expect the 30 MHz / 136 µs samples to be the hard ones again.


## Applicability


## Where this genuinely applies to your problem

**Strongest, most actionable transfer — the cascade construction (Hou et al. 2023).** Your 1D ViT / 2D ViT situation is a near-exact structural match to the published case that produced the largest per-confusion gain I found. Their I/Q model had one confusion it could not fix; they merged the confusable classes into a meta-class, let the I/Q model do the reduced-cardinality problem, and let a second-representation model do the split within the meta-class only. Your equivalent: 1D model does 7-way with {bluetooth, gsm, dsss} collapsed to one meta-class, spectrogram model arbitrates only within the triad, with **cw absent from its output alphabet**. Your arithmetic says this captures the 93 gsm→bluetooth errors the spectrogram fixed while excluding the 135 cw-absorption errors it introduced. In an episodic prototypical setup this is easy: enroll the triad-arbiter on triad classes only, and gate on the first model's nearest-prototype being in the triad. **This is the one recommendation I would make with confidence.**

**Calibrated expectations on ensemble gain.** PANNs is the cleanest, most-replicated instance of your exact scenario (1D-waveform model + 2D-spectrogram model, one clearly weaker): the fused model gained **+0.008 mAP over the better branch (0.439 vs 0.431)**, ~1.9% relative. If you do naive fusion rather than routing, budget for a gain of roughly that scale — from 0.876 to maybe 0.885 — not a jump to 0.92. And Wood et al. 2023's diversity/accuracy tradeoff predicts that fusing a 0.850 model that introduced a *new* error class with a 0.876 model can be net-negative. The 135-vs-93 arithmetic in your own numbers says exactly this.

**The absolute-time STFT parameterization.** This is my own extrapolation, not a literature finding, but it is arithmetic: your 63 distinct sample rates mean your 4096-sample window spans 136 µs to 4.096 ms, a 30× range, and your fixed 8-sample hop means the time axis of your spectrograms is not comparable across samples. Fixed 8×8 ViT patches therefore see physically incomparable regions. You already enforce frequency, scale, and power invariance in preprocessing; time-axis invariance is the missing one, and it is cheap to add. If the spectrogram branch is worth anything at all, this should raise its ceiling above 0.850.

**The cyclostationary suggestion is a hypothesis, not a finding.** The literature establishes that SCF/cyclic features are a standard AMC modality; it does not establish that they fix a GFSK/GMSK/DSSS triad. My reasoning that symbol rate (270.833 ksym/s vs 1 Msym/s vs 11 Mchip/s) is the exact physical invariant separating your triad is sound signal theory, but the estimation problem from 4096 samples at arbitrary burst offset is genuinely hard and I have no paper showing it works at that record length. Rate this as "highest-expected-value thing to try next," not "known to work."

## Where this research would NOT help you

**Constellation diagrams will not touch your triad.** This is the clearest negative. GFSK and GMSK are both constant-envelope continuous-phase — both map to the same ring in I/Q. A constellation branch is structurally blind to bluetooth-vs-gsm, which is your actual problem. It might separate dsss (real DBPSK/DQPSK/CCK constellation) from the ring pair, and would likely help ofdm, but the cost/benefit for the confusion you care about is poor. Additionally, constellation representations need carrier and symbol timing recovery and enough symbols to populate the cloud — with unknown symbol rate, arbitrary burst offset, and 136 µs windows at high sample rates, that is unreliable in your corpus specifically.

**Wavelets are not worth the cost here.** The one clean head-to-head (Phan 2024) went to spectrograms at every SNR and on 3 of 4 signal types, at 1/37th the compute, with the wavelet winning only on the impulsive class by +0.003 AUC. And the burst structure that distinguishes your classes (577 µs GSM bursts, 625 µs BT slots, 4.615 ms GSM frames) is *slow* structure well within STFT's reach — it is not the sharp-transient regime where wavelets earn their keep.

**Fusion cannot fix a label-taxonomy problem, and I suspect that is your real ceiling.** Nothing in this literature addresses your observation that "bluetooth" pools GFSK classic-connected and BLE advertising — two structurally different duty cycles and packet structures — into a single prototype, or that "gsm" pools GMSK normal bursts with 8PSK/QPSK/16QAM/32QAM EDGE bursts. If bluetooth is genuinely bimodal in embedding space, a single mean prototype is a misspecified model and **no input representation fixes that**. Your measurement that bluetooth-vs-gsm centroid z-distance (0.713) is *smaller than either class's own within-class spread* (1.495, 2.053) is the signature of exactly this: the classes aren't overlapping so much as each being too diffuse to have a meaningful centroid. That points at multi-prototype / subclass-aware few-shot methods (Allen et al., "Infinite Mixture Prototypes for Few-Shot Learning," ICML 2019, is the canonical reference) rather than at representation fusion. **I did not research that area — it was outside the brief — but I think it is a larger effect than anything in this report, and I would weight it above the fusion work.** A cheap test that requires no new architecture: enroll sub-profile-level prototypes (bluetooth-classic and bluetooth-le as separate prototypes) and map back to the coarse label at prediction time. If triad leakage drops, the problem was never the representation.

**Reported percentage gains should not be transferred.** Essentially all the AMC fusion numbers above come from RML2016.10a/10b: 128-sample or 1024-sample windows, single sample rate, textbook single-carrier modulations, full duty cycle, SNR as the only impairment axis. Your corpus is bursty, 30×-multi-rate, protocol-level (not modulation-level), synthetic from 37 profiles, with multipath/CFO/IQ-imbalance, and evaluated few-shot episodically. The +2.67% or +4.5% figures are not predictions for your setup; the *mechanisms* are what transfer.

**Two things the literature simply does not cover, where you are on your own:** (a) how to fuse representations *inside* an episodic prototypical pipeline — whether to concatenate-and-renormalize embeddings, keep per-branch prototypes and fuse distances, or learn per-branch temperatures; I found zero papers on this; and (b) any principled STFT parameter selection for AMC, where I found no sweep at all.


## Confidence


**Mixed, and deliberately tiered — I have marked every claim [A]/[B]/[C]/[D] inline.**

**Well-supported (multiple independent sources or textbook signal theory):**
- Ensemble-diversity theory (Krogh & Vedelsby 1995; Tumer & Ghosh 1996; Wood et al. 2023 JMLR) — established, and the diversity/accuracy tradeoff result specifically warns against your naive-fusion case.
- PANNs waveform+spectrogram fusion numbers (0.431 / 0.389 / 0.439 mAP) — the most-replicated instance of exactly your setup, verified from the paper and confirmed by the official repo.
- Constant-envelope modulations (GFSK/GMSK/MSK) carrying near-zero constellation information — this is signal theory, not a contested empirical claim.
- Constellation representations requiring carrier/symbol recovery and adequate symbol count — universally acknowledged.
- Standard STFT conventions (Hann/Hamming, 50–75% overlap, overlap does not improve underlying resolution).

**Single-paper, numbers verified from the source (I fetched and read them):**
- Hou et al. 2023 Sci Rep cascade: 92.27→94.94 / 93.47→96.69, +17/+18.2% on the confused pair. **This is the load-bearing citation of the whole report and I verified it directly.**
- Bai et al. 2022 dual-modal GAF+IQ: 87.6 / 85.3 / 92.1 at −10 dB.
- Phan 2024 spectrogram-vs-scalogram AUC table, including the valve exception and the 2.9h-vs-109h compute figure.
- Oladunni & Wong 2025 third-modality-gives-nothing result (biomedical, not RF).
- Schindler et al. 2017 +3.56% multi-resolution gain.
- Cheraghinia et al. 2025 raw-IQ (2,4096) wireless technology recognition at 99.47% on 8 classes.
- Medaiyese et al. 2021 wavelet-scattering 98.9% at 10 dB (no baseline reported).

**Single-paper, could NOT verify (paywall / MDPI 403 / PDF parse failure) — reported as claimed, do not build on:**
- Lin et al. 2026 early-vs-intermediate-vs-late fusion ranking (I have the conclusion text, not the numeric tables). This is the primary evidence for §2's ranking and it is unverified — treat "intermediate fusion is best" as a working hypothesis.
- M-LSCANet SNR-gating (93.4/95.8%).
- Han/Yu/Yang 2022 attention-fusion gains; also note this paper has no attention-vs-concat ablation, so the attention mechanism's value is unsupported even on its own terms.
- Both MDPI wavelet-AMC papers (Electronics 14/19/3801, Computers 11/11/162) — MDPI blocked all fetches.
- Waqas et al. 2023 constellation/burst passage — the technical claim is uncontroversial but the attribution came via a search snippet, not the source.
- Sensors 21/1/210 multi-window augmentation per-setting accuracies.
- Han et al. 2021 single-branch (~96% CNN, ~85% SAE) values, which came from a figure-reading summarizer; the fused 99.8% is stated in text.
- MCANet (arXiv:2510.18336) — abstract contains no numbers, no named datasets, no named modalities. **I do not consider it citable.**

**My own extrapolation, clearly not from any paper:**
- The reconciliation of "late fusion is worst" with "cascade fusion is best" (dilution vs routing).
- The 93-fixed-vs-135-introduced net-negative arithmetic on your specific confusion matrix and the resulting cascade recommendation.
- The 30× time-span analysis of your window and the absolute-time STFT parameterization recommendation. This is arithmetic from your stated sample rates and standard GSM/Bluetooth timing constants, but no paper told me to do it.
- The claim that cyclostationary symbol-rate features are the highest-value next representation for your triad specifically.
- The claim that your bluetooth/gsm sub-profile pooling is a larger effect than representation choice. **I did not research few-shot multi-prototype methods — that was outside the brief — so this is a hunch backed by your own diagnostic (centroid distance 0.713 < within-class spread 1.495/2.053), not by literature.**

**Two genuine gaps in the literature, not gaps in my search — I looked specifically and found nothing:**
1. No rigorous FFT-size × hop × window sweep with accuracy numbers for AMC or RF burst classification.
2. No work at all on multi-representation fusion inside episodic prototypical / few-shot AMC.


---

# Burst Detection + Cyclostationary

## Findings

## 1. Do CVNNs actually beat 2-channel real nets on I/Q classification?

**Verdict: at fair comparison, mostly no. The robust finding is parameter efficiency, not accuracy. Several of the loudest accuracy claims dissolve under scrutiny.**

### The decisive skeptical result
Ashutosh Kumar, "When do complex-valued neural networks help? A study of representation, geometry, and optimization," arXiv:2605.27673, May 2026 (single-author preprint, Owl Autonomous Imaging / RIT, not peer reviewed, but the most methodologically careful treatment of exactly your question that exists).

On a controlled 3-class RadioML 2018.01A subset (BPSK/QPSK/8PSK, 8 SNR levels), with a CReLU complex model:
- Under "matched-shared-trial" selection (same hyperparameter trial index for both families): complex **0.7293** vs best real baseline **0.4999**, a **+22.94 percentage point** gap.
- Under independent per-family tuning over the **same 16-trial search space**: the gap collapses to **+2.46 pp**.
- Gradient telemetry attributes the inflated gap to high-learning-rate first-step instability and dead seeds in the *real* baseline, not to complex superiority. Under ModReLU and ZReLU, where the real baselines stayed stable, the gap was small or reversed.
- He replicates the same artifact on synthetic data, so it is not a RadioML-loader quirk.

This is the single most important thing in this report: a 23-point "CVNN wins" result and a 2.5-point result came from the same data, same models, same search space, differing only in the selection rule.

### Positive claims, with their caveats
- **Krzyston, Bhattacharjea, Stark, "High-Capacity Complex Convolutional Neural Networks For I/Q Modulation Classification," arXiv:2010.10717 (2020)** (related ICC Workshops 2020 paper "Complex-Valued Convolutions for Modulation Recognition using Deep Learning"). Peak **92.4%** on RadioML 2016.10a; claims >10% improvement over parameter-comparable, speed-comparable real architectures, "statistically significant in all networks." This is the strongest positive AMC claim I found. **I could only verify abstract-level claims; I could not extract the per-baseline tables, so I cannot confirm how the real baselines were tuned.** Given Kumar's finding, an unverified ">10% over comparable params" claim from 2020 should be treated as provisional.
- **Chen, Wong, Hamdaoui, Elmaghbub, Sivanesan, Dorrance, Yang, "An Analysis of Complex-Valued CNNs for RF Data-Driven Wireless Device Classification," IEEE ICC 2022, arXiv:2202.09777.** Genuinely parameter-matched (e.g. both RVNN and CVNN at 54,400 params on the Wired dataset; the authors explicitly concede matched params is "not an exact apples-to-apples comparison"). CVNN beats RVNN by **16 to 34%** across four datasets (LoRa indoor, LoRa outdoor, NE Wired, NE WiFi). **Critical scope limit: this is RF fingerprinting, i.e. device identification from hardware impairments. The label literally is the I/Q-imbalance and phase-noise signature. Do not transfer this effect size to modulation or protocol classification.** Their most useful ablation: adding the second channel (I-or-Q-only to IQ) gains **7 to 24% for CVNNs but under 5% for RVNNs**, which is the cleanest published evidence that complex weights exploit the I/Q joint structure a stacked-real net does not.
- **Xu et al., "A Lightweight Dual-Branch Complex-Valued Neural Network for AMC," Sensors (2025), PMC12031408.** LDCVNN **62.41%** average on RML2016.10a at **9.0K params**, vs CNN2 51.46%, CLDNN 59.53%, ResNet 46.32% (21,450K params), CDSN 50.59% (1,336K), CSDNN 58.66% (327K). Uses weighted Frechet mean filtering for complex-scaling equivariance. **Not parameter-matched; the real baselines are stock weak reference implementations.** This is a parameter-efficiency result presented as an accuracy result.
- **Zhao, Wang, Lei, Zhang, Wu, Huang, "Dualformer," arXiv:2606.31352 (2026).** DualNN 99.50% / CVNN 99.35% / DualNN-Comparable 99.70% / RVNN 96.87%. **Their "RVNN"/"Realformer" processes the I channel ONLY** (stated explicitly: "Models processing only the I-channel with real-valued parameters are designated Realformer"). The RVNN comparison is therefore worthless for your question. What *is* valuable: their CVNN converged slowest (best epoch 84 vs DualNN 46 vs RVNN 23), and CVNN was **substantially worse in low-data regimes** (500/1000/2000 samples per class). Their theory frame is the useful part: CVNNs have lower *approximation* error but higher *estimation* error because the complex loss surface traps optimizers in spurious local minima.

### Explicit negative results
- **Monning & Manandhar, "Evaluation of Complex-Valued Neural Networks on Real-Valued Classification Tasks," arXiv:1811.12351 (2018):** at matched capacity, complex models perform "equal to or slightly worse" than real. They also observe the imaginary weight parts simply track the real parts when the data has no complex structure.
- **Barrachina, Ren, Morisseau, Vieillard, Ovarlez, "Complex-Valued vs. Real-Valued Neural Networks for Classification Perspectives: An Example on Non-Circular Data," ICASSP 2021, arXiv:2009.08340.** CVNN shows statistically higher mean and median accuracy with lower variance, and less overfitting without regularization. **The stated mechanism is non-circularity: real and imaginary parts being statistically dependent, i.e. nonzero pseudo-covariance E[zz^T].** This is the key theoretical caveat, see the applicability section.
- Survey-level consensus (Bassey/Qian/Li arXiv:2101.12249; "Comprehensive Survey of CVNNs," arXiv:2407.19258; IEEE TNNLS "CVNNs: A Comprehensive Survey," 2022) is that gains over comparable real models are modest and domain-conditional. A claim circulates that for most complex activations, real-valued backprop reduces infinite-width complex training dynamics to ordinary real dynamics. **I saw this asserted in search summaries but could not pin it to a specific verified paper. Treat as unverified.**

---

## 2. Complex-valued attention / transformers for signals

- **Dong, Peng, Yang, Lu, Shi, "Signal Transformer: Complex-valued Attention and Meta-Learning for Signal Recognition," arXiv:2106.04392, published ICASSP 2024** (Tongji, PKU, CMU, IBM Watson). Proposes CAMEL, a complex-valued attentional meta-learner: complex attention plus the first complex-valued MAML with first-order stationary point convergence guarantees. Evaluated 5-way 1-shot and 5-shot on RadioML 2016.04C, plus an SNR>=0 split, an SNR=0 split, and a 5-class prediction/other transfer split. Claims best accuracy and fastest convergence among MAML-family meta-learners. **This is the closest published analogue to your episodic few-shot setup. I could not extract Tables 1 to 3 from the PDF text layer, so I cannot quote the actual numbers. Treat the magnitude as unverified.** Note it is MAML, not prototypical.
- **Huang et al., "Holographic Transformers for Complex-Valued Signal Processing: Integrating Phase Interference into Self-Attention," arXiv:2509.19331 (Sept 2025, rev Oct 2025).** "Holographic attention" modulates interactions by *relative* phase and coherently superimposes values; a dual-headed decoder that also reconstructs the input is used to fight "phase collapse." Proved to implement a discrete interference operator maintaining phase consistency under linear mixing. Evaluated on PolSAR classification and wireless channel prediction, **not on modulation or protocol classification.** Reports increased robustness to phase perturbations. Single paper.
- **Zhao et al., "Dualformer," arXiv:2606.31352 (2026).** The transferable idea: **DualNN shares one real parameter set across both I and Q channels**, a weight-tying that captures much of the complex-multiplication structure at half the parameters and FLOPs of a CVNN. It matched or beat the full CVNN (99.50 vs 99.35) with markedly better optimization stability and far better low-data behavior. Their argument: shared weights bound the maximum eigenvalue of the weight matrices, lowering the global Lipschitz constant and flattening the loss surface into wider basins.
- **Hioki, "Complex-Valued Phase-Coherent Transformer (PCT)," arXiv:2605.10123 (2026).** Non-RF benchmarks. Its stated prediction, confirmed in their runs: **"without an inductive advantage, the vanilla real transformer outperforms the vanilla complex transformer."** Naive complex lifting of softmax attention does not help; they argue softmax row-normalization (token competition) is inherently misaligned with phase-preserving computation, and replace it with a real-valued elementwise smooth gate on L2-normalized complex cosine similarity. They also cite an open report of "no remarkable difference vs real baselines" for softmax-of-complex attention. Single-author preprint, treat the fix as a hypothesis; the diagnosis is nonetheless a plausible explanation for why complex transformers underdeliver.
- **AMC transformer hybrids (CBADNN and similar, 2025)** report 64.02% / 65.50% overall accuracy on RadioML 2016.10a/10b. These are averaged over SNRs down to -20 dB and are not comparable to your 0.891.

---

## 3. Group/phase equivariance: helpful, or does it discard information?

**The literature says: enforcing strict equivariance is usually worse than soft/approximate equivariance, and it is provably beneficial only when the target is exactly invariant. Your 0.653-0.716 vs 0.891 has a direct published precedent.**

### The published precedent for your exact failure
**Chakraborty, Xing, Yu, "SurReal: Complex-Valued Learning as Principled Transformations on a Scaling and Rotation Manifold," arXiv:1910.11334v3 (2020), accepted IEEE TNNLS.** This is the closest published architecture to your z-plane operator: complex numbers modeled as a product manifold of nonzero scaling and planar rotation, convolution defined as a weighted Frechet mean that is **equivariant** to the scaling/rotation (phase) group, and a distance transform that is **invariant** to it.

RadioML 2016.10a result, quoted from the paper: at SNR 10 dB SurReal achieves **76.1%**, versus O'Shea's real-valued baseline **72.7%** and the Deep Complex Network baseline **76.3%**, at **0.7%** of O'Shea's parameters and 58% of DCN's. And the paper states plainly: **"Our SurReal underperforms the baselines at lower SNRs."**

The important reading: 76.1% at 10 dB is far below the ~92.4% that Krzyston's *unconstrained* complex CNN reports on the same dataset. On the one benchmark where a hard rotation-and-scaling-invariant architecture was measured, invariance bought roughly 100x parameter efficiency and paid for it with a materially lower accuracy ceiling. That is structurally identical to your result.

### The theory of why it costs you
- **Kumar 2026, Proposition 1.** A 2x2 real channel-mixing matrix W commutes with every rotation R_phi if and only if W = aI + bJ, which is exactly complex scalar multiplication. **A complex-valued linear layer is precisely the U(1)-equivariant subspace of a stacked-real two-channel layer.** Corollary that matters for you: your stacked-real CNN is a strict superset. It can already learn the phase-equivariant operator, plus the phase-non-equivariant channel interactions. Going phase-equivariant strictly removes capacity. Whether that helps depends entirely on whether the discarded non-equivariant component carries label information.
- **Elesedy & Zaidi, "Provably Strict Generalisation Benefit for Equivariant Models," ICML 2021, arXiv:2102.10333**, and **Elesedy, "Provably Strict Generalisation Benefit for Invariance in Kernel Methods," NeurIPS 2021.** The provably nonzero generalization benefit holds **only when the target distribution is itself invariant/equivariant** with respect to the compact group. The proofs decompose function space into symmetric and anti-symmetric parts under group averaging; the equivariant model's error inherits the entire anti-symmetric component of the target. Enforcing a symmetry the data does not exactly satisfy converts variance reduction into irreducible bias.
- **Wang, Walters, Yu, "Approximately Equivariant Networks for Imperfectly Symmetric Dynamics," ICML 2022 (PMLR v162), arXiv:2201.11969.** Relaxed-equivariance models **outperform both unconstrained baselines and strictly equivariant baselines** on simulated turbulence and real multi-stream jet flow. Replicated in spirit by "Approximate Equivariance in Reinforcement Learning" (arXiv:2411.04225), "Learning (Approximately) Equivariant Networks via Constrained Optimization" (arXiv:2505.13631), "Almost Equivariance via Lie Algebra Convolutions" (arXiv:2310.13164), and PEnGUiN (arXiv:2503.15615). **This is now reasonably well replicated across domains and is the single most transferable theoretical result for your situation.**
- **Layer equivariance does not give end-to-end invariance.** Kumar states this explicitly and demonstrates it: biases, nonlinearities, pooling, readout, and the training distribution all break it. His fixed-rotation RF row (unseen test-time carrier phase): complex **0.252**, real-stack **0.246**, phase-only **0.262**, magnitude-only **0.328**. Everything with coordinate dependence collapses toward chance; only the trivially-invariant magnitude view survives. After random rotation augmentation: complex **0.654**, real-stack **0.580**, phase-only **0.574**. So the "equivariant by construction" claim generally is not true of the whole network anyway.
- **The well-supported alternative is augmentation, not architecture.** Huang, Pan, Zhang, Qian, Gao, Wu, "Data Augmentation for Deep Learning-Based Radio Modulation Classification," IEEE Access 2020, arXiv:1912.03026. Rotation augmentation outperforms flip, both outperform Gaussian noise. Headline: **joint rotation-plus-flip augmentation on 12.5% of the training set beat the un-augmented baseline trained on 100% of it.**

### On why your specific operator scored 0.653-0.716 (my extrapolation, not from any paper)
Three causes are confounded in your experiment, and phase equivariance is likely the smallest of them.
1. The U(1) constraint removes at most half the channel-mixing degrees of freedom (2 free parameters instead of 4 per 2x2 block). By itself that should not cost 0.18 accuracy.
2. **The rational pole/zero spectral parameterization is the real restriction.** That is an LTI filterbank with a magnitude gate. A globally-LTI front end separates signals essentially by power-spectrum shape until a nonlinearity intervenes, and after your bandwidth normalization, GFSK, GMSK, and 11 Mbps DSSS have broadly similar smooth spectra. What actually separates them (symbol rate, hop rate, burst duty cycle, CPM modulation index, chip-rate cyclostationarity) is cyclostationary and higher-order, not linear-spectral. This is the more likely capacity bottleneck, and it is a property of the FNO/rational parameterization, not of phase equivariance.
3. **The mid-training collapse (0.716 at ep 1666, crash to 0.29, no recovery) is the known signature of unconstrained pole/zero parameterizations.** Poles migrating to or outside the unit circle makes the operator ill-conditioned and the gradient explode. The deep state-space-model literature treats this as a first-class problem: a discrete-time LTI SSM is asymptotically stable iff its poles lie strictly inside the unit circle, and the standard fix is a diagonal parameterization that is stable by construction (S4/S4D/S5 lineage; see also "Layer-Adaptive State Pruning for Deep State Space Models," arXiv:2411.02824, and "AIRE-Prune," arXiv:2602.00534, both of which restate that zero/center-of-stability initialization does **not** guarantee stability during training and that states can diverge and make training infeasible).
4. **modReLU is empirically the worst complex activation, independent of any symmetry argument.** Cole et al., "Analysis of Deep Complex-Valued Convolutional Neural Networks for MRI Reconstruction," arXiv:2004.01738, found CReLU significantly better than modReLU and zReLU in all cases. Kumar's activation ablation agrees. modReLU's learnable dead-zone bias can kill all units, and it is non-holomorphic with its defect concentrated at the origin. Choosing phase equivariance forced you into modReLU, and modReLU is a bad activation.
5. The sep-CMA-ES run at 0.653 vs backprop's 0.716 says nothing about the architecture. Diagonal-covariance CMA-ES on 56k dimensions is a sample-starved hill climber.

---

## 4. Does phase matter at all for modulation classification?

**Well-supported answer: it is task-conditional, and the effect is real but modest, and it flips sign between signal families. This is one of the more solidly replicated findings here, confirmed independently in at least three papers.**

- **Kumar 2026, Table 2 (controlled synthetic RF stress tests), accuracy by input view:**
  - PSK-only: complex **0.821**, phase-only **0.735**, real-stack **0.728**, magnitude-only **0.333 (chance)**.
  - QAM-only: **magnitude-only wins at 0.524**, complex 0.509, real-stack 0.502, phase-only 0.487.
  - Mixed PSK+QAM: complex 0.507, phase 0.487, real-stack 0.481, magnitude 0.365.
  - Low-SNR PSK: parameter-matched real 0.526 = complex 0.526, phase 0.479, magnitude 0.326. Phase views degrade fastest under noise.
  - High-SNR PSK: complex 0.949, real-stack 0.897, phase 0.892, magnitude 0.333.
  - Unit-magnitude mixed: phase 0.491, real-stack 0.490, complex 0.476, magnitude 0.200.
- **Chen et al., ICC 2022, magnitude (R) vs phase (T) vs both (RT) ablation:** on the OSU-LoRa datasets, magnitude-only beats phase-only by **20 to 60%** and beats RT by 11 to 20%. On the NE Wired/WiFi datasets the result **inverts**: phase-only beats magnitude-only by **37 to 62%**, and RT is roughly equivalent to phase-only. Same task family, opposite answers, purely dataset-dependent.
- **Zhang, Zhou, Zhang, Li, Li, "CSPMNet: Pareto-Efficient AMC With Learnable Complex Subband Phase Motion," arXiv:2605.25099 (2026).** The most direct published test of "phase alone vs phase plus magnitude." Ablation Table III, overall accuracy on RadioML2022.01A / RadioML2016.10B: **"PhaseMotion only" 61.36 / 59.30** vs full **CSPMNet 71.09 / 63.26**, i.e. removing the amplitude-preserving path costs **9.73 / 3.96 points**. Their explicit design rationale is that normalizing a phase descriptor to unit modulus (making it purely angular, i.e. amplitude-invariant) lets unreliable low-energy phase estimates be treated as equally trustworthy, and is therefore harmful. Headline efficiency: 71.09% OA at 0.241M params, 68.4x fewer parameters and 17.0x fewer FLOPs than IQCM-Net.
- **Input-representation literature generally:** reported deltas between raw I/Q and amplitude/phase inputs for CNN classifiers are small, commonly under 1 to 2%. One review reports up to 2% over I/Q and 12% over frequency-domain data at medium-to-high SNR for amplitude/phase. **I could not verify the primary source of the 2%/12% figure; treat as unverified.** The consistent qualitative message across reviews (for example "Deep Learning for Automatic Modulation Classification: A Review," Electronics 15(10):2163, 2025) is that input representation is worth single-digit percent. Raw I/Q preserves symbol rate, multipath, and CFO structure; FFT/STFT representations are prone to phase ambiguity and frequency-offset distortion and often assume prior synchronization or burst segmentation.
- **Your own 2D STFT log-magnitude ViT result is a clean instance of this trade.** Discarding phase cut gsm-to-bluetooth leakage from 236/614 to 143/614 and simultaneously created a cw-absorption failure (bluetooth-to-cw 59/615, am-to-cw 46/589, fm-to-cw 30/615) that no time-domain model had. That is exactly what the literature predicts: magnitude-only removes the phase-dominant confusions and introduces envelope-dominant ones.

## Applicability

## Where this genuinely applies to your problem

**1. Abandon the phase-equivariant operator line. This is the strongest, best-supported conclusion.** SurReal is the published precedent and it behaves exactly like your z-plane operator: hard-wired scaling/rotation invariance, ~100x parameter efficiency, and an accuracy ceiling (76.1% at 10 dB on RML2016.10a) well below unconstrained complex CNNs on the same data (92.4%), plus explicit underperformance at low SNR. Your 0.653-0.716 vs 0.891 is the same shape of result, not an implementation bug. Equivariance bought parameter efficiency in the literature too, never a higher ceiling.

**2. Kumar's U(1)-subspace proposition gives you a clean, citable reason to stop.** A complex linear layer is exactly the U(1)-equivariant subspace of a stacked-real 2-channel layer. Your production CNN already *contains* every phase-equivariant operator you could build. Nothing strictly equivariant can beat it in-family; you can only lose the non-equivariant capacity. The only remaining argument for equivariance is sample efficiency, and you have a regenerable simulator, so you can buy sample efficiency with data instead.

**3. Your production model is already in the regime the equivariance literature recommends, and you should not have expected to beat it with a strict-equivariance model.** A soft phase-invariant channel (the 12 hand features, all of which are phase-invariant by construction: c20/c40/c41/c42/c60/c63, envelope stats, IF spread) concatenated to an unconstrained learned waveform path is precisely the "approximately equivariant" design that Wang/Walters (ICML 2022) found beats both the unconstrained and the strictly-equivariant extremes. This is a positive finding about your existing baseline, not a to-do item.

**4. The complex-CLDNN at 0.857-and-climbing is the one CVNN direction the literature actually supports** (complex convs plus explicit envelope and instantaneous-frequency branches plus recurrent temporal pooling). But calibrate expectations from Kumar (real gap over a well-tuned real baseline: ~2.5 pp, not 23) and Dualformer (CVNN converges slowest, best epoch 84 vs 23, and is markedly worse in low-data regimes). Expect ~0.89 to 0.91, budget a long run, and tune the real and complex families independently, never at a shared trial index. If you compare them at the same LR and get a 20-point gap, Kumar's paper says you have measured a dead real seed, not an architecture.

**5. One cheap, testable middle ground: DualNN-style weight sharing** (one real parameter set applied to both I and Q). Half the parameters of a CVNN, most of the structure, empirically flatter loss surface and better low-data behavior. Worth a run before committing to a full CVNN.

**6. If you keep any rational/pole-zero spectral parameterization anywhere, constrain pole radii to be stable by construction** (log-magnitude or negative-real-part parameterization, the S4D/S5 approach). Your 0.716-to-0.29 collapse with no recovery is the textbook divergence signature, and initialization inside the stable region provably does not keep you there during training.

**7. Two findings bear directly on the triad, though neither is from the CVNN literature.**

  a. **Infinite Mixture Prototypes (Allen, Shelhamer, Shin, Tenenbaum, ICML 2019, PMLR 97:232-241) is the best-supported intervention I found for your actual failure mode.** It represents each class by an inferred *set* of clusters rather than one mean, interpolating between prototypical and nearest-neighbor, and reports **10 to 25% absolute accuracy improvement over prototypical networks specifically on super-class distributions**. Your bluetooth (2 sub-profiles), gsm (7 sub-profiles), and ofdm (~20 sub-profiles) classes are literally super-classes. Your own diagnostic clinches it: bluetooth-vs-gsm centroid z-distance 0.713 is *smaller* than either class's own within-class spread (gsm 2.053, bluetooth 1.495). A single mean prototype is the wrong estimator for a multimodal class, and no amount of repulsion loss fixes an estimator problem, which is exactly why interventions 2, 3, and 4 all failed the way they did (strong repulsion at lambda=8.0 improved leakage to 145/162 but collapsed bluetooth's own accuracy from 0.968 to 0.279, which is the signature of squeezing a bimodal cloud toward one point).

  b. **Cyclostationary conjugate/non-conjugate structure separates your exact triad, and none of your current features capture it.** From the classical CSP literature (Gardner lineage; principally documented on Chad Spooner's cyclostationary.blog and in blind-modulation-classification theses such as escholarship qt6vp4k5h0): DSSS-BPSK exhibits **many** non-conjugate cycle frequencies (chip rate and harmonics) whereas most single-carrier PSK/QAM signals have only one; and **GMSK possesses only conjugate spectral-correlation features, while FSK and GFSK have significant features in both the non-conjugate and conjugate planes.** That is a direct physical separator for bluetooth (GFSK) vs gsm (GMSK) vs dsss, and the conjugate/non-conjugate distinction is itself a phase-symmetry statement, so it survives arbitrary carrier phase. Caveats: this is classical DSP, not a modern DL benchmark result; cycle frequencies scale with sample rate so your 63-value sample-rate spread and resampling stage must be accounted for; and your gsm class pools 8PSK/QPSK/16QAM/32QAM EDGE bursts which are *not* GMSK and would not show the GMSK conjugate-only signature, which argues again for sub-class prototypes rather than a single gsm feature.

## Where this does NOT apply, and where it would not help

- **Class mismatch.** Essentially the entire cited literature is 11-to-24-class *modulation* classification on RadioML at 128 to 1024 samples. You have 7 *protocol/technology* classes at 4096 samples with pooled heterogeneous sub-profiles. RadioML's canonical confusions are QAM16-vs-QAM64 and WBFM-vs-AM-DSB. **No paper I found reports or diagnoses a bluetooth/gsm/dsss triad.** Nothing here is a direct precedent for your failure mode.
- **The CVNN question is orthogonal to your bottleneck.** Nothing in the CVNN or equivariance literature addresses class heterogeneity, prototype geometry, or enrollment. Even if a CVNN delivered the literature-typical +2 to 3 pp, it would not touch a confusion that survived oversampling, two repulsion-loss variants, per-sample contrastive repulsion, a feature-normalization bug fix, both preprocessing conditions, a 750x training-length increase, hyperparameter sweeps, and a full corpus regeneration.
- **Do not transfer Chen et al.'s 16-34% CVNN gains.** That is RF fingerprinting, where the label *is* the hardware phase/IQ-imbalance signature. Wrong task family.
- **Number incomparability.** Most AMC overall-accuracy figures (60 to 72%) are averaged over SNRs down to -20 dB. They are not comparable to your 0.891.
- **A theoretical caveat that is my own extrapolation, not anyone's published result:** an I/Q signal with arbitrary carrier phase is *circular* (proper), meaning its distribution is invariant to multiplication by e^{j theta} and its pseudo-covariance E[zz^T] vanishes. Barrachina's demonstrated CVNN advantage is explicitly attributed to **non**-circularity. If your preprocessing (downconvert to DC, RMS normalize, arbitrary phase convention) leaves your data close to circular, the theoretical basis for a CVNN advantage over your stacked-real net largely evaporates. Your 75% impaired fraction does inject I/Q imbalance, which *is* a non-circularity-inducing impairment, but that is a nuisance variable, not your label. I found no paper testing this argument on protocol classification. Treat it as a hypothesis worth one cheap experiment (measure the pseudo-covariance of your preprocessed corpus) rather than a conclusion.
- **Nothing found tests CVNNs under episodic *prototypical* few-shot on RF.** The one few-shot complex-attention paper (CAMEL, ICASSP 2024) is MAML-based, and I could not extract its numbers.
- **My diagnosis of your z-plane collapse** (pole instability plus modReLU rather than phase equivariance per se) is inference from the SSM-stability and complex-activation literatures applied to your reported symptoms. It is the most parsimonious explanation available, but nobody has published this specific diagnosis for a z-plane FNO on RF.

## Confidence

Mixed, and I have tiered it deliberately.

**Well-replicated across independent groups (high confidence):**
- Strict equivariance underperforms approximate/soft equivariance when the symmetry is only approximate. Wang/Walters/Yu ICML 2022 plus at least four independent follow-ups (arXiv:2411.04225, 2505.13631, 2310.13164, 2503.15615) across turbulence, RL, and multi-agent domains.
- The generalization benefit of equivariance is provably conditional on the target actually being invariant. Two separate proof-bearing papers from Elesedy and Zaidi (ICML 2021, NeurIPS 2021).
- CVNNs deliver parameter efficiency at comparable accuracy. Consistent across SurReal, LDCVNN, CSPMNet, Dualformer, and the survey literature.
- Phase-vs-magnitude dominance is task-conditional and flips sign between signal families. Independently confirmed in Kumar 2026 (synthetic PSK vs QAM), Chen et al. ICC 2022 (LoRa vs Wired/WiFi, opposite directions), and CSPMNet 2026 (phase-only ablation costs 9.73 / 3.96 points).
- CReLU outperforms modReLU and zReLU. Confirmed in Cole et al. arXiv:2004.01738 and Kumar 2026 independently.

**Single-paper claims (medium confidence, do not over-weight):**
- SurReal's specific RadioML numbers (76.1% vs 72.7% vs 76.3% at SNR 10). Verified by extracting the paper's own text, so the quotation is reliable, but it is one paper on one dataset with a 2016-era real baseline.
- Kumar's 22.94-to-2.46 pp selection artifact. Verified from the PDF, has an internal synthetic replication, and is the most decision-relevant result in this report. But it is a **single-author, non-peer-reviewed 2026 preprint on a 3-class RadioML subset**. It deserves to change your priors, not to be treated as settled.
- CSPMNet's phase-only ablation, Dualformer's DualNN, all four complex-transformer papers. All single-paper, three of them unreviewed 2026 preprints.
- Krzyston's 92.4% and ">10% over comparable-parameter architectures." I verified the abstract but could not extract the baseline tables. Given Kumar's finding about selection rules, treat the effect size as unconfirmed.

**Could not verify (stated explicitly rather than presented as fact):**
- CAMEL/Signal Transformer's actual accuracy tables. PDF text layer would not yield them.
- The "2% over I/Q and 12% over frequency-domain for amplitude/phase inputs" figure. Appeared in a search summary; I could not trace the primary source.
- The claim that real-valued backprop reduces infinite-width complex training dynamics to real dynamics for most complex activations. Asserted in survey summaries; no verified primary source.
- Whether the 2022 Chen et al. CVNN advantage would survive independent per-family hyperparameter tuning. Nobody has redone it that way.

**My own informed extrapolation, labeled as such throughout:**
- The circularity argument (proper/circular I/Q removes the theoretical basis for a CVNN edge). Follows logically from Barrachina's stated mechanism but has never been tested on protocol classification.
- The diagnosis of your z-plane collapse as pole instability plus modReLU rather than phase equivariance. Parsimonious and consistent with the SSM-stability and complex-activation literatures, but not published for this architecture.
- Mapping the cyclostationary conjugate/non-conjugate distinction onto your specific bluetooth/gsm/dsss triad. The underlying CSP facts are classical and solid; the application to your exact confusion is mine, and it is complicated by your gsm class pooling non-GMSK EDGE profiles.

**Caveat on source recency:** four of the most on-point sources (arXiv:2605.27673, 2605.25099, 2606.31352, 2605.10123) are 2026 preprints with no peer review and, in two cases, single authors. I fetched and extracted each of them directly rather than relying on search summaries, so the quotations and numbers are accurate to the documents. Their scientific weight is a separate question.

**Overall:** the case for abandoning the strict phase-equivariant operator is strong and rests on multiple independent lines (SurReal's measured ceiling, the U(1)-subspace argument, the Elesedy proofs, and the replicated approximate-equivariance finding). The case that a CVNN will materially beat your 0.891 CNN is weak. The case that Infinite Mixture Prototypes addresses your actual triad failure is a strong single-paper claim with an exact structural match to your diagnostic numbers, and it is the one thing here I would test next.

---

# Complex-Valued / Equivariant

## Findings


# Current State of the Art in AMC / RF I/Q Classification (as of July 2026)

Caveat on method: this is a web-search survey. I fetched abstracts, HTML full texts, and PDFs where possible. MDPI and Wiley served 403/402 to my fetcher, so a few numbers below come from search-engine snippets of those pages rather than the papers themselves. I flag every such case. I also flag venue quality where it matters, because a meaningful fraction of the flashiest numbers in this field are self-published GitHub READMEs or predatory-adjacent journals.

---

## 1. WHAT ACTUALLY TOPS THE RADIOML LEADERBOARDS, AND WHAT THE NUMBERS MEAN

### Critical framing issue first
There is no maintained leaderboard. Papers With Code has no live RadioML benchmark page. What exists is a de-facto comparison table that gets copied paper to paper, using two metrics:
- **Average accuracy** over the full SNR sweep (-20 to +18 dB for 2016.10a, -20 to +30 dB for 2018.01a). This is the headline number and is dominated by the low-SNR regime where nothing works.
- **Peak / high-SNR accuracy**, typically at the top 2-3 SNR bins.

These are not comparable across papers because train/val/test splits are chosen by each author (this is an explicitly documented criticism of the datasets: they ship as a single set with no canonical split). Treat cross-paper deltas under ~1.5 points as noise.

### RadioML 2016.10a (11 classes, 2x128 samples, -20 to +18 dB)
The most credible recent head-to-head table I found is in **MoEformer** (Jiale Wang, Wupeng Xie, Yaxin Mu, Xin Liu, Zhilong Zhao, Jingwei Zhang, "Mixture-of-Experts Transformer for Automatic Modulation Recognition", arXiv:2606.09085v1, 8 June 2026), which reruns baselines under one protocol (6:2:2 split):

| Model | Avg acc | Peak acc | Params | Latency |
|---|---|---|---|---|
| ResNet | 57.08% | 84.45% | 96.3K | 0.032 ms |
| FEA-T | 59.80% | 90.50% | 168.6K | 0.081 ms |
| MCLDNN | 60.99% | 91.27% | 406.2K | 0.038 ms |
| AMC-Net | 61.96% | 91.64% | 466.1K | 0.034 ms |
| TLDNN | 63.00% | 93.23% | 243.3K | 0.041 ms |
| **MoEformer** | **63.74%** | **93.50%** | 187.7K | 0.059 ms |

On 2016.10b MoEformer reports 66.24% / 94.38%.

**The ceiling on 2016.10a peak accuracy is ~93-94%, and it is a dataset artifact, not a modeling failure.** This is well replicated across many papers: WBFM and AM-DSB cross-classify catastrophically even at +12 dB or higher, because the source audio has silent periods that reduce both to an unmodulated carrier. MoEformer explicitly states WBFM accuracy stays "below 50%" at high SNR for this reason. No architecture has ever fixed this, and none can, because the information is not in the signal.

### RadioML 2018.01a (24 classes, 2x1024 samples, -20 to +30 dB)
Best average accuracies cluster tightly at **64-65%**, best peak accuracies at **97-98.5%**:

| Model | Avg acc | Peak acc | Params | Source confidence |
|---|---|---|---|---|
| DP-DRSN (dual-path deep residual shrinkage) | 62.13% | 97.94% | lightweight | arXiv:2507.04586 / MDPI AI 6(8):195, 2025 |
| TLDNN | 63.17% | 97.60% | 276.7K | as rerun in MoEformer table |
| **MoEformer** | **64.22%** | 97.29% | 237.3K | arXiv:2606.09085, June 2026, fetched full text |
| SE-MSFN (multi-scale + squeeze-excite) | 64.50% | 98.5% | n/a | snippet only, could not fetch |
| **PHM-Net** | **64.59%** | **98.42%** | n/a | MDPI Electronics 15(12):2611, June 2026. Snippet only, MDPI 403'd me |
| MoE-AMC | **71.76%** (claimed) | n/a | n/a | arXiv:2312.02298, Dec 2023, no confirmed venue. **Outlier, distrust** |

On MoE-AMC: 71.76% is roughly 7 points above everything else and has not been reproduced by any of the 2026 comparison tables I found (MoEformer and PHM-Net both omit it). Its architecture is a gating MLP that routes to a ResNet "high-SNR expert" and a Transformer "low-SNR expert", with baselines quoted at MCFormer 61.77%, LSTM 62.51%, FEA-T 61.81%, HSRM-alone 66.05%, LSRM-alone 52.30%. My read: the gate is very likely leaking SNR label information, or the split differs. **Single-paper claim, do not treat as SOTA.**

**Claimed high-SNR record: 98.8%.** "AMC-Transformer: Automatic Modulation Classification based on Enhanced Attention Model" (2025), reported as 98.8% at SNR >= 10 dB on 2018.01a, tokenizing raw I/Q into patches with learned positional embeddings and multi-head self-attention, no convolutions, no handcrafted features. **Confidence: low.** I could locate it only as a repository record (real.mtak.hu/232834) and a personal GitHub reimplementation. I could not verify the venue or read the methods.

### Bottom line on leaderboards
Average-accuracy SOTA has moved roughly **63% -> 64.6%** on 2018.01a over about three years of published work, while parameter counts went **down**. That is a field in the flat part of its curve. Peak accuracy has been pinned at 97-98.5% since roughly 2021. **This is a plateau, and the literature knows it.**

---

## 2. GENUINELY NEW ARCHITECTURES, 2025-2026 (not CNN tweaks)

Five distinct families are actually new. In rough order of how much they matter:

### (a) Mixture-of-Experts routing
- **MoEformer** (arXiv:2606.09085, June 2026). Four experts, each operating on a **time-domain-resampled copy of the input at a different scale factor [0.5, 1.0, 1.5, 2.0]**, features FFT-aligned back to a common length, Local Representation Blocks with depthwise-separable convs, soft dense routing via MLP+softmax (not top-k), then a Transformer with **Rotary Position Embeddings** and a **convolutional FFN** replacing the position-wise FFN, then learnable attention pooling over time. 187-237K params. This is the most directly relevant new idea for your problem: multi-scale experts explicitly exist to handle the fact that different signal classes have different natural symbol/burst timescales.
- **MoE-AMC** (arXiv:2312.02298): SNR-regime routing, ResNet for high SNR, Transformer for low SNR. Numbers suspect, idea sound.
- **LWM-Spectro** (Namhyun Kim, Sadjad Alikhani, Ahmed Alkhateeb, ASU, arXiv:2601.08780, Jan 2026): three **protocol-specialized** expert encoders (WiFi / LTE / 5G) with a 2-layer router (d=64), top-1 selection at inference.

### (b) State space models (see section 3)

### (c) RF foundation models with self-supervised pretraining
This is the biggest structural shift in the field in 2025-2026.
- **LWM-Spectro** (arXiv:2601.08780): 12 Transformer blocks, d=128, 8 heads, 4x4 patches over STFT power spectrograms, max 1024 tokens, 70% mask ratio, pretrained on **9.2M spectrograms** from DeepMIMO ray-tracing across 20 city scenarios. Reported few-shot dominance: on joint SNR+Doppler classification with **5 samples/class**, fine-tuned LWM hits 76.53% vs Deep CNN 44.64%, ResNet-18 28.27%, EfficientNet-B0 30.74%, MobileNet-V3 7.12%. At 100/class: 94.43% vs 82.27% / 48.86% / 40.13% / 36.42%. Multi-protocol MoE-router reaches 89.8% F1 at 100 samples/class where a Deep CNN needs ~30x more data.
- **LatentWave: JEPA Pretraining for Wireless Foundation Models** (arXiv:2606.06373, June 2026). Joint-embedding predictive architecture over spectrograms and CSI. I did not fetch full numbers.
- **RIS-MAE** (Yunfei Liu, Mingxuan Liu, Wupeng Xie et al., arXiv:2508.00274, Aug 2025): masked autoencoder on **raw I/Q sequences, deliberately not time-frequency images**, random masking + reconstruction. Claims strong few-shot and cross-domain transfer on four datasets. Numeric results not in the abstract.
- **RF-MAE**: adaptive frequency masking that masks frequency components by energy distribution.
- **IQFM** (arXiv:2506.06718): multi-task SSL on raw I/Q, shared features for modulation classification + RF fingerprinting + AoA.
- **WavesFM / 6G WavesFM** (arXiv:2504.14100): ViT backbone, task-specific heads, LoRA adaptation.

### (d) Physics-guided coarse-to-fine (see section 5)

### (e) Metric-embedding / domain-generalization for protocol ID
- **"Approaching domain generalization with embeddings for robust discrimination and recognition of RF communication signals"** (Fraunhofer, arXiv:2510.23186, 27 Oct 2025). This is the closest published work to your setup and deserves your attention. They train **entirely on synthetic protocols** (1000 randomly-parameterized protocols per run, drawing from ASK/PSK/APSK/QAM/M-FSK/OFDM/FDM/CSS/DSSS, with phase offset, CFO sigma=0.02, 3GPP TDL channels, AWGN 3-30 dB, 50 instances per protocol per epoch, 20 epochs) and evaluate on **18 real captured protocol classes** including Bluetooth, Bluetooth LE, WiFi, WiFi-DSSS, DECT6, DroneID, and various RC links. Modified ResNet50: RN-1D (16.4M params, input 2x16384) and RN-2D (23.9M params, input 2x128x128 STFT). Embedding dim 128. Losses compared: plain softmax, norm-softmax, ArcFace (m=0.5, s=8).

### (f) Cross-domain multi-representation
- **DKDNet** (Shuang Wang, Chenxu Wang, Hantong Xing, Hanlin Mo, Lirong Han, Licheng Jiao, arXiv:2607.08031, 9 July 2026). Evaluates five signal representations, selects **I/Q + amplitude-phase + autocorrelation function (ACF)**, multi-representation encoder with a dynamic lightweight fusion unit, trained with classification + adversarial domain alignment. The ACF choice is notable and physically motivated.

### What is NOT happening
I found **no** credible KAN (Kolmogorov-Arnold Network) AMC paper, **no** equivariant/Lie-group AMC paper, **no** graph-neural-network AMC paper of consequence, and **no** diffusion-classifier AMC paper in 2025-2026. Diffusion appears only as a data augmentation generator (Zhao et al., IET Communications, 2025). Neural-operator approaches to AMC appear to be essentially unpublished, which is consistent with your FNO result: nobody has made that work either.

---

## 3. TRANSFORMERS, MAMBA, AND SSMs IN AMC

### Transformers: yes, but they do not win by much
FEA-T (59.80% avg on 2016.10a) actually **loses** to MCLDNN (a CNN-LSTM, 60.99%) and to TLDNN. Pure attention over raw I/Q has consistently underperformed convolutional-recurrent hybrids on short sequences. The transformers that do well are hybrids (MoEformer's conv-FFN + LRB front end; PHM-Net; CNN-Transformer). **Well-replicated finding: a plain patchified ViT over raw time-domain I/Q is not competitive with a good conv hybrid.** This matches your 1D ViT plateauing at 0.876 below your 38K-param CNN's 0.891.

Where transformers clearly do win is **patch-based long-sequence tokenization for technology recognition**. Cheraghinia, De Poorter, Fontaine, Debbah, Shahid, "Foundation Model for Wireless Technology Recognition Using IQ Timeseries" (IDLab Ghent/imec + Khalifa, arXiv:2505.19390, May 2025): 4 encoder layers, ~860K params, input **(2, 4096)** raw I/Q, patch length P=128 stride S=128 giving ~32 tokens, dual objective (cross-entropy + masked patch reconstruction). 8 technology classes (Sigfox, LoRa, 802.15.4g, 802.11ah, LTE, WiFi 802.11ax, 5G-NR, DVB-T) captured at **1/10/20 MHz sampling rates**. Results:

| Model | 8-class acc |
|---|---|
| CNN | 54.89% |
| Vanilla Transformer | 65.32% |
| LSTM | 69.01% |
| iTransformer | 74.19% |
| Autoencoder+CNN | 78.01% |
| **Patch Transformer (theirs)** | **99.47%** |

Their patch-size ablation is directly useful to you: **going from patch 8 to patch 128 cuts training time 75% while holding >99.5% accuracy; patches of 512+ degrade sharply.** They also attribute the CNN/LSTM failure specifically to **heterogeneous sampling rates**, which is exactly your 63-distinct-sample-rate situation, and the patch transformer's robustness to it is the paper's central claim. Training cost: 81 s/epoch vs 3385 s/epoch for a vanilla transformer and 2422 s/epoch for LSTM; 1.2 GB vs 11.8 GB vs 33.3 GB memory.

Note the input shape is **(2, 4096)**, identical to yours. Note also this is a supervised 8-class problem on real captures, not few-shot prototypical, and 99.47% suggests the task is easier than yours (their classes are more spectrally distinct than GFSK-vs-GMSK).

### Mamba / SSM: yes, applied, and the result is more interesting than the headline
**MAMCA** (Yezhuo Zhang, Zinan Zhou, Yichao Cao, Guangyu Li, Xuanpeng Li), arXiv:2405.11263, **published in IEEE Communications Letters 2024, DOI 10.1109/LCOMM.2024.3474519**. Code at github.com/ZhangYezhuo/MAMC. Selective SSM backbone with reduced state-matrix dimensions plus a denoising unit (soft-thresholding with an attention-learned threshold, via abs -> global average pool -> linear projection). 16.8M params at length 4096.

Its own Table II is the load-bearing result:

| Dataset (length) | MAMCA | MCLDNN | Transformer | PET-CGDNN |
|---|---|---|---|---|
| RML2016.10a (128) | **60.79%** | **61.42%** | - | 60.58% |
| TorchSig-QAM (128-4096) | **89.67%** | 81.02% | 74.43% | - |

**Read that carefully. On 128-sample sequences Mamba LOSES to a 2020-era CNN-LSTM. On sequences up to 4096 it wins by 8.6 points over MCLDNN and 15.2 points over a Transformer.** This is the single most decision-relevant finding in this entire report for you, and I want to be explicit that it is a **single-paper claim** (one team, their own datasets, their own baselines), even though the venue is respectable.

Efficiency at length 4096: 73.9 MFLOPs (lowest of all baselines), 1.61 s/epoch training, 0.246 ms inference, described as 1.5-64.5% of competing models' training time and 0.27-31% of their inference time. At -15 dB it holds >75% accuracy, about 20 points above second-best, which the authors attribute to the denoising unit.

Other SSM work:
- **ConvMamba** (Applied Sciences 15(17):9805, 2025, "Modulation Recognition Algorithm for Long-Sequence, High-Order Modulated Signals Based on Mamba Architecture"). CNN front end + **Mamba2**. Claims better efficiency/accuracy balance than CNN or Transformer for long sequences. MDPI 403'd me; snippet only.
- **AWMN, Adaptive Wavelet Mamba Network** (MDPI AI 6(12):323, 2025). Lifting-scheme adaptive wavelet transform for time-frequency features + Mamba for long-range temporal dependencies, aimed at OFDM AMC. Reported accuracies 62.39% / 64.50% / 74.95% on three datasets. MDPI 403'd me; **I could not verify which number maps to which dataset.**

I found **no** S4 / S5 / structured-SSM (pre-Mamba) AMC paper, and **no** bidirectional-Mamba AMC paper with published numbers. Bidirectional SSM is established elsewhere for exactly this kind of data (Audio Mamba, arXiv:2406.03344; EEGMamba, arXiv:2407.20254) but the RF community has not published it.

---

## 4. THE HIGH-SNR CEILING AND SATURATION

**Yes, saturation is real, documented, and attributed to specific irreducible class pairs.** This is a well-replicated finding, not a single-paper claim.

- **2016.10a**: peak ~93-94%. Cause: WBFM vs AM-DSB, from silent audio segments. Universal across papers.
- **2018.01a**: peak 97-98.5%, with one unverified 98.8% claim. Causes, consistently reported: (i) high-order densely-packed constellations, 128APSK / 256QAM / 128QAM / 32APSK confusing with adjacent PSK/APSK/QAM at minimal Euclidean distance under fading; (ii) the AM family, AM-DSB-SC vs AM-DSB-WC vs AM-SSB-SC vs AM-SSB-WC. One source reports **AM-DSB-SC being confused with 128APSK and 128QAM specifically**, which is a suppressed-carrier-looks-like-dense-constellation failure. Another notes ~6 dB SNR is needed to reach 90% overall.
- MoEformer states directly that below -10 dB **all methods converge** because the features are destroyed, so avg-accuracy differences are entirely earned in the -10 to +10 dB band.
- Diminishing-returns evidence on depth: DP-DRSN reports going from 4 to 6 denoising blocks gained a marginal 62.77% average while FLOPs rose to 30.92 M and energy to 30.71 mJ with no speed gain.
- **Sequence length gives diminishing returns at high SNR.** Chain-Net-lineage studies (arXiv:2009.02023 and related) report that doubling 512 -> 1024 samples gives **+4.08% average at SNR <= 0 dB but only +0.72% at SNR >= 2 dB**, and per-doubling gains at -10 dB decay 6.88% -> 5.83% -> 1.87%. So longer windows buy you low-SNR robustness, not clean-signal discrimination.

**Interpretation I am confident in:** on both RadioML benchmarks the residual high-SNR error is Bayes error given the observation, not model error. Papers that report 98%+ are all bumping the same wall, and the remaining 1.5-2% is composed of physically ambiguous pairs.

**Interpretation that is my extrapolation, flagged as such:** your 0.876 ViT plateau across a 100x training increase, flat from episode ~20k, has the same signature. A plateau that is invariant to 750,000 episodes, to preprocessing, to hyperparameters, to loss modification, and to corpus regeneration is a statement about the observation, not the optimizer.

---

## 5. HIERARCHICAL AND COARSE-TO-FINE CLASSIFICATION

This is an old idea in AMC that is being rediscovered with modern machinery. Three tiers of evidence:

### (a) Classical (well established, pre-DL)
Hierarchical decision trees over higher-order statistics: separate amplitude-modulated from angle-modulated first, then subdivide. Hierarchical polynomial classifiers on high-order cumulants (Digital Signal Processing / Elsevier lineage). These are the ancestors of your 12 hand-features. Established, and the reason they were abandoned is error propagation: a wrong coarse decision is unrecoverable.

### (b) Modern hierarchical AMC (2025-2026)
- **PHM-Net: A Physics-Informed Hierarchical Multi-Scale Network for AMC**, MDPI Electronics 15(12):2611, June 2026. Three components: **Transient Feature Gating (TFG)**, **Cross-Resolution Signal Aggregation (CRSA)**, and a **Physics-Informed Hierarchical Loss (PI-HL)** that enforces consistency between coarse-grained and fine-grained predictions over the modulation taxonomy. Reported 64.59% avg / 98.42% peak on 2018.01a, claimed +11.14 avg over AMC-Net and +9.43 over CNN-Transformer. **Snippet-only; I could not read the paper or verify the loss formulation. The claimed margins over baselines are much larger than MoEformer's rerun of the same baselines, so treat the deltas skeptically even if the absolute number is plausible.**
- **ZoomSpec: A Physics-Guided Coarse-to-Fine Framework for Wideband Spectrum Sensing** (Zhentao Yang, Yixiang Luomei, Zhuoyang Liu, Zhenyu Liu, Feng Xu, arXiv:2604.13568, 15 April 2026, 14 pages / 8 figures / 5 tables, filed cs.CV). This one is worth reading because its front end **is your preprocessing pipeline**, formalized as a differentiable module. Components: **Log-Space STFT (LS-STFT)** for constant relative resolution instead of linear frequency spacing; a lightweight **Coarse Proposal Net (CPN)** screening the full band; an **Adaptive Heterodyne Low-Pass (AHLP)** module doing center-frequency alignment, bandwidth-matched filtering, and safe decimation; then a **Fine Recognition Net (FRN)** that **fuses the purified time-domain I/Q sequence with spectral magnitude via dual-domain attention**. Reports 78.1 mAP@0.5:0.95 on the SpaceNet real-world dataset. I could not extract the ablation table (PDF text extraction failed), so **the individual contribution of AHLP versus the fusion is unverified**.
- Earlier two-stage CNN work: coarse classification from second-order cyclostationary features with a first-level CNN, then selectively activating a second CNN on high-order features within the chosen subclass.

### (c) Hierarchical / multi-modal prototypes in few-shot learning (this is the part that maps to your architecture)
This literature is outside RF but is directly on point for "one prototype per class pools two structurally different sub-profiles":
- **Infinite Mixture Prototypes (IMP)**, Kelsey Allen, Evan Shelhamer, Hanul Shin, Joshua Tenenbaum, **ICML 2019** (PMLR v97, arXiv:1902.04552). Replaces one-prototype-per-class with a **DP-means-style infinite mixture**, inferring the number of clusters per class from the data rather than fixing it. The paper's headline motivation is verbatim your problem: *"by clustering each super-class into multiple modes, IMP is better able to generalize to sub-classes."* Their strongest gains are on **alphabet-level Omniglot** tasks, i.e. exactly the case where a single label spans structurally distinct members. I attempted to fetch both the arXiv and PMLR PDFs and my text extraction failed on both, so **I have not verified the specific accuracy deltas**. The method and its motivation are solidly established (well-cited ICML paper); the numbers I am not quoting because I could not read them.
- **Leveraging Hierarchical Structures for Few-Shot Musical Instrument Recognition** (arXiv:2107.07029, ISMIR 2021). Hierarchical prototypical network where **the prototype for a parent node in the class tree is the mean of its children's prototypes**, trained multi-task across levels. Directly transferable: your ofdm class (LTE + NR + WiFi-OFDM, ~20 profiles) and gsm class (7 profiles, GMSK plus EDGE 8PSK/QPSK/16QAM/32QAM) are textbook parent nodes.
- Related: multi-prototype networks with local descriptors (LMPNet, Pattern Recognition 2021), covariance-matrix prototypes for heterogeneous feature distributions, clustering-based prototype modification for few-shot relation classification (KBS 2023), Improved Prototypical Networks (IPN, Pattern Recognition Letters 2020) which reweights support samples by representativeness rather than mean-pooling them.

---

## 6. NEGATIVE RESULTS AND FAILED APPROACHES

These are the parts I would want if I were you.

1. **Complex-valued networks do not beat real-valued ones at matched capacity.** Mönning and Manandhar, "Evaluation of Complex-Valued Neural Networks on Real-Valued Classification Tasks" (arXiv:1811.12351): when parameter counts are matched, complex models are **equal to or slightly worse**. The follow-up consensus is that the AMC literature's pro-CVNN results (e.g. IEEE 9128039, "Complex-Valued Networks for Automatic Modulation Classification") mostly **do not control for parameter count**. A 2025 evaluation (ScienceDirect S2666827025001252) concludes CVNNs win only when the architecture is specifically designed around complex operations, not from complexification alone. **Your FNO result (0.716, then catastrophic collapse to 0.29) and your ES-trained variant (0.653) are consistent with the capacity-matched literature, not anomalous.**

2. **Mamba loses on short sequences.** MAMCA's own table: 60.79% vs MCLDNN 61.42% on 2016.10a at length 128. The authors publish this. The win only materializes at extended length.

3. **ArcFace is not universally better for RF embeddings, and specifically hurts 1D.** From the Fraunhofer paper (arXiv:2510.23186), TPR at FPR=1e-3 on 18 real protocols:

| Method | TPR@FPR=1e-3 | TPR@FPR=1e-4 |
|---|---|---|
| ModFeat (hand features) | 0.168 | 0.118 |
| SCF-PCA (cyclostationary) | 0.258 | 0.238 |
| RN-1D / Norm-Softmax | 0.606 +/- 0.035 | 0.519 +/- 0.052 |
| RN-2D / ArcFace | 0.685 +/- 0.055 | 0.582 +/- 0.053 |

For 1D input, **norm-softmax beat ArcFace**; the authors call ArcFace's 1D underperformance surprising. For 2D input ArcFace won. This is a **single-paper** result but a carefully controlled one.

4. **Same paper, a finding that contradicts yours and you should know about it: RN-2D consistently outperformed RN-1D.** Their 2D input was 128x128 STFT, their 1D input 2x16384. Your measurement was the opposite (2D ViT 0.850 vs 1D ViT 0.876, and 16x slower). Differences that plausibly explain the divergence: their sequence is 4x longer, their classes are 18 real protocols rather than 7 pooled families, their STFT resolution is much higher than your 64-bin/hop-8, and they use ResNet50 not ViT. **I flag this as a genuine unresolved disagreement rather than pretending your result generalizes.**

5. **Hand-crafted statistical features are far weaker than learned embeddings under strict FPR.** Same table: ModFeat 0.168, SCF-PCA 0.258 vs 0.606-0.685 for embeddings. But note the same paper found **SCF-based classification slightly outperformed embeddings at high SNR**, and 2D ResNet only surpassed embeddings below 6 dB. Cyclostationary features are not dead; they are complementary.

6. **Depth past a point is pure cost.** DP-DRSN: 4 -> 6 denoising blocks bought a marginal average while FLOPs went to 30.92M.

7. **Longer windows buy low-SNR robustness, not clean-signal separability** (the +0.72% at high SNR figure above).

8. **RadioML itself is criticized as a flawed benchmark.** RML16 has "errors and ad-hoc choices of parameters", which motivated **RML22** as a corrected regeneration. The datasets lack a canonical split, and hardware artifacts (IQ imbalance, phase noise) and real spectral congestion are absent. Several groups have moved to TorchSig/Sig53 and WidebandSig53 (550K samples, 53 classes, ~2M signals) or CSRD2025 instead.

9. **No published AMC work that I could find studies a class label that pools structurally distinct waveform sub-profiles.** RadioML labels are modulation-pure by construction. This is a real gap in the literature and it means the benchmarks provide no guidance on your specific pathology.


## Applicability


## Where this genuinely applies to your 7-class problem

**1. The Mamba length result is the single most actionable finding, and it is aimed exactly at your regime.** MAMCA (IEEE Comm. Letters 2024) shows Mamba losing to CNN-LSTM at 128 samples and beating it by 8.6 points at lengths up to 4096. Your input is 4096 complex samples. Your best early trajectory so far (0.857 at episode 1200, still climbing) came from the CLDNN-style model with per-branch BiLSTM temporal pooling, i.e. the architecture family Mamba is a drop-in replacement for. Swapping BiLSTM for a bidirectional selective SSM in that 98K-param multi-branch model is a small, cheap, well-motivated experiment. It also fixes the practical problem: MAMCA reports 0.246 ms inference and 73.9 MFLOPs at length 4096, which is compatible with shipping in an app. **Caveat: this is a single-team claim, and their 89.67% figure is on their own TorchSig-QAM construction, not a shared benchmark.**

**2. Multi-prototype / hierarchical prototypes address a defect you have already diagnosed but not acted on, and the literature says it is the right move.** You wrote that bluetooth pools GFSK-classic and GFSK-LE, gsm pools GMSK normal bursts and EDGE 8PSK/QPSK/16-32QAM, and ofdm pools ~20 LTE/NR/WiFi profiles, and that a single prototype forces each into one point. Allen et al.'s Infinite Mixture Prototypes (ICML 2019) exists precisely for that, and its strongest reported gains are on superclass-level tasks. The hierarchical-prototype variant (parent prototype = mean of child prototypes, ISMIR 2021) fits your case even better because you **already have the 37 profile labels**, so you do not need to infer clusters, you can supervise them. This is the intervention your #1-#9 list has not tried: every one of your nine interventions manipulated the loss, the data balance, the normalization, or the training budget, and none of them changed the **label granularity**. That is the untouched axis.

**3. Your ceiling has the documented signature of Bayes error, and the literature supports treating it that way.** The 2016.10a WBFM/AM-DSB wall and the 2018.01a AM-DSB-SC/128APSK wall are both cases where the field collectively stopped attributing the error to the model. Your bluetooth-vs-gsm centroid z-distance of 0.713, smaller than either class's own within-class spread (2.053 and 1.495), is a direct measurement of the same condition in your feature space. GFSK and GMSK are both CPFSK with similar modulation index; the physics says they are separable mainly by **symbol rate and burst timing structure** (GSM 270.833 ksym/s in 577 us bursts on a 4.615 ms frame; Bluetooth classic 1 Msym/s in 625 us slots hopping 1600/s), not by constellation or cumulant geometry. Nothing in your 12 phase-invariant hand-features encodes symbol rate or burst periodicity, and your preprocessing **resamples to a canonical fractional bandwidth of 0.5**, which is exactly the operation that destroys absolute symbol-rate information. That is likely why native beat resampled for every architecture you tried, and it is a strong argument for adding an explicit cyclostationary or autocorrelation channel rather than another backbone.

**4. DKDNet's representation choice and the SCF results back that up.** DKDNet (arXiv:2607.08031, July 2026) evaluated five representations and kept I/Q + amplitude-phase + **autocorrelation function**. The Fraunhofer paper found SCF-PCA slightly outperforming learned embeddings at high SNR on 18 real protocols including Bluetooth, BLE, WiFi, and WiFi-DSSS. DSSS in particular has a textbook chip-rate cyclic feature that no amount of time-domain convolution is guaranteed to surface. Your dsss-to-bluetooth leakage originating in the learned waveform path rather than the hand features (your own diagnostic: dsss/bluetooth z-distance 2.956, well separated in hand-feature space) is consistent with the learned path lacking the right inductive bias, not with the classes being ambiguous.

**5. ZoomSpec formalizes your preprocessing and adds the piece you are missing.** Its AHLP module is your downconvert-plus-bandwidth-match-plus-decimate, made differentiable, and its Fine Recognition Net **fuses purified time-domain I/Q with spectral magnitude via dual-domain attention**. Your 2D-STFT ViT reduced gsm-to-bluetooth leakage from 236/614 to 143/614 but introduced a cw-absorption confusion no time-domain model had. That is the textbook profile of two **complementary** error distributions, and the literature's answer to complementary errors is fusion, not choosing one. A dual-branch model keeping your 1D CNN path and adding a coarse spectrogram path is better supported by the evidence than either alone.

**6. Multi-scale experts (MoEformer) target the same thing.** Four experts on time-resampled copies at [0.5, 1.0, 1.5, 2.0], FFT-aligned, soft-routed. Your classes live at wildly different symbol rates across 63 sample rates and 16 bandwidths. A single receptive-field stack is a compromise across all of them.

## Where this would NOT help you

- **The RadioML numbers themselves are near-useless as targets.** 24 modulation-pure classes at 1024 samples with a fixed 200 kHz sample rate is not your task. Your 7 classes are technology families spanning heterogeneous profiles at 63 sample rates. Nobody's 98.4% peak accuracy tells you anything about whether 0.891 is good.
- **Foundation-model pretraining (LWM-Spectro, LatentWave, RF-MAE, IQFM) is out of reach and probably unnecessary.** LWM-Spectro pretrained on 9.2M spectrograms. Your corpus is 14k samples from one simulator, regenerable but from the same generative process, so masked-reconstruction SSL would mostly learn your simulator's quirks. The few-shot gains those papers report come from pretraining diversity you do not have.
- **The 2016.10a/2018.01a average-accuracy race is irrelevant to you.** The 63-65% band is dominated by SNR below -10 dB. Your corpus is 75% impaired but not driven to -18 dB, and your production metric is overall accuracy on an impairment-inclusive split.
- **Mamba will not fix the triad confusion by itself.** If the bluetooth/gsm overlap is genuinely Bayes error in your current observation, a better sequence model reallocates error rather than removing it. Note that your strong-repulsion experiment (lambda=8.0) already demonstrated exactly this: leakage improved to 145/162 while bluetooth's own accuracy collapsed 0.968 to 0.279 and overall dropped to 0.839. Any purely architectural intervention risks the same trade.
- **Nothing in the literature addresses your specific triad.** RadioML 2018.01a contains GMSK and OQPSK but **no GFSK, and no DSSS at all**. There is no published GFSK-vs-GMSK-vs-DSSS confusion study. The bluetooth/GSM/WiFi-DSSS classes appear together only in the Fraunhofer 18-protocol set, and that paper reports aggregate TPR without a per-class confusion breakdown.
- **The 2D-beats-1D finding from Fraunhofer directly contradicts your measurement** and I am not going to smooth that over. Their setup differs on four axes at once (16384 vs 4096 samples, 128x128 vs 64-bin/hop-8 STFT, ResNet50 vs ViT, 18 real classes vs 7 pooled synthetic families). It is a reason to re-test STFT resolution, not a reason to abandon your result.
- **KAN, equivariant/Lie-group, GNN, neural-operator, and diffusion-classifier approaches have no AMC track record.** Your FNO experiment is, as far as I can tell, ahead of the published literature, and its outcome (0.716 then collapse to 0.29) is consistent with the capacity-matched CVNN literature. I would not read that as "the idea needs more tuning."


## Confidence


Mixed, and I have tagged each claim inline. Breaking it down:

**Well-replicated, multiple independent sources (high confidence):**
- RadioML 2016.10a peak accuracy ceiling ~93-94%, caused by WBFM/AM-DSB silent-audio ambiguity. Appears in many papers over many years.
- RadioML 2018.01a peak ceiling 97-98.5%, caused by high-order QAM/APSK proximity and the AM-DSB/SSB carrier variants. Multiple sources.
- Average-accuracy SOTA on 2018.01a sitting at 64-65% with sub-1-point-per-year progress. Confirmed by two independent 2026 comparison tables (MoEformer, PHM-Net) plus several 2025 papers.
- Transformers on raw I/Q not beating conv-recurrent hybrids on short sequences. Multiple independent tables.
- RadioML's methodological weaknesses (no canonical split, ad-hoc parameters, missing hardware artifacts), which motivated RML22, TorchSig/Sig53, and CSRD2025.
- Complex-valued networks failing to beat capacity-matched real-valued networks. Multiple sources including an explicit 2018 study and a 2025 re-evaluation.
- Longer sequences helping mainly at low SNR. Consistent across the length-ablation literature.

**Single-paper claims (medium confidence, flagged as such in the findings):**
- MAMCA's crossover result (Mamba loses at length 128, wins by 8.6 points at extended length). One team, respectable venue (IEEE Communications Letters 2024), own baselines and own TorchSig-QAM construction. **This is the most consequential claim in the report and it rests on one paper.**
- Cheraghinia et al.'s 99.47% on 8-technology recognition with a 4-layer patch transformer, and their attribution of CNN/LSTM failure to sampling-rate heterogeneity.
- The Fraunhofer embedding results: RN-2D > RN-1D, ArcFace underperforming for 1D, norm-softmax winning for 1D, SCF beating embeddings at high SNR.
- MoEformer's 63.74% / 64.22% and its multi-scale-expert design.
- LWM-Spectro's few-shot numbers.
- MoE-AMC's 71.76%. **I explicitly distrust this one** and recommend against citing it.

**Verified only via search snippet, not by reading the source (low confidence on specifics, moderate on existence):**
- PHM-Net's 64.59% / 98.42% and its claimed +11.14 / +9.43 margins over AMC-Net and CNN-Transformer. MDPI returned 403. The claimed margins are inconsistent with MoEformer's independent rerun of the same baselines, so I would trust the absolute number more than the deltas.
- SE-MSFN 64.50% / 98.5%.
- AWMN's 62.39 / 64.50 / 74.95, where I could not determine the dataset mapping.
- ConvMamba's architecture and results.
- ZoomSpec's ablation table (I got the abstract and architecture description but PDF text extraction failed on the body, so the 78.1 mAP figure is confirmed but the per-module contribution is not).
- AMC-Transformer's 98.8% at SNR >= 10 dB. **Weakest item in the report.** Located only via a repository record and a personal GitHub. Venue unverified.
- Infinite Mixture Prototypes' specific accuracy numbers. Both PDF fetches failed text extraction. The method, motivation, authors (Allen, Shelhamer, Shin, Tenenbaum), and venue (ICML 2019, PMLR v97) are confirmed; the numeric results are not, so I deliberately did not quote any.

**My own informed extrapolation, clearly not from any paper:**
- That your 0.876 ViT plateau across 750,000 episodes is Bayes error rather than optimization failure. The reasoning is by analogy to the documented RadioML walls plus your own centroid-distance measurement, not a cited result.
- That your canonical-bandwidth resampling step destroys absolute symbol-rate information and that this explains both the native-beats-resampled result and part of the GFSK/GMSK confusion. This is a physics argument I constructed; I found no paper making it.
- That your 1D-vs-2D complementary error pattern (STFT fixing gsm-to-bluetooth while introducing cw absorption) calls for fusion rather than selection. Supported in spirit by ZoomSpec and DKDNet, but neither paper analyzes this specific complementarity.
- The specific recommendation to supervise sub-profile prototypes using your existing 37 profile labels rather than inferring clusters. That is my synthesis of IMP plus the ISMIR hierarchical-prototype work applied to your setup.

**Explicit gaps I could not close:**
- No published GFSK vs GMSK vs DSSS confusion analysis exists that I could find. RadioML 2018.01a has GMSK and OQPSK but no GFSK and no DSSS.
- No AMC paper studies class labels that pool structurally distinct waveform sub-profiles.
- No S4/S5 AMC paper, no bidirectional-Mamba AMC paper with numbers, no KAN/equivariant/GNN/neural-operator AMC paper of consequence.
- I could not access the two most relevant survey articles (Zheng et al., International Journal of Intelligent Systems 2025, paywalled 402; and the Signal Processing survey on data representations / model structures / regularization, paywalled), either of which would likely have a consolidated comparison table better than the one I assembled.


---

# Metric Learning for Confusable Classes

## Findings

# RESEARCH FINDINGS

Confidence tags used throughout: [A] = well-replicated / read directly from the primary PDF, [B] = single-paper claim, [C] = my own extrapolation or reasoning.

---

## 1. MULTI-PROTOTYPE / MULTI-CENTER METHODS

### 1.1 Infinite Mixture Prototypes (the closest literature analogue to your situation)

**Allen, Shelhamer, Shin, Tenenbaum. "Infinite Mixture Prototypes for Few-shot Learning." ICML 2019, PMLR v97:232-241. arXiv:1902.04552.**

Method: each class is represented by a *set* of clusters, not one mean. Cluster count is inferred per-episode by DP-means (Kulis & Jordan), where the creation threshold lambda is derived analytically from the concentration parameter alpha and the learned cluster variance sigma rather than tuned by hand. A new cluster is spawned whenever a point's minimum distance to all existing means exceeds lambda. As lambda goes to infinity IMP degenerates to ProtoNet; as lambda goes to 0 it degenerates to nearest-neighbour. IMP therefore *interpolates* between prototype and nearest-neighbour behaviour, per class, adaptively.

Reported numbers [B, extracted from the ar5iv rendering; the headline gain is independently corroborated by the abstract's "10-25% absolute improvement" claim on two sources]:

| Task | ProtoNet | IMP | 1-NN |
|---|---|---|---|
| Omniglot **alphabet** recognition, 10-way 10-shot | 65.6 +/- 0.4 | **92.0 +/- 0.1** | 92.4 +/- 0.2 |
| Alphabet-train to character-test, 20-way 1-shot | 82.1 +/- 0.4 | **95.4 +/- 0.2** | - |
| Held-out character modes (60% chars withheld), "both modes" | 82.9 +/- 0.4 | **96.6 +/- 0.2** | - |
| Held-out, "testing modes" only | 77.7 +/- 0.4 | **94.9 +/- 0.2** | - |
| Omniglot 5-way 1-shot (standard) | 98.2 +/- 0.3 | 98.4 +/- 0.3 | 98.4 |
| Omniglot 20-way 5-shot (standard) | 98.6 | 98.6 | 98.3 |
| miniImageNet 5-way 1-shot | 47.0 +/- 0.8 | 49.6 +/- 0.8 | 49.6 |
| miniImageNet 5-way 5-shot | 66.1 +/- 0.7 | 68.1 +/- 0.8 | 59.4 |

**The structure of the win matters more than the number.** The "alphabet" task is exactly your situation: a label that pools structurally distinct sub-populations (an alphabet is a superclass over its characters, the way "gsm" is a superclass over 7 burst profiles). On that task a single prototype gets 65.6% and multiple prototypes get 92.0%, a 26.4-point absolute gain. On standard Omniglot/miniImageNet, where classes are roughly unimodal, IMP is within noise of ProtoNet (98.4 vs 98.2). **This is a clean negative result too: multi-prototype buys you nothing when the class is genuinely unimodal.** It only pays when the label pools modes.

Note that 1-NN matches IMP on the alphabet task (92.4 vs 92.0). That is diagnostic: when a class is strongly multi-modal, even naive nearest-neighbour beats a mean-prototype badly. If you want a five-minute cheap test of whether your triad problem is a multi-modality problem, swap nearest-prototype for k-NN over the enrollment pool at inference time only, with no retraining. If dsss->bluetooth leakage drops materially, multi-modality is implicated. If it does not, it is not.

### 1.2 Nearest Class Multiple Centroids (NCMC), the older and better-measured result

**Mensink, Verbeek, Perronnin, Csurka. "Distance-Based Image Classification: Generalizing to New Classes at Near-Zero Cost." IEEE TPAMI 35(11), 2013.** (I read the extended book-chapter version PDF directly, so these are verbatim [A].)

Method: k centroids per class obtained by plain k-means on the within-class features; class posterior is the sum of the per-centroid soft-min posteriors, p(c|x) = sum_j p(m_cj | x). This corresponds to a Gaussian mixture per class with equal mixing weights and a shared covariance. Metric (low-rank projection W) learned by maximising log-likelihood of correct classification under this multi-centroid model.

ILSVRC'10, flat top-5 error (lower is better), 4K-dim Fisher vector features:

| Proj. dim | NCM (k=1) | NCMC-test (metric trained k=1, k>1 at test only) | NCMC k=5 | NCMC k=10 | NCMC k=15 |
|---|---|---|---|---|---|
| 128 | 39.0 | 36.3 (k=30) | 36.2 | **35.8** | 36.1 |
| 256 | 37.4 | 36.1 (k=20) | 35.0 | **34.8** | 35.3 |
| 512 | 37.0 | 36.2 (k=20) | 34.8 | **34.6** | 35.1 |

Reference points on the same features: one-vs-rest linear SVM baseline 38.2, k-NN with learned metric 39.0, WSABIE 38.5.

Four things worth extracting from this table:

1. **Multi-centroid beat the flat linear SVM.** NCMC k=10 at 512d gives 34.6 versus 38.2 for one-vs-rest SVM. That is 3.6 absolute points *in favour of* a distance-based classifier over a discriminatively trained linear head, once the class is allowed more than one centre. [A]
2. **Just switching to multi-centroid at inference time, with a metric trained for k=1, already recovers most of the gain.** 39.0 to 36.3 at 128d without retraining anything. That is a very cheap experiment for you: re-cluster your enrolled embeddings per class into k sub-prototypes and use min-over-sub-prototypes, with the existing trained network untouched. [A]
3. **There is an optimum k and it is moderate.** k=10 beat k=15 at every projection dimension. Too many centroids degrades toward 1-NN and overfits.
4. **Honest counter-evidence.** On the 64K-dim features, NCMC (k=10, d=512) got 29.4 versus NCM 30.7, but the linear SVM baseline got 28.0 and still won. So multi-centroid does not universally dominate a learned linear head. It dominated at 4K features and lost at 64K features. [A]

### 1.3 Deep Nearest Centroids (DNC), the modern version, trained end to end

**Wang, Han, Zhou, Liu. "Visual Recognition with Deep Nearest Centroids." ICLR 2023. arXiv:2209.07383.** (Read the PDF directly [A].)

Method: sub-centroids discovered by within-class clustering using Sinkhorn-iteration cluster assignment, momentum-updated in an external memory so it is trainable with small batches (256) on ImageNet, at about 5% training slowdown. Training alternates class-wise clustering and nearest-sub-centroid classification. There is no parametric classifier layer at all; the only learnable parameters are the embedding.

ImageNet val top-1, parametric softmax vs DNC:

| Backbone | Parametric | DNC | Delta |
|---|---|---|---|
| ResNet50 | 76.20 | 76.49 | +0.29 |
| ResNet101 | 77.52 | 77.80 | +0.28 |
| Swin-S | 83.02 | 83.26 | +0.24 |
| Swin-B | 83.36 | 83.68 | +0.32 |

Ablation on number of sub-centroids K (Table 5a), ImageNet, ResNet101:

| K | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| top-1 | 77.31 | 77.54 | 77.68 | **77.80** |

ADE20K semantic segmentation, mIoU vs K: K=1 43.2, K=5 44.0, K=10 **44.3**, K=20 44.0.

Interpretation: on ImageNet, where classes are curated to be reasonably coherent, going from 1 to 4 sub-centroids buys +0.49 top-1. On ADE20K, where a semantic class like "building" or "plant" genuinely pools wildly different appearances, K=1 to K=10 buys +1.1 mIoU and K=20 over-fragments. The size of the multi-prototype gain tracks how heterogeneous the classes actually are. [A] This is the same pattern as IMP.

Also relevant: the paper's explicit framing of the limitation of parametric heads, "for each class, only one single weight vector is learned ... thus they essentially assume unimodality for each class, less tolerant of intra-class variation." That criticism applies to a linear softmax head *as well as* to a single-prototype head. See section 4.

### 1.4 Other multi-prototype work

- **PALM, "Learning with Mixture of Prototypes for Out-of-Distribution Detection," ICLR 2024, arXiv:2402.02653.** Multiple prototypes per class, each sample soft-assigned to a subset of prototypes via reciprocal-neighbour weights, trained with an MLE compactness loss plus a *prototype-level* contrastive loss for inter-class separation. The prototype-level contrastive term is the interesting bit for you: it repels prototypes rather than sample means, which avoids the failure you hit in your intervention #3 (strong mean-repulsion collapsing bluetooth's own accuracy) because sub-prototypes of a bimodal class are not forced together first. [B, numbers not verified]
- **LMPNet, "Local descriptor-based multi-prototype network for few-shot learning," Pattern Recognition 2021.** Multiple local-descriptor prototypes per class via an sSE attention module. [B, numbers not verified]
- **Prototype Propagation Network / MahiNet**, coarse-to-fine few-shot where a coarse prototype aggregates descendant fine prototypes. [B]

---

## 2. LOSSES FOR FINE-GRAINED / CONFUSABLE CLASSES

### 2.1 The headline numbers (face recognition, where these losses were born)

- **SphereFace (Liu et al., CVPR 2017)**: multiplicative angular margin. LFW 99.42 versus 98.71 for plain softmax on the same setup. [B]
- **CosFace (Wang et al., CVPR 2018)**: additive cosine margin. LFW 99.73, YTF 97.6. [B]
- **ArcFace (Deng, Guo, Xue, Zafeiriou, CVPR 2019, arXiv:1801.07698)**: additive *angular* margin, so the margin corresponds exactly to geodesic distance on the hypersphere. LFW 99.83. MegaFace figures around 98% identification / VR@FAR 1e-6 depending on protocol; I saw a search summary claiming 96.98 VR@FAR1e-6 and could not verify the exact cell against the paper, so treat the MegaFace digit as unverified. The paper's own framing is that ArcFace "forms an upper envelope of CosFace" under fair comparison, and that the constant linear angular margin is the geometric advantage over SphereFace's and CosFace's nonlinear margins. [B]

### 2.2 The reality check, which is the more important finding for you

**Musgrave, Belongie, Lim. "A Metric Learning Reality Check." ECCV 2020. arXiv:2003.08505.**

The paper's thesis: metric-learning papers routinely claimed 50-100% relative improvement over contrastive loss and about 50% over triplet, but those claims rest on unfair comparisons (different backbones, no held-out validation, test-set model selection, mismatched embedding dims). Under a controlled protocol with 4-fold cross-validation, a fixed BN-Inception backbone, and Bayesian hyperparameter search per loss, the improvements shrink to marginal, frequently inside confidence intervals.

Reproduced numbers, concatenated 512-dim embeddings [B, extracted from ar5iv, digits not double-checked against the PDF]:

| Loss | CUB200 P@1 / MAP@R | Cars196 P@1 / MAP@R | SOP P@1 / MAP@R |
|---|---|---|---|
| Contrastive | 68.13 / 26.53 | 81.78 / 24.89 | 73.12 / 44.39 |
| Triplet | 64.24 / 23.69 | 79.13 / 23.02 | 72.65 / 43.37 |
| CosFace | 67.32 / 26.70 | 85.52 / 27.57 | - |
| ArcFace | 67.50 / 26.45 | 85.44 / 27.22 | 76.20 / 47.41 |
| SoftTriple | 67.27 / 26.51 | - | - |

Read that carefully: **on CUB200, the most fine-grained of the three, ArcFace and CosFace are slightly WORSE than a well-tuned plain contrastive loss** (67.50 and 67.32 versus 68.13 P@1). On Cars196 the margin losses win by 3.7 points; on SOP by 3.1. So angular margin is not a fine-grained panacea. Its documented strength is large-class open-set retrieval (thousands to millions of identities), which is a different regime from 7 classes. [B, but this is the single most-cited corrective in the field and I would weight it heavily]

### 2.3 Triplet with hard negative mining

**Hermans, Beyer, Leibe. "In Defense of the Triplet Loss for Person Re-Identification." arXiv:1703.07737, 2017.** Batch-hard (within each minibatch, for each anchor take the hardest positive and hardest negative) consistently beat batch-all, and matched offline semi-hard mining at a fraction of the cost (7h vs 20h training in their experiments). The paper is also candid that fully-hard mining outside the batch destabilises training and can collapse the embedding, which is why they restrict hardness to within-batch. [B]

This is directly relevant to your intervention #3: strong prototype repulsion (lambda=8.0) improved leakage 236->162 but collapsed bluetooth's own clean accuracy from 0.968 to 0.279. That is the classic hard-negative collapse mode. The literature's answer is not "turn the margin down" (you tried, nothing happened) but "restrict the hardness scope and give the class more than one attractor," which is the multi-prototype fix.

### 2.4 Center loss

**Wen, Zhang, Li, Qiao. "A Discriminative Feature Learning Approach for Deep Face Recognition." ECCV 2016.** Learns one center per class and penalises distance from features to their class center, jointly with softmax. Requires the softmax term for stability; center loss alone collapses. I did not verify specific numbers. [B]

**Structural warning [C]:** center loss and ArcFace/CosFace/SphereFace all *assume and enforce unimodality per class*. They explicitly optimise intra-class compactness around a single center or a single class weight direction. Applied to "gsm" (7 heterogeneous burst profiles) or "ofdm" (~20 LTE/NR/WiFi profiles), they will fight the data. I would predict, and this is extrapolation not a cited result, that ArcFace on your 7 coarse labels reproduces the exact failure of your intervention #3: it will pull the bimodal classes toward a single point, improve the between-mean margin, and destroy per-class accuracy on the heterogeneous classes. If you want angular margin, apply it at the **37-profile** level, not the 7-class level.

### 2.5 Supervised contrastive

**Khosla et al. "Supervised Contrastive Learning." NeurIPS 2020. arXiv:2004.11362.** ResNet-50 on ImageNet: 78.7% top-1 with **batch size 6144**. ResNet-200: 81.4%, reported as 0.8% above the best prior number for that architecture. [B]

The batch size is the load-bearing detail. SupCon's advantage comes from having many positives and many negatives per anchor. Your episodic protocol is 7-way 5-shot, roughly 70 samples per episode. Your intervention #4 (per-sample contrastive repulsion across all pairs in the triad) is essentially a small-batch SupCon and it moved nothing (215/237). That is consistent with the literature rather than contradicting it: SupCon at batch 70 is not SupCon.

---

## 3. HIERARCHICAL (COARSE FAMILY, THEN FINE LEAF) CLASSIFICATION

### 3.1 The strongest negative result, and it is quite strong

**Bertinetto, Mueller, Tertikas, Samangooei, Lord. "Making Better Mistakes: Leveraging Class Hierarchies with Deep Networks." CVPR 2020. arXiv:1912.09393.**

Top-1 **error** (lower better) [B, ar5iv extraction]:

| Method | tieredImageNet-H | iNaturalist-19 |
|---|---|---|
| Flat cross-entropy | **31.58** | **44.04** |
| HXE (hierarchical cross-entropy) alpha=0.2 | 32.31 | - |
| HXE alpha=0.5 | 37.24 | 52.06 |
| Soft-labels beta=30 | 31.72 | - |
| Soft-labels beta=10/5 | 37.29 (beta=5) | 55.54 (beta=10) |
| YOLO-v2 style top-down hierarchical classifier | 34.24 | 45.00 |
| DeViSE | 36.27 | - |
| Barz & Denzler | 41.53 | 60.11 |

Hierarchical distance of mistakes (lower better): flat CE 6.97 on tieredImageNet-H, best hierarchy-aware 6.58; on iNat19 flat CE 2.43, best 2.10.

**Every single hierarchy-aware method loses top-1 to flat cross-entropy.** The paper states this explicitly as "an inherent tension between performance in the top-1 sense and in the hierarchical sense." What hierarchy buys you is *less severe* mistakes, not *fewer* mistakes. Note in particular the top-down hierarchical classifier (YOLO-v2 style, which is the "coarse family then fine leaf" cascade you asked about): 34.24 vs 31.58 flat on tieredImageNet-H, and 45.00 vs 44.04 on iNat19. It is worse on both, on both metrics on tieredImageNet-H. [B, but the qualitative conclusion is stated by the authors and I regard it as solid]

### 3.2 Where hierarchy does win

- **HD-CNN (Yan et al., ICCV 2015)** embeds CNNs in a two-level hierarchy where a coarse classifier routes to fine classifiers, and reports state-of-the-art on CIFAR100 and ImageNet-1k at the time. I could not extract the magnitudes; treat as unverified. [B] The architectural principle that matters is that HD-CNN gives the fine classifiers *extra dedicated capacity* on the confusable subset. That is a different mechanism from hierarchy-aware *losses*, which just reweight the same flat model.
- **E-commerce / plant-species hierarchical taxonomies**: reported gains of "up to 2.2%" over flat on both fine and coarse accuracy in the two-view plant-species work (Neurocomputing 2021, arXiv:2005.09110). Small. [B]
- The consistent framing across this literature is that a cascade helps when the coarse decision is *reliable* and the fine stage gets something the flat model did not have (more capacity, a different input view, or a different feature set). A cascade that only re-partitions the same features generally loses to flat, because cascade errors are unrecoverable.

### 3.3 Hierarchical AMC specifically

Standard AMC practice is a two-stage family-then-order cascade (identify ASK/PSK/QAM/FSK family, then M within family). Recent work (e.g., PHM-Net, Electronics 2025) reports +11.14 and +9.43 percentage points average accuracy on RML2018.01a over comparison AMC methods using hierarchical taxonomic structure plus multi-scale features. I have not verified those numbers against the paper and the comparison baselines matter enormously in AMC literature. [B, low confidence on the magnitude]

The well-documented AMC confusion pairs are 16QAM/64QAM and QPSK/8PSK (same family, different order), and GFSK/CPFSK/B-FM at low SNR. Your bluetooth/gsm confusion is structurally identical to the QPSK/8PSK case: same modulation family (continuous-phase FSK), differing only in a scalar parameter.

---

## 4. IS NEAREST-CENTROID FUNDAMENTALLY WEAKER THAN A LEARNED HEAD FOR HETEROGENEOUS CLASSES?

This is the question where I can give you the most precise answer, and the answer has a twist.

### 4.1 The measured gap: learned head beats nearest-centroid on the SAME embedding

**Tian, Wang, Krishnan, Tenenbaum, Isola. "Rethinking Few-Shot Image Classification: A Good Embedding Is All You Need?" ECCV 2020. arXiv:2003.11539.** Table 5, ResNet-12 backbone, ablation of the base learner on a fixed embedding. (Read directly from the PDF [A].)

| Head | miniIN 1-shot | miniIN 5-shot | tieredIN 1-shot | tieredIN 5-shot | CIFAR-FS 1/5 | FC100 1/5 |
|---|---|---|---|---|---|---|
| NN (nearest neighbour, Euclidean) | 56.29 | 69.96 | 64.80 | 78.75 | 64.36 / 78.00 | 38.40 / 49.12 |
| LR (logistic regression) | 58.74 | **78.31** | 67.62 | **84.77** | 66.92 / 84.78 | 40.36 / 57.23 |
| LR + L2-normalised features | 61.56 | 79.27 | 69.53 | 85.08 | 71.24 / 85.63 | 42.77 / 58.86 |

The 5-shot gaps are 8.35 (miniIN), 6.02 (tieredIN), 6.78 (CIFAR-FS), 8.11 (FC100) points in favour of the learned head. The 1-shot gaps are 2.0-2.8 points, which makes sense because with one support sample per class the two heads are nearly the same object. The authors' own summary: "logistic regression significantly outperforms the nearest neighbour classifier, especially for the 5-shot case." [A]

**Important caveat I must flag.** Their "NN" is stated as *nearest neighbour with Euclidean distance*, not nearest **centroid**. In 5-shot, a prototype (mean of 5) is usually meaningfully better than 1-NN. So the true prototype-vs-LR gap on matched embeddings is probably smaller than 8.35 points. Cross-checking: Meta-Baseline (Chen et al., arXiv:2003.04390) reports ResNet-12 cosine-prototype 5-shot miniImageNet around 79, which is right on top of RFS's LR+L2 number of 79.27. So the honest reading is: **on unimodal benchmark classes, a well-normalised prototype head and a learned linear head are roughly equivalent, and the RFS table overstates the gap by using 1-NN as the stand-in.** [C, my reconciliation of two papers]

### 4.2 The geometry, which is the part that actually settles your question

This is elementary and I state it as established geometry rather than as a paper's claim [A on the mathematics, [C] on the application to your case]:

- A **single-prototype nearest-centroid** classifier over C classes induces a Voronoi tessellation. Every class's decision region is an intersection of half-spaces, hence **convex**.
- A **linear softmax head** assigns x to argmax_c (w_c . x + b_c). Every class's decision region is again an intersection of half-spaces, hence **also convex**.

So the linear head does **not** fix multi-modality. It cannot represent a non-convex class region either. What the linear head buys over nearest-centroid is that w_c is free rather than pinned to the class mean, so it can point in a direction that spans both modes and re-weight dimensions. That is a real but limited gain, and it is exactly why the measured LR-vs-prototype gap is a few points and not tens of points.

What actually fixes non-convexity is one of:
1. **Multiple prototypes per class** (union of convex cells, hence non-convex). Mensink, IMP, DNC, PALM.
2. **A nonlinear head** (MLP with a hidden layer).
3. **Training against the sub-labels and pooling to the coarse label**, which is a special case of (1) with the clusters given rather than inferred.

Chen et al., "A Closer Look at Few-shot Classification," ICLR 2019, is the relevant background for (2)-vs-prototype: Baseline (linear layer) vs Baseline++ (cosine-distance classifier, which is a learned prototype per class) are competitive with each other and with meta-learning once the backbone is deep enough. Their point is that the head is rarely the bottleneck; the representation is. [B]

### 4.3 Bottom line on question 4

Nearest-centroid is **not** fundamentally weaker than a linear head. Both are convex-region classifiers. It **is** fundamentally weaker than a *multi*-centroid classifier or a nonlinear head when a class has separated sub-populations, and in that regime the failure is severe rather than marginal: 65.6 vs 92.0 in IMP's alphabet experiment, 39.0 vs 34.6 top-5 error in Mensink's NCMC.

The specific pathology when a class is bimodal and you take the mean: the prototype lands in the *gap between* the two modes, which is territory that may belong to a neighbouring class, and every sample of both modes is now further from its own prototype than samples of the neighbour are. That is precisely the shape of your bluetooth situation, where two GFSK sub-profiles with different duty cycles and packet structures are averaged into one point, and gsm and dsss both leak *into* that point.

---

## 5. LABEL NOISE, INHERENTLY AMBIGUOUS CLASSES, AND TELLING "MODEL LIMITATION" FROM "NOT SEPARABLE"

### 5.1 The definitional point that resolves most of the confusion

The Bayes error is a property of the **joint distribution of (input representation, label)**, not of the model. Ishida, Yamane, Charoenphakdee, Niu, Sugiyama, "Is the Performance of My Deep Network Too Good to Be True? A Direct Approach to Estimating the Bayes Error in Binary Classification," ICLR 2023 (oral), arXiv:2202.00395, is built entirely on that: their estimator is "model-free and even instance-free," with no hyperparameters, and they use it to argue that ViTs may have reached the Bayes error on some benchmarks and to detect test-set overfitting. [B]

The critical corollary for you [C, but it follows directly]: **your preprocessing pipeline is part of the input representation, therefore it changes the Bayes error.** "bluetooth and gsm are not separable" is not a well-formed statement until you fix the preprocessing. If the resample-to-canonical-fractional-bandwidth step destroys absolute occupied bandwidth and absolute symbol rate, then you may have manufactured the irreducibility yourself.

Also relevant caveat: **Ishida's estimator requires soft labels** (multiple annotator labels per instance). You do not have those. A 2025 follow-up (arXiv:2505.20761, "Practical estimation of the optimal classification error with soft labels and calibration") reports that naively plugging CIFAR-10H soft labels into standard estimators produces "unreasonably high" Bayes error estimates, because the soft labels themselves come from a shifted labelling condition. So the ICLR-2023 tool is not directly usable on a synthetic corpus with hard generative labels. [B]

### 5.2 What IS usable: the Cover-Hart bound

Cover & Hart (1967): asymptotically, `Bayes_err <= NN_err <= Bayes_err * (2 - C/(C-1) * Bayes_err)`. For a binary sub-problem this gives roughly `Bayes_err >= NN_err / 2`. So: take the bluetooth-vs-gsm binary sub-problem, measure the asymptotic 1-NN error in your embedding on a very large sample, and you get a defensible *lower* bracket on the irreducible error in that representation. If 1-NN error on the pair is 35%, Bayes error in your representation is at least roughly 18-19%, which would tell you a large chunk of your leakage is genuinely irreducible **given the current preprocessing**. If 1-NN error is 4%, the model is at fault. This is cheap, requires no soft labels, and is exactly the diagnostic your question calls for. [A on the theorem, [C] on the application]

### 5.3 Data-complexity measures (this is what your z-distance diagnostic already is)

**Ho & Basu. "Complexity Measures of Supervised Classification Problems." IEEE TPAMI 24(3):289-300, 2002.** Attributes classification difficulty to (i) ambiguity of the classes, (ii) sparsity and dimensionality, (iii) complexity of the class boundary. Defines F1 (Fisher discriminant ratio), N2 (ratio of intra-class to inter-class nearest-neighbour distance), N4, T1, LSC. Survey update: Lorena et al., "How Complex Is Your Classification Problem? A Survey on Measuring Classification Complexity," ACM Computing Surveys 2019, arXiv:1808.03591. [B]

Your measurement (bluetooth-gsm centroid z-distance 0.713, smaller than either class's own within-class spread of 1.495 and 2.053) is essentially F1/N2 and it is the right instrument. Two important reads of it:

- **F1 and N2 are representation-relative, and they are also mode-blind.** A within-class spread of 2.053 for gsm is exactly what a 7-profile mixture produces. A large within-class spread from *multi-modality* is not the same as a large spread from *noise*, but F1 cannot tell them apart. **Recompute F1/N2 per sub-profile pair, not per class pair.** If bluetooth-classic-connected vs gsm-normal-burst has a healthy F1 while bluetooth (pooled) vs gsm (pooled) has F1 = 0.713, then your classes are separable and your *labels* are the problem, not the physics. That single computation would, in my view, settle your question. [C]
- Your bluetooth-vs-dsss (2.956) and gsm-vs-dsss (2.765) numbers already prove that the dsss->bluetooth leak of 217/573 is NOT an irreducible-overlap problem. Those classes are well separated in hand-feature space. That leak is a decision-geometry or learned-path problem, and it is exactly the kind multi-prototype fixes. [C, but it follows directly from your own measurement]

### 5.4 Hidden stratification: the paper that is about your exact structural situation

**Sohoni, Dunnmon, Angus, Gu, Ré. "No Subclass Left Behind: Fine-Grained Robustness in Coarse-Grained Classification Problems." NeurIPS 2020. arXiv:2011.12945.** ([A] on the framing, [B] on the number.)

Core observation: models trained on coarse labels have highly variable performance across the unlabelled subclasses inside those labels, and **unlabelled subclasses are usually separable in the feature space of the trained deep model**. GEORGE exploits this: cluster the feature space within each coarse class to recover approximate subclass labels, then use those as noisy groups in a group-DRO objective. Reported: worst-case subclass accuracy improves by **up to 22 percentage points** over standard ERM, with no subclass information supplied.

This is directly your situation, with one enormous advantage: **GEORGE has to infer the subclasses because nobody knows them. You already have them.** You have 37 named profiles. Every method in this section (IMP's DP-means, GEORGE's clustering, DNC's Sinkhorn clustering, PALM's soft assignment) exists to recover subclass structure that you already possess as ground truth. [C on the "you have it better" conclusion, but it is not a controversial inference]

### 5.5 Subclass distillation

**Müller, Kornblith, Hinton. "Subclass Distillation." arXiv:2002.03936, 2020.** When there are few classes, distillation transfers little information, so they force the teacher to invent subclasses within each class during supervised training, using an auxiliary loss that pushes different examples of the same class into different subclasses. The cross-entropy constrains only the class-level marginal probabilities, leaving subclass logits free. [B, numbers not verified]

Relevance: 7 classes is exactly the "few classes" regime this paper targets, and the mechanism (add subclass logits, marginalise to the class label) is a clean way to get sub-population structure into a *learned head* without needing to change your inference API.

### 5.6 Label errors and ontological overlap

**Northcutt, Jiang, Chuang. "Confident Learning: Estimating Uncertainty in Dataset Labels." JAIR 70, 2021, arXiv:1911.00068.** And **Northcutt, Athalye, Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks," NeurIPS 2021 Datasets & Benchmarks, arXiv:2103.14749.** Average 3.4% label errors across benchmark test sets, 6% in ImageNet val, ~10% estimated in QuickDraw. Crucially for you: confident learning was used to quantify **ontological class overlap** on ImageNet, e.g. estimating that 645 "missile" images are labelled as the parent class "projectile." [B]

That is the closest published analogue to "my class taxonomy has a genuine parent/child or sibling overlap." The method (estimate the joint distribution between noisy and true labels from out-of-sample predicted probabilities) is applicable as a **diagnostic** on your corpus even though your labels are not noisy: run it and inspect which bluetooth samples are confidently predicted gsm, then check whether they are concentrated in one of the two bluetooth sub-profiles. If they concentrate, you have a sub-population problem. If they are uniform across sub-profiles, you have a genuine overlap problem. [C on the application]

### 5.7 Enforced invariance as a *cause* of confusable-class errors

**"Understanding the Detrimental Class-level Effects of Data Augmentation." NeurIPS 2023. arXiv:2401.01764.** Augmentation improves average accuracy while hurting individual class accuracy by **up to 20% on ImageNet**. Using higher-quality multi-label annotations, the authors categorise the harmed classes and find the majority are "inherently ambiguous, co-occur, or involve fine-grained distinctions," and that augmentation "controls the model's bias towards one of the closely related classes." [B]

This is the mechanism by which an invariance you deliberately impose becomes the source of your worst confusion, and it is measured, not speculative. Related theoretical work: Jacobsen et al., "Excessive Invariance Causes Adversarial Vulnerability," ICLR 2019, and Tramèr et al., "Fundamental Tradeoffs between Invariance and Sensitivity to Adversarial Perturbations," arXiv:2002.04599.

---

## 6. ONE PHYSICS FINDING THAT BEARS DIRECTLY ON YOUR TRIAD

Verified against RF references (rfwireless-world GFSK vs GMSK comparison; standard Bluetooth and GSM specs) [B on the sourcing, but these are uncontroversial spec facts]:

- **GMSK is the special case of GFSK with modulation index exactly h = 0.5.** GSM uses BT = 0.3, h = 0.5, symbol rate 270.833 ksym/s, 200 kHz channel.
- **Bluetooth GFSK uses BT = 0.5 and h in the range 0.28 to 0.35** (spec tolerance window), 1 Msym/s, 1 MHz channel.
- 802.11b DSSS: 11 Mchip/s, ~22 MHz occupied bandwidth.

So bluetooth, gsm, and dsss differ by a factor of ~3.7 and ~11 in symbol rate, ~5x and ~22x in occupied bandwidth in Hz, and bluetooth vs gsm differ by roughly 40-60% in modulation index. **These are not subtle distinctions. They are the most discriminative scalar parameters available.**

The extrapolation [C]: your preprocessing estimates occupied bandwidth and then **resamples so that occupied fractional bandwidth hits a canonical 0.5**. That operation is precisely a projection that discards absolute bandwidth in Hz. In the "native" variant you skip the resample but you still have 63 distinct sample rates across the corpus, so absolute symbol rate is not recoverable from the sample sequence alone unless the sample rate is presented to the network as a feature. Your own result that native beat resampled for *every* architecture tried is consistent with this: native leaks slightly more of the scale information back in. Your 2D-STFT result is also consistent: changing the representation changed *which* classes collide (gsm->bluetooth halved, a new cw absorption appeared), which is the signature of a representation-induced confusion rather than an irreducible one.

I could not find a published paper that measures this specific effect on RF signal classification. I am presenting it as reasoning from physics plus your own measurements, not as a citation.

## 7. WHAT THE LITERATURE SAYS DID NOT WORK (negative results collected)

1. Hierarchy-aware losses and top-down cascades **lose top-1** to flat cross-entropy on both tieredImageNet-H and iNaturalist-19 (Bertinetto et al. 2020). They buy mistake *severity*, not mistake *rate*.
2. Angular margin losses (ArcFace, CosFace) are **not better than plain contrastive on CUB200**, the most fine-grained benchmark in the reality-check suite (67.50 / 67.32 vs 68.13 P@1). Their real domain is large-class open-set retrieval.
3. Ten years of claimed metric-learning progress largely evaporates under controlled comparison (Musgrave et al. 2020).
4. Multi-prototype methods buy **nothing** on unimodal classes: IMP equals ProtoNet on standard Omniglot (98.4 vs 98.2) and near-equals on miniImageNet.
5. NCMC's advantage over a discriminative linear SVM **reversed** on higher-dimensional features (29.4 NCMC vs 28.0 SVM at 64K dims), so multi-centroid does not universally dominate.
6. DNC's k-sub-centroid gain **saturates and then reverses**: ADE20K K=10 gives 44.3 mIoU, K=20 gives 44.0. Too many centroids over-fragments.
7. Naively plugging soft labels into Bayes-error estimators yields "unreasonably high" estimates (arXiv:2505.20761), so the ICLR-2023 estimator is not a plug-and-play answer to "is this irreducible."
8. Hard-negative mining outside the minibatch destabilises and collapses embeddings, which is why Hermans et al. restrict to batch-hard.

## Applicability

## HOW WELL THIS ACTUALLY APPLIES TO YOUR PROBLEM

### The strongest applicable result, and why

**Infinite Mixture Prototypes' alphabet experiment (ProtoNet 65.6% to IMP 92.0%) is a structural match to your setup, not a loose analogy.** In that experiment the label is a superclass (an alphabet) whose members (characters) are structurally distinct, and the measured failure of single-prototype ProtoNet is catastrophic rather than marginal. Your "gsm" label pools 7 profiles spanning GMSK normal bursts and 8PSK/QPSK/16QAM/32QAM EDGE bursts. Those are not variations of one thing, they are different modulations. Your "ofdm" pools ~20 LTE/NR/WiFi profiles. Your "bluetooth" pools classic-connected and LE-advertising. You are training a single-prototype classifier on superclass labels, which is the exact configuration where the literature says the method degrades most.

**And you have something every one of these papers had to work around: ground-truth subclass labels.** IMP infers clusters with DP-means. GEORGE clusters the feature space and treats the result as noisy supervision. DNC runs Sinkhorn clustering every batch. PALM does reciprocal-neighbour soft assignment. All of that machinery exists solely to recover sub-population structure that nobody labelled. You have 37 named profiles. The cheapest high-value experiment available to you is to train episodically over the **37 profiles** as the classes (or maintain 37 prototypes and take min-distance-within-coarse-class at inference), and map to the 7 output labels at the last step. Your API and UX stay identical. That is a strictly better-conditioned version of what IMP and GEORGE do approximately.

### The prediction I would make, and how to falsify it fast

Your own diagnostic already splits the triad into two different problems, and they have different prognoses:

- **dsss -> bluetooth (217/573)**: bluetooth-vs-dsss centroid z-distance is 2.956, well separated in hand-feature space. This confusion **cannot** be irreducible overlap. It is decision geometry or the learned path. Multi-prototype should attack this directly, and I would expect a real reduction. High confidence.
- **gsm -> bluetooth (220/614)**: centroid z-distance 0.713, below both classes' within-class spread. This *looks* irreducible, but the within-class spreads (gsm 2.053, bluetooth 1.495) are exactly what a multi-profile mixture produces. F1 and N2 cannot distinguish "overlapping" from "each class is a spread-out mixture." **Recompute the same z-distance per sub-profile pair** (bluetooth-classic vs gsm-normal-burst, bluetooth-le-adv vs gsm-8psk, and so on, 2 x 7 = 14 pairs). If several pairs are well separated while the pooled pair is not, your classes are separable and the pooling is the problem. If all 14 pairs are also below 1.0, it is genuinely the representation. That one computation costs an afternoon and settles the question.

### The cheapest possible test of the multi-prototype hypothesis, requiring zero retraining

Mensink's NCMC-test result is the relevant precedent: with the metric trained for k=1, simply using k>1 centroids at inference recovered most of the gain (39.0 to 36.3 top-5 error at 128d). Translate to your system: keep the current 38k-param network frozen, k-means your enrolled embeddings within each class into k sub-prototypes (k=2 for bluetooth, k=4-7 for gsm, k=4-8 for ofdm, k=1 for am/cw/fm), and classify by minimum distance over sub-prototypes. If the dsss->bluetooth count drops materially, you have confirmed the diagnosis for a few hours of work. Equivalently, run plain k-NN over the enrollment pool at inference; IMP's alphabet result had 1-NN at 92.4 versus prototype at 65.6, so k-NN is a sensitive detector of the multi-modality failure.

### Where this research would NOT help you, stated plainly

1. **Angular margin losses (ArcFace/CosFace/SphereFace) and center loss are the wrong tool at the 7-class level and I would advise against them.** They explicitly enforce intra-class compactness around a single center or a single weight direction. Applied to your heterogeneous labels they optimise for exactly the assumption that is false. I predict they reproduce your intervention #3 (strong repulsion improved leakage to 145/162 but collapsed bluetooth clean accuracy 0.968 to 0.279 and dropped overall to 0.839). You have already run the experiment in a different guise and it failed. If you want angular margin, apply it at the 37-profile level where the unimodality assumption is actually approximately true. Also note the reality-check data: on CUB200, the most fine-grained benchmark, ArcFace was *worse* than plain contrastive.

2. **Coarse-then-fine hierarchical cascade will probably not help as usually implemented.** Bertinetto et al. show every hierarchy-aware method, including the top-down cascade, loses top-1 to flat cross-entropy. Worse, in your taxonomy the natural coarse family would be "bursty constant-envelope digital," which contains bluetooth, gsm, AND dsss. A cascade would route all three into one bucket and hand the identical hard decision to stage 2, plus add an unrecoverable routing error. The only version worth trying is HD-CNN-style, where the fine stage gets something the flat model does not have: extra dedicated capacity, or a different input view (specifically, un-resampled input plus absolute sample rate and estimated symbol rate as explicit features).

3. **Supervised contrastive is not applicable at your batch size.** SupCon's ImageNet result used batch 6144. Your 7-way 5-shot episodes are ~70 samples. Your intervention #4 was effectively small-batch SupCon and moved nothing (215/237), which is consistent with rather than contradicting the literature.

4. **The Bayes error estimator from Ishida et al. (ICLR 2023) is not directly usable.** It requires soft labels from multiple annotators. Your labels are generative and hard. Use the Cover-Hart bound on the binary sub-problem instead (Bayes error >= asymptotic 1-NN error / 2), which needs nothing but a large sample.

5. **Confident learning will not fix anything** because your labels are not noisy, they are generated. It is still worth running as a *diagnostic* to see whether the misclassified bluetooth samples concentrate in one sub-profile.

6. **The RFS logistic-regression-beats-nearest-neighbour result (78.31 vs 69.96 at 5-shot) is weaker evidence than it looks.** Their "NN" is 1-NN, not nearest-centroid, and at 5 shots a centroid usually beats 1-NN. Cross-referencing Meta-Baseline, the true prototype-vs-linear-head gap on matched ResNet-12 embeddings looks small. Do not switch to a linear head expecting 8 points. More importantly, **a linear softmax head has convex decision regions just like nearest-centroid does**, so it does not solve multi-modality either. If you want to replace the head for capacity reasons, the choices that actually change the hypothesis class are multi-prototype or an MLP head, not a linear one.

### The thesis I would put forward, which the research supports and your data does not contradict

Your triad confusion is likely two problems wearing one costume: a **multi-modality problem** (bluetooth and gsm and ofdm are superclasses collapsed to single prototypes) sitting on top of a **representation problem** (your invariance preprocessing normalises away absolute occupied bandwidth and symbol rate, which are the strongest available discriminators between a 200 kHz / 270.8 ksym/s GMSK burst, a 1 MHz / 1 Msym/s GFSK burst, and a 22 MHz / 11 Mchip/s DSSS burst; and GSM's modulation index is exactly 0.5 while Bluetooth's is 0.28-0.35).

Three independent pieces of your own evidence point at the representation half. Native beat resampled for every architecture. The 2D STFT changed which classes collide rather than reducing collisions overall (gsm->bluetooth 236 to 143, but new cw absorption appeared). And nine interventions that all operate on the *loss* or the *sampling* while leaving the input map fixed all landed within noise. The NeurIPS 2023 augmentation paper is the closest published support: enforced invariance costs up to 20 points on individual classes, concentrated specifically on ambiguous and fine-grained pairs.

The multi-modality half is supported by IMP, NCMC, DNC, GEORGE, and PALM, four of which measure the gain and find it scales with how heterogeneous the classes actually are (IMP: +26.4 on superclass labels, +0 on unimodal ones; DNC: +0.5 top-1 on ImageNet, +1.1 mIoU on ADE20K).

Neither half is addressed by anything you have tried, which is a satisfying explanation for why nine interventions all produced the same confusion counts.

### Concrete ranked next steps implied by the research

1. Recompute your robust z-distance diagnostic **per sub-profile pair** rather than per class pair (settles the "irreducible or not" question; costs an afternoon).
2. Frozen-network multi-prototype at inference only, k-means within class (Mensink's NCMC-test protocol; costs hours, no retraining).
3. Train the episodic objective over **37 profile labels**, pool to 7 at output (the ground-truth-subclass version of IMP/GEORGE; this is the highest-expected-value change).
4. Add absolute occupied bandwidth in Hz, estimated symbol rate in Hz, and an estimated modulation index to the hand-feature vector, so the invariance you impose for robustness does not also delete the class signal.
5. Cover-Hart bracket on the bluetooth-vs-gsm binary sub-problem to get a defensible floor on irreducible error, computed once under current preprocessing and once with absolute-scale features restored. The difference between those two numbers is the amount of separability your preprocessing is destroying.

## Confidence

Mixed, and I have tagged claims inline. Breaking it down:

**Read directly from the primary PDF, high confidence in the exact digits:**
- Mensink et al. NCM vs NCMC on ILSVRC'10 (the full Tables 2, 3, 4 including the 64K-feature counterexample where linear SVM still wins).
- DNC (ICLR 2023) ImageNet Table 2, the K-ablation Table 5a, the segmentation Table 3, and the paper's own statement that parametric classifiers "essentially assume unimodality for each class."
- RFS (ECCV 2020) Table 5 base-learner ablation, NN vs LR vs L2 vs Aug vs Distill on four benchmarks, plus the authors' own summary sentence.

**Extracted via ar5iv rendering by a summarisation model, so the qualitative conclusion is solid but individual digits could be off by a decimal:**
- Infinite Mixture Prototypes result tables. The headline (65.6 to 92.0 on alphabets) is corroborated independently by the abstract's "10-25% absolute accuracy improvements" claim, which I saw from two separate sources, so I am confident in the finding and moderately confident in the exact cells.
- Making Better Mistakes top-1 error table. The qualitative conclusion (every hierarchy-aware method loses top-1, and the authors describe an "inherent tension") is stated explicitly by the authors and I regard it as solid.
- Metric Learning Reality Check reproduced P@1 / MAP@R numbers. The finding that gains are marginal is the paper's central and widely replicated thesis.

**Single-paper claims I did not verify numerically:**
- Center loss (Wen et al. 2016) specific results.
- Subclass Distillation (Müller/Kornblith/Hinton) results.
- HD-CNN magnitude of improvement on CIFAR100/ImageNet.
- PHM-Net's claimed +11.14 / +9.43 points on RML2018.01a. AMC papers vary enormously in baseline quality and I would not act on this without reading it.
- ArcFace's MegaFace cell specifically. LFW 99.83 is well established; the MegaFace figure I saw (96.98 VR@FAR 1e-6) I could not confirm against the paper and the protocol matters.
- GEORGE's "+22 percentage points worst-case subclass accuracy." Consistent across two sources but I did not open the paper's tables.

**My own extrapolation, clearly not established fact:**
- The entire argument that your resample-to-canonical-fractional-bandwidth step is destroying the absolute bandwidth and symbol rate cues that separate GSM (200 kHz, 270.833 ksym/s, h=0.5), Bluetooth (1 MHz, 1 Msym/s, h=0.28-0.35), and 802.11b DSSS (22 MHz, 11 Mchip/s). The physics facts are standard spec knowledge and I verified them against RF references. The inference that this explains your triad confusion is mine. I found no published paper measuring this effect on RF classification.
- The convexity argument (that a linear softmax head has convex decision regions exactly as nearest-centroid does, so switching to a linear head does not solve multi-modality). The geometry is elementary and correct; the application to your case is my reasoning.
- The reconciliation of RFS's 8.35-point NN-vs-LR gap against Meta-Baseline's prototype numbers, concluding the true prototype-vs-linear gap is small. This is my cross-paper inference and could be wrong.
- The claim that your dsss->bluetooth leak is fixable while gsm->bluetooth may not be. This follows from your own hand-feature z-distances (2.956 vs 0.713) but the split is my reading.
- The prediction that ArcFace at the 7-class level would reproduce your intervention-#3 collapse.
- The proposed diagnostic sequence (per-sub-profile F1/N2, frozen-network multi-prototype, Cover-Hart bracket). Cover-Hart itself is a theorem from 1967 and is solid; using it this way is my suggestion.

**What I could not find at all:** any published work measuring multi-prototype methods on RF signal classification with heterogeneous protocol families, and any work quantifying how bandwidth-normalisation preprocessing affects inter-class separability in AMC. If those exist I did not surface them, and the absence is itself informative: the specific combination you are working on does not appear to have a direct literature precedent, so the transfer is from vision and face-recognition analogues.

---

# Multi-Representation Fusion

## Findings

## SCOPE NOTE ON METHOD

I ran ~20 web searches and fetched ~15 primary sources. Several arXiv PDFs and one MDPI page would not yield text (binary/403); where that happened I used ar5iv HTML or the PMC mirror and I say so. Everything below is tagged: **[A] well-replicated**, **[B] single-paper / single-source claim**, **[C] my own derivation or extrapolation**. I have flagged three places where the literature simply does not contain what you asked for.

---

# (1) THE TWO-STAGE DETECT-THEN-CLASSIFY PIPELINE

## What actually exists

**WRIST — Nguyen, Vomvas, Vo-Huu, Noubir, "Spectro-Temporal RF Identification using Deep Learning," arXiv:2107.05114 (2021), Northeastern.** [B]
The canonical modern instance. Treats RF identification as *spectrogram object detection*: FFT → time-frequency image → modified YOLOv4 predicts bounding boxes whose coordinates are (center frequency, temporal position). Adds an "RF-centric compression layer." Five classes, and notably **exactly your confusion family**: 802.11 (WiFi), 802.15.1 (Bluetooth), 802.15.4 (ZigBee), DJI Lightbridge, XPD cordless mic. Processes 100 MHz / >6 Gbps I&Q in real time.
Reported: **92.21 mAP @ IoU 0.75** (clean), **88.42 mAP** (anechoic chamber), **78.89 mAP** ("in the wild," congested). The optimized SYL-3/SYL-4 variants give 2.2× speedup over stock YOLO at competitive accuracy.

**Vagollari, Schram, et al., "Joint Detection and Classification of RF Signals Using Deep Learning," IEEE VTC2021-Spring** (IEEE Xplore doc 9449073). [B] Same formulation — wideband spectrogram, YOLO-family detector, detect+localize+classify jointly.

**"Large Scale Radio Frequency Wideband Signal Detection & Recognition," arXiv:2211.10335.** [B] Same family, larger scale, simulated analog+digital modulations.

**Xing, Zhang, Chang, Ren, Zhang, Xu, Cui, "Joint Signal Detection and Automatic Modulation Classification via Deep Learning," arXiv:2405.00736 (2024).** [B] Introduces the CRML23 dataset (multiple coexisting signals at different carrier frequencies) and a joint framework (JDM) coupling a detection module and an AMC module through a shared "proposal" data structure — an explicitly Faster-R-CNN-shaped design. **I could not extract head-to-head joint-vs-two-stage accuracy numbers**; the abstract only claims "effectiveness," and the PDF would not render to text.

**Uvaydov et al., "Stitching the Spectrum: Semantic Spectrum Segmentation with Wideband Signal Stitching," INFOCOM 2024 (arXiv:2402.03465)** and **CV-MuSeNet (arXiv:2506.11048, complex-valued multi-signal segmentation).** [B] These do pixel-level (semantic) segmentation of the spectrum rather than boxes. One reported figure from this line: a non-local-block segmentation network gives **+7% accuracy on the hardest-to-segment signals vs. U-Net**.

**Detection methods used in practice, ranked as the literature ranks them:** [A]
- *Energy detection* — cheapest, standard, fails at low SNR. Universally used as the trigger in real receivers.
- *Cyclostationary detection* (cyclic-feature / "cycle detectors," chi-squared test for presence of cyclostationarity) — substantially more robust than energy detection at low SNR and *signal-selective* (can detect signal A in the presence of signal B), but far costlier.
- *Matched filter / correlation on known sync words* — optimal but requires knowing the standard, so it is a per-protocol detector, not a general one.
- *Deep-learning burst detection* — dominated by the spectrogram-object-detection formulation above.

## THE HONEST NEGATIVE RESULT ON YOUR ACTUAL QUESTION

**I could not find any paper that runs the controlled A/B you are implicitly asking about: same classifier, same data, bursts aligned/segmented vs. randomly-offset fixed windows.** The two-stage papers above adopt detection-first because in a *wideband multi-emitter scene you cannot classify what you have not localized in frequency* — the frequency-domain localization is doing the work, not the time alignment. None of them ablates time alignment. WRIST explicitly "does not directly compare against traditional fixed-window classification approaches."

This matters for you because your windows are already single-signal and already frequency-centered by your preprocessing. **The literature's justification for two-stage does not transfer to your setup.** I flag this as the single most important corrective to the framing in your brief.

---

# (2) PUBLISHED QUANTIFICATION OF MISALIGNMENT COST

**This is a genuine hole in the literature, and I want to be blunt about it rather than manufacture a number.** Three of the most directly relevant papers all *deliberately step around* the question:

**Harper, Thornton, Larson, "Automatic Modulation Classification with Deep Neural Networks," arXiv:2301.11773 (2023), SMU.** [B] This is the closest anyone gets. They explicitly randomize alignment — "a uniformly random starting point is determined for each signal such that a contiguous sample of the desired duration, starting at the random point, is chosen" — and then **do not study it**. What they *do* ablate is *duration*, from 1.024 ms baseline down:
- 512 µs: minimal degradation
- 256 µs: degradation begins to be noticeable
- 128 µs: more pronounced
- 64 µs / 32 µs: significant and further loss
- 16 µs: still significantly above chance (chance = 4.2%, 24 classes)
Exact per-duration accuracies are in figures, not tables; I could not extract numeric values.

**Huang et al., "Data Augmentation for Deep Learning-based Radio Modulation Classification," arXiv:1912.03026 (2019).** [B] Tests rotation, flip, Gaussian noise. **Time shift is not tested at all.** Rotation gives ~8% absolute at SNR −6 to −2 dB and ~2% at SNR ≥ 4 dB; joint rotation+flip with 12.5% of training data matches the un-augmented 100%-data baseline. Gaussian noise: <2% and "trivial."

**Clark IV, Hauser, Headley, Michaels, "Training Data Augmentation for Deep Learning Radio Frequency Systems," arXiv:2010.00178 (2020).** [B] Tests SNR, frequency offset, and sample-rate mismatch — which the authors frame explicitly as **"post-detection imperfections."** Time shift is explicitly out of scope. Quality ordering found: captured > KDE-augmented > synthetic-assumption-augmented.

**Why the hole exists (my read, [C]):** the canonical AMC benchmarks (RadioML 2016/2018, CSPB.ML) are built from *continuously modulated* signals with random start phase. For a continuously-modulated signal, time alignment is genuinely irrelevant — the signal is (cyclo)stationary and any window is as good as any other. Alignment only becomes a variable when the signal is *bursty*, and the field's benchmark datasets essentially do not contain bursty signals. Your corpus does. **You are operating outside the regime the benchmarks cover, which is exactly why you cannot find the number and why the standard recipes are not helping you.**

**Closest quantified analogue, from RF fingerprinting** (different variable — channel/CFO, not time offset, so do not treat this as your number): "Wireless Fingerprinting via Deep Learning: The Impact of Confounding Factors," arXiv:2002.10791 — under changed channel conditions accuracy drops from ~72% → ~46% (I/Q input) and ~82% → ~6% (FFT input). [B] The lesson that transfers is only the general one: *deep RF models latch onto nuisance structure and collapse when it moves.*

---

# (3) ARCHITECTURES THAT LEARN ALIGNMENT END-TO-END

**O'Shea, Pemula, Batra, Clancy, "Radio Transformer Networks: Attention Models for Learning to Synchronize in Wireless Systems," Asilomar 2016 (arXiv:1605.00716).** [B] — **the canonical paper, and the result is weak. Report this as a negative.**
- Architecture: Spatial-Transformer-Network-style localization net emitting **five parameters (θ₀–θ₄)**: a 1-D affine transform (time offset *and* scale/symbol-rate) plus mixing with a complex sinusoid (frequency offset and initial phase). Fully differentiable, trained only by the classification loss.
- Data: RadioML 2016.04C, 11 classes (8 digital, 3 analog), with oscillator drift, clock drift, fading.
- **Result: "slightly increased performance over the model without attention" — characterized as "similar accuracies at slightly lower SNR values (≈1 dB)."** No table of accuracies; Figure 6 only.
- The authors' own caveat is the important part: *"we have no real reason to expect perfect synchronization from a classification task, just enough normalization to make things easier on the discriminative network."* Constellation plots show only *partial* synchronization.
- **Interpretation [C]: ten years on, the flagship "learn alignment end-to-end" paper bought ~1 dB. This is evidence *against* the hypothesis that a learned alignment module will break your triad plateau.**

**X-vector / statistics-pooling architectures (Harper et al. 2020, 2023).** [B] Mean+variance statistics pooling over convolutional filter outputs → fixed-length representation from variable-length input, fully convolutional, no padding or resampling needed. **Note: your production 1-D CNN already does mean+std pooling — you already have the x-vector trick.** Its alignment-invariance is exactly the global-average-pooling kind, which is why the extra capacity is not showing up.

**Attention/transformer pooling.** AMC-Transformer (Research Square rs-4751132; arXiv:2606.09085 MoE variant; ScienceDirect S2590123025008606 on additive attention): tokenize raw I/Q into patches + learnable positional embeddings + multi-head self-attention; scaled additive attention pools by "assigning larger weights to more discriminative time positions." Reported **98.8% at SNR ≥ 10 dB on RadioML2018.01A, +4.44% relative over a CNN and +1.96% relative over a ResNet reimplementation.** [B] Standard benchmark, no bursty signals — **this is the same architecture family that already plateaued at 0.876 for you, and the reported gains are relative-percent on a benchmark where alignment is a non-issue.**

**Learned segmentation.** Semantic spectrum segmentation (Stitching the Spectrum, INFOCOM 2024; CV-MuSeNet) produces per-time-frequency-bin labels rather than a global label. [B] This is the one branch of (3) I would take seriously for you, and only because it addresses the *pooled-class / variable-visible-structure* problem, not because it aligns anything.

---

# (4) CYCLOSTATIONARY ANALYSIS: DOES IT SEPARATE YOUR TRIAD?

## Established structure [A] (Spooner, cyclostationary.blog, corroborated across multiple posts)

**The hard second-order separator — conjugate SCF.**
For **BPSK, MSK, OOK, GMSK** the *conjugate* spectral correlation function is non-zero at some cycle frequencies. For **QPSK, 8PSK, 16QAM** it is **identically zero**. Specifically, for MSK/GMSK there are **two conjugate cycle frequencies, centered at the doubled carrier 2f_c and separated by the symbol rate.** MSK is exactly OQPSK with half-sine pulses, so OQPSK/MSK/GMSK analyses coincide.

**DSSS.** Ordinary PSK/QAM has "only a single non-conjugate cycle frequency and no conjugate cycle frequencies." **DSSS has many non-conjugate cycle frequencies and sometimes many conjugate ones.** The dominant feature is the **chip-rate feature at harmonics of 1/T_chip**, arising from the interaction of the data signal's α=0 component with the periodic chipping waveform. Crucially: *"features at the chip rate dwarf those at the symbol rate, a phenomenon that strengthens with increased processing gain."* Processing gain N_c = T_sym/T_chip.

**GSM specifically.** GSM's spectral correlation is "quite complex relative to its base modulation of GMSK, with the appearance of many cycle frequencies separated by a harmonic of the frame rate," and the frame rate is quoted as **216.7 Hz** — i.e. GSM has GMSK's symbol-rate/doubled-carrier features *plus* a dense comb at multiples of 1/4.615 ms from the TDMA framing.

**FSK taxonomy (Spooner, "Cyclostationarity of Frequency-Shift-Keyed Signals," 2023).** [B] Three regimes with *markedly different* cyclostationarity:
- *Incoherent FSK*: symbol-rate harmonics only for n=2m; no conjugate cyclostationarity (random carrier phases average away).
- *Carrier-phase-coherent FSK*: many CFs, β = Σ(n_j − 2m_j)F_j ± k/T₀, strong conjugate components.
- *Clock-phase-coherent FSK*: "cycle frequencies similar to those for BPSK, except odd-order cyclic cumulants can be nonzero."
GFSK/GMSK are CPM and sit in the coherent regimes.

**Two robustness facts that matter enormously for your 75%-impaired corpus** [A]:
- **Multipath distorts the spectrum and the SCF *shape* but does NOT change the cycle-frequency *locations*.**
- Non-conjugate cycle frequencies live at k/T₀ (carrier-independent) and are therefore **immune to CFO**. Conjugate CFs live at 2f_c ± … and *do* move with CFO (by 2δ for offset δ).

**A documented limit [A]:** *"Some distinct signal types give rise to identical spectral correlation functions"* — MSK/GMSK/OQPSK/SQPSK collide at second order. Spooner's resolution: *"higher-order statistics for these signals are, fortunately, not identical."* **So second-order SCF alone is insufficient for your triad; you need higher-order cyclic cumulants.**

## The Snoap / Latshaw / Popescu / Spooner result — the most important thing I found

**"Deep-Learning-Based Classification of Digitally Modulated Signals Using Capsule Networks and Cyclic Cumulants," Sensors 2023, 23(12):5735** (doi 10.3390/s23125735; also arXiv:2211.00232 as "Robust Classification…"). Numbers below are from the PMC mirror of the Sensors paper (the MDPI page 403'd me). [B, but methodologically airtight and consistent across two papers]

- **Features:** cyclic cumulants of orders **n ∈ {2, 4, 6}**, at cycle frequencies **α = (n − 2m)·f₀ ± k/T₀, k = 0…5** → 11 CFs × 15 (n,m) pairs = **165 CC estimates per signal example**, blindly estimated (symbol rate and CFO via SSCA).
- **Classes (8):** BPSK, QPSK, 8PSK, π/4-DQPSK, **MSK**, 16QAM, 64QAM, 256QAM. Chance = 12.5%.
- **Datasets:** CSPB.ML.2018 and CSPB.ML.2022, 112,000 signals each, **32,768 samples per instance**. The datasets differ *only* in the carrier-frequency-offset distribution (max CFO in 2018 = 0.001; min CFO in 2022 = 0.01, an order of magnitude larger) and slightly in SNR range (0–12 dB vs 1–18 dB). Symbol period and roll-off overlap.
- **Results (Table 3):**

| Model | train 2018 → test 2018 | train 2018 → test 2022 | train 2022 → test 2022 | train 2022 → test 2018 |
|---|---|---|---|---|
| **I/Q-trained capsule net** | **97.5%** | **23.7%** | **97.7%** | **25.7%** |
| **CC-trained capsule net** | 92.3% | **91.6%** | 92.5% | **93.1%** |
| Conventional CSP (no ML) | 82.0% | 82.0% | 82.0% | 82.0% |

**Read that again: a raw-I/Q network at 97.5% falls to 23.7% — barely above 12.5% chance — under a shift in one nuisance parameter's distribution. The cyclic-cumulant front end loses ~0.7 points.** Related arXiv:2211.00232 numbers for the CNN variants: I/Q-CNN 2018→2022 = 41.6%, 2022→2018 = 45.3%; CC-CNN 91.4% / 92.4%.
Per-class, the conventional CSP baseline was worst on high-order QAM (41.7% 64QAM, 55.9% 256QAM, 72.5% 16QAM), lifted by the CC net to ~62.3% / ~74% / ~97.5%.

## Does it separate YOUR three? My assessment

**DSSS vs. {Bluetooth, GSM}: YES, decisively. [A-grade mechanism]**
802.11b HR/DSSS: chip rate **11 Mchip/s**, occupied bandwidth **~22 MHz**, Barker-11 spreading for 1/2 Mbps and CCK-8 for 5.5/11 Mbps. The chip-rate non-conjugate CF sits at **11 MHz = BW/2**, i.e. at normalized cycle frequency **0.5 × occupied bandwidth** — and *that ratio survives your bandwidth normalization*. For Barker modes you additionally get a symbol-rate CF at 1 MHz with an **11:1 chip:symbol CF ratio**, a smoking gun. For CCK-11M the code is data-selected from 64 complementary codes, so the *code-repetition* feature washes out but the 11 MHz chip-clock feature remains. No CPM signal has anything at 0.5×BW.
*Caveat [A]:* long-code DSSS defeats this — "when the PN code length is bigger than the observation window duration… the cyclostationary properties are destroyed." **Not your problem**: 802.11b codes are 8–11 chips.

**GSM-GMSK vs. Bluetooth-Classic-GFSK: YES, via modulation index. [C — my derivation, standard CPM theory, one step beyond what I could cite]**
Verified anchor [A]: for GMSK (h=0.5) the two conjugate CFs are centered at 2f_c and **separated by the symbol rate R_s**.
Derivation [C]: binary CPM places tones at f_c ± hR_s/2; the conjugate (quadratic) nonlinearity puts conjugate CFs at 2f_c ± hR_s, so **conjugate-CF spacing = 2hR_s**. Setting h=0.5 recovers the verified spacing R_s — consistent. The non-conjugate CF is at R_s. Therefore:

**ratio = (conjugate-CF spacing)/(non-conjugate symbol-rate CF) = 2h**

- GSM GMSK, h = 0.5 → **2h = 1.00**
- Bluetooth Classic BR, h ∈ [0.28, 0.35], nominal 0.32 → **2h = 0.56–0.70**

This ratio is **dimensionless: invariant to sample rate, to your resample-to-0.5-fractional-bandwidth step, and to absolute symbol rate.** It is exactly the quantity your current feature set cannot see.

**GSM-GMSK vs. Bluetooth-LE-GFSK: NO, not from modulation index. [A on the parameters, C on the consequence]**
**BLE (LE 1M PHY) uses h = 0.5** (spec range 0.45–0.55) — *identical to GSM's GMSK*. Both are BT-filtered binary CPM at h=0.5. They differ only in:
- **BT product**: GSM **0.3**, BLE **0.5**. Lower BT = narrower Gaussian = *narrower* spectrum relative to R_s, so **R_s / BW_99% is larger for GSM than for BLE**. Real but modest, and it is a second-order-shape difference, not a CF-location difference.
- **Absolute symbol rate**: 270.833 ksym/s vs 1 Msym/s — **destroyed by your bandwidth normalization.**
- **Burst/frame structure** — see §5.

**This is, I believe, the mechanistic root of your measured bluetooth↔gsm centroid z-distance of 0.713.** Your "bluetooth" class pools h=0.32 (Classic) with h=0.5 (LE); the h=0.5 half of it is *physically almost the same modulation as GSM*, differing mainly in a Gaussian filter BT product and an absolute rate you deliberately normalize away. No amount of prototype repulsion can fix a class whose second mode genuinely coincides with another class's mode — which is precisely why your λ=8.0 repulsion experiment traded bluetooth accuracy 0.968 → 0.279 rather than fixing anything.

---

# (5) PHYSICAL/STATISTICAL DISCRIMINANTS AND WHAT SURVIVES A SHORT UNALIGNED WINDOW

## Verified parameter table [A]

| | **Bluetooth Classic BR** | **Bluetooth LE (1M)** | **GSM** | **802.11b HR/DSSS** |
|---|---|---|---|---|
| Modulation | GFSK (CPM) | GFSK (CPM) | GMSK (CPM) | DSSS-BPSK/QPSK, CCK |
| Symbol/chip rate | 1 Msym/s | 1 Msym/s | **270.833 ksym/s** | **11 Mchip/s** |
| Mod. index h | **0.28–0.35** (nom. 0.32) | **0.5** (0.45–0.55) | **0.5** | n/a |
| Gaussian BT | 0.5 | 0.5 | **0.3** | n/a |
| Channel / occ. BW | 1 MHz | 2 MHz | 200 kHz | **~22 MHz** |
| Spreading | none | none | none | Barker-11 (1/2M), CCK-8 (5.5/11M) |
| Hop / channel plan | **1600 hops/s**, 79 ch | 40 ch; 3 adv ch (37/38/39) | 8-slot TDMA | fixed channel |
| Burst duration | slot 625 µs; DH1 ≤366 µs | adv pkt ~80–376 µs | **normal burst 576.9 µs** (156.25 bit periods incl. 8.25-bit guard ≈ 30.46 µs) | preamble+hdr 192 µs (long) / 96 µs (short); frame up to ms |
| Burst repetition | 625 µs slot | irregular, ms–s | **TDMA frame 4.615 ms → 216.7 Hz per-timeslot rate**, duty 1/8 | CSMA, aperiodic |

## Recoverability from YOUR 4096-sample window — the analysis that matters [C, arithmetic on verified parameters]

Your window duration is **4096/f_s**, and f_s spans 1–30 MHz over 63 values. Nyquist forces f_s ≳ 2×BW, so:

| Class | plausible f_s | **window duration** | symbols/chips in window | burst edges visible? |
|---|---|---|---|---|
| GSM (200 kHz) | 1–2 MHz | **2.0–4.1 ms** | ~550–1100 symbols | **Yes, several bursts + guards** |
| BT Classic (1 MHz) | 2–4 MHz | **1.0–2.0 ms** | 1000–2000 symbols | Yes, 1.6–3.3 slot periods |
| BT LE (2 MHz) | 4–8 MHz | **0.5–1.0 ms** | 500–1000 symbols | Usually yes (packet + gap) |
| 802.11b (22 MHz) | **≥ 24–30 MHz** | **137–171 µs** | ~1500–1900 chips | **Usually NO — window sits inside the packet** |

**Four consequences you should act on:**

1. **Symbol-rate and chip-rate cyclic features ARE recoverable at 4096 samples for every class.** You have 500–2000 symbol/chip periods in every window. Cycle-frequency resolution ≈ 1/(4096/f_s), which is ~0.02% of R_s for GSM and ~0.07% of the chip rate for DSSS. **This is the good news and it is the main actionable finding.**

2. **Hop rate (1600 Hz) and GSM frame rate (216.7 Hz) are NOT recoverable.** Their periods are 625 µs and 4.615 ms; you'd need tens of ms. **Do not build features on hop rate or frame rate** — it is arithmetically impossible at your window length except marginally for GSM at f_s = 1 MHz. This kills one of the "structural observations not yet acted on" in your brief.

3. **Burst-envelope structure is available for GSM and Bluetooth but essentially absent for DSSS** — a structural asymmetry created by your *fixed sample count* combined with DSSS's 22 MHz bandwidth. So there is no burst-structure feature the model can apply uniformly across the triad. Worse: *within* GSM and Bluetooth, how much burst structure is visible varies with f_s across your 63 sample rates. **That is a mechanism that directly inflates within-class variance — which is what you measured** (gsm within-class median spread 2.053 and bluetooth 1.495, both *larger* than the 0.713 between-centroid distance). Your own bug report is corroborating evidence: GSM's 6th-order cumulants had norms up to 1.15M "from near-silent burst-edge captures" — GSM windows land on burst edges; DSSS windows do not.

4. **Your 12 hand-features are all α = 0 statistics.** c20/c40/c41/c42/c60/c63 computed as plain time averages are the **zero-cycle-frequency slice** of the cyclic cumulants. For constant-envelope CPM this slice is close to degenerate: GFSK and GMSK are both constant-modulus binary CPM with similar pulse shaping, so their α=0 cumulants nearly coincide. **The information that separates them (h, and R_s relative to bandwidth) lives at α ≠ 0 and your feature extractor integrates it away.** [C, but strongly supported: Snoap et al. get 92% from α≠0 CCs on classes where the conventional α=0-ish baseline gets 82%, and the classes they separate include MSK against QAM.]

## Concrete discriminant ranking, most to least robust for your setup

| Feature | Separates | Survives your preprocessing? | Estimable @4096? |
|---|---|---|---|
| Non-conjugate CF at chip rate, position **relative to occupied BW** (≈0.5·BW) | **DSSS vs all** | Yes (ratio is scale-free) | **Yes, strongly** |
| Chip-rate : symbol-rate CF ratio (=11 for Barker) | DSSS sub-modes; DSSS vs all | Yes | Yes for Barker modes; weak for CCK |
| **Conjugate-CF spacing ÷ non-conj. symbol-rate CF = 2h** | **BT-Classic (0.56–0.70) vs GSM/BLE (1.00)** | Yes (dimensionless) | Yes, but needs residual CFO ≪ h·R_s |
| Presence of *any* non-zero conjugate SCF | CPM/BPSK family vs QPSK/QAM/OFDM | Yes | Yes |
| R_s ÷ BW_99% (encodes Gaussian BT product) | GSM (BT=0.3) vs BLE (BT=0.5) | Yes | Yes, weakly |
| Burst duty cycle / envelope on-off ratio | GSM & BT vs DSSS | Yes | **Only at low f_s — inconsistent** |
| Hop rate, TDMA frame rate | Would separate all three | Yes | **NO — window far too short** |
| Absolute symbol rate | All three | **NO — destroyed by resampling** | n/a |
| α=0 higher-order cumulants (your current features) | Weak within CPM family | Yes | Yes — but nearly degenerate here |

## Additional negative results worth having

- **RTN's ~1 dB is the state of the art for learned alignment.** Do not expect a step change from an STN-style module.
- **Second-order SCF alone collides MSK/GMSK/OQPSK/SQPSK** — you need order ≥4 cyclic cumulants, not just the SCF.
- **Magnitude-only spectrogram representations discard the instantaneous-phase structure that the CPM classes live in.** Your own 2-D ViT result (gsm→bluetooth 236→143, but a new bluetooth→cw 59, am→cw 46, fm→cw 30 absorption) is a clean instance of this: log-magnitude STFT helps the burst-structure classes and destroys the constant-envelope/tone distinction.
- **Energy detection specifically fails at low SNR**; if you do build a burst segmenter, an energy detector will be the weakest link at the impaired end of your corpus.
- The augmentation literature (Huang 2019; Clark 2020) has **never** tested time-shift augmentation, so there is no published prior on whether it helps.

## One high-value check before you build anything

**Verify that Atom-SignalLab actually renders h = 0.32 for `bluetooth-classic-connected`, h = 0.5 for `bluetooth-le-advertising`, BT = 0.3 for `gsm-normal-burst` and BT = 0.5 for the Bluetooth profiles.** If the simulator uses one shared GFSK/GMSK modulator with a single modulation index and BT product, then *the discriminative information is not in your data at all*, and no front end — cyclostationary, neural, or otherwise — can recover it. That would fully explain a plateau that survived nine independent interventions, 750,000 episodes, and a corpus regeneration. This is cheap to check and it dominates every other next step.

---

## SOURCES

- [WRIST / Spectro-Temporal RF Identification (arXiv:2107.05114)](https://ar5iv.labs.arxiv.org/html/2107.05114)
- [Joint Detection and Classification of RF Signals Using Deep Learning (VTC2021)](https://ieeexplore.ieee.org/document/9449073/)
- [Joint Signal Detection and AMC via Deep Learning (arXiv:2405.00736)](https://arxiv.org/abs/2405.00736)
- [Large Scale RF Wideband Signal Detection & Recognition (arXiv:2211.10335)](https://ar5iv.labs.arxiv.org/html/2211.10335)
- [Stitching the Spectrum, INFOCOM 2024 (arXiv:2402.03465)](https://arxiv.org/abs/2402.03465)
- [Radio Transformer Networks (arXiv:1605.00716)](https://ar5iv.labs.arxiv.org/html/1605.00716)
- [AMC with Deep Neural Networks, Harper et al. (arXiv:2301.11773)](https://ar5iv.labs.arxiv.org/html/2301.11773)
- [Data Augmentation for DL Radio Modulation Classification (arXiv:1912.03026)](https://ar5iv.labs.arxiv.org/html/1912.03026)
- [Training Data Augmentation for DL RF Systems (arXiv:2010.00178)](https://ar5iv.labs.arxiv.org/html/2010.00178)
- [Capsule Networks and Cyclic Cumulants, Sensors 2023 23(12):5735 (PMC mirror)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10302682/)
- [Robust Classification Using Capsule Networks and Cyclic Cumulant Features (arXiv:2211.00232)](https://arxiv.org/html/2211.00232)
- [Cyclostationarity of DSSS Signals (Spooner)](https://cyclostationary.blog/2017/03/28/cyclostationarity-of-direct-sequence-spread-spectrum-signals/)
- [A Gallery of Spectral Correlation (Spooner)](https://cyclostationary.blog/2016/01/28/a-gallery-of-spectral-correlation/)
- [Cyclostationarity of Frequency-Shift-Keyed Signals (Spooner, 2023)](https://cyclostationary.blog/2023/04/25/cyclostationarity-of-frequency-shift-keyed-signals/)
- [Shifted Dataset CSPB.ML.2022 / generalization challenge](https://cyclostationary.blog/2022/01/23/shifted-dataset-for-the-machine-learning-challenge-how-well-does-a-modulation-recognition-dnn-generalize-100th-csp-blog-post/)
- [Infinite Mixture Prototypes for Few-Shot Learning (arXiv:1902.04552)](https://arxiv.org/abs/1902.04552)
- [Deep Learning for Interference Identification (arXiv:1905.08054)](https://ar5iv.labs.arxiv.org/html/1905.08054)
- [Wireless Fingerprinting via Deep Learning: Impact of Confounding Factors (arXiv:2002.10791)](https://arxiv.org/pdf/2002.10791)
- [GSM 05.01/05.02 (ETSI), burst and frame timing](https://www.etsi.org/deliver/etsi_gts/05/0502/05.00.00_60/gsmts_0502v050000p.pdf)
- [GFSK vs GMSK modulation index comparison](https://www.rfwireless-world.com/terminology/gfsk-vs-gmsk)
- [Silicon Labs UG103.14: Bluetooth LE Fundamentals (h=0.5)](https://www.silabs.com/documents/public/user-guides/ug103-14-fundamentals-ble.pdf)
- [Rohde & Schwarz, Generating Signals for WLANs Part I: IEEE 802.11b](https://scdn.rohde-schwarz.com/ur/pws/dl_downloads/dl_application/application_notes/1gp49/1GP49_1E.pdf)

## Applicability

## WHERE THIS APPLIES DIRECTLY

**1. The single highest-leverage finding is NOT the two-stage pipeline — it is that your 12 hand-features are the α=0 slice of the cyclic cumulants, and the information separating GFSK from GMSK lives at α≠0.** This is a mechanistic explanation for your measured bluetooth-vs-gsm centroid z-distance of 0.713 being smaller than either class's within-class spread. Adding ~10–20 cyclic features (chip/symbol-rate CF position normalized to occupied bandwidth; conjugate-CF spacing ÷ non-conjugate CF ≈ 2h; conjugate-SCF presence/absence) to your existing 12-feature concat branch is a small, cheap, contained change that fits your KISS preference and does not touch the architecture. It is testable in a day.

**2. Cyclic-feature magnitudes are inherently time-shift invariant** (from R_x^α(τ) = ∫x(t)x*(t−τ)e^{−j2παt}dt, delaying by t₀ multiplies by e^{−j2παt₀}, so |R| is unchanged). This means the cyclostationary route *dissolves* your burst-alignment problem rather than requiring you to solve it first. That is a strictly better option than building a burst segmenter.

**3. The Snoap/Spooner generalization result is a direct warning about your 0.891.** Their I/Q capsule net hit 97.5% in-distribution and 23.7% under a shift in *one* nuisance parameter's distribution. Your corpus is synthetic from a single generator with a fixed parameter distribution. Your 0.891 may be substantially optimistic in exactly this way, and your 750k-episode plateau at 0.876 for the ViT is consistent with a model that has saturated the *nuisance* structure available rather than the *physical* structure.

**4. The pooled-class problem is confirmed as physically real, not a labeling nit.** BLE uses h=0.5, identical to GSM's GMSK. Your "bluetooth" class contains a mode that is genuinely almost the same modulation as GSM. Infinite Mixture Prototypes (Allen, Shelhamer, Shin, Tenenbaum, ICML 2019, arXiv:1902.04552) reports **25% absolute** improvement over standard prototypical networks precisely on "alphabet" tasks where a class pools sub-types — the closest published analogue to your situation. Splitting to per-profile prototypes (37 sub-prototypes, aggregate at inference by min-distance-over-subprototypes) is the mechanically correct fix and is a small change to your episodic protocol.

## WHERE THIS WOULD NOT HELP YOU

**A burst detector / segmenter is probably the wrong investment.** The two-stage literature exists for *wideband multi-emitter scenes* where you cannot classify what you have not localized in frequency. Your windows are already single-signal and already frequency-centered by preprocessing. No paper I found runs the aligned-vs-unaligned A/B, and WRIST explicitly does not compare against fixed-window classification. Segmentation would mainly reduce within-class variance from the varying-visible-burst-structure effect; the cyclic route gets that for free via shift invariance.

**Learned end-to-end alignment (RTN / spatial transformer) is a documented weak result** — the canonical paper bought roughly 1 dB equivalent SNR and its own authors declined to claim it achieves synchronization. Given nine interventions have already landed within noise, a 1 dB-equivalent mechanism will not move a 35–39% leakage rate.

**Attention/transformer pooling is already exhausted for you.** The AMC-Transformer results (98.8% at high SNR on RadioML2018.01A) come from a benchmark of continuously-modulated signals where alignment is a non-issue. Your 1-D ViT at 0.876 across 750k episodes is the same architecture family against a harder problem; the published gains will not transfer.

**Hop-rate and TDMA-frame-rate features are arithmetically unavailable.** 1600 hops/s and 216.7 Hz need tens of milliseconds; your window is 137 µs to 4.1 ms. This closes off one of the "structural observations not yet acted on" in your brief. If you want those features you must lengthen the window, and for DSSS at 30 MHz that means ~10⁵–10⁶ samples.

**Modulation index will not separate GSM from Bluetooth LE.** Both are h=0.5 binary CPM. The only physical differences that survive your bandwidth normalization are the Gaussian BT product (0.3 vs 0.5, a second-order shape effect) and burst structure (unreliable, f_s-dependent). Expect residual gsm↔BLE confusion even with a perfect cyclostationary front end. h=0.32 vs h=0.5 will cleanly separate Bluetooth *Classic* from both.

**Chip/symbol-rate CF estimation at 4096 samples is 8× less resolved than Snoap et al.'s 32,768-sample instances.** Their 92% is not a number you should expect to inherit. Low-SNR impaired samples will give noisy CC estimates, and your corpus is 75% impaired.

## THE PREREQUISITE THAT DOMINATES EVERYTHING

Before building any cyclostationary front end, verify Atom-SignalLab actually renders distinct modulation indices (0.32 vs 0.5) and distinct Gaussian BT products (0.3 vs 0.5) across the bluetooth-classic / bluetooth-le / gsm profiles. If it uses one shared CPM modulator with fixed h and BT, the discriminative information is absent from the corpus and no front end can recover it — which would explain a plateau that survived nine interventions, three corpus scales, a generator upgrade, and 750,000 episodes better than any modeling hypothesis does. This check is cheap and it gates everything else.

## Confidence

**Mixed, and deliberately stratified. Read the [A]/[B]/[C] tags in the findings — they are not decoration.**

**Well-supported, multiple independent sources ([A]):**
- The cyclostationary structure facts: DSSS has many non-conjugate CFs dominated by the chip-rate feature that dwarfs the symbol-rate feature and strengthens with processing gain; MSK/GMSK/BPSK/OOK have non-zero conjugate SCF while QPSK/8PSK/16QAM have it identically zero; GMSK's two conjugate CFs are centered at 2f_c and separated by the symbol rate; multipath changes SCF shape but not cycle-frequency locations; long-code DSSS defeats cyclostationarity while short-code does not. Corroborated across several Spooner posts plus secondary sources.
- All standard parameters in the §5 table (h, BT product, symbol rates, burst durations, GSM 156.25-bit-period / 4.615 ms framing, 802.11b 11 Mchip/s / 22 MHz / Barker-11 / CCK-8, BLE h=0.5 vs Classic 0.28–0.35). Verified against ETSI GSM 05.02, Silicon Labs UG103.14, Rohde & Schwarz 1GP49, and multiple secondary sources.
- The general result that energy detection degrades badly at low SNR relative to cyclostationary detection.

**Single-source / single-paper ([B]) — treat as indicative, not settled:**
- Every specific accuracy number I quote. In particular the Snoap et al. Table 3 numbers (97.5 → 23.7 vs 92.3 → 91.6) come from one research group. They are internally consistent across two papers (Sensors 2023 and arXiv:2211.00232, whose CNN variants show 41.6% / 45.3% cross-dataset), which raises my confidence, but this is one lab with a strong prior position against raw-I/Q deep learning and it has not been independently replicated as far as I could find.
- WRIST's mAP figures, the AMC-Transformer 98.8%, IMP's 25%-absolute alphabet gain, the +7% segmentation figure.
- Harper et al.'s duration ablation is qualitative in my extraction — I could not get numeric per-duration accuracies, only the shape of the curve.

**My own derivation / extrapolation ([C]) — the parts you should push back on:**
- **The 2h ratio.** I verified the h=0.5 special case (conjugate CFs at 2f_c separated by R_s) from a primary source and extended it to general binary CPM as conjugate-CF spacing = 2hR_s. The extension is standard CPM theory and self-consistent at h=0.5, but I did not find a source stating it in that form. **Test it numerically on your own corpus before relying on it** — it is a ten-line experiment: take a bluetooth-classic sample and a gsm sample, compute the conjugate cyclic autocorrelation, and check whether the conjugate-CF spacing ÷ symbol-rate CF lands near 0.64 and 1.00 respectively.
- The entire §5 "recoverability at 4096 samples" table. It is arithmetic on verified parameters plus assumptions about which sample rates your generator pairs with which bandwidths. **You can replace my assumed f_s values with the actual per-profile values from your corpus and redo it in minutes** — do that rather than trusting my column.
- The claim that your c20…c63 features are effectively the α=0 cyclic-cumulant slice, and that this is why bluetooth and gsm collapse to z-distance 0.713. Mechanically sound and consistent with your diagnostics, but it is my inference connecting your measurements to the literature, not a published finding about your setup.
- The BT-product → R_s/BW_99% direction (GSM BT=0.3 gives larger R_s/BW than BLE BT=0.5). The direction follows from Gaussian filter behavior; the magnitude is unverified and may be too small to be useful.

**Explicit gaps I could not close:**
- **Item (2) has essentially no literature.** I searched it six different ways. No paper runs the aligned-vs-misaligned ablation. I am reporting that as a finding rather than filling it with a plausible number, and I would rather you know the number does not exist than be handed a fabricated one.
- I could not extract joint-vs-two-stage comparative numbers from Xing et al. 2024 (arXiv:2405.00736); several arXiv PDFs returned binary that would not convert to text, and the MDPI page for Sensors 23(12):5735 returned HTTP 403 (I used the PMC mirror instead, which may differ in table formatting).
- I did not find any paper classifying exactly your seven classes on exactly your corpus type, so all applicability reasoning is transfer, not direct evidence.

**Overall:** the physics and cyclostationary structure in §4–§5 I would stake a lot on. The specific accuracy numbers I would treat as order-of-magnitude indicators. The 2h discriminant is my best single actionable hypothesis and it is also the least externally verified thing here — verify it empirically before building on it, and verify the generator actually varies h at all before doing even that.

---

