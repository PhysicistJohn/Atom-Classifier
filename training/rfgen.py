"""Synthetic complex-I/Q modulator bank + impairment channel.

This is the training-data engine. It produces labelled complex baseband for a
bank of modulation families, then passes each realisation through a randomised
impairment channel that models the real-world nuisance variation a deployed
receiver sees: additive noise, carrier/phase/timing offset, IQ imbalance, DC
offset, phase noise, multipath fading, and power/bandwidth variation.

Two independent draws of the same modulation are, by construction, a positive
pair for the contrastive/prototypical objective; two draws of different
modulations are negatives. So the impairment model is simultaneously our realism
model and our augmentation policy.

Everything is seeded through an explicit numpy Generator so a given seed
reproduces a given corpus bit-for-bit (the repo cares about reproducibility).
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi

# Modulation families. `KNOWN` are trained on and get enrolled prototypes.
# `FEWSHOT` are held out of training entirely and enrolled from K shots at test
# time. `NOVEL` are never enrolled and must read as open-set "unknown".
KNOWN = ["cw", "am", "fm", "bpsk", "qpsk", "qam16", "qam64", "gfsk", "ofdm"]
FEWSHOT = ["psk8", "dsss"]
NOVEL = ["noise", "chirp"]
ALL_CLASSES = KNOWN + FEWSHOT + NOVEL


# --------------------------------------------------------------------------
# Pulse shaping
# --------------------------------------------------------------------------
def rrc_taps(sps: int, span: int, beta: float) -> np.ndarray:
    """Root-raised-cosine filter taps, unit-energy normalised."""
    n = np.arange(-span * sps, span * sps + 1, dtype=np.float64)
    t = n / sps
    beta = max(beta, 1e-6)
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-8:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(
                np.pi * ti * (1.0 + beta)
            )
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def _pulse_shape(symbols: np.ndarray, sps: int, rng: np.random.Generator) -> np.ndarray:
    """Upsample complex symbols by `sps` and root-raised-cosine filter."""
    beta = float(rng.uniform(0.2, 0.4))
    taps = rrc_taps(sps, span=6, beta=beta)
    up = np.zeros(len(symbols) * sps, dtype=np.complex128)
    up[::sps] = symbols
    return np.convolve(up, taps, mode="same")


# --------------------------------------------------------------------------
# Symbol constellations
# --------------------------------------------------------------------------
def _psk(order: int, n: int, rng: np.random.Generator) -> np.ndarray:
    k = rng.integers(0, order, size=n)
    return np.exp(1j * TWO_PI * k / order)


def _qam(side: int, n: int, rng: np.random.Generator) -> np.ndarray:
    levels = np.arange(-(side - 1), side, 2, dtype=np.float64)
    i = rng.choice(levels, size=n)
    q = rng.choice(levels, size=n)
    c = i + 1j * q
    return c / np.sqrt(np.mean(np.abs(c) ** 2) + 1e-12)


def _lowpass_message(n: int, rng: np.random.Generator, bw: float = 0.05) -> np.ndarray:
    """Band-limited real message in [-1, 1] for analog modulations."""
    x = rng.standard_normal(n)
    # simple one-pole low-pass to get an audio-like envelope
    a = np.exp(-TWO_PI * bw)
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc = a * acc + (1 - a) * x[i]
        y[i] = acc
    y -= y.mean()
    m = np.max(np.abs(y)) + 1e-9
    return y / m


# --------------------------------------------------------------------------
# Modulators — each returns complex baseband of exactly `length` samples
# --------------------------------------------------------------------------
def _make_symbols_len(length: int, sps: int) -> int:
    return length // sps + 12


def generate_clean(cls: str, length: int, sps: int, rng: np.random.Generator) -> np.ndarray:
    nsym = _make_symbols_len(length, sps)
    if cls == "cw":
        # bare carrier; a residual tone offset makes it a spectral line
        f0 = rng.uniform(-0.02, 0.02)
        sig = np.exp(1j * TWO_PI * f0 * np.arange(length))
    elif cls == "am":
        msg = _lowpass_message(length, rng, bw=rng.uniform(0.02, 0.08))
        depth = rng.uniform(0.4, 0.9)
        sig = (1.0 + depth * msg).astype(np.complex128)
    elif cls == "fm":
        msg = _lowpass_message(length, rng, bw=rng.uniform(0.02, 0.08))
        kf = rng.uniform(0.05, 0.2)
        phase = TWO_PI * kf * np.cumsum(msg)
        sig = np.exp(1j * phase)
    elif cls == "bpsk":
        sig = _pulse_shape((_psk(2, nsym, rng)).astype(np.complex128), sps, rng)
    elif cls == "qpsk":
        sig = _pulse_shape(_psk(4, nsym, rng), sps, rng)
    elif cls == "psk8":
        sig = _pulse_shape(_psk(8, nsym, rng), sps, rng)
    elif cls == "qam16":
        sig = _pulse_shape(_qam(4, nsym, rng), sps, rng)
    elif cls == "qam64":
        sig = _pulse_shape(_qam(8, nsym, rng), sps, rng)
    elif cls == "gfsk":
        h = rng.uniform(0.3, 0.7)  # modulation index
        data = rng.integers(0, 2, size=nsym) * 2 - 1
        # Gaussian-filtered symbol stream -> continuous phase
        up = np.zeros(nsym * sps)
        up[::sps] = data
        gt = np.arange(-2 * sps, 2 * sps + 1) / sps
        gauss = np.exp(-(gt ** 2) / (2 * 0.35 ** 2))
        gauss /= gauss.sum()
        freq = np.convolve(up, gauss, mode="same")
        phase = np.pi * h * np.cumsum(freq) / sps
        sig = np.exp(1j * phase)
    elif cls == "ofdm":
        nfft = rng.choice([64, 128])
        cp = nfft // 4
        active = int(nfft * rng.uniform(0.5, 0.8))
        nsyms = length // (nfft + cp) + 2
        out = []
        for _ in range(nsyms):
            freq = np.zeros(nfft, dtype=np.complex128)
            idx = np.arange(1, active + 1)
            freq[idx] = _qam(4, active, rng)
            t = np.fft.ifft(freq)
            out.append(np.concatenate([t[-cp:], t]))
        sig = np.concatenate(out)
    elif cls == "dsss":
        sf = int(rng.choice([8, 16]))  # spreading factor
        ndata = nsym // sf + 2
        data = rng.integers(0, 2, size=ndata) * 2 - 1
        pn = rng.integers(0, 2, size=sf) * 2 - 1
        chips = np.repeat(data, sf) * np.tile(pn, ndata)
        sig = _pulse_shape(chips.astype(np.complex128), max(2, sps // 2), rng)
    elif cls == "noise":
        sig = (rng.standard_normal(length) + 1j * rng.standard_normal(length)) / np.sqrt(2)
    elif cls == "chirp":
        rate = rng.uniform(0.1, 0.4) * rng.choice([-1, 1])
        t = np.arange(length) / length
        sig = np.exp(1j * np.pi * rate * (t * length) * t)
    else:
        raise ValueError(f"unknown class {cls}")

    sig = sig[:length]
    if len(sig) < length:
        sig = np.pad(sig, (0, length - len(sig)))
    # unit average power before the channel
    p = np.sqrt(np.mean(np.abs(sig) ** 2) + 1e-12)
    return (sig / p).astype(np.complex128)


# --------------------------------------------------------------------------
# Impairment channel
# --------------------------------------------------------------------------
def apply_channel(
    sig: np.ndarray,
    rng: np.random.Generator,
    snr_db: float,
    multipath: bool = True,
    max_cfo: float = 0.02,
) -> np.ndarray:
    n = len(sig)
    x = sig.copy()

    # multipath: a short random tapped-delay line (frequency-selective fading)
    if multipath and rng.random() < 0.7:
        ntap = rng.integers(2, 5)
        delays = rng.integers(1, 8, size=ntap)
        gains = (rng.standard_normal(ntap) + 1j * rng.standard_normal(ntap)) * 0.35
        h = np.zeros(int(delays.max()) + 1, dtype=np.complex128)
        h[0] = 1.0
        for d, g in zip(delays, gains):
            h[d] += g
        x = np.convolve(x, h, mode="same")

    # carrier frequency offset + Wiener phase noise
    cfo = rng.uniform(-max_cfo, max_cfo)
    pn_std = rng.uniform(0.0, 0.03)
    phase = TWO_PI * cfo * np.arange(n) + np.cumsum(rng.standard_normal(n) * pn_std)
    x = x * np.exp(1j * phase)

    # IQ imbalance (gain g + phase skew phi)
    g = rng.uniform(-0.08, 0.08)
    phi = rng.uniform(-0.1, 0.1)
    i = (1.0 + g) * x.real
    q = (1.0 - g) * (x.imag * np.cos(phi) + x.real * np.sin(phi))
    x = i + 1j * q

    # DC offset
    x = x + (rng.uniform(-0.05, 0.05) + 1j * rng.uniform(-0.05, 0.05))

    # additive white Gaussian noise at the requested SNR
    sp = np.mean(np.abs(x) ** 2) + 1e-12
    npow = sp / (10.0 ** (snr_db / 10.0))
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(npow / 2.0)
    x = x + noise

    # random overall power (removed later by amplitude normalisation; present so
    # the embedding must learn power-invariance)
    x = x * (10.0 ** (rng.uniform(-1.0, 1.0)))
    return x.astype(np.complex64)


def synth(
    cls: str,
    rng: np.random.Generator,
    raw_len: int = 4096,
    sps_range=(2, 16),
    snr_range=(0.0, 30.0),
) -> tuple[np.ndarray, float]:
    """One impaired realisation of `cls`. Returns (iq, sps_used)."""
    sps = int(rng.integers(sps_range[0], sps_range[1] + 1))
    snr = float(rng.uniform(*snr_range))
    clean = generate_clean(cls, raw_len, sps, rng)
    iq = apply_channel(clean, rng, snr)
    return iq, float(sps)
