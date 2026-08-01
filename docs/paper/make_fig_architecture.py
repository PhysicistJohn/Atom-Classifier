"""DACS architecture figure: three-lane grid, Manhattan-routed arrows.

Lanes (canvas 100x48): prototype head y=38, main spine y=22, decoder y=10,
escalation loop y=3.5. Real spectrograms from the production corpus.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parent.parent / "training/artifacts/longdwell-production-corpus"

# ---------------------------------------------------------------- data thumbs
manifest = json.loads((CORPUS / "manifest.json").read_text())
row = next(r for r in manifest["rows"] if r["cls"] == "gsm" and r["role"] == "eval")
noisy = np.load(CORPUS / "noisy.npy", mmap_mode="r")[row["row"], :200_000]
clean = np.load(CORPUS / "clean.npy", mmap_mode="r")[row["row"], :200_000]


def spec(x, nfft=256, hop=128):
    frames = 1 + (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(frames)[:, None]
    S = np.fft.fftshift(np.fft.fft(x[idx] * np.hanning(nfft), axis=1), axes=1)
    return 20 * np.log10(np.abs(S).T + 1e-6)


S_in, S_cl = spec(np.asarray(noisy)), spec(np.asarray(clean))
vmax = S_in.max()

# ------------------------------------------------------------------- palette
C = dict(enc="#dbe9f5", encE="#4878a8", z="#cfd8e8", zE="#5a6a8a",
         proto="#eef7ec", protoE="#5a8a5a", conf="#ece8f8", confE="#7a68b0",
         gate="#faf3dc", gateE="#b08828", ans="#e2f0e2", ansE="#3a7a3a",
         dec="#fdf0e0", decE="#c08838", loss="#fdeaea", lossE="#b04848",
         loop="#b03030", gray="#666666", dim="#999999")

fig, ax = plt.subplots(figsize=(9.5, 4.55))
ax.set_xlim(0, 100); ax.set_ylim(0, 48)
ax.axis("off")

FS_MAIN, FS_SUB, FS_TINY = 13, 10.5, 9


def arrow(p0, p1, color="#333333", lw=1.6, ls="-", z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))


def seg(p0, p1, color, lw=1.6, ls="-", z=2):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, ls=ls,
            zorder=z, solid_capstyle="round")


def chip(x0, y0, x1, y1, fc, ec, lines, z=4):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.35", fc=fc, ec=ec,
                                lw=1.3, zorder=z))
    cy = (y0 + y1) / 2
    if len(lines) == 1:
        ax.text((x0 + x1) / 2, cy, lines[0][0], ha="center", va="center",
                fontsize=lines[0][1], color=lines[0][2], zorder=z + 1,
                style=lines[0][3])
    else:
        offs = [+1.55, -1.55] if len(lines) == 2 else [+2.2, 0, -2.2]
        for (txt, fs, col, st), dy in zip(lines, offs):
            ax.text((x0 + x1) / 2, cy + dy, txt, ha="center", va="center",
                    fontsize=fs, color=col, zorder=z + 1, style=st)


# ------------------------------------------------------- training-only group
ax.add_patch(FancyBboxPatch((38.5, 5.6), 47.5, 10.2,
                            boxstyle="round,pad=0.4", fc="none",
                            ec="#aaaaaa", lw=1.1, ls=(0, (4, 3)), zorder=1))
ax.text(41.2, 15.05, "training-only supervision", ha="left", va="center",
        fontsize=FS_TINY - 0.5, color=C["gray"], style="italic", zorder=2)

# ------------------------------------------------------------ input thumbnail
ax.imshow(S_in, extent=(3, 15, 16, 28), origin="lower", aspect="auto",
          cmap="magma", vmin=vmax - 60, vmax=vmax, zorder=3)
ax.add_patch(Rectangle((3, 16), 12, 12, fc="none", ec="#444444", lw=1.0,
                       zorder=4))
ax.text(9, 29.3, r"capture $x(0{:}\ell_j)$ — any dwell", ha="center",
        va="bottom", fontsize=FS_SUB, color="#333333")
ax.text(9.5, 14.9, r"STFT $S_x$ $\cdot$ mask $m$ (train)", ha="center",
        va="top", fontsize=FS_TINY, color=C["gray"])

# ------------------------------------------------------------------- encoder
arrow((15.4, 22), (18.6, 22))
ax.add_patch(Polygon([(19, 15.5), (19, 28.5), (29, 25), (29, 19)],
                     fc=C["enc"], ec=C["encE"], lw=1.4, zorder=4))
ax.text(24, 23.1, "encoder $E$", ha="center", va="center", fontsize=FS_MAIN,
        color="#1a3a5a", zorder=5)
ax.text(24, 20.6, r"time $\downarrow$32$\times$", ha="center", va="center",
        fontsize=FS_TINY, color="#4a6a8a", zorder=5)
arrow((29.4, 22), (31.6, 22))

# -------------------------------------------------------------------- latent
ax.add_patch(FancyBboxPatch((32, 17.5), 3.0, 9.0, boxstyle="round,pad=0.3",
                            fc=C["z"], ec=C["zE"], lw=1.4, zorder=4))
ax.text(33.5, 22, "$z$", ha="center", va="center", fontsize=FS_MAIN + 1,
        color="#2a3a5a", zorder=5)
ax.text(33.5, 16.1, "[192, T/32]", ha="center", va="top", fontsize=FS_TINY,
        color=C["gray"])

# fan-out: straight to confidence; risers to prototype and decoder lanes
arrow((35.4, 22), (40.6, 22))
seg((36.8, 22), (36.8, 38), "#333333"); arrow((36.8, 38), (40.6, 38))
seg((36.8, 22), (36.8, 10), "#333333"); arrow((36.8, 10), (40.6, 10))

# ---------------------------------------------------- prototype head (top)
ax.add_patch(FancyBboxPatch((41, 31), 19, 14, boxstyle="round,pad=0.35",
                            fc=C["proto"], ec=C["protoE"], lw=1.3, zorder=4))
ax.text(50.5, 43.3, "prototype space (128-d)", ha="center", va="center",
        fontsize=FS_TINY + 0.5, color="#2a5a2a", zorder=6)
rng = np.random.default_rng(7)
cents = np.array([[45.5, 36.5], [48.3, 40.2], [51.8, 34.2], [54.6, 39.6],
                  [57.2, 36.0], [46.8, 33.4], [55.6, 33.0]])
cols = plt.cm.tab10(np.linspace(0, 0.9, 7))
q = np.array([49.3, 37.3])
for k in np.argsort(np.linalg.norm(cents - q, axis=1))[:3]:
    seg(q, cents[k], "#888888", lw=0.9, ls=(0, (2, 2)), z=5)
ax.scatter(cents[:, 0], cents[:, 1], s=52, c=cols, ec="white", lw=0.8,
           zorder=6)
ax.scatter(*q, marker="*", s=150, c="#222222", zorder=7)
ax.text(50.5, 32.3, r"$\hat y = \arg\min_k\, d(z, \mathbf{c}_k)$",
        ha="center", va="center", fontsize=FS_TINY + 0.5, color="#2a5a2a",
        zorder=6)
arrow((60.4, 38), (62.6, 38), color=C["protoE"])
chip(63, 34.2, 86, 41.8, "#e6efe6", C["protoE"],
     [(r"episodic CE $+$", FS_SUB, "#2a5a2a", "normal"),
      ("worst-dwell auxiliary", FS_SUB, "#2a5a2a", "normal"),
      ("know the class", FS_TINY, C["gray"], "italic")])

# ---------------------------------------------------- confidence head (mid)
chip(41, 18.6, 55, 25.4, C["conf"], C["confE"],
     [("confidence $g$", FS_SUB + 0.5, "#4a3a80", "normal"),
      ("know thyself", FS_TINY, C["gray"], "italic")])
arrow((55.4, 22), (58.6, 22), color=C["confE"])
ax.add_patch(Polygon([(59, 22), (63.5, 25.6), (68, 22), (63.5, 18.4)],
                     fc=C["gate"], ec=C["gateE"], lw=1.4, zorder=4))
ax.text(63.5, 22, r"$g \geq \tau$?", ha="center", va="center",
        fontsize=FS_SUB + 0.5, color="#6a5010", zorder=5)
arrow((68.4, 22), (71.6, 22), color=C["ansE"])
ax.text(70, 23.1, "yes", ha="center", va="bottom", fontsize=FS_TINY,
        color=C["ansE"])
chip(72, 19, 87.5, 25, C["ans"], C["ansE"],
     [(r"answer $(\hat y,\, \ell_j)$", FS_SUB + 0.5, "#1a5a1a", "normal")])

# ------------------------------------------------------- decoder head (low)
ax.add_patch(Polygon([(41, 8.8), (41, 11.2), (51, 14.2), (51, 5.8)],
                     fc=C["dec"], ec=C["decE"], lw=1.4, zorder=4))
ax.text(46, 10, "decoder $D$", ha="center", va="center", fontsize=FS_SUB + 1,
        color="#7a5010", zorder=5)
arrow((51.4, 10), (52.6, 10), color=C["decE"])
ax.imshow(S_cl, extent=(53, 62, 6.2, 13.8), origin="lower", aspect="auto",
          cmap="magma", vmin=vmax - 60, vmax=vmax, zorder=3)
ax.add_patch(Rectangle((53, 6.2), 9, 7.6, fc="none", ec="#444444", lw=1.0,
                       zorder=4))
arrow((62.4, 10), (63.6, 10), color=C["lossE"])
chip(64, 6.2, 86, 13.8, C["loss"], C["lossE"],
     [(r"$\lambda_r\, \|\hat S - S_{\mathrm{clean}}\|_1$",
       FS_SUB + 0.5, "#8a2a2a", "normal"),
      ("rests are targets too", FS_SUB - 1, "#8a2a2a", "normal"),
      ("know the channel", FS_TINY, C["gray"], "italic")])

# -------------------------------------------------------- escalation loop
seg((63.5, 18.4), (63.5, 17.0), C["loop"], ls=(0, (5, 3)))
seg((63.5, 17.0), (89.5, 17.0), C["loop"], ls=(0, (5, 3)))
seg((89.5, 17.0), (89.5, 2.2), C["loop"], ls=(0, (5, 3)))
seg((89.5, 2.2), (1.2, 2.2), C["loop"], ls=(0, (5, 3)))
seg((1.2, 2.2), (1.2, 22), C["loop"], ls=(0, (5, 3)))
arrow((1.2, 22), (2.9, 22), color=C["loop"], ls=(0, (5, 3)), lw=1.6)
ax.text(64.4, 17.5, "no", ha="left", va="bottom", fontsize=FS_TINY,
        color=C["loop"])
ax.text(49.25, 2.9, r"no: extend dwell $\ell_j \rightarrow \ell_{j+1}$ — "
        "keep listening (Algorithm 2)", ha="center", va="bottom",
        fontsize=FS_SUB, color=C["loop"])

fig.savefig(HERE / "fig_architecture.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("wrote fig_architecture.png")
