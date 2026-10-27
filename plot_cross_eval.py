"""Per-channel cross-evaluated score distributions: baseline vs HPO.

Reads the .npz histograms written by cross_eval.py and produces, for each
channel, a figure with the signal and background score distributions of both
configurations, plus a table of the separation and overtraining numbers.

  python plot_cross_eval.py --indir results/cross_eval --outdir results/cross_eval
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "legend.fontsize": 6.5})

CONFIG_STYLE = {
    "baseline": dict(color="#7f8c8d", ls="--", label="baseline (repo defaults)"),
    "baseline_units": dict(color="#e67e22", ls=":", label="baseline, units fixed"),
    "hpo": dict(color="#c0392b", ls="-", label="HPO recipe"),
}


def load_hist(indir, ch, cfg):
    f = os.path.join(indir, f"hist__{ch}__{cfg}.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f)
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/cross_eval")
    ap.add_argument("--outdir", default="results/cross_eval/figures")
    ap.add_argument("--configs", nargs="+", default=["baseline", "hpo"])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    summary = pd.read_csv(os.path.join(args.indir, "summary.csv"))
    channels = sorted(summary.channel.unique())

    for ch in channels:
        rows = summary[summary.channel == ch]
        fig, axes = plt.subplots(1, len(args.configs), figsize=(4.1 * len(args.configs), 3.0),
                                 sharey=True)
        if len(args.configs) == 1:
            axes = [axes]
        for ax, cfg in zip(axes, args.configs):
            h = load_hist(args.indir, ch, cfg)
            if h is None:
                ax.set_visible(False)
                continue
            edges = h["edges"]
            ctr = 0.5 * (edges[:-1] + edges[1:])
            hs, hb = h["h_sig"], h["h_bkg"]
            # normalise each to unit area so the shapes are comparable
            hs = hs / hs.sum() if hs.sum() > 0 else hs
            hb = hb / hb.sum() if hb.sum() > 0 else hb
            st = CONFIG_STYLE.get(cfg, dict(label=cfg))
            ax.step(ctr, hs, where="mid", color="#c0392b", lw=1.4, label="signal")
            ax.step(ctr, hb, where="mid", color="#2980b9", lw=1.4, label="background")
            # a log axis with a filled area shades the whole panel, so the
            # shapes are drawn as lines only; the floor keeps empty bins visible
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1.0)
            r = rows[rows.config == cfg]
            ttl = st["label"]
            if len(r):
                r = r.iloc[0]
                ttl += f"\nZ={r.z:.2f}  AUC={r.auc:.4f}"
            ax.set_title(ttl, fontsize=8)
            ax.set_xlabel("BDT score")
            ax.set_xlim(0, 1)
            ax.legend(loc="upper center")
        axes[0].set_ylabel("normalised events / bin")
        fig.suptitle(f"{ch} — cross-evaluated (DF1→DF2 and DF2→DF1 combined)",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"score__{ch}.png"))
        plt.close(fig)
        print(f"  {ch}")

    # an overview: Z of both configs per channel
    piv = summary.pivot_table(index="channel", columns="config", values="z")
    piv = piv.reindex(channels)
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.28 * len(channels))))
    y = np.arange(len(piv))
    for i, cfg in enumerate(args.configs):
        if cfg not in piv:
            continue
        st = CONFIG_STYLE.get(cfg, dict(label=cfg))
        ax.barh(y + (i - 0.5 * (len(args.configs) - 1)) * 0.38, piv[cfg], height=0.36,
                color=st["color"], alpha=0.9, label=st["label"])
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"cross-evaluated Asimov $Z$")
    ax.legend(fontsize=7)
    ax.set_title("baseline vs HPO, all channels")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "overview_z.png"))
    plt.close(fig)

    print(f"\nfigures in {args.outdir}")
    for f in sorted(glob.glob(os.path.join(args.outdir, "*.png")))[:5]:
        print(" ", os.path.basename(f))


if __name__ == "__main__":
    main()
