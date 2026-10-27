"""Produce the figures and tables that go into the README."""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})

pool = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/*__*__s*.csv"))],
                 ignore_index=True)
rep = pd.read_csv("results/replication.csv")
sub = pd.read_csv("results/sub/sub_studies.csv")

# ---------------------------------------------------------------- #
# fig 1: validation Z vs held-out Z -- why the objective is weak
# ---------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
ax = axes[0]
ch = "4L_eeee"
d = pool[pool.channel == ch]
ax.hexbin(d.z_unbinned, d.te_z_unbinned, gridsize=35, cmap="viridis", mincnt=1)
r = spearmanr(d.z_unbinned, d.te_z_unbinned).statistic
ax.set_xlabel(r"validation $Z$ (what the HPO sees)")
ax.set_ylabel(r"held-out $Z$ (DF2)")
ax.set_title(f"{ch}: $\\rho$ = {r:.2f}")
ax = axes[1]
for name, dd in [("4L_eeee", pool[pool.channel == "4L_eeee"]),
                 ("3L_eeeJ", pool[pool.channel == "3L_eeeJ"]),
                 ("4L_eeemu", pool[pool.channel == "4L_eeemu"])]:
    ax.scatter(dd.z_unbinned / dd.z_unbinned.mean(),
               dd.te_z_unbinned / dd.te_z_unbinned.mean(), s=3, alpha=0.3, label=name)
ax.set_xlabel("validation $Z$ / mean")
ax.set_ylabel("held-out $Z$ / mean")
ax.legend(markerscale=4, fontsize=7)
ax.set_title("the validation ranking barely transfers")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_val_vs_truth.png")
plt.close(fig)

# ---------------------------------------------------------------- #
# fig 2: replication -- what survives when the halves are swapped
# ---------------------------------------------------------------- #
ctrl = pd.read_csv("results/replication.csv")
m = []
for ch in ctrl.channel.unique():
    r = pd.read_csv(f"results/replication_{ch}.csv")
    m.append(dict(channel=ch, ctrl=r.te_z_unbinned.mean()))
ctrl = pd.DataFrame(m)
rep = rep.merge(ctrl, on="channel")
rep["gain"] = rep.rev_z - rep.ctrl

g = rep.groupby(["sep", "over", "A", "B"]).agg(
    gain=("gain", "mean"), gain_err=("gain", lambda x: x.std() / np.sqrt(len(x))),
    ot=("rev_ot", "mean")).reset_index()
g["label"] = g.sep + ("" if (g.B == 0).all() else " - " + g.B.astype(str) + g.over)
g = g.sort_values("gain")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
cols = ["#c0392b" if v > 0 else "#7f8c8d" for v in g.gain]
ax.barh(range(len(g)), g.gain, xerr=g.gain_err, color=cols, alpha=0.85,
        error_kw=dict(lw=0.8, capsize=2))
ax.set_yticks(range(len(g)))
ax.set_yticklabels([f"{r.sep} $-$ {r.B:g}$\\cdot${r.over}" for r in g.itertuples()],
                   fontsize=7)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel(r"held-out $Z$ gain over a random choice (replication direction)")
ax.set_title("does penalising overtraining help? (red = it does)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_replication.png")
plt.close(fig)

# ---------------------------------------------------------------- #
# fig 3: early stopping never reaches the cap
# ---------------------------------------------------------------- #
e = sub[sub.study == "early_stopping"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axes[0]
for ch, d in e.groupby("channel"):
    ax.plot(d.cap, d.best_iteration, "o-", ms=3, lw=1, label=ch)
ax.plot([100, 8000], [100, 8000], "k--", lw=0.8, label="cap")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("n_estimators cap"); ax.set_ylabel("best iteration")
ax.legend(fontsize=6, ncol=2)
ax.set_title("early stopping converges far below the cap")
ax = axes[1]
for ch, d in e.groupby("channel"):
    ax.plot(d.cap, d.z_te, "o-", ms=3, lw=1, label=ch)
ax.set_xscale("log")
ax.set_xlabel("n_estimators cap"); ax.set_ylabel("held-out $Z$")
ax.set_title("more trees do not buy significance")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_early_stopping.png")
plt.close(fig)

# ---------------------------------------------------------------- #
# fig 4: the algorithm comparison
# ---------------------------------------------------------------- #
conv = pd.read_csv("results/convergence.csv")
norm = pool.groupby("channel").te_z_unbinned.max()
conv["rel"] = conv.best_z / conv.channel.map(norm)
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for algo, d in conv.groupby("algo"):
    s = d.groupby("trial").rel.agg(["mean", "sem"])
    ax.fill_between(s.index, s["mean"] - s["sem"], s["mean"] + s["sem"], alpha=0.15)
    ax.plot(s.index, s["mean"], lw=1.4, label=algo)
ax.set_xlabel("model fits")
ax.set_ylabel("best held-out $Z$ / best in pool")
ax.legend(fontsize=8)
ax.set_title("all four searches converge by ~10 fits")
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_algorithms.png")
plt.close(fig)

# ---------------------------------------------------------------- #
# fig 5: min_child_weight and the cuts
# ---------------------------------------------------------------- #
mcw = sub[sub.study == "mcw"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axes[0]
piv = mcw.pivot_table(index="channel", columns="policy", values="z_te")
piv = piv[["tiny", "default", "suggested"]]
piv.plot(kind="bar", ax=ax, width=0.8)
ax.set_ylabel("held-out $Z$")
ax.set_title("min_child_weight: the suggested value costs little")
ax.tick_params(axis="x", rotation=45, labelsize=7)
ax.legend(fontsize=7)
ax = axes[1]
piv2 = mcw.pivot_table(index="channel", columns="policy", values="ot_te")
piv2 = piv2[["tiny", "default", "suggested"]]
piv2.plot(kind="bar", ax=ax, width=0.8)
ax.set_ylabel("held-out train-vs-test KS gap")
ax.set_title("... and halves the overtraining gap")
ax.tick_params(axis="x", rotation=45, labelsize=7)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_mcw.png")
plt.close(fig)

cuts = pd.read_csv("results/cuts.csv")
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for ch, d in cuts.groupby("channel"):
    ax.plot(d.bkg_eff, d.z_te, "o-", label=ch)
ax.set_xlabel("background efficiency of the cuts")
ax.set_ylabel("held-out $Z$")
ax.legend(fontsize=8)
ax.set_title("preselection cuts on the 1L channels")
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_cuts.png")
plt.close(fig)

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f)
