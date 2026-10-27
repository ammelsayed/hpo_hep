"""Replication test: does an HPO choice survive a swap of the data halves?

Every region has two statistically independent half-samples. The pool was built
on DF1 (train) -> DF2 (held out). This script takes the hyper-parameter sets
that each objective variant selected, retrains them on DF2, and measures them on
DF1. A choice that is real must score well in both directions; a choice driven
by statistical noise will not.

Only a sample of points is refit: the top-k of every variant (they overlap
heavily) plus a random control sample of the rest, which is enough to place the
chosen points on the reversed-direction distribution.

Usage:
  python replicate.py --pooldir results/pool --out results/replication.csv \
      --topk 8 --ncontrol 60
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from study_data import Channel, SPACE

GUARD = -1e6
# the objective variants the pool comparison ranked highest, plus controls
VARIANTS = [
    ("z_unbinned", "ks_gap", 1.0, 1.0),
    ("z_unbinned", "ks_gap", 1.0, 0.0),     # separation only
    ("z_unbinned", "ks_gap", 1.0, 2.0),     # overtraining punished twice
    ("z_unbinned", "ks_gap", 2.0, 1.0),
    ("z_eff10", "ks_gap", 1.0, 1.0),
    ("z_eff30", "ks_gap", 1.0, 1.0),
    ("z_binned", "ks_gap", 1.0, 1.0),
    ("auc", "ks_gap", 1.0, 1.0),
    ("mean_sep", "ks_gap", 1.0, 1.0),
    ("ks_sep", "ks_gap", 1.0, 1.0),
    ("z_unbinned", "chi2", 1.0, 1.0),
    ("z_unbinned", "mean_gap", 1.0, 1.0),
    ("z_unbinned", "dz_eff10", 1.0, 1.0),
]


def score_col(df, sep, over, A, B):
    bad = df["degenerate"] | df["inverted"]
    x = df[sep].astype(float)
    col = over if over in df.columns else f"ot_{over}"
    s = A * x - B * df[col].astype(float)
    return s.where(np.isfinite(s) & ~bad, GUARD).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooldir", default="results/pool")
    ap.add_argument("--out", default="results/replication.csv")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--ncontrol", type=int, default=60)
    ap.add_argument("--channels", nargs="+",
                    default=["4L_eeee", "4L_eeemu", "4L_eemumu", "4L_emumumu",
                             "4L_mumumumu", "3L_eeeJ", "3L_eee", "3L_mumumuJ"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nthread", type=int, default=1)
    args = ap.parse_args()

    rows = []
    for ch_name in args.channels:
        pool = pd.concat([pd.read_csv(f) for f in
                          sorted(glob.glob(f"{args.pooldir}/{ch_name}__*.csv"))],
                         ignore_index=True).drop_duplicates(subset="point")
        pool = pool.reset_index(drop=True)
        rng = np.random.RandomState(0)

        # which points to refit: every variant's top-k, plus a random control
        need = set()
        for sep, over, A, B in VARIANTS:
            s = score_col(pool, sep, over, A, B)
            need.update(pool.iloc[np.argsort(-s)[: args.topk]].point)
        rest = [p for p in pool.point if p not in need]
        if len(rest) > args.ncontrol:
            rest = list(rng.choice(rest, args.ncontrol, replace=False))
        need.update(rest)

        ch2 = Channel(ch_name, device=args.device, nthread=args.nthread, train_half=2)
        print(f"{ch_name}: refitting {len(need)} of {len(pool)} points on the "
              f"other half", flush=True)
        rev = {}
        for i, pt in enumerate(sorted(need)):
            r = pool[pool.point == pt].iloc[0]
            p = {k: (int(r[k]) if k == "max_depth" else float(r[k])) for k in SPACE}
            model, _ = ch2.fit(p)
            rev[pt] = ch2.evaluate(model)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(need)}", flush=True)
        rev = pd.DataFrame(rev).T.reset_index().rename(columns={"index": "point"})
        rev["channel"] = ch_name
        rev.to_csv(f"results/replication_{ch_name}.csv", index=False)
        rev_oracle = float(rev["te_z_unbinned"].max())
        fwd_oracle = float(pool["te_z_unbinned"].max())

        for sep, over, A, B in VARIANTS:
            s = score_col(pool, sep, over, A, B)
            order = np.argsort(-s)
            sel = pool.iloc[order[0]]
            top = pool.iloc[order[: args.topk]]
            rt = rev[rev.point.isin(top.point)]
            rs = rev[rev.point == sel.point]
            rows.append(dict(
                channel=ch_name, sep=sep, over=over, A=A, B=B,
                fwd_z=float(sel["te_z_unbinned"]), fwd_ot=float(sel["te_ot_ks_gap"]),
                rev_z=float(rs["te_z_unbinned"].iloc[0]) if len(rs) else np.nan,
                rev_ot=float(rs["te_ot_ks_gap"].iloc[0]) if len(rs) else np.nan,
                rev_oracle=rev_oracle, fwd_oracle=fwd_oracle,
                rev_z_topk=float(rt["te_z_unbinned"].mean()),
                rev_ot_topk=float(rt["te_ot_ks_gap"].mean()),
                fwd_z_topk=float(top["te_z_unbinned"].mean()),
                n_topk=len(rt)))
            print(f"  {sep:11s}-{over:9s} fwd {rows[-1]['fwd_z']:.3f} "
                  f"rev {rows[-1]['rev_z']:.3f} rev_topk {rows[-1]['rev_z_topk']:.3f} "
                  f"(rev oracle {rev_oracle:.3f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n=== replication summary (mean over channels) ===")
    g = df.groupby(["sep", "over", "A", "B"]).agg(
        fwd_z=("fwd_z", "mean"), rev_z=("rev_z", "mean"),
        rev_oracle=("rev_oracle", "mean"), fwd_oracle=("fwd_oracle", "mean"),
        rev_topk=("rev_z_topk", "mean"), fwd_topk=("fwd_z_topk", "mean"),
        rev_ot=("rev_ot", "mean"), rev_ot_topk=("rev_ot_topk", "mean"),
        n=("channel", "count")).reset_index()
    # how much of the reversed-direction oracle the choice captured
    g["capture"] = g.rev_z - g.rev_oracle
    g["capture_topk"] = g.rev_topk - g.rev_oracle
    g = g.sort_values("capture_topk", ascending=False)
    pd.set_option("display.width", 250)
    print(g.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
