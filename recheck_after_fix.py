"""Re-check the study conclusions after the tie-blocking fix in metrics.py.

The fix changes z_unbinned for models with many tied scores. The pool was built
with the old metric, so the stored z_unbinned column cannot be reused: the
comparison has to be rebuilt from the stored scores. Since the pool did not keep
per-event scores, this re-runs the key comparison on a fresh, smaller pool.

It checks the two claims that the fix could plausibly break:
  1. a constant classifier now scores Z = 0 (the bug that was fixed);
  2. A*x - B*y with B > 0 is still worse than B = 0.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics import evaluate_scores, zmax, separation_metrics
from search import Space
from study_data import Channel, SPACE, objective_from_row

CHANNELS = ["4L_eeee", "4L_eeemu", "4L_eemumu", "4L_emumumu",
            "4L_mumumumu", "3L_eeeJ", "3L_eee", "3L_mumumuJ"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="results/recheck.csv")
    args = ap.parse_args()

    sp = Space(SPACE)
    rng = np.random.RandomState(0)
    points = [sp.decode(u) for u in rng.rand(args.n, sp.dim)]

    rows = []
    for ch_name in CHANNELS:
        ch = Channel(ch_name, device="cpu", nthread=1)
        for i, p in enumerate(points):
            p = {k: (int(v) if k == "max_depth" else float(v)) for k, v in p.items()}
            model, _ = ch.fit(p)
            ev = ch.evaluate(model)
            ev.update(p)
            ev.update(channel=ch_name, i=i)
            rows.append(ev)
        print(f"{ch_name}: {len(points)} points", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 250)
    print("\n=== is any model degenerate, and does z_unbinned respect it? ===")
    print(f"  degenerate: {int(df.degenerate.sum())} / {len(df)}")
    bad = df[df.degenerate]
    if len(bad):
        print(f"  their z_unbinned: max {bad.z_unbinned.max():.4f} "
              f"(must be 0)")

    print("\n=== objective variants, held-out truth ===")
    from scipy.stats import spearmanr
    for sep, over, A, B in [("z_unbinned", "ks_gap", 1.0, 0.0),
                            ("z_unbinned", "ks_gap", 1.0, 1.0),
                            ("z_unbinned", "ks_gap", 1.0, 2.0)]:
        s = np.array([objective_from_row(r, sep, over, A, B) for _, r in df.iterrows()])
        ok = s > -1e5
        rho = spearmanr(s[ok], df.te_z_unbinned[ok]).statistic if ok.sum() > 10 else np.nan
        sel = df.iloc[int(np.argmax(s))]
        print(f"  {sep} - {B}*{over}: rho={rho:+.3f}  "
              f"selected held-out Z={sel.te_z_unbinned:.3f}")

    print("\n=== z_unbinned vs held-out z_unbinned, per channel ===")
    for ch_name, d in df.groupby("channel"):
        r = spearmanr(d.z_unbinned, d.te_z_unbinned).statistic
        print(f"  {ch_name:14s} rho={r:+.3f}  "
              f"z_va median {d.z_unbinned.median():.3f}  "
              f"z_te median {d.te_z_unbinned.median():.3f}")


if __name__ == "__main__":
    main()
