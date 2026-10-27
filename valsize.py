"""Why is the objective only weakly correlated with the truth? Sample size.

The validation split is 20% of one half-sample; the held-out sample is the whole
other half, roughly five times larger. The Asimov Z of a small sample is a noisy
estimate of the same model's Z on a large one, so the ranking it produces is
partly noise. That is a property of the *evaluation*, not of the search, and it
caps what any HPO algorithm can achieve.

This script measures the effect: for one channel, train the same models and
score them on validation splits of increasing size, then correlate each with the
held-out truth.

  python valsize.py --channel 4L_eeee --n 40
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from search import Space
from study_data import (Channel, SPACE, load, collapse_background, split_df1,
                        to_xyw, balance)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="4L_eeee")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nthread", type=int, default=1)
    args = ap.parse_args()

    ch = Channel(args.channel, device=args.device, nthread=args.nthread)
    print(f"{args.channel}: train {len(ch.y_tr)}, val {len(ch.y_va)}, "
          f"test {len(ch.y_te)} events", flush=True)

    # a spread of hyper-parameter points: the pooled best plus random draws
    sp = Space(SPACE)
    rng = np.random.RandomState(0)
    params = [sp.decode(u) for u in rng.rand(args.n, sp.dim)]

    rows = []
    for i, p in enumerate(params):
        model, _ = ch.fit(p)
        s_tr = model.predict_proba(ch.X_tr)[:, 1]
        s_te = model.predict_proba(ch.X_te)[:, 1]
        # the truth: scored on the whole held-out half
        from metrics import evaluate_scores
        truth = evaluate_scores(ch.y_tr, s_tr, ch.w_tr, ch.y_te, s_te, ch.w_te)
        rec = dict(i=i, z_te=truth["z_unbinned"], z_te_eff10=truth["z_eff10"],
                   auc_te=truth["auc"], ot_te=truth["ot_ks_gap"])
        # the same model scored on validation subsets of increasing size
        for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
            n = int(frac * len(ch.y_va))
            idx = rng.choice(len(ch.y_va), n, replace=False)
            s_sub = model.predict_proba(ch.X_va[idx])[:, 1]
            ev = evaluate_scores(ch.y_tr, s_tr, ch.w_tr,
                                 ch.y_va[idx], s_sub, ch.w_va[idx])
            rec[f"z_va_{int(frac*100)}"] = ev["z_unbinned"]
            rec[f"eff10_va_{int(frac*100)}"] = ev["z_eff10"]
            rec[f"auc_va_{int(frac*100)}"] = ev["auc"]
        rows.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(params)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"results/valsize_{args.channel}.csv", index=False)

    print(f"\n=== correlation with the held-out truth, vs validation size ===")
    out = []
    for frac in [20, 40, 60, 80, 100]:
        n = int(frac / 100 * len(ch.y_va))
        out.append(dict(
            val_frac=f"{frac}%", n_val=n,
            rho_z=spearmanr(df[f"z_va_{frac}"], df.z_te).statistic,
            rho_eff10=spearmanr(df[f"eff10_va_{frac}"], df.z_te_eff10).statistic,
            rho_auc=spearmanr(df[f"auc_va_{frac}"], df.auc_te).statistic))
    print(pd.DataFrame(out).round(3).to_string(index=False))
    print("\nEven at 100% of the validation split the rank correlation is limited")
    print("by the size of that split, not by the search.")


if __name__ == "__main__":
    main()
