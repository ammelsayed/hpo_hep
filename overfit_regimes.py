"""When does overtraining actually become real?

The early-stopping study showed that at the recommended hyper-parameters, even
4000 trees without early stopping do not hurt the held-out sample. This script
finds the regime where it does: remove the leaf floor, deepen the trees and
switch the subsampling off, i.e. the settings that let a tree memorise.

For each regime the train-vs-held-out KS gap and the held-out significance are
reported, together with the ratio held-out/validation, which is the quantity a
search would be fooled by.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from study_data import Channel, evaluate_scores
import xgboost as xgb

REGIMES = {
    # name: (params, n_estimators, early_stopping)
    "recommended": (dict(max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, reg_alpha=1e-3, reg_lambda=1.0,
                         gamma=0.1, min_child_weight=1.0), 4000, 40),
    "no_floor": (dict(max_depth=5, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, reg_alpha=1e-3, reg_lambda=1.0,
                      gamma=0.0, min_child_weight=1e-3), 4000, 40),
    "memorise": (dict(max_depth=12, learning_rate=0.3, subsample=1.0,
                      colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
                      gamma=0.0, min_child_weight=1e-6), 4000, None),
    "memorise_few_trees": (dict(max_depth=12, learning_rate=0.3, subsample=1.0,
                                colsample_bytree=1.0, reg_alpha=0.0,
                                reg_lambda=0.0, gamma=0.0,
                                min_child_weight=1e-6), 60, None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+",
                    default=["4L_eeee", "4L_eeemu", "3L_eeeJ", "3L_eee"])
    ap.add_argument("--out", default="results/overfit_regimes.csv")
    args = ap.parse_args()
    rows = []
    for name in args.channels:
        ch = Channel(name, device="cpu", nthread=1)
        for reg, (p, n_est, es) in REGIMES.items():
            q = {**ch.base, **p, "n_estimators": n_est, "early_stopping_rounds": es}
            m = xgb.XGBClassifier(**q)
            kw = dict(sample_weight=ch.w_tr)
            if es is not None:
                kw.update(eval_set=[(ch.X_va, ch.y_va)],
                          sample_weight_eval_set=[ch.w_va])
            m.fit(ch.X_tr, ch.y_tr, verbose=False, **kw)
            s_tr = m.predict_proba(ch.X_tr)[:, 1]
            s_va = m.predict_proba(ch.X_va)[:, 1]
            s_te = m.predict_proba(ch.X_te)[:, 1]
            va = evaluate_scores(ch.y_tr, s_tr, ch.w_tr, ch.y_va, s_va, ch.w_va)
            te = evaluate_scores(ch.y_tr, s_tr, ch.w_tr, ch.y_te, s_te, ch.w_te)
            rows.append(dict(channel=name, regime=reg,
                             trees=int(m.get_booster().num_boosted_rounds()),
                             gap_va=va["ot_ks_gap"], gap_te=te["ot_ks_gap"],
                             z_va=va["z_unbinned"], z_te=te["z_unbinned"],
                             auc_te=te["auc"], spike=te["sm_spike"],
                             ratio=te["z_unbinned"] / va["z_unbinned"]))
            r = rows[-1]
            print(f"{name:12s} {reg:20s} trees={r['trees']:5d} "
                  f"gap_va={r['gap_va']:.3f} gap_te={r['gap_te']:.3f} "
                  f"z_va={r['z_va']:.3f} z_te={r['z_te']:.3f} "
                  f"auc={r['auc_te']:.4f} spike={r['spike']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n=== mean over channels ===")
    print(df.groupby("regime").agg(trees=("trees", "mean"),
                                   gap_va=("gap_va", "mean"),
                                   gap_te=("gap_te", "mean"),
                                   z_te=("z_te", "mean"),
                                   auc_te=("auc_te", "mean"),
                                   spike=("spike", "mean"),
                                   ratio=("ratio", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
