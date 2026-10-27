"""Is early stopping what controls overtraining, rather than the hyper-parameters?

If so, the overtraining term in `A*x - B*y` is measuring something that the
training procedure has already fixed, and the `B` weight only spends separation.

For one channel, the same hyper-parameter points are trained three ways:
  * early stopping on the validation log-loss (what the study uses),
  * no early stopping at the full 4000 rounds,
  * no early stopping and no validation set at all.
The train-vs-held-out gap and the held-out significance are compared.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from search import Space
from study_data import Channel, SPACE, evaluate_scores

P = dict(max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
         reg_alpha=1e-3, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0)


def one(ch, mode, n_est=4000):
    if mode == "early":
        model, _ = ch.fit(P)
    else:
        import xgboost as xgb
        p = {**ch.base, **P, "n_estimators": n_est}
        if mode == "no_es":
            p["early_stopping_rounds"] = None
            model = xgb.XGBClassifier(**p)
            model.fit(ch.X_tr, ch.y_tr, sample_weight=ch.w_tr,
                      eval_set=[(ch.X_va, ch.y_va)],
                      sample_weight_eval_set=[ch.w_va], verbose=False)
        else:                                   # no validation set at all
            p["early_stopping_rounds"] = None
            model = xgb.XGBClassifier(**p)
            model.fit(ch.X_tr, ch.y_tr, sample_weight=ch.w_tr, verbose=False)
    s_tr = model.predict_proba(ch.X_tr)[:, 1]
    s_va = model.predict_proba(ch.X_va)[:, 1]
    s_te = model.predict_proba(ch.X_te)[:, 1]
    va = evaluate_scores(ch.y_tr, s_tr, ch.w_tr, ch.y_va, s_va, ch.w_va)
    te = evaluate_scores(ch.y_tr, s_tr, ch.w_tr, ch.y_te, s_te, ch.w_te)
    return dict(mode=mode, n_trees=int(model.get_booster().num_boosted_rounds()),
                gap_va=va["ot_ks_gap"], gap_te=te["ot_ks_gap"],
                z_va=va["z_unbinned"], z_te=te["z_unbinned"],
                auc_te=te["auc"], ratio=te["z_unbinned"] / va["z_unbinned"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+",
                    default=["4L_eeee", "4L_eeemu", "3L_eeeJ", "3L_eee"])
    ap.add_argument("--out", default="results/early_stop_effect.csv")
    args = ap.parse_args()
    rows = []
    for name in args.channels:
        ch = Channel(name, device="cpu", nthread=1)
        for mode in ["early", "no_es", "no_val"]:
            r = one(ch, mode)
            r["channel"] = name
            rows.append(r)
            print(f"{name:12s} {mode:7s} trees={r['n_trees']:5d} "
                  f"gap_va={r['gap_va']:.3f} gap_te={r['gap_te']:.3f} "
                  f"z_va={r['z_va']:.3f} z_te={r['z_te']:.3f} "
                  f"auc={r['auc_te']:.4f} te/va={r['ratio']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n=== mean over channels ===")
    print(df.groupby("mode").agg(n_trees=("n_trees", "mean"),
                                 gap_va=("gap_va", "mean"),
                                 gap_te=("gap_te", "mean"),
                                 z_te=("z_te", "mean"),
                                 auc_te=("auc_te", "mean"),
                                 ratio=("ratio", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
