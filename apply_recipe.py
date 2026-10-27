"""Apply the recipe from the study to the remaining channels.

The study found (README section 5) that the optimiser and the overtraining term
do not matter, while three things do: the weight-scaled leaf floor, a fixed
early-stopping budget, and -- for the 1L channels only -- a MET preselection.
This script applies exactly that, with no search, and reports the held-out
performance of the resulting model for every channel it is given.

The hyper-parameters are the median of the best pooled points, which is what a
search would have landed on, so this is the "apply the winner" step rather than
a new tuning round.

  python apply_recipe.py --channels 2SSL_eeJ 2OSL_emuJ ...
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cutflow import Filter
from metrics import evaluate_scores
from study_data import (Channel, collapse_background, load, split_df1,
                        suggest_mcw, to_xyw, balance)
import xgboost as xgb

# the median of the best pooled points (README section 4.6)
BASE = dict(max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1e-3, reg_lambda=1.0, gamma=0.1)

# 1L gets the MET preselection (README section 4.8); everything else gets none
CUTS_1L = {"MET": "MET_MET > 200"}


def run(ch_name, use_cuts, device="cpu", nthread=1):
    sig1, bkg1, feats = load(ch_name, 1)
    sig2, bkg2, _ = load(ch_name, 2)
    bkg1 = collapse_background(bkg1, feats)
    bkg2 = collapse_background(bkg2, feats)

    if use_cuts:
        f1 = Filter(sig1, bkg1); f1.apply_cuts(CUTS_1L)
        sig1, bkg1 = f1.get_dataframes()
        f2 = Filter(sig2, bkg2); f2.apply_cuts(CUTS_1L)
        sig2, bkg2 = f2.get_dataframes()

    s_tr, b_tr, s_va, b_va = split_df1(sig1, bkg1, feats)
    X_tr, y_tr, w = to_xyw(s_tr, b_tr, feats)
    w_bal = balance(y_tr, w)
    w_scale = float(np.mean(w_bal))
    w_tr = w_bal / w_scale
    X_va, y_va, w_va = to_xyw(s_va, b_va, feats)
    X_te, y_te, w_te = to_xyw(sig2, bkg2, feats)

    # the leaf floor comes from the weights, in normalised units
    mcw = suggest_mcw(y_tr, w_tr)
    params = dict(objective="binary:logistic", tree_method="hist", device=device,
                  nthread=nthread, random_state=42, verbosity=0,
                  n_estimators=4000, early_stopping_rounds=40,
                  min_child_weight=mcw, **BASE)
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)],
          sample_weight_eval_set=[w_va], verbose=False)

    s_tr_s = m.predict_proba(X_tr)[:, 1]
    s_va_s = m.predict_proba(X_va)[:, 1]
    s_te_s = m.predict_proba(X_te)[:, 1]
    va = evaluate_scores(y_tr, s_tr_s, w_tr, y_va, s_va_s, w_va)
    te = evaluate_scores(y_tr, s_tr_s, w_tr, y_te, s_te_s, w_te)
    return dict(
        channel=ch_name, cuts=bool(use_cuts), mcw=float(mcw),
        n_sig_train=len(s_tr), n_bkg_train=len(b_tr),
        n_sig_test=int(y_te.sum()), n_bkg_test=len(y_te) - int(y_te.sum()),
        trees=int(m.get_booster().num_boosted_rounds()),
        z_va=va["z_unbinned"], z_te=te["z_unbinned"],
        z_eff10_te=te["z_eff10"], auc_te=te["auc"],
        ot_te=te["ot_ks_gap"], spike_te=te["sm_spike"],
        **{k: params[k] for k in ("max_depth", "learning_rate", "subsample",
                                  "colsample_bytree", "reg_alpha", "reg_lambda",
                                  "gamma")},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--out", default="results/recipe.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nthread", type=int, default=1)
    args = ap.parse_args()
    rows = []
    for ch in args.channels:
        use_cuts = ch.startswith("1L")
        r = run(ch, use_cuts, args.device, args.nthread)
        rows.append(r)
        print(f"{ch:14s} cuts={str(use_cuts):5s} trees={r['trees']:5d} "
              f"Z_va={r['z_va']:.3f} Z_te={r['z_te']:.3f} "
              f"AUC={r['auc_te']:.4f} ot={r['ot_te']:.3f} "
              f"spike={r['spike_te']:.1f}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwritten to {args.out}")
    print(df[["channel", "z_te", "z_eff10_te", "auc_te", "ot_te", "trees"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
