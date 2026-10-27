"""Do preselection cuts help the 1L channels, where S/B ~ 1e-4?

The user's suggestion: if the classification cannot improve without overtraining,
cut away most of the background while keeping most of the signal. This tests it
on the 1L regions, the ones with the largest background, by measuring the
held-out significance of a default BDT before and after a set of cuts.

Cuts are chosen from the physics of the final state, not tuned on the data:
a lepton with a high-pT fat jet, a large MET, and a hard HT are the usual
suppression handles for ttbar and W+jets, which dominate 1L.

  python cuts_1L.py --channels 1L_eJJ 1L_muJJ
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cutflow import Filter
from metrics import evaluate_scores
from study_data import load, collapse_background, split_df1, to_xyw, balance
import xgboost as xgb


def run(ch, cuts, device="cpu", nthread=1):
    sig1, bkg1, feats = load(ch, 1)
    sig2, bkg2, _ = load(ch, 2)
    bkg1 = collapse_background(bkg1, feats)
    bkg2 = collapse_background(bkg2, feats)

    flt = Filter(sig1, bkg1)
    flt.apply_cuts(cuts)
    s1, b1 = flt.get_dataframes()

    # the same cuts on the held-out half
    f2 = Filter(sig2, bkg2)
    f2.apply_cuts(cuts)
    s2, b2 = f2.get_dataframes()

    if len(s1) < 50 or len(b1) < 50:
        return None
    s_tr, b_tr, s_va, b_va = split_df1(s1, b1, feats)
    X_tr, y_tr, w = to_xyw(s_tr, b_tr, feats)
    w_tr = balance(y_tr, w)
    X_va, y_va, w_va = to_xyw(s_va, b_va, feats)
    X_te, y_te, w_te = to_xyw(s2, b2, feats)

    params = dict(max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, reg_alpha=1e-3, reg_lambda=1.0, gamma=0.1,
                  min_child_weight=1.0, objective="binary:logistic",
                  tree_method="hist", device=device, nthread=nthread,
                  random_state=42, verbosity=0, n_estimators=4000,
                  early_stopping_rounds=40)
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr, sample_weight=w_tr / w_tr.mean(),
          eval_set=[(X_va, y_va)], sample_weight_eval_set=[w_va], verbose=False)
    s_tr_s = m.predict_proba(X_tr)[:, 1]
    s_va_s = m.predict_proba(X_va)[:, 1]
    s_te_s = m.predict_proba(X_te)[:, 1]
    ev = evaluate_scores(y_tr, s_tr_s, w_tr, y_te, s_te_s, w_te)
    return dict(
        channel=ch, n_cuts=len(cuts), n_sig_train=len(s1), n_bkg_train=len(b1),
        n_sig_test=len(s2), n_bkg_test=len(b2),
        sig_eff=len(s1) / len(sig1), bkg_eff=len(b1) / len(bkg1),
        sb_train=w_tr[y_tr == 1].sum() / w_tr[y_tr == 0].sum(),
        z_te=ev["z_unbinned"], z_eff10=ev["z_eff10"], auc=ev["auc"],
        ot_te=ev["ot_ks_gap"], n_trees=int(m.best_iteration or 0) + 1,
    )


CUTSETS = {
    "none": {},
    "met": {"MET": "MET_MET > 200"},
    "ht": {"HT": "HT > 800"},
    "jet": {"Jet": "Pt_SoftDroppedFatJet0 > 200"},
    "met+ht": {"MET": "MET_MET > 200", "HT": "HT > 800"},
    "met+ht+jet": {"MET": "MET_MET > 200", "HT": "HT > 800",
                   "Jet": "Pt_SoftDroppedFatJet0 > 200"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", default=["1L_eJJ", "1L_muJJ"])
    ap.add_argument("--out", default="results/cuts.csv")
    args = ap.parse_args()
    rows = []
    for ch in args.channels:
        for name, cuts in CUTSETS.items():
            r = run(ch, cuts)
            if r is None:
                print(f"{ch:10s} {name:12s} -> too few events after the cuts")
                continue
            r["cuts"] = name
            rows.append(r)
            print(f"{ch:10s} {name:12s} n_bkg={r['n_bkg_train']:8d} "
                  f"eps_b={r['bkg_eff']:.3f} eps_s={r['sig_eff']:.3f} "
                  f"Z_te={r['z_te']:.2f} AUC={r['auc']:.4f} ot={r['ot_te']:.3f}",
                  flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwritten to {args.out}")
    if len(df):
        print(df.groupby("cuts").agg(z_te=("z_te", "mean"),
                                     z_eff10=("z_eff10", "mean"),
                                     auc=("auc", "mean"),
                                     eps_b=("bkg_eff", "mean"),
                                     eps_s=("sig_eff", "mean"),
                                     ot=("ot_te", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
