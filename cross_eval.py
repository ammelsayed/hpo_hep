"""Cross-evaluated BDT score distributions for every channel.

Each region has two statistically independent half-samples, DF1 and DF2. A score
histogram that is honest must contain, for every event, a score from a model
that never saw that event. That is obtained by crossing the two halves:

    model A: trained on DF1  ->  scores every event of DF2
    model B: trained on DF2  ->  scores every event of DF1
    histogram: the union of the two sets of scores

so no event is scored by a model that trained on it, and all the events are
used. (Training a model and then predicting the same sample would put the
training events into the plot and is exactly the bias this study is about.)

Each training half is split 80/20 internally: 80 % to grow the trees, 20 % for
early stopping. The other half is only ever predicted, never used to fit or to
stop.

Configurations compared:

  baseline        the XGBoost defaults as `bdt_methods.py` shipped them, with
                  the raw physical weights
  baseline_units  the same defaults, but with the weights normalised to mean 1
                  (the control: it separates the units bug from the HPO gain)
  hpo             the recipe of README section 5: the median good point of the
                  pooled search, a leaf floor taken from the event weights,
                  early stopping

Usage:
  python cross_eval.py --channels 4L_eeee 3L_eee --outdir results/cross_eval
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import xgboost as xgb
from cutflow import Filter
from metrics import weighted_ks, zmax
from study_data import (balance, collapse_background, load, split_df1,
                        suggest_mcw, to_xyw)

NBINS = 30

CONFIGS = {
    # the repo defaults: no explicit max_depth/learning_rate/... in `self.params`
    "baseline": dict(
        params=dict(max_depth=6, learning_rate=0.3, subsample=1.0,
                    colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=1.0,
                    gamma=0.0, min_child_weight=1.0),
        normalise=False, mcw="fixed", n_estimators=3000, es=20),
    # the same defaults, units fixed: the control for the comparison
    "baseline_units": dict(
        params=dict(max_depth=6, learning_rate=0.3, subsample=1.0,
                    colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=1.0,
                    gamma=0.0, min_child_weight=1.0),
        normalise=True, mcw="fixed", n_estimators=3000, es=20),
    # the recipe of README section 5
    "hpo": dict(
        params=dict(max_depth=5, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=1e-3, reg_lambda=1.0,
                    gamma=0.1),
        normalise=True, mcw="suggested", n_estimators=4000, es=40),
}

# 1L gets the MET preselection (README section 4.8), applied identically to
# every configuration so that the comparison isolates the model.
CUTS_1L = {"MET": "MET_MET > 200"}


def fit(cfg, X_tr, y_tr, w_tr_raw, X_va, y_va, w_va):
    w = balance(y_tr, w_tr_raw)
    if cfg["normalise"]:
        w = w / np.mean(w)
    p = dict(cfg["params"])
    if cfg["mcw"] == "suggested":
        p["min_child_weight"] = suggest_mcw(y_tr, w)
    m = xgb.XGBClassifier(
        objective="binary:logistic", tree_method="hist", device="cpu",
        nthread=1, random_state=42, verbosity=0,
        n_estimators=cfg["n_estimators"], early_stopping_rounds=cfg["es"], **p)
    m.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_va, y_va)],
          sample_weight_eval_set=[w_va], verbose=False)
    return m


def load_half(ch, half, use_cuts):
    sig, bkg, feats = load(ch, half)
    bkg = collapse_background(bkg, feats)
    if use_cuts:
        f = Filter(sig, bkg)
        f.apply_cuts(CUTS_1L)
        sig, bkg = f.get_dataframes()
    return sig, bkg, feats


def run_channel(ch, cfg_name, use_cuts, device="cpu"):
    cfg = CONFIGS[cfg_name]
    s1, b1, feats = load_half(ch, 1, use_cuts)
    s2, b2, _ = load_half(ch, 2, use_cuts)
    if len(s1) < 20 or len(b1) < 20 or len(s2) < 20 or len(b2) < 20:
        return None

    # ---- direction A: train on DF1, score DF2 ------------------------------
    a_tr, b_tr, a_va, b_va = split_df1(s1, b1, feats)
    Xa, ya, wa = to_xyw(a_tr, b_tr, feats)
    Xav, yav, wav = to_xyw(a_va, b_va, feats)
    mA = fit(cfg, Xa, ya, wa, Xav, yav, wav)
    X2s, _, _ = to_xyw(s2, s2.iloc[:0], feats)      # signal rows only
    X2b, _, _ = to_xyw(s2.iloc[:0], b2, feats)      # background rows only
    s2_score = mA.predict_proba(X2s)[:, 1]
    b2_score = mA.predict_proba(X2b)[:, 1]

    # ---- direction B: train on DF2, score DF1 ------------------------------
    a2_tr, b2_tr, a2_va, b2_va = split_df1(s2, b2, feats)
    Xb, yb, wb = to_xyw(a2_tr, b2_tr, feats)
    Xbv, ybv, wbv = to_xyw(a2_va, b2_va, feats)
    mB = fit(cfg, Xb, yb, wb, Xbv, ybv, wbv)
    X1s, _, _ = to_xyw(s1, s1.iloc[:0], feats)
    X1b, _, _ = to_xyw(s1.iloc[:0], b1, feats)
    s1_score = mB.predict_proba(X1s)[:, 1]
    b1_score = mB.predict_proba(X1b)[:, 1]

    # ---- the cross-evaluated union ----------------------------------------
    sig_score = np.concatenate([s1_score, s2_score])
    bkg_score = np.concatenate([b1_score, b2_score])
    sig_w = np.concatenate([s1.flat_weight.values, s2.flat_weight.values])
    bkg_w = np.concatenate([b1.flat_weight.values, b2.flat_weight.values])

    # every event must appear exactly once
    assert len(sig_score) == len(s1) + len(s2), (len(sig_score), len(s1), len(s2))
    assert len(bkg_score) == len(b1) + len(b2), (len(bkg_score), len(b1), len(b2))

    y = np.concatenate([np.ones(len(sig_score), int), np.zeros(len(bkg_score), int)])
    sc = np.concatenate([sig_score, bkg_score])
    w = np.concatenate([sig_w, bkg_w])

    edges = np.linspace(0.0, 1.0, NBINS + 1)
    h_s, _ = np.histogram(sig_score, bins=edges, weights=sig_w)
    h_b, _ = np.histogram(bkg_score, bins=edges, weights=bkg_w)
    h_sn, _ = np.histogram(sig_score, bins=edges, weights=np.ones_like(sig_w))
    h_bn, _ = np.histogram(bkg_score, bins=edges, weights=np.ones_like(bkg_w))

    from sklearn.metrics import roc_auc_score
    return dict(
        channel=ch, config=cfg_name, cuts=bool(use_cuts),
        n_sig=len(sig_score), n_bkg=len(bkg_score),
        w_sig=float(sig_w.sum()), w_bkg=float(bkg_w.sum()),
        z=zmax(y, sc, w),
        auc=float(roc_auc_score(y, sc, sample_weight=w)),
        ks_sep=float(weighted_ks(sig_score, sig_w, bkg_score, bkg_w)),
        mean_s=float(np.average(sig_score, weights=sig_w)),
        mean_b=float(np.average(bkg_score, weights=bkg_w)),
        trees_A=int(mA.get_booster().num_boosted_rounds()),
        trees_B=int(mB.get_booster().num_boosted_rounds()),
        h_sig=h_s, h_bkg=h_b, h_sig_n=h_sn, h_bkg_n=h_bn, edges=edges,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--outdir", default="results/cross_eval")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    for ch in args.channels:
        use_cuts = ch.startswith("1L")
        for cfg_name in args.configs:
            r = run_channel(ch, cfg_name, use_cuts)
            if r is None:
                print(f"{ch} {cfg_name}: too few events after the cuts", flush=True)
                continue
            hists = {k: r.pop(k) for k in ("h_sig", "h_bkg", "h_sig_n", "h_bkg_n",
                                           "edges")}
            np.savez(os.path.join(args.outdir, f"hist__{ch}__{cfg_name}.npz"),
                     **hists)
            rows.append(r)
            print(f"{ch:14s} {cfg_name:15s} Z={r['z']:6.2f} AUC={r['auc']:.4f} "
                  f"KS={r['ks_sep']:.3f} trees={r['trees_A']}/{r['trees_B']}",
                  flush=True)

    df = pd.DataFrame(rows)
    # one file per channel as well as the merged one, so that the runs can be
    # launched in parallel without overwriting each other's summary
    for ch, g in df.groupby("channel"):
        g.to_csv(os.path.join(args.outdir, f"summary__{ch}.csv"), index=False)
    out = os.path.join(args.outdir, "summary.csv")
    df.to_csv(out, index=False)
    print(f"\nwritten to {out}")
    if len(df):
        piv = df.pivot_table(index="channel", columns="config", values="z")
        print("\n=== cross-evaluated Asimov Z ===")
        print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
