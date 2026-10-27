"""Sub-studies that the pooled search cannot answer on its own.

1. `min_child_weight`: should it be searched, left at the default, fixed to the
   value suggested by the event weights, or effectively switched off?
2. Early stopping: is the user's intuition right that a well-chosen parameter
   set keeps improving the validation metric until the cap, so that the
   `n_estimators` cap never bites?

Each policy is evaluated on the same train/val split and on the held-out half,
with the rest of the hyper-parameters fixed at the median of the best pooled
points, so that only the policy varies.

Usage: python sub_studies.py --channels 4L_eeee 3L_eeeJ ... --out results/sub
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from study_data import Channel, SPACE, suggest_mcw

# the median of the best pooled points per channel, i.e. a "typical good" set
BASE = dict(max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1e-3, reg_lambda=1.0, gamma=0.1)


def mcw_ablation(ch_name, device="cpu", nthread=1):
    ch = Channel(ch_name, device=device, nthread=nthread)
    mcw_sug = suggest_mcw(ch.y_tr, ch.w_tr)
    policies = {
        "searched_median": 1.0,     # what the search typically lands on
        "default": 1.0,             # XGBoost's default, in normalised units
        "suggested": mcw_sug,       # from the event weights
        "tiny": 1e-2,               # effectively no leaf floor
    }
    # the searched range, to say where the suggestion falls inside it
    lo, hi = SPACE["min_child_weight"][0], SPACE["min_child_weight"][1]
    rows = []
    for name, mcw in policies.items():
        model, fit_s = ch.fit({**BASE, "min_child_weight": float(mcw)})
        ev = ch.evaluate(model)
        rows.append(dict(channel=ch_name, study="mcw", policy=name, mcw=float(mcw),
                         fit_s=round(fit_s, 2), n_trees=ev["n_trees"],
                         z_va=ev["z_unbinned"], z_te=ev["te_z_unbinned"],
                         ot_va=ev["ot_ks_gap"], ot_te=ev["te_ot_ks_gap"],
                         auc_te=ev["te_auc"], spike=ev["te_sm_spike"],
                         mcw_suggested=mcw_sug,
                         in_range=bool(lo <= mcw <= hi)))
    return rows


def early_stopping_study(ch_name, caps=(100, 500, 1000, 2000, 4000, 8000),
                         device="cpu", nthread=1):
    """Does the validation metric keep improving, or does the cap bite?

    For each cap the model is trained with early stopping *disabled* and the
    validation log-loss and the held-out significance are recorded at the cap.
    If the metric is still falling at the largest cap, a higher cap would have
    helped and `n_estimators` is not a free parameter to ignore.
    """
    ch = Channel(ch_name, device=device, nthread=nthread)
    rows = []
    for cap in caps:
        model, fit_s = ch.fit({**BASE, "min_child_weight": 1.0},
                              n_estimators=int(cap))
        ev = ch.evaluate(model)
        hist = model.evals_result()["validation_0"]["logloss"]
        best_i = int(np.argmin(hist))
        rows.append(dict(channel=ch_name, study="early_stopping", cap=cap,
                         fit_s=round(fit_s, 2),
                         n_trees=ev["n_trees"], best_iteration=best_i,
                         logloss_at_cap=float(hist[-1]),
                         logloss_best=float(hist[best_i]),
                         still_improving=bool(best_i >= len(hist) - 5),
                         z_va=ev["z_unbinned"], z_te=ev["te_z_unbinned"],
                         ot_te=ev["te_ot_ks_gap"], auc_te=ev["te_auc"]))
        print(f"  {ch_name} cap={cap:5d} best_it={best_i:5d} "
              f"still_improving={rows[-1]['still_improving']} "
              f"z_te={ev['te_z_unbinned']:.3f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--out", default="results/sub")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nthread", type=int, default=1)
    ap.add_argument("--skip-early", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for ch in args.channels:
        print(f"=== {ch} ===", flush=True)
        rows += mcw_ablation(ch, args.device, args.nthread)
        if not args.skip_early:
            rows += early_stopping_study(ch, device=args.device, nthread=args.nthread)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "sub_studies.csv"), index=False)

    pd.set_option("display.width", 250)
    m = df[df.study == "mcw"]
    print("\n=== min_child_weight policies ===")
    print(m.groupby("policy").agg(z_te=("z_te", "mean"), ot_te=("ot_te", "mean"),
                                  n_trees=("n_trees", "mean"),
                                  auc_te=("auc_te", "mean"),
                                  spike=("spike", "mean")).round(4))
    print("\nsuggested values vs the searched range:")
    print(m[["channel", "mcw_suggested"]].drop_duplicates().to_string(index=False))

    e = df[df.study == "early_stopping"]
    if len(e):
        print("\n=== does the cap bite? ===")
        print(e.groupby("cap").agg(still_improving=("still_improving", "mean"),
                                   z_te=("z_te", "mean"),
                                   ot_te=("ot_te", "mean")).round(4))
        print("\nbest iteration as a fraction of the cap:")
        e2 = e.assign(frac=e.best_iteration / e.cap)
        print(e2.groupby("cap").frac.agg(["mean", "max"]).round(3))


if __name__ == "__main__":
    main()
