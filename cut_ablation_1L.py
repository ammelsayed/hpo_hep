"""The 1L cut study with the cuts switched off, so the comparison is honest.

`cuts_1L.py` compared different models trained on differently cut samples. This
runs the *same* recipe with and without the MET cut on all four 1L channels, so
the significance change is attributable to the cut and nothing else.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from apply_recipe import run

ap = argparse.ArgumentParser()
ap.add_argument("--channels", nargs="+", default=["1L_eJ", "1L_eJJ", "1L_muJ", "1L_muJJ"])
ap.add_argument("--out", default="results/recipe_1L_cut_ablation.csv")
args = ap.parse_args()

rows = []
for ch in args.channels:
    for use in (False, True):
        r = run(ch, use, "cpu", 1)
        r["channel"] = ch
        rows.append(r)
        print(f"{ch:10s} cuts={str(use):5s} bkg={r['n_bkg_train']:7d} "
              f"Z_va={r['z_va']:.3f} Z_te={r['z_te']:.3f} "
              f"AUC={r['auc_te']:.4f} ot={r['ot_te']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)
piv = df.pivot_table(index="channel", columns="cuts", values="z_te")
piv.columns = ["no_cut", "with_cut"]
piv["gain"] = piv.with_cut - piv.no_cut
piv["gain_pct"] = 100 * piv.gain / piv.no_cut
print("\n=== held-out significance, same recipe with and without MET > 200 ===")
print(piv.round(3).to_string())
print(f"\nmean gain: {piv.gain.mean():+.3f} ({piv.gain_pct.mean():+.1f}%)")
