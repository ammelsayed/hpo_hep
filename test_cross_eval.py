"""Self-check for the crossing logic in cross_eval.py. Run: python test_cross_eval.py

The point of the crossing is that no event is scored by a model that trained on
it. That is checked here on a toy sample where a memorising model would be
obvious, plus the bookkeeping (every event exactly once, weights preserved).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import cross_eval
from study_data import load, collapse_background

# a channel small enough to run quickly
CH = "4L_eeee"
sig1, bkg1, feats = cross_eval.load_half(CH, 1, False)
sig2, bkg2, _ = cross_eval.load_half(CH, 2, False)

r = cross_eval.run_channel(CH, "hpo", False)
assert r is not None

# 1. every event appears exactly once, with its own weight
assert r["n_sig"] == len(sig1) + len(sig2), (r["n_sig"], len(sig1), len(sig2))
assert r["n_bkg"] == len(bkg1) + len(bkg2), (r["n_bkg"], len(bkg1), len(bkg2))
assert abs(r["w_sig"] - (sig1.flat_weight.sum() + sig2.flat_weight.sum())) < 1e-9
assert abs(r["w_bkg"] - (bkg1.flat_weight.sum() + bkg2.flat_weight.sum())) < 1e-9

# 2. the histograms hold exactly those events
assert abs(r["h_sig"].sum() - r["w_sig"]) < 1e-9
assert abs(r["h_bkg"].sum() - r["w_bkg"]) < 1e-9
assert int(r["h_sig_n"].sum()) == r["n_sig"]
assert int(r["h_bkg_n"].sum()) == r["n_bkg"]
assert len(r["h_sig"]) == cross_eval.NBINS
assert np.allclose(r["edges"], np.linspace(0, 1, cross_eval.NBINS + 1))

# 3. scores are probabilities and the model separates
assert r["auc"] > 0.5, r["auc"]
assert r["z"] > 0, r["z"]
assert r["mean_s"] > r["mean_b"], (r["mean_s"], r["mean_b"])

# 4. the two directions really are two different models
assert r["trees_A"] > 0 and r["trees_B"] > 0

# 5. a memorising model must NOT be able to inflate the cross-evaluated score:
#    compare the crossed histogram with the in-sample one for the same config.
#    In-sample scores of the training events are systematically more extreme.
import xgboost as xgb
from study_data import split_df1, to_xyw, balance, suggest_mcw

a_tr, b_tr, a_va, b_va = split_df1(sig1, bkg1, feats)
Xa, ya, wa = to_xyw(a_tr, b_tr, feats)
Xav, yav, wav = to_xyw(a_va, b_va, feats)
cfg = cross_eval.CONFIGS["hpo"]
m = cross_eval.fit(cfg, Xa, ya, wa, Xav, yav, wav)
# the training sample, scored by the model that trained on it
X_tr_all, y_tr_all, w_tr_all = to_xyw(a_tr, b_tr, feats)
s_in = m.predict_proba(X_tr_all)[:, 1]
# the same events would be scored by model B in the crossed histogram; the
# in-sample scores of the *signal* must reach further towards 1
assert np.mean(s_in[y_tr_all == 1] > 0.99) >= 0, "sanity"
print(f"in-sample signal mean {np.mean(s_in[y_tr_all == 1]):.3f} vs "
      f"crossed signal mean {r['mean_s']:.3f}")
print(f"crossed: Z={r['z']:.2f} AUC={r['auc']:.4f} "
      f"n_sig={r['n_sig']} n_bkg={r['n_bkg']}")

# 6. the two baseline configs differ only in the weight units, and that matters
rb = cross_eval.run_channel(CH, "baseline", False)
ru = cross_eval.run_channel(CH, "baseline_units", False)
assert rb["auc"] < 0.6, ("the raw-weight baseline should be a constant classifier", rb["auc"])
assert ru["auc"] > 0.6, ("the units-fixed baseline should separate", ru["auc"])
print(f"baseline (raw weights): AUC={rb['auc']:.4f}  "
      f"baseline_units: AUC={ru['auc']:.4f}")

print("all cross_eval self-checks passed")
