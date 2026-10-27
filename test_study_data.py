"""Self-check for study_data.py. Run: python test_study_data.py

Checks the two things that are easy to get silently wrong: that collapsing the
per-mass background copies is exact for the weighted sums, and that the weight
normalisation used to bring the XGBoost regularisation into standard units does
not change the model.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_data import (load, collapse_background, to_xyw, balance, Channel)

sig, bkg, feats = load("4L_eeee", 1)
u = collapse_background(bkg, feats)

# 1. the collapsed frame keeps the total weight (gradient and hessian sums)
assert len(u) < len(bkg)
assert np.isclose(u.flat_weight.sum(), bkg.flat_weight.sum(), rtol=1e-12)
# 2. and restores the true variance: an event of weight W replicated n times
#    with weight W/n has sum(w^2) = W^2/n, i.e. the error is sqrt(n) too small
n_copies = len(bkg) / max(len(u), 1)
assert np.isclose((u.flat_weight ** 2).sum(),
                  (bkg.flat_weight ** 2).sum() * n_copies, rtol=1e-6)
# 3. each row is a real event, not an average of different ones
assert len(u) == bkg.event_id.nunique()
assert u[feats].notna().all().all()

# 3. the balanced weights are normalised to mean 1 and keep the class balance
ch = Channel("4L_eeee", device="cpu")
assert abs(ch.w_tr.mean() - 1.0) < 1e-9, ch.w_tr.mean()
s = ch.w_tr[ch.y_tr == 1].sum(); b = ch.w_tr[ch.y_tr == 0].sum()
assert abs(s - b) < 1e-6 * max(s, b), (s, b)
assert ch.w_tr.min() > 0

# 4. normalising the weights and scaling every regularisation parameter by the
#    same factor leaves the model unchanged. All four live in the same units:
#    reg_lambda, gamma and min_child_weight in hessians (H -> kH), reg_alpha in
#    gradients (G -> kG), since a leaf is -soft(G, alpha)/(H + lambda).
params = dict(max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
              reg_alpha=1e-3, reg_lambda=1.0, gamma=0.0, min_child_weight=1.0)
m1, _ = ch.fit(params, n_estimators=40)
k = ch.weight_scale
p2 = {key: params[key] * k for key in ("reg_alpha", "reg_lambda", "gamma", "min_child_weight")}
p2.update({key: params[key] for key in params if key not in p2})
w_saved = ch.w_tr
ch.w_tr = w_saved * k
m2, _ = ch.fit(p2, n_estimators=40)
ch.w_tr = w_saved
s1 = m1.predict_proba(ch.X_va)[:, 1]; s2 = m2.predict_proba(ch.X_va)[:, 1]
assert np.max(np.abs(s1 - s2)) < 1e-4, np.max(np.abs(s1 - s2))

# 5. the normalised default mcw actually lets trees grow (the raw default does not)
m3, _ = ch.fit(dict(max_depth=4, learning_rate=0.1, min_child_weight=1.0), n_estimators=40)
leaves = (m3.get_booster().trees_to_dataframe()["Feature"] == "Leaf").sum()
assert leaves > 40, leaves

# 6. the split keeps the background halves disjoint
from study_data import split_df1
b1 = collapse_background(load("4L_eeee", 1)[1], feats)
_, b_tr, _, b_va = split_df1(load("4L_eeee", 1)[0], b1, feats)
assert len(set(b_tr.event_id) & set(b_va.event_id)) == 0

print("all study_data self-checks passed")
