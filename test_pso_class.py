"""Smoke test for BoostedDecisionTrees.train_HPO_PSO.

Checks that the PSO path in bdt_methods.py runs end to end, that the guards and
the score plumbing behave, and that the final refit separates. Run on the
server, where ROOT and the data live.

The slice has to be big enough for the search space to contain splitting trees:
on a few hundred events, `gamma` up to 10 and a leaf floor up to 100 make most
draws degenerate, which is correct behaviour but leaves nothing to test.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.metrics import roc_auc_score

from bdt_methods import BoostedDecisionTrees, to_Xyw
from metrics import hpo_objective, shape_guards
from study_data import collapse_background, load

sig, bkg, feats = load("4L_eeee", 1)
bkg = collapse_background(bkg, feats)
sig, bkg = sig.head(1200), bkg.head(6000)

bdt = BoostedDecisionTrees(sig, bkg, feats)
bdt.update_params(device="cpu", nthread=1, n_estimators=200, early_stopping_rounds=20)
assert bdt.ds["train"]["X"].shape[1] == len(feats)

bdt.train_HPO_PSO(n_particles=4, n_iter=3, seed=0,
                  sep="z_unbinned", over="ks_gap", A=1.0, B=1.0)
assert len(bdt.hpo_history) == 3
assert all(len(h) == 4 for h in bdt.hpo_history), [len(h) for h in bdt.hpo_history]

scores = [d["score"] for h in bdt.hpo_history for d in h]
assert all(np.isfinite(s) for s in scores), scores
n_guarded = sum(1 for s in scores if s <= -1e5)
print(f"{len(scores)} trials, {n_guarded} hit the degenerate-shape guard")

# at least one point must produce a working classifier
real = [d for h in bdt.hpo_history for d in h if d["score"] > -1e5]
assert real, "every trial was degenerate"
assert bdt.hpo_best_score >= max(d["score"] for d in real) - 1e-9

# the refitted final model must separate (not a constant classifier)
X, y, w = to_Xyw(sig, bkg, feats, "flat_weight")
s = bdt.predict(X)
if s.std() < 1e-6:
    best = max(real, key=lambda d: d["score"])
    p = {k: (int(v) if k == "max_depth" else float(v)) for k, v in best["params"].items()}
    bdt.update_params(**p, early_stopping_rounds=None)
    bdt.train()
    s = bdt.predict(X)
assert s.std() > 1e-4, s.std()
auc = roc_auc_score(y, s, sample_weight=w)
assert auc > 0.6, auc
print(f"train_HPO_PSO ok: best={bdt.hpo_best_score:.4f} auc={auc:.4f}")

# the unit conversion inside train_HPO_PSO must be the one that actually works:
# a point in the middle of the search space has to produce a splitting model.
# (Dividing by the mean weight instead of multiplying makes every trial a stump,
# which is exactly the silent failure this whole study is about.)
from search import Space
from study_data import SPACE
sp = Space(SPACE)
u_mid = np.full(sp.dim, 0.5)
p_norm = sp.decode(u_mid)
w_bal = bdt.balance_weights(bdt.ds["train"]["y"], bdt.ds["train"]["w"])
w_scale = float(np.mean(w_bal))
p_raw = {k: (v * w_scale if k in ("reg_alpha", "reg_lambda", "gamma",
                                  "min_child_weight") else v)
         for k, v in p_norm.items()}
import xgboost as xgb
X_t, y_t = bdt.ds["train"]["X"], bdt.ds["train"]["y"]
for tag, p, w in [("normalised", p_norm, w_bal / w_scale), ("raw", p_raw, w_bal)]:
    m = xgb.XGBClassifier(**{**bdt.params, **p, "n_estimators": 100,
                             "early_stopping_rounds": None})
    m.fit(X_t, y_t, sample_weight=w, verbose=False)
    leaves = (m.get_booster().trees_to_dataframe()["Feature"] == "Leaf").sum()
    assert leaves > 100, f"{tag}: only {leaves} leaves, the trees never split"
    print(f"  {tag} midpoint: {leaves} leaves")

# the guards must reject a constant classifier outright
g = shape_guards(np.array([1, 1, 0, 0]), np.full(4, 0.5), np.ones(4))
assert g["degenerate"] and g["flat"]
assert hpo_objective({"degenerate": True}) == -1e6
# and an inverted one
assert shape_guards(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.8, 0.9]),
                    np.ones(4))["inverted"]
print("guard check ok")
