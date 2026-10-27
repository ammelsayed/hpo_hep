"""Common data loading and model training for the HPO study.

A model is trained on the `train` part of one half-sample and evaluated on
  * the `val` part of the same half  (what the HPO objective is allowed to see),
  * the *other* half-sample         (never seen during the study; the "truth").

Every trial is scored once and stored, so the objective variants and the search
algorithms can be compared offline on the same pool of models.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import xgboost as xgb
from metrics import evaluate_scores

DATA = os.path.join(HERE, "data")

# Hyper-parameter space shared by every algorithm and every objective.
# reg_lambda, reg_alpha, gamma and min_child_weight are in *normalised* hessian
# units: `Channel` rescales the training weights to mean 1, which multiplies
# every hessian by k = 1/mean(w). A leaf value -soft(G, alpha)/(H + lambda) is
# invariant under (w, alpha, lambda, gamma, mcw) -> (k*w, k*alpha, k*lambda,
# k*gamma, k*mcw), so the search then spans the ranges XGBoost documents instead
# of the 1e-5..1e-1 the raw weights (~1e-4) would demand -- at those raw values
# the default mcw = 1 already exceeds the total root hessian and every tree is
# a stump (best_iteration = 0, constant score).
SPACE = {
    "max_depth": (3, 10, False),
    "learning_rate": (0.01, 0.3, True),
    "subsample": (0.5, 1.0, False),
    "colsample_bytree": (0.5, 1.0, False),
    "reg_alpha": (1e-4, 10.0, True),
    "reg_lambda": (1e-2, 100.0, True),
    "gamma": (0.0, 10.0, False),
    "min_child_weight": (1e-2, 100.0, True),
}

# `min_child_weight` policies, compared in the dedicated ablation
MCW = {
    "searched": None,          # a free hyper-parameter (in SPACE)
    "default": 1.0,            # the XGBoost default, in normalised units
    "suggested": "suggest",    # from the event weights, see suggest_mcw
    "tiny": 1e-2,              # effectively no leaf floor
}


def load(channel, half):
    d = os.path.join(DATA, channel)
    sig = pd.read_parquet(f"{d}/sig_DF{half}.parquet")
    bkg = pd.read_parquet(f"{d}/bkg_DF{half}.parquet")
    feats = json.load(open(f"{d}/features_SR.json"))
    return sig, bkg, feats


def collapse_background(df, feats, weight_col="flat_weight"):
    """Collapse the per-mass-hypothesis copies of every background event.

    The background is replicated once per signal mass hypothesis (21 copies)
    with the weight split between them, and `signal_mass` is not one of the
    training features. The copies are therefore exact duplicates: summing their
    weights into one row leaves the gradient and hessian sums unchanged, while
    an unweighted KS test on 21 copies of the same point would be meaningless.
    Note this also restores the true variance: an event of weight W replicated
    n times with weight W/n has sum(w^2) = W^2/n, i.e. an error sqrt(n) too
    small, which is what the statistical-error estimates are computed from.
    """
    if "event_id" not in df.columns:
        return df
    agg = {f: "first" for f in feats}
    agg[weight_col] = "sum"
    if "proc_name" in df.columns:
        agg["proc_name"] = "first"
    return df.groupby("event_id", as_index=False, sort=False).agg(agg)


def split_df1(sig, bkg, feats, train_fraction=0.8, seed=42):
    """Split one half-sample into train/val at the level of whole events."""
    from sklearn.model_selection import train_test_split
    s_tr, s_va = train_test_split(sig, train_size=train_fraction,
                                  random_state=seed, shuffle=True)
    ids = bkg["event_id"].unique() if "event_id" in bkg else None
    if ids is None:
        b_tr, b_va = train_test_split(bkg, train_size=train_fraction,
                                      random_state=seed, shuffle=True)
    else:
        rng = np.random.RandomState(seed)
        ids = ids.copy()
        rng.shuffle(ids)
        keep = set(ids[: int(len(ids) * train_fraction)])
        mask = bkg["event_id"].isin(keep)
        b_tr, b_va = bkg[mask], bkg[~mask]
    return s_tr, b_tr, s_va, b_va


def to_xyw(sig, bkg, feats, weight_col="flat_weight"):
    X = np.vstack([sig[feats].values, bkg[feats].values]).astype(np.float32)
    y = np.concatenate([np.ones(len(sig), int), np.zeros(len(bkg), int)])
    w = np.concatenate([sig[weight_col].values, bkg[weight_col].values]).astype(np.float64)
    return X, y, w


def balance(y, w):
    """Rescale each class to the same total weight (as bdt_methods does)."""
    s = np.sum(w[y == 1]); b = np.sum(w[y == 0])
    f = np.where(y == 1, (s + b) / (2 * s), (s + b) / (2 * b))
    return w * f


def suggest_mcw(y, w, min_events_in_leaf=100):
    """`min_child_weight` from the event weights, in normalised hessian units.

    A leaf holds `min_events_in_leaf` events of the minority class, each with a
    hessian of at most 0.25*w (logistic loss), so the floor is that many
    events' worth of hessian. `w` must already be the *normalised* balanced
    training weights used for the fit.
    """
    minority = min(np.sum(w[y == 1]), np.sum(w[y == 0]))
    n_min = min(np.sum(y == 1), np.sum(y == 0))
    mean_w_min = minority / max(n_min, 1)
    return float(min_events_in_leaf * 0.25 * mean_w_min)


class Channel:
    """One signal region, split once so that every trial sees the same data."""

    def __init__(self, name, device="cpu", nthread=1, seed=42, train_half=1):
        self.name = name
        self.train_half, self.test_half = train_half, (2 if train_half == 1 else 1)
        sig1, bkg1, self.feats = load(name, self.train_half)
        sig2, bkg2, _ = load(name, self.test_half)
        bkg1 = collapse_background(bkg1, self.feats)
        bkg2 = collapse_background(bkg2, self.feats)
        s_tr, b_tr, s_va, b_va = split_df1(sig1, bkg1, self.feats, seed=seed)

        self.X_tr, self.y_tr, w = to_xyw(s_tr, b_tr, self.feats)
        wb = balance(self.y_tr, w)
        # normalise to mean weight 1: see the note on SPACE
        self.weight_scale = float(np.mean(wb))
        self.w_tr = wb / self.weight_scale
        self.X_va, self.y_va, self.w_va = to_xyw(s_va, b_va, self.feats)
        self.X_te, self.y_te, self.w_te = to_xyw(sig2, bkg2, self.feats)

        self.device, self.nthread = device, nthread
        self.base = dict(
            objective="binary:logistic", tree_method="hist", device=device,
            nthread=nthread, random_state=seed, verbosity=0,
            n_estimators=4000, early_stopping_rounds=40,
        )

    def n_events(self):
        return dict(train=len(self.y_tr), val=len(self.y_va), test=len(self.y_te),
                    sig_train=int(self.y_tr.sum()), bkg_train=int((1 - self.y_tr).sum()))

    def fit(self, params, n_estimators=None):
        p = {**self.base, **params}
        if n_estimators is not None:
            p["n_estimators"] = int(n_estimators)
        m = xgb.XGBClassifier(**p)
        t0 = time.time()
        m.fit(self.X_tr, self.y_tr, sample_weight=self.w_tr,
              eval_set=[(self.X_va, self.y_va)],
              sample_weight_eval_set=[self.w_va], verbose=False)
        return m, time.time() - t0

    def evaluate(self, model):
        """Every metric the study needs, for one trained model."""
        s_tr = model.predict_proba(self.X_tr)[:, 1]
        s_va = model.predict_proba(self.X_va)[:, 1]
        s_te = model.predict_proba(self.X_te)[:, 1]
        m = evaluate_scores(self.y_tr, s_tr, self.w_tr, self.y_va, s_va, self.w_va)
        truth = evaluate_scores(self.y_tr, s_tr, self.w_tr, self.y_te, s_te, self.w_te)
        out = dict(m)
        out.update({f"te_{k}": v for k, v in truth.items()})
        out["best_iteration"] = int(getattr(model, "best_iteration", 0) or 0)
        out["n_trees"] = int(model.get_booster().num_boosted_rounds())
        return out


def objective_from_row(row, sep="z_unbinned", over="ks_gap", A=1.0, B=1.0, prefix=""):
    """Evaluate an objective variant on an already computed metric row."""
    if row.get(prefix + "degenerate") or row.get(prefix + "inverted"):
        return -1e6
    x = row.get(prefix + sep)
    y = row.get(prefix + over) if prefix + over in row else row.get(prefix + "ot_" + over)
    if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
        return -1e6
    return float(A * x - B * y)


def truth_from_row(row, **kw):
    """The same objective on the held-out metrics: the ground truth."""
    return objective_from_row(row, prefix="te_", **kw)
