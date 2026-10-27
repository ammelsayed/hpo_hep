"""Self-check for the HPO metrics in metrics.py. Run: python test_metrics.py"""
import numpy as np
from metrics import (separation_metrics, shape_guards, overtraining_metrics,
                     hpo_objective, evaluate_scores, weighted_ks,
                     z_at_signal_efficiency, asimov_z_curve, zmax)

rng = np.random.RandomState(0)


def make(n_s=4000, n_b=40000, sep=1.0):
    s = np.clip(rng.beta(5, 2, n_s) if sep > 0.5 else rng.uniform(0, 1, n_s), 0, 1)
    b = np.clip(rng.beta(2, 5, n_b) if sep > 0.5 else rng.uniform(0, 1, n_b), 0, 1)
    y = np.concatenate([np.ones(n_s, int), np.zeros(n_b, int)])
    sc = np.concatenate([s, b])
    w = np.concatenate([np.full(n_s, 5e-3), np.full(n_b, 2.5e-3)])
    return y, sc, w


# 1. a separated classifier must have a positive significance and a mean_sep > 2
y, sc, w = make()
m = separation_metrics(y, sc, w)
assert m["z_unbinned"] > 3, m
assert m["mean_sep"] > 2, m
assert 0.5 < m["auc"] < 1.0, m
assert 0.3 < m["ks_sep"] <= 1.0, m

# 2. a constant classifier is degenerate and its objective is the guard value
yc = y
g = shape_guards(yc, np.full_like(sc, 0.5), w)
assert g["degenerate"] and g["n_nonempty_bins"] == 1, g
assert hpo_objective({"degenerate": True}) == -1e6

# 3. an inverted classifier is caught by the guard
sc_inv = 1.0 - sc
m_inv = separation_metrics(y, sc_inv, w)
g_inv = shape_guards(y, sc_inv, w)
assert g_inv["inverted"], g_inv
assert hpo_objective({**m_inv, **g_inv}) == -1e6

# 4. overtraining: identical distributions -> ks_gap ~ 0, a shifted one -> large
y2, sc2, w2 = make()
ot_same = overtraining_metrics(y2, sc2, w2, y2, sc2, w2)
assert ot_same["ks_gap"] < 1e-9, ot_same
ot_shift = overtraining_metrics(y2, sc2, w2, y2, np.clip(sc2 + 0.3, 0, 1), w2)
assert ot_shift["ks_gap"] > 0.2, ot_shift
assert ot_shift["mean_gap"] > 0.5, ot_shift
assert ot_shift["chi2"] > ot_same["chi2"], (ot_shift["chi2"], ot_same["chi2"])

# 5. weighted_ks matches an unweighted ECDF comparison
a = rng.normal(size=500); b = rng.normal(1.0, size=800)
d = weighted_ks(a, np.ones_like(a), b, np.ones_like(b))
from scipy.stats import ks_2samp
assert abs(d - ks_2samp(a, b).statistic) < 1e-9, (d, ks_2samp(a, b).statistic)

# 6. z at a signal efficiency is monotone non-decreasing in the efficiency
z10 = z_at_signal_efficiency(y, sc, w, 0.10)
z50 = z_at_signal_efficiency(y, sc, w, 0.50)
assert z10 <= z50 + 1e-9, (z10, z50)

# 6b. a cut that keeps signal and no background must give a large Z, not zero
#     (the background regularisation is what makes this finite)
yt = np.array([1, 1, 1, 1, 0, 0, 0, 0])
st = np.array([.9, .8, .7, .6, .1, .2, .3, .4])
wt = np.ones(8)
assert z_at_signal_efficiency(yt, st, wt, 0.25) > 1.0, z_at_signal_efficiency(yt, st, wt, 0.25)
zt, _, _ = asimov_z_curve(yt, st, wt)
assert zt[3] > 1.0, zt          # after the 4th (highest) signal, b = 0

# 6c. a constant score separates nothing, so its max-cut Z must be zero.
#     Ties have to be blocked: walking tied events one at a time would let the
#     constant "cut" after all the signal and report a large Z for a model with
#     AUC = 0.5. This is the bug that made the raw-weight baseline look good.
y_const = np.concatenate([np.ones(500, int), np.zeros(2000, int)])
s_const = np.full(2500, 0.5)
w_const = np.ones(2500)
assert abs(zmax(y_const, s_const, w_const)) < 1e-9, zmax(y_const, s_const, w_const)
assert separation_metrics(y_const, s_const, w_const)["z_unbinned"] < 1e-9
# 6d. an inverted classifier also has its best cut at "everything passes", so it
#     must be rejected too rather than credited with the whole-sample Z
s_inv2 = np.where(y_const == 1, 0.3, 0.7)
assert zmax(y_const, s_inv2, w_const) < 1e-9, zmax(y_const, s_inv2, w_const)
assert shape_guards(y_const, s_inv2, w_const)["inverted"]
assert separation_metrics(y_const, s_inv2, w_const)["z_unbinned"] < 1e-9
# 6e. the same binary score, correctly oriented, must be credited normally
s_ok2 = np.where(y_const == 1, 0.7, 0.3)
assert zmax(y_const, s_ok2, w_const) > 10, zmax(y_const, s_ok2, w_const)
# 6f. with real separation the tie handling must not lose anything
assert zmax(y, sc, w) == max(asimov_z_curve(y, sc, w)[0])

# 7. objective ordering: a better classifier scores higher for every (sep, over)
m_good = separation_metrics(*make())
m_bad = separation_metrics(*make(sep=0.1))
for sep in ["z_unbinned", "auc", "z_eff10", "z_eff30", "z_binned", "mean_sep", "ks_sep"]:
    o_good = hpo_objective({**m_good, "degenerate": False, "inverted": False,
                            "ot_ks_gap": 0.05, "ot_mean_gap": 0.05, "ot_chi2": 0.1,
                            "ot_ks_sig": 0.05, "ot_ks_bkg": 0.05,
                            "ot_mean_gap_sig": 0.05, "ot_mean_gap_bkg": 0.05,
                            "ot_chi2_sig": 0.1, "ot_chi2_bkg": 0.1}, sep=sep, over="ks_gap")
    o_bad = hpo_objective({**m_bad, "degenerate": False, "inverted": False,
                           "ot_ks_gap": 0.05, "ot_mean_gap": 0.05, "ot_chi2": 0.1,
                           "ot_ks_sig": 0.05, "ot_ks_bkg": 0.05,
                           "ot_mean_gap_sig": 0.05, "ot_mean_gap_bkg": 0.05,
                           "ot_chi2_sig": 0.1, "ot_chi2_bkg": 0.1}, sep=sep, over="ks_gap")
    assert o_good > o_bad, (sep, o_good, o_bad)

# 8. evaluate_scores returns every key the objective variants need
e = evaluate_scores(y, sc, w, y, sc, w)
for k in ["z_unbinned", "z_eff10", "z_eff30", "z_binned", "auc", "mean_sep", "ks_sep",
          "degenerate", "inverted", "ot_ks_gap", "ot_mean_gap", "ot_chi2", "dz_eff10"]:
    assert k in e, k
assert abs(e["dz_eff10"]) < 1e-9, e["dz_eff10"]

print("all metric self-checks passed")
