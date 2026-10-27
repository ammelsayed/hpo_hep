import math
import numpy as np
from sklearn.metrics import roc_auc_score

def asimov_significance(s, b, b_err=0.0):
    """Discovery significance with background uncertainty."""
    if s <= 0.0:
        return 0.0
    if b <= 0.0:
        b = 1e-10  # small placeholder to avoid division by zero
        b_err = max(b_err, 1e-10)

    # No background uncertainty: Cowan et al. asymptotic formula
    if b_err <= 0.0:
        val = 2.0 * ((s + b) * math.log1p(s / b) - s)
        return math.sqrt(max(val, 0.0))

    sb2 = b_err * b_err
    denom = b * b + (s + b) * sb2
    if denom <= 0.0:
        return 0.0

    term1 = (s + b) * math.log(((s + b) * (b + sb2)) / denom)
    term2 = (b * b / sb2) * math.log(1.0 + (sb2 * s) / (b * (b + sb2)))
    val = 2.0 * (term1 - term2)
    return math.sqrt(max(val, 0.0))

vec_asimov = np.vectorize(asimov_significance)

## The expected signficance
## If you use the binned one then we are in trouble
## The binned expected signficance should only be used for
## getting a rough estimate of the final significance of the scores distributions
## and not be used as an optimization objective
def get_counts_and_errors(model, X, y, w, nbins = 30, weighted = True):
    y_pred_prob = model.predict_proba(X)[:, 1]
    bins = np.linspace(0, 1.0, nbins + 1)
    if weighted == False:
        w = np.array([1] * len(X), dtype=np.float32)
    counts_sig, _ = np.histogram(y_pred_prob[y==1], bins=bins, density=False, weights=w[y==1])
    counts_bkg, _ = np.histogram(y_pred_prob[y==0], bins=bins, density=False, weights=w[y==0])
    sum_w2_bkg, _ = np.histogram(y_pred_prob[y==0], bins=bins, density=False, weights=w[y==0]**2)
    errors_bkg = np.sqrt(sum_w2_bkg)
    return counts_sig, counts_bkg, errors_bkg, _

def expected_significance_binned(model, X, y, w, nbins = 30):
    counts_sig, counts_bkg, errors_bkg, _ = get_counts_and_errors(model, X, y, w, nbins = nbins)
    return np.sqrt(np.sum(vec_asimov(s=counts_sig, b=counts_bkg, b_err=errors_bkg)**2))

def print_bdt_histogram_bins(model, X, y, w, nbins = 30):
    np.set_printoptions(precision=2, suppress=False)
    y_pred_prob = model.predict_proba(X)[:, 1]
    bins = np.linspace(0, 1.0, nbins + 1)
    counts_sig, _ = np.histogram(y_pred_prob[y==1], bins=bins, density=False, weights=w[y==1])
    counts_bkg, _ = np.histogram(y_pred_prob[y==0], bins=bins, density=False, weights=w[y==0])
    print(f"  > Binned signal histogram values:")
    print(counts_sig)
    print(f"  > Binned background histogram values")
    print(counts_bkg)
    print(f"  > Bin edges:")
    print(_)

## A stable version of the expted signficance
def amsasimov(s_in, b_in):
    """
    asimov significance arXiv:1007.1727 eq. 97 (reduces to s/sqrt(b) if s<<b)
    """
    s=np.copy(s_in)
    b=np.copy(b_in)
    s=np.where( (b_in == 0) , 0., s_in)
    b=np.where( (b_in == 0) , 1., b)

    ams = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    ams = np.where((s < 0) | (b < 0), np.nan, ams)  # nan if unphysical values
    if np.isscalar(s_in):
        return float(ams)
    else:
        return  ams

def significance_vscore(y_true, y_score, sample_weight=None):
    """
    Calculate the significance using the Asimov method.
    """
    if sample_weight is None:
        sample_weight = np.full(len(y_true), 1.0)

    # Define bins for y_score (100 bins from 0 to 1.0)
    bins = np.linspace(0, 1.0, 101)

    # Fills s and b weighted binned distributions
    s_hist, bin_edges = np.histogram(
        y_score[y_true == 1], bins=bins, weights=sample_weight[y_true == 1]
    )
    b_hist, bin_edges = np.histogram(
        y_score[y_true == 0], bins=bins, weights=sample_weight[y_true == 0]
    )

    # Compute cumulative sums (from the right!)
    s_cumul = np.cumsum(s_hist[::-1])[::-1]
    b_cumul = np.cumsum(b_hist[::-1])[::-1]

    # Compute significance
    significance = amsasimov(s_cumul, b_cumul)

    # Find the bin with the maximum significance
    max_value = np.max(significance)

    return significance


def significance_reduction_ratio(weights):
    """
    Compute the ratio R = sqrt( sum(w) / (sum(w) + sum(w^2)) )
    which quantifies how much the significance is reduced when
    using weighted events compared to unweighted events (w=1).

    Parameters:
        weights : array-like or float
            If array-like: per‑event weights.
            If float: assumed constant weight for all events (flat weights).

    Returns:
        R : float
            The reduction factor (0 < R ≤ 1).
            Multiply the naive significance s/sqrt(b) by R to get
            the approximate significance with weight‑induced variance.

    For flat weights (all w_i = w):
        R = 1 / sqrt(1 + w)
    """
    if np.isscalar(weights):
        w = float(weights)
        return 1.0 / np.sqrt(1.0 + w)
    else:
        w = np.asarray(weights)
        sum_w = np.sum(w)
        sum_w2 = np.sum(w**2)
        return np.sqrt(sum_w / (sum_w + sum_w2))

def significance_vscore_scorer(y_true, y_score, sample_weight=None):
    """
    Returns the maximum significance score for the given true labels and predicted scores.
    This function is designed to be used as a scorer in model evaluation.
    """
    return np.max(significance_vscore(y_true, y_score, sample_weight=sample_weight))


def weighted_mean(x, w):
    return np.sum(x * w) / np.sum(w)



def bkg_rejection_rate(sig_x, sig_w, bkg_x, bkg_w, sig_eff=0.7, cut_direction=">="):
    # basic onput validation
    if not 0.0 <= sig_eff <= 1.0:
        raise ValueError(f"sig_eff must be between 0 and 1, got {sig_eff}")

    if cut_direction not in (">=", "<="):
        raise ValueError(f"cut_direction must be '>=' or '<=', got '{cut_direction}'")

    # Sort signal and compute cumulative weights
    idx = np.argsort(sig_x)
    sorted_sig_x = sig_x[idx]
    cumw = np.cumsum(sig_w[idx])
    total = cumw[-1]
    
    # Target cumulative weight depending on cut direction
    if cut_direction == ">=":
        target_cum = (1 - sig_eff) * total   
    else:  # cut_direction == "<="
        target_cum = sig_eff * total        
    
    # Find position where cumulative weight reaches or exceeds target
    pos = np.searchsorted(cumw, target_cum)
    
    # If exactly at a point, return that score
    # else interpolate threshold linearly between adjacent points
    # if you do not want to liner interpolation,
    # you can just use thr = sorted_sig_x[pos]
    if pos == 0:
        thr = sorted_sig_x[0]
    elif pos >= len(sorted_sig_x):
        thr = sorted_sig_x[-1]
    else:
        # w_prev: cumulative weight just below target
        # w_curr: cumulative weight just above or equal to target
        w_prev, w_curr = cumw[pos-1], cumw[pos]
        if w_curr == w_prev:
            thr = sorted_sig_x[pos]
        else:
            frac = (target_cum - w_prev) / (w_curr - w_prev)
            thr = sorted_sig_x[pos-1] + frac * (sorted_sig_x[pos] - sorted_sig_x[pos-1])
    
    # Apply cut to background and compute rejection
    if cut_direction == ">=":
        pass_bkg = bkg_x >= thr
    else:
        pass_bkg = bkg_x <= thr
    
    bkg_eff = np.sum(bkg_w[pass_bkg]) / np.sum(bkg_w)
    bkg_rej = 1 - bkg_eff
    return bkg_rej, thr


def threshold_from_signal_rejection(scores, weights, rejection):
    """
    Returns threshold T such that:  fraction of weighted signal with score <= T  = rejection.
    
    Parameters
    ----------
    scores : array-like, signal scores
    weights : array-like, signal weights
    rejection : float, desired signal rejection rate (e.g., 0.0005 for 0.05%)
    
    Returns
    -------
    threshold : float
    """
    idx = np.argsort(scores)
    sorted_scores = scores[idx]
    cumw = np.cumsum(weights[idx])
    target = rejection * cumw[-1]
    pos = np.searchsorted(cumw, target)
    # If exactly at a point, return that score; else linearly interpolate
    if pos == 0:
        return sorted_scores[0]
    if pos >= len(sorted_scores):
        return sorted_scores[-1]
    # Linear interpolation between sorted_scores[pos-1] and sorted_scores[pos]
    w_prev = cumw[pos-1]
    w_curr = cumw[pos]
    if w_curr == w_prev:
        return sorted_scores[pos]
    frac = (target - w_prev) / (w_curr - w_prev)
    return sorted_scores[pos-1] + frac * (sorted_scores[pos] - sorted_scores[pos-1])

def threshold_from_signal_efficiency(scores, weights, efficiency):
    """
    efficiency = fraction of signal that passes cut (score > threshold)
    """
    rejection = 1 - efficiency
    return threshold_from_signal_rejection(scores, weights, rejection)


def significance_with_uncertainty(S, B, sigma_B):
    return S / np.sqrt(B + sigma_B**2) if (B + sigma_B**2) > 0 else 0.0


# ---------------------------------------------------------------- #
# HPO objective metrics
# ---------------------------------------------------------------- #

def weighted_ks(x1, w1, x2, w2):
    """Weighted two-sample KS distance (max |ECDF1 - ECDF2|)."""
    x1, w1 = np.asarray(x1, float), np.asarray(w1, float)
    x2, w2 = np.asarray(x2, float), np.asarray(w2, float)
    k1, k2 = w1 > 0, w2 > 0
    x1, w1, x2, w2 = x1[k1], w1[k1], x2[k2], w2[k2]
    if x1.size == 0 or x2.size == 0:
        return np.nan
    xs = np.unique(np.concatenate([x1, x2]))

    def ecdf(x, w):
        o = np.argsort(x, kind="mergesort")
        cw = np.cumsum(w[o]) / np.sum(w)
        i = np.searchsorted(x[o], xs, side="right") - 1
        return np.where(i >= 0, cw[np.clip(i, 0, None)], 0.0)

    return float(np.max(np.abs(ecdf(x1, w1) - ecdf(x2, w2))))


def asimov_z_curve(y, score, w, b_reg=0.01):
    """Cumulative Asimov Z for the cut `score >= t`, at every event's score.

    `b_reg` is the HBC-style regularisation of the background, as a fraction of
    the total background weight. Without it a cut that keeps signal and no
    background has b = 0, and `amsasimov` maps that to Z = 0 -- the deepest,
    most sensitive part of the tail would score zero instead of the highest.
    With it, Z stays finite and grows with the signal that passes the cut.

    Events with the *same* score are processed as one block. A cut can only be
    placed between distinct scores, so walking tied events one at a time would
    let a constant classifier "cut" after all its signal and before its
    background -- reporting a large significance for a model that separates
    nothing. Blocking the ties makes a constant score return Z = 0, which is the
    correct answer.
    """
    score = np.asarray(score, float)
    y = np.asarray(y)
    w = np.asarray(w, float)
    o = np.argsort(-score, kind="mergesort")
    s_sorted, y_sorted, w_sorted = score[o], y[o], w[o]
    # the last event of each run of equal scores: the only places a cut exists
    last = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), len(s_sorted) - 1]
    ys, ws = y_sorted[last], w_sorted[last]
    # weight of the whole block ending at each cut point
    cw = np.cumsum(w_sorted)
    cws = np.cumsum(np.where(y_sorted == 1, w_sorted, 0.0))
    s = cws[last]
    b = cw[last] - s
    # the regularisation is a fraction of the *total* background, not of the
    # background that survives the cut: otherwise a cut that keeps no
    # background at all gets br = 0 and reports an infinite Z
    b_tot = float(cw[-1] - cws[-1])
    br = b_reg * b_tot if b_tot > 0 else 0.0
    z = np.nan_to_num(amsasimov(s, b + br), nan=0.0)
    return z, s, b


def z_at_signal_efficiency(y, score, w, eff, b_reg=0.01):
    """Asimov Z at the cut that keeps a fraction `eff` of the weighted signal."""
    z, s, _ = asimov_z_curve(y, score, w, b_reg)
    if s[-1] <= 0:
        return 0.0
    k = int(np.searchsorted(s, eff * s[-1]))
    return float(z[min(k, len(z) - 1)])


def zmax(y, score, w, b_reg=0.01):
    """The max-cut Asimov significance: `z_unbinned`.

    Returns 0 when the score has no usable shape. Two failures have to be
    excluded, both of which report the significance of the *whole sample* rather
    than of a separation:

    * a constant score has a single cut point, where all the signal and all the
      background pass (10.7 in the toy case);
    * an inverted score (background above signal) has its best "cut" at the
      lowest score, where again everything passes.
    """
    g = shape_guards(y, score, w)
    if g["degenerate"] or g["inverted"]:
        return 0.0
    z, _, _ = asimov_z_curve(y, score, w, b_reg)
    return float(z.max()) if len(z) else 0.0


def separation_metrics(y, score, w):
    """Separation-power candidates. All computed on weighted events."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    w = np.asarray(w, float)
    sig, bkg = y == 1, y == 0
    if sig.sum() < 2 or bkg.sum() < 2:
        return {}
    z, s, b = asimov_z_curve(y, score, w)
    s_tot, b_tot = s[-1], b[-1]

    m_s = np.average(score[sig], weights=w[sig])
    m_b = np.average(score[bkg], weights=w[bkg])
    # a constant score has one cut point, where everything passes, and the
    # Asimov formula then reports the significance of the whole sample; an
    # inverted score has the same problem at the other end. Neither is a
    # separation measure, so both are gated exactly as in zmax.
    degenerate = bool(np.ptp(score) < 1e-9 or m_s <= m_b)
    out = {
        "z_unbinned": 0.0 if degenerate else float(z.max()),
        "z_eff10": z_at_signal_efficiency(y, score, w, 0.10),
        "z_eff30": z_at_signal_efficiency(y, score, w, 0.30),
        "auc": float(roc_auc_score(y, score, sample_weight=w)),
        "ks_sep": weighted_ks(score[sig], w[sig], score[bkg], w[bkg]),
        "mean_s": float(m_s),
        "mean_b": float(m_b),
        # diverges as the two means are pushed to 1 and 0
        "mean_sep": float(min(1.0 / max(m_b, 1e-3) + 1.0 / max(1.0 - m_s, 1e-3), 1e6)),
        # symmetric form, finite for a reversed classifier (that case is
        # rejected by the shape guards anyway)
        "mean_sep_sym": float(min(1.0 / max(m_b, 1e-3) + 1.0 / max(1.0 - m_s, 1e-3)
                                  if m_s > m_b else 0.0, 1e6)),
        "s_tot": float(s_tot), "b_tot": float(b_tot),
        "sb": float(s_tot / b_tot) if b_tot > 0 else np.inf,
    }
    # binned Asimov sum in quadrature, 30 bins, same background regularisation
    h_s, edges = np.histogram(score[sig], bins=30, range=(0, 1), weights=w[sig])
    h_b, _ = np.histogram(score[bkg], bins=30, range=(0, 1), weights=w[bkg])
    out["z_binned"] = float(np.sqrt(np.nansum(amsasimov(h_s, h_b + 0.01 * b_tot) ** 2)))
    return out


def shape_guards(y, score, w, n_bins=30):
    """Detect classifiers that are useless whatever their objective value.

    `degenerate`  no usable shape: everything in one bin, or the score is
                  constant, or (almost) no event is left in the tail.
    `inverted`    background scores above signal scores -> the sign is wrong.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    w = np.asarray(w, float)
    sig, bkg = y == 1, y == 0
    if sig.sum() < 2 or bkg.sum() < 2:
        return {"degenerate": True, "inverted": True, "n_nonempty_bins": 0}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(score, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(float)
    n_occ = int((counts > 0).sum())
    top_share = float(counts.max() / counts.sum()) if counts.sum() else 1.0
    same_bin = bool(n_occ <= 1)
    # a score that is constant to numerical precision is the same failure
    flat = bool(np.ptp(score) < 1e-9)
    degenerate = bool(n_occ <= 1 or flat or top_share > 0.995)

    m_s = np.average(score[sig], weights=w[sig])
    m_b = np.average(score[bkg], weights=w[bkg])
    inverted = bool(m_s < m_b)
    return {
        "degenerate": degenerate,
        "inverted": inverted,
        "same_bin": same_bin,
        "flat": flat,
        "n_nonempty_bins": n_occ,
        "top_bin_share": top_share,
        "mean_s": float(m_s), "mean_b": float(m_b),
    }


def overtraining_metrics(y_tr, s_tr, w_tr, y_va, s_va, w_va, n_bins=30):
    """Overtraining candidates: compare the train score shape with the held-out one.

    Weighted: the quantity that matters downstream is the *yield* distribution,
    so a bin with many low-weight events must not count as much as a bin with a
    few high-weight ones. (The p-value of a KS test on 21 copies of the same
    background event would be meaningless, which is why `collapse_background`
    removes the copies before this is called.)
    """
    y_tr, s_tr = np.asarray(y_tr).astype(int), np.asarray(s_tr, float)
    y_va, s_va = np.asarray(y_va).astype(int), np.asarray(s_va, float)
    w_tr, w_va = np.asarray(w_tr, float), np.asarray(w_va, float)
    out = {}
    for name, lab in (("sig", 1), ("bkg", 0)):
        a, wa = s_tr[y_tr == lab], w_tr[y_tr == lab]
        b, wb = s_va[y_va == lab], w_va[y_va == lab]
        if a.size < 2 or b.size < 2:
            out[f"ks_{name}"] = np.nan
            out[f"mean_gap_{name}"] = np.nan
            out[f"chi2_{name}"] = np.nan
            continue
        out[f"ks_{name}"] = weighted_ks(a, wa, b, wb)
        sd = np.std(a)
        out[f"mean_gap_{name}"] = abs(a.mean() - b.mean()) / sd if sd > 0 else np.nan
        # binned chi2 of the two normalised shapes, with their statistical errors
        h1, _ = np.histogram(a, bins=n_bins, range=(0, 1), weights=wa)
        h2, _ = np.histogram(b, bins=n_bins, range=(0, 1), weights=wb)
        e1, _ = np.histogram(a, bins=n_bins, range=(0, 1), weights=wa ** 2)
        e2, _ = np.histogram(b, bins=n_bins, range=(0, 1), weights=wb ** 2)
        if h1.sum() <= 0 or h2.sum() <= 0:
            out[f"chi2_{name}"] = np.nan
            continue
        p1, p2 = h1 / h1.sum(), h2 / h2.sum()
        var = e1 / max(h1.sum(), 1e-12) ** 2 + e2 / max(h2.sum(), 1e-12) ** 2
        ok = var > 0
        out[f"chi2_{name}"] = float(np.sum((p1[ok] - p2[ok]) ** 2 / var[ok])
                                    / max(ok.sum() - 1, 1))

    def _agg(key):
        v = [out[f"{key}_sig"], out[f"{key}_bkg"]]
        return float(np.nanmax(v)) if np.isfinite(v).any() else np.nan

    out["ks_gap"] = _agg("ks")
    out["mean_gap"] = _agg("mean_gap")
    out["chi2"] = _agg("chi2")
    return out


def smoothness(y, score, w, n_bins=30):
    """How spiky the score distribution is.

    A classifier whose output is a set of isolated spikes cannot be used to fit
    a template: the shape has no meaning between the spikes. Measured as the
    largest ratio between a bin and the mean of its two neighbours, on the
    weighted background (the dominant sample), with empty neighbours floored so
    that the ratio stays finite.
    """
    score = np.asarray(score, float)
    w = np.asarray(w, float)
    bkg = np.asarray(y).astype(int) == 0
    h, _ = np.histogram(score[bkg], bins=n_bins, range=(0, 1), weights=w[bkg])
    if h.sum() <= 0:
        return {"spike": np.inf, "empty_bins": n_bins, "n_bins": n_bins}
    p = h / h.sum()
    nbr = np.zeros_like(p)
    nbr[1:-1] = 0.5 * (p[:-2] + p[2:])
    nbr[0], nbr[-1] = p[1], p[-2]
    ok = nbr > 0
    spike = float(np.max(p[ok] / nbr[ok])) if ok.any() else np.inf
    return {"spike": spike, "empty_bins": int((h == 0).sum()), "n_bins": n_bins}


def evaluate_scores(y_tr, s_tr, w_tr, y_va, s_va, w_va):
    """All HPO metrics for one trained model, in a single dict.

    Computed once per trial so that every objective variant can be compared
    offline on the same pool of models.
    """
    m = separation_metrics(y_va, s_va, w_va)
    m.update(shape_guards(y_va, s_va, w_va))
    m.update({f"ot_{k}": v for k, v in overtraining_metrics(y_tr, s_tr, w_tr,
                                                            y_va, s_va, w_va).items()})
    m.update({f"sm_{k}": v for k, v in smoothness(y_va, s_va, w_va).items()})
    # same operating point on the training sample: the d-AMS idea of the PSO
    # paper, expressed as a relative loss so that the 80/20 split does not bias it
    z_tr10 = z_at_signal_efficiency(y_tr, s_tr, w_tr, 0.10)
    z_va10 = m.get("z_eff10", 0.0)
    m["dz_eff10"] = float(max(0.0, (z_tr10 - z_va10) / max(z_va10, 1e-6)))
    m["z_eff10_train"] = float(z_tr10)
    return m


def hpo_objective(m, sep="z_unbinned", over="ks_gap", A=1.0, B=1.0, guard=-1e6):
    """`A * x - B * y`, with the degenerate-shape guards.

    Returns `guard` (a large negative number, not merely a small one) when the
    classifier has no usable shape, so that such points are never selected.
    """
    if not m:
        return guard
    if m.get("degenerate") or m.get("inverted"):
        return guard
    x = m.get(sep)
    y = m.get(f"ot_{over}", m.get(over))
    if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
        return guard
    return float(A * x - B * y)

def significance_above_cut(y_true, bdt_scores, weights=None, cut=0.90, sigma_B=10.0):
    """
    Returns significance (S/√(B+σ_B²)) for events with bdt_scores >= cut.

    Parameters:
    - bdt_scores : array-like, BDT output scores (e.g., probability)
    - y_true     : array-like, 1 for signal, 0 for background
    - cut        : float, threshold on BDT score (e.g., 0.90)
    - weights    : array-like (optional), per-event weights. If None, all events have weight 1.
    - sigma_B    : systematic uncertainty on the total background estimate (scalar)
    """
    pass_cut = np.array(bdt_scores) >= cut
    y = np.array(y_true)
    
    if weights is None:
        weights = np.ones_like(bdt_scores, dtype=float)
    else:
        weights = np.array(weights)
    
    S = np.sum(weights[(y == 1) & pass_cut])
    B = np.sum(weights[(y == 0) & pass_cut])
    
    return significance_with_uncertainty(S, B, sigma_B)
