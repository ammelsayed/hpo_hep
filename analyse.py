"""Compare objective variants and search algorithms from the model pool.

Pure post-processing of the CSVs written by run_pool.py: no model is trained
here, so every variant is compared on exactly the same models.

The question is not "which variant has the highest number" (all of them pick a
good model out of 48) but "does a higher objective really mean a better model
on data the objective never saw". That is measured by the rank correlation
between the objective and the held-out truth, and by the quality of the single
model the objective would have selected.

Usage:
  python analyse.py --pooldir results/pool --out results/analysis
"""
import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from study_data import objective_from_row

# separation x and overtraining y candidates to be compared
SEP = ["z_unbinned", "z_eff10", "z_eff30", "z_binned", "auc", "mean_sep", "ks_sep"]
OVER = ["ks_gap", "mean_gap", "chi2", "dz_eff10"]
WEIGHTS = [(1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]
GUARD = -1e6


def load_pool(pooldir):
    files = sorted(glob.glob(os.path.join(pooldir, "*.csv")))
    if not files:
        raise SystemExit(f"no pool CSVs in {pooldir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def score_column(df, sep, over, A, B, prefix=""):
    """Vectorised version of objective_from_row, for speed on 6k rows.

    `dz_eff10` is stored without the `ot_` prefix (it is already a relative
    train-vs-test loss), the other overtraining measures carry it.
    """
    bad = df[f"{prefix}degenerate"] | df[f"{prefix}inverted"]
    x = df[f"{prefix}{sep}"].astype(float)
    col = f"{prefix}{over}" if f"{prefix}{over}" in df.columns else f"{prefix}ot_{over}"
    y = df[col].astype(float)
    s = A * x - B * y
    s = s.where(np.isfinite(s) & ~bad, GUARD)
    return s.values


def truth_column(df, sep="z_unbinned", over="ks_gap", A=1.0, B=1.0):
    return score_column(df, sep, over, A, B, prefix="te_")


def evaluate_variant(df, sep, over, A, B, truth_sep="z_unbinned", truth_over="ks_gap"):
    """Rank correlation with the held-out truth, and the selected model's quality.

    Per run (channel, algo, seed) the best trial by the objective is selected and
    judged on DF2. `regret` is the DF2 significance lost relative to the best
    model the run actually contains, so 0 means the objective picked the oracle.
    """
    s_all = score_column(df, sep, over, A, B)
    t_all = truth_column(df, truth_sep, truth_over)
    ok = s_all > GUARD / 2
    rho = spearmanr(s_all[ok], t_all[ok]).statistic if ok.sum() > 10 else np.nan

    rows = []
    d = df.assign(_s=s_all, _t=t_all)
    for (ch, algo, seed), g in d.groupby(["channel", "algo", "seed"]):
        best = g.loc[g._s.idxmax()]
        z_oracle = float(g["te_z_unbinned"].max())
        z = float(best["te_z_unbinned"])
        rows.append(dict(channel=ch, algo=algo, seed=seed, z_truth=z,
                         ot_truth=float(best["te_ot_ks_gap"]),
                         regret=z_oracle - z, hit=z >= 0.95 * z_oracle,
                         degenerate=bool(best["te_degenerate"])))
    r = pd.DataFrame(rows)
    return dict(sep=sep, over=over, A=A, B=B, rho=rho,
                z_truth=r.z_truth.mean(), ot_truth=r.ot_truth.mean(),
                regret=r.regret.mean(), hit=r.hit.mean(),
                n_degenerate=int(r.degenerate.sum()), n_runs=len(r)), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooldir", default="results/pool")
    ap.add_argument("--out", default="results/analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load_pool(args.pooldir)
    print(f"pool: {len(df)} models, {df.channel.nunique()} channels, "
          f"{df.algo.nunique()} algorithms, {df.seed.nunique()} seeds")

    # spread of the truth: is there anything for an objective to choose between?
    print(f"held-out Z range across all models: {df.te_z_unbinned.min():.3f} .. "
          f"{df.te_z_unbinned.max():.3f} (median {df.te_z_unbinned.median():.3f})")
    print(f"degenerate models: {int(df.te_degenerate.sum())} / {len(df)}")

    # 1. objective variants
    res = []
    for sep, over, (A, B) in itertools.product(SEP, OVER, WEIGHTS):
        r, _ = evaluate_variant(df, sep, over, A, B)
        res.append(r)
    obj = pd.DataFrame(res).sort_values("rho", ascending=False)
    obj.to_csv(os.path.join(args.out, "objective_ranking.csv"), index=False)
    print("\n=== objective variants, ranked by rank-correlation with held-out truth ===")
    print(obj.head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== the same, ranked by the significance of the selected model ===")
    print(obj.sort_values("z_truth", ascending=False).head(12).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    # 2. algorithms under the best objective
    best = obj.iloc[0]
    print(f"\n=== algorithms under the best objective "
          f"({best.sep} - {best.B}*{best.over}) ===")
    alg_rows = []
    for algo, g in df.groupby("algo"):
        r, _ = evaluate_variant(g, best.sep, best.over, best.A, best.B)
        alg_rows.append(dict(algo=algo, **{k: v for k, v in r.items()
                                           if k not in ("sep", "over", "A", "B")}))
    alg = pd.DataFrame(alg_rows).sort_values("regret")
    alg.to_csv(os.path.join(args.out, "algorithm_ranking.csv"), index=False)
    print(alg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 3. convergence: best held-out Z found so far vs trials spent
    conv = []
    for (ch, algo, seed), g in df.groupby(["channel", "algo", "seed"]):
        g = g.sort_values("trial")
        conv.append(pd.DataFrame(dict(
            channel=ch, algo=algo, seed=seed, trial=g.trial.values,
            best_z=np.maximum.accumulate(g["te_z_unbinned"].values),
            wall=np.cumsum(g["wall_s"].values))))
    conv = pd.concat(conv, ignore_index=True)
    conv.to_csv(os.path.join(args.out, "convergence.csv"), index=False)

    # 4. single-metric sanity: correlation of every raw metric with the truth
    print("\n=== how well does each raw metric predict the held-out Z? ===")
    rows = []
    for c in ["z_unbinned", "z_eff10", "z_eff30", "z_binned", "auc", "mean_sep",
              "ks_sep", "sm_spike", "ot_ks_gap", "ot_mean_gap", "ot_chi2", "dz_eff10"]:
        if c not in df.columns:
            continue
        v = df[c].astype(float)
        ok = np.isfinite(v)
        if ok.sum() < 10:
            continue
        rows.append(dict(metric=c, rho=spearmanr(v[ok], df.te_z_unbinned[ok]).statistic,
                         rho_neg_ot=spearmanr(v[ok], df.te_ot_ks_gap[ok]).statistic))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "metric_correlation.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
