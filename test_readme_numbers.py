"""Check every number quoted in the README against the committed results."""
import glob
import numpy as np
import pandas as pd

ok = True
def check(label, got, want, tol=0.02):
    global ok
    good = abs(got - want) <= tol * max(abs(want), 1e-9) + 0.006
    ok &= good
    print(f"{'OK ' if good else 'BAD'} {label:58s} readme={want:<9} data={got:.4f}")

pool = pd.read_csv("results/pool.csv.gz")
rep = pd.read_csv("results/replication.csv")
sub = pd.read_csv("results/sub/sub_studies.csv")

print("=== section 3.4 / 4.6 pool size ===")
check("pool models", len(pool), 6144, 0)
print(f"     channels={pool.channel.nunique()} algos={pool.algo.nunique()} "
      f"seeds={pool.seed.nunique()} fits/run={len(pool)//(pool.channel.nunique()*pool.algo.nunique()*pool.seed.nunique())}")

print("\n=== section 4.1 replication gains ===")
ctrl = pd.concat([pd.read_csv(f"results/replication_{c}.csv").assign(channel=c)
                  for c in rep.channel.unique()])
ctrl = ctrl.groupby("channel").te_z_unbinned.mean().rename("ctrl")
m = rep.join(ctrl, on="channel")
m["gain"] = m.rev_z - m.ctrl
for sep, over, A, B, want in [
        ("z_unbinned", "ks_gap", 1.0, 0.0, 0.037),
        ("z_binned", "ks_gap", 1.0, 1.0, 0.031),
        ("z_unbinned", "ks_gap", 1.0, 1.0, 0.008),
        ("z_unbinned", "ks_gap", 1.0, 2.0, -0.019),
        ("mean_sep", "ks_gap", 1.0, 1.0, -0.022),
        ("auc", "ks_gap", 1.0, 1.0, -0.068),
        ("z_eff10", "ks_gap", 1.0, 1.0, -0.097)]:
    s = m[(m.sep == sep) & (m.over == over) & (m.A == A) & (m.B == B)]
    check(f"{sep} - {B}*{over}", s.gain.mean(), want, 0.06)

print("\n=== section 4.2 gap statistics ===")
check("mean gap train-vs-val", pool.ot_ks_gap.mean(), 0.143, 0.05)
check("mean gap train-vs-DF2", pool.te_ot_ks_gap.mean(), 0.110, 0.05)
r = [np.corrcoef(d.ot_ks_gap.rank(), d.te_z_unbinned.rank())[0, 1]
     for _, d in pool.groupby("channel")]
check("rho(ks_gap, held-out Z)", float(np.mean(r)), 0.27, 0.15)
check("fraction optimistic (te<va)",
      float((pool.te_z_unbinned < pool.z_unbinned).mean()), 0.0, 0.0)
check("median te/va", float((pool.te_z_unbinned / pool.z_unbinned).median()), 1.96, 0.03)

print("\n=== section 4.4 early stopping ===")
e = sub[sub.study == "early_stopping"]
check("median best iteration", float(pool.best_iteration.median()), 161, 0.05)
check("models at the 4000 cap", float((pool.n_trees >= 4000).sum()), 0, 0)
check("99th pct best iteration", float(pool.best_iteration.quantile(0.99)), 1065, 0.05)
check("still improving at cap=100",
      float(e[e.cap == 100].still_improving.mean()), 0.857, 0.02)
check("still improving at cap=500",
      float(e[e.cap == 500].still_improving.mean()), 0.0, 0.0)

print("\n=== section 4.5 regimes ===")
r = pd.read_csv("results/overfit_regimes.csv")
for reg, want_z, want_spike in [("recommended", 1.690, 17.3),
                                ("memorise", 1.967, 473.0)]:
    d = r[r.regime == reg]
    check(f"{reg} held-out Z", d.z_te.mean(), want_z, 0.03)
    check(f"{reg} spikiness", d.spike.mean(), want_spike, 0.15)
check("recommended gap", r[r.regime == "recommended"].gap_te.mean(), 0.164, 0.06)
check("memorise gap", r[r.regime == "memorise"].gap_te.mean(), 0.284, 0.06)

print("\n=== section 4.7 min_child_weight ===")
mcw = sub[sub.study == "mcw"]
for pol, wz, wot, wsp in [("suggested", 1.514, 0.079, 7.3),
                          ("default", 1.567, 0.144, 13.6),
                          ("tiny", 1.578, 0.162, 18.4)]:
    d = mcw[mcw.policy == pol]
    check(f"{pol} Z", d.z_te.mean(), wz, 0.02)
    check(f"{pol} gap", d.ot_te.mean(), wot, 0.10)
    check(f"{pol} spike", d.spike.mean(), wsp, 0.15)
check("suggested mcw min", mcw.mcw_suggested.min(), 30.0, 0.10)
check("suggested mcw max", mcw.mcw_suggested.max(), 130.6, 0.10)

print("\n=== section 4.8 / 4.9 cuts and recipe ===")
ca = pd.read_csv("results/recipe_1L_cut_ablation.csv")
piv = ca.pivot_table(index="channel", columns="cuts", values="z_te")
gain = (piv[True] - piv[False]) / piv[False] * 100
check("mean 1L cut gain (%)", float(gain.mean()), 43.0, 0.05)
for ch, want in [("1L_eJ", 1.260), ("1L_eJJ", 2.241), ("1L_muJ", 1.361),
                 ("1L_muJJ", 3.055)]:
    check(f"recipe {ch} Z", float(piv.loc[ch, True]), want, 0.02)
for ch, want in [("1L_eJ", 0.858), ("1L_eJJ", 1.682), ("1L_muJ", 0.931),
                 ("1L_muJJ", 2.096)]:
    check(f"no-cut {ch} Z", float(piv.loc[ch, False]), want, 0.02)

rec = pd.concat([pd.read_csv("results/recipe_rest.csv"),
                 pd.read_csv("results/recipe_1L.csv")])
check("recipe channels", len(rec), 20, 0)
for ch, want in [("2SSL_emuJ", 7.948), ("2OSL_emuJ", 9.961), ("3L_emumuJ", 5.378),
                 ("2OSL_eeJ", 1.650), ("2OSL_mumuJ", 1.549)]:
    check(f"recipe {ch} Z", float(rec[rec.channel == ch].z_te.iloc[0]), want, 0.02)

print("\n=== section 4.6 algorithm table ===")
alg = pool.loc[pool.groupby(["channel", "algo", "seed"]).score.idxmax()]
for a, want in [("tpe", 1.779), ("pso", 1.748), ("ga", 1.783), ("random", 1.774)]:
    check(f"{a} held-out Z", alg[alg.algo == a].te_z_unbinned.mean(), want, 0.02)

print("\n=== section 4.3 valsize ===")
for ch, w20, w100 in [("4L_eeee", 0.335, 0.702), ("3L_eeeJ", -0.075, 0.559)]:
    d = pd.read_csv(f"results/valsize_{ch}.csv")
    from scipy.stats import spearmanr
    check(f"{ch} rho at 20%", spearmanr(d.z_va_20, d.z_te).statistic, w20, 0.10)
    check(f"{ch} rho at 100%", spearmanr(d.z_va_100, d.z_te).statistic, w100, 0.05)

print("\n=== section 6: cross-evaluated distributions, all 26 channels ===")
ce = pd.read_csv("results/cross_eval/summary.csv")
check("cross_eval channels", len(ce.channel.unique()), 26, 0)
check("cross_eval configs", len(ce.config.unique()), 3, 0)
z = ce.pivot_table(index="channel", columns="config", values="z")
a = ce.pivot_table(index="channel", columns="config", values="auc")
check("mean Z, baseline", float(z.baseline.mean()), 3.00, 0.02)
check("mean Z, units fixed", float(z.baseline_units.mean()), 4.17, 0.02)
check("mean Z, HPO", float(z.hpo.mean()), 4.45, 0.02)
# the nine channels where the shipped defaults never split
never_split = ["3L_eeeJ", "3L_eemuJ", "3L_emumuJ", "3L_mumumuJ",
               "4L_eeee", "4L_eeemu", "4L_eemumu", "4L_emumumu", "4L_mumumumu"]
check("channels with baseline Z = 0", int((z.baseline == 0).sum()), 9, 0)
for ch in never_split:
    assert z.loc[ch, "baseline"] == 0.0, (ch, z.loc[ch, "baseline"])
    assert abs(a.loc[ch, "baseline"] - 0.5) < 1e-9, (ch, a.loc[ch, "baseline"])
check("HPO beats baseline on", int((z.hpo > z.baseline).sum()), 26, 0)
check("HPO beats units-fixed on", int((z.hpo > z.baseline_units).sum()), 21, 0)
# the quoted per-channel rows
for ch, want in [("1L_eJ", 1.78), ("1L_eJJ", 3.25), ("1L_muJJ", 3.59),
                 ("2OSL_eeJ", 1.98), ("2OSL_emuJ", 13.85), ("2OSL_mumuJ", 1.88),
                 ("2SSL_eeJ", 6.69), ("2SSL_mumuJ", 8.54), ("3L_eee", 2.54),
                 ("3L_eeeJ", 3.44), ("3L_eemuJ", 6.61), ("3L_emumuJ", 7.48),
                 ("3L_mumumuJ", 4.43), ("4L_eeee", 0.97), ("4L_mumumumu", 1.17)]:
    check(f"cross_eval HPO Z {ch}", float(z.loc[ch, "hpo"]), want, 0.02)
# the quoted AUC rows
for ch, col, want in [("3L_eeeJ", "baseline", 0.5000), ("4L_eeee", "baseline", 0.5000),
                      ("1L_eJJ", "baseline", 0.9686), ("1L_eJJ", "hpo", 0.9848),
                      ("2OSL_mumuJ", "baseline", 0.9047),
                      ("2OSL_mumuJ", "hpo", 0.9439),
                      ("2SSL_eeJ", "baseline_units", 0.9710)]:
    check(f"cross_eval AUC {ch} {col}", float(a.loc[ch, col]), want, 0.01)
# the 4L losses quoted in the text
for ch in ["4L_eeee", "4L_eeemu", "4L_eemumu", "4L_emumumu", "4L_mumumumu"]:
    assert z.loc[ch, "hpo"] < z.loc[ch, "baseline_units"], ch

print("\n" + ("ALL README NUMBERS CHECK OUT" if ok else "SOME NUMBERS DISAGREE"))
