# HPO for significance and overtraining in BDT-based searches

Hyper-parameter optimisation (HPO) of the XGBoost classifiers used in the
multi-lepton search, with an objective that rewards **separation power** and
punishes **overtraining** at the same time.

---

## 1. The problem

The analysis trains one boosted decision tree per signal region to separate a
heavy resonance (signal) from the SM backgrounds. Two things go wrong with
default or hand-picked hyper-parameters:

1. **The separation is not optimal for a discovery.** A BDT that maximises a
   generic metric (AUC, accuracy, log-loss) does not necessarily maximise the
   expected significance `Z` of the search. With `S/B ~ 10^-4` (1L) to `O(1)`
   (3L/4L), what matters is the *shape* of the score distribution in the
   high-score tail where the cut will be placed, not the global ranking.
2. **The classifier overtrain. ** With small signal samples (4L channels have
   O(10^3) signal events) and large backgrounds, a flexible model memorises its
   training events. The training score distribution then separates far better
   than the one on statistically independent data, the final significance is
   over-estimated, and the shape used to extract the result is not the shape
   that the data will show.

Both effects are controlled by the same hyper-parameters (depth, learning rate,
subsampling, regularisation), so they cannot be tuned independently: the
"best" separation is usually obtained by the most overtrained model. HPO must
therefore optimise a **single objective that combines the two**.

## 2. What we optimise

The objective is a scalar to be maximised

```
score = A * x  -  B * y
```

* `x` — separation power, larger is better.
* `y` — amount of overtraining, smaller is better.
* `A`, `B` — weights that put `x` and `y` on a common scale.

### 2.1 Separation `x` (candidates to be compared)

| id | definition |
|----|------------|
| `z_unbinned` | max over cuts of the unbinned Asimov significance `Z(s,b)` on the evaluation sample |
| `z_eff10` / `z_eff30` | Asimov significance at a fixed signal efficiency (10 %, 30 %) |
| `z_binned` | sum in quadrature of the per-bin Asimov significance (30 bins) |
| `auc` | weighted ROC AUC (weight-free ranking power, ignores the tail) |
| `mean_sep` | `1/mean(b) + 1/(1-mean(s))` on weighted mean scores; diverges for perfect separation |
| `ks_sep` | weighted KS distance between the signal and background score distributions |

### 2.2 Overtraining `y` (candidates to be compared)

| id | definition |
|----|------------|
| `dz` | `|Z_train - Z_eval|` at the same operating point (the `d-AMS` idea of Ref. [1]) |
| `ks_gap` | max of the train-vs-eval weighted KS distances for signal and background |
| `chi2` | binned `chi2/ndf` of train vs eval score distributions |
| `mean_gap` | `|mean_train - mean_eval|` in units of the score standard deviation |

### 2.3 Degenerate-shape guards

Some hyper-parameter points produce a classifier that cannot be used at all.
The objective is set to a large negative value (not just "small") when

* all signal **and** background scores fall into a single bin (no separation at
  all, e.g. `best_iteration = 0`), or
* the ordering is inverted: background scores near 1 and signal scores near 0.

## 3. How the study is organised

### 3.1 Data

Each region provides two statistically independent half-samples,
`sig_DF1/DF2` and `bkg_DF1/DF2`. A model is **trained on DF1 and evaluated on
DF2**, so the overtraining measure compares a genuinely held-out sample with
the training sample. Background events are duplicated once per signal mass
hypothesis (21 copies of the same event, with the weight split between the
copies); the overtraining metrics are computed on de-duplicated events,
otherwise the KS test is fed 21 copies of every point and its p-value is
meaningless.

Datasets used for the *method comparison* (small, fast): `4L_eeee`,
`4L_eeemu`, `4L_eemumu`, `4L_emumumu`, `4L_mumumumu`, `3L_eeeJ`, `3L_eee`.
The winner is then applied to the remaining regions.

### 3.2 Algorithms compared

| id | method |
|----|--------|
| `random` | quasi-random search over the same ranges (baseline) |
| `tpe` | Optuna TPE, i.e. the tree-structured Parzen estimator already used in `bdt_methods.py` |
| `pso` | particle swarm optimisation, as in Ref. [1] |
| `ga` | genetic algorithm, as in Ref. [1] |

All algorithms get the **same number of model fits**, so the comparison is
"quality per unit of computing time", not "quality per iteration".

### 3.3 Hyper-parameters

`max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`,
`reg_lambda`, `gamma`, and `min_child_weight`. The last one is the subject of a
dedicated sub-study: it can be (i) left at the XGBoost default, (ii) fixed from
the event weights (`suggest_min_child_weight`), or (iii) searched. It is a
hessian, not an event count, so its natural scale is set by the weights.

`n_estimators` is *not* tuned: it is set high and the ensemble is stopped by
early stopping. The claim that a well-chosen parameter set keeps improving the
validation metric until the cap is reached is checked explicitly.

### 3.4 What "best" means

The objective is only a proxy. Every configuration is finally judged on the
**held-out** sample (DF2) by

* the Asimov significance of the DF2 score distribution,
* the train-vs-DF2 overtraining measures,
* the smoothness of the score distribution (no spikes, no empty tail bins).

An objective is only useful if a high objective value implies a high held-out
significance **and** a small train/test gap.

## 4. GPU

The server has four A10 GPUs, but XGBoost was silently training on the CPU:
the installed wheel (3.4.1) is built against CUDA 13.3 while the driver
supports CUDA 12.7, and XGBoost reacts to "no usable GPU" with a warning and a
fallback to CPU. See Section 6 for the fix.

## 5. Results

*(to be filled in after the study)*

## 6. References

[1] L. Tani, D. Rand, C. Veelken, M. Kadastik, *Evolutionary algorithms for
hyperparameter optimization in machine learning for application in high energy
physics*, Eur. Phys. J. C 81 (2021) 170,
[doi:10.1140/epjc/s10052-021-08950-y](https://doi.org/10.1140/epjc/s10052-021-08950-y).

[2] G. Cowan, K. Cranmer, E. Gross, O. Vitells, *Asymptotic formulae for
likelihood-based tests of new physics*, Eur. Phys. J. C 71 (2011) 1554.
