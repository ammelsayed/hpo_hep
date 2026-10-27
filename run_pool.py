"""Generate the pool of trained models used by the whole study.

For each (channel, algorithm, seed) a fixed number of models is trained with
`n_estimators` fixed high and early stopping on the validation log-loss. Every
model is evaluated once on the validation half (which the HPO objective may use)
and once on DF2 (held out from all of the study's decisions). The resulting
metrics are written to a CSV; the objective comparison and the algorithm
comparison are then pure post-processing of that table, so the expensive part
runs exactly once.

Usage:
  python run_pool.py --channels 4L_eeee 3L_eeeJ --algos random tpe pso ga \
                     --budget 40 --seeds 0 1 2 --outdir results/pool
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from search import Space, make
from study_data import Channel, SPACE

SPACE_OBJ = Space(SPACE)


def run_one(channel, algo, seed, budget, batch, out, device, nthread):
    ch = Channel(channel, device=device, nthread=nthread, seed=42)
    alg = make(algo, SPACE_OBJ, seed=seed, n_batch=batch,
               n_iter=max(budget // batch, 1))
    rows, n_eval, best = [], 0, -np.inf
    while n_eval < budget:
        pts = alg.ask(batch)
        scores = []
        for u in pts:
            params = SPACE_OBJ.decode(u)
            t0 = time.time()
            model, fit_s = ch.fit(params)
            row = ch.evaluate(model)
            row.update(params)
            row.update(channel=channel, algo=algo, seed=seed, trial=n_eval,
                       fit_s=round(fit_s, 2), wall_s=round(time.time() - t0, 2),
                       point=u.tolist())
            row["score"] = row["z_unbinned"] - row["ot_ks_gap"] \
                if not (row["degenerate"] or row["inverted"]) else -1e6
            rows.append(row)
            scores.append(row["score"])
            best = max(best, row["score"])
            n_eval += 1
        alg.tell(pts, np.array(scores))
        print(f"[{channel}/{algo}/s{seed}] {n_eval}/{budget} best={best:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--algos", nargs="+", default=["random", "tpe", "pso", "ga"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nthread", type=int, default=1)
    ap.add_argument("--outdir", default="results/pool")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for channel in args.channels:
        for algo in args.algos:
            for seed in args.seeds:
                out = os.path.join(args.outdir, f"{channel}__{algo}__s{seed}.csv")
                if os.path.exists(out):
                    print(f"skip {out}", flush=True)
                    continue
                run_one(channel, algo, seed, args.budget, args.batch, out,
                        args.device, args.nthread)


if __name__ == "__main__":
    main()
