"""Merge the per-channel cross_eval summaries into one table."""
import glob
import os

import pandas as pd

indir = "results/cross_eval"
files = sorted(glob.glob(os.path.join(indir, "summary__*.csv")))
if not files:
    raise SystemExit("no summary__*.csv; re-run cross_eval.py per channel")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.drop_duplicates(subset=["channel", "config"])
df.to_csv(os.path.join(indir, "summary.csv"), index=False)
pd.set_option("display.width", 250)
print(f"{len(df)} rows, {df.channel.nunique()} channels, {df.config.nunique()} configs")
print(df.pivot_table(index="channel", columns="config", values="z").round(3).to_string())
