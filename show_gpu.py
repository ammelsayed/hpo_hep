"""Demonstrate the GPU diagnosis end to end, on the version installed in the venv.

Run twice: once with the venv's xgboost 3.4.1 (CUDA 13.3 wheel, driver 12.7) and
once with the 3.2.0 wheel that matches the driver. The point is that the first
case silently trains on the CPU while reporting `device="cuda"`.
"""
import json
import sys
import time

import numpy as np
import xgboost as xgb
from xgboost import build_info

print(f"xgboost {xgb.__version__}, built against CUDA {build_info()['CUDA_VERSION']}")
X = np.random.rand(200_000, 20).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
for dev in ("cpu", "cuda"):
    t = time.time()
    m = xgb.XGBClassifier(n_estimators=100, max_depth=6, device=dev,
                          tree_method="hist", verbosity=0)
    m.fit(X, y, verbose=False)
    real = json.loads(m.get_booster().save_config())["learner"]["generic_param"]["device"]
    print(f"  requested={dev:5s} actually_used={real:7s} {time.time() - t:6.2f}s")
