"""Self-check for the search algorithms in search.py. Run: python test_search.py

This tests the *mechanics* every algorithm must satisfy (the ask/tell contract,
bounds, movement, monotonic best), plus one quality guard: PSO must beat blind
guessing on a known 7-D optimum. Algorithm quality at a real budget is what the
study measures on 6144 fits, not what a 160-evaluation unit test can resolve --
at that budget GA is genuinely no better than random, so no assertion is made
about it here.
"""
import numpy as np
from search import Space, make

RANGES = {
    "max_depth": (3, 8, False),
    "learning_rate": (0.01, 0.3, True),
    "subsample": (0.5, 1.0, False),
    "colsample_bytree": (0.5, 1.0, False),
    "reg_alpha": (1e-4, 10.0, True),
    "reg_lambda": (1e-2, 100.0, True),
    "gamma": (0.0, 10.0, False),
}
space = Space(RANGES)
assert space.dim == 7

# decode is exact at the ends of the cube and respects the ranges
lo, hi = space.decode(np.zeros(7)), space.decode(np.ones(7))
assert lo["max_depth"] == 3 and hi["max_depth"] == 8, (lo, hi)
assert abs(lo["learning_rate"] - 0.01) < 1e-12 and abs(hi["learning_rate"] - 0.3) < 1e-9
assert isinstance(lo["max_depth"], int) and isinstance(hi["max_depth"], int)
# a log parameter must be log-spaced, not linear
mid = space.decode(np.full(7, 0.5))
assert abs(mid["reg_alpha"] - np.sqrt(1e-4 * 10.0)) < 1e-9, mid["reg_alpha"]

TARGET = np.array([0.7, 0.3, 0.9, 0.2, 0.5, 0.6, 0.1])


def f(u):
    return -np.sum((np.asarray(u) - TARGET) ** 2, axis=1)


N_BATCH, N_ITER = 8, 20
ALGOS = ["random", "tpe", "pso", "ga"]


def run(name, seed):
    """Run one algorithm to the full budget; return (best, best_x, history)."""
    alg = make(name, space, seed=seed, n_batch=N_BATCH, n_iter=N_ITER)
    best, best_x, hist = -np.inf, None, []
    for _ in range(N_ITER):
        pts = alg.ask(N_BATCH)
        assert pts.shape == (N_BATCH, 7), (name, pts.shape)
        assert np.all(pts >= 0) and np.all(pts <= 1), (name, "point outside the cube")
        sc = f(pts)
        alg.tell(pts, sc)
        hist.append(float(sc.max()))
        if sc.max() > best:
            best, best_x = float(sc.max()), pts[np.argmax(sc)]
    return best, best_x, hist


for name in ALGOS:
    best, best_x, _ = run(name, seed=0)
    assert np.isfinite(best) and best_x is not None, name
    assert np.linalg.norm(best_x - TARGET) < 1.5, (name, best_x, TARGET)

# PSO's global best must be monotone and the swarm must actually move
alg = make("pso", space, seed=0, n_batch=N_BATCH, n_iter=N_ITER)
hist, x0 = [], None
for it in range(N_ITER):
    pts = alg.ask(N_BATCH)
    if x0 is None:
        x0 = pts.copy()
    alg.tell(pts, f(pts))
    hist.append(alg.global_best_s)
assert all(b >= a - 1e-12 for a, b in zip(hist, hist[1:])), hist
assert hist[-1] > hist[0], hist
x1 = alg.ask(N_BATCH)
assert np.max(np.abs(x1 - x0)) > 1e-6, "the swarm did not move"

# GA must keep its population inside the cube even with mutation
alg = make("ga", space, seed=0, n_batch=N_BATCH, n_iter=N_ITER)
for it in range(N_ITER):
    pts = alg.ask(N_BATCH)
    alg.tell(pts, f(pts))
assert np.all(alg.pop >= 0) and np.all(alg.pop <= 1)

# quality guard: PSO must beat blind guessing, averaged over seeds so the check
# does not depend on one lucky draw
rng = np.random.RandomState(1)
blind = f(rng.rand(N_BATCH * N_ITER, 7)).max()
pso_mean = np.mean([run("pso", s)[0] for s in range(5)])
assert pso_mean > blind, (pso_mean, blind)
print(f"PSO mean best {pso_mean:.3f} vs blind {blind:.3f}")

print("all search self-checks passed")
