"""Hyper-parameter search algorithms used by the HPO study.

Every algorithm works on the unit cube [0, 1]^d and is given the same budget of
objective evaluations, so that they can be compared per unit of computing time.

The PSO and GA follow Tani et al., Eur. Phys. J. C 81 (2021) 170:
  * PSO: momentum + cognitive + social attraction, inertial weight decreasing
    linearly from 0.8 to 0.4, momenta initialised within a quarter of the range,
    particles clipped at the bounds with their momentum reset to zero.
  * GA: tournament selection, 1-point crossover, Gaussian mutation whose width
    decreases linearly to zero, elitism and culling.
"""
import numpy as np


class Space:
    """Hyper-parameter ranges: name -> (low, high, is_log)."""

    def __init__(self, ranges):
        self.names = list(ranges)
        self.low = np.array([ranges[k][0] for k in self.names], float)
        self.high = np.array([ranges[k][1] for k in self.names], float)
        self.log = np.array([bool(ranges[k][2]) for k in self.names])
        self.is_int = np.array([float(ranges[k][0]).is_integer() and
                                float(ranges[k][1]).is_integer() for k in self.names])

    @property
    def dim(self):
        return len(self.names)

    def decode(self, u):
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        v = self.low + u * (self.high - self.low)
        if self.log.any():
            lg = self.log
            v[lg] = np.exp(np.log(self.low[lg]) + u[lg] * (np.log(self.high[lg]) - np.log(self.low[lg])))
        out = {}
        for k, val, is_int in zip(self.names, v, self.is_int):
            out[k] = int(round(val)) if is_int else float(val)
        return out


class RandomSearch:
    """Quasi-random baseline: the same budget spent on uniform samples."""

    def __init__(self, space, seed=0, **kw):
        self.space, self.rng = space, np.random.RandomState(seed)

    def ask(self, n):
        return self.rng.rand(n, self.space.dim)

    def tell(self, points, scores):
        pass


class PSO:
    """Particle swarm optimisation (Tani et al., section 2)."""

    def __init__(self, space, seed=0, n_particles=10, n_iter=20,
                 c1=2.0, c2=2.0, w_start=0.8, w_end=0.4, n_informants=3):
        self.space = space
        self.n = n_particles
        self.rng = np.random.RandomState(seed)
        self.c1, self.c2 = c1, c2
        self.w_start, self.w_end = w_start, w_end
        self.n_inf = max(1, min(n_informants, n_particles))
        self.n_iter = max(n_iter, 1)
        self.x = self.rng.rand(n_particles, space.dim)
        # momenta within a quarter of the range (Tani et al., section 2)
        self.p = self.rng.uniform(-0.25, 0.25, size=(n_particles, space.dim))
        self.best_x = self.x.copy()
        self.best_s = np.full(n_particles, -np.inf)
        self.global_best_x = self.x[0].copy()
        self.global_best_s = -np.inf
        self.k = 0

    def ask(self, n):
        return self.x.copy()

    def tell(self, points, scores):
        scores = np.asarray(scores, float)
        better = scores > self.best_s
        self.best_s = np.where(better, scores, self.best_s)
        self.best_x[better] = points[better]

        # global best over a random subset ("informants") to avoid premature
        # convergence
        inf = self.rng.choice(self.n, size=self.n_inf, replace=False)
        j = inf[np.argmax(self.best_s[inf])]
        if self.best_s[j] > self.global_best_s:
            self.global_best_s = float(self.best_s[j])
            self.global_best_x = self.best_x[j].copy()

        w = self.w_start + (self.w_end - self.w_start) * (self.k / max(self.n_iter - 1, 1))
        r1 = self.rng.rand(self.n, self.space.dim)
        r2 = self.rng.rand(self.n, self.space.dim)
        force = self.c1 * r1 * (self.best_x - self.x) + self.c2 * r2 * (self.global_best_x - self.x)
        x_new = self.x + w * self.p + force
        # clip at the bounds, resetting the momentum of the clipped particles
        hit = (x_new < 0.0) | (x_new > 1.0)
        x_new = np.clip(x_new, 0.0, 1.0)
        p_new = x_new - self.x
        p_new[hit] = 0.0
        self.x, self.p = x_new, p_new
        self.k += 1


class GA:
    """Genetic algorithm (Tani et al., section 3), single population."""

    def __init__(self, space, seed=0, n_pop=10, n_iter=20, n_tour=3, p_tour=0.4,
                 p_mut=0.2, n_elite=2):
        self.space = space
        self.n = n_pop
        self.rng = np.random.RandomState(seed)
        self.n_tour, self.p_tour, self.p_mut = n_tour, p_tour, p_mut
        self.n_elite = max(1, min(n_elite, n_pop // 2))
        self.n_iter = max(n_iter, 1)
        self.k = 0
        self.pop = self.rng.rand(n_pop, space.dim)

    def _select(self, scores):
        """Tournament selection: highest score wins with probability p_tour."""
        cand = self.rng.choice(self.n, size=self.n_tour, replace=False)
        cand = cand[np.argsort(-scores[cand])]
        for c in cand:
            if self.rng.rand() < self.p_tour:
                return c
        return cand[-1]

    def ask(self, n):
        return self.pop.copy()

    def tell(self, points, scores):
        scores = np.asarray(scores, float)
        new = [points[i] for i in np.argsort(-scores)[:self.n_elite]]   # elitism
        width = 0.25 * (1.0 - self.k / max(self.n_iter - 1, 1))
        while len(new) < self.n:
            i, j = self._select(scores), self._select(scores)
            cut = self.rng.randint(1, self.space.dim) if self.space.dim > 1 else 0
            child = np.where(np.arange(self.space.dim) < cut, points[i], points[j])
            mut = self.rng.randn(self.space.dim) * width
            child = np.clip(child + (self.rng.rand(self.space.dim) < self.p_mut) * mut, 0.0, 1.0)
            new.append(child)
        self.pop = np.array(new)
        self.k += 1


class TPE:
    """Optuna's tree-structured Parzen estimator, in the same ask/tell form."""

    def __init__(self, space, seed=0, **kw):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.space = space
        self.study = optuna.create_study(direction="maximize",
                                         sampler=optuna.samplers.TPESampler(seed=seed))
        self.pending = []

    def ask(self, n):
        self.pending = [self.study.ask() for _ in range(n)]
        out = []
        for t in self.pending:
            u = []
            for k, lo, hi, lg, ii in zip(self.space.names, self.space.low,
                                         self.space.high, self.space.log, self.space.is_int):
                if ii:
                    v = t.suggest_int(k, int(lo), int(hi))
                    u.append((v - lo) / max(hi - lo, 1e-12))
                else:
                    v = t.suggest_float(k, float(lo), float(hi), log=bool(lg))
                    u.append((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if lg
                             else (v - lo) / max(hi - lo, 1e-12))
            out.append(u)
        return np.array(out)

    def tell(self, points, scores):
        for t, s in zip(self.pending, scores):
            self.study.tell(t, float(s))
        self.pending = []


def make(name, space, seed, n_batch, n_iter):
    if name == "random":
        return RandomSearch(space, seed)
    if name == "tpe":
        return TPE(space, seed)
    if name == "pso":
        return PSO(space, seed, n_particles=n_batch, n_iter=n_iter)
    if name == "ga":
        return GA(space, seed, n_pop=n_batch, n_iter=n_iter)
    raise ValueError(f"unknown algorithm {name!r}")
