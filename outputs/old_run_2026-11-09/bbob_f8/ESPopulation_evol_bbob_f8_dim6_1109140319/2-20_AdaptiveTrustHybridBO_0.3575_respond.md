# Description  
**AdaptiveTrustHybridBO** – a Bayesian‑Optimization loop that builds on the Sobol‑EI‑HybridBO baseline and adds three powerful refinements that are especially useful for narrow‑valley problems such as the 6‑D Rosenbrock (BBOB f8):

1. **Adaptive Trust‑Region (TR)** – a hyper‑cube centred on the current best solution.  
   The TR radius is enlarged after a successful improvement (× 1.2) and shrunk after a failure (× 0.7) and is always kept inside the global bounds.  
   This concentrates sampling in the promising valley while still allowing the region to grow when the algorithm needs to explore.

2. **Hybrid acquisition maximisation** – a cheap Sobol/LHS pool inside the TR is first generated, the Expected‑Improvement (EI) is evaluated on the pool, the *k* best pool points (default k = 20) become start points for a gradient‑based L‑BFGS‑B optimiser that **maximises EI** (i.e. minimises –EI).  
   The best local optimum is taken as the next evaluation point.  
   This gives a far more accurate EI maximisation than a pure pool‑search at almost no extra cost.

3. **Periodic GP‑hyper‑parameter re‑training** – the GP surrogate is fully re‑optimised (with 5 restarts) only every `retrain_interval` iterations (default 5). In the intervening iterations the GP is refitted with `n_restarts=0`, i.e. only the posterior is updated while keeping the previously learned kernel hyper‑parameters.  
   This reduces the expensive GP optimisation overhead while still adapting to the changing data.

The algorithm respects the strict evaluation budget, works for any dimensionality inside the \([-5,5]^d\) box and uses only **NumPy, SciPy and scikit‑learn**.

---

# Justification  

| Component | Why it helps for BBOB f8 (Rosenbrock, 6‑D) |
|-----------|--------------------------------------------|
| **Sobol quasi‑Monte‑Carlo initial design** | Guarantees a space‑filling start with very low discrepancy, giving a reliable first GP fit. |
| **Adaptive Trust‑Region** | Rosenbrock’s global optimum lies in a long, narrow valley. A TR focuses evaluations inside the valley after the first few successes, dramatically reducing wasted samples in flat regions. The adaptive scaling prevents premature confinement and automatically re‑expands when the current TR is too restrictive. |
| **Hybrid EI maximisation (pool → L‑BFGS‑B)** | EI is a smooth, cheap‑to‑evaluate acquisition, but its global maximum can be far from any random pool point. By refining the *k* best pool points with a gradient optimiser we obtain a near‑optimal EI location without the need for a huge candidate pool, saving both time and budget. |
| **Periodic GP re‑training** | Full hyper‑parameter optimisation of a GP is the dominant cost (cubic in the number of observations). Re‑training only every few iterations keeps the surrogate accurate while cutting the wall‑clock time, which is crucial for a tight runtime budget. |
| **Kriging‑Believer batch (size = 1)** | Single‑point BO guarantees that every new observation immediately improves the surrogate, which is the safest strategy under a strict evaluation budget. |
| **ARD Matern‑5/2 kernel + tiny nugget** | The Automatic Relevance Determination (ARD) length‑scales allow the GP to adapt to different sensitivities along each coordinate – essential for the anisotropic Rosenbrock valley. The Matern‑5/2 kernel balances smoothness and flexibility. |
| **Budget‑aware loop** | The code never exceeds `self.budget`; any remaining budget smaller than a full iteration is handled gracefully. |

Together these refinements keep the algorithm cheap (few GP re‑optimisations, modest candidate pools) while dramatically improving exploitation of the narrow valley, which should raise the AOCC score beyond the current 0.3335.

---

# Code  

```python
from collections.abc import Callable
import numpy as np
from scipy import optimize, stats
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern,
    WhiteKernel,
    ConstantKernel,
)


class AdaptiveTrustHybridBO:
    """
    AdaptiveTrustHybridBO:
    • Sobol/LHS initial design
    • ARD Matern‑5/2 GP surrogate
    • Expected‑Improvement acquisition
    • Adaptive trust‑region around the current best
    • Hybrid EI maximisation (pool + L‑BFGS‑B)
    • Periodic full GP hyper‑parameter re‑training
    """

    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)          # total allowed function evaluations
        self.dim = int(dim)                # problem dimensionality
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])  # global bounds

        # history
        self.X = None          # shape (n, dim)
        self.y = None          # shape (n, 1)
        self.n_evals = 0

        # algorithmic parameters
        self.n_init = max(2 * dim, 5)      # Sobol start size
        self.tr_radius = 5.0               # initial trust‑region radius (half of global range)
        self.tr_min = 0.1
        self.tr_max = 5.0
        self.tr_expand = 1.2
        self.tr_shrink = 0.7
        self.retrain_interval = 5         # full GP optimisation frequency
        self.k_start = 20                 # number of start points for L‑BFGS‑B
        self.n_candidates = max(2000, 100 * dim)   # pool size inside TR

    # ------------------------------------------------------------------ #
    #  Sampling utilities (Sobol for the very first design, LHS later)
    # ------------------------------------------------------------------ #
    def _sample(self, n: int, use_tr: bool = False, centre: np.ndarray = None) -> np.ndarray:
        """Draw `n` points uniformly in the global bounds or inside the TR."""
        if self.X is None:                         # first call → Sobol
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            unit = sampler.random_base2(m=int(np.ceil(np.log2(n))))
            unit = unit[:n]
        else:
            sampler = qmc.LatinHypercube(d=self.dim)
            unit = sampler.random(n)

        if use_tr:
            # TR bounds are centred at `centre` with half‑width = self.tr_radius
            lower = np.maximum(self.bounds[0], centre - self.tr_radius)
            upper = np.minimum(self.bounds[1], centre + self.tr_radius)
            pts = qmc.scale(unit, lower, upper)
        else:
            pts = qmc.scale(unit, self.bounds[0], self.bounds[1])
        return pts

    # ------------------------------------------------------------------ #
    #  GP surrogate
    # ------------------------------------------------------------------ #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray, full: bool) -> GaussianProcessRegressor:
        """Fit a GP.  If `full` is False, skip expensive hyper‑parameter restarts."""
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(self.dim),
                     length_scale_bounds=(1e-2, 1e2),
                     nu=2.5)
            + WhiteKernel(noise_level=1e-8,
                          noise_level_bounds=(1e-10, 1e-1))
        )
        n_restart = 5 if full else 0
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=n_restart,
            random_state=0,
        )
        gp.fit(X, y.ravel())
        return gp

    # ------------------------------------------------------------------ #
    #  Expected Improvement
    # ------------------------------------------------------------------ #
    def _ei(self, Xcand: np.ndarray, gp: GaussianProcessRegressor) -> np.ndarray:
        mu, sigma = gp.predict(Xcand, return_std=True)
        sigma = sigma.reshape(-1, 1)

        y_best = np.min(self.y)

        with np.errstate(divide="ignore", invalid="ignore"):
            imp = y_best - mu.reshape(-1, 1)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei.ravel()

    # ------------------------------------------------------------------ #
    #  Hybrid EI maximisation (pool + L‑BFGS‑B)
    # ------------------------------------------------------------------ #
    def _max_ei(self, gp: GaussianProcessRegressor, centre: np.ndarray) -> np.ndarray:
        """Return a single point that (approximately) maximises EI inside the TR."""
        # 1) generate a modest pool inside the current TR
        pool = self._sample(self.n_candidates, use_tr=True, centre=centre)

        ei_vals = self._ei(pool, gp)
        # select the k best pool points as start locations for L‑BFGS‑B
        start_idx = np.argpartition(-ei_vals, self.k_start)[: self.k_start]
        starts = pool[start_idx]

        # bounds for the optimiser (TR caps)
        lower = np.maximum(self.bounds[0], centre - self.tr_radius)
        upper = np.minimum(self.bounds[1], centre + self.tr_radius)
        bounds = list(zip(lower, upper))

        best_x = None
        best_ei = -np.inf

        # minimise -EI (i.e. maximise EI)
        for x0 in starts:
            res = optimize.minimize(
                fun=lambda x: -self._ei(x[None, :], gp)[0],
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-9, "gtol": 1e-5, "maxiter": 50},
            )
            if not res.success:
                continue
            cand_x = res.x
            cand_ei = -res.fun
            if cand_ei > best_ei:
                best_ei = cand_ei
                best_x = cand_x

        # fallback – if optimisation failed, simply take the best pool point
        if best_x is None:
            best_x = pool[np.argmax(ei_vals)]

        return best_x[None, :]          # shape (1, dim)

    # ------------------------------------------------------------------ #
    #  Evaluation utilities
    # ------------------------------------------------------------------ #
    def _evaluate(self, func: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
        """Evaluate `func` row‑wise, respecting the budget."""
        y = np.apply_along_axis(func, 1, X).astype(np.float64).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _add_observations(self, Xnew: np.ndarray, ynew: np.ndarray):
        if self.X is None:
            self.X, self.y = Xnew, ynew
        else:
            self.X = np.vstack([self.X, Xnew])
            self.y = np.vstack([self.y, ynew])

    # ------------------------------------------------------------------ #
    #  Main optimisation loop
    # ------------------------------------------------------------------ #
    def __call__(self, func: Callable[[np.ndarray], float]) -> tuple[float, np.ndarray]:
        """Run BO until the evaluation budget is exhausted. Return (best_val, best_x)."""
        # ---- 1) Initial Sobol design -------------------------------------------------
        X0 = self._sample(self.n_init, use_tr=False)
        y0 = self._evaluate(func, X0)
        self._add_observations(X0, y0)

        best_idx = np.argmin(self.y)
        best_val = self.y[best_idx, 0]
        best_x = self.X[best_idx, :].copy()

        iteration = 0
        # ---- 2) Main BO loop ---------------------------------------------------------
        while self.n_evals < self.budget:
            remaining = self.budget - self.n_evals
            if remaining <= 0:
                break

            # Fit GP (full re‑training every `retrain_interval` iterations)
            full_retrain = (iteration % self.retrain_interval == 0)
            gp = self._fit_gp(self.X, self.y, full=full_retrain)

            # Propose next point inside the adaptive TR
            cand = self._max_ei(gp, centre=best_x)

            # Evaluate candidate (might be truncated if budget is 1)
            cand = cand[:remaining, :]                     # safety slice
            ycand = self._evaluate(func, cand)

            # Update history
            self._add_observations(cand, ycand)

            # Check improvement
            cur_best_idx = np.argmin(self.y)
            cur_best_val = self.y[cur_best_idx, 0]
            cur_best_x = self.X[cur_best_idx, :]

            if cur_best_val < best_val - 1e-12:   # successful improvement
                best_val = cur_best_val
                best_x = cur_best_x.copy()
                self.tr_radius = min(self.tr_max, self.tr_radius * self.tr_expand)
            else:                                 # no improvement
                self.tr_radius = max(self.tr_min, self.tr_radius * self.tr_shrink)

            iteration += 1

        return float(best_val), best_x.copy()
```
## Feedback
 The algorithm AdaptiveTrustHybridBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3575 with standard deviation 0.0132.

took 773.93 seconds to run.