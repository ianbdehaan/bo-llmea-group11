You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of BBOB Function 8 (Rosenbrock, 6D). Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AdaptiveTrustHybridBO: 0.3575, 773.93 seconds, **AdaptiveTrustHybridBO** – a Bayesian‑Optimization loop that builds on the Sobol‑EI‑HybridBO baseline and adds three powerful refinements that are especially useful for narrow‑valley problems such as the 6‑D Rosenbrock (BBOB f8):

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


- TrustRegionAdaptiveSobolEI_BO: 0.3370, 73.87 seconds, **  
**TrustRegionAdaptiveSobolEI_BO** – a Bayesian‑Optimization algorithm that builds on the strong points of the original *AdaptiveSobolEI_BO* and adds three complementary mechanisms that are especially helpful for the narrow‑valley structure of the 6‑D Rosenbrock function (BBOB f8):

1. **Adaptive Trust‑Region (TR)** – a hyper‑cube centred on the current best solution.  
   The TR radius is enlarged after a successful improvement and shrunk after a failure, keeping the search focussed inside the valley while still allowing occasional global jumps.

2. **Hybrid candidate pool** – a small global Sobol pool (exploration) + a larger Sobol pool confined to the TR (exploitation).  
   This gives a cheap yet balanced mixture of long‑range and short‑range proposals.

3. **Periodic GP‑hyper‑parameter re‑training & warm‑start** – the GP is re‑optimised only every *retrain_interval* iterations, otherwise the previous kernel hyper‑parameters are kept and only the training data are updated.  This cuts the overhead dramatically without sacrificing model quality.

The rest of the pipeline (initial Sobol design, ARD Matern‑5/2 GP surrogate, Expected‑Improvement acquisition, L‑BFGS‑B refinement of the top‑k EI points, strict budget handling) is identical to the baseline, but the added TR logic yields a tighter exploitation of the Rosenbrock valley and improves the overall AOCC score.

---

**
- SobolEIHybridBO: 0.3335, 91.23 seconds, **  
**Sobol‑EI‑HybridBO** – a Bayesian‑Optimization framework that combines low‑discrepancy Sobol quasi‑Monte‑Carlo (QMC) initialization, a scalable Gaussian‑Process surrogate (Matern‑5/2 kernel) and an Expected‑Improvement (EI) acquisition.  After a cheap global Sobol design, the algorithm repeatedly (i) fits the GP, (ii) draws a large pool of candidate points with cheap Latin‑Hypercube sampling, (iii) evaluates EI on the pool, and (iv) selects a small batch (default = 1) of the highest‑EI points.  The batch is chosen by a simple “Kriging‑believer” update (the GP mean of already‑chosen points is used as a surrogate for their unknown function values) to keep the selection cheap while still providing a modest diversification effect.  The whole loop stops exactly at the user‑provided evaluation budget.

---

**
- TrustRegionSobolEI_BO: 0.3310, 57.24 seconds, **TrustRegionSobolEI_BO** – a Bayesian‑Optimization algorithm that builds on the proven AdaptiveSobolEI_BO framework and adds three powerful refinements that are especially useful for narrow‑valley problems such as the 6‑D Rosenbrock (BBOB f8):

1. **Adaptive trust‑region** centred on the current best solution.  
   The region is a hyper‑cube whose half‑width `r` is enlarged after a successful improvement (× 1.2) and shrunk after a failure (× 0.7).  
   This focuses the expensive GP‑based search inside the promising valley while still allowing global exploration when the region grows.

2. **Periodic GP re‑training** (default every 5 iterations).  
   Re‑fitting a GP is the most costly step; we keep the previous surrogate for a few iterations and only update the training data, dramatically reducing overhead without sacrificing model quality.

3. **Exploration‑exploitation balance** via a small random‑sampling injection (probability `ε = 0.1`).  
   With probability ε we ignore the trust‑region and propose a fresh Sobol point, preventing premature convergence to a sub‑optimal valley.

All other ingredients remain the same: Sobol quasi‑Monte‑Carlo initialization, an ARD Matern‑5/2 GP surrogate, Expected‑Improvement (EI) acquisition, and a cheap L‑BFGS‑B refinement of the best Sobol candidates.  
The algorithm respects the strict evaluation budget, works for any dimensionality inside the \([-5,5]^d\) box, and uses only NumPy, SciPy and scikit‑learn.

---




The selected solution to update is:
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


With code:
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
The algorithm AdaptiveTrustHybridBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3575 with standard deviation 0.0132.

took 773.93 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

