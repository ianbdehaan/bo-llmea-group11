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
**TrustRegionSobolEI_BO** – a Bayesian‑Optimization algorithm that builds on the proven AdaptiveSobolEI_BO framework and adds three powerful refinements that are especially useful for narrow‑valley problems such as the 6‑D Rosenbrock (BBOB f8):

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


With code:
```python
from collections.abc import Callable
import numpy as np
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class TrustRegionSobolEI_BO:
    """
    TrustRegionSobolEI_BO – Bayesian optimisation with:
      * Sobol QMC initial design,
      * ARD Matern‑5/2 GP surrogate,
      * Expected‑Improvement acquisition,
      * Adaptive trust‑region centred on the incumbent,
      * Periodic GP re‑training,
      * Small random‑exploration injection,
      * L‑BFGS‑B refinement of the best Sobol candidates.
    """

    def __init__(self, budget: int, dim: int,
                 trust_radius_init: float = 5.0,
                 trust_shrink: float = 0.7,
                 trust_expand: float = 1.2,
                 min_trust_radius: float = 0.5,
                 max_trust_radius: float = 5.0,
                 retrain_interval: int = 5,
                 explore_prob: float = 0.1):
        self.budget = budget                # total allowed function evaluations
        self.dim = dim
        self.bounds = np.column_stack((np.full(dim, -5.0),
                                       np.full(dim, 5.0)))   # (dim,2)

        # trust‑region parameters
        self.trust_radius = trust_radius_init
        self.trust_shrink = trust_shrink
        self.trust_expand = trust_expand
        self.min_trust_radius = min_trust_radius
        self.max_trust_radius = max_trust_radius

        self.retrain_interval = retrain_interval
        self.explore_prob = explore_prob

        # data containers
        self.X: np.ndarray = None           # (n_samples, dim)
        self.y: np.ndarray = None           # (n_samples, 1)
        self.n_evals = 0

        # initial design size (5 points per dimension, but ≤ half the budget)
        self.n_init = min(5 * dim, max(2, budget // 2))

        # internal GP cache
        self._gp: GaussianProcessRegressor = None
        self._last_fit_iter = -1            # iteration index when GP was (re)trained

    # ------------------------------------------------------------------ #
    # Sampling utilities
    # ------------------------------------------------------------------ #
    def _sobol_samples(self, n: int) -> np.ndarray:
        """Generate `n` Sobol points scaled to the problem bounds."""
        m = int(np.ceil(np.log2(n)))               # Sobol works with powers of two
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=0)
        unit = sampler.random_base2(m=m)[:n]
        return qmc.scale(unit, self.bounds[:, 0], self.bounds[:, 1])

    def _sobol_in_trust_region(self, n: int, centre: np.ndarray) -> np.ndarray:
        """Sobol points uniformly drawn inside the current trust‑region."""
        # compute region bounds and clip to global limits
        low = np.maximum(centre - self.trust_radius, self.bounds[:, 0])
        high = np.minimum(centre + self.trust_radius, self.bounds[:, 1])
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=None)
        unit = sampler.random(n)
        return qmc.scale(unit, low, high)

    # ------------------------------------------------------------------ #
    # Surrogate model
    # ------------------------------------------------------------------ #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """Fit a GP with ARD Matern‑5/2 kernel + tiny white noise."""
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(self.dim),
                           length_scale_bounds=(1e-2, 1e2), nu=2.5)
                  + WhiteKernel(noise_level=1e-8,
                                noise_level_bounds=(1e-10, 1e-1)))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      n_restarts_optimizer=5,
                                      normalize_y=True,
                                      random_state=0)
        gp.fit(X, y.ravel())
        return gp

    # ------------------------------------------------------------------ #
    # Expected Improvement
    # ------------------------------------------------------------------ #
    def _ei(self, X: np.ndarray, gp: GaussianProcessRegressor,
            y_best: float) -> np.ndarray:
        """Vectorised Expected Improvement for minimisation."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu = mu.reshape(-1, 1)

        with np.errstate(divide='ignore', invalid='ignore'):
            imp = y_best - mu                     # improvement (minimisation)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei.ravel()

    # ------------------------------------------------------------------ #
    # Acquisition optimisation (refinement)
    # ------------------------------------------------------------------ #
    def _refine_candidates(self, cand: np.ndarray, gp: GaussianProcessRegressor,
                           y_best: float, n_refine: int = 3) -> np.ndarray:
        """Refine the `n_refine` best Sobol candidates with L‑BFGS‑B on –EI."""
        ei_vals = self._ei(cand, gp, y_best)
        top_idx = np.argpartition(-ei_vals, n_refine)[:n_refine]
        refined = []

        bounds = tuple(self.bounds[i] for i in range(self.dim))

        for x0 in cand[top_idx]:
            res = minimize(lambda x: -self._ei(x.reshape(1, -1), gp, y_best)[0],
                           x0=x0,
                           method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 20, 'ftol': 1e-6})
            refined.append(res.x)

        return np.vstack(refined)

    # ------------------------------------------------------------------ #
    # Evaluation handling (budget‑aware)
    # ------------------------------------------------------------------ #
    def _evaluate(self, func: Callable[[np.ndarray], float],
                  X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate `func` on rows of X respecting the remaining budget."""
        remaining = self.budget - self.n_evals
        if X.shape[0] > remaining:
            X = X[:remaining]

        y_vals = np.apply_along_axis(func, 1, X).reshape(-1, 1)
        self.n_evals += X.shape[0]
        return X, y_vals

    def _append_data(self, X_new: np.ndarray, y_new: np.ndarray):
        """Add newly evaluated points to the internal dataset."""
        if self.X is None:
            self.X, self.y = X_new, y_new
        else:
            self.X = np.vstack((self.X, X_new))
            self.y = np.vstack((self.y, y_new))

    # ------------------------------------------------------------------ #
    # Main optimisation interface
    # ------------------------------------------------------------------ #
    def __call__(self,
                 func: Callable[[np.ndarray], float]) -> tuple[float, np.ndarray]:
        """
        Run the BO loop.
        Returns (best_y, best_x) where best_y is the minimal observed value.
        """

        # ---------- initial Sobol design ----------
        X0 = self._sobol_samples(self.n_init)
        X0, y0 = self._evaluate(func, X0)
        self._append_data(X0, y0)

        best_idx = int(np.argmin(self.y))
        best_y = float(self.y[best_idx, 0])
        best_x = self.X[best_idx].copy()

        iteration = 0
        while self.n_evals < self.budget:
            # -------------------------------------------------
            # (re)fit GP every `retrain_interval` iterations
            # -------------------------------------------------
            if (iteration - self._last_fit_iter) >= self.retrain_interval or self._gp is None:
                self._gp = self._fit_gp(self.X, self.y)
                self._last_fit_iter = iteration
            gp = self._gp

            # -------------------------------------------------
            # decide whether to explore globally or stay inside trust‑region
            # -------------------------------------------------
            if np.random.rand() < self.explore_prob:
                # pure global Sobol pool
                pool = self._sobol_samples(min(3000, 10 * self.dim * self.n_init))
            else:
                # Sobol points confined to the current trust‑region
                pool = self._sobol_in_trust_region(
                    min(3000, 10 * self.dim * self.n_init), centre=best_x)

            # -------------------------------------------------
            # refine a few promising candidates
            # -------------------------------------------------
            refined = self._refine_candidates(pool, gp, best_y, n_refine=3)

            # -------------------------------------------------
            # select the point with maximal EI (i.e. minimal -EI)
            # -------------------------------------------------
            ei_vals = self._ei(refined, gp, best_y)
            next_x = refined[np.argmax(ei_vals)].reshape(1, -1)

            # -------------------------------------------------
            # evaluate the black‑box function (budget‑aware)
            # -------------------------------------------------
            next_x, next_y = self._evaluate(func, next_x)
            self._append_data(next_x, next_y)

            # -------------------------------------------------
            # update incumbent and adapt trust‑region radius
            # -------------------------------------------------
            cur_best_idx = int(np.argmin(self.y))
            cur_best_y = float(self.y[cur_best_idx, 0])

            if cur_best_y < best_y - 1e-12:          # successful improvement
                best_y, best_x = cur_best_y, self.X[cur_best_idx].copy()
                self.trust_radius = min(self.trust_radius * self.trust_expand,
                                        self.max_trust_radius)
            else:                                   # no improvement
                self.trust_radius = max(self.trust_radius * self.trust_shrink,
                                        self.min_trust_radius)

            iteration += 1

        return best_y, best_x

```
The algorithm TrustRegionSobolEI_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3310 with standard deviation 0.0006.

took 57.24 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

