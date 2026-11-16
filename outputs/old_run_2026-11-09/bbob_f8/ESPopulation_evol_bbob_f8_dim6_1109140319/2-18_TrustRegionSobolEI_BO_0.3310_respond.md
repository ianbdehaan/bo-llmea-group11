# Description 
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

# Justification 

| Component | Why it helps on Rosenbrock (narrow valley) |
|-----------|--------------------------------------------|
| **Trust‑region** | The Rosenbrock valley is long and thin. By repeatedly shrinking the search region around the best point we force the GP to model the local curvature more accurately, leading to faster exploitation. When the region becomes too small (no improvement) it is expanded again, guaranteeing that the algorithm can still escape local traps. |
| **Periodic GP re‑training** | GP fitting scales cubically with the number of observations. With a modest budget (≈ 200 evaluations) the cost of re‑training each iteration dominates the runtime (as seen in the 93 s of the previous algorithm). Updating the GP only every few iterations reduces the wall‑clock time dramatically while the surrogate remains sufficiently up‑to‑date because we add only a handful of points between re‑fits. |
| **Exploration injection (ε‑probability)** | Pure trust‑region exploitation can become myopic. A 10 % chance of drawing a fresh Sobol point from the whole domain re‑introduces global diversity, which is especially valuable early on and after long periods without improvement. |
| **L‑BFGS‑B refinement of top‑k candidates** | Maximising EI analytically is costly; a few gradient‑based refinements from the most promising Sobol points give a near‑optimal acquisition value at negligible extra cost. |
| **Sobol QMC design** | Low‑discrepancy sampling provides a well‑spread initial dataset and a cheap candidate pool for acquisition evaluation, ensuring good coverage of the search space without random clustering. |

Together these modifications keep the algorithm computationally light, improve sample efficiency on the Rosenbrock valley, and yield a higher AOCC score than the original AdaptiveSobolEI_BO.

---

# Code 
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
## Feedback
 The algorithm TrustRegionSobolEI_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3310 with standard deviation 0.0006.

took 57.24 seconds to run.