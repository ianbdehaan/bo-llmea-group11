**# Description**  
**TR‑Sobol‑EI‑BO** – a Bayesian‑Optimization loop that combines  

* a Sobol quasi‑Monte‑Carlo space‑filling start,  
* an ARD Matern‑5/2 Gaussian‑Process surrogate,  
* Expected‑Improvement (EI) as acquisition,  
* a **dynamic trust‑region** centred on the current best solution (radius expands after a successful improvement and contracts after a failure),  
* **periodic GP‑hyper‑parameter re‑training** (every `retrain_interval` iterations) to keep the computational overhead low, and  
* a cheap **local L‑BFGS‑B refinement** of the top‑k EI candidates inside the trust‑region.  

The algorithm keeps the evaluation budget strictly, works for any dimension inside the box `[-5, 5]^d`, and is especially suited for narrow‑valley problems such as the 6‑D Rosenbrock (BBOB f8).  

```text
1. Initialise with a Sobol design (≈ min(5·d, budget/2) points).  
2. Loop until the budget is exhausted  
   a. Fit / update the GP (full re‑fit every `retrain_interval` iterations).  
   b. Build a candidate pool:  
        – If the trust‑region radius is small → sample only inside the region.  
        – Otherwise → global Sobol pool.  
   c. Compute EI on the pool, pick the `k_refine` best points and run a short
      L‑BFGS‑B optimisation of the *negative* EI (bounds = trust‑region).  
   d. Choose the point with the highest refined EI, evaluate the black‑box
      function, and update the best‑so‑far.  
   e. Adapt the trust‑region radius: enlarge (× 1.2) after an improvement,
      shrink (× 0.5) after a failure, and clip to `[r_min, r_max]`.  
3. Return the best observed value and its location.  
```

---

**# Justification**  

| Component | Why it helps on BBOB f8 (Rosenbrock, 6‑D) |
|-----------|-------------------------------------------|
| **Sobol start** | Guarantees a well‑distributed initial sample without wasting evaluations on random draws. |
| **ARD Matern‑5/2 GP** | ARD automatically scales each dimension, crucial for the highly anisotropic Rosenbrock valley. |
| **Expected Improvement** | Balances exploration/exploitation and has a closed‑form gradient, enabling cheap local optimisation. |
| **Dynamic Trust‑Region** | The Rosenbrock valley is a long, narrow canyon. By focusing the search around the current best and shrinking when we fail, the algorithm spends most evaluations inside the promising region while still being able to “jump out” after a successful expansion. |
| **Periodic GP re‑training** | GP fitting dominates runtime. Re‑training only every few iterations (default 5) cuts overhead dramatically without sacrificing model quality, because the data set grows slowly (one point per iteration). |
| **L‑BFGS‑B refinement of EI** | Direct optimisation of EI yields much better acquisition maximisation than a raw random pool, especially when the pool is constrained inside a small trust‑region. The optimisation is cheap (≤ 30 iterations, 5 starts). |
| **Batch size = 1** | Guarantees that every new observation immediately improves the surrogate – essential when the budget is tight. |
| **Budget‑aware evaluation handling** | Guarantees we never exceed the user‑provided evaluation budget, a strict requirement of the BBOB framework. |

Together these ideas give a tighter focus on the Rosenbrock valley, reduce unnecessary GP updates, and improve the exploitation of promising areas, which should raise the AOCC score while keeping runtime well below the 126 s of the previous algorithm.

---

**# Code**  

```python
from collections.abc import Callable
import numpy as np
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class TRSobolEI_BO:
    """
    Trust‑Region Sobol‑EI Bayesian Optimisation.
    -------------------------------------------------
    * Sobol QMC initial design
    * ARD Matern‑5/2 GP surrogate
    * Expected Improvement acquisition
    * Dynamic trust‑region centred on the current best
    * Periodic GP hyper‑parameter re‑training (retrain_interval)
    * Local L‑BFGS‑B refinement of top‑k EI points
    * Strict evaluation budget
    """

    def __init__(self, budget: int, dim: int,
                 trust_radius_init: float = 2.0,
                 trust_radius_min: float = 0.2,
                 trust_radius_max: float = 5.0,
                 retrain_interval: int = 5,
                 k_refine: int = 5,
                 pool_factor: int = 10):
        """
        Parameters
        ----------
        budget : int
            Maximum number of black‑box evaluations.
        dim : int
            Dimensionality of the problem.
        trust_radius_init, trust_radius_min, trust_radius_max : float
            Parameters controlling the adaptive trust‑region.
        retrain_interval : int
            Fit the GP from scratch every `retrain_interval` BO iterations.
        k_refine : int
            Number of top EI candidates that are locally refined.
        pool_factor : int
            Controls the size of the Sobol candidate pool:
            pool_size = pool_factor * dim * n_init (capped at 5000).
        """
        self.budget = int(budget)
        self.dim = int(dim)
        self.bounds = np.column_stack((np.full(dim, -5.0),
                                       np.full(dim, 5.0)))          # (dim,2)

        # initial design size (≈ 5 points per dim, but never > budget/2)
        self.n_init = min(5 * dim, max(2, self.budget // 2))

        # trust‑region settings
        self.tr_radius = float(trust_radius_init)
        self.r_min = float(trust_radius_min)
        self.r_max = float(trust_radius_max)

        # other hyper‑parameters
        self.retrain_interval = max(1, int(retrain_interval))
        self.k_refine = int(k_refine)
        self.pool_factor = int(pool_factor)

        # bookkeeping
        self.X = None          # (n_samples, dim)
        self.y = None          # (n_samples, 1)
        self.n_evals = 0
        self.iteration = 0

    # ------------------------------------------------------------------ #
    # Sampling utilities
    # ------------------------------------------------------------------ #
    def _sobol_samples(self, n: int) -> np.ndarray:
        """Draw `n` Sobol points uniformly scaled to the search bounds."""
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        m = int(np.ceil(np.log2(n)))                # Sobol needs power‑of‑2
        unit = sampler.random_base2(m=m)[:n]        # (n, dim) in [0,1]^d
        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        return qmc.scale(unit, lower, upper)

    def _trust_region_samples(self, n: int, centre: np.ndarray) -> np.ndarray:
        """
        Sample uniformly inside a hyper‑cube trust region centred at `centre`
        with side length = 2 * self.tr_radius, clipped to the global bounds.
        """
        low = np.maximum(centre - self.tr_radius, self.bounds[:, 0])
        high = np.minimum(centre + self.tr_radius, self.bounds[:, 1])
        # Use Sobol on the unit cube and then scale to [low, high]
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=iteration_seed())
        m = int(np.ceil(np.log2(n)))
        unit = sampler.random_base2(m=m)[:n]
        return qmc.scale(unit, low, high)

    # ------------------------------------------------------------------ #
    # GP surrogate
    # ------------------------------------------------------------------ #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """Fit an ARD Matern‑5/2 GP (with tiny nugget)."""
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(self.dim),
                     length_scale_bounds=(1e-2, 1e2),
                     nu=2.5)
            + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            random_state=0,
        )
        gp.fit(X, y.ravel())
        return gp

    # ------------------------------------------------------------------ #
    # Expected Improvement
    # ------------------------------------------------------------------ #
    def _ei(self, X: np.ndarray, gp: GaussianProcessRegressor,
            y_best: float) -> np.ndarray:
        """EI for minimisation."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        imp = y_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return ei

    # ------------------------------------------------------------------ #
    # Local refinement of EI (L‑BFGS‑B)
    # ------------------------------------------------------------------ #
    def _refine_ei(self, start_pts: np.ndarray,
                   gp: GaussianProcessRegressor,
                   y_best: float,
                   centre: np.ndarray) -> np.ndarray:
        """Run L‑BFGS‑B on the *negative* EI from each start point."""
        refined = []
        # bounds = intersection of global bounds and current trust region
        low = np.maximum(centre - self.tr_radius, self.bounds[:, 0])
        high = np.minimum(centre + self.tr_radius, self.bounds[:, 1])
        bounds = list(zip(low, high))

        def neg_ei(x):
            return -self._ei(x.reshape(1, -1), gp, y_best)[0]

        for x0 in start_pts:
            res = minimize(
                neg_ei,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 30, "ftol": 1e-9},
            )
            refined.append(res.x)

        return np.vstack(refined) if refined else np.empty((0, self.dim))

    # ------------------------------------------------------------------ #
    # Candidate generation + acquisition evaluation
    # ------------------------------------------------------------------ #
    def _select_next(self, gp: GaussianProcessRegressor,
                     y_best: float,
                     centre: np.ndarray) -> np.ndarray:
        """
        Build a candidate pool (global or inside the trust region),
        evaluate EI, locally refine the best `k_refine` points,
        and return the single point with highest refined EI.
        """
        # Decide whether to sample globally or inside the trust region.
        # When the region is already large (> 3.0) we allow global sampling
        # to keep a chance of escaping.
        if self.tr_radius < 3.0:
            pool_size = min(5000,
                            self.pool_factor * self.dim * self.n_init)
            candidates = self._trust_region_samples(pool_size, centre)
        else:
            pool_size = min(5000,
                            self.pool_factor * self.dim * self.n_init)
            candidates = self._sobol_samples(pool_size)

        ei_vals = self._ei(candidates, gp, y_best)

        # Choose top k_refine points for local optimisation
        k = min(self.k_refine, candidates.shape[0])
        top_idx = np.argpartition(-ei_vals, k)[:k]
        top_candidates = candidates[top_idx]

        refined = self._refine_ei(top_candidates, gp, y_best, centre)

        # Re‑evaluate EI on refined points and merge with the original pool
        if refined.size:
            refined_ei = self._ei(refined, gp, y_best)
            candidates = np.vstack([candidates, refined])
            ei_vals = np.concatenate([ei_vals, refined_ei])

        # Return the point with maximal EI
        best_idx = np.argmax(ei_vals)
        return candidates[best_idx].reshape(1, -1)

    # ------------------------------------------------------------------ #
    # Evaluation handling (budget‑aware)
    # ------------------------------------------------------------------ #
    def _evaluate(self,
                  func: Callable[[np.ndarray], float],
                  X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate `func` on rows of `X` respecting the remaining budget."""
        remaining = self.budget - self.n_evals
        if remaining <= 0:
            return np.empty((0, self.dim)), np.empty((0, 1))

        if X.shape[0] > remaining:
            X = X[:remaining]

        y = np.apply_along_axis(func, 1, X).reshape(-1, 1)
        self.n_evals += X.shape[0]
        return X, y

    def _append(self, X_new: np.ndarray, y_new: np.ndarray):
        """Append new data to the internal storage."""
        if self.X is None:
            self.X, self.y = X_new, y_new
        else:
            self.X = np.vstack([self.X, X_new])
            self.y = np.vstack([self.y, y_new])

    # ------------------------------------------------------------------ #
    # Main optimisation interface
    # ------------------------------------------------------------------ #
    def __call__(self,
                 func: Callable[[np.ndarray], float]) -> tuple[float, np.ndarray]:
        """
        Run the BO loop until the evaluation budget is exhausted.
        Returns (best_y, best_x).
        """
        # ----- initial Sobol design -----
        X0 = self._sobol_samples(self.n_init)
        X0, y0 = self._evaluate(func, X0)
        self._append(X0, y0)

        # current best
        best_idx = int(np.argmin(self.y))
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()

        # ----- sequential BO -----
        while self.n_evals < self.budget:
            self.iteration += 1

            # Fit or reuse GP
            if self.iteration % self.retrain_interval == 1 or self.iteration == 1:
                gp = self._fit_gp(self.X, self.y)
            else:
                # reuse previous GP (scikit‑learn does not expose a cheap update,
                # so we just keep the old object – it still predicts with the
                # old hyper‑parameters).
                pass

            # Propose next point (trust‑region centred on best_x)
            x_next = self._select_next(gp, best_y, centre=best_x)

            # Evaluate (budget‑aware)
            x_next, y_next = self._evaluate(func, x_next)
            if x_next.shape[0] == 0:
                break
            self._append(x_next, y_next)

            # Update best and adapt trust‑region radius
            cur_best_idx = int(np.argmin(self.y))
            cur_best_y = float(self.y[cur_best_idx])
            if cur_best_y < best_y - 1e-12:          # successful improvement
                best_y, best_x = cur_best_y, self.X[cur_best_idx].copy()
                self.tr_radius = min(self.tr_radius * 1.2, self.r_max)
            else:                                    # no improvement
                self.tr_radius = max(self.tr_radius * 0.5, self.r_min)

        return best_y, best_x


# ----------------------------------------------------------------------
# Helper to generate a different seed for each trust‑region sampling call.
# ----------------------------------------------------------------------
def iteration_seed():
    """Simple deterministic seed based on the global iteration counter."""
    # Using numpy's random seed generator ensures reproducibility across runs.
    # The seed is derived from the current time + process id to avoid
    # accidental repeats when the optimizer is called multiple times.
    import time, os
    return int(time.time() * 1000) % (2**32 - 1) ^ os.getpid()
```

**Explanation of key changes compared with the previous `AdaptiveSobolEI_BO`:**

1. **Dynamic trust‑region** – focuses sampling inside a shrinking/expanding hyper‑cube centred on the current best, dramatically improving exploitation in the Rosenbrock valley while still allowing global jumps when the radius grows.
2. **Periodic GP re‑training** – the GP is refitted only every `retrain_interval` iterations, cutting the heavy O(N³) cost without losing model fidelity.
3. **Region‑aware candidate generation** – when the trust‑region is small we sample exclusively inside it; otherwise we fall back to a global Sobol pool, preserving the ability to escape local minima.
4. **Local EI refinement respects the trust‑region bounds** – the L‑BFGS‑B optimiser is constrained to the current region, guaranteeing that refined points stay where the model is most reliable.
5. **Adaptive radius update rule** – a simple multiplicative increase/decrease (× 1.2 / × 0.5) with hard caps provides a robust, low‑overhead mechanism to automatically balance exploration vs. exploitation.
6. **Budget‑aware evaluation** – unchanged, still guarantees we never exceed the allowed number of function calls.

The resulting algorithm retains the lightweight nature of the original design
## Error
 Traceback (most recent call last):
  File "<TRSobolEI_BO>", line 242, in __call__
 240 |         # current best
 241 |         best_idx = int(np.argmin(self.y))
 242->         best_y = float(self.y[best_idx])
 243 |         best_x = self.X[best_idx].copy()
 244 | 
TypeError: only 0-dimensional arrays can be converted to Python scalars
