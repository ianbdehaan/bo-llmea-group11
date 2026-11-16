**# Description**  
**SparseAdaptiveBO** – a lightweight Bayesian‑Optimization framework designed for strict evaluation budgets.  
Key ideas  

* **Sobol quasi‑Monte‑Carlo initial design** – space‑filling, deterministic, cheap and well‑suited for the ‑5/5 box.  
* **Gaussian‑Process surrogate** with an ARD **Matern‑5/2** kernel (constant + white‑noise term). The kernel hyper‑parameters are re‑optimised only once per BO iteration (very cheap for ≤ 200 points).  
* **Expected‑Improvement (EI)** acquisition evaluated on a modest Sobol candidate pool (≈ 10 · dim points).  
* **Kriging‑Believer batch selection** – the GP is virtually updated with its own mean prediction after each selected candidate, providing cheap diversification without inner optimisation.  
* **Robust bookkeeping** – all array shapes are kept 1‑D where possible, eliminating the “only 0‑dimensional arrays can be converted to Python scalars” error and guaranteeing that the algorithm never exceeds the user‑provided budget.  

The algorithm is computationally cheap (no gradient‑based acquisition optimisation, no large candidate pools) yet retains the exploration‑exploitation balance of classic BO, making it well‑suited for the 6‑D Rosenbrock BBOB problem under a tight evaluation budget.

---

**# Justification**  

1. **Sobol initial design** – deterministic low‑discrepancy points give a good coverage of the search space with very few evaluations. Using `scramble=True` adds a stochastic element that prevents pathological alignment with the objective’s structure.  

2. **Matern‑5/2 ARD kernel** – the Rosenbrock function exhibits long narrow valleys; the Matern‑5/2 kernel models moderate smoothness and the ARD length‑scales allow the GP to adapt to different sensitivities along each coordinate.  

3. **Expected Improvement** – EI is scale‑free, works well with noisy or noise‑free objectives, and only requires the GP mean and variance, which are cheap to compute for the whole candidate pool.  

4. **Kriging‑Believer** – Instead of performing an expensive inner optimisation for each batch element, we pretend that the GP’s mean is the true function value after selecting a point. This cheap “belief” update keeps the batch selection fast while still providing diversification.  

5. **Shape‑safe handling** – All outputs of the GP are reshaped to 1‑D vectors (`ravel()`) and scalar extraction uses `.item()`. This removes the conversion error that occurred in the previous version.  

6. **Dynamic budget handling** – The code automatically truncates the initial design and any subsequent batch if the remaining budget is smaller, guaranteeing that the user‑specified budget is never exceeded.  

Overall, the refined algorithm keeps the simplicity and speed of the original Sparse‑GP‑BO while fixing the bug and adding a few robustness tweaks that improve its reliability on the BBOB test suite.

---

**# Code**  

```python
from collections.abc import Callable
from typing import Tuple

import numpy as np
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern,
    WhiteKernel,
    ConstantKernel,
)


class SparseAdaptiveBO:
    """
    SparseAdaptiveBO – Bayesian optimisation with
    Sobol initial design, Matern‑5/2 GP surrogate and
    batch Expected‑Improvement using a Kriging‑Believer update.
    """

    def __init__(self, budget: int, dim: int, random_state: int = 0):
        """
        Parameters
        ----------
        budget : int
            Maximum number of allowed objective evaluations.
        dim : int
            Dimensionality of the search space.
        random_state : int, optional
            Seed for reproducibility of Sobol sampling and GP optimiser.
        """
        self.budget = int(budget)
        self.dim = int(dim)
        self.random_state = random_state

        # Search space bounds: [-5, 5]^dim
        self.bounds = np.column_stack((np.full(dim, -5.0), np.full(dim, 5.0)))

        # Data containers (filled after the first evaluation)
        self.X: np.ndarray | None = None          # shape (n_samples, dim)
        self.y: np.ndarray | None = None          # shape (n_samples, )
        self.n_evals = 0

        # Initial design size – at most half the budget, at least 2*dim
        self.n_init = min(max(2 * self.dim, 5), max(1, self.budget // 2))

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------
    def _sample_sobol(self, n: int) -> np.ndarray:
        """Draw `n` Sobol points uniformly inside the bounded box."""
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.random_state)
        # Sobol generates 2**k points, so we request the smallest power of two ≥ n
        m = int(np.ceil(np.log2(n)))
        unit = sampler.random_base2(m=m)[:n]          # shape (n, dim) in [0,1]^d
        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        return qmc.scale(unit, lower, upper)          # map to [-5,5]^d

    # ------------------------------------------------------------------
    # Surrogate model
    # ------------------------------------------------------------------
    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """
        Fit a Gaussian Process with a constant * Matern‑5/2 * WhiteKernel.
        The hyper‑parameters are optimised by maximizing the log‑marginal likelihood.
        """
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=np.ones(self.dim),
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_state,
        )
        gp.fit(X, y.ravel())
        return gp

    # ------------------------------------------------------------------
    # Acquisition – Expected Improvement
    # ------------------------------------------------------------------
    def _ei(self, Xcand: np.ndarray, gp: GaussianProcessRegressor, y_best: float) -> np.ndarray:
        """Compute EI for every candidate in `Xcand`."""
        mu, sigma = gp.predict(Xcand, return_std=True)
        sigma = sigma.reshape(-1) + 1e-12                     # avoid division by zero
        delta = y_best - mu
        Z = delta / sigma
        ei = delta * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei                                            # shape (n_cand,)

    # ------------------------------------------------------------------
    # Kriging‑Believer batch selection
    # ------------------------------------------------------------------
    def _select_batch(
        self,
        batch_size: int,
        gp: GaussianProcessRegressor,
        y_best: float,
    ) -> np.ndarray:
        """
        Return `batch_size` points selected from a Sobol candidate pool.
        After each selection the GP is virtually updated with its own mean
        (Kriging‑Believer) to diversify the batch.
        """
        n_cand = max(10 * self.dim, 100)          # modest pool, cheap EI eval
        cand = self._sample_sobol(n_cand)

        selected = []
        gp_tmp = gp

        for _ in range(batch_size):
            ei = self._ei(cand, gp_tmp, y_best)
            idx = np.argmax(ei)
            x_next = cand[idx : idx + 1]          # (1, dim)
            selected.append(x_next)

            # Kriging‑Believer virtual update with the GP mean
            mu_next = gp_tmp.predict(x_next)      # (1,)
            X_aug = np.vstack([gp_tmp.X_train_, x_next])
            y_aug = np.append(gp_tmp.y_train_, mu_next)
            gp_tmp = self._fit_gp(X_aug, y_aug)

            # Remove the chosen candidate to avoid picking it again
            cand = np.delete(cand, idx, axis=0)

        return np.vstack(selected)                # (batch_size, dim)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _evaluate(self, func: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
        """Evaluate `func` on each row of `X` respecting the remaining budget."""
        remaining = self.budget - self.n_evals
        if X.shape[0] > remaining:
            X = X[:remaining]
        y = np.apply_along_axis(func, 1, X).astype(float)   # shape (n,)
        self.n_evals += X.shape[0]
        return y

    def _add_data(self, Xnew: np.ndarray, ynew: np.ndarray) -> None:
        """Append newly evaluated data to the internal storage."""
        if self.X is None:
            self.X = Xnew
            self.y = ynew
        else:
            self.X = np.vstack([self.X, Xnew])
            self.y = np.concatenate([self.y, ynew])

    # ------------------------------------------------------------------
    # Main optimisation interface
    # ------------------------------------------------------------------
    def __call__(self, func: Callable[[np.ndarray], float]) -> Tuple[float, np.ndarray]:
        """
        Run the BO loop until the evaluation budget is exhausted.

        Returns
        -------
        best_y : float
            Smallest observed objective value.
        best_x : np.ndarray, shape (dim,)
            Decision vector that achieved `best_y`.
        """
        # ---------- Initial design ----------
        X0 = self._sample_sobol(self.n_init)
        y0 = self._evaluate(func, X0)
        self._add_data(X0, y0)

        best_idx = int(np.argmin(self.y))
        best_y = float(self.y[best_idx].item())
        best_x = self.X[best_idx].copy()

        # ---------- BO iterations ----------
        while self.n_evals < self.budget:
            gp = self._fit_gp(self.X, self.y)

            # Simple single‑point batch (can be increased if desired)
            batch_sz = min(1, self.budget - self.n_evals)

            X_next = self._select_batch(batch_sz, gp, best_y)
            y_next = self._evaluate(func, X_next)

            self._add_data(X_next, y_next)

            # Update incumbent
            cur_best = int(np.argmin(y_next))
            if y_next[cur_best] < best_y:
                best_y = float(y_next[cur_best].item())
                best_x = X_next[cur_best].copy()

        return best_y, best_x
```

The class can be instantiated and used as follows:

```python
def rosenbrock(x):
    # classic Rosenbrock (sum of 100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2)
    return sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

bo = SparseAdaptiveBO(budget=200, dim=6, random_state=42)
best_val, best_x = bo(rosenbrock)
print("Best value:", best_val)
print("Best point:", best_x)
```

The implementation respects the evaluation budget, avoids the previous scalar‑conversion error, and incorporates a few robustness improvements while staying computationally lightweight.
## Error
 Traceback (most recent call last):
  File "<SparseAdaptiveBO>", line 5, in <module>
   3 |     return sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
   4 | 
   5-> bo = SparseAdaptiveBO(budget=200, dim=6, random_state=42)
   6 | best_val, best_x = bo(rosenbrock)
   7 | print("Best value:", best_val)
NameError: name 'SparseAdaptiveBO' is not defined
