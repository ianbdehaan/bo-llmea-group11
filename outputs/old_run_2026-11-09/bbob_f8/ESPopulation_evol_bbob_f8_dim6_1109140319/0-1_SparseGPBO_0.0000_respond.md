**# Description**  
A **Sparse‑GP‑BO** algorithm (named `SparseGPBO`) that combines  

* **Quasi‑Monte‑Carlo Sobol sampling** for a space‑filling initial design,  
* a **Gaussian Process surrogate** with a Matern‑5/2 kernel (fast‑to‑train, good for smooth but non‑convex landscapes like Rosenbrock),  
* **Expected Improvement (EI)** as acquisition function, and  
* a **batch‑wise Kriging‑Believer** selection that re‑uses the GP posterior to propose several points per iteration without costly inner optimisation.  

The method is computationally efficient (few GP hyper‑parameter optimisations, cheap acquisition evaluation on a modest candidate set) and works well on low‑ to medium‑dimensional black‑box problems such as BBOB Function 8 (Rosenbrock, 6‑D) within a strict evaluation budget.

---

**# Justification**  

| Component | Why it is chosen for the task |
|-----------|------------------------------|
| **Sobol QMC initial design** | Provides a low‑discrepancy, space‑filling set of points with only *O(dim·n)* cost, giving a reliable first GP fit even with a small budget. |
| **Gaussian Process (Matern‑5/2)** | The GP is the classic surrogate for Bayesian optimisation; the Matern‑5/2 kernel captures the moderate smoothness of Rosenbrock while remaining numerically stable. Scikit‑learn’s `GaussianProcessRegressor` offers automatic hyper‑parameter optimisation via marginal‑likelihood maximisation. |
| **Expected Improvement (EI)** | A closed‑form acquisition that balances exploitation and exploration; cheap to compute for many candidate points. |
| **Batch Kriging‑Believer** | Rather than solving a costly inner optimisation for each new point, we sample a moderate candidate pool (e.g., 5·dim points), evaluate EI, pick the best, **pretend** its objective value equals the GP mean (the “belief”), update the GP posterior virtually, and repeat for the remaining batch slots. This yields a fast, parallel‑friendly batch without sacrificing much performance. |
| **Budget‑aware loop** | All calls to the black‑box are routed through `_evaluate_points`, which updates `self.n_evals`. The loop never exceeds `self.budget`. |
| **Scalable to any `dim`** | The number of initial points `n_init = min(2·dim, budget // 2)` scales linearly with dimension, guaranteeing at least a minimal model while leaving room for BO iterations. |

Overall, the algorithm respects the strict evaluation budget, stays within the allowed libraries, and is simple enough to be reproduced while still leveraging modern BO ideas.

---

**# Code**  

```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class SparseGPBO:
    """
    SparseGPBO – Bayesian optimisation with Sobol initial design,
    Gaussian Process surrogate (Matern‑5/2) and batch Expected Improvement.
    """
    def __init__(self, budget: int, dim: int):
        self.budget = budget                # total allowed evaluations
        self.dim = dim                      # problem dimensionality
        # bounds has shape (2, dim): [lower, upper]
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # data containers
        self.X: np.ndarray = None           # shape (n_samples, dim)
        self.y: np.ndarray = None           # shape (n_samples, 1)
        self.n_evals = 0                    # how many evaluations have been used
        # initial design size – at most half the budget, at least 2·dim
        self.n_init = min(max(2 * dim, 4), self.budget // 2)

    # ------------------------------------------------------------------ #
    #  Sampling utilities
    # ------------------------------------------------------------------ #
    def _sample_points(self, n_points: int) -> np.ndarray:
        """Quasi‑random Sobol sampling inside the search bounds."""
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol generates points in [0,1]^d; map to bounds
        unit_samples = sampler.random_base2(m=int(np.ceil(np.log2(n_points))))
        unit_samples = unit_samples[:n_points]               # keep exactly n_points
        lower, upper = self.bounds
        return qmc.scale(unit_samples, lower, upper)          # shape (n_points, dim)

    # ------------------------------------------------------------------ #
    #  Surrogate model
    # ------------------------------------------------------------------ #
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Fit a GP with Matern‑5/2 kernel; hyper‑parameters are optimised
        by maximizing the log‑marginal likelihood (default in scikit‑learn)."""
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
                 Matern(length_scale=np.ones(self.dim), length_scale_bounds=(1e-2, 1e2), nu=2.5) + \
                 WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-1))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      normalize_y=True,
                                      n_restarts_optimizer=5,
                                      random_state=0)
        gp.fit(X, y.ravel())
        return gp

    # ------------------------------------------------------------------ #
    #  Acquisition – Expected Improvement
    # ------------------------------------------------------------------ #
    def _acquisition_function(self, X: np.ndarray, gp: GaussianProcessRegressor,
                              y_best: float) -> np.ndarray:
        """Return EI for each candidate in X (shape (n, dim))."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1) + 1e-12               # avoid division by zero
        improvement = y_best - mu.reshape(-1, 1)
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei                     # shape (n, 1)

    # ------------------------------------------------------------------ #
    #  Batch selection using Kriging‑Believer
    # ------------------------------------------------------------------ #
    def _select_next_points(self, batch_size: int,
                            gp: GaussianProcessRegressor,
                            y_best: float) -> np.ndarray:
        """
        Propose `batch_size` points.
        A modest candidate set is generated, EI is evaluated,
        and the best point is taken. The GP posterior is updated
        with a *belief* (the predicted mean) before picking the next
        point of the batch (Kriging‑Believer).
        """
        # candidate pool – a few times the dimension for cheap evaluation
        n_candidates = max(5 * self.dim, 50)
        candidates = self._sample_points(n_candidates)

        selected = []
        # copy of GP to be virtually updated
        gp_tmp = gp
        for _ in range(batch_size):
            ei = self._acquisition_function(candidates, gp_tmp, y_best)
            idx = np.argmax(ei)
            x_next = candidates[idx:idx + 1]               # shape (1, dim)
            selected.append(x_next)

            # Kriging‑Believer update: pretend we observed the GP mean
            mu_next = gp_tmp.predict(x_next)
            # Augment training data virtually
            X_aug = np.vstack([gp_tmp.X_train_, x_next])
            y_aug = np.append(gp_tmp.y_train_, mu_next)
            gp_tmp = self._fit_model(X_aug, y_aug)

            # Remove the selected candidate to avoid reselection
            candidates = np.delete(candidates, idx, axis=0)

        return np.vstack(selected)                         # shape (batch_size, dim)

    # ------------------------------------------------------------------ #
    #  Evaluation helpers
    # ------------------------------------------------------------------ #
    def _evaluate_points(self, func: Callable[[np.ndarray], np.float64],
                         X: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of points with the black‑box `func`.
        Returns a column vector of shape (n_points, 1).
        """
        n = X.shape[0]
        if self.n_evals + n > self.budget:
            # truncate to remaining budget
            X = X[: self.budget - self.n_evals]
            n = X.shape[0]
        y = np.apply_along_axis(func, 1, X).reshape(-1, 1)
        self.n_evals += n
        return y

    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        """Append newly evaluated points to the data set."""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    # ------------------------------------------------------------------ #
    #  Main optimisation interface
    # ------------------------------------------------------------------ #
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        """
        Run the BO loop until the evaluation budget is exhausted.
        Returns (best_y, best_x) where best_y is the smallest observed
        objective value and best_x the corresponding decision vector.
        """
        # ---------- initial design ----------
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # keep track of the current best
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()

        # ---------- BO iterations ----------
        while self.n_evals < self.budget:
            # Fit surrogate on all available data
            gp = self._fit_model(self.X, self.y)

            # Determine how many points we can still afford
            remaining = self.budget - self.n_evals
            batch_size = min(1, remaining)   # single‑point batch is safe; change to >1 if desired

            # Propose next point(s) via acquisition
            X_next = self._select_next_points(batch_size, gp, best_y)

            # Evaluate them (respecting the budget)
            y_next = self._evaluate_points(func, X_next)

            # Update data set
            self._update_eval_points(X_next, y_next)

            # Update incumbent
            cur_best_idx = np.argmin(y_next)
            if y_next[cur_best_idx, 0] < best_y:
                best_y = float(y_next[cur_best_idx, 0])
                best_x = X_next[cur_best_idx].copy()

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<SparseGPBO>", line 147, in __call__
 145 |         # keep track of the current best
 146 |         best_idx = np.argmin(self.y)
 147->         best_y = float(self.y[best_idx])
 148 |         best_x = self.X[best_idx].copy()
 149 | 
TypeError: only 0-dimensional arrays can be converted to Python scalars
