**# Description**  
A **Sobol‑Matern‑EI BO** algorithm – a Bayesian optimisation method that combines  

* **Quasi‑Monte‑Carlo Sobol sampling** for a space‑filling initial design,  
* a **Gaussian Process surrogate** with a **Matern‑5/2 kernel** (flexible smoothness, good for Rosenbrock‑type valleys),  
* the **Expected‑Improvement (EI)** acquisition function, and  
* a **batch‑wise Kriging‑Believer** strategy with cheap L‑BFGS‑B local refinement to propose the next evaluation point(s).  

The method is lightweight, fully vectorised with NumPy, and respects the strict evaluation budget required by the BBOB test suite.

---

**# Justification**  

* **Sobol QMC initial design** – provides low‑discrepancy points that cover the 6‑D search space much better than pure random sampling, giving a reliable GP fit with only a few dozen evaluations.  

* **Matern‑5/2 kernel** – captures the moderate smoothness of the Rosenbrock function while keeping the GP hyper‑parameter optimisation inexpensive (only length‑scale and variance). Adding a small WhiteKernel prevents numerical issues.  

* **Expected Improvement** – classic acquisition that balances exploitation (low predicted mean) and exploration (high predictive variance) without requiring extra hyper‑parameters.  

* **Kriging‑Believer batch** – after selecting a point we “pretend” its EI‑value equals the predicted mean, update the GP virtually and continue selecting the rest of the batch. This yields a cheap batch strategy without re‑training the full GP after each single evaluation.  

* **Local refinement with L‑BFGS‑B** – the acquisition surface can have narrow ridges (typical for Rosenbrock). Starting from the best Sobol candidates and performing a few gradient‑based steps quickly sharpens the solution while keeping computational overhead low.  

All components are available in the allowed libraries (NumPy, SciPy, scikit‑learn), and the implementation stays within the given code skeleton.

---

**# Code**  

```python
from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.optimize import minimize

class SobolMaternEI_BO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget                # total allowed evaluations
        self.dim = dim
        # bounds has shape (2, dim); bounds[0] = lower, bounds[1] = upper
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None           # evaluated points, shape (n, dim)
        self.y: np.ndarray = None           # corresponding function values, shape (n, 1)
        self.n_evals = 0                    # counter of used evaluations

        # initial design size – at least 2*dim but never larger than budget/2
        self.n_init = min(max(2 * dim, 5), budget // 2)

    # --------------------------------------------------------------------- #
    # 1. Sampling (Sobol QMC)
    # --------------------------------------------------------------------- #
    def _sample_points(self, n_points: int) -> np.ndarray:
        """Return `n_points` Sobol samples scaled to the problem bounds."""
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol generates points in [0,1]^d; transform to bounds
        raw = sampler.random_base2(m=int(np.ceil(np.log2(n_points))))
        raw = raw[:n_points]                     # keep exactly n_points
        lower, upper = self.bounds
        return lower + raw * (upper - lower)

    # --------------------------------------------------------------------- #
    # 2. Surrogate model (Gaussian Process with Matern kernel)
    # --------------------------------------------------------------------- #
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Fit a GP to data (X, y) and return the trained model."""
        # Kernel: constant * Matern(5/2) + small white noise
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
                 Matern(length_scale=np.ones(self.dim), length_scale_bounds=(1e-2, 1e2), nu=2.5) + \
                 WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0,
                                      normalize_y=True,
                                      n_restarts_optimizer=5,
                                      random_state=0)
        gp.fit(X, y.ravel())
        return gp

    # --------------------------------------------------------------------- #
    # 3. Acquisition: Expected Improvement (EI)
    # --------------------------------------------------------------------- #
    def _acquisition_function(self, X: np.ndarray, gp: GaussianProcessRegressor,
                              y_best: float) -> np.ndarray:
        """Compute EI for each candidate in X."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu = mu.reshape(-1, 1)

        # avoid division by zero
        with np.errstate(divide='ignore'):
            imp = y_best - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    # --------------------------------------------------------------------- #
    # 4. Next point selection (batch Kriging‑Believer + local refinement)
    # --------------------------------------------------------------------- #
    def _select_next_points(self, batch_size: int, gp: GaussianProcessRegressor,
                            y_best: float) -> np.ndarray:
        """
        Return `batch_size` new points.
        Uses Sobol candidates + L‑BFGS‑B refinement, and the Kriging‑Believer
        trick to fill the batch.
        """
        # 1) generate a moderate candidate set (e.g., 5 * dim * 10)
        n_candidates = 5 * self.dim * 10
        candidates = self._sample_points(n_candidates)

        # 2) evaluate EI on the candidates
        ei = self._acquisition_function(candidates, gp, y_best).ravel()
        # keep the top `batch_size` candidates as starting points
        start_idx = np.argpartition(-ei, batch_size)[:batch_size]
        start_points = candidates[start_idx]

        # 3) local refinement for each start point
        refined = []
        bounds = [(low, high) for low, high in zip(self.bounds[0], self.bounds[1])]

        for x0 in start_points:
            res = minimize(
                fun=lambda x: -self._acquisition_function(x.reshape(1, -1), gp, y_best),
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-6}
            )
            refined.append(res.x if res.success else x0)

        # 4) Kriging‑Believer: sequentially add points, pretending each new point
        #    has the GP mean as its observed value.
        batch = []
        X_tmp = self.X.copy()
        y_tmp = self.y.copy()
        gp_tmp = gp

        for x in refined:
            batch.append(x)
            # pretend observation = GP mean at x
            mu_x = gp_tmp.predict(x.reshape(1, -1))
            X_tmp = np.vstack([X_tmp, x])
            y_tmp = np.vstack([y_tmp, mu_x])
            # re‑fit surrogate on the "augmented" data (cheap because dim is small)
            gp_tmp = self._fit_model(X_tmp, y_tmp)

        return np.array(batch)

    # --------------------------------------------------------------------- #
    # 5. Evaluation helper
    # --------------------------------------------------------------------- #
    def _evaluate_points(self, func: Callable[[np.ndarray], np.float64],
                         X: np.ndarray) -> np.ndarray:
        """Evaluate `func` on each row of X respecting the budget."""
        n = X.shape[0]
        if self.n_evals + n > self.budget:
            # truncate to the remaining budget
            X = X[: self.budget - self.n_evals]
            n = X.shape[0]

        y = np.empty((n, 1), dtype=np.float64)
        for i, xi in enumerate(X):
            y[i, 0] = func(xi)
        self.n_evals += n
        return y

    # --------------------------------------------------------------------- #
    # 6. Update internal data structures
    # --------------------------------------------------------------------- #
    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        """Append newly evaluated points to the history."""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    # --------------------------------------------------------------------- #
    # 7. Main optimisation loop
    # --------------------------------------------------------------------- #
    def __call__(self,
                 func: Callable[[np.ndarray], np.float64]
                 ) -> tuple[np.float64, np.ndarray]:
        """
        Run Bayesian optimisation until the evaluation budget is exhausted.
        Returns the best observed value and its location.
        """
        # ---- Initial design -------------------------------------------------
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # best-so-far
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()

        # ---- Main BO loop --------------------------------------------------
        while self.n_evals < self.budget:
            # Fit surrogate on all data gathered so far
            gp = self._fit_model(self.X, self.y)

            # Decide batch size (1 for simplicity, can be larger if budget allows)
            batch_size = 1
            # Ensure we do not request more points than remaining budget
            remaining = self.budget - self.n_evals
            batch_size = min(batch_size, remaining)

            # Select next candidate(s)
            next_X = self._select_next_points(batch_size, gp, best_y)

            # Evaluate them
            next_y = self._evaluate_points(func, next_X)

            # Update history
            self._update_eval_points(next_X, next_y)

            # Update best
            cur_best_idx = np.argmin(next_y)
            if next_y[cur_best_idx, 0] < best_y:
                best_y = float(next_y[cur_best_idx, 0])
                best_x = next_X[cur_best_idx].copy()

        return best_y, best_x
```

The class **`SobolMaternEI_BO`** follows the required interface (`__init__(budget, dim)` and `__call__(func)`) and respects the evaluation budget while providing a modern, efficient Bayesian optimisation pipeline suitable for the 6‑D Rosenbrock problem from the BBOB suite.
## Error
 Traceback (most recent call last):
  File "<SobolMaternEI_BO>", line 167, in __call__
 165 |         # best-so-far
 166 |         best_idx = np.argmin(self.y)
 167->         best_y = float(self.y[best_idx])
 168 |         best_x = self.X[best_idx].copy()
 169 | 
TypeError: only 0-dimensional arrays can be converted to Python scalars
