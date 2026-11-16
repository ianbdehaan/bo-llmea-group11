from collections.abc import Callable
import numpy as np
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern,
    WhiteKernel,
    ConstantKernel,
)

class AdaptiveSparseBO:
    """
    AdaptiveSparseBO – Bayesian optimisation with Sobol initial design,
    ARD Matern‑5/2 GP surrogate and Kriging‑Believer batch Expected Improvement.
    """

    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)          # total allowed function evaluations
        self.dim = int(dim)                # problem dimensionality
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])  # (2, dim)

        # data containers (filled after the first evaluation)
        self.X: np.ndarray | None = None   # (n_samples, dim)
        self.y: np.ndarray | None = None   # (n_samples, 1)

        self.n_evals = 0                   # number of evaluations already used

        # size of the Sobol initial design: at least 2*dim, at most half the budget
        self.n_init = min(max(2 * self.dim, 4), max(2, self.budget // 2))

    # ------------------------------------------------------------------ #
    # Sampling utilities
    # ------------------------------------------------------------------ #
    def _sample_points(self, n_points: int) -> np.ndarray:
        """Sobol quasi‑random points uniformly mapped to the search bounds."""
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol generates 2**m points; choose the smallest power of two >= n_points
        m = int(np.ceil(np.log2(n_points)))
        unit = sampler.random_base2(m=m)[:n_points]          # (n_points, dim) in [0,1]
        lower, upper = self.bounds
        return qmc.scale(unit, lower, upper)                 # map to [-5,5]^dim

    # ------------------------------------------------------------------ #
    # Surrogate model
    # ------------------------------------------------------------------ #
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """Fit a GP with ARD Matern‑5/2 kernel; hyper‑parameters are optimised."""
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
            random_state=0,
        )
        gp.fit(X, y.ravel())
        return gp

    # ------------------------------------------------------------------ #
    # Acquisition – Expected Improvement
    # ------------------------------------------------------------------ #
    def _ei(self, X: np.ndarray, gp: GaussianProcessRegressor, y_best: float) -> np.ndarray:
        """Vectorised Expected Improvement for a batch of candidates."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1) + 1e-12                # avoid zero division
        improvement = y_best - mu.reshape(-1, 1)           # we minimise
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei                                          # (n_candidates, 1)

    # ------------------------------------------------------------------ #
    # Kriging‑Believer batch selection
    # ------------------------------------------------------------------ #
    def _select_batch(
        self,
        batch_size: int,
        gp: GaussianProcessRegressor,
        y_best: float,
    ) -> np.ndarray:
        """
        Propose `batch_size` points using a Kriging‑Believer strategy.
        A modest candidate set is generated once; after each selection the GP
        is virtually updated with its own mean prediction.
        """
        n_candidates = max(5 * self.dim, 50)
        candidates = self._sample_points(n_candidates)   # (n_candidates, dim)

        selected = []
        gp_tmp = gp

        for _ in range(batch_size):
            ei = self._ei(candidates, gp_tmp, y_best)
            idx = np.argmax(ei)                           # best EI index
            x_next = candidates[idx : idx + 1]            # (1, dim)
            selected.append(x_next)

            # Kriging‑Believer update: pretend we observed the GP mean
            mu_next = gp_tmp.predict(x_next)              # (1,)
            X_aug = np.vstack([gp_tmp.X_train_, x_next])
            y_aug = np.append(gp_tmp.y_train_, mu_next)
            gp_tmp = self._fit_model(X_aug, y_aug)

            # Remove the chosen candidate to avoid duplication
            candidates = np.delete(candidates, idx, axis=0)

        return np.vstack(selected)                         # (batch_size, dim)

    # ------------------------------------------------------------------ #
    # Evaluation helpers
    # ------------------------------------------------------------------ #
    def _evaluate(self, func: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
        """
        Evaluate the black‑box `func` on a set of points respecting the remaining budget.
        Returns a column vector (n_points, 1).
        """
        n_remaining = self.budget - self.n_evals
        if X.shape[0] > n_remaining:
            X = X[:n_remaining]
        y = np.apply_along_axis(func, 1, X).reshape(-1, 1)
        self.n_evals += X.shape[0]
        return y

    def _append_data(self, X_new: np.ndarray, y_new: np.ndarray):
        """Append newly evaluated points to the internal dataset."""
        if self.X is None:
            self.X, self.y = X_new, y_new
        else:
            self.X = np.vstack([self.X, X_new])
            self.y = np.vstack([self.y, y_new])

    # ------------------------------------------------------------------ #
    # Main optimisation interface
    # ------------------------------------------------------------------ #
    def __call__(self, func: Callable[[np.ndarray], float]) -> tuple[float, np.ndarray]:
        """
        Run the BO loop until the evaluation budget is exhausted.
        Returns (best_y, best_x) where best_y is the smallest observed value.
        """
        # ---------- Initial design ----------
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate(func, X_init)
        self._append_data(X_init, y_init)

        best_idx = int(np.argmin(self.y))
        best_y = float(self.y[best_idx].item())
        best_x = self.X[best_idx].copy()

        # ---------- BO iterations ----------
        while self.n_evals < self.budget:
            # Fit surrogate on all data collected so far
            gp = self._fit_model(self.X, self.y)

            # Decide batch size (at most √remaining, at least 1)
            remaining = self.budget - self.n_evals
            batch_size = max(1, int(np.sqrt(remaining)))
            batch_size = min(batch_size, remaining)

            # Propose next points
            X_next = self._select_batch(batch_size, gp, best_y)

            # Evaluate them (budget is enforced inside _evaluate)
            y_next = self._evaluate(func, X_next)

            # Update dataset
            self._append_data(X_next, y_next)

            # Update incumbent
            cur_best = np.argmin(y_next)
            if y_next[cur_best, 0] < best_y:
                best_y = float(y_next[cur_best, 0].item())
                best_x = X_next[cur_best].copy()

        return best_y, best_x
