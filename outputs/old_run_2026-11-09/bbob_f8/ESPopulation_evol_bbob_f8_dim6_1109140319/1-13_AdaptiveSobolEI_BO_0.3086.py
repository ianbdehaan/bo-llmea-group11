from collections.abc import Callable
import numpy as np
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class AdaptiveSobolEI_BO:
    """
    AdaptiveSobolEI_BO – Bayesian optimisation with:
        • Sobol quasi‑Monte‑Carlo initial design,
        • ARD Matern‑5/2 GP surrogate,
        • Expected Improvement acquisition,
        • Large Sobol candidate pool + L‑BFGS‑B refinement of top‑k EI points,
        • Batch size = 1 (greedy), strict evaluation budget.
    """

    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)          # total number of allowed func evaluations
        self.dim = int(dim)                # problem dimensionality
        self.bounds = np.column_stack((np.full(dim, -5.0),
                                       np.full(dim, 5.0)))   # shape (dim, 2)

        # data containers
        self.X: np.ndarray | None = None   # shape (n_samples, dim)
        self.y: np.ndarray | None = None   # shape (n_samples, 1)

        self.n_evals = 0                   # evaluations performed so far

        # initial design: 5 points per dimension, but never more than half the budget
        self.n_init = min(5 * dim, max(2, self.budget // 2))

    # --------------------------------------------------------------------- #
    # Sampling utilities
    # --------------------------------------------------------------------- #
    def _sobol_samples(self, n: int) -> np.ndarray:
        """
        Return `n` Sobol points uniformly scaled to the search bounds.
        """
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        # Sobol generates 2**m points, so we ask for the smallest power of two >= n
        m = int(np.ceil(np.log2(n)))
        unit = sampler.random_base2(m=m)[:n]                     # shape (n, dim)
        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        return qmc.scale(unit, lower, upper)                     # shape (n, dim)

    # --------------------------------------------------------------------- #
    # Surrogate model
    # --------------------------------------------------------------------- #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """
        Fit a Gaussian Process with ARD Matern‑5/2 kernel + tiny white noise.
        """
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

    # --------------------------------------------------------------------- #
    # Expected Improvement acquisition
    # --------------------------------------------------------------------- #
    def _ei(self, X: np.ndarray, gp: GaussianProcessRegressor, y_best: float) -> np.ndarray:
        """
        Compute EI for minimisation.
        Returns an (n, ) array.
        """
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-12)                # avoid division by zero
        imp = y_best - mu                               # improvement (negative because we minimise)
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return ei

    # --------------------------------------------------------------------- #
    # Local refinement of EI (L‑BFGS‑B)
    # --------------------------------------------------------------------- #
    def _refine_ei(self,
                   start_points: np.ndarray,
                   gp: GaussianProcessRegressor,
                   y_best: float,
                   n_refine: int = 5) -> np.ndarray:
        """
        Starting from `start_points` (k × dim) run a few L‑BFGS‑B optimisations of
        the *negative* EI and return the refined points (still ≤ k points).
        """
        refined = []

        # bounds for scipy optimizer
        bounds = list(self.bounds)

        # objective: negative EI (because scipy minimizes)
        def neg_ei(x):
            return -self._ei(x.reshape(1, -1), gp, y_best)[0]

        for x0 in start_points[:n_refine]:
            res = minimize(
                neg_ei,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 30, "ftol": 1e-9},
            )
            refined.append(res.x)

        return np.vstack(refined) if refined else np.empty((0, self.dim))

    # --------------------------------------------------------------------- #
    # Candidate generation + acquisition evaluation
    # --------------------------------------------------------------------- #
    def _select_next(self,
                     gp: GaussianProcessRegressor,
                     y_best: float,
                     batch_size: int = 1) -> np.ndarray:
        """
        Generate a Sobol candidate pool, evaluate EI, refine the best few,
        and return `batch_size` points with highest refined EI.
        """
        # pool size: a few thousand points – cheap for dim ≤ 10
        pool_size = min(5000, 10 * self.dim * self.n_init)
        candidates = self._sobol_samples(pool_size)

        ei_vals = self._ei(candidates, gp, y_best)

        # pick the top `k_refine` points for local optimisation
        k_refine = min(5, pool_size)
        top_idx = np.argpartition(-ei_vals, k_refine)[:k_refine]
        top_candidates = candidates[top_idx]

        # local refinement (optional but improves exploitation)
        refined = self._refine_ei(top_candidates, gp, y_best, n_refine=k_refine)

        # evaluate EI again on refined points (if any)
        if refined.size:
            refined_ei = self._ei(refined, gp, y_best)
            # concatenate with original pool (so we keep the global picture)
            candidates = np.vstack([candidates, refined])
            ei_vals = np.concatenate([ei_vals, refined_ei])

        # finally pick the best `batch_size` points
        idx = np.argpartition(-ei_vals, batch_size)[:batch_size]
        return candidates[idx]

    # --------------------------------------------------------------------- #
    # Evaluation handling (budget‑aware)
    # --------------------------------------------------------------------- #
    def _evaluate(self,
                  func: Callable[[np.ndarray], float],
                  X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate `func` on each row of `X` respecting the remaining budget.
        Returns the (possibly truncated) X and the corresponding y values.
        """
        remaining = self.budget - self.n_evals
        if remaining <= 0:
            return np.empty((0, self.dim)), np.empty((0, 1))

        if X.shape[0] > remaining:
            X = X[:remaining]

        # vectorised evaluation via np.apply_along_axis (fast enough for small budgets)
        y = np.apply_along_axis(func, 1, X).reshape(-1, 1)
        self.n_evals += X.shape[0]
        return X, y

    def _append_data(self, X_new: np.ndarray, y_new: np.ndarray):
        """Append new observations to the internal dataset."""
        if self.X is None:
            self.X, self.y = X_new, y_new
        else:
            self.X = np.vstack([self.X, X_new])
            self.y = np.vstack([self.y, y_new])

    # --------------------------------------------------------------------- #
    # Main optimisation interface
    # --------------------------------------------------------------------- #
    def __call__(self,
                 func: Callable[[np.ndarray], float]) -> tuple[float, np.ndarray]:
        """
        Run the BO loop until the evaluation budget is exhausted.
        Returns (best_y, best_x) where best_y is the minimal observed value.
        """
        # ---------- initial Sobol design ----------
        X0 = self._sobol_samples(self.n_init)
        X0, y0 = self._evaluate(func, X0)
        self._append_data(X0, y0)

        # current best
        best_idx = int(np.argmin(self.y))
        best_y = float(self.y[best_idx].item())
        best_x = self.X[best_idx].copy()

        # ---------- sequential BO ----------
        while self.n_evals < self.budget:
            # fit surrogate
            gp = self._fit_gp(self.X, self.y)

            # propose next point (batch size = 1)
            x_next = self._select_next(gp, best_y, batch_size=1)

            # evaluate
            x_next, y_next = self._evaluate(func, x_next)
            if x_next.shape[0] == 0:          # budget exhausted during selection
                break
            self._append_data(x_next, y_next)

            # update best
            cur_best_idx = int(np.argmin(self.y))
            cur_best_y = float(self.y[cur_best_idx].item())
            if cur_best_y < best_y:
                best_y = cur_best_y
                best_x = self.X[cur_best_idx].copy()

        return best_y, best_x
