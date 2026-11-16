from collections.abc import Callable
import numpy as np
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern,
    WhiteKernel,
    ConstantKernel,
)


class AdaptiveSobolMaternEI_BO:
    """
    Adaptive Sobol‑Matern‑EI Bayesian Optimiser.

    Parameters
    ----------
    budget : int
        Total number of allowed function evaluations.
    dim : int
        Dimensionality of the search space (bounds are [-5, 5]^dim).
    """

    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)
        self.dim = int(dim)

        # Bounds: shape (2, dim)  –  lower row, upper row
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim], dtype=np.float64)

        # History containers (filled lazily)
        self.X: np.ndarray | None = None          # (n, dim)
        self.y: np.ndarray | None = None          # (n, 1)
        self.n_evals = 0

        # Initial design size: at least 2*dim, never more than 30% of budget
        self.n_init = min(max(2 * dim, 5), max(1, budget // 3))

        # Max batch size (can be reduced when remaining budget is small)
        self.max_batch = 5

    # ------------------------------------------------------------------ #
    # 1. Sobol sampling (scrambled)
    # ------------------------------------------------------------------ #
    def _sobol(self, n: int) -> np.ndarray:
        """Return `n` Sobol points scaled to the problem bounds."""
        # Sobol generates points in [0,1]^d; we need exactly n points.
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=0)
        # Use power‑of‑2 sampling then truncate
        m = int(np.ceil(np.log2(n)))
        raw = sampler.random_base2(m=m)[:n]
        lower, upper = self.bounds
        return lower + raw * (upper - lower)

    # ------------------------------------------------------------------ #
    # 2. GP surrogate (Matern‑5/2 ARD)
    # ------------------------------------------------------------------ #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """Fit a GP to (X, y) and return the trained regressor."""
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=np.ones(self.dim),
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=0,
        )
        gp.fit(X, y.ravel())
        return gp

    # ------------------------------------------------------------------ #
    # 3. Expected Improvement (minimisation)
    # ------------------------------------------------------------------ #
    def _ei(self, Xcand: np.ndarray, gp: GaussianProcessRegressor, f_best: float) -> np.ndarray:
        """Expected Improvement for minimisation (vectorised)."""
        mu, sigma = gp.predict(Xcand, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu = mu.reshape(-1, 1)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = f_best - mu          # improvement (positive = good)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei.ravel()               # shape (n_candidates,)

    # ------------------------------------------------------------------ #
    # 4. Local refinement of a point (L‑BFGS‑B on -EI)
    # ------------------------------------------------------------------ #
    def _refine(self, x0: np.ndarray, gp: GaussianProcessRegressor, f_best: float) -> np.ndarray:
        """Run a few L‑BFGS‑B steps to maximise EI starting from x0."""
        bounds = [(low, high) for low, high in zip(self.bounds[0], self.bounds[1])]

        def neg_ei(x):
            return -self._ei(x.reshape(1, -1), gp, f_best)

        res = minimize(
            fun=neg_ei,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 30, "ftol": 1e-6},
        )
        return res.x if res.success else x0

    # ------------------------------------------------------------------ #
    # 5. Batch selection – Kriging‑Believer + refinement
    # ------------------------------------------------------------------ #
    def _select_batch(
        self,
        batch_size: int,
        gp: GaussianProcessRegressor,
        f_best: float,
    ) -> np.ndarray:
        """
        Return `batch_size` new points using a Sobol candidate pool,
        EI ranking, local refinement and the Kriging‑Believer trick.
        """
        # 1) candidate pool (5 * dim * 10 points → reasonable coverage)
        n_cand = max(5 * self.dim * 10, batch_size * 10)
        cand = self._sobol(n_cand)

        # 2) compute EI on the pool
        ei_vals = self._ei(cand, gp, f_best)

        # 3) pick the top `batch_size` candidates as start points
        start_idx = np.argpartition(-ei_vals, batch_size)[:batch_size]
        start_pts = cand[start_idx]

        # 4) refine each start point locally
        refined = [self._refine(pt, gp, f_best) for pt in start_pts]

        # 5) Kriging‑Believer sequential addition
        batch = []
        X_tmp = self.X.copy() if self.X is not None else np.empty((0, self.dim))
        y_tmp = self.y.copy() if self.y is not None else np.empty((0, 1))
        gp_tmp = gp

        for pt in refined:
            batch.append(pt)
            # pretend observation = GP mean at pt
            mu_pt = gp_tmp.predict(pt.reshape(1, -1))
            X_tmp = np.vstack([X_tmp, pt])
            y_tmp = np.vstack([y_tmp, mu_pt.reshape(-1, 1)])
            gp_tmp = self._fit_gp(X_tmp, y_tmp)   # cheap because only a few rows added

        return np.array(batch)

    # ------------------------------------------------------------------ #
    # 6. Budget‑aware evaluation helper
    # ------------------------------------------------------------------ #
    def _evaluate(self, func: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
        """
        Evaluate `func` on rows of X, respecting the remaining budget.
        Returns an (n,1) array of function values.
        """
        remaining = self.budget - self.n_evals
        if remaining <= 0:
            return np.empty((0, 1))

        if X.shape[0] > remaining:
            X = X[:remaining]

        y = np.empty((X.shape[0], 1), dtype=np.float64)
        for i, xi in enumerate(X):
            y[i, 0] = func(xi)
        self.n_evals += X.shape[0]
        return y

    # ------------------------------------------------------------------ #
    # 7. History update
    # ------------------------------------------------------------------ #
    def _append_history(self, Xnew: np.ndarray, ynew: np.ndarray):
        if self.X is None:
            self.X = Xnew
            self.y = ynew
        else:
            self.X = np.vstack([self.X, Xnew])
            self.y = np.vstack([self.y, ynew])

    # ------------------------------------------------------------------ #
    # 8. Main optimisation entry point
    # ------------------------------------------------------------------ #
    def __call__(self, func: Callable[[np.ndarray], float]) -> tuple[float, np.ndarray]:
        """
        Run the optimiser until the evaluation budget is exhausted.

        Returns
        -------
        best_val : float
            Lowest observed function value.
        best_x : np.ndarray
            Corresponding location (shape (dim,)).
        """
        # ---------- Initial Sobol design ----------
        X0 = self._sobol(self.n_init)
        y0 = self._evaluate(func, X0)
        self._append_history(X0, y0)

        # current best
        best_idx = np.argmin(self.y)            # self.y is (n,1)
        best_val = float(self.y[best_idx, 0])
        best_x = self.X[best_idx].copy()

        # ---------- BO loop ----------
        while self.n_evals < self.budget:
            # Fit surrogate on all collected data
            gp = self._fit_gp(self.X, self.y)

            # Adaptive batch size (1 … max_batch, never exceed remaining budget)
            remaining = self.budget - self.n_evals
            batch_sz = min(self.max_batch, remaining)

            # Select next points
            X_next = self._select_batch(batch_sz, gp, best_val)

            # Evaluate them (budget guard inside)
            y_next = self._evaluate(func, X_next)

            # Update history
            self._append_history(X_next, y_next)

            # Update best-so-far
            cur_best_idx = np.argmin(y_next)
            if y_next[cur_best_idx, 0] < best_val:
                best_val = float(y_next[cur_best_idx, 0])
                best_x = X_next[cur_best_idx].copy()

        return best_val, best_x
