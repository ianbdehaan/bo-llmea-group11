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

class QuasiAdaptiveMaternBO:
    """
    Quasi‑Adaptive Bayesian Optimiser with Sobol initial design,
    ARD Matern‑5/2 GP surrogate and Expected‑Improvement acquisition.
    """

    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)          # total allowed evaluations
        self.dim = int(dim)
        # bounds: shape (2, dim) – lower row, upper row
        self.bounds = np.array([[-5.0] * self.dim, [5.0] * self.dim])
        self.X: np.ndarray | None = None   # evaluated points, shape (n, dim)
        self.y: np.ndarray | None = None   # function values, shape (n, 1)
        self.n_evals = 0                    # counter of used evaluations

        # initial design: at least 2*dim points, never more than half the budget
        self.n_init = min(max(2 * self.dim, 5), self.budget // 2)

        # deterministic Sobol seed for reproducibility
        self._sobol_seed = 0

    # ------------------------------------------------------------------ #
    # 1. Sobol quasi‑Monte‑Carlo sampling
    # ------------------------------------------------------------------ #
    def _sample_points(self, n_points: int) -> np.ndarray:
        """Return `n_points` Sobol samples scaled to the problem bounds."""
        # Sobol generates points in [0,1]^d; we need exactly n_points (not necessarily a power of two)
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self._sobol_seed)
        # generate enough points (next power of two) and truncate
        m = int(np.ceil(np.log2(max(1, n_points))))
        raw = sampler.random_base2(m=m)[:n_points]
        lower, upper = self.bounds
        return lower + raw * (upper - lower)

    # ------------------------------------------------------------------ #
    # 2. Gaussian‑Process surrogate (ARD Matern‑5/2)
    # ------------------------------------------------------------------ #
    def _fit_model(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """Fit a GP to data (X, y) and return the trained model."""
        # kernel = constant * Matern(5/2, ARD) + tiny white noise
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
    # 3. Expected Improvement acquisition
    # ------------------------------------------------------------------ #
    def _ei(self, X: np.ndarray, gp: GaussianProcessRegressor, y_best: float) -> np.ndarray:
        """Vectorised Expected Improvement for a set of candidates."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)               # (n,1)
        mu = mu.reshape(-1, 1)

        # avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = y_best - mu                     # improvement (minimisation)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei.squeeze()                       # shape (n,)

    # ------------------------------------------------------------------ #
    # 4. Select next batch (Kriging‑Believer + local refinement)
    # ------------------------------------------------------------------ #
    def _select_next_points(
        self,
        batch_size: int,
        gp: GaussianProcessRegressor,
        y_best: float,
    ) -> np.ndarray:
        """
        Return `batch_size` new points.
        Steps:
        1) generate a moderate Sobol candidate pool,
        2) compute EI and keep the `batch_size` best as start points,
        3) locally refine each start point with L‑BFGS‑B on -EI,
        4) sequentially add points using the Kriging‑Believer trick.
        """
        # 1) candidate pool
        n_candidates = max(5 * self.dim * 10, 100)   # at least 100 points
        candidates = self._sample_points(n_candidates)

        # 2) EI evaluation & keep top‑k starts
        ei_vals = self._ei(candidates, gp, y_best)
        start_idx = np.argpartition(-ei_vals, batch_size)[:batch_size]
        start_points = candidates[start_idx]

        # 3) local refinement of each start point
        bounds = [(low, high) for low, high in zip(self.bounds[0], self.bounds[1])]
        refined = []
        for x0 in start_points:
            res = minimize(
                fun=lambda x: -self._ei(x.reshape(1, -1), gp, y_best)[0],
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 30, "ftol": 1e-6},
            )
            refined.append(res.x if res.success else x0)

        # 4) Kriging‑Believer sequential batch construction
        batch = []
        X_tmp = self.X.copy()
        y_tmp = self.y.copy()
        gp_tmp = gp

        for x in refined:
            batch.append(x)
            # pretend observation = GP mean at x
            mu_x = gp_tmp.predict(x.reshape(1, -1))
            X_tmp = np.vstack([X_tmp, x])
            y_tmp = np.vstack([y_tmp, mu_x.reshape(-1, 1)])
            gp_tmp = self._fit_model(X_tmp, y_tmp)

        return np.array(batch)

    # ------------------------------------------------------------------ #
    # 5. Evaluation respecting the budget
    # ------------------------------------------------------------------ #
    def _evaluate_points(
        self,
        func: Callable[[np.ndarray], np.float64],
        X: np.ndarray,
    ) -> np.ndarray:
        """Evaluate `func` on each row of X, truncating if budget would be exceeded."""
        remaining = self.budget - self.n_evals
        if remaining <= 0:
            return np.empty((0, 1), dtype=np.float64)

        X = X[:remaining]                     # cut to remaining budget
        n = X.shape[0]
        y = np.empty((n, 1), dtype=np.float64)

        for i, xi in enumerate(X):
            y[i, 0] = func(xi)

        self.n_evals += n
        return y

    # ------------------------------------------------------------------ #
    # 6. History update
    # ------------------------------------------------------------------ #
    def _update_history(self, new_X: np.ndarray, new_y: np.ndarray) -> None:
        """Append newly evaluated points to the internal history."""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    # ------------------------------------------------------------------ #
    # 7. Main optimisation loop
    # ------------------------------------------------------------------ #
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[float, np.ndarray]:
        """
        Run Bayesian optimisation until the evaluation budget is exhausted.
        Returns (best_value, best_location).
        """
        # ----- Initial Sobol design -------------------------------------------------
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_history(X_init, y_init)

        # best‑so‑far
        best_idx = np.argmin(self.y)               # y is (n,1)
        best_y = float(self.y[best_idx, 0])        # scalar
        best_x = self.X[best_idx].copy()

        # ----- Main BO loop ---------------------------------------------------------
        while self.n_evals < self.budget:
            # Fit GP on all collected data
            gp = self._fit_model(self.X, self.y)

            # Determine batch size (1 is safe; can be larger if many evaluations remain)
            remaining = self.budget - self.n_evals
            batch_size = min(1, remaining)          # keep it simple for robustness

            # Propose new points
            next_X = self._select_next_points(batch_size, gp, best_y)

            # Evaluate them (budget‑aware)
            next_y = self._evaluate_points(func, next_X)

            if next_y.shape[0] == 0:                # budget exhausted inside _evaluate_points
                break

            # Update history
            self._update_history(next_X, next_y)

            # Update best if improved
            cur_best_idx = np.argmin(next_y)
            cur_best_val = float(next_y[cur_best_idx, 0])
            if cur_best_val < best_y:
                best_y = cur_best_val
                best_x = next_X[cur_best_idx].copy()

        return best_y, best_x
