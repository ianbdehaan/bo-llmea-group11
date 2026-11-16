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

class AdaptiveTRHybridEI_BO:
    """
    AdaptiveTRHybridEI_BO:
    Bayesian optimisation with Sobol start, adaptive trust‑region,
    hybrid EI maximisation (pool + L‑BFGS‑B) and periodic GP re‑training.
    """

    def __init__(self, budget: int, dim: int):
        self.budget = budget                # total allowed evaluations
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])   # (2, dim)

        # bookkeeping
        self.X = None          # shape (n, dim)
        self.y = None          # shape (n, 1)
        self.n_evals = 0

        # Sobol initial design
        self.n_init = max(2 * dim, 5)

        # trust‑region parameters
        self.tr_radius = 2.0                     # initial half‑side length
        self.tr_radius_min = 0.2
        self.tr_radius_max = 5.0
        self.tr_success_factor = 1.2
        self.tr_failure_factor = 0.7

        # GP re‑training schedule
        self.retrain_interval = 3
        self.iter_since_retrain = 0
        self.gp = None

    # ------------------------------------------------------------------ #
    # Sampling utilities (Sobol for first design, LHS afterwards)
    # ------------------------------------------------------------------ #
    def _sample(self, n: int, within_tr: bool = False) -> np.ndarray:
        """
        Return `n` points uniformly in the global bounds.
        If `within_tr` is True, points are sampled inside the current
        trust‑region hyper‑cube (clipped to the global bounds).
        """
        if self.X is None:                     # first call → Sobol
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            unit = sampler.random_base2(m=int(np.ceil(np.log2(n))))
            unit = unit[:n]
        else:
            sampler = qmc.LatinHypercube(d=self.dim)
            unit = sampler.random(n)

        # scale to appropriate bounds
        if within_tr:
            # centre = best point, half‑side = tr_radius
            centre = self.best_x
            lower = np.maximum(self.bounds[0], centre - self.tr_radius)
            upper = np.minimum(self.bounds[1], centre + self.tr_radius)
            scaled = qmc.scale(unit, lower, upper)
        else:
            scaled = qmc.scale(unit, self.bounds[0], self.bounds[1])
        return scaled

    # ------------------------------------------------------------------ #
    # GP surrogate
    # ------------------------------------------------------------------ #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray):
        """Fit (or re‑fit) the GP with an ARD Matern‑5/2 kernel."""
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(self.dim),
                     length_scale_bounds=(1e-2, 1e2),
                     nu=2.5)
            + WhiteKernel(noise_level=1e-8,
                          noise_level_bounds=(1e-10, 1e-1))
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
    # Expected Improvement (EI) and its negative for minimisation
    # ------------------------------------------------------------------ #
    def _ei(self, X: np.ndarray, gp: GaussianProcessRegressor) -> np.ndarray:
        """Vectorised EI (minimisation). Returns shape (n, 1)."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        y_best = np.min(self.y)

        with np.errstate(divide="ignore", invalid="ignore"):
            imp = y_best - mu.reshape(-1, 1)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def _neg_ei(self, x: np.ndarray, gp: GaussianProcessRegressor) -> float:
        """Scalar –EI for use with SciPy's L‑BFGS‑B."""
        x = x.reshape(1, -1)
        ei = self._ei(x, gp)
        return -ei.item()

    # ------------------------------------------------------------------ #
    # Hybrid EI maximisation (pool + local refinement)
    # ------------------------------------------------------------------ #
    def _max_ei(self, gp: GaussianProcessRegressor) -> np.ndarray:
        """
        Return the point that (approximately) maximises EI.
        Procedure:
          1. Sample a large pool inside the trust‑region.
          2. Evaluate EI on the whole pool.
          3. Take the `k_best` points as starts for L‑BFGS‑B on –EI.
          4. Return the best local optimum.
        """
        pool_size = max(5000, 200 * self.dim)
        pool = self._sample(pool_size, within_tr=True)
        ei_vals = self._ei(pool, gp).ravel()

        # select a few promising starts
        k_best = min(5, pool_size)
        start_idx = np.argpartition(-ei_vals, k_best - 1)[:k_best]
        starts = pool[start_idx]

        best_x = None
        best_ei = -np.inf

        # local optimisation from each start
        for s in starts:
            res = minimize(
                fun=self._neg_ei,
                x0=s,
                args=(gp,),
                method="L-BFGS-B",
                bounds=list(zip(self.bounds[0], self.bounds[1])),
                options={"maxiter": 30, "ftol": 1e-6},
            )
            cand = res.x.reshape(1, -1)
            cand_ei = self._ei(cand, gp).item()
            if cand_ei > best_ei:
                best_ei = cand_ei
                best_x = cand

        # fallback – if optimisation failed, use the best pool point
        if best_x is None:
            best_x = pool[np.argmax(ei_vals)].reshape(1, -1)

        return best_x

    # ------------------------------------------------------------------ #
    # Evaluation utilities
    # ------------------------------------------------------------------ #
    def _evaluate(self, func: Callable[[np.ndarray], float],
                  X: np.ndarray) -> np.ndarray:
        """Evaluate `func` row‑wise, update counter, and return column vector."""
        y_vals = np.apply_along_axis(func, 1, X).astype(np.float64)
        y_vals = y_vals.reshape(-1, 1)
        self.n_evals += len(X)
        return y_vals

    def _append_history(self, X_new: np.ndarray, y_new: np.ndarray):
        """Append new data to the history and update the incumbent."""
        if self.X is None:
            self.X = X_new
            self.y = y_new
        else:
            self.X = np.vstack([self.X, X_new])
            self.y = np.vstack([self.y, y_new])

        # update best point
        cur_best_idx = np.argmin(self.y)
        self.best_y = self.y[cur_best_idx, 0]
        self.best_x = self.X[cur_best_idx, :].copy()

    # ------------------------------------------------------------------ #
    # Trust‑region update
    # ------------------------------------------------------------------ #
    def _update_tr(self, improved: bool):
        """Adapt the trust‑region radius based on the last iteration."""
        if improved:
            self.tr_radius = min(
                self.tr_radius * self.tr_success_factor,
                self.tr_radius_max,
            )
        else:
            self.tr_radius = max(
                self.tr_radius * self.tr_failure_factor,
                self.tr_radius_min,
            )

    # ------------------------------------------------------------------ #
    # Main optimisation entry point
    # ------------------------------------------------------------------ #
    def __call__(self,
                 func: Callable[[np.ndarray], float]) -> tuple[float, np.ndarray]:
        """
        Run the optimisation under the supplied evaluation budget.
        Returns (best_value, best_point).
        """
        # ---------- 1) Sobol initial design ----------
        init_X = self._sample(self.n_init, within_tr=False)
        init_y = self._evaluate(func, init_X)
        self._append_history(init_X, init_y)

        # initialise GP (first fit)
        self.gp = self._fit_gp(self.X, self.y)
        self.iter_since_retrain = 0

        # ---------- 2) Main BO loop ----------
        while self.n_evals < self.budget:
            remaining = self.budget - self.n_evals
            batch = 1  # single‑point BO (budget‑friendly)

            # Re‑fit GP only every `retrain_interval` iterations
            if self.iter_since_retrain >= self.retrain_interval:
                self.gp = self._fit_gp(self.X, self.y)
                self.iter_since_retrain = 0
            else:
                # keep the same GP object (no kernel optimisation)
                self.iter_since_retrain += 1

            # Propose next point via hybrid EI maximisation inside TR
            next_X = self._max_ei(self.gp)          # shape (1, dim)

            # Ensure we stay inside global bounds (numerical safety)
            next_X = np.clip(next_X, self.bounds[0], self.bounds[1])

            # Evaluate
            next_y = self._evaluate(func, next_X)

            # Record
            prev_best = self.best_y
            self._append_history(next_X, next_y)

            # Trust‑region adaptation
            improved = self.best_y < prev_best - 1e-12   # strict improvement
            self._update_tr(improved)

        return self.best_y, self.best_x
