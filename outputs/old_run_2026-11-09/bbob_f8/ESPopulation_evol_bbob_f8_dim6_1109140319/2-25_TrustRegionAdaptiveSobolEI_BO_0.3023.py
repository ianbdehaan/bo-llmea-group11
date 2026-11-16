from collections.abc import Callable
import numpy as np
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class TrustRegionAdaptiveSobolEI_BO:
    """
    TrustRegionAdaptiveSobolEI_BO – Bayesian optimisation with:
        • Sobol quasi‑Monte‑Carlo initial design,
        • ARD Matern‑5/2 GP surrogate,
        • Expected Improvement (EI) acquisition,
        • Adaptive trust‑region around the current best,
        • Hybrid global+local Sobol candidate pools,
        • L‑BFGS‑B refinement of the top‑k EI points,
        • GP re‑training only every `retrain_interval` iterations,
        • Batch size = 1, strict evaluation budget.
    """

    def __init__(self,
                 budget: int,
                 dim: int,
                 *,
                 trust_init: float = 2.0,          # initial TR half‑side length
                 trust_expand: float = 1.2,        # factor after success
                 trust_shrink: float = 0.7,        # factor after failure
                 trust_min: float = 0.1,
                 trust_max: float = 5.0,
                 retrain_interval: int = 5,
                 global_pool_factor: float = 3.0,  # *dim points globally each iteration
                 local_pool_factor: float = 20.0,  # *dim points inside TR each iteration
                 k_refine: int = 5):
        self.budget = int(budget)
        self.dim = int(dim)
        self.bounds = np.column_stack((np.full(dim, -5.0),
                                       np.full(dim, 5.0)))          # (dim,2)

        # ---- bookkeeping -------------------------------------------------
        self.n_evals = 0
        self.X = None            # (n, dim)
        self.y = None            # (n, 1)

        # ---- initial design ------------------------------------------------
        self.n_init = min(5 * dim, max(2, self.budget // 2))

        # ---- trust‑region parameters ---------------------------------------
        self.tr_radius = trust_init
        self.tr_expand = trust_expand
        self.tr_shrink = trust_shrink
        self.tr_min = trust_min
        self.tr_max = trust_max

        # ---- other hyper‑parameters ----------------------------------------
        self.retrain_interval = max(1, retrain_interval)
        self.global_pool_factor = global_pool_factor
        self.local_pool_factor = local_pool_factor
        self.k_refine = k_refine

        # internal counter for GP refitting
        self._iters_since_retrain = 0

    # --------------------------------------------------------------------- #
    # Sampling utilities
    # --------------------------------------------------------------------- #
    def _sobol_samples(self, n: int) -> np.ndarray:
        """Draw `n` Sobol points uniformly scaled to the search bounds."""
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        m = int(np.ceil(np.log2(n)))
        unit = sampler.random_base2(m=m)[:n]                     # (n, dim)
        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        return qmc.scale(unit, lower, upper)

    def _sobol_in_tr(self, n: int, centre: np.ndarray) -> np.ndarray:
        """Sobol points inside the current trust‑region (clipped to global bounds)."""
        low = np.maximum(self.bounds[:, 0], centre - self.tr_radius)
        high = np.minimum(self.bounds[:, 1], centre + self.tr_radius)
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42 + self.n_evals)
        m = int(np.ceil(np.log2(n)))
        unit = sampler.random_base2(m=m)[:n]
        return qmc.scale(unit, low, high)

    # --------------------------------------------------------------------- #
    # Surrogate model
    # --------------------------------------------------------------------- #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """Fit a GP with ARD Matern‑5/2 kernel + tiny white noise."""
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
            n_restarts_optimizer=3,
            normalize_y=True,
            random_state=0,
        )
        gp.fit(X, y.ravel())
        return gp

    # --------------------------------------------------------------------- #
    # Expected Improvement
    # --------------------------------------------------------------------- #
    def _ei(self, X: np.ndarray, gp: GaussianProcessRegressor, y_best: float) -> np.ndarray:
        """EI for minimisation."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        imp = y_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return ei

    # --------------------------------------------------------------------- #
    # Local refinement of EI (L‑BFGS‑B)
    # --------------------------------------------------------------------- #
    def _refine_ei(self,
                   start_pts: np.ndarray,
                   gp: GaussianProcessRegressor,
                   y_best: float) -> np.ndarray:
        """Run L‑BFGS‑B from each start point to (locally) maximise EI."""
        refined = []
        bounds = list(self.bounds)

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

    # --------------------------------------------------------------------- #
    # Candidate generation + acquisition evaluation
    # --------------------------------------------------------------------- #
    def _select_next(self,
                     gp: GaussianProcessRegressor,
                     y_best: float,
                     centre: np.ndarray) -> np.ndarray:
        """
        Build a hybrid candidate pool (global + local TR), evaluate EI,
        locally refine the top `k_refine` points and return the best one.
        """
        # ----- global pool -------------------------------------------------
        pool_global = int(np.ceil(self.global_pool_factor * self.dim))
        Xg = self._sobol_samples(pool_global)

        # ----- local TR pool ------------------------------------------------
        pool_local = int(np.ceil(self.local_pool_factor * self.dim))
        Xl = self._sobol_in_tr(pool_local, centre)

        # union of both pools
        candidates = np.vstack([Xg, Xl])

        # EI on the raw candidates
        ei_vals = self._ei(candidates, gp, y_best)

        # top‑k for refinement
        k = min(self.k_refine, candidates.shape[0])
        top_idx = np.argpartition(-ei_vals, k)[:k]
        top_candidates = candidates[top_idx]

        # refinement
        refined = self._refine_ei(top_candidates, gp, y_best)

        # evaluate EI on refined points and merge
        if refined.size:
            refined_ei = self._ei(refined, gp, y_best)
            candidates = np.vstack([candidates, refined])
            ei_vals = np.concatenate([ei_vals, refined_ei])

        # pick the single best point
        best_idx = np.argmax(ei_vals)          # EI is larger → better improvement
        return candidates[best_idx].reshape(1, -1)

    # --------------------------------------------------------------------- #
    # Budget‑aware evaluation
    # --------------------------------------------------------------------- #
    def _evaluate(self,
                  func: Callable[[np.ndarray], float],
                  X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate `func` row‑wise respecting the remaining budget."""
        remaining = self.budget - self.n_evals
        if remaining <= 0:
            return np.empty((0, self.dim)), np.empty((0, 1))

        if X.shape[0] > remaining:
            X = X[:remaining]

        # vectorised evaluation (still sequential per point)
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
        # ---- initial Sobol design -----------------------------------------
        X0 = self._sobol_samples(self.n_init)
        X0, y0 = self._evaluate(func, X0)
        self._append_data(X0, y0)

        # current best solution
        best_idx = int(np.argmin(self.y))
        best_y = float(self.y[best_idx].item())
        best_x = self.X[best_idx].copy()

        # ---- sequential BO -----------------------------------------------
        while self.n_evals < self.budget:
            # decide whether to refit the GP
            if self._iters_since_retrain == 0:
                gp = self._fit_gp(self.X, self.y)
            else:
                # reuse previously fitted GP (scikit‑learn keeps the kernel)
                gp = self._fit_gp(self.X, self.y) if self._iters_since_retrain >= self.retrain_interval else self._last_gp
            self._last_gp = gp

            # propose next point using the current trust‑region centre = best_x
            x_next = self._select_next(gp, best_y, centre=best_x)

            # evaluate (budget‑aware)
            x_next, y_next = self._evaluate(func, x_next)
            if x_next.shape[0] == 0:
                break
            self._append_data(x_next, y_next)

            # update best and trust‑region radius
            cur_best_idx = int(np.argmin(self.y))
            cur_best_y = float(self.y[cur_best_idx].item())
            if cur_best_y < best_y - 1e-12:          # successful improvement
                best_y = cur_best_y
                best_x = self.X[cur_best_idx].copy()
                self.tr_radius = min(self.tr_radius * self.tr_expand, self.tr_max)
            else:
                # no improvement → shrink TR
                self.tr_radius = max(self.tr_radius * self.tr_shrink, self.tr_min)

            # bookkeeping
            self._iters_since_retrain = (self._iters_since_retrain + 1) % self.retrain_interval

        return best_y, best_x
