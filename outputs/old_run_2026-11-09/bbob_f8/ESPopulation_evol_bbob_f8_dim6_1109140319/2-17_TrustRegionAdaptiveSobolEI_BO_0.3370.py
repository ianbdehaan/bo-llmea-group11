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
        • Expected Improvement acquisition,
        • Adaptive trust‑region centred on the current best,
        • Hybrid global + TR candidate pool,
        • L‑BFGS‑B refinement of top‑k EI points,
        • Periodic GP hyper‑parameter re‑training,
        • Strict evaluation budget.
    """

    # -------------------------- constructor -------------------------- #
    def __init__(self, budget: int, dim: int):
        self.budget = int(budget)            # total allowed evaluations
        self.dim = int(dim)                  # problem dimensionality
        self.bounds = np.column_stack(
            (np.full(dim, -5.0), np.full(dim, 5.0))
        )                                   # (dim, 2)

        # data containers
        self.X: np.ndarray | None = None     # (n_samples, dim)
        self.y: np.ndarray | None = None     # (n_samples, 1)

        self.n_evals = 0                     # evaluations used so far

        # initial Sobol design (max half the budget, at least 2·dim points)
        self.n_init = min(5 * dim, max(2 * dim, self.budget // 2))

        # ---- trust‑region parameters ----
        self.tr_radius = 2.0                 # initial half‑side length
        self.min_radius = 0.2
        self.max_radius = 5.0
        self.inc_factor = 1.5                # enlarge after success
        self.dec_factor = 0.7                # shrink after failure
        self.success_tol = 1e-3              # improvement needed to count as success

        # ---- GP re‑training schedule ----
        self.retrain_interval = 5            # re‑fit GP every 5 iterations
        self._iters_since_retrain = 0

    # ----------------------------------------------------------------- #
    # Sampling utilities
    # ----------------------------------------------------------------- #
    def _sobol_samples(self, n: int) -> np.ndarray:
        """Generate `n` Sobol points uniformly scaled to the search bounds."""
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        m = int(np.ceil(np.log2(n)))               # Sobol needs power‑of‑2
        unit = sampler.random_base2(m=m)[:n]
        lower, upper = self.bounds[:, 0], self.bounds[:, 1]
        return qmc.scale(unit, lower, upper)

    def _sobol_inside_tr(self, n: int, centre: np.ndarray) -> np.ndarray:
        """
        Sample `n` points uniformly inside the current trust‑region
        (hyper‑cube centred at `centre` with half‑side `self.tr_radius`).
        Points falling outside the global bounds are clipped.
        """
        lower = np.maximum(self.bounds[:, 0], centre - self.tr_radius)
        upper = np.minimum(self.bounds[:, 1], centre + self.tr_radius)
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42 + self.n_evals)
        m = int(np.ceil(np.log2(n)))
        unit = sampler.random_base2(m=m)[:n]
        return qmc.scale(unit, lower, upper)

    # ----------------------------------------------------------------- #
    # Surrogate model
    # ----------------------------------------------------------------- #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray,
                warm_start: GaussianProcessRegressor | None = None
                ) -> GaussianProcessRegressor:
        """
        Fit a GP with ARD Matern‑5/2 kernel + tiny white noise.
        If `warm_start` is given we keep its hyper‑parameters (no re‑optimisation).
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
            n_restarts_optimizer=0,          # we handle re‑training manually
            normalize_y=True,
            random_state=0,
        )
        if warm_start is not None:
            # clone with existing hyper‑parameters, then fit new data
            gp.kernel_ = warm_start.kernel_
        gp.fit(X, y.ravel())
        return gp

    # ----------------------------------------------------------------- #
    # Expected Improvement acquisition
    # ----------------------------------------------------------------- #
    def _ei(self, X: np.ndarray,
            gp: GaussianProcessRegressor,
            y_best: float) -> np.ndarray:
        """EI for minimisation (vectorised)."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-12)          # avoid division by zero
        imp = y_best - mu                         # improvement (positive = good)
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
        return ei

    # ----------------------------------------------------------------- #
    # Local refinement of EI (L‑BFGS‑B)
    # ----------------------------------------------------------------- #
    def _refine_ei(self,
                   start_points: np.ndarray,
                   gp: GaussianProcessRegressor,
                   y_best: float,
                   n_refine: int = 5) -> np.ndarray:
        """Run a few L‑BFGS‑B optimisations of the *negative* EI."""
        refined = []
        bounds = list(self.bounds)

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

    # ----------------------------------------------------------------- #
    # Candidate generation + acquisition evaluation
    # ----------------------------------------------------------------- #
    def _select_next(self,
                     gp: GaussianProcessRegressor,
                     y_best: float,
                     best_x: np.ndarray,
                     batch_size: int = 1) -> np.ndarray:
        """
        Hybrid candidate pool:
            * 10 % global Sobol points,
            * 90 % Sobol points inside the trust‑region.
        After EI evaluation the top‑k points are locally refined.
        Returns `batch_size` points with highest refined EI.
        """
        pool_size = min(5000, 10 * self.dim * self.n_init)

        # split into global / local
        n_global = max(1, pool_size // 10)
        n_local = pool_size - n_global

        global_cand = self._sobol_samples(n_global)
        local_cand = self._sobol_inside_tr(n_local, best_x)

        candidates = np.vstack([global_cand, local_cand])

        ei_vals = self._ei(candidates, gp, y_best)

        # top‑k for local refinement (k <= 5)
        k_refine = min(5, candidates.shape[0])
        top_idx = np.argpartition(-ei_vals, k_refine)[:k_refine]
        top_candidates = candidates[top_idx]

        refined = self._refine_ei(top_candidates, gp, y_best, n_refine=k_refine)

        if refined.size:
            refined_ei = self._ei(refined, gp, y_best)
            candidates = np.vstack([candidates, refined])
            ei_vals = np.concatenate([ei_vals, refined_ei])

        # pick best `batch_size` points
        idx = np.argpartition(-ei_vals, batch_size)[:batch_size]
        return candidates[idx]

    # ----------------------------------------------------------------- #
    # Evaluation handling (budget‑aware)
    # ----------------------------------------------------------------- #
    def _evaluate(self,
                  func: Callable[[np.ndarray], float],
                  X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate `func` row‑wise, respecting the remaining budget."""
        remaining = self.budget - self.n_evals
        if remaining <= 0:
            return np.empty((0, self.dim)), np.empty((0, 1))

        if X.shape[0] > remaining:
            X = X[:remaining]

        # simple loop is fine for the modest budgets we face
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

    # ----------------------------------------------------------------- #
    # Trust‑region update
    # ----------------------------------------------------------------- #
    def _update_tr(self, new_best_y: float, old_best_y: float):
        """Adapt the trust‑region radius based on success/failure."""
        if old_best_y - new_best_y > self.success_tol:      # success
            self.tr_radius = min(self.tr_radius * self.inc_factor,
                                 self.max_radius)
        else:                                               # failure
            self.tr_radius = max(self.tr_radius * self.dec_factor,
                                 self.min_radius)

    # ----------------------------------------------------------------- #
    # Main optimisation interface
    # ----------------------------------------------------------------- #
    def __call__(self,
                 func: Callable[[np.ndarray], float]
                 ) -> tuple[float, np.ndarray]:
        """
        Run the BO loop until the evaluation budget is exhausted.
        Returns (best_y, best_x) where best_y is the minimal observed value.
        """
        # ------------------- initial Sobol design ------------------- #
        X0 = self._sobol_samples(self.n_init)
        X0, y0 = self._evaluate(func, X0)
        self._append_data(X0, y0)

        # current best point & value
        best_idx = int(np.argmin(self.y))
        best_y = float(self.y[best_idx].item())
        best_x = self.X[best_idx].copy()

        # -------------------------- BO loop -------------------------- #
        while self.n_evals < self.budget:
            # decide whether we need a full GP re‑fit
            if self._iters_since_retrain >= self.retrain_interval or \
               self._iters_since_retrain == 0:
                gp = self._fit_gp(self.X, self.y)          # full optimisation
                self._iters_since_retrain = 0
            else:
                # warm‑start: keep previous hyper‑parameters
                gp = self._fit_gp(self.X, self.y, warm_start=gp)
                self._iters_since_retrain += 1

            # propose next point(s)
            x_next = self._select_next(gp, best_y, best_x, batch_size=1)

            # evaluate (budget‑aware)
            x_next, y_next = self._evaluate(func, x_next)
            if x_next.shape[0] == 0:
                break
            self._append_data(x_next, y_next)

            # update best and trust‑region
            old_best_y = best_y
            cur_best_idx = int(np.argmin(self.y))
            cur_best_y = float(self.y[cur_best_idx].item())
            if cur_best_y < best_y:
                best_y = cur_best_y
                best_x = self.X[cur_best_idx].copy()

            self._update_tr(best_y, old_best_y)

        return best_y, best_x
