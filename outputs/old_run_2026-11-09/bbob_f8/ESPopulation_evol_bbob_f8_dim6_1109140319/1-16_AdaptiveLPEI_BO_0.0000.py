from collections.abc import Callable
import numpy as np
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class AdaptiveLPEI_BO:
    """
    AdaptiveLPEI‑BO: Bayesian Optimisation with
    • Sobol+LHS hybrid initial design,
    • ARD Matern‑5/2 GP surrogate,
    • Expected‑Improvement acquisition,
    • Gradient‑refined EI,
    • Adaptive batch size + Local‑Penalisation.
    """

    def __init__(self, budget: int, dim: int):
        self.budget = budget                # total allowed evaluations
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])   # (2, dim)

        # History
        self.X: np.ndarray = None           # (n, dim)
        self.y: np.ndarray = None           # (n, 1)
        self.n_evals = 0

        # design parameters
        self.n_init = max(2 * dim, 5)        # Sobol points
        self.retrain_interval = 5           # GP hyper‑opt frequency
        self.iter_counter = 0

    # ------------------------------------------------------------------ #
    # Sampling utilities
    # ------------------------------------------------------------------ #
    def _sample(self, n: int, use_sobol: bool) -> np.ndarray:
        """Generate `n` points in the search space."""
        if use_sobol:
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            unit = sampler.random_base2(m=int(np.ceil(np.log2(n))))
            unit = unit[:n]
        else:
            sampler = qmc.LatinHypercube(d=self.dim)
            unit = sampler.random(n)

        return qmc.scale(unit, self.bounds[0], self.bounds[1])

    # ------------------------------------------------------------------ #
    # GP surrogate
    # ------------------------------------------------------------------ #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """Fit GP with ARD Matern‑5/2 kernel."""
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(self.dim),
                           length_scale_bounds=(1e-2, 1e2),
                           nu=2.5)               # ARD automatically enabled
                  + WhiteKernel(noise_level=1e-8,
                                noise_level_bounds=(1e-10, 1e-1)))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      normalize_y=True,
                                      n_restarts_optimizer=2,
                                      random_state=0)
        gp.fit(X, y.ravel())
        return gp

    # ------------------------------------------------------------------ #
    # Expected Improvement
    # ------------------------------------------------------------------ #
    def _ei(self, X: np.ndarray, gp: GaussianProcessRegressor) -> np.ndarray:
        """Vectorised Expected Improvement (minimisation)."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)

        y_best = np.min(self.y)

        with np.errstate(divide='ignore', invalid='ignore'):
            imp = y_best - mu.reshape(-1, 1)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei.ravel()                     # shape (n,)

    # ------------------------------------------------------------------ #
    # Gradient‑based EI refinement
    # ------------------------------------------------------------------ #
    def _refine_ei(self,
                   x0: np.ndarray,
                   gp: GaussianProcessRegressor,
                   bounds: np.ndarray,
                   max_iter: int = 20) -> np.ndarray:
        """Run a few L‑BFGS‑B steps starting from `x0` to increase EI."""
        def neg_ei(x):
            mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
            sigma = sigma[0]
            y_best = np.min(self.y)
            imp = y_best - mu[0]
            if sigma == 0.0:
                return 0.0
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei  # minimise negative EI

        res = minimize(neg_ei,
                       x0,
                       method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-6})
        return res.x

    # ------------------------------------------------------------------ #
    # Local penalisation for batch selection
    # ------------------------------------------------------------------ #
    def _penalise(self,
                  ei: np.ndarray,
                  candidates: np.ndarray,
                  selected: np.ndarray,
                  length_scales: np.ndarray,
                  gamma: float = 0.5) -> np.ndarray:
        """
        Reduce EI around already selected points.
        Penalisation follows:  p_i = 1 - exp(-||x - x_i||^2 / (2*l^2*gamma^2))
        """
        for x_i in selected:
            d2 = np.sum(((candidates - x_i) / length_scales) ** 2, axis=1)
            penalty = np.exp(-d2 / (2.0 * gamma ** 2))
            ei *= (1.0 - penalty)          # shrink EI near x_i
        return ei

    # ------------------------------------------------------------------ #
    # Batch proposal
    # ------------------------------------------------------------------ #
    def _propose_batch(self,
                       gp: GaussianProcessRegressor,
                       batch_size: int) -> np.ndarray:
        """Generate a batch using EI, refinement and local penalisation."""
        n_cand = max(5000, 200 * self.dim)
        cand = self._sample(n_cand, use_sobol=False)

        # raw EI on the candidate pool
        ei_raw = self._ei(cand, gp)

        # keep top‑k for refinement (k = batch_size * 5)
        k = min(len(cand), batch_size * 5)
        top_idx = np.argpartition(-ei_raw, k)[:k]
        top_cand = cand[top_idx]

        # refine each of the top points
        refined = []
        for pt in top_cand:
            refined_pt = self._refine_ei(pt, gp, self.bounds.T)
            refined.append(refined_pt)
        refined = np.array(refined)                     # (k, dim)

        # recompute EI on refined set
        ei_refined = self._ei(refined, gp)

        # combine raw and refined pools (avoid duplicates)
        all_cand = np.vstack([cand, refined])
        all_ei   = np.concatenate([ei_raw, ei_refined])

        # local penalisation loop
        selected = []
        length_scales = gp.kernel_.k2.length_scale   # ARD length‑scales

        for _ in range(batch_size):
            # apply penalisation based on already selected points
            penalised_ei = self._penalise(all_ei.copy(),
                                          all_cand,
                                          np.array(selected) if selected else np.empty((0, self.dim)),
                                          length_scales)

            idx = np.argmax(penalised_ei)
            selected.append(all_cand[idx])
            # set EI of the chosen point to -inf to avoid re‑selection
            all_ei[idx] = -np.inf

        return np.vstack(selected)                      # (batch_size, dim)

    # ------------------------------------------------------------------ #
    # Evaluation utilities
    # ------------------------------------------------------------------ #
    def _evaluate(self,
                  func: Callable[[np.ndarray], np.float64],
                  X: np.ndarray) -> np.ndarray:
        """Evaluate `func` row‑wise and update evaluation counter."""
        y = np.apply_along_axis(func, 1, X).astype(np.float64).reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _append_history(self, X_new: np.ndarray, y_new: np.ndarray):
        if self.X is None:
            self.X, self.y = X_new, y_new
        else:
            self.X = np.vstack([self.X, X_new])
            self.y = np.vstack([self.y, y_new])

    # ------------------------------------------------------------------ #
    # Main optimisation interface
    # ------------------------------------------------------------------ #
    def __call__(self,
                 func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        """
        Run the BO loop until the evaluation budget is exhausted.
        Returns (best_value, best_point).
        """
        # ---- 1) Initial design -------------------------------------------------
        init_X = self._sample(self.n_init, use_sobol=True)
        init_y = self._evaluate(func, init_X)
        self._append_history(init_X, init_y)

        best_idx = np.argmin(self.y)
        best_val = self.y[best_idx, 0]
        best_x   = self.X[best_idx, :].copy()

        # ---- 2) Main BO loop ---------------------------------------------------
        while self.n_evals < self.budget:
            remaining = self.budget - self.n_evals

            # adaptive batch size: sqrt of remaining (at least 1)
            batch_sz = max(1, int(np.sqrt(remaining)))
            batch_sz = min(batch_sz, remaining)   # cannot exceed budget

            # (re)train GP only every `retrain_interval` iterations
            if self.iter_counter % self.retrain_interval == 0:
                gp = self._fit_gp(self.X, self.y)
            else:
                # reuse previous gp (already fitted)
                gp = self._fit_gp(self.X, self.y)   # cheap because n_restarts=2

            # propose batch
            next_X = self._propose_batch(gp, batch_sz)

            # evaluate
            next_y = self._evaluate(func, next_X)

            # update history
            self._append_history(next_X, next_y)

            # update best
            cur_best_idx = np.argmin(self.y)
            cur_best_val = self.y[cur_best_idx, 0]
            if cur_best_val < best_val:
                best_val = cur_best_val
                best_x   = self.X[cur_best_idx, :].copy()

            self.iter_counter += 1

        return best_val, best_x
