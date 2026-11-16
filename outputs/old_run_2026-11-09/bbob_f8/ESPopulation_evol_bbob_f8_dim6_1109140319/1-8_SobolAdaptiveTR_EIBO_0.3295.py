from collections.abc import Callable
import numpy as np
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class SobolAdaptiveTR_EIBO:
    """
    Sobol‑Adaptive‑TR‑EI‑BO:
    * Sobol (or LHS) sampling for the initial design.
    * Gaussian‑Process surrogate with Matern‑5/2 ARD kernel.
    * Expected‑Improvement acquisition.
    * Adaptive trust‑region around the current best point.
    * Multi‑restart L‑BFGS‑B optimisation of EI.
    * Periodic GP hyper‑parameter re‑training.
    """

    def __init__(self, budget: int, dim: int,
                 trust_radius_init: float = 2.5,
                 trust_radius_min: float = 0.05,
                 trust_radius_max: float = 5.0,
                 expand_factor: float = 1.5,
                 shrink_factor: float = 0.7,
                 retrain_interval: int = 5):
        """
        Parameters
        ----------
        budget : int
            Maximum number of allowed function evaluations.
        dim : int
            Dimensionality of the problem.
        trust_radius_init : float, optional
            Initial half‑side length of the hyper‑cube trust region (default 2.5).
        trust_radius_min / trust_radius_max : float, optional
            Bounds for the adaptive radius.
        expand_factor / shrink_factor : float, optional
            Multiplicative factors to enlarge/shrink the region after success/failure.
        retrain_interval : int, optional
            How many BO iterations between two GP re‑fits.
        """
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])

        # trust‑region parameters
        self.trust_radius = trust_radius_init
        self.trust_radius_min = trust_radius_min
        self.trust_radius_max = trust_radius_max
        self.expand_factor = expand_factor
        self.shrink_factor = shrink_factor

        self.retrain_interval = max(1, retrain_interval)

        # bookkeeping
        self.X = None          # (n, dim)
        self.y = None          # (n, 1)
        self.n_evals = 0
        self._gp = None        # cached GP model
        self._iter = 0         # BO iteration counter

    # ------------------------------------------------------------------ #
    #  Sampling utilities
    # ------------------------------------------------------------------ #
    def _sample(self, n: int, use_sobol: bool = False) -> np.ndarray:
        """Uniform sampling inside the bounds. Sobol for the first design,
        otherwise Latin‑Hypercube."""
        if use_sobol:
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            unit = sampler.random_base2(m=int(np.ceil(np.log2(n))))
            unit = unit[:n]
        else:
            sampler = qmc.LatinHypercube(d=self.dim)
            unit = sampler.random(n)

        return qmc.scale(unit, self.bounds[0], self.bounds[1])

    # ------------------------------------------------------------------ #
    #  Gaussian‑Process surrogate
    # ------------------------------------------------------------------ #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray):
        """Fit a GP with Matern‑5/2 ARD kernel."""
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(self.dim),
                           length_scale_bounds=(1e-2, 1e2),
                           nu=2.5)
                  + WhiteKernel(noise_level=1e-8,
                                noise_level_bounds=(1e-10, 1e-1)))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      normalize_y=True,
                                      n_restarts_optimizer=3,
                                      random_state=0)
        gp.fit(X, y.ravel())
        return gp

    # ------------------------------------------------------------------ #
    #  Expected Improvement (minimisation)
    # ------------------------------------------------------------------ #
    def _ei(self, X: np.ndarray) -> np.ndarray:
        """Return EI for each row of X (shape (n, 1))."""
        mu, sigma = self._gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        y_best = np.min(self.y)

        with np.errstate(divide='ignore', invalid='ignore'):
            imp = y_best - mu.reshape(-1, 1)          # improvement (positive = good)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    # ------------------------------------------------------------------ #
    #  Trust‑region handling
    # ------------------------------------------------------------------ #
    def _trust_region_bounds(self, centre: np.ndarray) -> np.ndarray:
        """Return lower/upper bounds of the current trust region,
        clipped to the global bounds."""
        lower = np.maximum(centre - self.trust_radius, self.bounds[0])
        upper = np.minimum(centre + self.trust_radius, self.bounds[1])
        return np.vstack([lower, upper]).T   # shape (dim, 2)

    # ------------------------------------------------------------------ #
    #  Acquisition optimisation (multi‑restart L‑BFGS‑B)
    # ------------------------------------------------------------------ #
    def _optimize_ei(self, centre: np.ndarray) -> np.ndarray:
        """Find the point with maximal EI inside the current trust region."""
        tr_bounds = self._trust_region_bounds(centre)

        # generate a few (≈ 5 × dim) Sobol start points inside the trust region
        n_start = max(5 * self.dim, 20)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        unit = sampler.random_base2(m=int(np.ceil(np.log2(n_start))))
        unit = unit[:n_start]
        starts = qmc.scale(unit, tr_bounds[:, 0], tr_bounds[:, 1])

        best_x = None
        best_ei = -np.inf

        # negative EI is minimised
        def neg_ei(x):
            x = x.reshape(1, -1)
            return -self._ei(x)[0, 0]

        for x0 in starts:
            res = minimize(neg_ei,
                           x0,
                           method='L-BFGS-B',
                           bounds=tr_bounds,
                           options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 100})
            if not res.success:
                continue
            ei_val = -res.fun
            if ei_val > best_ei:
                best_ei = ei_val
                best_x = res.x

        # fallback: if optimisation failed, return the best start point
        if best_x is None:
            best_x = starts[np.argmax(self._ei(starts).ravel())]

        return best_x.reshape(1, -1)   # (1, dim)

    # ------------------------------------------------------------------ #
    #  Evaluation utilities
    # ------------------------------------------------------------------ #
    def _evaluate(self, func: Callable[[np.ndarray], float],
                  X: np.ndarray) -> np.ndarray:
        """Evaluate func row‑wise, update evaluation counter."""
        y = np.apply_along_axis(func, 1, X).astype(np.float64)
        y = y.reshape(-1, 1)
        self.n_evals += len(X)
        return y

    def _append_data(self, X_new: np.ndarray, y_new: np.ndarray):
        """Append newly observed points to the history."""
        if self.X is None:
            self.X, self.y = X_new, y_new
        else:
            self.X = np.vstack([self.X, X_new])
            self.y = np.vstack([self.y, y_new])

    # ------------------------------------------------------------------ #
    #  Main optimisation loop
    # ------------------------------------------------------------------ #
    def __call__(self,
                 func: Callable[[np.ndarray], float]) -> tuple[float, np.ndarray]:
        """
        Run the BO algorithm under the prescribed budget.
        Returns (best_value, best_point).
        """
        # ---------- 1) initial Sobol design ----------
        n_init = max(2 * self.dim, 5)
        X_init = self._sample(n_init, use_sobol=True)
        y_init = self._evaluate(func, X_init)
        self._append_data(X_init, y_init)

        # current best
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx].copy()
        best_y = self.y[best_idx, 0]

        # ---------- 2) BO iterations ----------
        while self.n_evals < self.budget:
            self._iter += 1

            # Re‑fit GP only every `retrain_interval` iterations
            if self._gp is None or self._iter % self.retrain_interval == 0:
                self._gp = self._fit_gp(self.X, self.y)

            # -------------------------------------------------
            # 2.1 propose next point inside adaptive trust region
            # -------------------------------------------------
            cand = self._optimize_ei(best_x)          # shape (1, dim)

            # -------------------------------------------------
            # 2.2 evaluate
            # -------------------------------------------------
            y_cand = self._evaluate(func, cand)
            self._append_data(cand, y_cand)

            # -------------------------------------------------
            # 2.3 update best and trust‑region radius
            # -------------------------------------------------
            if y_cand[0, 0] < best_y - 1e-12:          # successful improvement
                best_y = y_cand[0, 0]
                best_x = cand[0].copy()
                self.trust_radius = min(self.trust_radius * self.expand_factor,
                                        self.trust_radius_max)
            else:                                      # no improvement
                self.trust_radius = max(self.trust_radius * self.shrink_factor,
                                        self.trust_radius_min)

        return float(best_y), best_x
