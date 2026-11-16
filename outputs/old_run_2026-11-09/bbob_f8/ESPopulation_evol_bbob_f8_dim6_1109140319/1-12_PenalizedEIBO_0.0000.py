# -*- coding: utf-8 -*-
"""
PenalizedEI_BO
----------------
A refined Bayesian Optimisation algorithm for low‑budget continuous problems.
It extends SobolEIHybridBO with:
  * hybrid (global+local) candidate generation,
  * local‑penalised batch selection,
  * adaptive batch size,
  * occasional GP hyper‑parameter re‑optimisation.

Author: <your name>
"""

from collections.abc import Callable
import numpy as np
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

# --------------------------------------------------------------------------- #
# Helper: penalisation factor based on GP length‑scale (González et al., 2016)
# --------------------------------------------------------------------------- #
def _penalisation_factor(x, selected, lengthscales, eta=1.0):
    """
    Compute a multiplicative penalty for point x given already selected points.
    The penalty is exp( - (||x - x_i|| / (eta * l_i))^2 ), where l_i is the
    ARD length‑scale for dimension i.
    """
    if selected.size == 0:
        return 1.0
    # broadcast distance computation
    diff = x - selected   # shape (n_selected, dim)
    scaled = diff / lengthscales   # ARD scaling
    d2 = np.sum(scaled ** 2, axis=1)   # squared Mahalanobis distance
    return np.exp(-d2 / (eta ** 2)).max()   # max over selected points


class PenalizedEIBO:
    """
    PenalizedEI‑BO: Bayesian Optimisation with Sobol init, Matern‑5/2 GP,
    hybrid global+local candidate pools and locally‑penalised batch EI.
    """

    # --------------------------------------------------------------------- #
    #  Construction
    # --------------------------------------------------------------------- #
    def __init__(self, budget: int, dim: int):
        self.budget = budget                # total allowed evaluations
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])   # (2, dim)

        # data containers
        self.X: np.ndarray = None           # (n, dim)
        self.y: np.ndarray = None           # (n, 1)
        self.n_evals = 0

        # design parameters
        self.n_init = max(2 * dim, 5)        # Sobol initial points
        self._gp: GaussianProcessRegressor | None = None
        self._last_hyper_opt = -1           # iteration of last hyper‑opt

    # --------------------------------------------------------------------- #
    #  Sampling utilities
    # --------------------------------------------------------------------- #
    def _global_pool(self, n: int) -> np.ndarray:
        """Sobol QMC pool covering the whole search space."""
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol works best with power‑of‑2 sizes; we round up then slice.
        m = int(np.ceil(np.log2(n)))
        unit = sampler.random_base2(m=m)[:n]
        return qmc.scale(unit, self.bounds[0], self.bounds[1])

    def _local_pool(self, n: int, centre: np.ndarray, lengthscales: np.ndarray) -> np.ndarray:
        """
        Latin‑Hypercube pool inside a hyper‑rectangle centred at `centre`
        with half‑width = 2 * lengthscale (trust‑region).
        """
        radius = 2.0 * lengthscales      # shape (dim,)
        low = np.maximum(self.bounds[0], centre - radius)
        high = np.minimum(self.bounds[1], centre + radius)

        sampler = qmc.LatinHypercube(d=self.dim)
        unit = sampler.random(n)                     # in [0,1]^d
        scaled = low + unit * (high - low)           # affine map
        return scaled

    # --------------------------------------------------------------------- #
    #  Surrogate model
    # --------------------------------------------------------------------- #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray, optimise: bool = True) -> GaussianProcessRegressor:
        """Create / update the GP. Hyper‑parameters are optimised only if `optimise`. """
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(self.dim),
                           length_scale_bounds=(1e-2, 1e2),
                           nu=2.5)
                  + WhiteKernel(noise_level=1e-8,
                                noise_level_bounds=(1e-10, 1e-1)))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      normalize_y=True,
                                      n_restarts_optimizer=5 if optimise else 0,
                                      random_state=0)
        gp.fit(X, y.ravel())
        return gp

    # --------------------------------------------------------------------- #
    #  Expected Improvement
    # --------------------------------------------------------------------- #
    def _ei(self, X: np.ndarray, gp: GaussianProcessRegressor) -> np.ndarray:
        """Vectorised EI for minimisation."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)

        y_best = np.min(self.y)                     # current best (min)
        imp = y_best - mu.reshape(-1, 1)            # improvement (positive = good)
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei.ravel()                           # (n,)

    # --------------------------------------------------------------------- #
    #  Penalised batch selection
    # --------------------------------------------------------------------- #
    def _select_batch(self,
                      pool: np.ndarray,
                      gp: GaussianProcessRegressor,
                      batch_size: int) -> np.ndarray:
        """
        Select `batch_size` points from `pool` using EI + local penalisation.
        No GP re‑fit is performed inside the loop.
        """
        # retrieve ARD length‑scales (array of shape (dim,))
        lengthscales = gp.kernel_.k2.length_scale
        selected = np.empty((0, self.dim))

        # copy EI values – we will modify them in‑place
        ei_vals = self._ei(pool, gp)

        for _ in range(batch_size):
            # apply penalisation to all candidates
            if selected.shape[0] > 0:
                penalties = np.apply_along_axis(
                    _penalisation_factor,
                    1,
                    pool,
                    selected,
                    lengthscales,
                    eta=1.0
                )
                penalised_ei = ei_vals * penalties
            else:
                penalised_ei = ei_vals

            idx = np.argmax(penalised_ei)
            chosen = pool[idx:idx + 1]               # (1, dim)
            selected = np.vstack([selected, chosen])

            # after picking a point we set its EI to -inf so it cannot be re‑chosen
            ei_vals[idx] = -np.inf

        return selected                               # (batch_size, dim)

    # --------------------------------------------------------------------- #
    #  Evaluation utilities
    # --------------------------------------------------------------------- #
    def _evaluate(self, func: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
        """Evaluate `func` row‑wise, update evaluation counter."""
        y = np.apply_along_axis(func, 1, X).astype(np.float64)
        y = y.reshape(-1, 1)
        self.n_evals += X.shape[0]
        return y

    def _append_data(self, X_new: np.ndarray, y_new: np.ndarray):
        if self.X is None:
            self.X, self.y = X_new, y_new
        else:
            self.X = np.vstack([self.X, X_new])
            self.y = np.vstack([self.y, y_new])

    # --------------------------------------------------------------------- #
    #  Main optimisation loop
    # --------------------------------------------------------------------- #
    def __call__(self, func: Callable[[np.ndarray], float]) -> tuple[float, np.ndarray]:
        """
        Run the optimizer under the supplied evaluation budget.
        Returns (best_value, best_point).
        """
        # --------------------------------------------------------------- #
        # 1) Initial Sobol design
        # --------------------------------------------------------------- #
        init_X = self._global_pool(self.n_init)
        init_y = self._evaluate(func, init_X)
        self._append_data(init_X, init_y)

        best_idx = np.argmin(self.y)
        best_val = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()

        iteration = 0

        # --------------------------------------------------------------- #
        # 2) Main BO loop
        # --------------------------------------------------------------- #
        while self.n_evals < self.budget:
            remaining = self.budget - self.n_evals

            # ---- adaptive batch size (1 → 3) -------------------------------- #
            if remaining > 3 * self.dim:
                batch_size = 3
            elif remaining > self.dim:
                batch_size = 2
            else:
                batch_size = 1

            # ---- decide whether to re‑optimise GP hyper‑parameters  --------- #
            # re‑optimise every ceil(dim/2) iterations (or on the first loop)
            optimise_hyper = (iteration - self._last_hyper_opt) >= max(1, self.dim // 2)

            # fit / update GP
            self._gp = self._fit_gp(self.X, self.y, optimise=optimise_hyper)
            if optimise_hyper:
                self._last_hyper_opt = iteration

            # ---- build hybrid candidate pool -------------------------------- #
            n_global = max(2000, 200 * self.dim)   # enough to keep exploration
            global_pool = self._global_pool(n_global)

            # local pool centred on current best, radius = 2 * lengthscale
            lengthscales = self._gp.kernel_.k2.length_scale
            n_local = max(500, 50 * self.dim)
            local_pool = self._local_pool(n_local, best_x, lengthscales)

            # merge and shuffle (shuffle avoids bias toward one pool)
            pool = np.vstack([global_pool, local_pool])
            np.random.shuffle(pool)

            # ---- batch selection with penalisation -------------------------- #
            next_X = self._select_batch(pool, self._gp, batch_size)

            # ---- evaluate --------------------------------------------------- #
            next_y = self._evaluate(func, next_X)
            self._append_data(next_X, next_y)

            # ---- update best ------------------------------------------------ #
            cur_best_idx = np.argmin(self.y)
            cur_best_val = float(self.y[cur_best_idx])
            if cur_best_val < best_val:
                best_val = cur_best_val
                best_x = self.X[cur_best_idx].copy()

            iteration += 1

        return best_val, best_x
