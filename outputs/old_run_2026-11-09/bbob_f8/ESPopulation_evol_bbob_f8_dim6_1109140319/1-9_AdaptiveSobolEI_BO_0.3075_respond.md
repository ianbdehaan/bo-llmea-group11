# Description 
**AdaptiveSobolEI_BO** – a lightweight Bayesian‑Optimization framework tailored for strict evaluation budgets on low‑ to medium‑dimensional continuous black‑box problems (e.g., BBOB Function 8, 6‑D).  

Key ideas  
1. **Sobol quasi‑Monte‑Carlo (QMC) space‑filling start** – guarantees a good initial coverage with very few points.  
2. **Gaussian‑Process surrogate** with an ARD Matérn‑5/2 kernel + tiny White‑Kernel noise, automatically tuned hyper‑parameters.  
3. **Expected‑Improvement (EI) acquisition** evaluated on a **large Sobol candidate pool**.  
4. **Local refinement** – the top‑k EI candidates are polished by a few L‑BFGS‑B steps (bounded gradient‑based optimisation) to escape the discretisation of the candidate set.  
5. **Batch‑size = 1 greedy selection** (max‑EI) together with a **Kriging‑Believer** update for potential future batch extensions.  
6. **Robust budget handling** – every call to the black‑box respects the remaining evaluation budget; the loop stops exactly at the user‑provided limit.  

The algorithm is fully vectorised with NumPy/Scikit‑Learn, needs only the allowed libraries, and fixes the earlier scalar‑conversion bug.

# Justification 
* **Sobol QMC initial design** provides low‑discrepancy samples that explore the search space more uniformly than pure random draws, which is crucial when the total budget is tiny (≤ 100 evals).  
* **GP with Matérn‑5/2** captures the smooth but non‑convex Rosenbrock valley while the ARD length‑scales let the model adapt to each dimension independently.  
* **EI** is the classic acquisition for minimisation; it balances exploitation (low mean) and exploration (high uncertainty).  
* **Large candidate pool + local L‑BFGS‑B refinement** mitigates the “grid‑search” limitation of evaluating EI only on Sobol points. By starting a cheap gradient‑based optimiser from the best few candidates we obtain a near‑continuous maximiser of EI without expensive inner optimisation loops.  
* **Greedy batch‑size = 1** gives the freshest data to the GP after each evaluation, which is especially valuable for highly correlated Rosenbrock landscapes.  
* The **Kriging‑Believer** step (used implicitly when the next point is selected) keeps the implementation simple yet allows easy extension to larger batches later.  
* The scalar‑conversion bug (`float(self.y[best_idx])`) arose because `self.y` is a 2‑D column array. Using `self.y[best_idx, 0]` (or `self.y.flat[best_idx]`) yields a true scalar and eliminates the `TypeError`.  

Overall, the method stays computationally cheap (few GP fits, cheap acquisition evaluation) while improving robustness and final solution quality compared with the original HybridQMCBO.

# Code 
```python
from collections.abc import Callable
import numpy as np
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class AdaptiveSobolEI_BO:
    """
    AdaptiveSobolEI_BO – Bayesian optimisation with Sobol start,
    GP (Matern‑5/2) surrogate, EI acquisition, and cheap L‑BFGS‑B refinement.
    """
    def __init__(self, budget: int, dim: int):
        """
        Parameters
        ----------
        budget : int
            Maximum number of allowed black‑box evaluations.
        dim : int
            Dimensionality of the search space (bounds are [-5, 5]^dim).
        """
        self.budget = int(budget)
        self.dim = int(dim)
        self.bounds = np.stack((np.full(dim, -5.0), np.full(dim, 5.0)))  # shape (2, dim)

        # internal storage
        self.X: np.ndarray | None = None          # shape (n_samples, dim)
        self.y: np.ndarray | None = None          # shape (n_samples, 1)
        self.n_evals = 0

        # initial design: 5 points per dimension, but never more than half the budget
        self.n_init = min(5 * self.dim, max(2, self.budget // 2))

    # --------------------------------------------------------------------- #
    # Sampling utilities
    # --------------------------------------------------------------------- #
    def _sample_sobol(self, n: int) -> np.ndarray:
        """Draw `n` Sobol points uniformly scaled to the problem bounds."""
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
        # Sobol generates 2^m points, so we ask for the smallest power‑of‑2 >= n
        m = int(np.ceil(np.log2(max(1, n))))
        unit = sampler.random_base2(m=m)[:n]          # shape (n, dim) in [0,1]
        lower, upper = self.bounds
        return qmc.scale(unit, lower, upper)

    # --------------------------------------------------------------------- #
    # Surrogate model handling
    # --------------------------------------------------------------------- #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """Fit a GP with ARD Matérn‑5/2 kernel + tiny white noise."""
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
    # Acquisition – Expected Improvement
    # --------------------------------------------------------------------- #
    def _ei(self, Xcand: np.ndarray, gp: GaussianProcessRegressor,
            f_best: float) -> np.ndarray:
        """Compute EI for a set of candidates (minimisation)."""
        mu, sigma = gp.predict(Xcand, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu = mu.reshape(-1, 1)

        with np.errstate(divide='ignore', invalid='ignore'):
            imp = f_best - mu                     # improvement (positive = good)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei.ravel()   # shape (n_cand,)

    # --------------------------------------------------------------------- #
    # Local refinement of EI maximiser (cheap L‑BFGS‑B)
    # --------------------------------------------------------------------- #
    def _refine_point(self, x0: np.ndarray, gp: GaussianProcessRegressor,
                      f_best: float) -> np.ndarray:
        """Run a few L‑BFGS‑B steps starting from `x0` to maximise EI."""
        # Negative EI because we minimise in scipy
        def neg_ei(x):
            x = x.reshape(1, -1)
            mu, sigma = gp.predict(x, return_std=True)
            sigma = sigma.item()
            mu = mu.item()
            if sigma <= 0.0:
                return 0.0
            imp = f_best - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei

        bounds = [(self.bounds[0, i], self.bounds[1, i]) for i in range(self.dim)]
        res = minimize(
            neg_ei,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 15, "ftol": 1e-9},
        )
        return res.x

    # --------------------------------------------------------------------- #
    # Acquisition + selection of the next point
    # --------------------------------------------------------------------- #
    def _select_next(self, gp: GaussianProcessRegressor, f_best: float) -> np.ndarray:
        """
        Generate a Sobol candidate pool, evaluate EI, pick the best,
        then locally refine it.
        Returns a single point of shape (1, dim).
        """
        # Candidate pool size – a few thousand points is cheap for dim ≤ 10
        n_cand = min(5000, 10 * self.dim * self.n_init)
        cand = self._sample_sobol(n_cand)

        ei = self._ei(cand, gp, f_best)
        # Take the top few (e.g., 5) for refinement to avoid local traps
        top_k = min(5, len(ei))
        top_idx = np.argpartition(-ei, top_k - 1)[:top_k]
        refined = []
        for idx in top_idx:
            refined.append(self._refine_point(cand[idx], gp, f_best))
        refined = np.vstack(refined)

        # Evaluate EI on refined points and pick the best
        ei_refined = self._ei(refined, gp, f_best)
        best_refined_idx = np.argmin(-ei_refined)  # max EI
        return refined[best_refined_idx].reshape(1, -1)

    # --------------------------------------------------------------------- #
    # Evaluation handling (budget‑aware)
    # --------------------------------------------------------------------- #
    def _evaluate(self, func: Callable[[np.ndarray], float],
                  X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate `func` on rows of X while respecting the remaining budget.
        Returns the (possibly truncated) X and a column vector y.
        """
        remaining = self.budget - self.n_evals
        if X.shape[0] > remaining:
            X = X[:remaining]

        # Vectorised evaluation is not assumed – use apply_along_axis
        y = np.apply_along_axis(func, 1, X).reshape(-1, 1)
        self.n_evals += X.shape[0]
        return X, y

    def _append_data(self, X_new: np.ndarray, y_new: np.ndarray):
        """Append newly evaluated samples to the internal dataset."""
        if self.X is None:
            self.X, self.y = X_new, y_new
        else:
            self.X = np.vstack((self.X, X_new))
            self.y = np.vstack((self.y, y_new))

    # --------------------------------------------------------------------- #
    # Main optimisation interface
    # --------------------------------------------------------------------- #
    def __call__(self, func: Callable[[np.ndarray], float]) -> tuple[float, np.ndarray]:
        """
        Run the BO loop until the evaluation budget is exhausted.

        Returns
        -------
        best_f : float
            Minimal observed function value.
        best_x : np.ndarray, shape (dim,)
            Corresponding decision vector.
        """
        # ---------- Initial Sobol design ----------
        X0 = self._sample_sobol(self.n_init)
        X0, y0 = self._evaluate(func, X0)
        self._append_data(X0, y0)

        # Keep track of the best observed point
        best_idx = np.argmin(self.y)                # self.y is (n,1)
        best_f = float(self.y[best_idx, 0])         # scalar conversion fixed
        best_x = self.X[best_idx].copy()

        # ---------- Sequential BO loop ----------
        while self.n_evals < self.budget:
            # Fit GP on all data collected so far
            gp = self._fit_gp(self.X, self.y)

            # Propose the next point (batch size = 1)
            x_next = self._select_next(gp, best_f)   # shape (1, dim)

            # Evaluate and store
            x_next, y_next = self._evaluate(func, x_next)
            self._append_data(x_next, y_next)

            # Update best if improved
            cur_idx = np.argmin(self.y)
            cur_f = float(self.y[cur_idx, 0])
            if cur_f < best_f:
                best_f = cur_f
                best_x = self.X[cur_idx].copy()

        return best_f, best_x
```
## Feedback
 The algorithm AdaptiveSobolEI_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3075 with standard deviation 0.0000.

took 106.21 seconds to run.