You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of BBOB Function 8 (Rosenbrock, 6D). Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- SobolEIHybridBO: 0.3335, 91.23 seconds, **  
**Sobol‑EI‑HybridBO** – a Bayesian‑Optimization framework that combines low‑discrepancy Sobol quasi‑Monte‑Carlo (QMC) initialization, a scalable Gaussian‑Process surrogate (Matern‑5/2 kernel) and an Expected‑Improvement (EI) acquisition.  After a cheap global Sobol design, the algorithm repeatedly (i) fits the GP, (ii) draws a large pool of candidate points with cheap Latin‑Hypercube sampling, (iii) evaluates EI on the pool, and (iv) selects a small batch (default = 1) of the highest‑EI points.  The batch is chosen by a simple “Kriging‑believer” update (the GP mean of already‑chosen points is used as a surrogate for their unknown function values) to keep the selection cheap while still providing a modest diversification effect.  The whole loop stops exactly at the user‑provided evaluation budget.

---

**
- SobolAdaptiveTR_EIBO: 0.3295, 691.21 seconds, **Sobol‑Adaptive‑TR‑EI‑BO** – a Bayesian‑Optimization framework that builds on the Sobol‑EI‑HybridBO idea but adds three key refinements that are known to speed up convergence on narrow‑valley functions such as the 6‑D Rosenbrock (BBOB f8):

1. **Adaptive trust‑region** around the current best solution.  
   The search region is a hyper‑cube whose radius is enlarged after a successful improvement and shrunk after a failure, keeping the optimisation focussed on the promising valley while still allowing global exploration when the radius is large.

2. **Multi‑restart local optimisation of Expected‑Improvement**.  
   Instead of evaluating EI on a huge random pool, a modest Sobol/LHS pool is used only to generate starting points for a gradient‑based L‑BFGS‑B optimiser (minimise –EI).  The best local optimum is taken as the next evaluation point.  This gives a much more accurate maximisation of EI for the same computational budget.

3. **Periodic GP‑hyper‑parameter re‑training**.  
   GP fitting is expensive; we refit the GP only every `retrain_interval` iterations (default 5) and otherwise reuse the previous surrogate, updating only the training data.  This reduces overhead while still tracking the landscape.

The algorithm respects the strict evaluation budget, works for any dimensionality, and stays within the allowed libraries (NumPy, SciPy, scikit‑learn).  


- AdaptiveSobolEI_BO: 0.3189, 93.23 seconds, **  
**AdaptiveSobolEI_BO** – a Bayesian‑Optimization algorithm that  

* starts with a Sobol quasi‑Monte‑Carlo (QMC) space‑filling design,  
* builds a Gaussian‑Process (GP) surrogate with an ARD Matern‑5/2 kernel,  
* evaluates the Expected‑Improvement (EI) acquisition on a large Sobol candidate pool,  
* refines the *k* best candidates with a cheap local L‑BFGS‑B optimisation of the (negative) EI, and  
* selects the single point with the highest refined EI (batch‑size = 1).  

The loop repeats until the user‑provided evaluation budget is exhausted.  
All operations are fully vectorised with NumPy / SciPy and respect the strict budget.

---

**
- AdaptiveSobolEI_BO: 0.3086, 126.13 seconds, **AdaptiveSobolEI_BO** – a Bayesian‑Optimization algorithm that  

* starts with a space‑filling Sobol quasi‑Monte‑Carlo design,  
* builds an ARD Matern‑5/2 Gaussian‑Process surrogate,  
* uses **Expected Improvement (EI)** as acquisition,  
* evaluates EI on a large Sobol candidate pool and **refines the top‑k candidates with L‑BFGS‑B** (gradient‑based local optimisation of the acquisition),  
* selects a single point per iteration (batch = 1) so that every evaluation immediately updates the model,  
* respects the strict evaluation budget and works for any dimensionality inside the \([-5,5]^d\) box.

The algorithm is computationally light (few GP hyper‑parameter optimisations, cheap candidate generation) yet gains extra exploitation power from the local refinement of EI, which is especially helpful on the narrow Rosenbrock valley of BBOB Function 8.

---




The selected solution to update is:
**Sobol‑Adaptive‑TR‑EI‑BO** – a Bayesian‑Optimization framework that builds on the Sobol‑EI‑HybridBO idea but adds three key refinements that are known to speed up convergence on narrow‑valley functions such as the 6‑D Rosenbrock (BBOB f8):

1. **Adaptive trust‑region** around the current best solution.  
   The search region is a hyper‑cube whose radius is enlarged after a successful improvement and shrunk after a failure, keeping the optimisation focussed on the promising valley while still allowing global exploration when the radius is large.

2. **Multi‑restart local optimisation of Expected‑Improvement**.  
   Instead of evaluating EI on a huge random pool, a modest Sobol/LHS pool is used only to generate starting points for a gradient‑based L‑BFGS‑B optimiser (minimise –EI).  The best local optimum is taken as the next evaluation point.  This gives a much more accurate maximisation of EI for the same computational budget.

3. **Periodic GP‑hyper‑parameter re‑training**.  
   GP fitting is expensive; we refit the GP only every `retrain_interval` iterations (default 5) and otherwise reuse the previous surrogate, updating only the training data.  This reduces overhead while still tracking the landscape.

The algorithm respects the strict evaluation budget, works for any dimensionality, and stays within the allowed libraries (NumPy, SciPy, scikit‑learn).  


With code:
```python
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

```
The algorithm SobolAdaptiveTR_EIBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3295 with standard deviation 0.0235.

took 691.21 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

