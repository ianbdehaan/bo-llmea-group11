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
**  
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
With code:
```python
from collections.abc import Callable
import numpy as np
from scipy.stats import qmc, norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class AdaptiveSobolEI_BO:
    """
    AdaptiveSobolEI_BO – Bayesian optimisation with Sobol initialization,
    GP (Matern‑5/2) surrogate and Expected‑Improvement acquisition.
    A small L‑BFGS‑B refinement of the best Sobol candidates is performed
    before the greedy selection (batch size = 1).
    """

    def __init__(self, budget: int, dim: int):
        self.budget = budget                # total allowed evaluations
        self.dim = dim
        self.bounds = np.column_stack((np.full(dim, -5.0),
                                       np.full(dim, 5.0)))   # shape (dim, 2)

        # data containers
        self.X: np.ndarray = None           # (n_samples, dim)
        self.y: np.ndarray = None           # (n_samples, 1)
        self.n_evals = 0

        # initial design: 5 points per dimension, but never more than half the budget
        self.n_init = min(5 * dim, max(2, budget // 2))

    # ------------------------------------------------------------------ #
    # Sampling utilities
    # ------------------------------------------------------------------ #
    def _sobol_samples(self, n: int) -> np.ndarray:
        """Generate `n` Sobol points scaled to the problem bounds."""
        # Sobol works with powers of two; we oversample and then truncate
        m = int(np.ceil(np.log2(n)))
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        unit = sampler.random_base2(m=m)
        unit = unit[:n]                     # keep exactly n points
        return qmc.scale(unit, self.bounds[:, 0], self.bounds[:, 1])

    # ------------------------------------------------------------------ #
    # Surrogate model
    # ------------------------------------------------------------------ #
    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """Fit a GP with ARD Matern‑5/2 kernel + tiny white noise."""
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(self.dim),
                           length_scale_bounds=(1e-2, 1e2), nu=2.5)
                  + WhiteKernel(noise_level=1e-8,
                                noise_level_bounds=(1e-10, 1e-1)))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      n_restarts_optimizer=5,
                                      normalize_y=True,
                                      random_state=0)
        gp.fit(X, y.ravel())
        return gp

    # ------------------------------------------------------------------ #
    # Expected Improvement
    # ------------------------------------------------------------------ #
    def _ei(self, X: np.ndarray, gp: GaussianProcessRegressor,
            y_best: float) -> np.ndarray:
        """Vectorised Expected Improvement for minimisation."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu = mu.reshape(-1, 1)

        with np.errstate(divide='ignore', invalid='ignore'):
            imp = y_best - mu                     # improvement (minimisation)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei.ravel()                         # 1‑D array

    # ------------------------------------------------------------------ #
    # Acquisition optimisation (refinement)
    # ------------------------------------------------------------------ #
    def _refine_candidates(self, cand: np.ndarray, gp: GaussianProcessRegressor,
                           y_best: float, n_refine: int = 3) -> np.ndarray:
        """
        Starting from the `n_refine` best Sobol candidates, run a short
        L‑BFGS‑B optimisation of the negative EI (bounds are respected).
        Returns the refined points (still inside bounds).
        """
        # keep the n_refine candidates with highest EI
        ei_vals = self._ei(cand, gp, y_best)
        top_idx = np.argpartition(-ei_vals, n_refine)[:n_refine]
        refined = []

        bounds = tuple(self.bounds[i] for i in range(self.dim))

        for x0 in cand[top_idx]:
            # minimise -EI  <=> maximise EI
            res = minimize(lambda x: -self._ei(x.reshape(1, -1), gp, y_best)[0],
                           x0=x0,
                           method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 20, 'ftol': 1e-6})
            refined.append(res.x)

        return np.vstack(refined)

    # ------------------------------------------------------------------ #
    # Evaluation handling (budget‑aware)
    # ------------------------------------------------------------------ #
    def _evaluate(self, func: Callable[[np.ndarray], float],
                  X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate `func` on rows of X respecting the remaining budget."""
        remaining = self.budget - self.n_evals
        if X.shape[0] > remaining:
            X = X[:remaining]

        y_vals = np.apply_along_axis(func, 1, X).reshape(-1, 1)
        self.n_evals += X.shape[0]
        return X, y_vals

    def _append_data(self, X_new: np.ndarray, y_new: np.ndarray):
        """Add newly evaluated points to the internal dataset."""
        if self.X is None:
            self.X, self.y = X_new, y_new
        else:
            self.X = np.vstack((self.X, X_new))
            self.y = np.vstack((self.y, y_new))

    # ------------------------------------------------------------------ #
    # Main optimisation interface
    # ------------------------------------------------------------------ #
    def __call__(self,
                 func: Callable[[np.ndarray], float]) -> tuple[float, np.ndarray]:
        """
        Run the BO loop.
        Returns (best_y, best_x) where best_y is the minimal observed value.
        """

        # ---------- initial Sobol design ----------
        X0 = self._sobol_samples(self.n_init)
        X0, y0 = self._evaluate(func, X0)
        self._append_data(X0, y0)

        best_idx = int(np.argmin(self.y))
        best_y = float(self.y[best_idx, 0])
        best_x = self.X[best_idx].copy()

        # ---------- sequential BO ----------
        while self.n_evals < self.budget:
            # Fit GP on all data collected so far
            gp = self._fit_gp(self.X, self.y)

            # Build a large Sobol candidate pool (cheap)
            n_pool = min(5000, 10 * self.dim * self.n_init)
            pool = self._sobol_samples(n_pool)

            # Refine the top few candidates (typically 3)
            refined = self._refine_candidates(pool, gp, best_y, n_refine=3)

            # Evaluate EI on the refined set and pick the best point
            ei_refined = self._ei(refined, gp, best_y)
            next_x = refined[np.argmin(ei_refined)]  # min because we used -EI in refine

            # Ensure shape (1, dim) for evaluation routine
            next_x = next_x.reshape(1, -1)

            # Evaluate the black‑box function (budget‑aware)
            next_x, next_y = self._evaluate(func, next_x)
            self._append_data(next_x, next_y)

            # Update incumbent
            cur_best_idx = int(np.argmin(self.y))
            cur_best_y = float(self.y[cur_best_idx, 0])
            if cur_best_y < best_y:
                best_y = cur_best_y
                best_x = self.X[cur_best_idx].copy()

        return best_y, best_x

```
The algorithm AdaptiveSobolEI_BO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3189 with standard deviation 0.0166.

took 93.23 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

