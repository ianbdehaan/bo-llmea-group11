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
- SparseGPBO: 0.0000, 0.00 seconds, **  
A **Sparse‑GP‑BO** algorithm (named `SparseGPBO`) that combines  

* **Quasi‑Monte‑Carlo Sobol sampling** for a space‑filling initial design,  
* a **Gaussian Process surrogate** with a Matern‑5/2 kernel (fast‑to‑train, good for smooth but non‑convex landscapes like Rosenbrock),  
* **Expected Improvement (EI)** as acquisition function, and  
* a **batch‑wise Kriging‑Believer** selection that re‑uses the GP posterior to propose several points per iteration without costly inner optimisation.  

The method is computationally efficient (few GP hyper‑parameter optimisations, cheap acquisition evaluation on a modest candidate set) and works well on low‑ to medium‑dimensional black‑box problems such as BBOB Function 8 (Rosenbrock, 6‑D) within a strict evaluation budget.

---

**
- SobolMaternEI_BO: 0.0000, 0.00 seconds, **  
A **Sobol‑Matern‑EI BO** algorithm – a Bayesian optimisation method that combines  

* **Quasi‑Monte‑Carlo Sobol sampling** for a space‑filling initial design,  
* a **Gaussian Process surrogate** with a **Matern‑5/2 kernel** (flexible smoothness, good for Rosenbrock‑type valleys),  
* the **Expected‑Improvement (EI)** acquisition function, and  
* a **batch‑wise Kriging‑Believer** strategy with cheap L‑BFGS‑B local refinement to propose the next evaluation point(s).  

The method is lightweight, fully vectorised with NumPy, and respects the strict evaluation budget required by the BBOB test suite.

---

**
- HybridQMCBO: 0.0000, 0.00 seconds, **  
A **HybridQMC‑BO** algorithm – a Bayesian Optimizer that combines  

* **Quasi‑Monte‑Carlo (Sobol) initial design** for space‑filling coverage,  
* **Gaussian Process surrogate** with a Matern‑5/2 kernel (automatic ARD),  
* **Expected‑Improvement (EI) acquisition** evaluated on a large Sobol candidate set,  
* **Batch‑size = 1 greedy selection** (the best EI point) to keep computational overhead low.  

The method is well‑suited for low‑ to medium‑dimensional continuous black‑box problems such as BBOB Function 8 (Rosenbrock, 6‑D) and respects a strict evaluation budget.

---

**


The selected solution to update is:
**  
A **HybridQMC‑BO** algorithm – a Bayesian Optimizer that combines  

* **Quasi‑Monte‑Carlo (Sobol) initial design** for space‑filling coverage,  
* **Gaussian Process surrogate** with a Matern‑5/2 kernel (automatic ARD),  
* **Expected‑Improvement (EI) acquisition** evaluated on a large Sobol candidate set,  
* **Batch‑size = 1 greedy selection** (the best EI point) to keep computational overhead low.  

The method is well‑suited for low‑ to medium‑dimensional continuous black‑box problems such as BBOB Function 8 (Rosenbrock, 6‑D) and respects a strict evaluation budget.

---

**
With code:
```python
from collections.abc import Callable
from scipy.stats import qmc  # Sobol low‑discrepancy sequence
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class HybridQMCBO:
    """
    HybridQMCBO – Bayesian optimisation with Sobol initial design,
    GP surrogate (Matern‑5/2) and Expected Improvement acquisition.
    """
    def __init__(self, budget: int, dim: int):
        self.budget = budget               # total allowed function evaluations
        self.dim = dim                     # problem dimensionality
        # bounds has shape (2, <dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0                    # number of function evaluations performed
        # initial design size – 5 points per dimension but never exceed half the budget
        self.n_init = min(5 * dim, max(2, budget // 2))

    # --------------------------------------------------------------------- #
    # Sampling utilities
    # --------------------------------------------------------------------- #
    def _sample_points(self, n_points: int) -> np.ndarray:
        """
        Sample `n_points` uniformly in the search space using a Sobol QMC sequence.
        Returns an array of shape (n_points, dim).
        """
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol generates points in [0,1]^d; scale to bounds
        unit_samples = sampler.random_base2(m=int(np.ceil(np.log2(n_points))))
        unit_samples = unit_samples[:n_points]               # keep exactly n_points
        lower, upper = self.bounds
        samples = qmc.scale(unit_samples, lower, upper)
        return samples

    # --------------------------------------------------------------------- #
    # Surrogate model handling
    # --------------------------------------------------------------------- #
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """
        Fit a Gaussian Process with a Matern‑5/2 kernel (ARD) and a small noise term.
        Returns the fitted GP model.
        """
        # Constant * Matern + WhiteKernel (noise) – hyper‑parameters are optimized by default
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
                 Matern(length_scale=np.ones(self.dim), length_scale_bounds=(1e-2, 1e2), nu=2.5) + \
                 WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-1))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      n_restarts_optimizer=5,
                                      normalize_y=True,
                                      random_state=42)
        gp.fit(X, y.ravel())
        return gp

    # --------------------------------------------------------------------- #
    # Acquisition – Expected Improvement
    # --------------------------------------------------------------------- #
    def _acquisition_function(self, X: np.ndarray, gp: GaussianProcessRegressor,
                              y_best: float) -> np.ndarray:
        """
        Compute Expected Improvement (EI) for each candidate in X.
        Returns an array of shape (n_points, 1).
        """
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu = mu.reshape(-1, 1)

        # Avoid division by zero
        with np.errstate(divide='ignore'):
            imp = y_best - mu          # improvement (minimisation)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    # --------------------------------------------------------------------- #
    # Selection of the next evaluation point
    # --------------------------------------------------------------------- #
    def _select_next_points(self, batch_size: int, gp: GaussianProcessRegressor,
                            y_best: float) -> np.ndarray:
        """
        Generate a large Sobol candidate set, evaluate EI, and return the
        `batch_size` points with highest EI.
        """
        # Candidate pool size – a few thousand points is cheap and enough for dim ≤ 10
        n_candidates = min(5000, 10 * self.dim * self.n_init)
        candidates = self._sample_points(n_candidates)

        ei = self._acquisition_function(candidates, gp, y_best)
        # Pick the top points (greedy, batch size = 1 in practice)
        idx = np.argpartition(-ei.ravel(), batch_size)[:batch_size]
        return candidates[idx]

    # --------------------------------------------------------------------- #
    # Evaluation handling
    # --------------------------------------------------------------------- #
    def _evaluate_points(self, func: Callable[[np.ndarray], np.float64],
                         X: np.ndarray) -> np.ndarray:
        """
        Evaluate the black‑box function on each row of X.
        Returns an array of shape (n_points, 1).
        """
        # Respect the global budget – truncate if needed
        remaining = self.budget - self.n_evals
        if X.shape[0] > remaining:
            X = X[:remaining]

        y_vals = np.apply_along_axis(func, 1, X).reshape(-1, 1)
        self.n_evals += X.shape[0]
        return X, y_vals

    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        """
        Append newly evaluated points to the internal data set.
        """
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack((self.X, new_X))
            self.y = np.vstack((self.y, new_y))

    # --------------------------------------------------------------------- #
    # Main optimisation interface
    # --------------------------------------------------------------------- #
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        """
        Run the optimisation loop.
        Returns (best_y, best_x) where best_y is the minimal observed function value.
        """
        # ------------------- initial design ------------------- #
        X_init = self._sample_points(self.n_init)
        X_init, y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Track the best solution seen so far
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()

        # ------------------- sequential BO loop ------------------- #
        while self.n_evals < self.budget:
            # Fit surrogate on all collected data
            gp = self._fit_model(self.X, self.y)

            # Choose next point(s) – we use batch_size = 1 for maximal feedback
            batch_size = 1
            next_X = self._select_next_points(batch_size, gp, best_y)

            # Evaluate and update
            next_X, next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Update best observed value
            cur_best_idx = np.argmin(self.y)
            cur_best_y = float(self.y[cur_best_idx])
            if cur_best_y < best_y:
                best_y = cur_best_y
                best_x = self.X[cur_best_idx].copy()

        return best_y, best_x

```
An error occurred : Traceback (most recent call last):
  File "<HybridQMCBO>", line 145, in __call__
 143 |         # Track the best solution seen so far
 144 |         best_idx = np.argmin(self.y)
 145->         best_y = float(self.y[best_idx])
 146 |         best_x = self.X[best_idx].copy()
 147 | 
TypeError: only 0-dimensional arrays can be converted to Python scalars


Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

