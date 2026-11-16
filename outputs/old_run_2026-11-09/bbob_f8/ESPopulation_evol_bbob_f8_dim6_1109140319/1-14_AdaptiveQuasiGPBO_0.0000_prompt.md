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
A **Sobol‑Matern‑EI BO** algorithm – a Bayesian optimisation method that combines  

* **Quasi‑Monte‑Carlo Sobol sampling** for a space‑filling initial design,  
* a **Gaussian Process surrogate** with a **Matern‑5/2 kernel** (flexible smoothness, good for Rosenbrock‑type valleys),  
* the **Expected‑Improvement (EI)** acquisition function, and  
* a **batch‑wise Kriging‑Believer** strategy with cheap L‑BFGS‑B local refinement to propose the next evaluation point(s).  

The method is lightweight, fully vectorised with NumPy, and respects the strict evaluation budget required by the BBOB test suite.

---

**
With code:
```python
from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.optimize import minimize

class SobolMaternEI_BO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget                # total allowed evaluations
        self.dim = dim
        # bounds has shape (2, dim); bounds[0] = lower, bounds[1] = upper
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None           # evaluated points, shape (n, dim)
        self.y: np.ndarray = None           # corresponding function values, shape (n, 1)
        self.n_evals = 0                    # counter of used evaluations

        # initial design size – at least 2*dim but never larger than budget/2
        self.n_init = min(max(2 * dim, 5), budget // 2)

    # --------------------------------------------------------------------- #
    # 1. Sampling (Sobol QMC)
    # --------------------------------------------------------------------- #
    def _sample_points(self, n_points: int) -> np.ndarray:
        """Return `n_points` Sobol samples scaled to the problem bounds."""
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol generates points in [0,1]^d; transform to bounds
        raw = sampler.random_base2(m=int(np.ceil(np.log2(n_points))))
        raw = raw[:n_points]                     # keep exactly n_points
        lower, upper = self.bounds
        return lower + raw * (upper - lower)

    # --------------------------------------------------------------------- #
    # 2. Surrogate model (Gaussian Process with Matern kernel)
    # --------------------------------------------------------------------- #
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Fit a GP to data (X, y) and return the trained model."""
        # Kernel: constant * Matern(5/2) + small white noise
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
                 Matern(length_scale=np.ones(self.dim), length_scale_bounds=(1e-2, 1e2), nu=2.5) + \
                 WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0,
                                      normalize_y=True,
                                      n_restarts_optimizer=5,
                                      random_state=0)
        gp.fit(X, y.ravel())
        return gp

    # --------------------------------------------------------------------- #
    # 3. Acquisition: Expected Improvement (EI)
    # --------------------------------------------------------------------- #
    def _acquisition_function(self, X: np.ndarray, gp: GaussianProcessRegressor,
                              y_best: float) -> np.ndarray:
        """Compute EI for each candidate in X."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        mu = mu.reshape(-1, 1)

        # avoid division by zero
        with np.errstate(divide='ignore'):
            imp = y_best - mu
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    # --------------------------------------------------------------------- #
    # 4. Next point selection (batch Kriging‑Believer + local refinement)
    # --------------------------------------------------------------------- #
    def _select_next_points(self, batch_size: int, gp: GaussianProcessRegressor,
                            y_best: float) -> np.ndarray:
        """
        Return `batch_size` new points.
        Uses Sobol candidates + L‑BFGS‑B refinement, and the Kriging‑Believer
        trick to fill the batch.
        """
        # 1) generate a moderate candidate set (e.g., 5 * dim * 10)
        n_candidates = 5 * self.dim * 10
        candidates = self._sample_points(n_candidates)

        # 2) evaluate EI on the candidates
        ei = self._acquisition_function(candidates, gp, y_best).ravel()
        # keep the top `batch_size` candidates as starting points
        start_idx = np.argpartition(-ei, batch_size)[:batch_size]
        start_points = candidates[start_idx]

        # 3) local refinement for each start point
        refined = []
        bounds = [(low, high) for low, high in zip(self.bounds[0], self.bounds[1])]

        for x0 in start_points:
            res = minimize(
                fun=lambda x: -self._acquisition_function(x.reshape(1, -1), gp, y_best),
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-6}
            )
            refined.append(res.x if res.success else x0)

        # 4) Kriging‑Believer: sequentially add points, pretending each new point
        #    has the GP mean as its observed value.
        batch = []
        X_tmp = self.X.copy()
        y_tmp = self.y.copy()
        gp_tmp = gp

        for x in refined:
            batch.append(x)
            # pretend observation = GP mean at x
            mu_x = gp_tmp.predict(x.reshape(1, -1))
            X_tmp = np.vstack([X_tmp, x])
            y_tmp = np.vstack([y_tmp, mu_x])
            # re‑fit surrogate on the "augmented" data (cheap because dim is small)
            gp_tmp = self._fit_model(X_tmp, y_tmp)

        return np.array(batch)

    # --------------------------------------------------------------------- #
    # 5. Evaluation helper
    # --------------------------------------------------------------------- #
    def _evaluate_points(self, func: Callable[[np.ndarray], np.float64],
                         X: np.ndarray) -> np.ndarray:
        """Evaluate `func` on each row of X respecting the budget."""
        n = X.shape[0]
        if self.n_evals + n > self.budget:
            # truncate to the remaining budget
            X = X[: self.budget - self.n_evals]
            n = X.shape[0]

        y = np.empty((n, 1), dtype=np.float64)
        for i, xi in enumerate(X):
            y[i, 0] = func(xi)
        self.n_evals += n
        return y

    # --------------------------------------------------------------------- #
    # 6. Update internal data structures
    # --------------------------------------------------------------------- #
    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        """Append newly evaluated points to the history."""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    # --------------------------------------------------------------------- #
    # 7. Main optimisation loop
    # --------------------------------------------------------------------- #
    def __call__(self,
                 func: Callable[[np.ndarray], np.float64]
                 ) -> tuple[np.float64, np.ndarray]:
        """
        Run Bayesian optimisation until the evaluation budget is exhausted.
        Returns the best observed value and its location.
        """
        # ---- Initial design -------------------------------------------------
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # best-so-far
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()

        # ---- Main BO loop --------------------------------------------------
        while self.n_evals < self.budget:
            # Fit surrogate on all data gathered so far
            gp = self._fit_model(self.X, self.y)

            # Decide batch size (1 for simplicity, can be larger if budget allows)
            batch_size = 1
            # Ensure we do not request more points than remaining budget
            remaining = self.budget - self.n_evals
            batch_size = min(batch_size, remaining)

            # Select next candidate(s)
            next_X = self._select_next_points(batch_size, gp, best_y)

            # Evaluate them
            next_y = self._evaluate_points(func, next_X)

            # Update history
            self._update_eval_points(next_X, next_y)

            # Update best
            cur_best_idx = np.argmin(next_y)
            if next_y[cur_best_idx, 0] < best_y:
                best_y = float(next_y[cur_best_idx, 0])
                best_x = next_X[cur_best_idx].copy()

        return best_y, best_x

```
An error occurred : Traceback (most recent call last):
  File "<SobolMaternEI_BO>", line 167, in __call__
 165 |         # best-so-far
 166 |         best_idx = np.argmin(self.y)
 167->         best_y = float(self.y[best_idx])
 168 |         best_x = self.X[best_idx].copy()
 169 | 
TypeError: only 0-dimensional arrays can be converted to Python scalars


Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

