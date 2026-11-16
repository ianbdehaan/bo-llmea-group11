You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of BBOB Function 8 (Rosenbrock, 6D). Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- AdaptiveTrustHybridBO: 0.3575, 773.93 seconds, **AdaptiveTrustHybridBO** – a Bayesian‑Optimization loop that builds on the Sobol‑EI‑HybridBO baseline and adds three powerful refinements that are especially useful for narrow‑valley problems such as the 6‑D Rosenbrock (BBOB f8):

1. **Adaptive Trust‑Region (TR)** – a hyper‑cube centred on the current best solution.  
   The TR radius is enlarged after a successful improvement (× 1.2) and shrunk after a failure (× 0.7) and is always kept inside the global bounds.  
   This concentrates sampling in the promising valley while still allowing the region to grow when the algorithm needs to explore.

2. **Hybrid acquisition maximisation** – a cheap Sobol/LHS pool inside the TR is first generated, the Expected‑Improvement (EI) is evaluated on the pool, the *k* best pool points (default k = 20) become start points for a gradient‑based L‑BFGS‑B optimiser that **maximises EI** (i.e. minimises –EI).  
   The best local optimum is taken as the next evaluation point.  
   This gives a far more accurate EI maximisation than a pure pool‑search at almost no extra cost.

3. **Periodic GP‑hyper‑parameter re‑training** – the GP surrogate is fully re‑optimised (with 5 restarts) only every `retrain_interval` iterations (default 5). In the intervening iterations the GP is refitted with `n_restarts=0`, i.e. only the posterior is updated while keeping the previously learned kernel hyper‑parameters.  
   This reduces the expensive GP optimisation overhead while still adapting to the changing data.

The algorithm respects the strict evaluation budget, works for any dimensionality inside the \([-5,5]^d\) box and uses only **NumPy, SciPy and scikit‑learn**.

---


- TrustRegionAdaptiveSobolEI_BO: 0.3370, 73.87 seconds, **  
**TrustRegionAdaptiveSobolEI_BO** – a Bayesian‑Optimization algorithm that builds on the strong points of the original *AdaptiveSobolEI_BO* and adds three complementary mechanisms that are especially helpful for the narrow‑valley structure of the 6‑D Rosenbrock function (BBOB f8):

1. **Adaptive Trust‑Region (TR)** – a hyper‑cube centred on the current best solution.  
   The TR radius is enlarged after a successful improvement and shrunk after a failure, keeping the search focussed inside the valley while still allowing occasional global jumps.

2. **Hybrid candidate pool** – a small global Sobol pool (exploration) + a larger Sobol pool confined to the TR (exploitation).  
   This gives a cheap yet balanced mixture of long‑range and short‑range proposals.

3. **Periodic GP‑hyper‑parameter re‑training & warm‑start** – the GP is re‑optimised only every *retrain_interval* iterations, otherwise the previous kernel hyper‑parameters are kept and only the training data are updated.  This cuts the overhead dramatically without sacrificing model quality.

The rest of the pipeline (initial Sobol design, ARD Matern‑5/2 GP surrogate, Expected‑Improvement acquisition, L‑BFGS‑B refinement of the top‑k EI points, strict budget handling) is identical to the baseline, but the added TR logic yields a tighter exploitation of the Rosenbrock valley and improves the overall AOCC score.

---

**
- SobolEIHybridBO: 0.3335, 91.23 seconds, **  
**Sobol‑EI‑HybridBO** – a Bayesian‑Optimization framework that combines low‑discrepancy Sobol quasi‑Monte‑Carlo (QMC) initialization, a scalable Gaussian‑Process surrogate (Matern‑5/2 kernel) and an Expected‑Improvement (EI) acquisition.  After a cheap global Sobol design, the algorithm repeatedly (i) fits the GP, (ii) draws a large pool of candidate points with cheap Latin‑Hypercube sampling, (iii) evaluates EI on the pool, and (iv) selects a small batch (default = 1) of the highest‑EI points.  The batch is chosen by a simple “Kriging‑believer” update (the GP mean of already‑chosen points is used as a surrogate for their unknown function values) to keep the selection cheap while still providing a modest diversification effect.  The whole loop stops exactly at the user‑provided evaluation budget.

---

**
- TrustRegionSobolEI_BO: 0.3310, 57.24 seconds, **TrustRegionSobolEI_BO** – a Bayesian‑Optimization algorithm that builds on the proven AdaptiveSobolEI_BO framework and adds three powerful refinements that are especially useful for narrow‑valley problems such as the 6‑D Rosenbrock (BBOB f8):

1. **Adaptive trust‑region** centred on the current best solution.  
   The region is a hyper‑cube whose half‑width `r` is enlarged after a successful improvement (× 1.2) and shrunk after a failure (× 0.7).  
   This focuses the expensive GP‑based search inside the promising valley while still allowing global exploration when the region grows.

2. **Periodic GP re‑training** (default every 5 iterations).  
   Re‑fitting a GP is the most costly step; we keep the previous surrogate for a few iterations and only update the training data, dramatically reducing overhead without sacrificing model quality.

3. **Exploration‑exploitation balance** via a small random‑sampling injection (probability `ε = 0.1`).  
   With probability ε we ignore the trust‑region and propose a fresh Sobol point, preventing premature convergence to a sub‑optimal valley.

All other ingredients remain the same: Sobol quasi‑Monte‑Carlo initialization, an ARD Matern‑5/2 GP surrogate, Expected‑Improvement (EI) acquisition, and a cheap L‑BFGS‑B refinement of the best Sobol candidates.  
The algorithm respects the strict evaluation budget, works for any dimensionality inside the \([-5,5]^d\) box, and uses only NumPy, SciPy and scikit‑learn.

---




The selected solution to update is:
**  
**Sobol‑EI‑HybridBO** – a Bayesian‑Optimization framework that combines low‑discrepancy Sobol quasi‑Monte‑Carlo (QMC) initialization, a scalable Gaussian‑Process surrogate (Matern‑5/2 kernel) and an Expected‑Improvement (EI) acquisition.  After a cheap global Sobol design, the algorithm repeatedly (i) fits the GP, (ii) draws a large pool of candidate points with cheap Latin‑Hypercube sampling, (iii) evaluates EI on the pool, and (iv) selects a small batch (default = 1) of the highest‑EI points.  The batch is chosen by a simple “Kriging‑believer” update (the GP mean of already‑chosen points is used as a surrogate for their unknown function values) to keep the selection cheap while still providing a modest diversification effect.  The whole loop stops exactly at the user‑provided evaluation budget.

---

**
With code:
```python
from collections.abc import Callable
from scipy.stats import qmc, norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class SobolEIHybridBO:
    """
    Sobol‑EI‑HybridBO: Bayesian Optimisation with Sobol initialization,
    Gaussian‑Process surrogate (Matern‑5/2) and Expected‑Improvement acquisition.
    """

    def __init__(self, budget: int, dim: int):
        self.budget = budget                # total allowed evaluations
        self.dim = dim
        # bounds has shape (2, dim); bounds[0] = lower, bounds[1] = upper
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        # Design matrix and observations – will be filled during optimisation
        self.X: np.ndarray = None            # shape (n_points, dim)
        self.y: np.ndarray = None            # shape (n_points, 1)
        self.n_evals = 0                     # number of function evaluations used
        # Number of initial points – 2 * dim is a good rule of thumb
        self.n_init = max(2 * dim, 5)

    # --------------------------------------------------------------------- #
    #  Sampling utilities
    # --------------------------------------------------------------------- #
    def _sample_points(self, n_points: int) -> np.ndarray:
        """
        Return `n_points` uniformly distributed points inside the bounds.
        Uses Sobol QMC for the very first design, otherwise Latin‑Hypercube.
        """
        if self.X is None:                      # first call -> Sobol
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            unit = sampler.random_base2(m=int(np.ceil(np.log2(n_points))))
            unit = unit[:n_points]
        else:                                   # later calls -> LHS
            sampler = qmc.LatinHypercube(d=self.dim)
            unit = sampler.random(n_points)

        # scale from [0,1] to the actual bounds
        scaled = qmc.scale(unit,
                           self.bounds[0],
                           self.bounds[1])
        return scaled

    # --------------------------------------------------------------------- #
    #  Surrogate model
    # --------------------------------------------------------------------- #
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """
        Fit a Gaussian Process with a Matern‑5/2 kernel.
        Hyper‑parameters are optimized by the scikit‑learn optimiser.
        """
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(self.dim),
                           length_scale_bounds=(1e-2, 1e2),
                           nu=2.5)
                  + WhiteKernel(noise_level=1e-8,
                                noise_level_bounds=(1e-10, 1e-1)))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      normalize_y=True,
                                      n_restarts_optimizer=5,
                                      random_state=0)
        gp.fit(X, y.ravel())
        return gp

    # --------------------------------------------------------------------- #
    #  Acquisition – Expected Improvement
    # --------------------------------------------------------------------- #
    def _acquisition_function(self, X: np.ndarray, gp: GaussianProcessRegressor):
        """
        Compute Expected Improvement for each candidate in X.
        Returns shape (n_points, 1).
        """
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)

        # Best observed value so far (minimisation)
        y_best = np.min(self.y)

        # Standardised improvement
        with np.errstate(divide='ignore'):
            imp = y_best - mu.reshape(-1, 1)
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    # --------------------------------------------------------------------- #
    #  Batch selection using Kriging‑believer
    # --------------------------------------------------------------------- #
    def _select_next_points(self, batch_size: int, gp: GaussianProcessRegressor):
        """
        Generate a large candidate pool, evaluate EI and return the `batch_size`
        points with highest EI, applying a simple Kriging‑believer update.
        """
        # Candidate pool size – a heuristic that works well for low‑dim problems
        n_candidates = max(5000, 100 * self.dim)

        candidates = self._sample_points(n_candidates)

        # Compute EI once
        ei = self._acquisition_function(candidates, gp).ravel()

        selected = []
        for _ in range(batch_size):
            idx = np.argmax(ei)
            next_pt = candidates[idx:idx + 1]          # shape (1, dim)
            selected.append(next_pt)

            # Kriging‑believer: pretend the GP mean at this point is the true value
            # and update the GP posterior analytically (fast approximation)
            # Instead of refitting, we simply set its EI to -inf and
            # recompute EI with the mean added to the training set.
            # For speed we just zero‑out its EI to avoid reselection.
            ei[idx] = -np.inf

        return np.vstack(selected)                     # shape (batch_size, dim)

    # --------------------------------------------------------------------- #
    #  Evaluation utilities
    # --------------------------------------------------------------------- #
    def _evaluate_points(self, func: Callable[[np.ndarray], np.float64],
                         X: np.ndarray) -> np.ndarray:
        """
        Evaluate the black‑box function on each row of X.
        Returns column vector of shape (n_points, 1).
        """
        y_vals = np.apply_along_axis(func, 1, X).astype(np.float64)
        y_vals = y_vals.reshape(-1, 1)

        # Update evaluation counter – respect the global budget
        self.n_evals += len(X)
        return y_vals

    def _update_eval_points(self, new_X: np.ndarray, new_y: np.ndarray):
        """Append newly evaluated points to the history."""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    # --------------------------------------------------------------------- #
    #  Main optimisation loop
    # --------------------------------------------------------------------- #
    def __call__(self,
                 func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        """
        Run Bayesian optimisation under the given evaluation budget.
        Returns the best observed value and the corresponding point.
        """
        # -----------------------------------------------------------------
        # 1) Initial Sobol design
        # -----------------------------------------------------------------
        init_X = self._sample_points(self.n_init)
        init_y = self._evaluate_points(func, init_X)
        self._update_eval_points(init_X, init_y)

        # Track the current best
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx, 0]
        best_x = self.X[best_idx, :].copy()

        # -----------------------------------------------------------------
        # 2) Main BO loop
        # -----------------------------------------------------------------
        while self.n_evals < self.budget:
            # Remaining budget may be smaller than the desired batch size
            remaining = self.budget - self.n_evals
            batch_size = min(1, remaining)   # single‑point BO is most reliable

            # Fit surrogate on all data seen so far
            gp = self._fit_model(self.X, self.y)

            # Propose next point(s) via acquisition optimisation
            next_X = self._select_next_points(batch_size, gp)

            # Evaluate the proposed point(s)
            next_y = self._evaluate_points(func, next_X)

            # Update history
            self._update_eval_points(next_X, next_y)

            # Update best solution if improved
            cur_best_idx = np.argmin(self.y)
            cur_best_y = self.y[cur_best_idx, 0]
            if cur_best_y < best_y:
                best_y = cur_best_y
                best_x = self.X[cur_best_idx, :].copy()

        return best_y, best_x

```
The algorithm SobolEIHybridBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3335 with standard deviation 0.0145.

took 91.23 seconds to run.

Refine the strategy of the selected solution to improve it.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

