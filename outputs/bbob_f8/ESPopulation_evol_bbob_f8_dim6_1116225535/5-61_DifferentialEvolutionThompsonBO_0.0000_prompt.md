You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems


The optimization algorithm is evaluated on BBOB function 8 (Rosenbrock, original), instance 1, dimension 6, minimization. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `__call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
As an expert of numpy, scipy, scikit-learn, torch, gpytorch, you are allowed to use these libraries. Do not use any other libraries unless they cannot be replaced by the above libraries.  Do not remove the comments from the code.
Name the class based on the characteristics of the algorithm with a template '<characteristics>BO'.
The algorithm must be reproducible: the evaluator calls np.random.seed(seed) and torch.manual_seed(seed) before each run. Every other random generator must be created from np.random, e.g. `self.rng = np.random.default_rng(np.random.randint(2**31 - 1))`, and passed on explicitly, e.g. `qmc.Sobol(d=dim, scramble=True, seed=self.rng)`, `qmc.LatinHypercube(d=dim, seed=self.rng)`, `GaussianProcessRegressor(..., random_state=int(self.rng.integers(2**31 - 1)))`. Never create an unseeded generator.
Each candidate is run 10 times in parallel, one CPU core per run, and is stopped and counted as failed if the runs take longer than 120 seconds. Keep the cost per iteration small (e.g. few optimizer restarts for the GP hyperparameters and the acquisition function). The algorithm will later also be run with a budget of 600 evaluations, so it must work for any budget and its cost should grow moderately with the number of evaluated points.
A BO algorithm can be built from many different components, for example:
- initial design: Sobol, Latin hypercube, random, maximin designs, different sizes (e.g. d+1 up to 4d points)
- surrogate model: GP with an anisotropic (ARD) Matern or RBF kernel, GP on log-transformed or rank-transformed y, random forest or extra trees (uncertainty from the spread of the trees), an ensemble of models, a local GP fitted only near the best point
- acquisition function: expected improvement, log expected improvement, lower confidence bound with a schedule for the exploration weight, probability of improvement, Thompson sampling from the GP posterior, a portfolio that switches between several acquisition functions
- acquisition function optimizer: L-BFGS-B from several starting points, differential evolution, a small evolution strategy, random candidates plus local refinement, candidates sampled around the best points
- exploration strategy: a trust region around the best point that grows and shrinks (as in TuRBO), restarts when progress stops, a switch from global to local search, batches of diverse points
Do not always use the same default choices (e.g. a scikit-learn GP with expected improvement on random candidates); choose the components that suit the problem.

Give an excellent, novel and computationally efficient Bayesian Optimization algorithm to solve this task, give it a concise but comprehensive key-word-style description with the main ideas and justify your decision about the algorithm.

The current population of algorithms already evaluated(name, score, runtime and description):
- TrustRegionGPBO: 0.3340, 208.71 seconds, TrustRegionGPBO is a Bayesian Optimization algorithm that introduces a trust region strategy, a local Gaussian Process (GP) with RBF kernel, and differential evolution for acquisition function optimization. The algorithm dynamically expands the trust region around the best solution when progress is made, enabling efficient navigation of the Rosenbrock function's elongated valley. It combines global acquisition function optimization within the trust region with local model adaptation, balancing exploration and exploitation.


- TrustRegionEnsembleThompsonBO: 0.3281, 988.68 seconds, TrustRegionEnsembleThompsonBO is a Bayesian Optimization algorithm that integrates trust region management into the EnsembleThompsonBO framework. It uses a Random Forest ensemble for surrogate modeling and Thompson Sampling for acquisition optimization, but restricts exploration to a dynamically sized trust region around the best solution. The trust region expands when progress is made and shrinks when no improvement occurs, enabling efficient navigation of the Rosenbrock function's challenging landscape. Key innovations include:  
- **Trust Region Strategy**: Focuses search within a local region that expands/shrinks based on progress.  
- **Efficient Initialization**: Uses a minimal Latin Hypercube design of size `d+1` to reduce initial evaluations.  
- **Local Candidate Sampling**: Generates Sobol candidates within the trust region, ensuring exploration in promising areas.  
- **Dynamic Trust Region Adjustment**: Balances exploration (expansion) and exploitation (contraction) through geometric scaling of the trust region.  


- TrustRegionHybridBO: 0.3175, 257.99 seconds, TrustRegionHybridBO is a Bayesian Optimization algorithm that integrates a trust region strategy with a hybrid acquisition function optimizer combining random candidates and local L-BFGS-B optimization. The algorithm employs a local Gaussian Process (GP) with an anisotropic RBF kernel to model the Rosenbrock function's elongated valley. The trust region dynamically expands around the best solution when progress is made, balancing exploration and exploitation. Key innovations include:  
- **Hybrid Acquisition Function Optimization**: Generates 50 random candidates within the trust region, followed by L-BFGS-B refinement for each candidate, enabling efficient global-local search.  
- **Budget-Aware Design**: Reduces computational overhead by replacing differential evolution with faster local optimization, ensuring scalability for larger budgets.  
- **Adaptive Trust Region**: Expands or shrinks the trust region based on progress, maintaining focus on promising areas while avoiding premature convergence.  


- MaternGPThompsonBO: 0.3121, 147.17 seconds, MaternGPThompsonBO is a Bayesian Optimization algorithm that replaces the Random Forest ensemble with a Gaussian Process (GP) using an anisotropic Matern kernel and Thompson Sampling. The algorithm leverages the GP's ability to capture smoothness and dimension-specific length scales in the Rosenbrock function's elongated valley. Key components include:
- **Anisotropic Matern Kernel**: Models varying smoothness across dimensions, crucial for the Rosenbrock function's structure.
- **Efficient GP Fitting**: Limited hyperparameter restarts reduce computational cost while maintaining model accuracy.
- **Thompson Sampling**: Directly samples from the GP posterior to balance exploration and exploitation.
- **Latin Hypercube Initialization**: Ensures diverse initial sampling for robust model fitting.
- **Acquisition Function Optimization**: Combines global Sobol candidates with L-BFGS-B refinement for effective next-point selection.




The selected solution to update is:
MaternGPThompsonBO is a Bayesian Optimization algorithm that replaces the Random Forest ensemble with a Gaussian Process (GP) using an anisotropic Matern kernel and Thompson Sampling. The algorithm leverages the GP's ability to capture smoothness and dimension-specific length scales in the Rosenbrock function's elongated valley. Key components include:
- **Anisotropic Matern Kernel**: Models varying smoothness across dimensions, crucial for the Rosenbrock function's structure.
- **Efficient GP Fitting**: Limited hyperparameter restarts reduce computational cost while maintaining model accuracy.
- **Thompson Sampling**: Directly samples from the GP posterior to balance exploration and exploitation.
- **Latin Hypercube Initialization**: Ensures diverse initial sampling for robust model fitting.
- **Acquisition Function Optimization**: Combines global Sobol candidates with L-BFGS-B refinement for effective next-point selection.


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class MaternGPThompsonBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 3 * dim  # Latin Hypercube design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points):
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, self.bounds[0], self.bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,  # Reduced from default 1 to 5 for stability
            optimizer='fmin_l_bfgs_b',
            alpha=1e-6,  # Small noise for numerical stability
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)  # Avoid division by zero
        samples = mu + sigma * self.rng.normal(size=mu.shape)
        return samples.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        candidates = self._sample_points(50)
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values, axis=0)[:, 0]
        top_candidates = candidates[idxs[-5:]]  # Last 5 (highest acq values)

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1)).item()
            res = minimize(
                acq_wrapper, 
                x0, 
                bounds=[(l, u) for l, u in zip(self.bounds[0], self.bounds[1])],
                method='L-BFGS-B'
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return candidates[[idxs[0]]].reshape(1, -1)
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points)
        best_idx = np.argmax(acq_values_opt)
        return optimized_points[best_idx].reshape(1, -1)

    def _evaluate_points(self, func, X):
        n_new = len(X)
        if self.n_evals + n_new > self.budget:
            n_new = self.budget - self.n_evals
            X = X[:n_new]
        y = np.zeros((n_new, 1))
        for i in range(n_new):
            y[i, 0] = func(X[i])
        self.n_evals += n_new
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_y = np.min(self.y)
        best_x = self.X[np.argmin(self.y)]

        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)
            
            if new_y[0] < best_y:
                best_y = new_y[0]
                best_x = next_point[0]
            
            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x

```
The algorithm MaternGPThompsonBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3121 with standard deviation 0.0252.

took 147.17 seconds to run.

Replace the acquisition function optimizer of the selected solution by a substantially different alternative (options: L-BFGS-B from several starting points, differential evolution, a small evolution strategy, random candidates plus local refinement, candidates sampled around the best points), and adapt the rest of the algorithm where needed. Explain in the justification why the new acquisition function optimizer should work better.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

