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


- EnsembleThompsonBO: 0.3121, 880.22 seconds, EnsembleThompsonBO is a Bayesian Optimization algorithm that replaces the Gaussian Process surrogate model in LatinHypercubeThompsonBO with a Random Forest ensemble. It retains Thompson Sampling for acquisition function optimization but leverages the ensemble's diversity to estimate uncertainty. Key innovations include:  
- **Random Forest Ensemble**: Uses predictions from multiple trees to approximate the posterior distribution, avoiding hyperparameter optimization.  
- **Uncertainty Estimation**: Calculates the standard deviation of tree predictions as the acquisition function's sigma.  
- **Efficient Initialization**: Maintains Latin Hypercube design for diverse initial sampling.  
- **Budget-Aware Optimization**: Balances exploration and exploitation through Thompson Sampling and local optimization with L-BFGS-B.  


- LatinHypercubeThompsonBO: 0.3121, 146.54 seconds, LatinHypercubeThompsonBO is a Bayesian Optimization algorithm that improves upon the ThompsonSamplingBO by replacing the Sobol-based initial design with a Latin Hypercube design. It retains the Gaussian Process (GP) with anisotropic Matérn kernel as the surrogate model and uses Thompson Sampling for acquisition function optimization. The algorithm balances exploration and exploitation by sampling from the GP posterior and optimizing the sampled functions. The Latin Hypercube initialization ensures diverse coverage across each dimension, improving the initial model's accuracy for the Rosenbrock function's elongated valley structure.  


- EnsembleEIBO: 0.3044, 44.76 seconds, EnsembleEIBO is a Bayesian Optimization algorithm that leverages an ensemble of Random Forests as the surrogate model and Expected Improvement (EI) acquisition function. The algorithm dynamically shifts from global to local exploration by using a hybrid acquisition optimization strategy: (1) generating diverse candidates via Sobol sampling, (2) refining these candidates with L-BFGS-B, and (3) prioritizing regions near the best solution found so far. Key innovations include:  
- **Random Forest Ensemble**: Uncertainty estimation via prediction variance across trees, avoiding GP hyperparameter optimization.  
- **Dynamic Candidate Refinement**: Combines global Sobol candidates with local optimization and prioritizes regions near the best solution.  
- **Budget-Aware Initialization**: Reduces initial design points (2d instead of 3d) to save evaluations for later iterations.  




The selected solution to update is:
EnsembleThompsonBO is a Bayesian Optimization algorithm that replaces the Gaussian Process surrogate model in LatinHypercubeThompsonBO with a Random Forest ensemble. It retains Thompson Sampling for acquisition function optimization but leverages the ensemble's diversity to estimate uncertainty. Key innovations include:  
- **Random Forest Ensemble**: Uses predictions from multiple trees to approximate the posterior distribution, avoiding hyperparameter optimization.  
- **Uncertainty Estimation**: Calculates the standard deviation of tree predictions as the acquisition function's sigma.  
- **Efficient Initialization**: Maintains Latin Hypercube design for diverse initial sampling.  
- **Budget-Aware Optimization**: Balances exploration and exploitation through Thompson Sampling and local optimization with L-BFGS-B.  


With code:
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class EnsembleThompsonBO:
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
        model = RandomForestRegressor(
            n_estimators=20,
            max_depth=10,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Collect predictions from all trees in the forest
        predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        mu = np.mean(predictions, axis=0)
        sigma = np.std(predictions, axis=0)
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
The algorithm EnsembleThompsonBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3121 with standard deviation 0.0252.

took 880.22 seconds to run.

Replace the exploration strategy of the selected solution by a substantially different alternative (options: a trust region around the best point that grows and shrinks (as in TuRBO), restarts when progress stops, a switch from global to local search, batches of diverse points), and adapt the rest of the algorithm where needed. Explain in the justification why the new exploration strategy should work better.



Give the response in the format:
# Description 
<description>
# Justification 
<justification for the key components of the algorithm or the changes made>
# Code 
<code>

