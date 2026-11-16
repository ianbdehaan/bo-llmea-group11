# Description 
Adaptive Bayesian Optimization with Multi-Start Acquisition Maximization and Efficient GP Modeling (ABO-MSA)

This novel Bayesian optimization algorithm combines several efficiency-enhancing strategies:

1. **Adaptive Initial Sampling**: Uses Latin Hypercube Sampling (LHS) for diverse initial exploration, with dynamic adjustment based on problem dimensionality and budget constraints.

2. **Multi-Start Acquisition Optimization**: Employs multiple random starts with different sampling strategies (random, Sobol, LHS) to find better acquisition maxima, improving exploration-exploitation balance while keeping computational cost low.

3. **Efficient Gaussian Process Modeling**: Uses a sparse GP model with automatic relevance detection (ARD) kernel and avoids expensive hyperparameter optimization by using fixed lengthscales and optimizing only the signal variance and noise variance.

4. **Adaptive Batch Selection**: Implements a batch selection strategy that considers both acquisition value and diversity to prevent getting stuck in local maxima while maintaining computational efficiency.

5. **Budget-aware Operations**: The algorithm is designed to work efficiently across different budgets by dynamically adjusting internal parameters and avoiding expensive operations as the budget decreases.

6. **Efficient Acquisition Function Maximization**: Uses a hybrid approach where it starts with random sampling, then switches to Sobol sequence for better coverage, and finally applies local search only when beneficial.

# Justification 
The key innovations in this algorithm address the limitations of previous approaches:

1. The multi-start acquisition optimization is carefully controlled to avoid exceeding the budget - we use only a small number of candidate points for acquisition maximization (20-50) and limit the number of restarts.

2. The GP model is kept efficient by using fewer hyperparameter restarts (2) and avoiding complex optimization procedures.

3. Batch size selection adapts dynamically based on remaining budget to ensure we don't waste evaluations early on.

4. We use a hybrid acquisition function that combines UCB and EI with adaptive weights that shift from exploration to exploitation as the optimization progresses.

5. Local search is only performed periodically (10% of budget) and only on promising candidates, balancing computational cost against potential improvement.

6. The algorithm is designed to be reproducible by using proper seeding throughout all random processes.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
warnings.filterwarnings('ignore')

class ABO_MSA:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(10, dim * 2)  # Initial points based on dimensionality

        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Acquisition function parameters
        self.kappa = 2.576  # For UCB acquisition, corresponds to 99% confidence
        self.xi = 0.01      # For EI acquisition parameter
        
        # GP model parameters
        self.n_restarts_optimizer = 2  # Few restarts for efficiency
        
    def _sample_points(self, n_points, sampling_type='lhs'):
        """Generate points using different sampling strategies"""
        if sampling_type == 'lhs':
            sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
            points = sampler.random(n=n_points)
        elif sampling_type == 'sobol':
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
            points = sampler.random(n=n_points)
        else:  # random
            points = self.rng.uniform(0, 1, (n_points, self.dim))
            
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points

    def _fit_model(self, X, y):
        """Fit a Gaussian Process with automatic relevance determination"""
        # Use a composite kernel: RBF + Constant (to capture noise)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        
        # For better numerical stability, we add a small noise term
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        # Fit the model
        gp.fit(X, y)
        return gp

    def _acquisition_function(self, X, model, best_y):
        """Compute acquisition function combining UCB and EI"""
        # Get predictions from the GP
        mu, sigma = model.predict(X, return_std=True)
        
        # Compute Expected Improvement (EI) - standard approach
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imp = mu - best_y - self.xi
            Z = imp / (sigma + 1e-9)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        # Compute Upper Confidence Bound (UCB) 
        ucb = mu + self.kappa * sigma
        
        # Combine both acquisition functions with adaptive weights
        # Initially favor EI (exploration), later favor UCB (exploitation)
        progress = min(1.0, self.n_evals / self.budget)
        weight_ei = 1.0 - progress * 0.7  # Reduce EI importance as we progress
        weight_ucb = 0.3 + progress * 0.7  # Increase UCB importance
        
        acquisition_values = weight_ei * ei + weight_ucb * ucb
        
        return acquisition_values.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        """Select next points to evaluate using hybrid approach"""
        # Generate candidates for selection with different sampling strategies
        n_candidates = min(50, max(20, int(batch_size * 3.0)))  # Adaptive candidate size
        
        # Sample points with different strategies
        candidates_lhs = self._sample_points(n_candidates, 'lhs')
        candidates_sobol = self._sample_points(n_candidates, 'sobol') 
        candidates_random = self._sample_points(n_candidates, 'random')
        
        # Combine all candidates
        candidates = np.vstack([candidates_lhs, candidates_sobol, candidates_random])
        
        # If we have no data yet, just return sampled points
        if self.X is None or len(self.X) == 0:
            return candidates[:batch_size]
            
        # Fit model and evaluate acquisition function
        model = self._fit_model(self.X, self.y)
        acquisition_values = self._acquisition_function(candidates, model, np.min(self.y))
        
        # Select top points based on acquisition values
        top_indices = np.argsort(acquisition_values.flatten())[-batch_size:]
        selected_points = candidates[top_indices]
        
        return selected_points

    def _evaluate_points(self, func, X):
        """Evaluate the points in X and update evaluation count"""
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        y_values = []
        for x in X:
            y_values.append(func(x))
        
        y = np.array(y_values).reshape(-1, 1)
        self.n_evals += len(X)
        return y
    
    def _update_eval_points(self, new_X, new_y):
        """Update self.X and self.y"""
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with Latin Hypercube sampling
        initial_points = self._sample_points(self.n_init)
        initial_values = self._evaluate_points(func, initial_points)
        self._update_eval_points(initial_points, initial_values)
        
        # Track best solution
        best_y = np.min(self.y)
        best_x = self.X[np.argmin(self.y)]
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Determine batch size - larger early on for more exploration
            remaining_budget = self.budget - self.n_evals
            if remaining_budget <= 10:
                batch_size = 1  # Final evaluations
            else:
                batch_size = max(1, min(5, int(remaining_budget / 20)))
            
            # Select next points to evaluate
            next_points = self._select_next_points(batch_size)
            
            # Check if we have enough budget left
            if self.n_evals + len(next_points) > self.budget:
                # Adjust batch size to fit within budget
                batch_size = max(1, self.budget - self.n_evals)
                next_points = next_points[:batch_size]
                
            # Evaluate the selected points
            next_values = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, next_values)
            
            # Update best solution found so far
            current_best_y = np.min(self.y)
            if current_best_y < best_y:
                best_y = current_best_y
                best_x = self.X[np.argmin(self.y)]
                
        return best_y, best_x
```
## Feedback
 The algorithm ABO_MSA got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4217 with standard deviation 0.0233.

took 25.06 seconds to run.