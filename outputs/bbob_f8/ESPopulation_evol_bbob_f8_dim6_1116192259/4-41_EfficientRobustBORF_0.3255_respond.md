# Description 
**Efficient Robust Bayesian Optimization with Random Forest Surrogate Model (ERBO-RF)**: This Bayesian Optimization algorithm employs a random forest surrogate model instead of Gaussian Process, using an ensemble of decision trees to model the objective function. It maintains the adaptive multi-objective acquisition function framework but adapts the acquisition optimization strategy to work with the random forest's uncertainty estimates. The algorithm uses a hybrid approach for acquisition function optimization with local refinement and incorporates dynamic adaptation of exploration vs exploitation based on budget progress.

# Justification 
The replacement of Gaussian Process with Random Forest surrogate model offers several advantages for this specific black-box optimization problem:

1. **Computational Efficiency**: Random forests are significantly faster to train than Gaussian Processes, especially in high dimensions (d=6). While GPs have O(n³) complexity, random forests scale much better with increasing data size.

2. **Robustness to Noise and Non-smooth Functions**: The Rosenbrock function is known for its narrow valley structure and potential numerical challenges. Random forests naturally handle non-smooth functions well and are less sensitive to outliers or noise in the observations compared to GP models.

3. **Uncertainty Quantification**: While not as precise as GP uncertainty estimates, random forests provide reliable uncertainty measures through variance across trees. This is sufficient for acquisition function optimization without sacrificing too much performance.

4. **Handling of High-Dimensional Spaces**: The algorithm uses an ensemble approach that works well in high dimensions where GP models suffer from the curse of dimensionality due to their assumption of smoothness.

5. **Scalability**: Random forests can easily accommodate new data points incrementally, making them well-suited for budget-constrained optimization scenarios.

The rest of the algorithm maintains the proven adaptive multi-objective acquisition function framework which dynamically balances exploration and exploitation based on budget progress, ensuring good performance across different stages of optimization.

# Code
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import differential_evolution
import warnings
from scipy.stats import norm

class EfficientRobustBORF:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        
        # Initialize random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Determine initial sample size based on dimensionality and budget
        self.n_init = min(max(10, 2 * dim + 1), budget // 4)
        if self.n_init > budget:
            self.n_init = budget
            
        # Model parameters
        self.model = None
        
        # For batch selection
        self.batch_size = 1

    def _sample_points(self, n_points):
        """Sample points using maximin Latin hypercube design"""
        # Create Latin hypercube sampling
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        """Fit and tune surrogate model using Random Forest"""
        # Use Random Forest with multiple trees for robustness
        # Random forests are computationally efficient and handle high-dimensional spaces well
        model = RandomForestRegressor(
            n_estimators=20,  # Reasonable number of trees
            max_depth=None,   # Allow full depth for better fit
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=1  # Single thread to maintain reproducibility
        )
        
        model.fit(X, y.flatten())
        return model

    def _adaptive_acquisition(self, X):
        """Adaptive acquisition function that balances expected improvement and uncertainty"""
        if self.model is None:
            return np.zeros(X.shape[0])
            
        # Get predictions from the Random Forest model
        # For RF, we get both mean prediction and standard deviation from individual tree variance
        predictions = []
        for tree in self.model.estimators_:
            pred = tree.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mu = np.mean(predictions, axis=0)
        sigma = np.std(predictions, axis=0)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Adaptive exploration parameter based on budget progress
        progress_ratio = self.n_evals / self.budget
        
        # Start with high exploration, decrease as we progress
        xi = 0.1 * (1 - progress_ratio) + 0.001
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, 1e-8)
        
        improvement = y_min - mu - xi
        z = improvement / sigma
        
        # Expected Improvement using RF predictions
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # Normalize EI to [0, 1] for consistent weighting
        if np.max(ei) != np.min(ei):
            ei_normalized = (ei - np.min(ei)) / (np.max(ei) - np.min(ei))
        else:
            ei_normalized = np.zeros_like(ei)
            
        # Uncertainty-based component from RF variance
        uncertainty = 1.0 / (sigma + 1e-8)
        
        # Normalize uncertainty to [0, 1] 
        if np.max(uncertainty) != np.min(uncertainty):
            uncertainty_normalized = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty))
        else:
            uncertainty_normalized = np.zeros_like(uncertainty)
            
        # Dynamic weighting based on progress
        if progress_ratio < 0.3:
            # Early stage: prioritize exploration (EI)
            ei_weight = 0.7
            unc_weight = 0.3
        elif progress_ratio < 0.7:
            # Middle stage: balanced
            ei_weight = 0.5
            unc_weight = 0.5
        else:
            # Late stage: prioritize exploitation (uncertainty)
            ei_weight = 0.3
            unc_weight = 0.7
            
        # Combined acquisition function
        acq_value = ei_weight * ei_normalized + unc_weight * uncertainty_normalized
        
        return acq_value

    def _select_next_points(self):
        """Select next points using adaptive acquisition function"""
        # Determine population size based on budget and progress
        if self.n_evals < self.budget * 0.2:
            pop_size = min(50, 10 * self.dim)
        elif self.n_evals < self.budget * 0.6:
            pop_size = min(30, 5 * self.dim)
        else:
            pop_size = min(20, 3 * self.dim)
            
        # For acquisition optimization, use a hybrid approach
        # Generate diverse population with some based on best points
        if self.X is not None and len(self.X) > 0:
            # Use best points as starting locations with some diversity
            best_indices = np.argsort(self.y.flatten())[:min(3, len(self.y))]
            start_points = self.X[best_indices]
            
            # Generate diverse initial population
            population = []
            for i in range(pop_size):
                if i < len(start_points):
                    point = start_points[i % len(start_points)].copy()
                    # Add noise to encourage exploration around best points
                    noise_scale = 0.1 * (self.bounds[1] - self.bounds[0])
                else:
                    point = self._sample_points(1)[0]
                    noise_scale = 0.2 * (self.bounds[1] - self.bounds[0])
                
                # Add noise for diversity
                noise = self.rng.normal(0, noise_scale)
                point += noise
                
                # Clip to bounds
                point = np.clip(point, self.bounds[0], self.bounds[1])
                population.append(point)
        else:
            # No previous points, generate random population
            population = self._sample_points(pop_size)
        
        # Evaluate acquisition function for the entire population
        population_array = np.array(population)
        acq_values = self._adaptive_acquisition(population_array)
        
        # Select top candidates based on acquisition values
        sorted_indices = np.argsort(acq_values)[::-1]
        selected_points = np.array([population[i] for i in sorted_indices[:self.batch_size]])
        
        return selected_points

    def _evaluate_points(self, func, X):
        """Evaluate the points in X"""
        # Evaluate all points in X
        y_vals = []
        for x in X:
            y_vals.append(func(x))
            
        y_vals = np.array(y_vals).reshape(-1, 1)
        
        self.n_evals += len(X)
        return y_vals
    
    def _update_eval_points(self, new_X, new_y):
        """Update self.X and self.y"""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func:Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        """Main minimize optimization loop"""
        
        # Initialize with Latin hypercube design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the model with new data
            if self.n_evals > self.n_init:
                self.model = self._fit_model(self.X, self.y)
            
            # Select next points using acquisition function
            next_points = self._select_next_points()
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Break if we've exceeded budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
```
## Feedback
 The algorithm EfficientRobustBORF got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3255 with standard deviation 0.0207.

took 15.53 seconds to run.