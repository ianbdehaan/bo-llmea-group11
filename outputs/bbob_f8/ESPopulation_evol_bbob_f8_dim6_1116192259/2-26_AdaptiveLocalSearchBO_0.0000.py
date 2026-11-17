from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

class AdaptiveLocalSearchBO:
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
            
        # Acquisition function parameters
        self.noise_level = 1e-6
        
        # Model parameters
        self.model = None
        self.kernel_type = 'rbf'  # Start with RBF kernel
        
        # For adaptive local search
        self.local_search_active = False
        self.local_search_center = None
        self.local_search_radius = 1.0
        
        # Batch size
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
        """Fit and tune surrogate model with adaptive kernel selection"""
        # Determine which kernel to use based on function behavior
        if len(y) > 10:
            # Analyze the function landscape for smoothness
            distances = cdist(X, X, 'euclidean')
            np.fill_diagonal(distances, 1.0)
            
            # Calculate local variance in y values
            local_variances = []
            for i in range(len(X)):
                # Get neighbors within a certain distance
                neighbor_indices = np.where(distances[i] < np.median(distances[i]))[0]
                if len(neighbor_indices) > 2:
                    local_y = y[neighbor_indices].flatten()
                    local_variances.append(np.var(local_y))
            
            if len(local_variances) > 0:
                avg_local_variance = np.mean(local_variances)
                # If variance is small, prefer Matern (smoother) kernel
                # If variance is large, prefer RBF (more flexible)
                if avg_local_variance < 1e-2:
                    self.kernel_type = 'matern'
                else:
                    self.kernel_type = 'rbf'
        
        # Choose kernel based on analysis
        if self.kernel_type == 'matern':
            kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        else:
            kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        
        # Add white noise for numerical stability
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        # Fit GP with reasonable number of restarts to balance efficiency and accuracy
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=2,  # Reduced for computational efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _curvature_aware_acquisition(self, X):
        """Acquisition function that incorporates local curvature information"""
        # If no model exists, return zeros
        if self.model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from the GP model
        mu, sigma = self.model.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Calculate curvature information by computing second derivatives
        # This is a simplified approximation using finite differences
        curvature_info = np.zeros(X.shape[0])
        if len(self.X) > 10 and self.n_evals > self.n_init + 5:
            # Compute approximate Hessian using local points
            distances = cdist(X, self.X, 'euclidean')
            # Get the nearest points for each query point
            nearest_indices = np.argmin(distances, axis=1)
            if len(nearest_indices) > 0:
                # Simple curvature estimate based on distance to nearest points
                min_distances = np.min(distances, axis=1)
                # Curvature inversely related to distance (closer = higher curvature)
                curvature_info = 1.0 / (min_distances + 1e-8)
        
        # Standard Expected Improvement calculation
        improvement = y_min - mu
        # Add curvature information to acquisition function
        # High curvature regions are more promising for local search
        acquisition = improvement * (1 + 0.5 * curvature_info.reshape(-1, 1))
        
        return acquisition.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        """Select the next points to evaluate using hybrid optimization strategy"""
        # Use a hybrid approach for acquisition function optimization
        # First, generate a population of candidate points
        
        # Determine search strategy based on evaluation progress
        progress_ratio = self.n_evals / self.budget
        
        if progress_ratio < 0.3 or len(self.X) < 10:
            # Early stage - global exploration
            # Use stochastic sampling with good diversity
            n_candidates = min(50, max(20, 10 * self.dim))
            candidates = self._sample_points(n_candidates)
            
            # Evaluate acquisition function for candidates
            acq_values = self._curvature_aware_acquisition(candidates).flatten()
            
            # Select top candidates
            selected_indices = np.argsort(acq_values)[-batch_size:]
            selected_points = candidates[selected_indices]
        else:
            # Later stage - local exploitation
            # Use gradient-based optimization around promising regions
            
            if len(self.X) > 0:
                # Find best points and use them as starting points
                best_indices = np.argsort(self.y.flatten())[:min(3, len(self.y))]
                
                # Collect candidates by sampling near the best points
                candidates = []
                for idx in best_indices:
                    # Sample around the current best point
                    center = self.X[idx]
                    # Use local search with adaptive radius
                    search_radius = 0.1 * (self.bounds[1] - self.bounds[0])
                    
                    # Generate points around this center
                    local_samples = self._sample_points(10)
                    # Adjust samples to be near the center
                    for i in range(len(local_samples)):
                        if i < len(local_samples):
                            local_samples[i] = center + (local_samples[i] - center) * 0.5
                    
                    candidates.extend(local_samples)
                
                candidates = np.array(candidates[:20])  # Limit to avoid too many points
            else:
                # No previous points, random sampling
                candidates = self._sample_points(20)
            
            # Evaluate acquisition function for candidates
            acq_values = self._curvature_aware_acquisition(candidates).flatten()
            
            # Select best candidates
            selected_indices = np.argsort(acq_values)[-batch_size:]
            selected_points = candidates[selected_indices]
        
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
            next_points = self._select_next_points(self.batch_size)
            
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
