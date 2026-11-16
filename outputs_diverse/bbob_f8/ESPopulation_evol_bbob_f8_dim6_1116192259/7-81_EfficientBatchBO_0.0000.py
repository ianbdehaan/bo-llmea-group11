from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import warnings

class EfficientBatchBO:
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
            
        # GP model parameters
        self.gp_model = None
        
        # Hybrid model parameters
        self.local_model = None
        self.local_radius = 0.5
        
        # Acquisition function parameters
        self.noise_level = 1e-6
        self.acq_noise = 1e-8
        self.exploitation_weight = 0.7
        self.exploration_weight = 0.3
        self.adaptation_threshold = 0.1
        
        # Batch parameters
        self.batch_size = min(5, max(1, budget // 20))
        
        # Performance tracking
        self.best_y_history = []
        self.iteration = 0
        self.progress_window = 10

    def _sample_points(self, n_points):
        """Sample points using Sobol sequence for better space-filling properties"""
        # Create Sobol sequence sampling
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_gp_model(self, X, y):
        """Fit Gaussian Process surrogate model with optimized hyperparameters"""
        # Use a GP with ARD Matern kernel for smooth functions
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        kernel += WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-10, 1e1))
        
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            n_restarts_optimizer=2,  # Reduced restarts for efficiency
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1))
        )
        
        model.fit(X, y)
        return model

    def _fit_local_model(self, X, y):
        """Fit a local polynomial regression model near the best point"""
        if len(X) < 5:
            # Not enough points for local modeling
            return None
            
        # Find the best point
        best_idx = np.argmin(y)
        best_x = X[best_idx]
        
        # Compute distances to all points
        distances = np.linalg.norm(X - best_x, axis=1)
        
        # Select points within local radius
        local_indices = np.where(distances <= self.local_radius)[0]
        
        if len(local_indices) < 3:
            # Not enough local points, use all points
            local_indices = np.arange(len(X))
            
        X_local = X[local_indices]
        y_local = y[local_indices]
        
        # Fit polynomial regression
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly_features.fit_transform(X_local)
        
        model = LinearRegression()
        model.fit(X_poly, y_local.flatten())
        
        return {
            'model': model,
            'poly_features': poly_features,
            'center': best_x,
            'radius': self.local_radius
        }

    def _acquisition_function(self, X):
        """Novel acquisition function combining Expected Improvement and Uncertainty Sampling"""
        if self.gp_model is None:
            return np.zeros((X.shape[0], 1))
            
        # Get predictions from GP
        mu, sigma = self.gp_model.predict(X, return_std=True)
        
        # Find best observed value (y_min)
        y_min = np.min(self.y)
        
        # Avoid numerical issues
        sigma = np.maximum(sigma, self.acq_noise)
        
        # Expected Improvement calculation
        improvement = y_min - mu
        ei = improvement * norm.cdf(improvement / sigma) + sigma * norm.pdf(improvement / sigma)
        
        # Uncertainty sampling (high variance points)
        uncertainty = sigma
        
        # Adaptive weights based on optimization progress
        if len(self.best_y_history) > self.progress_window:
            recent_progress = np.mean(np.diff(self.best_y_history[-self.progress_window:]))
            if recent_progress < -self.adaptation_threshold:
                # More exploration needed
                exploitation_weight = 0.4
                exploration_weight = 0.6
            else:
                # More exploitation
                exploitation_weight = 0.7
                exploration_weight = 0.3
        else:
            exploitation_weight = self.exploitation_weight
            exploration_weight = self.exploration_weight
            
        # Combined acquisition function
        acq_value = exploitation_weight * ei + exploration_weight * uncertainty
        
        return acq_value.reshape(-1, 1)

    def _optimize_acquisition(self, n_starting_points=50):
        """Optimize acquisition function using multiple starting points"""
        if self.X is None or len(self.X) == 0:
            # If no points yet, use Sobol sampling
            candidates = self._sample_points(n_starting_points)
            acq_values = self._acquisition_function(candidates)
            return candidates[np.argmax(acq_values)].reshape(1, -1)
        
        # Generate multiple starting points using Sobol sequence
        starting_points = self._sample_points(n_starting_points)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(starting_points)
        
        # Select the best point from candidates
        best_candidate_idx = np.argmax(acq_values.flatten())
        return starting_points[best_candidate_idx].reshape(1, -1)

    def _select_diverse_batch(self, n_candidates=20):
        """Select diverse batch using clustering to avoid redundant evaluations"""
        # Sample more candidates than needed for diversity
        all_candidates = self._sample_points(n_candidates)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(all_candidates)
        
        # Select the top candidates based on acquisition value
        top_indices = np.argsort(acq_values.flatten())[-self.batch_size:]
        selected_candidates = all_candidates[top_indices]
        
        # Further refine diversity using clustering
        if len(selected_candidates) > 1:
            # Use KMeans to cluster and select diverse points
            kmeans = KMeans(n_clusters=min(self.batch_size, len(selected_candidates)), 
                           random_state=int(self.rng.integers(2**31 - 1)))
            kmeans.fit(selected_candidates)
            
            # For each cluster, select the candidate closest to the cluster center
            diverse_candidates = []
            for i in range(min(self.batch_size, len(selected_candidates))):
                cluster_points = selected_candidates[kmeans.labels_ == i]
                if len(cluster_points) > 0:
                    center = kmeans.cluster_centers_[i]
                    closest_point = cluster_points[np.argmin(np.linalg.norm(cluster_points - center, axis=1))]
                    diverse_candidates.append(closest_point)
            
            if len(diverse_candidates) >= self.batch_size:
                return np.array(diverse_candidates[:self.batch_size])
        
        return selected_candidates

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
        
        # Initialize with Sobol design
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial GP model
        self.gp_model = self._fit_gp_model(self.X, self.y)
        
        # Initialize history tracking
        self.best_y_history = [np.min(self.y)]
        
        # Main optimization loop
        while self.n_evals < self.budget:
            # Update the models with new data
            if self.n_evals > self.n_init:
                self.gp_model = self._fit_gp_model(self.X, self.y)
                
                # Fit local model for fine-tuning near optimum
                self.local_model = self._fit_local_model(self.X, self.y)
            
            # Select next batch of points using diverse sampling
            next_points = self._select_diverse_batch(n_candidates=20)
            
            # Evaluate these points
            y_next = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, y_next)
            
            # Update best value history
            self.best_y_history.append(np.min(self.y))
            
            # Break if we've exceeded budget
            if self.n_evals >= self.budget:
                break
                
        # Return best result
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x

# Note: The norm import is required for the acquisition function
from scipy.stats import norm
