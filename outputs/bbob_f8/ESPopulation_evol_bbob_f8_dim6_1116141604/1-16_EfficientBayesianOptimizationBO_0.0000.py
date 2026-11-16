from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.optim import Adam
import warnings
warnings.filterwarnings('ignore')

class EfficientBayesianOptimizationBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(10, min(20, int(np.ceil(np.sqrt(self.dim) * 5))))
        
        # For reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # Algorithm parameters
        self.acq_func_type = "ei+"  # Expected Improvement plus variant for better exploration
        self.n_restarts = 3  # Number of restarts for acquisition function optimization
        self.batch_size = 1  # Initial batch size
        self.model_update_threshold = 5  # Minimum number of evaluations before updating model
        
        # For GPyTorch model
        self.gp_model = None
        self.likelihood = None
        self.is_model_trained = False
        
    def _sample_points(self, n_points):
        # Use Latin Hypercube Sampling for good space coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        points = sampler.random(n=n_points)
        # Scale to bounds
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        return points
    
    def _fit_model(self, X, y):
        # Use GPyTorch for more robust GP fitting
        if X.shape[0] < 2:
            return None
            
        # Convert to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.flatten(), dtype=torch.float32)
        
        # Create GP model with RBF kernel
        class ExactGPModel(ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = ConstantMean()
                self.covar_module = ScaleKernel(RBFKernel())
            
            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
        # Initialize likelihood and model
        self.likelihood = GaussianLikelihood()
        self.gp_model = ExactGPModel(X_tensor, y_tensor, self.likelihood)
        
        # Set model to training mode and optimize hyperparameters
        self.gp_model.train()
        self.likelihood.train()
        
        # Use a small number of optimization steps for efficiency
        optimizer = Adam([{'params': self.gp_model.parameters()}], lr=0.1)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.gp_model)
        
        # Optimization with limited iterations
        n_iter = min(20, max(5, int(np.ceil(np.log(self.X.shape[0]) * 5))))
        for _ in range(n_iter):
            optimizer.zero_grad()
            output = self.gp_model(X_tensor)
            loss = -mll(output, y_tensor)
            loss.backward()
            optimizer.step()
            
        # Set model to evaluation mode
        self.gp_model.eval()
        self.likelihood.eval()
        self.is_model_trained = True
        
        return self.gp_model
    
    def _acquisition_function(self, X):
        # Expected Improvement plus variant for better exploration
        if self.y is None or self.X is None or not self.is_model_trained:
            return np.ones((X.shape[0], 1)) * 1e-8
            
        # Get the best observed value so far (lower is better for minimization)
        best_y = np.min(self.y)
        
        # Convert to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Predict mean and std at candidate points
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.gp_model(X_tensor)
            mu = predictions.mean.detach().numpy()
            sigma = np.sqrt(predictions.variance.detach().numpy())
        
        # Avoid numerical issues with zero std
        sigma = np.maximum(sigma, 1e-8)
        
        # Expected Improvement calculation
        improvement = best_y - mu
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        
        # EI+ variant: Add distance-based exploration term
        if self.X is not None and len(self.X) > 1:
            # Compute pairwise distances to existing points
            distances = np.array([np.min(np.linalg.norm(X - x, axis=1)) for x in self.X])
            # Normalize distances
            max_dist = np.max(distances)
            if max_dist > 0:
                distances = distances / max_dist
            
            # Apply distance-based exploration bonus (higher bonus for points farther from existing ones)
            exploration_bonus = 0.01 * (1 - distances)  # Bonus decreases with proximity to existing points
            ei = ei + exploration_bonus.reshape(-1, 1)
        
        # Apply adaptive factor to control exploration/exploitation balance
        # In early iterations, favor exploration; later, exploit more
        exploitation_factor = max(0.1, 1.0 - (self.n_evals / self.budget) * 0.8)
        ei *= exploitation_factor
        
        return ei.reshape(-1, 1)
    
    def _select_next_points(self, batch_size):
        # Use a hybrid strategy to optimize acquisition function
        # Start with multiple random initial points for better exploration
        n_candidates = max(100, batch_size * 50)
        
        # Sample candidates using LHS
        candidates = self._sample_points(n_candidates)
        
        # Evaluate acquisition function for all candidates
        acq_values = self._acquisition_function(candidates)
        
        # Select top candidates based on acquisition value
        top_indices = np.argsort(acq_values.flatten())[-batch_size:][::-1]
        selected_points = candidates[top_indices]
        
        return selected_points
    
    def _evaluate_points(self, func, X):
        # Evaluate the points in X
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        if X is None or len(X) == 0:
            return np.array([])
            
        y_values = []
        for x in X:
            y_values.append(func(x))
            
        self.n_evals += len(X)
        return np.array(y_values).reshape(-1, 1)
    
    def _update_eval_points(self, new_X, new_y):
        # Update self.X and self.y
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])
    
    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Main minimize optimization loop
        # func: takes array of shape (n_dims,) and returns np.float64. 
        # !!! Do not call func directly. Use _evaluate_points instead and be aware of the budget when calling it. !!!
        # Return a tuple (best_y, best_x)
        
        # Initialize with random points
        init_points = self._sample_points(self.n_init)
        init_values = self._evaluate_points(func, init_points)
        self._update_eval_points(init_points, init_values)
        
        # Fit initial model
        self._fit_model(self.X, self.y)
        
        # Adaptive batch size adjustment
        if self.budget > 200:
            self.batch_size = min(5, max(1, int(np.ceil((self.budget - self.n_evals) / 50))))
        else:
            self.batch_size = 1
            
        # Main optimization loop
        while self.n_evals < self.budget and self.n_evals + self.batch_size <= self.budget:
            # If we have enough points, fit the model and optimize acquisition function
            if self.X.shape[0] > 1 and self.n_evals >= self.model_update_threshold:
                # Select next batch of points to evaluate
                next_batch = self._select_next_points(self.batch_size)
                
                # Evaluate the batch
                batch_values = self._evaluate_points(func, next_batch)
                
                # Update the evaluation history
                self._update_eval_points(next_batch, batch_values)
                
                # Only update model if we have enough evaluations or are near the end
                if (self.n_evals >= len(self.X) + self.model_update_threshold or 
                    self.budget - self.n_evals < 10):
                    self._fit_model(self.X, self.y)
                
                # Update batch size based on remaining budget
                if self.budget > 200:
                    self.batch_size = min(5, max(1, int(np.ceil((self.budget - self.n_evals) / 50))))
            else:
                # If we don't have enough points or are in early phase, evaluate more initial points
                remaining_budget = self.budget - self.n_evals
                if remaining_budget > 0:
                    additional_points = self._sample_points(min(remaining_budget, 10))
                    additional_values = self._evaluate_points(func, additional_points)
                    self._update_eval_points(additional_points, additional_values)
        
        # Return best point found
        best_idx = np.argmin(self.y)
        best_y = float(self.y[best_idx])
        best_x = self.X[best_idx].copy()
        
        return best_y, best_x
