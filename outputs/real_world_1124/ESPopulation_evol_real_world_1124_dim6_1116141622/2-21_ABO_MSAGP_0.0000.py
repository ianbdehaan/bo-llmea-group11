from collections.abc import Callable
from scipy.stats import qmc #If you are using QMC sampling, qmc from scipy is encouraged. Remove this line if you have better alternatives.
from scipy.stats import norm
import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import HalfCauchyPrior
import warnings

class ABO_MSAGP:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        # bounds has shape (2,<dimension>), bounds[0]: lower bound, bounds[1]: upper bound
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        # X has shape (n_points, n_dims), y has shape (n_points, 1)
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0 # the number of function evaluations
        self.n_init = max(5, min(20, budget // 4))  # Adaptive initial points based on budget

        # Random number generator for reproducibility
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        
        # GP model with fixed hyperparameters for efficiency
        self.gp_model = None
        self.is_gp_fitted = False
        
        # Acquisition function parameters
        self.noise_var = 1e-6  # Fixed noise variance to avoid expensive optimization
        self.exploration_factor = 0.01  # Adaptive exploration factor
        self.ucb_weight = 2.0  # Weight for UCB component in acquisition function
        
        # For numerical stability
        self.min_length_scale = 1e-4
        self.max_length_scale = 1e2

    def _sample_points(self, n_points):
        # Sample points using Latin Hypercube Sampling for good coverage
        sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
        samples = sampler.random(n=n=n_points)
        # Scale to bounds
        samples = self.bounds[0] + samples * (self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        # Fit a Gaussian Process with fixed hyperparameters for efficiency
        try:
            # Convert to torch tensors
            if isinstance(X, np.ndarray):
                X_torch = torch.FloatTensor(X)
                y_torch = torch.FloatTensor(y).reshape(-1)
            else:
                X_torch = X
                y_torch = y
            
            # Use GPyTorch for better numerical stability and flexibility
            if self.gp_model is None:
                class SimpleGPModel(ExactGP):
                    def __init__(self, train_x, train_y, likelihood):
                        super(SimpleGPModel, self).__init__(train_x, train_y, likelihood)
                        self.mean_module = ConstantMean()
                        # Use fixed length scale for computational efficiency
                        self.covar_module = ScaleKernel(
                            RBFKernel(length_scale_constraint=gpytorch.constraints.Interval(self.min_length_scale, self.max_length_scale))
                        )
                        
                    def forward(self, x):
                        mean_x = self.mean_module(x)
                        covar_x = self.covar_module(x)
                        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                
                # Initialize likelihood and model
                likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-8, 1e-2))
                self.gp_model = SimpleGPModel(X_torch, y_torch, likelihood)
                self.gp_model.likelihood.noise = torch.tensor(self.noise_var)
                
            # Set model to training mode
            self.gp_model.train()
            self.gp_model.likelihood.train()
            
            # Optimization using Adam
            optimizer = torch.optim.Adam([
                {'params': self.gp_model.parameters()},  # Includes likelihood parameters
            ], lr=0.1)
            
            # Use a simple optimization loop for efficiency
            mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
            
            # Limit optimization steps for computational efficiency
            num_steps = min(5, max(1, self.budget // 100))
            
            with gpytorch.settings.max_cholesky_size(1000), \
                 gpytorch.settings.fast_computations(log_prob=False, covar_diag=False):
                for _ in range(num_steps):
                    optimizer.zero_grad()
                    output = self.gp_model(X_torch)
                    loss = -mll(output, y_torch)
                    loss.backward()
                    optimizer.step()
            
            # Set model to evaluation mode
            self.gp_model.eval()
            self.gp_model.likelihood.eval()
            self.is_gp_fitted = True
            
        except Exception as e:
            warnings.warn(f"GP fitting failed with error: {e}")
            self.is_gp_fitted = False

    def _acquisition_function(self, X):
        # Expected Improvement + UCB hybrid acquisition function
        if not self.is_gp_fitted or self.y is None or len(self.y) == 0:
            # If no data yet or model not fitted, use a simple heuristic
            return np.ones((X.shape[0], 1)) * 0.1
        
        try:
            # Convert to torch tensor
            if isinstance(X, np.ndarray):
                X_torch = torch.FloatTensor(X)
            else:
                X_torch = X
            
            # Get GP predictions
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                mean, std = self.gp_model(X_torch).mean, self.gp_model(X_torch).stddev
                
            # Best observed value so far (for EI calculation)
            y_best = np.min(self.y)
            
            # Avoid numerical issues
            std = torch.clamp(std, min=1e-8)
            
            # Calculate Expected Improvement
            improvement = y_best - mean
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            
            # Calculate UCB component (for exploration)
            ucb = mean + self.ucb_weight * std
            
            # Hybrid acquisition: EI + weighted UCB
            hybrid_acq = ei + 0.1 * ucb  # 0.1 weight for UCB to balance exploitation/exploration
            
            return hybrid_acq.detach().numpy().reshape(-1, 1)
            
        except Exception as e:
            warnings.warn(f"Acquisition function calculation failed: {e}")
            return np.ones((X.shape[0], 1)) * 0.1

    def _select_next_points(self, batch_size):
        # Multi-start acquisition optimization for better exploration
        best_acq_value = -np.inf
        best_x = None
        
        # Try multiple random starts with different sampling strategies
        n_starts = min(5, max(1, batch_size * 2))  # Limit starts to avoid overhead
        
        # For diversity, we use a combination of different sampling methods
        for start_idx in range(n_starts):
            try:
                # Sample points using Sobol sequence for better distribution
                n_samples = min(1000, max(100, self.budget * 5))
                
                # Use different sampling strategies for each start
                if start_idx % 4 == 0:
                    # Random sampling
                    candidates = np.random.uniform(self.bounds[0], self.bounds[1], (n_samples, self.dim))
                elif start_idx % 4 == 1:
                    # Sobol sequence
                    sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
                    sobol_samples = sampler.random(n=n_samples)
                    candidates = self.bounds[0] + sobol_samples * (self.bounds[1] - self.bounds[0])
                elif start_idx % 4 == 2:
                    # LHS sampling
                    candidates = self._sample_points(n_samples)
                else:
                    # More diverse points using grid sampling
                    n_grid = int(np.ceil(n_samples ** (1.0 / self.dim)))
                    grid_points = np.linspace(self.bounds[0], self.bounds[1], num=n_grid)
                    meshgrid = np.meshgrid(*[grid_points] * self.dim, indexing='ij')
                    candidates = np.column_stack([g.ravel() for g in meshgrid])[:n_samples]
                
                # Evaluate acquisition function for all candidates
                acq_values = self._acquisition_function(candidates)
                
                # Select the best point among candidates
                max_idx = np.argmax(acq_values)
                if acq_values[max_idx] > best_acq_value:
                    best_acq_value = acq_values[max_idx]
                    best_x = candidates[max_idx].reshape(1, -1)
                    
            except Exception as e:
                warnings.warn(f"Start {start_idx} failed: {e}")
                continue
        
        # Fallback to simple sampling if no good point found
        if best_x is None:
            return self._sample_points(batch_size)
        
        # Return the best point found
        return np.array(best_x) if best_x is not None else self._sample_points(batch_size)

    def _evaluate_points(self, func, X):
        # Evaluate the points in X using function evaluation
        # func: takes array of shape (n_dims,) and returns np.float64.
        # return array of shape (n_points, 1)
        
        if X is None or len(X) == 0:
            return np.array([])
            
        y = []
        for x in X:
            try:
                y.append(func(x))
            except Exception as e:
                warnings.warn(f"Function evaluation failed: {e}")
                # Fallback to a large value if evaluation fails
                y.append(1e6)
            
        self.n_evals += len(X)
        return np.array(y).reshape(-1, 1)

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
        
        # Initialize with initial samples
        if self.X is None:
            X_init = self._sample_points(self.n_init)
            y_init = self._evaluate_points(func, X_init)
            self._update_eval_points(X_init, y_init)
        
        # Fit the model on initial data
        self._fit_model(self.X, self.y)
        
        # Optimization loop
        while self.n_evals < self.budget:
            # Fit model with current data
            try:
                self._fit_model(self.X, self.y)
            except Exception as e:
                warnings.warn(f"Model fitting failed: {e}")
                # Fallback to simple sampling if model fitting fails
                break
            
            # Adaptive batch sizing based on remaining budget and problem characteristics
            remaining_budget = self.budget - self.n_evals
            if remaining_budget <= 10:
                batch_size = 1
            elif remaining_budget <= 50:
                batch_size = min(3, max(1, remaining_budget // 10))
            else:
                # Use larger batches when more budget is available
                batch_size = min(5, max(1, remaining_budget // 20))
            
            # Occasionally increase exploration factor if we're not making progress
            if self.n_evals > self.budget * 0.3 and np.std(self.y) < 0.1:
                self.ucb_weight = max(1.0, self.ucb_weight * 0.95)
            
            # Select next points using acquisition function
            next_X = self._select_next_points(batch_size)
            
            # Evaluate the selected points
            next_y = self._evaluate_points(func, next_X)
            
            # Update the data
            self._update_eval_points(next_X, next_y)
            
            # Early termination if budget is almost exhausted
            if self.n_evals >= self.budget:
                break
                
        # Return best solution found
        best_idx = np.argmin(self.y)
        return float(self.y[best_idx]), self.X[best_idx].copy()
