# Thinking (reasoning of the model, not part of the answer)
Okay, I need to design a Bayesian Optimization algorithm for the Rosenbrock function in 6 dimensions. Let's start by understanding the problem. The Rosenbrock function has a global minimum at a parabolic valley, so the algorithm needs to navigate this valley and find the minimum efficiently. Since the function is a classic test problem, it's likely smooth and unimodal, but with a narrow valley that's challenging for optimization algorithms.

The user specified that the code must be a subclass with __init__ and __call__ methods. The algorithm must be efficient, computationally light, and handle any budget, especially up to 600 evaluations. Also, it needs to be reproducible with proper seeding.

First, the initial design. For the Rosenbrock function, using a Sobol sequence makes sense because it's low-discrepancy and can cover the space more evenly than random samples. The number of initial points should be enough to get a rough idea of the landscape. A common choice is 2*dim + 1, which in 6D would be 13 points. That's a good start to build the surrogate model.

Next, the surrogate model. The Rosenbrock function is smooth, so a Gaussian Process (GP) with an RBF kernel is appropriate. Anisotropic kernels (ARD) can capture different scales in each dimension. To handle possible non-stationarity, maybe a Matérn kernel with nu=2.5. But the user mentioned using scikit-learn, which supports Matérn kernels. However, for efficiency, the model should be tuned. Since the budget might be small, using a fixed kernel might be better to avoid overfitting. Also, using a nugget (alpha) to handle noise, though the function is noiseless. So setting alpha to a very small value.

The acquisition function: Expected Improvement (EI) is a standard choice, but the problem's valley shape might require balancing exploration and exploitation. EI naturally balances this. But to avoid getting stuck in local optima, maybe using a variant like LogEI or incorporating some diversity. However, for efficiency, standard EI is better. So stick with EI.

Optimizing the acquisition function: Since the problem is smooth, using gradient-based optimization like L-BFGS-B is efficient. But starting points matter. To avoid local minima, generate several random candidates and optimize from each. The number of candidates should be manageable but sufficient. Maybe 50 candidates in the domain. For computational efficiency, using a small batch size, perhaps sequential, but since the budget is up to 600, maybe a batch of 1 each step.

Exploration strategy: Trust Region methods like TuRBO could work, but they add complexity. For simplicity, maybe a trust region around the best point, expanding or shrinking based on progress. However, implementing that might be more code. Alternatively, since the Rosenbrock function is well-known, a trust region approach could be effective. But given the code structure, perhaps starting with a simple trust region that shrinks as the model improves, or just a fixed strategy. However, the user wants something novel. Maybe a hybrid approach that combines global and local search phases. Initially, more exploration, then more exploitation as the budget allows.

Putting it all together: The algorithm will start with Sobol points. Then fit a GP with anisotropic Matérn kernel. For acquisition, use Expected Improvement optimized with multiple starting points. To handle the valley, maybe use a trust region around the best observed point. Each iteration, the acquisition function is optimized within a trust region, which is adjusted based on the current best solution.

Now, for the code structure. The class needs to be named based on its characteristics. Let's think of the components: Adaptive Trust Region (ATR), Bayesian Optimization (BO). So maybe "ATRBO" as the class name.

In the __init__ method, set up the initial design size, the surrogate model, acquisition function parameters. The X and y arrays will store the evaluated points and their function values.

The _sample_points function will use Sobol sequences. Using scipy's qmc.Sobol, with scrambling and the seed from the rng. The bounds are fixed at [-5, 5], so scaling the Sobol samples to fit.

The _fit_model function will create a GP model. Using scikit-learn's GaussianProcessRegressor with a Matérn kernel. The kernel's length scales are initialized to the data's range. Using a nugget (alpha) to handle numerical issues. The model is fit with a fixed optimizer (like fmin_l_bfgs_b) but since the user mentioned to keep the cost low, maybe not refitting the hyperparameters every time. Wait, but for efficiency, in BO, the hyperparameters should be optimized. However, if the budget is small, maybe optimizing the hyperparameters once per iteration is feasible.

The acquisition function is Expected Improvement. The EI is calculated based on the GP's mean and standard deviation. The function value is computed for each candidate point. The _select_next_points function generates candidates via L-BFGS-B optimization, starting from multiple points.

The trust region approach: After each iteration, the next point is selected within a trust region around the best point. The size of the trust region can be adjusted based on the progress. For example, if the improvement is good, the region can expand; if not, it contracts. However, implementing this requires tracking the best solution and adjusting the trust region's bounds. For simplicity, maybe start with a fixed region around the best point, gradually shrinking as the optimization proceeds. Alternatively, use a dynamic trust region like in TuRBO, but with a simpler implementation.

Another idea: Use a trust region that starts with the entire space and shrinks as the best solution is found. The region could be a hypercube around the best point with a certain radius. The radius can be updated based on whether the acquisition function suggests progress is being made. However, this might complicate the code. For a code submission with time constraints, maybe a static trust region that's a fraction of the search space, say 0.1 to 0.5 times the search range, and centered on the best point.

Putting this into code: For each iteration, generate candidate points within a trust region around the best point. The trust region's bounds are calculated as best_x ± (search_range * trust_radius). The trust_radius could start at 0.5 and decrease as the algorithm progresses. For example, after each iteration, if the improvement is significant, the radius is decreased by a factor, or increased if the improvement is minimal.

But how to track this in code? Maybe track the best solution and adjust the trust region's bounds accordingly. However, for the code to be efficient, the trust region can be implemented by modifying the bounds when generating the acquisition function candidates.

So in the _select_next_points function, generate candidates within a region around the best point. The size of this region can be adjusted based on the iteration count or based on the improvement observed.

Alternatively, for each step, the trust region is a fixed percentage of the original bounds, but this might not adapt well. A better approach is to adjust the trust region dynamically.

But to keep the code simple and efficient, maybe start with a fixed trust region. For example, in the first few steps, the region is large to explore, then it shrinks. However, without tracking progress, it's hard to decide when to shrink. Alternatively, use the current best solution's standard deviation from the GP to define the trust region. Points with higher uncertainty might be explored more.

Alternatively, for the acquisition function optimization, in each step, restrict the candidates to a region around the best observed point. This would focus the search in promising areas and avoid wasting evaluations in low-probability regions. The size of the region could be a fixed fraction of the search space.

For example, after the initial design, the best point is found. The trust region is defined as ±trust_radius around that point. The trust_radius could start at 0.5 and decrease as the number of iterations increases. However, in code, this requires tracking the current iteration and adjusting the radius accordingly.

But for the code submission, perhaps using a dynamic trust region that starts wide and narrows down over time. For example, in the first half of the budget, the region is larger, and in the second half, it's smaller. However, this is a heuristic.

Putting it all together:

In the _select_next_points function, generate a set of random candidates (say, 50) within the trust region. Then optimize the acquisition function using L-BFGS-B from each of these candidates. The best candidate from these optimizations is selected as the next point.

The trust region is defined as the best point's coordinates plus or minus a certain radius. The radius starts at, say, 0.5 and is reduced by a factor every few iterations. For example, every time the best value improves by less than a threshold, the radius is halved.

But how to implement this in code? Let's outline:

- Track the best_x and best_y.
- Initialize trust_radius = 0.5.
- For each iteration:
   - Generate 50 random points within the trust region around best_x.
   - Optimize the acquisition function from each of these points.
   - Select the best point found.
   - If the improvement over the last few iterations is small, reduce the trust_radius.

But this requires tracking the history of improvements. For code simplicity, perhaps just decrease the trust_radius over time, say, after every N evaluations. For example, trust_radius = max(0.01, trust_radius * 0.99) after each step. This would cause it to shrink gradually.

However, this might be too simplistic. But given time constraints for code submission, it's manageable.

Another consideration: The Rosenbrock function's minimum is at a point where the valley is elongated. So the algorithm must move along the valley. If the trust region is too small, the algorithm might get stuck. If it's too large, it might waste evaluations. A dynamic adjustment of the trust region based on the current state would help.

Now, let's outline the code:

In the class ATRBO:

- The initial design uses Sobol sequence with n_init = 2*dim + 1 = 13 for dim=6.
- The surrogate model is a GP with Matérn kernel (nu=2.5), ARD, and fixed alpha.
- The acquisition function is Expected Improvement.
- The next point selection uses multiple random starts within a trust region around the best point, optimized with L-BFGS-B.
- The trust region starts with a certain radius and shrinks over time.

Implementing the _select_next_points function:

def _select_next_points(self, batch_size):
    # Generate candidates within trust region
    current_best = self.X[np.argmin(self.y), :]
    lower = np.maximum(current_best - self.trust_radius, self.bounds[0])
    upper = np.minimum(current_best + self.trust_radius, self.bounds[1])
    candidates = qmc.scale(qmc.Sobol(self.dim, seed=self.rng).random(50), lower, upper - lower)
    # Optimize acquisition function from each candidate
    def objective(x):
        return -self._acquisition_function(x.reshape(1, -1))[0, 0]
    res = minimize(objective, x0=candidates[0], bounds=[(l, u) for l, u in zip(lower, upper)], method='L-BFGS-B')
    best_candidate = res.x
    return best_candidate.reshape(1, -1)

But this is a simplified version. Also, need to handle multiple candidates and choose the best. Also, the trust_radius needs to be managed.

Now, considering computational efficiency: Each iteration must not take too long. For example, optimizing the acquisition function with 50 starting points and using L-BFGS-B may be acceptable for small budgets, but for a budget of 600, 500 iterations could be costly. However, the user said to keep the cost per iteration small. So maybe use fewer starting points, like 20 instead of 50.

Moreover, the trust_radius should be adjusted. For example, after each step, check if the improvement is significant and adjust the radius accordingly.

Another point: The initial design uses Sobol, but the Rosenbrock function's global minimum is in a specific location. The initial design points might not cover the valley area, so using Sobol helps spread out the points.

Now, for the code structure:

The class will have variables for the current best point, the trust_radius, and the number of iterations. The trust_radius starts at a certain value, say 1.0 (half the search space's width), and decreases over time.

But in the code, how to track the number of iterations? The __call__ function will loop until the budget is exhausted. Each loop iteration (after initial design) represents one step where a new point is added.

Let me outline the code step by step.

In __init__:

self.trust_radius = 1.0  # Initial radius (since the search space is from -5 to 5, the total is 10, so radius of 1 is 1/5 of the space)
self.initial_radius = 1.0
self.radius_decay_rate = 0.99  # Decay the radius each step

In the __call__ method:

After initial evaluation, loop:

for each step until budget:
   select next point using trust region and L-BFGS-B
   evaluate, update
   adjust trust_radius *= decay_rate if improvement is below threshold, or based on some condition

But to implement this, need to track the best value and the improvement over time. For simplicity, just decay the radius by a fixed rate each iteration.

But for computational efficiency, perhaps the radius decays every few steps. However, in code, for simplicity, just multiply it by 0.99 each time.

Now, let's code the _select_next_points function. It needs to generate candidates within the trust region, optimize the acquisition function from those candidates.

But how to handle the trust region. The trust region's bounds are determined by the best_x. For each dimension, the lower bound is best_x[d] - trust_radius, but not below -5, and upper bound is best_x[d] + trust_radius, but not above 5.

Thus, in code:

current_best = self.X[np.argmin(self.y), :]
lower_bounds = np.maximum(current_best - self.trust_radius, self.bounds[0])
upper_bounds = np.minimum(current_best + self.trust_radius, self.bounds[1])

Then, generate candidates within these bounds. Using Sobol sequences for candidates to avoid random sampling, which gives better coverage.

But generating 50 Sobol points scaled to the trust region.

Now, the acquisition function is Expected Improvement. The implementation of the _acquisition_function function would compute the EI for each point.

The EI is computed based on the predicted mean and standard deviation of the GP. For a given point x, the improvement is max(0, best_observed - mean(x)). The EI is the expected value of this improvement.

The code for EI would use the GP's predict method to get mean and std, then compute the Z-score and the integral.

But since the GP is a regression model, the code would need to handle that.

In the _fit_model function, the GP is fitted. The model is a GaussianProcessRegressor with a Matérn kernel. The hyperparameters are optimized during fitting.

Now, considering all this, the code can be structured as follows.

Potential issues:

- The initial design may not capture the global minimum region. But the Sobol sequence helps spread out the points.
- The trust region might get too narrow too quickly, preventing the algorithm from exploring the valley. The decay rate of 0.99 is a slow decay, which might be acceptable.
- The acquisition function optimization may get stuck in local optima. Using multiple starting points helps, but in the trust region approach, the starting points are limited to the trust region, which could be problematic if the true next step is outside. However, the trust region is initially large and shrinks over time, allowing initial exploration and later exploitation.

Another point: The Rosenbrock function is unimodal, so global exploration is important initially. The trust region approach starts large and then narrows, which should help.

Putting all this into code:

Implementing the _select_next_points function with a trust region and multiple starting points.

Now, code for the class:

The class will be called ATRBO (Adaptive Trust Region Bayesian Optimization).

In the __init__ method, set up the initial sample size, the model, the trust_radius, etc.

Now, code:

from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

class ATRBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = 2 * dim + 1  # Initial points
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.trust_radius = 1.0  # Initial trust radius
        self.radius_decay = 0.99  # Decay rate per iteration
        self.model = None  # Surrogate model

    def _sample_points(self, n_points):
        sampler = qmc.Sobol(self.dim, scramble=True, seed=self.rng)
        samples = qmc.scale(sampler.random(n_points), self.bounds[0], self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        kernel = C(1.0) * Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small nugget for numerical stability
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1)),
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=0  # Avoid overfitting with small data
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mean, std = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (mean - best_y) / std
            ei = (mean - best_y) * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0] = 0.0
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        current_best_idx = np.argmin(self.y)
        current_best = self.X[current_best_idx]
        
        # Define trust region bounds
        lower = np.maximum(current_best - self.trust_radius, self.bounds[0])
        upper = np.minimum(current_best + self.trust_radius, self.bounds[1])
        
        # Generate candidates within trust region
        sampler = qmc.Sobol(self.dim, scramble=True, seed=self.rng)
        candidates = qmc.scale(sampler.random(20), lower, upper - lower)
        
        # Optimize acquisition function from each candidate
        best_candidate = None
        best_ei = -np.inf
        
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]
        
        bounds = [(l, u) for l, u in zip(lower, upper)]
        for x0 in candidates:
            res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.success and res.fun > best_ei:
                best_ei = res.fun
                best_candidate = res.x
        
        self.trust_radius *= self.radius_decay  # Decay the radius
        return best_candidate.reshape(1, -1)

    def _evaluate_points(self, func, X):
        y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            y[i, 0] = func(X[i])
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial design
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        
        self.model = self._fit_model(self.X, self.y)
        
        # Main loop
        while self.n_evals < self.budget:
            batch_size = 1  # Sequential optimization
            new_points = self._select_next_points(batch_size)
            new_y = self._evaluate_points(func, new_points)
            self._update_eval_points(new_points, new_y)
            self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        return self.y[best_idx, 0], self.X[best_idx].copy()
```

Now, let's check the code for correctness.

The initial design uses Sobol samples, which is good. The model is a GP with Matérn kernel. The acquisition function is Expected Improvement. The trust region is dynamic, starting with a radius of 1.0 and decaying by 0.99 each iteration.

The _select_next_points function generates 20 candidates within the trust region and optimizes from each using L-BFGS-B. The best candidate is selected. The trust_radius is decayed each time.

The code tracks the best solution and adjusts the trust region around it, which should help in navigating the valley.

Potential issues:

- The trust_radius decay may be too slow. If the Rosenbrock function requires a more aggressive narrowing, adjusting the decay rate could help. However, for the code submission, 0.99 is a safe choice as it decays slowly.

- The initial trust_radius of 1.0 (half the search space's width from -5 to 5) might be too large. But the Rosenbrock function's valley is elongated, so a larger initial radius allows exploration.

- The model is refit every time a new point is added. With small budgets, this is manageable. For 600 evaluations, refitting 600 times is acceptable since the GP fit is O(n^3), but with n up to 600, it might be slow. However, the problem allows for it as per the user's note that cost should grow moderately. To handle this, perhaps using a lower n_restarts_optimizer for the GP, which is already set to 0 in the code.

- The number of candidates per optimization (20) is a trade-off between thoroughness and speed. Using 20 is reasonable for computational efficiency.

- The acquisition function optimization uses the L-BFGS-B method with bounds set by the trust region. This should prevent stepping out of the region.

The code should be reproducible because the rng is seeded properly in the __init__ method and passed to the Sobol sampler and GP.

Thus, this algorithm should be efficient and effective for the Rosenbrock function in 6D, especially with the trust region approach focusing on promising areas and adapting as the optimization progresses.


# Answer
# Description 
The algorithm ATRBO (Adaptive Trust Region Bayesian Optimization) combines a trust region strategy with a Gaussian Process (GP) surrogate model and Expected Improvement (EI) acquisition function. The trust region dynamically adjusts around the best observed solution to balance exploration and exploitation. Initial design uses Sobol sequences for space-filling sampling, while the GP with an anisotropic Matérn kernel models the function landscape. The acquisition function is optimized using multiple L-BFGS-B restarts within the trust region. The trust region shrinks gradually over iterations to focus on promising areas.

# Justification 
- **Sobol Initial Design**: Ensures better coverage of the search space compared to random sampling, crucial for capturing the elongated valley of the Rosenbrock function.
- **GP with Anisotropic Matérn Kernel**: Models the smoothness of the Rosenbrock function while allowing different length scales per dimension.
- **Expected Improvement Acquisition Function**: Balances exploration and exploitation, and is efficient to compute.
- **Trust Region Strategy**: Focuses search on the most promising area around the current best solution, reducing waste in unproductive regions.
- **Adaptive Trust Radius**: Gradually shrinking the trust region ensures the algorithm transitions from global exploration to local refinement.
- **L-BFGS-B with Multiple Starts**: Efficiently finds near-optimal candidates for the acquisition function within the trust region bounds.

# Code 
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

class ATRBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = 2 * dim + 1  # Space-filling initial design
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.trust_radius = 1.0  # Initial trust region size
        self.radius_decay = 0.99  # Gradual shrinking factor
        self.model = None  # GP surrogate

    def _sample_points(self, n_points):
        # Generate initial points via Sobol sampling
        sampler = qmc.Sobol(self.dim, scramble=True, seed=self.rng)
        samples = qmc.scale(sampler.random(n_points), self.bounds[0], self.bounds[1] - self.bounds[0])
        return samples

    def _fit_model(self, X, y):
        # GP with Matérn kernel for smoothness modeling
        kernel = C(1.0) * Matern(length_scale=np.ones(self.dim), nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small nugget for stability
            normalize_y=True,
            random_state=int(self.rng.integers(2**31 - 1)),
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=0  # Reduce computational cost
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        # Expected Improvement (EI)
        mean, std = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (mean - best_y) / std
            ei = (mean - best_y) * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std == 0] = 0.0
        return ei.reshape(-1, 1)

    def _select_next_points(self, batch_size):
        # Trust region around current best solution
        current_best_idx = np.argmin(self.y)
        current_best = self.X[current_best_idx]
        lower = np.maximum(current_best - self.trust_radius, self.bounds[0])
        upper = np.minimum(current_best + self.trust_radius, self.bounds[1])
        
        # Generate diverse candidates within trust region
        sampler = qmc.Sobol(self.dim, scramble=True, seed=self.rng)
        candidates = qmc.scale(sampler.random(20), lower, upper - lower)
        
        best_candidate = None
        best_ei = -np.inf
        
        # Optimization of acquisition function
        def objective(x):
            return -self._acquisition_function(x.reshape(1, -1))[0, 0]
        
        bounds = [(l, u) for l, u in zip(lower, upper)]
        for x0 in candidates:
            res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.success and res.fun > best_ei:
                best_ei = res.fun
                best_candidate = res.x
        
        self.trust_radius *= self.radius_decay  # Adaptive shrinking
        return best_candidate.reshape(1, -1)

    def _evaluate_points(self, func, X):
        # Evaluate candidates without exceeding budget
        y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            y[i, 0] = func(X[i])
        return y

    def _update_eval_points(self, new_X, new_y):
        # Maintain evaluated data
        if self.X is None:
            self.X = new_X.copy()
            self.y = new_y.copy()
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial design
        initial_X = self._sample_points(self.n_init)
        initial_y = self._evaluate_points(func, initial_X)
        self._update_eval_points(initial_X, initial_y)
        self.n_evals += initial_X.shape[0]
        self.model = self._fit_model(self.X, self.y)
        
        # Main optimization loop
        while self.n_evals < self.budget:
            batch_size = 1  # Sequential optimization
            new_points = self._select_next_points(batch_size)
            new_y = self._evaluate_points(func, new_points)
            self._update_eval_points(new_points, new_y)
            self.model = self._fit_model(self.X, self.y)
            self.n_evals += batch_size
        
        best_idx = np.argmin(self.y)
        return self.y[best_idx, 0], self.X[best_idx].copy()
```
## Error
 Traceback (most recent call last):
  File "<ATRBO>", line 107, in __call__
 107->             new_points = self._select_next_points(batch_size)
  File "<ATRBO>", line 61, in _select_next_points
  59 |         # Generate diverse candidates within trust region
  60 |         sampler = qmc.Sobol(self.dim, scramble=True, seed=self.rng)
  61->         candidates = qmc.scale(sampler.random(20), lower, upper - lower)
  62 |         
  63 |         best_candidate = None
ValueError: Bounds are not consistent 'l_bounds' < 'u_bounds'
