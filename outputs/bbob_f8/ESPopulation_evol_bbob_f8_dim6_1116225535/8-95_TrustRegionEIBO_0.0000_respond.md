# Thinking (reasoning of the model, not part of the answer)
Okay, let's tackle this problem. The user wants me to design a new Bayesian Optimization algorithm that's different from the existing ones mentioned. The previous algorithms all use a Matérn kernel with LCB or EI acquisition, and Sobol sequences for initial design and acquisition optimization. My task is to come up with something novel.

First, the current solutions focus on Matérn kernels with exponential decay in the LCB. To differentiate, maybe I can switch the surrogate model. The existing ones use Gaussian Processes. What if I use a different model, like a Random Forest? But the user mentioned GPs are part of the allowed libraries, but maybe combining with other models could help. However, the problem is a smooth function (Rosenbrock), so perhaps a GP is better. Alternatively, maybe a more adaptive approach, like using a local GP model around the best point. That could reduce computational overhead and focus on the promising regions.

Next, acquisition functions. The existing ones use LCB with exponential decay. Maybe a different acquisition function, like Expected Improvement (EI), but with a twist. Or perhaps a portfolio approach that switches between acquisition functions. But the user wants computational efficiency. Another idea: using a trust region approach. TuRBO (Trust Region Bayesian Optimization) uses expanding/shrinking trust regions around the best point. This is different from the global approach of the previous algorithms.

Trust region methods adaptively adjust the search space around the best solution. This could be more efficient in navigating the elongated valley of the Rosenbrock function. The algorithm would start with a small trust region and expand it as the search progresses, allowing exploitation close to the current best and exploration further out when needed.

The surrogate model could still be a GP, but localized to the trust region. This reduces the model's complexity and might lead to better modeling in the vicinity of the current best point. The acquisition function could use Expected Improvement (EI) within the trust region, or a hybrid approach. The trust region size could be adjusted based on the success of the previous steps.

Initial design could use Sobol, but with a smaller initial set since the trust region will expand. The acquisition optimization can be done within the current trust region, potentially using a smaller number of candidates and L-BFGS-B. The trust region management is crucial here. Parameters like the expansion factor and the minimum/maximum size of the trust region need to be set.

For the code structure: the class will need to track the current trust region bounds. Each iteration selects a point within the trust region, evaluates it, and updates the region based on whether the new point is better than the current best. The GP model is retrained each time with the new data, but only the data within the trust region is considered? Wait, no. The trust region is about the search for the next point, not the model. The model uses all data. Hmm, maybe the trust region is just for the acquisition function's candidate selection. So the acquisition function is optimized within the trust region bounds.

Let me outline the steps:

1. Initialize with a Sobol design, maybe 2d points (since the previous used 3d). The initial trust region could be the entire space.

2. For each iteration:
   a. Optimize the acquisition function within the current trust region.
   b. Evaluate the new point.
   c. If the new point is better, expand the trust region. If not, shrink it.
   
3. Use a GP with an anisotropic kernel (like Matérn) for the surrogate model.
4. Acquisition function could be EI, which is different from the previous LCBs. EI tends to favor exploitation but can be tuned.

Why EI? It's a standard acquisition function that balances exploration and exploitation. But combining it with a trust region strategy would adaptively control the search area, which is different from the previous algorithms. The trust region approach could help in efficiently navigating the valley of the Rosenbrock function by focusing the search where the function is likely to be lower.

Also, the previous algorithms used exponential decay for the LCB's exploration parameter. The trust region method doesn't need that since the exploration is controlled by the region size. 

Implementation steps in code:

- The class needs to track the current trust region bounds. For example, a lower and upper bound for each dimension.
- In the _select_next_points method, generate candidates within the current trust region.
- The trust region could start as the entire bounds and adjust based on the success rate of evaluations. For example, if the new point is better than the current best, expand the region; if not, shrink it. The expansion and contraction factors can be tuned.

Potential issues: How to adjust the trust region. TuRBO uses a certain strategy for that. Maybe I can use similar logic: track the number of steps where the new point is better than the best in the trust region. If it's a certain number, expand; otherwise, shrink.

But given the time constraints and computational budget, perhaps a simpler approach. For example, if the new point is better, expand the trust region by a factor (say 1.5) in each dimension. If not, shrink by 0.9. But the expansion and shrinkage should be done in a way that doesn't let the trust region become too large or too small.

Another consideration: the initial Sobol design. Maybe use fewer initial points since the trust region will expand. However, the Rosenbrock function has a valley, so initial points should cover it. The previous used 3d, maybe use 2d here to save on budget, but need to balance between coverage and budget.

The surrogate model remains a GP with Matérn kernel, but the acquisition function is Expected Improvement (EI). The trust region strategy changes how the candidates are selected and optimized, focusing the search more locally as the optimization progresses.

Testing this approach could lead to better performance by balancing global and local search through the trust region. The EI acquisition function is known to work well with trust regions, as it focuses on improving the current best solution.

Now, the code:

The class will need to track the trust region bounds. Let's call it self.tr_bounds, initialized as the original bounds. Each iteration, when selecting the next point, generate candidates within the current tr_bounds. After evaluating, adjust the bounds based on whether the new point is better.

The _select_next_points method will generate candidates within the tr_bounds, optimize the acquisition (EI), and then select the best point. Then, based on the result, adjust the trust region.

For the acquisition function, compute EI using the GP's mean and standard deviation. The EI formula is:

EI(x) = (μ(x) - best_f) * Φ((μ(x) - best_f)/σ(x)) + σ(x) * φ((μ(x) - best_f)/σ(x))

Where Φ is the standard normal CDF and φ is the PDF.

But since we're minimizing, the acquisition function would be negated. Wait, the Rosenbrock is a minimization problem. So, we want to maximize the Expected Improvement. The EI is defined for minimization as the expected improvement over the current best. So the acquisition function is:

current_best = min(y)
improvement = max(0, current_best - mu(x))  # Wait, no. For minimization, improvement is current_best - y, but since we want to minimize, we want to find x where y is smaller. So the expected improvement is the amount by which we expect the new point to be better.

The standard EI formula for minimization is:

EI(x) = E[ max( current_best - f(x), 0 ) ]

Which can be computed with:

z = (current_best - mu) / sigma
if sigma == 0:
    ei = 0
else:
    ei = (current_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

But since we're trying to minimize, we need to maximize this EI. So the acquisition function will return the EI value, and we select the x with the highest EI.

In code, using scipy.stats.norm for the cdf and pdf.

So, in the _acquisition_function method, compute the mu and sigma, then calculate the EI.

Now, the trust region adjustment. Let's use a simple approach: if the new point improves the best value, expand the trust region. If not, shrink it. The expansion and contraction can be controlled with factors.

For example:

- expansion_factor = 1.5
- contraction_factor = 0.9

But to avoid the trust region becoming too large, we can set a maximum size (original bounds) and minimum size.

So, after evaluating a new point, check if the new y is better than the current best. If it is, expand the trust region in each dimension by expansion_factor. If not, shrink by contraction_factor. But the expansion and contraction can't go beyond the original bounds.

However, in the code, each trust region adjustment would need to update the bounds. Let's think about how to implement this.

In the __call__ method, after evaluating a new point and updating the best, we adjust the trust region:

If new_y < best_y:
    expand the trust region in each dimension towards the best direction.
Else:
    shrink the trust region.

But how to compute the direction for expansion? Maybe expand the trust region in all directions. Alternatively, adjust based on the position of the new point relative to the current best. But this could get complex. A simpler approach is to uniformly expand or shrink the trust region in all dimensions.

Let me outline the code steps:

- Initialize tr_bounds as self.bounds.
- For each iteration after the initial phase:
   - Generate candidates within tr_bounds.
   - Select the candidate with highest EI.
   - Evaluate the candidate.
   - If the new point's y is better than the best, expand the tr_bounds.
   - Else, shrink the tr_bounds.

Expanding and shrinking the bounds:

For each dimension, the current tr_bounds are a lower and upper limit. To expand, we can increase the range by a factor. For example, if the current tr is [l, u], expanding by 1.5 would mean:

new_l = best_x[dim] - 1.5 * (u - best_x[dim]) / 2
Wait, maybe a better way is to adjust the center and width. Suppose the current tr is centered around the best_x (since we're in the trust region). But how to determine the center?

Alternatively, the trust region can be centered around the current best_x, with a certain radius. For example, the trust region is a hypercube centered at best_x, with side length determined by the radius. Then, expanding the radius or shrinking it based on the success.

This is similar to the TuRBO approach, but since this is a simplified version, let's proceed.

So, the trust region is a hypercube around the current best_x. Let's say the radius starts at the full space (initially, but that might not be efficient). Wait, the initial trust region is the entire space. Then, in each iteration, we adjust the radius based on whether the new point is better.

But implementing this requires tracking the radius. However, for simplicity, perhaps track the current bounds and adjust them each time.

For example:

current best_x is x.

If the new point is better, the trust region is expanded in all directions. So the new bounds for each dimension would be:

lower = max(current_lower, x[dim] - expansion_factor * (current_upper - x[dim]))
upper = min(current_upper, x[dim] + expansion_factor * (current_upper - x[dim]))

Wait, maybe not. Alternatively, expand the trust region to cover more of the space. But the exact method depends on the strategy.

Alternatively, use a fixed expansion/contraction factor. Let's say the trust region is defined by a lower and upper bound for each dimension. When expanding, we increase the range in each dimension by a factor. However, the trust region can't exceed the original bounds.

But how to compute the new lower and upper? For example, the current lower and upper for dimension d is [ld, ud]. The best_x in this dimension is x[d]. To expand, maybe extend the bounds by a factor, keeping x[d] as the center.

So new_lower = x[d] - expansion_factor * (ud - x[d])
new_upper = x[d] + expansion_factor * (x[d] - ld) ?

Wait, that's not right. Alternatively, the current width of the trust region in dimension d is (ud - ld). Expanding by a factor of 1.5 would make the new width 1.5 times the current width. The center remains the best_x[d].

So new_center = x[d]
new_width = expansion_factor * (ud - ld)
new_lower = new_center - new_width / 2
new_upper = new_center + new_width / 2

But this assumes that the trust region is symmetric around the best_x. However, if the best_x is near a boundary, expanding could cause the new_lower or new_upper to go beyond the original bounds. So we need to clip them to the original bounds.

This seems manageable. Let's code this logic.

In the __call__ method, after evaluating a new point, check if it's better. Then adjust the trust region bounds accordingly.

So, in code:

For example:

if new_y < best_y:
    # expand the trust region
    expansion_factor = 1.5
    for d in range(self.dim):
        center = best_x[d]
        current_width = tr_bounds[1][d] - tr_bounds[0][d]
        new_width = expansion_factor * current_width
        new_lower = center - new_width / 2
        new_upper = center + new_width / 2
        # clip to original bounds
        new_lower = max(new_lower, self.bounds[0][d])
        new_upper = min(new_upper, self.bounds[1][d])
        tr_bounds[0][d] = new_lower
        tr_bounds[1][d] = new_upper
else:
    # shrink the trust region
    contraction_factor = 0.9
    for d in range(self.dim):
        center = best_x[d]
        current_width = tr_bounds[1][d] - tr_bounds[0][d]
        new_width = contraction_factor * current_width
        new_lower = center - new_width / 2
        new_upper = center + new_width / 2
        # clip to original bounds
        new_lower = max(new_lower, self.bounds[0][d])
        new_upper = min(new_upper, self.bounds[1][d])
        tr_bounds[0][d] = new_lower
        tr_bounds[1][d] = new_upper

This way, the trust region expands or contracts symmetrically around the current best_x, but can't exceed the original bounds.

Now, in the code structure, the initial tr_bounds are self.bounds. Each time a new point is evaluated, the trust region is adjusted.

The initial design can be smaller, say 2d points, since the trust region will expand. This saves initial evaluations and allows more iterations for the trust region to adapt.

The acquisition function uses Expected Improvement. The code for _acquisition_function would need to compute EI.

Putting it all together, the new algorithm uses a Trust Region with Expected Improvement acquisition and GP model with Matérn kernel. The key difference from previous algorithms is the use of a trust region to control the search area, adapting it based on the success of the evaluations. This approach focuses the search more on promising regions while allowing controlled exploration when necessary.

Now, code implementation:

- The class needs to track tr_bounds. Initially, it's self.bounds.

In __init__:

self.tr_bounds = self.bounds.copy()

In _select_next_points, generate candidates within tr_bounds.

In __call__, after evaluating the new point and updating best, adjust the tr_bounds.

Now, code:

Implementing the _acquisition_function for EI.

Using scipy.stats.norm for the calculations.

So in the code:

import scipy.stats as stats

def _acquisition_function(self, X):
    mu, sigma = self.model.predict(X, return_std=True)
    sigma = np.clip(sigma, 1e-9, None)
    current_best = np.min(self.y)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (current_best - mu) / sigma
        ei = (current_best - mu) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        ei[sigma == 0] = 0.0
    return ei  # Maximize EI

Then, in _select_next_points, generate candidates within tr_bounds, evaluate EI, select top candidates for optimization.

Now, the code for _select_next_points:

def _select_next_points(self, batch_size):
    # Generate candidates within trust region bounds
    bounds = self.tr_bounds
    candidates = self._sample_points_in_bounds(200, bounds)
    acq_values = self._acquisition_function(candidates)
    idxs = np.argsort(-acq_values)  # Maximize EI
    top_candidates = candidates[idxs[:20]]  # Top 20 for refinement

    optimized_points = []
    for x0 in top_candidates:
        def acq_wrapper(x):
            return -self._acquisition_function(x.reshape(1, -1)).item()  # Minimize negative EI
        res = minimize(
            acq_wrapper, 
            x0, 
            bounds=[(b[0][d], b[1][d]) for d, b in enumerate(bounds)],  # Wait, need to get the bounds for each dimension
            method='L-BFGS-B'
        )
        if res.success:
            optimized_points.append(res.x)
    if not optimized_points:
        return candidates[[idxs[0]]].reshape(1, -1)
    optimized_points = np.array(optimized_points)
    acq_values_opt = self._acquisition_function(optimized_points)
    best_idx = np.argmax(acq_values_opt)  # Choose point with highest EI
    return optimized_points[best_idx].reshape(1, -1)

Wait, in the bounds for minimize, each dimension's lower and upper bound is in self.tr_bounds. So for the bounds parameter in minimize, it should be a list of tuples:

bounds = [(tr_bounds[0][d], tr_bounds[1][d]) for d in range(self.dim)]

So in the code, during the minimize step, the bounds are passed correctly.

Now, the function _sample_points_in_bounds. How to generate Sobol samples within arbitrary bounds?

The existing _sample_points uses the original bounds. So I need a new method that generates samples within the current tr_bounds.

But the current _sample_points uses qmc.Sobol, scaled to the given bounds. So perhaps modify the _sample_points method to accept bounds.

In the existing code, _sample_points has a bounds parameter. So in the _select_next_points, pass the current tr_bounds.

Wait, the _sample_points method in the code is defined as:

def _sample_points(self, n_points, bounds=None):

So, in _select_next_points, when generating candidates, call self._sample_points(200, bounds=self.tr_bounds).

But the _sample_points method scales to the given bounds. So this is correct.

Now, the trust region adjustment in __call__:

After evaluating the new point, adjust the trust region.

In the __call__ method:

while self.n_evals < self.budget:
    next_point = self._select_next_points(1)
    new_y = self._evaluate_points(func, next_point)
    self._update_eval_points(next_point, new_y)

    if new_y[0] < best_y:
        best_y = new_y[0]
        best_x = next_point[0]
        expansion_factor = 1.5
    else:
        expansion_factor = 0.9  # Wait, but in code, need to decide whether to expand or contract.

Wait, the adjustment of the trust region depends on whether the new point improved the best. So:

if new_y[0] < best_y:
    # expand the trust region
else:
    # shrink it

But how to implement this. Let's code it.

In the __call__ method, after evaluating the new point:

if new_y[0] < best_y:
    improved = True
else:
    improved = False

Then, adjust the trust region bounds based on improved.

But code-wise, in __call__:

    if new_y[0] < best_y:
        best_y = new_y[0]
        best_x = next_point[0]
        improved = True
    else:
        improved = False

    # Adjust trust region bounds
    if improved:
        expansion_factor = 1.5
        contraction_factor = 1.0  # No contraction
    else:
        expansion_factor = 1.0  # No expansion
        contraction_factor = 0.9

    # Apply expansion or contraction
    for d in range(self.dim):
        center = best_x[d]
        current_lower = self.tr_bounds[0][d]
        current_upper = self.tr_bounds[1][d]
        current_width = current_upper - current_lower

        if improved:
            new_width = expansion_factor * current_width
        else:
            new_width = contraction_factor * current_width

        new_lower = center - new_width / 2
        new_upper = center + new_width / 2

        # Clip to original bounds
        new_lower = max(new_lower, self.bounds[0][d])
        new_upper = min(new_upper, self.bounds[1][d])

        self.tr_bounds[0][d] = new_lower
        self.tr_bounds[1][d] = new_upper

But this code assumes that best_x is updated only when the new point is better. However, even if the new point is not better, best_x remains the current best. So when not improved, the center is still the current best_x.

This makes sense. The trust region is always around the current best_x. So if the new point is not better, we shrink the trust region around the best_x, and if it's better, we expand.

This should help in focusing the search around the best solution, expanding when progress is made, and shrinking when it's not.

Now, in the code, the initial design is 2d points instead of 3d. Let's say n_init = 2 * dim. This saves evaluations for the initial phase, allowing more iterations for the trust region to adapt.

Putting it all together, the new algorithm is called TrustRegionEIBO. The key components are the trust region strategy for adaptive search space control, Expected Improvement acquisition function, and the GP model with Matérn kernel. The trust region adjusts dynamically based on evaluation success, allowing focused exploitation and controlled exploration.

The justification is that this approach adapts the search area to the current best solution, which is efficient for functions like Rosenbrock with elongated valleys. The trust region allows for local refinement and prevents the algorithm from wandering too far when progress stalls, while expanding when improvements are found. This dynamic adjustment, combined with the EI acquisition function, which balances exploration and exploitation, should lead to better convergence.


# Answer
# Description 
**TrustRegionEIBO** introduces a novel Bayesian Optimization algorithm that employs a **Trust Region strategy** with **Expected Improvement (EI)** acquisition and a **Matérn-5/2 kernel GP**. Unlike prior algorithms that use global LCB with static exploration schedules, TrustRegionEIBO dynamically adjusts a hypercube-shaped trust region around the current best solution. This trust region expands when evaluations improve the best value and contracts otherwise, enabling adaptive exploration-exploitation balance. The algorithm generates candidates within the trust region, optimizing EI with L-BFGS-B and Sobol candidates, while the GP model remains globally calibrated. Initial design uses 2d points to save evaluations for trust region refinement.

# Justification 
**1. Trust Region Strategy**  
Unlike previous global LCB approaches, TrustRegionEIBO localizes search to a hypercube around the current best solution. This adapts to the Rosenbrock function's elongated valley by focusing on promising regions while avoiding premature global search. Trust regions dynamically expand/shrink based on evaluation success, balancing exploitation (when progress is made) and exploration (when progress stalls).  

**2. Expected Improvement (EI)**  
EI outperforms LCB on unimodal functions like Rosenbrock by directly modeling the probability of improvement over the current best. This aligns better with gradient-like progress in valleys compared to LCB’s fixed exploration weight.  

**3. Efficient Acquisition Optimization**  
Generating 200 Sobol candidates within the trust region and refining top 20 with L-BFGS-B ensures computational efficiency while maintaining diversity. Trust regions reduce candidate search space, accelerating optimization.  

**4. Anisotropic Matérn-5/2 Kernel**  
Robust to local non-smoothness while maintaining flexibility. GP hyperparameters are optimized with 3 restarts for stability.  

**5. Initialization**  
2d initial Sobol points reduce overhead for budget-limited runs (e.g., 600 evaluations). Trust region expansion compensates for fewer initial points, focusing refinement on the valley.  

**6. Trust Region Management**  
Trust regions expand by 1.5x the current width when improvement is made, and contract by 90% otherwise. This prevents overfitting to noisy evaluations and adapts to the function’s curvature.  

# Code 
```python
from collections.abc import Callable
from scipy.stats import qmc, norm
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class TrustRegionEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.tr_bounds = self.bounds.copy()  # Trust region bounds
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 2 * dim  # Smaller initialization for trust region
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)  # Anisotropic Matérn-5/2
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=1e-8,
            optimizer='fmin_l_bfgs_b',
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        current_best = np.min(self.y)
        z = (current_best - mu) / sigma
        with np.errstate(divide='ignore', invalid='ignore'):
            ei = (current_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma == 0] = 0.0
        return ei  # Maximize EI

    def _select_next_points(self, batch_size):
        bounds = self.tr_bounds
        candidates = self._sample_points(200, bounds=bounds)
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(-acq_values)  # Maximize EI
        top_candidates = candidates[idxs[:20]]  # Top 20 for refinement

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return -self._acquisition_function(x.reshape(1, -1)).item()  # Minimize negative EI
            res = minimize(
                acq_wrapper,
                x0,
                bounds=[(b[0][d], b[1][d]) for d in range(self.dim)],
                method='L-BFGS-B'
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return candidates[[idxs[0]]].reshape(1, -1)
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points)
        best_idx = np.argmax(acq_values_opt)  # Choose point with highest EI
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
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_point = self._select_next_points(1)
            new_y = self._evaluate_points(func, next_point)
            self._update_eval_points(next_point, new_y)

            if new_y[0] < best_y:
                best_y = new_y[0]
                best_x = next_point[0]
                improved = True
            else:
                improved = False

            # Adjust trust region bounds
            for d in range(self.dim):
                center = best_x[d]
                current_lower, current_upper = self.tr_bounds[0][d], self.tr_bounds[1][d]
                current_width = current_upper - current_lower

                if improved:
                    factor = 1.5  # Expand
                else:
                    factor = 0.9  # Contract

                new_width = factor * current_width
                new_lower = center - new_width / 2
                new_upper = center + new_width / 2

                # Clip to original bounds
                new_lower = max(new_lower, self.bounds[0][d])
                new_upper = min(new_upper, self.bounds[1][d])

                self.tr_bounds[0][d] = new_lower
                self.tr_bounds[1][d] = new_upper

            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
```
## Error
 Traceback (most recent call last):
  File "<TrustRegionEIBO>", line 104, in __call__
 104->             next_point = self._select_next_points(1)
  File "<TrustRegionEIBO>", line 64, in _select_next_points
  62 |                 acq_wrapper,
  63 |                 x0,
  64->                 bounds=[(b[0][d], b[1][d]) for d in range(self.dim)],
  65 |                 method='L-BFGS-B'
  66 |             )
NameError: name 'b' is not defined
