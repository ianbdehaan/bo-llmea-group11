# Thinking (reasoning of the model, not part of the answer)
Okay, let's see. I need to refine the AdaptiveMaternLCBGPBO algorithm to improve its performance on the Rosenbrock function. The current score is 0.3412 with a runtime of 562 seconds. The user wants a better AOCC score and maybe reduce runtime. 

First, I'll look at the current components. The algorithm uses an anisotropic Matérn kernel with ν=2.5 (since nu=5/2 is 2.5), which is good for modeling non-smooth functions. The exponential decay of κ in the LCB is set to 2.0 * exp(-3*progress), which rapidly reduces exploration. The initial design is 3d points using Sobol sequences. Acquisition function optimization uses 200 Sobol candidates and top 20 refined with L-BFGS-B. The GP is trained with 3 restarts.

Possible improvements: 

1. **Initial Design**: The current initial design is 3d points. For the Rosenbrock function, which has a long valley, maybe more initial points could help. But the user already tried expanding to 3d. Wait, in the code, n_init is 3*dim. For dim=6, that's 18 points. Maybe increasing it further could help? But the previous version had 3d, which might not be enough. Wait, in the given code, the n_init is 3*dim. But the previous version in the history had "Expanded initialization: 3d initial points provide better coverage". Maybe increasing to 5*dim? Let's check the code again.

In the code: self.n_init = 3 * dim. So for dim=6, that's 18 points. The Rosenbrock function's valley is challenging, so a better initial design might capture the shape. Maybe using more initial points, like 4*dim or 5*dim. But the code allows for it. However, the user's previous attempts had 3d. Let me see in the history: EnhancedAnisotropicLCBBO had 3d initial. Maybe increasing the initial points could help in better coverage.

2. **Acquisition Function Optimization**: The current approach is 200 Sobol candidates, top 20 refined. Maybe increasing the number of candidates or the number of refinement steps could improve finding the optimal next point. Alternatively, using a different sampling method for candidates.

3. **GP Training**: The current model uses 3 restarts for hyperparameters. Maybe increasing to 5 or 4? But that might increase runtime. However, the user mentioned computational efficiency, so perhaps not. Alternatively, maybe using a different optimization method for hyperparameters, but the current setup uses L-BFGS-B, which is standard.

4. **Kappa Schedule**: The current exponential decay is 2.0 * exp(-3*progress). Maybe adjusting the coefficients could balance exploration and exploitation better. For example, starting with a higher exploration weight or decaying slower. Also, the current κ schedule might be too aggressive in reducing exploration. Maybe using a different schedule, like a sigmoid or a different exponential base. Or adding a lower bound to prevent κ from becoming too small.

5. **Batch Selection**: The current code uses a single-point batch selection. Maybe using a multi-point batch could leverage parallel evaluations. However, the evaluator mentions that each candidate is run in parallel (10 runs). Wait, the problem says the algorithm will be run with a budget of 600, and the cost should grow moderately. So perhaps using a small batch size, like 5, to reduce the number of iterations and model fits. But in the current code, it's selecting one point at a time. Maybe changing the batch size to something like 5, but ensuring that the total budget is respected. However, the user's code is for a single point each iteration. So maybe introducing a small batch could help speed up convergence.

6. **Trust Region or Local Search**: The Rosenbrock function's valley is long, so focusing more on local search around the best point might help. The current algorithm uses global search with Sobol candidates. Maybe combining global and local strategies. For example, after some progress, switch to a local search around the best point. Or using a trust region that adapts to the current best point.

7. **Acquisition Function**: The current LCB is mu - κ*sigma. The user has been using LCB with exponential decay. What if using a different acquisition function, like Expected Improvement (EI) with a custom trade-off? However, the previous versions used LCB with good results. But perhaps switching to a combination of LCB and EI could help. Alternatively, using Thompson Sampling, which might be computationally expensive, but maybe a hybrid approach.

8. **Model Choice**: The Matérn kernel is good for non-smooth functions. What if using a combination of kernels, like Matérn plus RBF? Or using a different kernel for different parts of the search. The current setup uses anisotropic Matérn with ν=2.5. Maybe adjusting the nu parameter. For example, using ν=3/2, which is less smooth, or ν=5/2 for more smoothness? Wait, the code uses nu=2.5 (which is 5/2). That's correct for Matérn kernel. So maybe that's appropriate.

9. **Alpha in GP**: The current alpha is 1e-8, which is good for deterministic functions. But maybe using a very small alpha is causing overfitting. However, since the function is deterministic, a very small alpha is acceptable.

10. **Progress Calculation**: The progress ratio is calculated as n_evals / budget. Maybe using a different measure for progress, such as the improvement over time. But that's hard to track without knowing the optimal value.

11. **Refinement in Acquisition Optimization**: The current code refines the top 20 candidates with L-BFGS-B. Maybe increasing the number of starting points or the number of refinement steps. Alternatively, using a different optimizer for the acquisition function, like differential evolution.

12. **Candidate Pooling**: The candidate generation uses Sobol sequences. Maybe using a different low-discrepancy sequence or Latin Hypercube could improve coverage, but Sobol is already good. However, Sobol may not be the best for high-dimensional spaces. But the current code is for dim=6, which is manageable.

13. **Early Stopping**: The algorithm fits the GP model every iteration. Maybe reducing the number of model refits by batching evaluations could save time. For example, using a batch size of 5 and refitting the model after each batch instead of each iteration.

Now, considering the user's previous algorithms and the current code, the main areas for improvement are the initial design, acquisition function optimization, and maybe the kappa schedule. Let's think step by step.

First, the initial design. The current code uses 3*dim points. For the Rosenbrock function, which is tricky due to its long valley, starting with more diverse points could help the GP model better approximate the landscape. Increasing the initial design to 5*dim might help. However, the user's previous version had 3d initial points. Let's check the code again: in the current code, self.n_init = 3*dim. So for dim=6, 18 points. Maybe increasing to 5*dim (30 points) would provide better coverage. However, this would increase the initial cost but may lead to better model accuracy.

Second, the acquisition function optimization. The current approach is 200 candidates, top 20 refined. Maybe increasing the number of candidates to 300 or 400. However, the user mentioned that the cost should grow moderately. So, perhaps 200 is sufficient. Alternatively, using a different method for generating candidates, like Latin Hypercube, but the code uses Sobol. Another idea: instead of refining the top 20, refine more candidates but with a smaller number, like 10 or 15. However, in the current code, the user has top 20. Maybe refining 20 is already good.

Third, the kappa schedule. The current formula is 2.0 * exp(-3 * progress_ratio). Maybe adjusting the coefficient for the exponent. For example, using a lower decay rate (like exp(-2 * progress_ratio)) would maintain exploration longer. Alternatively, starting with a lower kappa. Let's see: the initial value when progress_ratio is 0 is 2.0. Maybe that's too high. For example, using 1.5 or 1.0 as initial kappa. But the exponential decay is already aggressive. Another approach is to use a sigmoid function that reduces exploration more gradually. However, exponential decay is simple and effective.

Fourth, batch selection. The current algorithm uses single-point batches. Switching to a small batch size, say 5, could reduce the number of iterations and model fits. For example, using q-EI (batched acquisition function) or q-UCB. However, implementing batch acquisition functions can be complex. Alternatively, using a simple heuristic to select multiple points in each iteration. For instance, in each iteration, generate 5 candidates and evaluate them. But the user's code is written for single-point batches. Implementing a batched version would require changes to the acquisition function and the selection process. However, given the time constraint (120 seconds per run), batches might speed up the process by reducing the number of iterations.

Fifth, trust region or local search. The current algorithm uses global search with Sobol candidates. Introducing a trust region that focuses around the best point could help in navigating the valley. However, implementing a trust region would require tracking the region and adjusting it based on the model's performance. This could complicate the code but may be beneficial.

Putting these ideas together, possible improvements:

- Increase initial design to 5*dim to better capture the Rosenbrock valley.
- Adjust the kappa schedule to maybe start with a lower exploration coefficient and decay more slowly. For example, 1.5 * exp(-2 * progress_ratio).
- Implement batched acquisition function with a small batch size, such as 5, to reduce the number of model fits.
- Use a hybrid approach for acquisition function optimization, perhaps combining global candidates with local search around the best point.

Now, considering the code structure. The current code has a function _select_next_points which selects one point. Changing it to select multiple points (batch) would require modifying the _select_next_points function to return multiple points, and adjusting the main loop to handle batches. However, the user's evaluator allows for parallel runs, so using batches could be beneficial.

Another point: in the current code, the GP is refit every iteration. If we use batches, we can reduce the number of model refits, which could save time. For example, if the budget is 600 and the batch size is 5, then we have 120 iterations instead of 600. However, the current code has a budget of 600, and the user's code is for a general budget. So, making the batch size dependent on the budget could be a problem. Alternatively, using a fixed batch size, say 5, and adjusting the code to handle it.

Another possible change: in the _fit_model function, using a different kernel. The current code uses Matern with nu=2.5. What if using a Matern kernel with nu=3/2 (1.5), which is less smooth but might adapt better to the valley's structure? Or using a combination of kernels? However, the current setup is working, so changing the kernel might not help. Alternatively, allowing the kernel to be optimized more, but the current code uses n_restarts_optimizer=3.

Another idea: using a different acquisition function. The user has been using LCB with exponential decay. Maybe switching to Expected Improvement (EI) with a custom trade-off between exploration and exploitation. However, the current algorithm uses LCB, which has shown good results. But perhaps combining LCB and EI in a portfolio approach could be beneficial. However, this might complicate the code.

Let me think about the code again. The current _select_next_points function generates 200 Sobol candidates, selects the top 20, refines them with L-BFGS-B, then picks the best. Maybe refining more candidates. For example, generating 300 candidates, selecting top 30, refining them. But this would increase computational cost. However, the user's code must be efficient. The current code uses 200 candidates and 20 refinements. Maybe increasing the number of refinements to 25 or 30. However, the user's code is already using 20, which is a balance between computational cost and quality.

Another idea: the current code uses a fixed number of 200 candidates. Maybe using a dynamic number based on the current progress. For example, in early stages, use more candidates to explore, then reduce the number later. However, this complicates the code.

Now, considering the time constraint. The current runtime is 562 seconds. If we can reduce the number of model refits by using batches, that could save time. For example, using a batch size of 5 would reduce the number of model fits by a factor of 5, but generating and optimizing acquisition function for batches is more complex.

Let me outline the proposed changes:

1. **Increase Initial Design**: Change n_init from 3*dim to 5*dim to better cover the initial search space. This could help the GP model capture the valley's direction earlier.

2. **Adjust Kappa Schedule**: The current kappa starts at 2.0. Maybe using a lower initial value and slower decay. For example, 1.5 * exp(-2 * progress_ratio). This maintains more exploration in the middle phases.

3. **Batched Acquisition Function**: Implement a small batch size, like 5, to reduce the number of iterations and model refits. This requires modifying _select_next_points to return multiple points and adjusting the main loop.

4. **Refinement of More Candidates**: Increase the number of candidates refined in acquisition function optimization. For example, from 20 to 30. However, this increases computational cost. Alternatively, keep it at 20 but use a different refinement method.

But the user's code must be efficient. Let's focus on the batched acquisition function. Implementing a simple batched approach where each iteration selects 5 points. However, the acquisition function for batches (q-UCB or q-EI) is more complex. To keep it simple, perhaps selecting 5 candidates based on the top of the acquisition function and refining them. But the current code selects 1 point. 

Alternatively, using a multi-point acquisition function that considers the diversity of the batch. For example, using a greedy approach where each point is selected to be as diverse as possible from the previous ones. However, implementing this would require more code.

Given the time constraints, perhaps the most straightforward improvement is to increase the initial design and adjust the kappa schedule. Let's proceed with that.

So, in the code:

- Change self.n_init = 3 * dim to 5 * dim.

- Modify the _compute_kappa function to use a different decay, perhaps 1.5 * exp(-2 * progress_ratio).

Additionally, in the acquisition function optimization, maybe increase the number of candidates. For example, from 200 to 300. But the user mentioned computational efficiency, so maybe 200 is acceptable.

Another possible tweak: in the current code, the GP is refit every iteration. If we use a batch size of 5, then refitting occurs every 5 iterations, which could save time. However, this would require modifying the loop to handle batches.

Let me outline the code changes step by step.

First, in the __init__ method:

self.n_init = 5 * dim

Then, in the _compute_kappa function:

return 1.5 * np.exp(-2 * progress_ratio)

Then, in the _select_next_points function, perhaps increase the number of candidates. For example, from 200 to 300. Also, maybe increase the number of refined candidates from 20 to 30. However, this may increase computational cost. Alternatively, keep 200 candidates and 20 refinements, but with the updated kappa schedule.

Wait, but the current code has the _select_next_points function returning a single point. To implement a batch, the function would need to return multiple points. For example, changing the batch_size parameter in _select_next_points from 1 to 5. Then, in the main loop, the algorithm evaluates 5 points each iteration.

But the user's code structure currently evaluates one point per iteration. So modifying this would require changes in the __call__ function's loop.

For example, in the __call__ function:

while self.n_evals < self.budget:
    next_points = self._select_next_points(5)
    new_y = self._evaluate_points(func, next_points)
    self._update_eval_points(next_points, new_y)
    self.model = self._fit_model(self.X, self.y)

This would reduce the number of iterations from budget to budget / batch_size.

However, the current _select_next_points function is designed to return a single point. So, the _select_next_points function needs to be modified to return multiple points. For example, generating multiple candidates, selecting the top N, and refining them.

But this would require a significant change to the _select_next_points function. For simplicity, perhaps using a greedy approach to select diverse points in each batch. For example, selecting the top N points based on the acquisition function.

However, with a batch size of 5, the code would need to return 5 points. The current code is returning 1. So, modifying the _select_next_points function to generate 200 candidates, select the top 20, refine them, and pick the top 5.

But in the current code, the _select_next_points function returns the best point. To handle batches, perhaps the function would return the top 5 points after refinement.

This would require changing the return statement from optimized_points[best_idx].reshape(1, -1) to returning the top 5 points.

But this is a major change. Given the time constraints, perhaps it's better to focus on initial design and kappa schedule for now.

Another idea: in the _select_next_points function, after refining the top 20 candidates, instead of picking the single best, select the top 5. But the code would then need to handle batches.

But the user's evaluator may not allow for parallel evaluations unless it's implemented as batches. However, the problem statement says "each of the optimization functions has a search space... Each candidate is run 10 times in parallel..." So the algorithm can return multiple points each iteration, and the evaluator will run them in parallel. Therefore, modifying the algorithm to select batches could be beneficial.

Given that, I'll proceed to modify the code to use batches. Let's outline the steps:

1. Modify the _select_next_points function to return a batch of 5 points.

2. In the __call__ function's loop, process these 5 points each iteration.

3. Adjust the _evaluate_points function to handle multiple points, ensuring the budget is not exceeded.

4. Update the model refitting to occur after each batch.

This would reduce the number of model refits and potentially speed up the algorithm.

Now, implementing this in the code:

In the _select_next_points function, instead of returning one point, return the top 5 points. The current code picks the top 20 refined candidates and selects the best among them. For batches, we need to return 5 points.

In the _select_next_points function:

if not optimized_points:
    return candidates[[idxs[0]]].reshape(1, -1)
optimized_points = np.array(optimized_points)
acq_values_opt = self._acquisition_function(optimized_points)
best_idx = np.argmin(acq_values_opt)
return optimized_points[best_idx].reshape(1, -1)

Change this to select the top 5 points. So:

sorted_indices = np.argsort(acq_values_opt)
top5_indices = sorted_indices[:5]
top5_points = optimized_points[top5_indices]
return top5_points

But the code must ensure that the number of points doesn't exceed the budget. So in the __call__ function's loop, when evaluating the next points, we need to check the remaining budget and select the appropriate number.

Modifying the _select_next_points function to return multiple points. Let's adjust the code:

def _select_next_points(self, batch_size):
    candidates = self._sample_points(200)  # Increased candidate pool
    acq_values = self._acquisition_function(candidates)
    idxs = np.argsort(acq_values)  # Minimize LCB
    top_candidates = candidates[idxs[:20]]  # Top 20 candidates for refinement

    optimized_points = []
    for x0 in top_candidates:
        def acq_wrapper(x):
            return self._acquisition_function(x.reshape(1, -1)).item()
        bounds = [(l, u) for l, u in zip(self.bounds[0], self.bounds[1])]
        res = minimize(
            acq_wrapper, 
            x0, 
            bounds=bounds,
            method='L-BFGS-B'
        )
        if res.success:
            optimized_points.append(res.x)
    if not optimized_points:
        # Return top batch_size candidates
        return candidates[idxs[:batch_size]]
    optimized_points = np.array(optimized_points)
    acq_values_opt = self._acquisition_function(optimized_points)
    best_indices = np.argsort(acq_values_opt)[:batch_size]
    return optimized_points[best_indices]

Then, in the __call__ function:

    while self.n_evals < self.budget:
        batch_size = 5
        next_points = self._select_next_points(batch_size)
        new_y = self._evaluate_points(func, next_points)
        self._update_eval_points(next_points, new_y)

        # Update best solution
        for i in range(len(new_y)):
            if new_y[i] < best_y:
                best_y = new_y[i]
                best_x = next_points[i]

        self.model = self._fit_model(self.X, self.y)

This way, each iteration selects 5 points, reduces the number of model refits from 600 to 120, which could save time.

Another adjustment: in the _evaluate_points function, ensure that the number of points doesn't exceed the budget.

def _evaluate_points(self, func, X):
    n_new = len(X)
    remaining_budget = self.budget - self.n_evals
    if n_new > remaining_budget:
        n_new = remaining_budget
        X = X[:n_new]
    y = np.zeros((n_new, 1))
    for i in range(n_new):
        y[i, 0] = func(X[i])
    self.n_evals += n_new
    return y

This handles cases where the batch size would exceed the remaining budget.

Additionally, in the __call__ function, when updating the best solution, loop through each new point and check if it's better than the current best.

With these changes, the algorithm would use a batch size of 5, reducing the number of iterations and model fits. This should improve efficiency.

Now, the code changes would be significant. But this is a possible refinement. Let's summarize the changes:

- Increase initial design to 5*dim.
- Adjust kappa schedule to 1.5 * exp(-2 * progress_ratio).
- Use batch acquisition with batch_size=5.
- Modify _select_next_points to return batch_size points.
- Adjust the main loop to handle batches.

These changes aim to improve efficiency and performance by reducing the number of model refits and better exploration in the initial phase.


# Answer
# Description 
**TrustRegionAdaptiveMaternLCBGPBO** is a refined Bayesian Optimization algorithm that introduces an adaptive trust region around the best point to focus search in the elongated valley of the Rosenbrock function. Key improvements include:  
- **Anisotropic Matérn Kernel (ν=5/2)** for robust modeling of local smoothness and curvature.  
- **Dynamic Trust Region (TuRBO)** that grows/shrinks based on progress to focus exploitation near the minimum while maintaining global exploration.  
- **Exponential κ-Schedule** (1.5·e^(-2·progress)) for slower decay of exploration to balance convergence in the valley.  
- **Hybrid Candidate Pooling** with 250 Sobol candidates and 25 refinement points to improve acquisition optimization.  
- **Expanded Initialization (5d points)** for robust coverage of the search space.  
- **Batched Optimization (batch_size=5)** to reduce the number of model refits and improve runtime efficiency.  

# Justification 
The Rosenbrock function's long valley requires adaptive exploration-exploitation strategies. The trust region mechanism (TuRBO-inspired) ensures the algorithm stays near the minimum while allowing expansion when progress stalls. The slower exponential decay in κ (1.5·e^(-2·progress)) prevents premature over-exploitation. Batched acquisition reduces the number of expensive GP refits (from 600 to 120) while maintaining diversity in candidate selection. Expanding the initial design (5d) improves early coverage of the elongated valley. The hybrid candidate pooling increases the chance of finding optimal candidates while keeping computational cost moderate.  

# Code 
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.optimize import minimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class TrustRegionAdaptiveMaternLCBGPBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = 5 * dim  # Expanded initialization for better coverage
        self.batch_size = 5
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.trust_radius = 2.0  # Initial trust region radius

    def _sample_points(self, n_points, bounds=None):
        if bounds is None:
            bounds = self.bounds
        sampler = qmc.Sobol(d=self.dim, scramble=True, seed=self.rng)
        samples = sampler.random(n=n_points)
        scaled_samples = qmc.scale(samples, bounds[0], bounds[1])
        return scaled_samples

    def _fit_model(self, X, y):
        kernel = Matern(length_scale=np.ones(self.dim), nu=2.5)  # Anisotropic Matérn kernel with ν=5/2
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,  # More robust hyperparameter optimization
            alpha=1e-8,  # Reduced noise assumption for deterministic function
            optimizer='fmin_l_bfgs_b',
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        mu, sigma = self.model.predict(X, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        current_kappa = self._compute_kappa()
        return mu - current_kappa * sigma  # Minimize LCB

    def _compute_kappa(self):
        progress_ratio = self.n_evals / self.budget
        return 1.5 * np.exp(-2 * progress_ratio)  # Exponential decay of exploration weight

    def _select_next_points(self, batch_size):
        # Expand bounds based on trust region
        best_x = self.X[np.argmin(self.y)]
        trust_bounds = np.array([best_x - self.trust_radius, best_x + self.trust_radius])
        trust_bounds = np.clip(trust_bounds, self.bounds[0], self.bounds[1])  # Stay within global bounds

        candidates = self._sample_points(250, bounds=trust_bounds)  # Increased candidate pool
        acq_values = self._acquisition_function(candidates)
        idxs = np.argsort(acq_values)  # Minimize LCB
        top_candidates = candidates[idxs[:25]]  # Top 25 candidates for refinement

        optimized_points = []
        for x0 in top_candidates:
            def acq_wrapper(x):
                return self._acquisition_function(x.reshape(1, -1)).item()
            bounds = [(l, u) for l, u in zip(trust_bounds[0], trust_bounds[1])]
            res = minimize(
                acq_wrapper, 
                x0, 
                bounds=bounds,
                method='L-BFGS-B'
            )
            if res.success:
                optimized_points.append(res.x)
        if not optimized_points:
            return candidates[idxs[:batch_size]]
        optimized_points = np.array(optimized_points)
        acq_values_opt = self._acquisition_function(optimized_points)
        best_indices = np.argsort(acq_values_opt)[:batch_size]
        return optimized_points[best_indices]

    def _update_trust_region(self, new_X, new_y):
        best_x = self.X[np.argmin(self.y)]
        best_new_x = new_X[np.argmin(new_y)]
        # Adjust trust region based on progress
        if new_y.min() < self.y.min():
            # Successful step: expand trust region
            self.trust_radius = min(self.trust_radius * 1.5, 2.0)
        else:
            # No progress: shrink trust region
            self.trust_radius = max(self.trust_radius * 0.5, 0.5)

    def _evaluate_points(self, func, X):
        n_new = len(X)
        remaining_budget = self.budget - self.n_evals
        if n_new > remaining_budget:
            n_new = remaining_budget
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
        # Initial Sobol design for diverse sampling
        self.X = self._sample_points(self.n_init)
        self.y = self._evaluate_points(func, self.X)
        best_idx = np.argmin(self.y)
        best_x = self.X[best_idx]
        best_y = self.y[best_idx]

        # Initialize GP model
        self.model = self._fit_model(self.X, self.y)

        while self.n_evals < self.budget:
            next_points = self._select_next_points(self.batch_size)
            new_y = self._evaluate_points(func, next_points)
            self._update_eval_points(next_points, new_y)

            # Update best solution
            for i in range(len(new_y)):
                if new_y[i] < best_y:
                    best_y = new_y[i]
                    best_x = next_points[i]

            # Update trust region
            self._update_trust_region(next_points, new_y)

            # Refit GP model
            self.model = self._fit_model(self.X, self.y)

        return best_y, best_x
```
## Feedback
 The algorithm TrustRegionAdaptiveMaternLCBGPBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.3037 with standard deviation 0.0137.

took 82.23 seconds to run.