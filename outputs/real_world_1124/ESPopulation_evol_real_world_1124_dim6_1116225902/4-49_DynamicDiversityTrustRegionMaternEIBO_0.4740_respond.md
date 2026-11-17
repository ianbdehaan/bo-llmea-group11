# Thinking (reasoning of the model, not part of the answer)
Okay, I need to refine the RefinedTrustRegionSobolMaternEIBO algorithm to improve its performance. Let's look at what's already in the code and think about possible improvements.

First, the current algorithm uses a trust region approach with Sobol sampling and Expected Improvement (EI) as the acquisition function. The trust region adjusts dynamically based on progress. The initial design uses Sobol points, which is good for space coverage. The model is a Gaussian Process with a Matern kernel and reduced restarts for speed.

Looking at possible areas to refine:

1. **Initial Design**: The initial Sobol points are set to max(5, 2*dim). For higher dimensions, that's enough, but maybe increasing the initial points could lead to a better model start. However, budget is limited, so maybe not. Wait, the current n_init is max(5, 2*dim). For dim=6, that's 12 points. Maybe increasing to 2*dim+1 or 3*dim could help? Let's check the code.

2. **Candidate Generation**: The algorithm generates 100 Sobol candidates per iteration. Maybe varying the number based on the trust region's size or the current state could help. Also, using batch selection with some diversity could be improved. For example, ensuring the selected candidates are not too close to each other.

3. **Trust Region Dynamics**: The current adjustment is 1.1 and 0.8 for expansion and contraction. Maybe making these factors adaptive based on the improvement amount could help. Also, the trust region's center is updated to the best point found, but perhaps incorporating the uncertainty from the model to adjust the center could prevent premature convergence.

4. **Model Refitting**: The model is refitted every 2 iterations. Maybe adjusting this based on the improvement or the model's confidence could be better. For instance, if the model's predictions are uncertain (high variance), more frequent refitting might help.

5. **Acquisition Function Optimization**: The current EI is calculated, and top 2 candidates are selected. Maybe using a more sophisticated method for batch selection, like Expected Improvement Plus (EIP) or q-EI, which considers the correlation between points. But q-EI could be computationally heavy. Alternatively, enforcing diversity among selected candidates.

6. **Handling Plateaus**: If the best point doesn't improve for several iterations, maybe the trust region should expand more aggressively to explore other areas.

7. **Kernel Parameters**: The current Matern kernel uses nu=2.5 (Matern 5/2), which is good for smooth functions. Maybe allowing the model to switch between different kernels based on the problem's characteristics? But that might complicate things. Alternatively, using a hybrid kernel, but again, not sure.

8. **Scaling the Bounds**: The trust region's bounds are clamped to the domain, but maybe using a different scaling approach could help.

9. **Exploiting Parallelism**: Since the algorithm selects 2 points per iteration, ensuring that those points are diverse and not too close could help exploration.

10. **Early Stopping**: If the model isn't improving over several iterations, perhaps resetting the trust region or expanding it to explore more.

Let me think of specific changes.

First, the initial design. The current code uses max(5, 2*dim) points. For dim=6, 12 points. Maybe increasing this to 2*dim+1=13, but perhaps 2*dim+1 is already considered. The initial design could be more diverse. Alternatively, using a maximin distance approach for the initial points to ensure they are spread out. But with Sobol sequences, the points are already well-distributed, so maybe not necessary. Alternatively, using Latin hypercube for the initial design? But the code already uses Sobol.

Next, candidate selection. The current approach is to generate 100 candidates in the trust region and select the top 2. To ensure diversity, perhaps selecting the top 2 but with a distance constraint between them. For example, after selecting the top candidate, the next one is the best that's at least a certain distance away. Alternatively, using a clustering method to pick diverse points. However, adding this might increase computational cost. Since each candidate selection is already using Sobol points, which are designed to be diverse, maybe the current approach is sufficient, but adding some diversity check could help.

Another point: the trust region's center is always the current best point. But if the best point's prediction is uncertain (high variance), maybe the center should be adjusted. For example, using the mean of the best points or a weighted average based on their uncertainty. Alternatively, moving the center towards areas with lower variance if the current best is uncertain. But that might complicate the algorithm.

The model refitting every 2 iterations is a cost-saving measure. However, if the model's predictions become unreliable (like in areas of high uncertainty), more frequent refitting could help. Maybe checking the variance of the best point and refitting more often when the variance is high.

Also, the trust region's expansion and contraction factors. The current 1.1 and 0.8 are smooth, but maybe making them adaptive. For example, if the improvement is significant, expand more, otherwise, just a small expansion. Or if the improvement is marginal, contract a bit more. Alternatively, using logarithmic scaling.

Another idea: when the trust region radius is very small, the algorithm might get stuck in a local minimum. To prevent this, after a certain number of iterations without improvement, expand the trust region more aggressively. For example, if the radius hasn't changed for N iterations, expand it to a larger factor.

In the code, the acquisition function is Expected Improvement. Maybe using Lower Confidence Bound (LCB) with a dynamic exploration parameter could balance exploration and exploitation better. However, the user's examples included EI-based algorithms, so maybe sticking with EI is better. Alternatively, combining EI with a diversity term.

Also, the initial radius is 1.0, but maybe starting with a radius that's a function of the domain size. For example, initial radius is (upper - lower) * 0.1, which for [-5,5] is 10, so 0.1*10=1.0, which is the current setting. So that's correct.

Another aspect: the model uses the Matern 5/2 kernel with ARD. The length scales are initialized to 1, but maybe using a heuristic based on the problem's scale could help. However, the model's optimization process will find the optimal length scales, so initializing them to 1 might be sufficient.

For the code, let's think of possible modifications.

Change 1: Adjust the number of candidates per iteration. The current code uses 100 candidates. Maybe increasing this to 200 to allow more exploration, but with a higher computational cost. However, the algorithm must be efficient. Alternatively, using a smaller number during the initial phases and increasing later. But that might complicate the code.

Change 2: Instead of always selecting top 2 candidates, enforce a minimum distance between them. For example, after selecting the best candidate, the next one should be at least a certain distance away. This would encourage diversity in the batch. The minimum distance can be based on the trust region radius or a fixed value. This could prevent the algorithm from evaluating very similar points in parallel, which might slow down convergence but improve exploration.

Change 3: Modify the trust region update logic. Currently, if there's an improvement, the radius expands by 10%, else it contracts by 20%. Maybe adjust the factors to be more aggressive. For example, expand by 1.2 if there's a significant improvement, but only 1.05 otherwise. Or contract by 0.5 if there's no improvement for several iterations. However, this requires defining what constitutes a significant improvement, which could be problem-dependent.

Change 4: Incorporate the uncertainty from the model into the trust region update. For example, if the best point has high variance, the trust region might need to expand more to explore. But how to quantify that? The current code doesn't track uncertainty in the best point's prediction. Alternatively, when the model's variance is high, adjust the trust region more aggressively.

Change 5: The model refitting frequency. The current code refits every 2 iterations. Maybe refitting when the trust region changes (expands/contracts) and after a certain number of evaluations. Alternatively, checking if the model's fit has degraded (e.g., based on some metric like the model's predictive variance) and refitting more often when needed.

Change 6: Use a different kernel for the Gaussian Process. The current code uses Matern 5/2. Maybe switching to a Matern 3/2 kernel could offer better performance for functions with less smoothness. However, this is problem-dependent, and the FM wave parameter estimation might benefit from a specific kernel. The user's prior examples used Matern, so perhaps that's the best choice.

Change 7: The acquisition function optimization could use a more sophisticated method. For example, instead of evaluating 100 candidates and picking top 2, using a local optimization with bounds (like L-BFGS-B) starting from several random points. However, this would increase computational cost but could lead to better acquisition function maxima.

But given the time constraints (algorithm must run under 120 seconds), adding more computational steps (like multiple L-BFGS-B runs) might not be feasible. The current approach uses Sobol candidates, which is efficient.

Another point: the current code uses a fixed batch size of 2. Maybe adjusting the batch size based on the trust region's size. For example, when the trust region is large, using a larger batch, and when it's small, using a smaller batch. This could balance exploration and exploitation.

But the problem states that each candidate is run 10 times in parallel, so the batch size must be 10? Wait, the problem says "each of the optimization functions has a search space...", but the user mentions "each candidate is run 10 times in parallel". Wait, no, the code is for a single algorithm. The note says: "Each candidate is run 10 times in parallel, one CPU core per run..." So the algorithm's batch size should be 10, but in the current code, the batch size is 2. Wait, the code's _select_next_points returns top 2 candidates. But the evaluator runs each candidate 10 times in parallel? Wait, that's confusing.

Wait, the problem says: "each candidate is run 10 times in parallel, one CPU core per run, and is stopped and counted as failed if the runs take longer than 120 seconds." So the algorithm's __call__ function must submit 10 candidate points to be evaluated in parallel. However, the current code's _select_next_points function returns 2 points, and each iteration processes those 2 points. But if the evaluator is expecting 10 in parallel, that would be a problem.

Wait, looking at the problem description again:

"Each candidate is run 10 times in parallel, one CPU core per run, and is stopped and counted as failed if the runs take longer than 120 seconds. Keep the cost per iteration small (e.g. few optimizer restarts for the GP hyperparameters and the acquisition function). The algorithm will later also be run with a budget of 600 evaluations, so it must work for any budget and its cost should grow moderately with the number of evaluated points."

Wait, the wording is confusing. The "each candidate is run 10 times in parallel" likely refers to the algorithm being evaluated in a setup where each algorithm's run is executed 10 times in parallel on different CPU cores. But the algorithm itself must submit function evaluations in batches. For example, in each iteration, the algorithm may evaluate 2 points (batch size 2), and since each of the 10 parallel runs does that, the total number of evaluations would be 10 * number of iterations * batch size. However, the problem statement says that the budget is the maximum number of evaluations allowed. So the code must manage its own batch size, ensuring that the total evaluations (n_evals) do not exceed the budget.

In the current code, the _select_next_points returns 2 points. So each iteration adds 2 evaluations. The initial design uses 2*dim or 5 points, then each iteration adds 2. For example, with budget=600, the number of iterations would be (600 - n_init) / 2. So the code's current batch size is 2, which is acceptable if the evaluator allows it. However, if the problem requires the algorithm to evaluate 10 points in parallel (i.e., batch size 10), then the code's current batch size of 2 is incorrect. But the problem's initial code for RefinedTrustRegionSobolMaternEIBO uses a batch size of 2. So perhaps the batch size is set to 2, and the 10 runs are the multiple independent runs of the algorithm by the evaluator. For example, the evaluator runs 10 separate instances of the algorithm, each with their own budget. Each instance of the algorithm evaluates 2 points per iteration. So the code's batch size is correct.

Therefore, the current code's batch size is 2, and that's acceptable.

Back to refining the algorithm. Let's think about the trust region's center and radius. Another idea: instead of always centering the trust region on the best point, perhaps use a weighted average of the best points, or the mean of the best points. This could help avoid getting stuck in a narrow region.

Another possible refinement is to adjust the trust region's radius based on the improvement's magnitude. For example, if the improvement is large, expand more, if it's small, contract less. However, quantifying what's a large improvement is problem-dependent.

Additionally, when the trust region's radius is small, the algorithm might not explore enough. To prevent this, periodically (after a certain number of iterations) reset the trust region to a larger radius if no improvement has been made. This could be implemented by tracking the number of consecutive iterations without improvement and expanding the radius if that count is too high.

In the current code, the trust region is updated every iteration. If there's no improvement, it contracts. Maybe adding a counter for iterations without improvement and expanding the radius after a threshold.

Let's outline the proposed changes:

1. **Increase Candidate Diversity in Batch Selection**: After selecting the best candidate, enforce a minimum distance between the two selected candidates to ensure diversity.

2. **Dynamic Model Refitting Based on Variance**: Check the variance of the best point. If the variance is high, refit the model more frequently to capture the uncertainty better.

3. **Trust Region Reset on Plateaus**: If the algorithm hasn't improved for a certain number of iterations, reset the trust region to a larger radius to encourage exploration.

4. **Adaptive Expansion/Contraction Factors**: Adjust the expansion and contraction factors based on the improvement's magnitude.

Implementing these changes step by step.

First, for candidate diversity: after selecting the top candidate, the second candidate must be at least a certain distance away. Let's say, for example, that the second candidate must be at least 0.25 distance units away from the first. The distance can be the Euclidean distance. This can be implemented by checking the distance between the top candidates and selecting the next one that's sufficiently far away.

Second, dynamic model refitting: after each model fit, check the variance at the best point. If the variance is above a threshold (e.g., 0.1), refit the model more frequently. However, the current code already fits the model every 2 iterations, which is sufficient. Maybe not necessary.

Third, trust region reset on plateaus: track the number of consecutive iterations without improvement. If it exceeds a threshold (say, 5), expand the trust region to a larger radius.

Fourth, adaptive expansion/contraction: when there's an improvement, the expansion factor can be larger. For example, if the improvement is significant (e.g., more than 5% of the current best), expand by 1.2; otherwise, 1.1. Similarly, if there's no improvement for multiple iterations, contract by a larger factor.

Let's proceed with implementing these changes.

First, modifying the trust region update to include a plateau detection mechanism. Add a counter for iterations without improvement. If this counter reaches a threshold (e.g., 5), expand the trust region to the maximum radius.

Second, modifying the candidate selection to ensure diversity between the two selected points.

Third, adjusting the expansion/contraction factors based on the improvement amount.

Let's adjust the code accordingly.

In the code:

- Track the number of consecutive iterations without improvement. Let's add a variable self.no_improvement_count = 0. Initialize it in __init__.

- In the _update_eval_points method, after updating the best_idx, check if there's an improvement. If yes, reset the no_improvement_count. If not, increment it.

- In the trust region update logic, if self.no_improvement_count exceeds a threshold (e.g., 5), expand the trust region to the maximum radius.

For candidate diversity:

In the _select_next_points method, after selecting the top 2 candidates, check the Euclidean distance between them. If it's below a threshold (e.g., 0.2), then replace the second candidate with the next best one that is sufficiently distant.

This may require a loop to find the best second candidate.

Another approach: generate all 100 candidates, compute the acquisition function, then select the first candidate (best), then for the second candidate, select the best among the remaining candidates that are at least a certain distance away from the first.

This can be implemented as follows:

- Find the first candidate (best EI).

- Find the second candidate from the remaining candidates with maximum EI, but at least a certain distance away from the first.

The distance threshold can be a function of the trust region's radius. For example, 0.1 * tr_radius.

In code:

After selecting the top_indices, check the distance between the two selected candidates. If they are too close, find the next best candidate that is sufficiently far.

Alternatively, when selecting the top 2, replace the second candidate if they are too close.

But implementing this may require more code. Let's think of the current code's _select_next_points function:

candidates = ... 
acq_vals = ...
top_indices = np.argsort(acq_vals)[-2:][::-1]
next_X = candidates[top_indices]

So, the first and second best candidates.

We can check the distance between them. If it's less than a threshold, then for the second candidate, select the best among the remaining candidates that are sufficiently far from the first.

The code can be modified as follows:

Compute the indices sorted by EI.

sorted_indices = np.argsort(acq_vals, axis=0)[::-1]  # descending order

first_candidate_idx = sorted_indices[0]
first_candidate = candidates[first_candidate_idx]

remaining_indices = sorted_indices[1:]

# Find the best candidate among remaining_indices that is at least distance min_dist away from first_candidate
min_dist = 0.2  # arbitrary value, can be adjusted
best_second_idx = None
best_second_ei = -np.inf
for idx in remaining_indices:
    candidate = candidates[idx]
    dist = np.linalg.norm(candidate - first_candidate)
    if dist >= min_dist:
        ei_val = acq_vals[idx]
        if ei_val > best_second_ei:
            best_second_ei = ei_val
            best_second_idx = idx

if best_second_idx is not None:
    second_candidate_idx = best_second_idx
else:
    # No candidate found, just take the next best, even if close
    second_candidate_idx = sorted_indices[1]

Then, next_X is [candidates[first_candidate_idx], candidates[second_candidate_idx]]

This adds some computational overhead but could improve diversity.

Now, the code needs to implement this.

Another aspect: the min_dist threshold. Maybe set it to a fraction of the trust region radius, like 0.1 * tr_radius.

Additionally, in the trust region update, adding the plateau detection.

Now, in the code:

In __init__:

self.no_improvement_count = 0
self.no_improvement_threshold = 5  # adjust as needed

In the loop during evaluation:

after evaluating new points, check if the new best is better than the previous best. If not, increment self.no_improvement_count. Else, reset it to 0.

In the code:

current_best_idx = np.argmin(self.y)
new_best = next_y[0] < self.y[best_idx]

Wait, in the current code, new_best is set based on whether the first of the new points (next_y[0]) is better than the previous best. But in reality, after evaluating new points, the new best could be among any of the new points. For example, if the new points are added, the current best could be the best among all evaluations, not just the previous best.

So the current logic is not entirely correct. The code should check whether the new evaluation includes a better point than the previous best.

So, the current code has:

new_best = next_y[0] < self.y[best_idx]

But if the new points include a better point (either next_y[0] or next_y[1]), then new_best should be True.

Thus, the code should compare the new best of next_y to the previous best.

Let me adjust that.

In the current code:

new_best = next_y[0] < self.y[best_idx]

But after evaluating next_y, the new best could be the minimum of next_y.

Wait, the code evaluates next_X (2 points) and gets next_y (2 values). Then, the new best is whether the new evaluations include a better point than the previous best.

So, current_best_value = self.y[best_idx]

new_best = np.min(next_y) < current_best_value

But this is not correct because the new best could be any of the previous evaluations or the new ones. However, in the code, best_idx is updated to current_best_idx after the new points are added.

Wait, in the code:

After evaluating new points and updating the self.X and self.y, current_best_idx is set to the index of the current best (np.argmin(self.y)), which includes the new points. So the new_best variable should indicate whether the new points include a better point than the previous best.

So, the current best before adding new points is self.y[best_idx]. The new points have next_y. So new_best is whether np.min(next_y) < self.y[best_idx].

But in the current code, new_best is set as next_y[0] < self.y[best_idx]. This is incorrect because the second point may be better.

So the code should compare the new points' minimum to the previous best.

So, in the code:

new_best = np.min(next_y) < self.y[best_idx]

That's a correction needed. The current code checks only the first new evaluation point, which is not correct. This mistake could lead to incorrect trust region updates.

Fixing that would improve the algorithm's performance.

Another correction: the current code uses next_y[0] to determine if there's a new best. This is a bug. The new_best should be whether any of the new points is better than the previous best.

So, in the code:

new_best = np.min(next_y) < self.y[best_idx]

This is a necessary correction.

Now, with these considerations, let's outline the code changes:

1. Correct the new_best logic to compare the minimum of the new points to the previous best.

2. Add plateau detection based on self.no_improvement_count.

3. Enforce diversity between the two selected candidates in each batch.

4. Adjust the expansion/contraction factors based on improvement magnitude.

Let's proceed to code these changes.

First, in the _select_next_points method, modify the candidate selection to ensure diversity.

Second, track the no_improvement_count and use it to expand the trust region.

Third, correct the new_best calculation.

Fourth, adjust expansion/contraction factors.

Let's start coding these changes.

In the _select_next_points method:

def _select_next_points(self):
    ... 
    candidates = ...
    acq_vals = ...
    sorted_indices = np.argsort(acq_vals, axis=0)[::-1]  # descending order
    if len(sorted_indices) < 2:
        return candidates[sorted_indices]  # handle if only 1 candidate
    first_idx = sorted_indices[0]
    first_candidate = candidates[first_idx]
    remaining_indices = sorted_indices[1:]
    min_dist = 0.1 * self.tr_radius  # 10% of the current radius
    best_second_idx = None
    best_second_ei = -np.inf
    for idx in remaining_indices:
        dist = np.linalg.norm(candidates[idx] - first_candidate)
        if dist >= min_dist:
            ei_val = acq_vals[idx]
            if ei_val > best_second_ei:
                best_second_ei = ei_val
                best_second_idx = idx
    if best_second_idx is not None:
        second_idx = best_second_idx
    else:
        # No candidate found with sufficient distance; take the next best
        second_idx = sorted_indices[1]
    next_X = np.array([candidates[first_idx], candidates[second_idx]])
    return next_X

This adds diversity between the two selected candidates.

In the trust region update logic, after evaluating new points, check if the new best is better, and update the no_improvement_count:

In the main loop:

    next_y = self._evaluate_points(func, next_X)
    self._update_eval_points(next_X, next_y)
    
    # Update trust region
    current_best_idx = np.argmin(self.y)
    # Check if any of the new points improved the best
    previous_best_value = self.y[best_idx]
    new_best_value = self.y[current_best_idx]
    new_best = new_best_value < previous_best_value
    
    if new_best:
        # Reset plateau counter
        self.no_improvement_count = 0
        # Expand trust region
        expansion_factor = 1.1
        if (new_best_value - previous_best_value) / abs(previous_best_value + 1e-8) > 0.05:
            expansion_factor = 1.3  # larger expansion for significant improvement
        self.tr_radius = min(self.tr_radius * expansion_factor, self.max_tr_radius)
    else:
        # Increment plateau counter
        self.no_improvement_count += 1
        # Contract trust region
        contraction_factor = 0.8
        if self.no_improvement_count >= self.no_improvement_threshold:
            # Reset to maximum radius if no improvement
            self.tr_radius = self.max_tr_radius
        else:
            self.tr_radius = max(self.tr_radius * contraction_factor, self.min_tr_radius)
    
    # Update best solution and trust region center
    best_idx = current_best_idx
    self.tr_center = self.X[best_idx].copy()
    ...

This adds a plateau counter and adjusts the expansion/contraction factors based on improvement magnitude.

Now, adding the no_improvement_count and no_improvement_threshold to the __init__:

In __init__:

        self.no_improvement_count = 0
        self.no_improvement_threshold = 5  # adjust as needed

Also, in the code, after the initial model fitting, we need to initialize self.best_value, or track the previous best for comparison.

Wait, in the code, after the initial evaluation, best_idx is set to the best of the initial points. Then, during the loop, previous_best_value is tracked as self.y[best_idx], and new_best_value is the current best.

Thus, the code compares the new best to the previous best.

With these changes, the algorithm should:

- Ensure diversity in the selected candidates.
- Adjust trust region expansion/contraction based on improvement.
- Reset the trust region if no improvement for a threshold number of iterations.
- Correctly determine if new points improved the best.

These changes should lead to better exploration and exploitation, avoiding local minima and improving convergence.

Another possible change is to adjust the number of candidates generated per iteration. For example, when the trust region is large, generate more candidates, and when small, fewer. However, this would complicate the code. Given the time constraints, perhaps sticking to 100 candidates is sufficient.

Now, let's code these changes into the original code.


# Answer
# Description 
This refined Bayesian Optimization algorithm, **DynamicDiversityTrustRegionMaternEIBO**, enhances RefinedTrustRegionSobolMaternEIBO with three key improvements: 1) Enforces diversity in batch selection by ensuring selected candidates are sufficiently distant, 2) Implements adaptive trust region expansion/contraction based on improvement magnitude, 3) Introduces plateau detection to reset trust regions when no progress is made. The algorithm uses Sobol sampling for candidate generation, Expected Improvement acquisition function, and dynamically adjusts trust region size (initial radius 1.0) based on evaluation progress. Diversity constraints in batch selection and plateau-aware trust region updates improve exploration-exploitation balance and prevent premature convergence.

# Justification 
The algorithm's key improvements address critical limitations of its predecessor: 
1) **Candidate Diversity**: By enforcing a minimum distance between selected candidates, the algorithm avoids evaluating redundant points, maintaining diverse exploration while preserving parallelism. 
2) **Adaptive Trust Region Dynamics**: Expansion/contraction factors are now proportional to improvement magnitude (1.3 for significant gains, 1.1 for minor gains), while plateau detection triggers aggressive trust region resets. This prevents the algorithm from getting stuck in local minima.
3) **Accurate New Best Detection**: Corrects a bug in progress tracking by comparing the new batch's minimum objective value to the previous best, ensuring trust region updates reflect actual improvements.
The algorithm maintains computational efficiency through reduced model refitting frequency and low-cost Sobol candidate generation, while the trust region strategy ensures focused local search with global exploration capabilities.

# Code 
```python
from collections.abc import Callable
from scipy.stats import qmc
from scipy.stats import norm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class DynamicDiversityTrustRegionMaternEIBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(5, 2 * dim)
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.sobol_gen = qmc.Sobol(d=dim, scramble=True, seed=int(self.rng.integers(2**31 - 1)))
        # Trust region parameters
        self.tr_center = None
        self.tr_radius = 1.0  # Smaller initial radius
        self.max_tr_radius = 5.0  # Half of domain range
        self.min_tr_radius = 0.1  # Minimum allowable radius
        self.iteration = 0  # Track iterations for refitting frequency
        self.no_improvement_count = 0
        self.no_improvement_threshold = 5

    def _sample_points(self, n_points, low=None, high=None):
        """Generate Sobol candidates within specified bounds (default: overall domain)."""
        if low is None:
            low = self.bounds[0]
        if high is None:
            high = self.bounds[1]
        samples = self.sobol_gen.random(n_points)
        scaled_samples = qmc.scale(samples, low, high)
        return scaled_samples

    def _fit_model(self, X, y):
        """Fit Gaussian Process with Matern kernel and ARD, fewer restarts for speed."""
        kernel = Matern(nu=2.5, length_scale=np.ones(self.dim), length_scale_bounds=(1e-3, 1e3))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=1,  # Reduced restarts for speed
            random_state=int(self.rng.integers(2**31 - 1))
        )
        model.fit(X, y)
        return model

    def _acquisition_function(self, X):
        """Expected Improvement for minimization."""
        mean, std = self.model.predict(X, return_std=True)
        best_y = np.min(self.y)
        improvement = best_y - mean
        with np.errstate(divide='ignore', invalid='ignore'):
            z = improvement / std
            ei = improvement * norm.cdf(z) + std * norm.pdf(z)
            ei[std == 0.0] = 0.0  # Handle zero variance
        return ei

    def _select_next_points(self):
        """Generate candidates within current trust region, select top 2 with diversity."""
        if self.tr_center is None:
            return self._sample_points(50)  # Fallback to global search if no trust region
        
        # Compute trust region bounds (clamped to domain)
        low = np.clip(self.tr_center - self.tr_radius, self.bounds[0], self.bounds[1])
        high = np.clip(self.tr_center + self.tr_radius, self.bounds[0], self.bounds[1])
        
        candidates = self._sample_points(100, low=low, high=high)  # 100 candidates in trust region
        acq_vals = self._acquisition_function(candidates)
        
        # Sort indices by acquisition value descending
        sorted_indices = np.argsort(acq_vals, axis=0)[::-1]  # descending
        if len(sorted_indices) < 2:
            return candidates[sorted_indices]
        
        # First candidate is the best
        first_idx = sorted_indices[0]
        first_candidate = candidates[first_idx]
        
        # Find second candidate with sufficient distance from first
        min_dist = 0.1 * self.tr_radius  # 10% of current radius
        remaining_indices = sorted_indices[1:]
        best_second_idx = None
        best_second_ei = -np.inf
        
        for idx in remaining_indices:
            candidate = candidates[idx]
            dist = np.linalg.norm(candidate - first_candidate)
            if dist >= min_dist:
                ei_val = acq_vals[idx]
                if ei_val > best_second_ei:
                    best_second_ei = ei_val
                    best_second_idx = idx
        
        if best_second_idx is not None:
            second_idx = best_second_idx
        else:
            second_idx = sorted_indices[1]  # fall back to closest available
        
        next_X = np.array([candidates[first_idx], candidates[second_idx]])
        return next_X

    def _evaluate_points(self, func, X):
        """Evaluate new points and update counter."""
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        """Update evaluation history."""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation with Sobol
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Set initial trust region center
        best_idx = np.argmin(self.y)
        self.tr_center = self.X[best_idx].copy()
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        while self.n_evals < self.budget:
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Update trust region
            current_best_idx = np.argmin(self.y)
            previous_best_value = self.y[best_idx]
            new_best_value = self.y[current_best_idx]
            
            if new_best_value < previous_best_value:
                # Reset plateau counter
                self.no_improvement_count = 0
                
                # Adaptive expansion based on improvement magnitude
                improvement_ratio = (previous_best_value - new_best_value) / (abs(previous_best_value) + 1e-8)
                if improvement_ratio > 0.05:  # 5% significant improvement
                    expansion_factor = 1.3
                else:
                    expansion_factor = 1.1
                    
                self.tr_radius = min(self.tr_radius * expansion_factor, self.max_tr_radius)
            else:
                # Increment plateau counter
                self.no_improvement_count += 1
                contraction_factor = 0.8
                
                if self.no_improvement_count >= self.no_improvement_threshold:
                    # Aggressive reset when stuck in plateau
                    self.tr_radius = self.max_tr_radius
                else:
                    # Normal contraction
                    self.tr_radius = max(self.tr_radius * contraction_factor, self.min_tr_radius)
            
            # Update best solution and trust region center
            best_idx = current_best_idx
            self.tr_center = self.X[best_idx].copy()
            
            # Refit model every 2 iterations to reduce overhead
            self.iteration += 1
            if self.iteration % 2 == 0:
                self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm DynamicDiversityTrustRegionMaternEIBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4740 with standard deviation 0.0064.

took 6.77 seconds to run.