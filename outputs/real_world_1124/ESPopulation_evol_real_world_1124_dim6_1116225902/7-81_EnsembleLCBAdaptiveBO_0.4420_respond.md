# Thinking (reasoning of the model, not part of the answer)
Okay, I need to design a new Bayesian Optimization algorithm that's different from the existing ones, especially the TrustRegionSobolMaternEIBO. Let me start by understanding the current algorithms and their components.

The existing TrustRegionSobolMaternEIBO uses a dynamic trust region with Sobol candidates and Expected Improvement (EI) as the acquisition function. The trust region adjusts based on whether progress is made, expanding or contracting the area to sample. It starts with a Sobol sequence for the initial design and uses a Gaussian Process (GP) with a Matérn kernel. The main components are the trust region strategy, Sobol candidates, and EI.

To create a novel approach, I need to change at least one key component. Let's consider changing the surrogate model, acquisition function, or search strategy. Since the existing algorithm uses a GP with a Matérn kernel and EI, maybe switching to a different model like an ensemble of Random Forests or using a different acquisition function like Thompson Sampling could help. Also, the trust region strategy is central to their approach, so perhaps replacing that with a different exploration strategy.

Wait, the user mentioned the existing algorithms use different exploration strategies. For example, ThompsonSamplingEnsembleMaternBO uses Thompson Sampling. So maybe combining an ensemble model with a different acquisition function could be a way. But I need to ensure it's distinct.

Another idea: using a different surrogate model. For example, instead of a single GP, use an ensemble of models to capture uncertainty better. Also, maybe using an acquisition function like Lower Confidence Bound (LCB) with a dynamic exploration parameter. The LCB can balance exploration and exploitation by adjusting the exploration weight based on the current state of the search.

Alternatively, instead of a trust region, perhaps implement a multi-fidelity approach, but the problem here is a black box without fidelity information. So that's not applicable.

What about using a different candidate generation method? The current one uses Sobol within a trust region. Maybe using a different sampling method, like Latin Hypercube Sampling for the initial design, but that's already in some algorithms. Hmm.

Wait, another angle: the current algorithms focus on local trust regions. Maybe instead, use a global exploration strategy but with a dynamic switch to local search as the budget increases. However, the TrustRegionSobolMaternEIBO already does that to some extent.

Let me think about the components again. The existing one uses a GP with anisotropic Matérn kernel. Maybe using a different kernel, but the user mentioned that the algorithm must be computationally efficient, so changing the kernel might not help. Alternatively, perhaps using a different model like a Random Forest or another type of ensemble model. Random Forests can handle non-Gaussian assumptions and might be faster in some cases, but they don't provide uncertainty estimates directly. However, the acquisition functions like Thompson Sampling can work with ensembles by sampling from the trees.

Wait, the user allows using scikit-learn's RandomForestRegressor. So maybe using an ensemble of Random Forests as the surrogate model. Then, use Thompson Sampling to generate candidates. Also, instead of a trust region, maybe use a different mechanism to balance exploration and exploitation. For example, using a dynamic exploration parameter in the acquisition function.

Another idea: instead of the trust region, use a dynamic exploration parameter in the Lower Confidence Bound (LCB) acquisition function. The LCB acquisition function is defined as μ(x) - β_t * σ(x), where β_t controls the exploration-exploitation trade-off. If β_t is adjusted based on the progress or the current budget, it can adaptively shift from exploration to exploitation.

Let me outline the new algorithm:

- Surrogate Model: Use an ensemble of Random Forests (multiple models) to estimate the mean and variance. Each tree's prediction can be used to compute uncertainty estimates.
- Acquisition Function: Lower Confidence Bound with a dynamic β parameter. The β value could be increased when the algorithm detects that it's in an exploration phase and decreased when exploiting.
- Search Strategy: Instead of a trust region, generate candidates globally using Latin Hypercube Sampling (LHS) but with the acquisition function optimized using a combination of random sampling and local optimization.

Wait, but the user mentioned to avoid using the same components. The existing algorithms use Sobol sequences, GPs, and trust regions. So using an ensemble model and LCB with dynamic β would be different.

Let me check the components again. The algorithm should be named based on characteristics. Maybe "EnsembleLCBAdaptiveBO" or something like that.

Let me structure the key components:

1. **Initial Design**: Latin Hypercube Sampling (LHS) to ensure diverse initial points. This is different from the Sobol used in the existing algorithms.

2. **Surrogate Model**: An ensemble of Random Forests. Each tree in the ensemble is trained on a subset of the data. The predictions from all trees can be used to compute the mean and variance, which are necessary for the LCB acquisition function.

3. **Acquisition Function**: Lower Confidence Bound (LCB) with an adaptive β parameter. The β value starts high to encourage exploration and decreases over time as more data is collected. Alternatively, β can be adjusted based on the observed progress (e.g., if improvements stop for a few iterations, increase β to explore more).

4. **Candidate Generation**: Use LHS to generate a large number of candidates, then optimize the acquisition function by selecting the top candidates. This combines global exploration with local optimization.

5. **Dynamic Adjustment**: Monitor the progress. If there's no improvement over a certain number of iterations, increase the β parameter to encourage more exploration. Conversely, if progress is steady, reduce β to exploit more.

Now, let's think about the code structure. The existing code uses Sobol for initial design, GP with Matérn kernel, and EI. The new code would replace these with LHS, Random Forest ensemble, and LCB with adaptive β.

Implementing the ensemble model would involve using scikit-learn's RandomForestRegressor. For the acquisition function, the LCB requires the mean and standard deviation predictions from the model. The dynamic β could be updated based on the number of iterations or the presence of improvements.

For the acquisition function optimization, instead of using the Sobol candidates in a trust region, generate a large number of candidates via LHS and select the best ones based on the LCB value.

This approach should be different enough from the existing algorithms, using an ensemble model and a different acquisition function. It also introduces a dynamic adjustment of the exploration parameter, which isn't present in the existing algorithms that use fixed parameters or trust regions.

Now, the code. Let's outline the steps:

- Initialize with LHS for the initial design. The number of initial points could be 2*dim.
- Train an ensemble of Random Forests. For each iteration, generate LHS candidates, compute the LCB for each, select the top ones.
- The dynamic β can be adjusted based on whether there's improvement. For example, if there's no improvement in the last few iterations, increase β by a factor.

But how to implement the ensemble? The Random Forest can be a single model, but to get uncertainty estimates, perhaps using multiple models with different random states or bootstrapping. Alternatively, using the variance across the trees as the uncertainty.

Wait, the RandomForestRegressor in scikit-learn outputs the mean prediction, but the variance can be estimated by the variance of the predictions from individual trees. However, scikit-learn doesn't provide this directly. So, for each tree in the forest, we can get the prediction and compute the variance across them. That would require modifying the model to return both mean and variance.

Alternatively, use a single Random Forest and bootstrap the variance. But this might complicate things. Maybe using a single model and using the tree's variance as the uncertainty estimate. However, that might not be accurate. Alternatively, using a Gaussian Process with a different kernel, but we already know that the existing algorithm uses GP.

Wait, perhaps using a Quantile Regression Forest. But that's not in scikit-learn. Another approach is to use the variance of the out-of-bag predictions. But this might not be straightforward.

Alternatively, use multiple Random Forest models trained on different subsets of the data and average their predictions. The variance across these models can be used as the uncertainty estimate. This would be an ensemble of ensembles. For example, train 5 Random Forests, each on different bootstrap samples, and then for prediction, take the average of the 5 models' predictions and compute the variance across them.

Yes, that could work. So, the surrogate model is an ensemble of multiple Random Forest models. For prediction, take the mean of the models' predictions and the variance across them as the uncertainty.

Now, in the code, the _fit_model function would create this ensemble. Each model in the ensemble is a RandomForestRegressor with different hyperparameters or random states.

But how to implement that in scikit-learn? Let's see. For example, create 5 different RandomForest models, each with different random states and possibly different parameters. Then, for a new X, predict with all models, compute the mean and variance.

This approach would be more computationally intensive than a single GP, but given that the existing algorithm runs in 38 seconds for 600 evaluations, this should be feasible as long as the number of models isn't too high.

Now, the acquisition function: Lower Confidence Bound (LCB) = mean - β * sqrt(variance). The β starts at a higher value and decreases as the algorithm progresses. Alternatively, adjust β based on the progress. For example, if there's no improvement in the last few iterations, increase β.

The candidate generation: Use Latin Hypercube Sampling to generate a large set of candidates (e.g., 100 points), then select the top ones based on the LCB value. This is more efficient than using optimization algorithms like L-BFGS-B.

This approach should be different from the existing algorithms. Let's check the components again:

- Surrogate Model: Ensemble of Random Forests.
- Acquisition Function: LCB with adaptive β.
- Search Strategy: LHS for initial design and candidate generation.
- Exploration Strategy: Dynamic β based on progress.

Yes, this should be distinct. Now, the code structure.

The class name could be EnsembleLCBAdaptiveBO. The initial design uses LHS. The surrogate model is an ensemble of Random Forests. The acquisition function is LCB with β adjusted based on progress. The candidate selection uses LHS.

Potential issues: The ensemble of Random Forests may be slower to train and predict compared to a GP, especially with many trees. However, the code must be efficient enough to run within the time constraints. Since the existing algorithm uses a GP with 3 restarts for hyperparameters, the ensemble approach should be designed to be computationally feasible.

Another consideration: The initial design with LHS. The code can use qmc.LatinHypercube for generating the initial points.

For the acquisition function optimization, generating 100 candidates via LHS each iteration and selecting the top one would be manageable. The number of candidates can be adjusted based on the problem's dimensionality.

Now, the dynamic adjustment of β. For example, starting with β=5.0 and decreasing by 0.5 each time there's improvement. If there's no improvement for a certain number of iterations, increase β by 1.0. But this requires tracking the best value over iterations and adjusting β accordingly.

Let me outline the code steps:

In __init__:

- Set initial design to Latin Hypercube.
- Initialize the ensemble of Random Forest models.
- Define the initial β and a mechanism to adjust it.

In _fit_model:

- Train multiple RandomForest models on the data.

In _acquisition_function:

- Compute the mean and variance from the ensemble's predictions.
- Return LCB value.

In _select_next_points:

- Generate candidates via LHS.
- Evaluate LCB for each candidate.
- Select the best candidate(s).

In the main loop:

- After evaluating points, check if there's improvement.
- Adjust β accordingly.

Now, implementing the ensemble. Let's say we create 5 Random Forest models, each with a different random state. For each model in the ensemble, we train it on the data. Then, for a new X, we get the mean of all models' predictions and the variance across them.

But wait, the RandomForestRegressor can be trained on the same data, but with different subsets. However, using scikit-learn's RandomForestRegressor, each model in the ensemble already uses bootstrapping. So perhaps using multiple models isn't necessary. Alternatively, using a single model and using the variance of the individual tree predictions. Let me check how to get that.

In scikit-learn, each RandomForest has a feature to get the variance across trees. For example, after fitting the model, we can get the predictions from each tree and compute the variance. However, this isn't directly accessible, but we can loop through the trees and collect their predictions.

But that would be computationally expensive. For each prediction step, we'd have to loop through all trees in the forest, which could be slow for high-dimensional problems. Given the time constraints, this might not be feasible. Alternatively, approximate the variance using the standard deviation provided by the model. But the RandomForestRegressor doesn't provide uncertainty estimates by default.

Hmm, this is a problem. The LCB acquisition function requires an estimate of the variance. Without a reliable way to get that from a Random Forest, this approach may not work. So perhaps using a Gaussian Process instead, but that's what the existing algorithm does. Alternatively, use a different model that provides uncertainty estimates.

Another idea: use a Quantile Regression Forest, which provides quantiles that can be used to estimate the variance. However, scikit-learn's RandomForest doesn't support this directly. Maybe using another library like sklearn's GradientBoostingRegressor with quantile loss, but that's more complex.

Alternatively, use a Bayesian Random Forest, but that's not part of sc4ikit-learn. Hmm.

This suggests that using a Random Forest as the surrogate model might not be straightforward for getting variance estimates. Therefore, perhaps using a different model that can provide both mean and variance. For example, a Bayesian Neural Network, but that's not in the allowed libraries (the user allows torch, so maybe possible). However, the user said to use numpy, scipy, scikit-learn, torch, gpytorch. So maybe using a Bayesian Neural Network is possible, but that would require writing code to train it.

But the problem is time. The algorithm must be efficient. Training a Bayesian Neural Network for each iteration might be too slow. The existing algorithm uses a GP with 3 restarts, which is manageable.

Alternative approach: Use a single Gaussian Process but with a different kernel, say an RBF kernel, and use the Lower Confidence Bound as the acquisition function. But the existing algorithm uses a Matérn kernel and Expected Improvement, so that's different but not novel enough.

Wait, the user wants the algorithm to be clearly different. So maybe combining the LCB acquisition function with the ensemble approach, but how to get the variance. Maybe approximate the variance using the out-of-bag error of the Random Forest. The OOB error can be used to estimate the variance in the predictions.

Alternatively, use the variance of the predictions from different trees in the forest. For example, in scikit-learn's RandomForest, we can get the predictions from each tree and compute the variance across them. Here's how:

For a given X, the prediction of each tree can be obtained by looping through the `estimators_` attribute of the RandomForest. Then, the variance of these predictions can be computed as the variance across trees.

But for each prediction call, this would be O(number of trees) in time, which might be slow. However, given that the existing algorithm uses a GP with 3 restarts, and the problem's time constraint is 120 seconds for 10 runs, this might be acceptable.

Let me proceed with this approach.

In the code:

The _fit_model function would create a single RandomForestRegressor. Then, for prediction, collect the predictions from all trees in the forest, compute the mean and variance.

But how to collect predictions from each tree? For each tree in the forest's estimators_, predict and collect the results.

For example:

def _predict(self, X):
    preds = np.array([tree.predict(X).ravel() for tree in self.model.estimators_])
    mean = np.mean(preds, axis=0)
    var = np.var(preds, axis=0)
    return mean, var

Yes, but for high-dimensional X, this could be slow. However, the problem is of dimension 6, which is manageable.

So, in code:

- The surrogate model is a RandomForestRegressor with multiple trees (say, 50 trees). The predictions from each tree are used to compute the mean and variance.

Now, the code for the new algorithm.

Let me outline the code step by step.

The class name: EnsembleLCBAdaptiveBO.

The __init__ method initializes the budget, dim, bounds, etc. Uses LHS for initial design.

The _sample_points method generates Latin Hypercube samples.

The _fit_model method creates a RandomForestRegressor with n_estimators=20, for example, and fits it.

The _acquisition_function computes the LCB using the mean and variance from the model's predictions.

The _select_next_points generates candidates via LHS, evaluates LCB, selects the best.

Dynamic β adjustment: initial β is set to a value, say 2.0. If there's no improvement for a certain number of iterations, increase β by 0.5. If there is improvement, decrease β by 0.1, but not below 0.5.

This dynamic adjustment can be tracked by comparing the current best value with the previous ones.

Now, writing the code.

Potential code steps:

import numpy as np
from scipy.stats import qmc
from sklearn.ensemble import RandomForestRegressor

class EnsembleLCBAdaptiveBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0]*dim, [5.0]*dim])
        self.X = None
        self.y = None
        self.n_evals = 0
        self.n_init = max(5, 2 * dim)
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_gen = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.beta = 2.0  # Initial beta value for LCB
        self.min_beta = 0.5
        self.max_beta = 5.0
        self.no_improvement_count = 0  # Counter for no improvement

    def _sample_points(self, n_points, low=None, high=None):
        if low is None:
            low = self.bounds[0]
        if high is None:
            high = self.bounds[1]
        samples = self.lhs_gen.random(n_points)
        scaled_samples = qmc.scale(samples, low, high)
        return scaled_samples

    def _fit_model(self, X, y):
        model = RandomForestRegressor(
            n_estimators=20,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=-1
        )
        model.fit(X, y.ravel())
        return model

    def _predict(self, X):
        # Collect predictions from all trees in the forest
        preds = np.array([tree.predict(X).ravel() for tree in self.model.estimators_])
        mean = np.mean(preds, axis=0)
        var = np.var(preds, axis=0)
        return mean, var

    def _acquisition_function(self, X):
        mean, var = self._predict(X)
        std = np.sqrt(var)
        best_y = np.min(self.y)
        lcb = mean - self.beta * std
        return lcb

    def _select_next_points(self):
        candidates = self._sample_points(100)  # Generate 100 candidates via LHS
        acq_vals = self._acquisition_function(candidates)
        next_idx = np.argmin(acq_vals)  # Since LCB is for minimization, we minimize
        next_X = candidates[next_idx].reshape(1, -1)
        return next_X

    def _evaluate_points(self, func, X):
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.array]:
        # Initial evaluation with Latin Hypercube
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)
        
        # Fit initial model
        self.model = self._fit_model(self.X, self.y)
        
        best_y = np.min(self.y)
        best_idx = np.argmin(self.y)
        previous_best = best_y.copy()
        
        while self.n_evals < self.budget:
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)
            
            # Update best solution
            current_best = np.min(self.y)
            if current_best < best_y:
                best_y = current_best
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
            
            # Adjust beta based on no improvement count
            if self.no_improvement_count >= 3:
                # Increase beta if no improvement for 3 iterations
                self.beta = min(self.beta + 0.5, self.max_beta)
            else:
                # Decrease beta if there's improvement
                self.beta = max(self.beta - 0.1, self.min_beta)
            
            # Refit model with updated data
            self.model = self._fit_model(self.X, self.y)
        
        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x

Wait, but in the _select_next_points method, it generates 100 candidates via LHS. The acquisition function is LCB, which is minimized. The best candidate is the one with the lowest LCB value. This is correct for minimization.

The dynamic β adjustment increases when no improvement for 3 iterations and decreases otherwise. This should encourage more exploration when the algorithm is stuck.

This approach uses an ensemble (Random Forest with multiple trees) to provide uncertainty estimates via variance across trees. The acquisition function is LCB with adaptive β. The initial design uses Latin Hypercube Sampling instead of Sobol. The candidate generation uses LHS again.

This is different from the existing algorithms, which use GPs with Sobol and trust regions. The surrogate model is an ensemble model, the acquisition function is LCB with dynamic parameter, and the search strategy is global LHS.

Potential improvements: The number of candidates generated per iteration could be adjusted based on the dimension. For example, generate 2*dim candidates. Also, the number of trees in the Random Forest could be tuned for efficiency.

Another point: The code uses a single Random Forest model with multiple trees, rather than an ensemble of multiple models. But the variance is computed across the trees in the model, so it's effectively using the ensemble's predictions to estimate uncertainty. This is correct.

Testing this code would require checking if the Random Forest model's predictions can be obtained in the _predict method. The code loops through each tree in the model's estimators_ and collects their predictions.

Yes, this is possible in scikit-learn. Each tree in a RandomForestRegressor is stored in model.estimators_.

This should work. The key difference from the existing algorithm is the use of an ensemble model (Random Forest) with variance from the trees, the acquisition function as LCB with adaptive β, and the use of Latin Hypercube Sampling for both initial design and candidate generation.

Now, the justification. The new algorithm uses an ensemble model to capture uncertainty, a dynamic acquisition function to balance exploration and exploitation, and a global sampling strategy. This should differ significantly from the trust region and EI-based approaches.

The code should be efficient enough, as the Random Forest with 20 trees and 100 candidates per iteration is manageable within the time constraints.


# Answer
# Description  
**EnsembleLCBAdaptiveBO** introduces a novel Bayesian Optimization approach leveraging ensemble Random Forests for uncertainty estimation, dynamic Lower Confidence Bound (LCB) acquisition with adaptive exploration parameters, and global Latin Hypercube Sampling (LHS) candidates. It avoids trust regions entirely, instead relying on the ensemble's statistical variance across trees to quantify uncertainty. The LCB acquisition dynamically adjusts its exploration parameter (β) based on optimization progress, increasing β during stagnation and decreasing it during improvement to balance global exploration and local exploitation. This design decouples the surrogate model from the trust-region strategy, enabling adaptive exploration without region constraints.

# Justification  
1. **Ensemble Surrogate Model**:  
   - Random Forest ensembles (with 20 trees) provide robust uncertainty estimates via tree-level prediction variance. This is critical for the LCB acquisition, which requires both mean and variance. Unlike GPs, Random Forests handle non-smooth, non-Gaussian objectives efficiently and scale better with large budgets.  

2. **LCB Acquisition with Adaptive β**:  
   - Unlike fixed β in standard LCB, this algorithm dynamically adjusts β (initial β=2.0, min β=0.5, max β=5.0) based on whether progress is made (no improvement over 3 iterations triggers β increase). This self-regulating mechanism avoids overexploitation in narrow regions while avoiding premature convergence.  

3. **Global LHS Candidates**:  
   - Replaces Sobol sequences and trust regions with Latin Hypercube Sampling for candidate generation. This ensures diverse global sampling while maintaining low computational overhead compared to constrained optimization methods.  

4. **Efficient Ensemble Training**:  
   - Uses `n_estimators=20` for the Random Forest, balancing model complexity and training speed. The variance across trees provides reliable uncertainty estimates without requiring kernel hyperparameter tuning (as in GPs).  

5. **Progress-Driven Exploration**:  
   - The β parameter adaptation introduces a feedback loop between model exploration and optimization progress. This is a key differentiator from fixed-schedule exploration methods like Thompson Sampling or static EI thresholds.  

6. **Computational Efficiency**:  
   - The algorithm avoids costly GP hyperparameter optimization and trust-region bookkeeping. It uses parallelizable Random Forest training (`n_jobs=-1`) and simple candidate evaluation, ensuring compatibility with the 120-second runtime constraint for 600 evaluations.  

# Code  
```python
from collections.abc import Callable
from scipy.stats import qmc
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class EnsembleLCBAdaptiveBO:
    def __init__(self, budget: int, dim: int):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([[-5.0] * dim, [5.0] * dim])
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.n_evals = 0
        self.n_init = max(5, 2 * dim)  # Ensure at least 5 initial points
        self.rng = np.random.default_rng(np.random.randint(2**31 - 1))
        self.lhs_gen = qmc.LatinHypercube(d=dim, seed=int(self.rng.integers(2**31 - 1)))
        self.beta = 2.0  # Initial beta for Lower Confidence Bound
        self.min_beta = 0.5
        self.max_beta = 5.0
        self.no_improvement_count = 0  # Track progress for adaptive beta

    def _sample_points(self, n_points, low=None, high=None):
        """Generate Latin Hypercube candidates within specified bounds."""
        if low is None:
            low = self.bounds[0]
        if high is None:
            high = self.bounds[1]
        samples = self.lhs_gen.random(n_points)
        scaled_samples = qmc.scale(samples, low, high)
        return scaled_samples

    def _fit_model(self, X, y):
        """Train a Random Forest ensemble for mean and variance prediction."""
        model = RandomForestRegressor(
            n_estimators=20,
            random_state=int(self.rng.integers(2**31 - 1)),
            n_jobs=-1  # Enable parallel training
        )
        model.fit(X, y.ravel())
        return model

    def _predict(self, X):
        """Estimate mean and variance using tree-level predictions."""
        preds = np.array([tree.predict(X).ravel() for tree in self.model.estimators_])
        mean = np.mean(preds, axis=0)
        var = np.var(preds, axis=0)
        return mean, var

    def _acquisition_function(self, X):
        """Lower Confidence Bound for minimization."""
        mean, var = self._predict(X)
        std = np.sqrt(var)
        lcb = mean - self.beta * std
        return lcb

    def _select_next_points(self):
        """Generate 100 global candidates and select the top LCB."""
        candidates = self._sample_points(100)  # 100 LHS candidates
        acq_vals = self._acquisition_function(candidates)
        best_idx = np.argmin(acq_vals)  # Minimize LCB
        next_X = candidates[best_idx].reshape(1, -1)
        return next_X

    def _evaluate_points(self, func, X):
        """Evaluate new points, respecting evaluation budget."""
        max_points = self.budget - self.n_evals
        n_points = min(len(X), max_points)
        y = np.zeros((n_points, 1))
        for i in range(n_points):
            y[i] = func(X[i])
            self.n_evals += 1
        return y

    def _update_eval_points(self, new_X, new_y):
        """Update stored evaluation history."""
        if self.X is None:
            self.X = new_X
            self.y = new_y
        else:
            self.X = np.vstack([self.X, new_X])
            self.y = np.vstack([self.y, new_y])

    def __call__(self, func: Callable[[np.ndarray], np.float64]) -> tuple[np.float64, np.ndarray]:
        # Initial evaluation with Latin Hypercube Sampling
        X_init = self._sample_points(self.n_init)
        y_init = self._evaluate_points(func, X_init)
        self._update_eval_points(X_init, y_init)

        # Initialize best solution and model
        best_idx = np.argmin(self.y)
        self.model = self._fit_model(self.X, self.y)
        previous_best = np.min(self.y).copy()

        while self.n_evals < self.budget:
            next_X = self._select_next_points()
            next_y = self._evaluate_points(func, next_X)
            self._update_eval_points(next_X, next_y)

            # Track progress for adaptive beta
            current_best = np.min(self.y)
            if current_best < previous_best:
                self.no_improvement_count = 0
                previous_best = current_best
            else:
                self.no_improvement_count += 1

            # Dynamically adjust beta
            if self.no_improvement_count >= 3:
                self.beta = min(self.beta + 0.5, self.max_beta)  # Increase exploration
            else:
                self.beta = max(self.beta - 0.1, self.min_beta)  # Decrease exploration

            # Refit model
            self.model = self._fit_model(self.X, self.y)

        best_idx = np.argmin(self.y)
        best_y = self.y[best_idx]
        best_x = self.X[best_idx]
        return best_y, best_x
```
## Feedback
 The algorithm EnsembleLCBAdaptiveBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4420 with standard deviation 0.0271.

took 22.10 seconds to run.