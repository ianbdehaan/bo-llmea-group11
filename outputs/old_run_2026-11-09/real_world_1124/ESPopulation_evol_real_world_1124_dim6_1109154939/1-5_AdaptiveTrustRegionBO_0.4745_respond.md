**# Description**  
**Adaptive‑Trust‑Region Bayesian Optimiser (ATR‑BO)**  

- **Sobol quasi‑Monte‑Carlo seed** – a deterministic space‑filling design (≈ 2·dim points) that quickly gives a diverse initial model.  
- **Gaussian‑Process surrogate** – scikit‑learn GP with a Matern‑5/2 kernel, normalised outputs and a small nugget for numerical stability.  
- **Upper‑Confidence‑Bound (UCB) acquisition** – `UCB(x)=μ(x)‑β·σ(x)` (minimisation).  The exploration weight β grows logarithmically with the iteration number, guaranteeing theoretical regret bounds.  
- **Adaptive trust‑region** – the search is confined to a hyper‑cube centred on the current best solution.  The radius shrinks after a configurable number of consecutive failures and expands after a success, focusing the optimisation where the surrogate predicts improvement while still allowing global exploration early on.  
- **Local refinement of the acquisition** – a cheap Sobol sample inside the trust region provides a good starting point; a single L‑BFGS‑B run (bounded to the region) refines the UCB optimum.  
- **Sequential (batch‑size = 1) update** – after each function evaluation the GP is refitted, guaranteeing that the surrogate always reflects the latest information.  

The algorithm respects a strict evaluation budget, works in any dimension, and uses only NumPy, SciPy and scikit‑learn – all permitted libraries.  By combining a global Sobol start, a theoretically‑grounded UCB acquisition, and a trust‑region that automatically balances exploration/exploitation, ATR‑BO is well‑suited for the 6‑D Frequency‑Modulated Sound Waves BBOB problem where the budget is limited.

---

**# Justification**  

| Component | Reasoning |
|-----------|-----------|
| **Sobol initial design** | Guarantees low‑discrepancy coverage without random variance; with only a few dozen evaluations (budget ≈ 200 for BBOB) it yields a reliable GP fit. |
| **Matern‑5/2 kernel** | Provides smoothness similar to many BBOB functions while remaining flexible enough for the highly‑oscillatory FM‑Sound‑Waves landscape. |
| **UCB acquisition** | Unlike EI, UCB has a closed‑form expression and a simple, provably convergent β‑schedule; the minimisation problem is easier to solve with gradient‑based local optimisation. |
| **Adaptive trust‑region** | Early iterations keep a large region (global search).  When the optimiser stagnates the region contracts, concentrating evaluations where the GP predicts improvement – essential for high‑dimensional, multi‑modal problems under tight budgets. |
| **Sobol candidate + L‑BFGS‑B refinement** | The Sobol sample gives a good global guess inside the region; a single deterministic gradient step efficiently reaches a local UCB optimum, avoiding expensive multi‑start optimisation. |
| **Sequential update** | Guarantees that each new observation immediately improves the surrogate, which is crucial when the total number of evaluations is small. |
| **Budget‑aware loop** | The implementation tracks the number of calls and never exceeds `self.budget`, returning the best point found so far. |

Overall, ATR‑BO builds on the strengths of the existing QMC‑EI‑BO (Sobol start, GP surrogate) but replaces the batch‑EI scheme with a theoretically‑grounded, adaptively focused UCB search that makes better use of each evaluation in a limited‑budget setting.

---

**# Code**  

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


class AdaptiveTrustRegionBO:
    """
    Adaptive‑Trust‑Region Bayesian Optimiser (ATR‑BO).

    Parameters
    ----------
    budget : int
        Maximum number of allowed calls to the black‑box function.
    dim : int
        Dimensionality of the search space (bounds are always [-5, 5]^dim).
    init_samples : int, optional
        Number of Sobol points used for the initial design.
        If None, defaults to 2 * dim (a common rule‑of‑thumb).
    radius_init : float, optional
        Initial trust‑region half‑width as a fraction of the total bound length
        (default 0.5 → region spans the whole space).
    success_patience : int, optional
        Number of consecutive successes needed to enlarge the region.
    failure_patience : int, optional
        Number of consecutive failures after which the region shrinks.
    """

    def __init__(self,
                 budget,
                 dim,
                 init_samples=None,
                 radius_init=0.5,
                 success_patience=3,
                 failure_patience=5):
        self.budget = int(budget)
        self.dim = int(dim)
        self.lower = -5.0
        self.upper = 5.0
        self.bounds = np.column_stack((np.full(dim, self.lower),
                                       np.full(dim, self.upper)))
        self.init_samples = init_samples if init_samples is not None else 2 * dim
        self.radius_init = radius_init
        self.success_patience = success_patience
        self.failure_patience = failure_patience

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def _scale_sobol(self, points):
        """Scale Sobol points from [0,1]^d to the problem bounds."""
        return self.lower + points * (self.upper - self.lower)

    def _initial_design(self):
        """Generate Sobol initial points."""
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol requires a power‑of‑two number of points; we take the next power.
        n = 1 << int(np.ceil(np.log2(self.init_samples)))
        raw = sampler.random_base2(m=int(np.log2(n)))
        X = self._scale_sobol(raw)[:self.init_samples]
        return X

    def _fit_gp(self, X, y):
        """Fit a GP with a Matern‑5/2 kernel."""
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(self.dim),
                           length_scale_bounds=(1e-2, 1e2),
                           nu=2.5)
                  + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2)))
        gp = GaussianProcessRegressor(kernel=kernel,
                                      normalize_y=True,
                                      n_restarts_optimizer=5,
                                      random_state=0)
        gp.fit(X, y)
        return gp

    def _beta(self, t, delta=0.1):
        """Exploration weight for GP‑UCB (minimisation version)."""
        # Standard theoretical schedule (Srinivas et al., 2010)
        return np.sqrt(2 * np.log(self.dim * (t ** 2) * (np.pi ** 2) / (6 * delta)))

    def _ucb(self, x, gp, beta):
        """Upper‑Confidence‑Bound acquisition (to be minimised)."""
        mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
        return mu[0] - beta * sigma[0]

    def _trust_region_bounds(self, centre, radius_frac):
        """Compute axis‑aligned trust‑region bounds."""
        half_width = radius_frac * (self.upper - self.lower) / 2.0
        low = np.maximum(centre - half_width, self.lower)
        high = np.minimum(centre + half_width, self.upper)
        return np.column_stack((low, high))

    def _candidate_in_region(self, n_candidates, centre, radius_frac):
        """Sobol candidates confined to the current trust region."""
        tr_bounds = self._trust_region_bounds(centre, radius_frac)
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Use a power‑of‑two number of points for Sobol.
        m = int(np.ceil(np.log2(n_candidates)))
        raw = sampler.random_base2(m=m)[:n_candidates]
        # Scale each dimension independently.
        cand = tr_bounds[:, 0] + raw * (tr_bounds[:, 1] - tr_bounds[:, 0])
        return cand

    # --------------------------------------------------------------------- #
    # Main optimisation entry point
    # --------------------------------------------------------------------- #
    def __call__(self, func):
        """
        Run the optimisation.

        Parameters
        ----------
        func : callable
            Black‑box objective to minimise.  Must accept a 1‑D NumPy array
            of length `self.dim` and return a scalar.

        Returns
        -------
        best_x : ndarray, shape (dim,)
            The best point found within the allocated budget.
        best_f : float
            Function value at `best_x`.
        """
        # -----------------------------------------------------------------
        # 1. Initial design
        # -----------------------------------------------------------------
        X = self._initial_design()
        n_initial = X.shape[0]
        y = np.empty(n_initial)

        evals = 0
        for i in range(n_initial):
            if evals >= self.budget:
                break
            y[i] = func(X[i])
            evals += 1

        # If the budget is exhausted already, return the best initial point.
        if evals >= self.budget:
            best_idx = np.argmin(y[:evals])
            return X[best_idx], float(y[best_idx])

        # -----------------------------------------------------------------
        # 2. Main loop
        # -----------------------------------------------------------------
        # GP model on the data collected so far.
        gp = self._fit_gp(X[:evals], y[:evals])

        # Trust‑region state
        centre = X[np.argmin(y[:evals])]          # current best point
        radius_frac = self.radius_init           # fraction of full width
        success_counter = 0
        failure_counter = 0

        # Main optimisation loop (sequential, one evaluation per iteration)
        while evals < self.budget:
            t = evals + 1  # iteration index for β schedule (starts at 1)
            beta_t = self._beta(t)

            # -------------------------------------------------------------
            # 2.1 Generate Sobol candidates inside the current trust region
            # -------------------------------------------------------------
            n_cand = 2 ** 7  # 128 candidates – cheap yet enough diversity
            cand = self._candidate_in_region(n_cand, centre, radius_frac)

            # -------------------------------------------------------------
            # 2.2 Evaluate acquisition on candidates, pick the best as start
            # -------------------------------------------------------------
            acq_vals = np.apply_along_axis(self._ucb, 1, cand, gp, beta_t)
            start_idx = np.argmin(acq_vals)  # we minimise UCB
            x0 = cand[start_idx]

            # -------------------------------------------------------------
            # 2.3 Local refinement of the acquisition (L‑BFGS‑B)
            # -------------------------------------------------------------
            tr_bounds = self._trust_region_bounds(centre, radius_frac)

            res = minimize(fun=lambda x: self._ucb(x, gp, beta_t),
                           x0=x0,
                           bounds=tr_bounds,
                           method='L-BFGS-B',
                           options={'maxiter': 30, 'ftol': 1e-6})

            x_next = np.clip(res.x, self.lower, self.upper)

            # -------------------------------------------------------------
            # 2.4 Evaluate the true objective
            # -------------------------------------------------------------
            f_next = func(x_next)
            evals += 1

            # -------------------------------------------------------------
            # 2.5 Update data, GP and trust‑region state
            # -------------------------------------------------------------
            # Append new observation
            X = np.vstack([X, x_next])
            y = np.append(y, f_next)

            # Re‑fit GP (cheap for the small data sets typical in BBOB)
            gp = self._fit_gp(X, y)

            # Update best‑so‑far
            if f_next < y.min():
                centre = x_next
                success_counter += 1
                failure_counter = 0
            else:
                failure_counter += 1
                success_counter = 0

            # Adapt radius
            if success_counter >= self.success_patience:
                radius_frac = min(1.0, radius_frac * 1.5)   # expand
                success_counter = 0
            if failure_counter >= self.failure_patience:
                radius_frac = max(0.05, radius_frac * 0.5)  # contract
                failure_counter = 0

        # -----------------------------------------------------------------
        # 3. Return best point found
        # -----------------------------------------------------------------
        best_idx = np.argmin(y)
        best_x = X[best_idx]
        best_f = float(y[best_idx])
        return best_x, best_f
```

The class `AdaptiveTrustRegionBO` can be instantiated with a given evaluation budget and dimensionality, then called with any black‑box function conforming to the described interface.  It respects the budget, works for arbitrary dimensions, and leverages only the allowed scientific Python stack.
## Feedback
 The algorithm AdaptiveTrustRegionBO got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.4745 with standard deviation 0.0090.

took 96.49 seconds to run.