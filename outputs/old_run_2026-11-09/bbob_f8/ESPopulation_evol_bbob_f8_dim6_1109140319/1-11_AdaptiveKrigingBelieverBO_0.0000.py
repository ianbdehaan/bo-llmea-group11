optimizer = AdaptiveKrigingBelieverBO(budget=200, dim=6)
best_val, best_x = optimizer(rosenbrock_function)   # user‑provided callable
