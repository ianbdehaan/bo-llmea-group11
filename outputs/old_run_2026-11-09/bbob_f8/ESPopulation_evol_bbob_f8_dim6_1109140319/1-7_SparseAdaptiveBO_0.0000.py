def rosenbrock(x):
    # classic Rosenbrock (sum of 100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2)
    return sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

bo = SparseAdaptiveBO(budget=200, dim=6, random_state=42)
best_val, best_x = bo(rosenbrock)
print("Best value:", best_val)
print("Best point:", best_x)
