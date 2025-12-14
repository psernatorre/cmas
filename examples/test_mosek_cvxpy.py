import cvxpy as cp
import numpy as np

# Problem data
m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

# Solve the problem using MOSEK with custom parameters
settings = {
    'solver': cp.MOSEK,
    'verbose': True,
}
result = prob.solve(**settings)

print(f"Optimal objective value: {result}")
print(f"Optimal variable x: {x.value}")
