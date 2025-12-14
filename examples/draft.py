import numpy as np
from scipy.linalg import solve_continuous_are, eigvals

from cmaspy.nash_state_feedback import lyapunov_iteration_cares, newton_raphson_cares

A = np.array([[-1, 0],
              [0, -1]])
nag = 2
B = [None] * 2
B[0] = np.array([[1],
                 [0]])
B[1] = np.array([[0],
                 [1]])


Q = [None] * nag
Q[0] = 2*np.eye(2)
Q[1] = np.eye(2)

R = [None] * nag
R[0] = 1
R[1] = 2

niter=3
P, K, CARE, Acl_sys = lyapunov_iteration_cares(A, B, Q, R, niter)
Pfin, F, P, CARE, Acl = newton_raphson_cares(A, B, Q, R, niter)

X = np.eye(3)
for row in range(3):
    for col in range(row, 3):
        X[row, col] = 20

print('ok')