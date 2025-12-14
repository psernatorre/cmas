import numpy as np
from scipy.linalg import solve_continuous_are, eigvals, block_diag

from cmaspy.partial_state_feedback import single_agent_output_feedback
import cvxpy as cp

A = np.array([  [0, 1, 0],
                [0, 1, 1],
                [0, 13, 0]])

B =  np.array([ [0], 
                [0], 
                [1]])

C = np.array([[0,   5, -1],
              [-1, -1,  0]])

D = np.zeros((C.shape[0], B.shape[1]))

Q = block_diag(1, 3, 0.1)
R = 10**(-4)

solve_settings = {'solver': cp.MOSEK,
                  'verbose': False}

P = solve_continuous_are(A, B, Q, R)
F = single_agent_output_feedback(A, B, C, D, Q, R, P, **solve_settings)

print(F)


A = np.array([ [-0.0366, 0.0271, 0.0188, -0.4555],
               [0.0482, -1.0100, 0.0024, -4.0208],
               [0.1002, 0.3681, -0.7070, 1.4200],
               [0,          0,      1,      0]])

B = np.array([[0.4422, 0.1761],
              [3.5446, -7.5922],
              [-5.52, 4.49],
              [0, 0]])

C = np.array([[0, 1, 0, 0]])

D = np.zeros((C.shape[0], B.shape[1]))

Q = np.eye(4)
R = np.eye(2)

P = solve_continuous_are(A, B, Q, R)
F = single_agent_output_feedback(A, B, C, D, Q, R, P, **solve_settings)

print(F)

print('Example completed')