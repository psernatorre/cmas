""""
This example is extracted from Section IV of the paper: From LQR to Static Output Feedback: A New LMI Approach, Luis Rodrigues, 2022
"""
# Import standard and third-party python packages
import numpy as np
from scipy.linalg import solve_continuous_are, eigvals, block_diag
import cvxpy as cp

# Import packages from cmaspy
from cmaspy.partial_state_feedback import single_agent_output_feedback, mas_output_feedback

# Numerical matrices
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

# Solve CARE to obtain P
P = solve_continuous_are(A, B, Q, R)

# Use Rodriguez approach
F = single_agent_output_feedback(A, B, C, D, Q, R, P, **solve_settings)

# Use CMASPy MAS output feedback
alpha_coef = 1000
beta_coef = 0
gamma_coef = 0
mas_out = mas_output_feedback(A, [B], [C], [D], [Q], [R], [P], alpha_coef, beta_coef, gamma_coef, **solve_settings)
print(F)
print(mas_out.F[0])