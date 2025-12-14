# Import standard and third-party python packages
import numpy as np
from scipy.linalg import solve_continuous_are, eigvals, inv, block_diag
import cvxpy as cp

# Import packages from cmaspy
from cmaspy.utils import compute_eigenvalues, check_matrix_dims

def single_agent_output_feedback(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                                 Q: np.ndarray, R: np.ndarray, P: np.ndarray,
                                 **cvxpy_solve_settings):
    
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    C = np.atleast_2d(C)
    D = np.atleast_2d(D)
    Q = np.atleast_2d(Q)
    R = np.atleast_2d(R)

    n = A.shape[0]
    m = B.shape[1]
    ny = C.shape[0]

    # Define matrix F
    F = cp.Variable((m,ny))

    # Define parameter alpha
    alpha = cp.Variable()

    # Define auxiliary variables
    N = P @ B @ (F @ D @ inv(R) + inv(R) @ D.T @ F.T) @ B.T @ P
    B_bar = B @ (np.eye(m) + F @ D)

    # Define constraints
    constraints = [ cp.bmat([[Q - P @ B @ F @ C - C.T @ F.T @ B.T @ P + N,     P @ B_bar],
                                [B_bar.T @ P,                                     R ] ] ) >> alpha * np.eye(n+m)]

    # Define objective function
    objective = cp.Maximize(alpha)

    # Define optimization problem
    prob = cp.Problem(objective, constraints)

    # Solve problem
    prob.solve(**cvxpy_solve_settings)

    solver_status = prob.status

    alpha = alpha.value
    F = F.value

    print(f'SDP status: {solver_status}')
    print(f'Obj. value: {alpha}')

    Acl = A + B @ F @ C

    print('Closed-loop system eigenvalues: ')

    Acl_eigvals = compute_eigenvalues(Acl, show=True)
    
    return F


