# Import standard and third-party python packages
import numpy as np
from scipy.linalg import solve_continuous_are, eigvals, inv, block_diag, sqrtm
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

def mas_output_feedback(A: np.ndarray, B: list[np.ndarray], C: list[np.ndarray], D: list[np.ndarray], 
                        Q: list[np.ndarray], R: list[np.ndarray], P: list[np.ndarray], 
                        alpha_coef: float, beta_coef: float, gamma_coef: float, **cvxpy_solve_settings):
    
    print('+-------------------------------------------------------------------+')
    print('Synthesis of partial-state feedback control for a multi-agent system')
    print('+-------------------------------------------------------------------+ \n')

    # Check dimensions of matrices, number of agents, and format matrices if they are require
    A, B, Q, R = check_matrix_dims(A, B, Q, R)
    
    # Number of agents
    nag = len(B) 

    # Number of states of the system
    n = len(A)

    # Dimension of matrix B
    m = np.hstack(B).shape[1]

    print(f'Number of states: {n}')
    print(f'Number of inputs: {m}')
    print(f'Number of agents: {nag}')

    # List to save matrix F for agents
    F = [None for _ in range(nag)]

    for j in range(nag):
        ncolsB = B[j].shape[1]
        nrowsC = C[j].shape[0]
        F[j] =  cp.Variable((ncolsB, nrowsC))
    
    # Lists to save alpha, beta and gamma
    alpha = [cp.Variable() for _ in range(nag)]
    beta = [cp.Variable() for _ in range(nag)]
    gamma = [cp.Variable() for _ in range(nag)]

    # Define auxiliary matrices
    Gamma = [None for _ in range(nag)]
    Upsilon = [None for _ in range(nag)]
    B_bar = [None for _ in range(nag)]
    D_bar = [None for _ in range(nag)]
    R_bar = [None for _ in range(nag)]

    for j in range(nag):
        sum1 = 0
        sum2 = 0
        D_bar[j] = []
        R_bar[j] = []
        
        for k in range(nag):
            if k != j:
                sum1 = sum1 + B[k] @ inv(R[k]) @ B[k].T @ P[k]
                D_bar[j].append( P[k] @ B[k] @ F[k] @ D[k] )
                R_bar[j].append( R[k] )
            
            sum2 = sum2 + B[k] @ F[k] @ C[k]

        D_bar[j] = np.hstack(D_bar[j])
        R_bar[j] = block_diag(*R_bar[j])
        Gamma[j] = Q[j] - P[j] @ ( sum1 + sum2 ) - ( sum1 + sum2 ).T @ P[j]
        Upsilon[j] =  P[j] @ B[j] @ ( F[j] @ D[j] @ inv(R[j]) + inv(R[j]) @ D[j].T @ F[j].T ) @ B[j].T @ P[j]
        B_bar[j] = B[j] @ ( np.eye(B[j].shape[1]) + F[j] @ D[j] )

    # Define list of constraints
    constraints = []

    for j in range(nag):
        M = np.zeros((B[j].shape[1], D_bar[j].shape[1]))
        constraints += [cp.bmat([   [Gamma[j] + Upsilon[j],     P[j] @ B_bar[j],    D_bar[j]],
                                    [B_bar[j].T @ P[j],         R[j],               M ],
                                    [D_bar[j.T],                M.T,               R_bar[j]] ] >> alpha[j]*np.eye(n + m)) ]
        M = sqrtm(R[j]) @ B[j].T @ P[j] + sqrtm(R[j]) @ F[j] @ C[j]
        constraints += [cp.bmat([   [beta[j] @ np.eye(B[j].shape[1]),   M],
                                    [M.T,                               beta[j] @ np.eye(n)]]) >> 0]
        
        M = sqrtm(R[j]) * F[j] * D[j]
        constraints += [cp.bmat([   [gamma[j] @ np.eye(B[j].shape[1]),  M],
                                    [M.T,                               gamma[j] @ np.eye(B[j].shape[1])]]) >> 0]
    
    sum_obj = 0
    for j in range(nag):
         sum_obj  += -alpha_coef * alpha[j] + beta_coef * beta[j] + gamma_coef * gamma[j]
    
    # Define objective function
    objective = cp.Minimize( sum_obj )

    # Define optimization problem
    prob = cp.Problem(objective, constraints)

    # Solve problem
    prob.solve(**cvxpy_solve_settings)

    solver_status = prob.status
    sum_obj =  sum_obj.value

    print(f'SDP status: {solver_status}')
    print(f'Obj. value: {sum_obj}')

    for j in range(nag):
        F[j] = F[j].value
        alpha[j] = alpha[j].value
        beta[j] = beta[j].value
        gamma[j] = gamma[j].value


    Acl_F = A 
    
    for j in range(nag):
        Acl_F += B[j] @ F[j] @ C[j]
    

    print('Closed-loop system eigenvalues: ')

    Acl_eigvals = compute_eigenvalues(Acl_F, show=True)

    return F, alpha, beta, gamma, prob, Acl_F