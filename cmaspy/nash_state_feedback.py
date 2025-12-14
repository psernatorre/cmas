# Import standard and third-party python packages
import numpy as np
from scipy.linalg import solve_continuous_are, eigvals, inv, block_diag
import cvxpy as cp

# Import packages from cmaspy
from cmaspy.utils import compute_eigenvalues, check_matrix_dims

def lyapunov_iteration_cares(A: np.ndarray,
                             B: list[np.ndarray],
                             Q: list[np.ndarray],
                             R: list[np.ndarray],
                             niter: int):

    print('+---------------------------------+')
    print('Lyapunov iterations to solve CAREs')
    print('+---------------------------------+ \n')

    # Check dimensions of matrices and format matrices if they are require
    A, B, Q, R = check_matrix_dims(A, B, Q, R)
    
    # Number of agents
    nag = len(B) 

    # List of matrices S, e.g, S[0] for 1st agent, S[1] for 2nd agent, etc.
    S = [B[i] @ inv(R[i]) @ B[i].T for i in range(nag)]

    # List of closed-loop matrix A respect to each agent.
    # First we initialize by assigning the matrix A to each agent
    # This list will be updated as iterations progress
    Acl = [A for _ in range(nag) ]

    # List to save matrix P and K for agents, 
    # e.g., P[0] for 1st agent, P[1] for 2nd agent
    P = [None] * nag
    F = [None] * nag

    # List to save the 
    CARE = [None] * nag
    CARE_abs_error = [None] * nag

    print(f"{'Iteration':<9}  {'Max(|CARE|)':<11}  {'Re(Lambda)':<10}")
    print(f"="*34)
    
    for i in range(niter):

        for j in range(nag):
            try:
                P[j] = solve_continuous_are(Acl[j], B[j], Q[j], R[j])
            except Exception as e:
                print(f"Error in solving ARE in iteration {i}, agent {j}: {e}")
                return None
    
        for j in range(nag):
            cum = 0
            for k in range(nag):
                if k != j:
                    cum = cum - S[k] @ P[k]
                
            Acl[j] = A + cum
            Acl_sys = A + cum - S[j] @ P[j]
            maxReEig = np.max(eigvals(Acl_sys).real)
            CARE[j] = Acl[j].T @ P[j] + P[j] @ Acl[j] - P[j] @ S[j] @ P[j] + Q[j] 
            CARE_abs_error[j] = np.max(np.abs(CARE[j]))
            F[j] = -inv(R[j]) @ B[j].T @ P[j] 

        print(f"{i :>9}  {np.max(CARE_abs_error):>11.2e}  {maxReEig:>10.2e}")
        print(f"-"*34)

    print('\n')
    print('Closed-loop system eigenvalues')
    Acl_eigvals = compute_eigenvalues(Acl_sys, show=True)

    return P, F, CARE, Acl_sys


def newton_raphson_cares(A: np.ndarray, 
                         B: list[np.ndarray], 
                         Q: list[np.ndarray], 
                         R: list[np.ndarray], 
                         niter: int):

    # Check dimensions of matrices, number of agents, and format matrices if they are require
    A, B, Q, R = check_matrix_dims(A, B, Q, R)
    
    # Number of agents
    nag = len(B) 

    # Number of states of the system
    nst = len(A)

    # List to save matrix P for agents and iterations
    # For example, if niter = 3 and nag = 2, then
    # P = [ [None, None, None], [None, None, None]]
    # To access P of 2nd agent of 3rd iteration: P[1][2]
    P = [[None for _ in range(niter)] for _ in range(nag)]
    

    # List to the matrix CARE
    CARE = [None for _ in range(nag)]
    maxCARE = [None for _ in range(nag)]
    F = [None for _ in range(nag)]
    Pfin = [None for _ in range(nag)]

    print('====================================')
    print('Newton-Raphson method to solve CAREs')
    print('==================================== \n')

    S = [B[i] @ inv(R[i]) @ B[i].T for i in range(nag)]

    print(f"{'Iteration':<9}  {'Solver status':<24}  {'Alpha':<9}  {'Max(|CARE|)':<11}  {'Re(Lambda)':<11}")
    print(f"="*58)

    for i in range(niter):

        if i == 0:

            # Initialize by solving LQR
            B_lqr = np.hstack(B)
            Q_lqr = np.sum(Q, axis=0)
            R_lqr = block_diag(*R)

            P_lqr = solve_continuous_are(A, B_lqr, Q_lqr, R_lqr)
            
            for j in range(nag):
                P[j][i] = P_lqr

            alpha = 0
            solver_status = 'optimal'

        else: 

            for j in range(nag):
                # Define stabilizing solution
                P[j][i] = cp.Variable((nst,nst), symmetric=True)

            # Define slack parameter
            alpha = cp.Variable()

            # Define list of constraints
            constraints = []

            for j in range(nag):
                sumx = 0
                for k in range(nag):
                    if k != j:
                        sumx = sumx + -P[k][i] @ S[k] @ P[j][i-1]

                LHS = (Acl.T @ P[j][i] + P[j][i] @ Acl 
                    + sumx + P[j][i-1] @ S[j] @ P[j][i-1] + Q[j])
        
                # The idea is to make LHS equal to zero because it is a CARE.
                # However, using ==0 could be very strict. Instead, we use the
                # parameter 'alpha'.
    
                for col in range(nst):
                    for row in range(col, nst):
                        constraints += [cp.abs(LHS[row,col]) <= alpha]
            
                constraints += [P[j][i] >> 0]
        
            constraints += [alpha >= 0]

            objective = cp.Minimize(alpha)
                
            prob = cp.Problem(objective, constraints)

            prob.solve()

            solver_status = prob.status

            alpha = alpha.value
            
            for j in range(nag):
                P[j][i] = P[j][i].value
                
        # State-matrix of the closed-loop system. This is then used in the next
        # iteration
        sumx = 0
        for j in range(nag):
            sumx = sumx - S[j] @ P[j][i]
    
        Acl = A + sumx;         

        # Maximum of the eigenvalue real parts. This is to check stability.
        maxReEig = np.max(eigvals(Acl).real)

        # Calculate the left-hand side of the CARE for each agent. In theory,
        # every CARE should be equal to zero.

        for k in range(nag):
            Pfin[k] = P[k][i]
            CARE[k] = Acl.T @ Pfin[k] + Pfin[k] @ Acl + Pfin[k] @ S[k] @ Pfin[k] + Q[k] 
            maxCARE[k] = np.max(np.abs(CARE[k]))
            F[k] = -inv(R[k]) @ B[k].T @ Pfin[k]
            
        print(f"{i :<9}  {solver_status :<24}  {alpha:<9.2e}  {np.max(maxCARE):<11.2e}  {maxReEig:<11.2e}")


    print('\n')
    print('Closed-loop system eigenvalues: ')

    Acl_eigvals = compute_eigenvalues(Acl, show=True)

    return Pfin, F, P, CARE, Acl