import numpy as np
from scipy.linalg import eigvals
import pandas as pd

def check_matrix_dims(A: np.ndarray, B: list[np.ndarray], Q: list[np.ndarray], R: list[np.ndarray]):
    '''Function that check the dimensions of A, matrices in B, matrices in Q, matrices in R.
    Args: 
    ----
    A (numpy array): state matrix of the system
    B (list of numpy arrays): B[0] is the input matrix of 1st agent, B[1] is the input matrix of 2nd agent, and so forth.
    Q (list of numpy arrays): Q[0] is the input matrix of 1st agent, Q[1] is the input matrix of 2nd agent, and so forth.
    R (list of numpy arrays): R[0] is the input matrix of 1st agent, R[1] is the input matrix of 2nd agent, and so forth.
    
    Returns:
    -------
    A, B, Q, R: returns the inputs, with some additional format, if all tests are passed.
    
    '''

    B_len = len(B)
    Q_len = len(Q)
    R_len = len(R)

    # Convert all matrices to 2D-arrays
    # By doing this, we convert A=1 into a numpy array A = np.array([[1]])
    # This prevent errors when taking inv or other matrix operations 
    A = np.atleast_2d(A)
    B = [np.atleast_2d(B[j]) for j in range(B_len)]
    Q = [np.atleast_2d(Q[j]) for j in range(Q_len)]
    R = [np.atleast_2d(R[j]) for j in range(R_len)]

    assert B_len == Q_len == R_len, (f"The length of B, Q, R are different. \n"
                                     f"len(B) = {B_len}, len(Q) = {Q_len}, len(R) = {R_len}")

    nag = len(B)

    assert A.shape[0] == A.shape[1], "A must a square matrix."

    for j in range(nag):

        nrowsQ = Q[j].shape[0]
        ncolsQ = Q[j].shape[1]
        assert nrowsQ == ncolsQ, f"Q[{j}] is not a square matrix."

        dimA = A.shape[0] # number of rows of A
        dimQ = Q[j].shape[0] # number of rows of Q[j]
        assert dimA == dimQ, f"Dimension of Q[{j}] = {dimQ}, does not match dimension of A = {dimA} "

        nrowsB = B[j].shape[0]
        ncolsB = B[j].shape[1]
        nrowsR = R[j].shape[0]
        ncolsR = R[j].shape[1]
        assert nrowsB == dimA, f"Number of rows of B[{j}] = {nrowsB}, does not match dimension fo A = {dimA} "
        assert nrowsR == ncolsR, f"R[{j}] is not a square matrix."
        assert nrowsR == ncolsB, f"Dimension of R[{j}] = {nrowsR}, does not match number of columns of B[{j}] = {ncolsB}"



    return A, B, Q, R


def compute_eigenvalues(  A : np.ndarray, 
                          show : bool = False, 
                          print_settings : dict = {'index' : True, 
                                            'tablefmt': "grid",
                                            'numalign': "right", 
                                            'floatfmt': '.3f'}):
    '''Computes eigenvalues, natural frequency, damping ratio, time constant. It also has the option to display a
    pretty table when the function is executed.
    
    Args:
    ----
    A (numpy array): Matrix A of state-space model:
    
    show (Boolean): True (print table), False (do not print). By default is False.
    
    print_settings (dict): setting applied to tabulate package to print the pandas dataframe.

    Returns:
    -------

    df (Dataframe) : It contains eigenvalues, real, imag parts, natural frequency, damping ratio, and time constant.
    
    '''

    eigenvalues = eigvals(A)

    df = pd.DataFrame(data=eigenvalues, columns= ['eigenvalue'])
    df['real'] = df.apply(lambda row: row['eigenvalue'].real, axis=1)
    df['imag'] = df.apply(lambda row: row['eigenvalue'].imag, axis=1)
    df['natural_frequency'] = df.apply(lambda row: abs(row['eigenvalue']/(2*np.pi)), axis=1)
    df['damping_ratio'] =  df.apply(lambda row: -row['eigenvalue'].real/(abs(row['eigenvalue'])), axis=1)
    df['time_constant'] = df.apply(lambda row: -1/row['eigenvalue'].real, axis=1)
    df = df.sort_values(by='real', ascending=False, ignore_index=True)


    if show:
        df_to_print = df.copy()
        df_to_print = df_to_print[['real', 'imag', 'damping_ratio', 'natural_frequency', 'time_constant']]
        df_to_print.rename(columns={'real': 'Eigenvalue \n real part',
                                    'imag': 'Eigenvalue \n imaginary part',
                                    'damping_ratio': 'Damping \n ratio [p.u.]', 
                                    'natural_frequency': 'Natural \n frequency [Hz]',
                                    'time_constant': 'Time \n constant [s]'}, inplace=True)
        print(df_to_print.to_markdown(**print_settings))

    return df
