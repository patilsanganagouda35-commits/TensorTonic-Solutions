import numpy as np

def matrix_transpose(A):
    # Convert to numpy array if input is a list
    A = np.array(A)
    
    # Get dimensions of the original matrix
    # N = rows, M = columns
    N, M = A.shape
    
    # Initialize a new matrix with swapped dimensions (M x N)
    transposed = np.zeros((M, N), dtype=A.dtype)
    
    # Iterate through each element to perform manual indexing
    for i in range(N):
        for j in range(M):
            # Mapping logic: A[i, j] -> A^T[j, i]
            transposed[j, i] = A[i, j]
    
    return transposed