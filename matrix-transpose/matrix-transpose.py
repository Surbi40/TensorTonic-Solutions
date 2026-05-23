import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)  #Converting list to numpy array
    #get original dimensions
    rows, cols = A.shape

    #create an empty matrix with swapped dimensions
    result = np.zeros((cols,rows),dtype=A.dtype)
    #fill the transpose matrix
    for i in range(rows):
        for j in range(cols):
            result[j][i]=A[i][j]
    return result

    
