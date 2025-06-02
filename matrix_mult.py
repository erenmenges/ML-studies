import numpy as np

A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

def mult(pA, pB):
    C = np.zeros((pA.shape[0],pB.shape[1]))
    for i, row in enumerate(C):
        for j, singular_row_data in enumerate(row):
            sum1 = 0
            for k, original_singular_row_data in enumerate(pA[i]):
                sum1 += original_singular_row_data * (pB.T)[j][k]
            C[i][j] = sum1
    return C

