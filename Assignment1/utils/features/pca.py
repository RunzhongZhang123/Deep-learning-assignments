import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """

    ###############################################
    # TODO: Implement PCA by extracting        #
    # eigenvector.You may need to sort the      #
    # eigenvalues to get the top K of them.     #
    ###############################################
    ###############################################
    #          START OF YOUR CODE         #
    ###############################################
    N, D = X.shape
    P = np.zeros([K, D])
    T = []
    M = np.mean(X.T, axis=1)

    C = X - M

    V = np.corrcoef(C.T)
    # print(V.shape)

    value, vector = np.linalg.eig(V)
    # print(vector.shape)
    # print(vector)
    # print(value.shape)

    pair = [(np.abs(value[i]), vector[:,i]) for i in range(K)]
    pair.sort(key=lambda x: x[0], reverse=True)
    for i in range(K):
        P[i, :] = pair[i][1]
        T.append(pair[i][0])

    ###############################################
    #           END OF YOUR CODE         #
    ###############################################
    return (P, T)