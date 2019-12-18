import numpy as np
from random import shuffle

def sigmoid(x):
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                            #         
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    h = 1 / (1 + np.exp(-1 * x))
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################
    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = None
    # Initialize the gradient to zero
    dW = None

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    N = X.shape[0]
    D = X.shape[1]
    # Y = np.zeros([N, 2])
    Y1 = np.zeros([N, ])
    Y0 = np.zeros([N, ])
    for i in range(N):
        for j in range(D):
            Y1[i, ] += X[i, j] * W[j, 1]
        Y1[i, ] = sigmoid(Y1[i, ])
        Y0[i, ] = 1 - Y1[i, ]
        # Y[i, 0] = Y0[i, 0]
        # Y[i, 1] = Y1[i, 0]

    loss = 0
    for i in range(N):
    	loss += -1 * y[i] * np.log(Y1[i]) - (1 - y[i]) * np.log(Y0[i])
    loss /= N
    loss += 0.5 * reg * np.sum(W[:, 1] * W[:, 1])

    dW = np.zeros([W.shape[0], 1])
    for i in range(D):
    	for j in range(N):
    		dW[i, 0] += X.T[i, j] * (Y1[j, ] - y[j])
    	dW[i, 0] = dW[i, 0] / N + reg * W[i, 1]

    # pre = np.matmul(X, W)
    # pre = sigmoid(pre)
    # Label = []
    # for i in range(X.shape(0)):
    #     M = np.amax(pre[i,:])
    #     Label.append(int(np.where(pre[i,:] == np.amax(pre[i,:]))))

    # loss = 0
    # for i in range(X.shape(0)):
    #     loss += Label[i] * np.log(pre[i,1]) + (1 - Label[i]) * np.log(1 - pre[i,1])

    # reg_term  = 0
    # for i in range(W.shape(0)):
    #     for j in range(W.shape(1)):
    #         reg_term += W[i,j]**2
    # reg_term = np.sqrt(reg_term)

    # loss = loss / X.shape(0) + reg * reg_term
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = None
    # Initialize the gradient to zero
    dW = None

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no    # 
    # explicit loops.                                       #
    # Store the loss in loss and the gradient in dW. If you are not careful   #
    # here, it is easy to run into numeric instability. Don't forget the     #
    # regularization!                                       #
    ############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    Y = np.dot(X, W)[:, 1]
    Y1 = sigmoid(Y)
    Y0 = 1 - Y1
    loss = (-y * np.log(Y1) - (1 - y) * np.log(Y0)).mean() + 0.5 * reg * np.sum(W[:, 1] * W[:, 1])

    dW = np.dot(X.T, (sigmoid(Y) - y)) / X.shape[0] + reg * W[:, 1]
    dW = dW.reshape([W.shape[0], 1])
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW
