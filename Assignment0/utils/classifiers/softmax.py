import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    num_train = X.shape[0]
    num_class = W.shape[1]
    dW_each = np.zeros_like(W)
    scores = np.dot(X,W)
    scores_max = np.reshape(np.max(scores, axis=1), (num_train, 1))   # N by 1
    prob = np.exp(scores - scores_max) / np.sum(np.exp(scores - scores_max), axis=1, keepdims=True) # N by C
    
    true_class = np.zeros_like(prob)
    true_class[np.arange(num_train), y] = 1.0
    
    dW_each = np.zeros_like(W)
    for i in range(num_train):
        for j in range(num_class):
            loss += -(true_class[i, j] * np.log(prob[i, j]))
            dW_each[:, j] = -(true_class[i, j] - prob[i, j]) * X[i, :]
        dW += dW_each    
    
    loss /= num_train
    loss += reg*np.sum(W*W)
    
    dW /= num_train
    dW += 2*reg*W


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

 
    num_train = X.shape[0]
    
    scores = np.dot(X,W)
    scores_max = np.reshape(np.max(scores, axis=1), (num_train, 1))   # N by 1
    prob = np.exp(scores - scores_max) / np.sum(np.exp(scores - scores_max), axis=1, keepdims=True)
    true_class = np.zeros_like(prob)
    true_class[range(num_train), y] = 1.0
    
    loss += -np.sum(true_class * np.log(prob)) / num_train + reg * np.sum(W * W)
    dW += -np.dot(X.T, true_class - prob) / num_train + 2 * reg * W
    


    return loss, dW
