import numpy as np
from random import shuffle
import tensorflow as tf

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    C = W.shape[1]
    N = X.shape[0]
    for i in range(N):
        SUM = np.dot(X[i], W) 
        SUM = SUM - SUM.max()
        ans = np.sum(np.exp(SUM))
        tmp = np.exp(SUM[y[i]])
        loss += - np.log(tmp / ans)

        dW[:, y[i]] += (-1) * (ans - tmp) / ans * X[i]
        for j in range(C):
            if j == y[i]:
                continue
            dW[:, j] += np.exp(SUM[j]) / ans * X[i]

    loss = loss / N + reg * np.sum(W * W)
    dW = dW / N + 2 * reg * W
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    D = W.shape[1]
    N = X.shape[0]
    Y = np.dot(X, W)
    Y -= Y.max()
    Y = np.exp(Y)

    SUM = np.sum(Y, axis=1)
    ans = Y[range(N), y]
    loss = ans / SUM
    loss = -np.sum(np.log(loss))/N + reg * np.sum(W * W)

    tmp = np.divide(Y, SUM.reshape(N, 1))
    tmp[range(N), y] = - (SUM - ans) / SUM
    dW = X.T.dot(tmp) / N + 2 * reg * W
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW


def softmax_loss_tf(W, X, y, reg):
    W_tf = tf.placeholder(tf.float32, shape=(3073,10))
    X_tf = tf.placeholder(tf.float32, shape=(None, 3073))
    y_tf = tf.placeholder(tf.int32, shape=(None,))
    init_op = tf.global_variables_initializer()

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits= tf.matmul(X_tf, W_tf), labels=tf.one_hot(y_tf,10))
    loss0 = tf.reduce_mean(cross_entropy) + reg*tf.reduce_sum(W_tf*W_tf)
    grad0 = tf.gradients(loss0, W_tf)
    out0 = (loss0, grad0)
    with tf.Session() as sess:
        sess.run(init_op)
        loss_gt, grad_gt = sess.run(out0, feed_dict={W_tf: W, X_tf: X, y_tf: y})
    return loss_gt, grad_gt