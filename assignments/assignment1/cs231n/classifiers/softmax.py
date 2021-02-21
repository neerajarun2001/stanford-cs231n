from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Some variables we'll need.
    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
      
      # Forward pass to get class scores.
      scores = X[i].dot(W)

      # Normalization trick for numeric stability.
      scores -= np.amax(scores)

      # Exponentiate the scores so we can calc softmax.
      exp_scores = np.exp(scores)

      # Nonvectorized summation of e^f_j.
      sum_scores = 0.0
      for j in range(num_class):
        sum_scores += exp_scores[j] 
        
      # Initialize softmax -- need softmax of all scores.
      softmax = np.zeros(num_class)

      # Iterate through again to calc loss and gradient.
      for j in range(num_class):

        # Find softmax for jth score.
        softmax[j] = exp_scores[j] / sum_scores

        # Two versions of gradient.
        # Only the y_ith score contributes to the loss
        # because softmax is trying to maximize y_ith score. 
        if y[i] == j:
          loss -= np.log(softmax[j])
          dW[:, y[i]] += (softmax[j] - 1) * X[i]
        else: 
          dW[:, j] += softmax[j] * X[i]

    # Average
    loss /= num_train
    dW /= num_train
      
    # Regularization term
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Some variables we'll need.
    num_train = X.shape[0]
    num_class = W.shape[1]

    # Calc scores (500, 10) -- row i is vector of scores for X_i
    scores = X.dot(W)

    scores -= np.reshape(np.amax(scores, axis=1), (num_train, 1))

    exp_scores = np.exp(scores)
    sum_scores = np.sum(exp_scores, axis=1)

    softmax = exp_scores / sum_scores[:, None]

    log_softmax = -1 * np.log(softmax)
    loss += np.sum(log_softmax[np.arange(num_train), y])

    for i in range(num_train):
      for j in range(num_class):
        if j == y[i]:
          dW[:, y[i]] += (softmax[i, j] - 1) * X[i]
        else: 
          dW[:, j] += softmax[i, j] * X[i]
    


    # Average
    loss /= num_train
    dW /= num_train

    # Regularization term
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
