import numpy as np
from random import shuffle

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  scores = X.dot(W) 
  
  for i in range(num_train):
    f = scores[i]
    f -= np.max(f)
    p = np.exp(f) / np.sum(np.exp(f))
    loss -= np.log(p[y[i]])
    
    for j in range(num_classes):
        if j == y[i]:
          dW[:,j] += -X[i]*(1 -p[j]) 
        else:
          dW[:,j] += X[i]*p[j]
          
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W
    
    

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
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
  num_train = X.shape[0]
  
  scores = X.dot(W)
  
  scores_right = np.zeros(y.shape)
  scores_right_matrix = np.zeros(scores.shape)
  
  maxs = np.max(scores, axis=1)
  maxs = np.expand_dims(maxs, axis=1)
  scores -= maxs
  sums = np.sum(np.exp(scores), axis = 1)
  sums = np.expand_dims(sums, axis=1)
  scores = np.exp(scores)/ sums #scores is a matrix of probabilities for picture i to be in class j 
  
  
  for i in range(num_train):
    scores_right[i] = scores[i, y[i]]
    scores_right_matrix[i, y[i]] = 1
  
  loss = -np.sum(np.log(scores_right))
  
  scores_modified = scores - scores_right_matrix
  dW = X.T.dot(scores_modified)
  
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

