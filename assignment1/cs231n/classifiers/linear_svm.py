import numpy as np
from random import shuffle
import time

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  #print(X.shape)
  
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    num_exceed_classes = 0
    
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        num_exceed_classes += 1
    #print(num_exceed_classes)    
    dW[:,y[i]] += (-1)*num_exceed_classes*X[i]
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  loss = 0.0
  dW = np.zeros(W.shape)  # initialize the gradient as zero
  scores = X.dot(W)  # shape is (N,10)
  num_train = X.shape[0]

  zero_matrix = np.zeros(scores.shape)
  scores_right = np.zeros(y.shape)
  scores_right_matrix = np.zeros(scores.shape)


  # compute loss
  for i in range(num_train):
    scores_right[i] = scores[i, y[i]] - 1
    scores_right_matrix[i, y[i]] = 1

  scores_right = np.expand_dims(scores_right, axis=1)

  scores_adj = scores - scores_right - scores_right_matrix
  zero_matrix = np.zeros(scores.shape)
  scores_adj = np.maximum(zero_matrix, scores_adj)
  loss = np.sum(scores_adj) # верно!

  #compute dW
  scores_adj_normed = scores_adj > 0

  scores_y = np.sum(scores_adj_normed, axis=1)
  scores_y = np.expand_dims(scores_y, axis=1)
  scores_y = scores_y * scores_right_matrix * (-1)

  a = scores_y + scores_adj_normed
  dW = X.T.dot(a)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  return loss, dW

