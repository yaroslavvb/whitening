import tensorflow as tf
import numpy as np
import util as u
import sys


def test_numpy(X):
  dsize = X.shape[1]
  cov = X @ X.T
  #  cov = cov/np.max(cov)
  precision = np.linalg.inv(cov)
  
  n = cov.shape[0]
  B = np.zeros((4,4))
  lr = 0.01
  losses = []
  for i in range(100):
    R = B @ X - X
    G = 2 * R @ X.T
    np.fill_diagonal(G, 0)

    resvar = np.asarray([np.linalg.norm(r)**2 for r in R])
    losses.append(np.sum(resvar))
    D2 = np.diag(1/resvar)
    precision2 = D2 @ (np.identity(n) - B)
    
    err = (precision2 - precision)
    loss2 = np.trace(err @ err.T)
    B = B - lr * G
    print(loss2)

  test_points = 10
  losses = np.asarray(losses)[:test_points]
  target_losses = [118., 41.150800000000004, 33.539355199999996,29.747442032320002, 27.450672271574934, 25.95846376879459,24.917943341139274, 24.139761502111114, 23.519544126307142,22.998235729589265]

  u.check_equal(losses[:test_points], target_losses[:test_points])
  print('mismatch is ', np.max(losses-target_losses))
    
if __name__=='__main__':
  numbers=[(x+1)**3 for x in range(16)]
  list(u.chunks(numbers, 4))
  X = np.array(list(u.chunks(numbers, 4)))
  
  X = np.asarray([[5, 1, 0, 4], [0, 4, 1, 2], [1, 0, 3, 3], [4, 2, 0, 4]])
  test_numpy(X)

