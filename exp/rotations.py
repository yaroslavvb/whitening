# Inverting random rotations using various optimization methods
# Accompanying notebook
# rotations.py
# https://www.wolframcloud.com/objects/f24e0b36-5340-4d82-9dd0-4aa25be2ac05

import numpy as np
import tensorflow as tf
import util as u
from util import t  # transpose
from util import c2v
from util import v2c
from util import v2r
from util import kr  # kroneckre
from util import Kmat # commutation matrix

dtype = np.float64

def simple_test():
  X0 = np.genfromtxt('data/rotations_simple_X0.csv',
                     delimiter= ",")
  W0f = np.genfromtxt('data/rotations_simple_W0f.csv',
                      delimiter= ",")
  fs = np.genfromtxt('data/rotations_simple_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  u.check_equal(fs, [10,2,2])

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  def Wdim(i): return f(i), f(i-1)  # dimension of W[i]
  
  dsize = X0.shape[1]
  assert f(-1) == dsize
  
  # load W0f and do shape checks (can remove)
  W0s = u.unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)
  Wdims = [Wdim(i) for i in range(1, n+1)]
  sizes = [dim[0] * dim[1] for dim in Wdims]
  Wf_size = np.sum(sizes)
  Wf_holder = tf.placeholder(dtype, shape=(Wf_size,))
  Wf = tf.Variable(Wf_holder, name="Wf")
  Wf_copy = tf.Variable(Wf_holder, name="Wf_copy")
  assert Wf.shape == W0f.shape
  init_dict = {Wf_holder: W0f}

  # Create W's
  W = u.unflatten(Wf, fs[1:])
  X = tf.constant(X0)
  W.insert(0, X)
  for (numpy_W, tf_W) in zip(W0s, W):
    u.check_equal(numpy_W.shape, u.fix_shape(tf_W.shape))

  # Create A's
  # A[1] == X
  # A[n+1] == Y
  A = [0]*(n+2)
  A[0] = u.Identity(dsize)
  for i in range(n+1):
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
  assert len(A) == n+2

  assert W[0].get_shape() == X0.shape
  assert A[n+1].get_shape() == X0.shape
  assert A[1].get_shape() == X0.shape

  err = X - A[n+1]
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(1.0, dtype=dtype)
  
  # Create B's
  B = [0]*(n+1)
  B[n] = -err/dsize
  Bn = [0]*(n+1)            # Newton-modified backprop
  Bn[n] = u.Identity(fs[-1])
  for i in range(n-1, -1, -1):
    B[i] = t(W[i+1]) @ B[i+1]
    Bn[i] = t(W[i+1]) @ Bn[i+1]

  # Create U's
  U = [list(range(n+1)) for _ in range(n+1)]
  for bottom in range(n+1):
    for top in range(n+1):
      prod = u.Identity(fs[top+1])
      for i in range(top, bottom-1, -1):
        prod = prod @ W[i]
      U[bottom][top] = prod

  # Block i, j gives hessian block between layer i and layer j
  blocks = [list(range(n+1)) for _ in range(n+1)]
  for i in range(1, n+1):
    for j in range(1, n+1):
      if i == j:
        blocks[i][j] = kr(A[i]@t(A[i]), Bn[i]@t(Bn[i]))/dsize
      elif i < j:
        blocks[i][j] = (kr(A[i]@t(A[j]), Bn[i]@t(Bn[j])) -
                       kr((A[i]@t(B[j])), U[i+1][j-1]) @ Kmat(f(j),f(j-1)))
      else:
        blocks[i][j] = (kr(A[i]@t(A[j]), Bn[i]@t(Bn[j])) -
                       kr(t(U[j+1][i-1]), B[i]@t(A[j])) @ Kmat(f(j),f(j-1)))
        
  # remove leftmost blocks (those are with respect to W[0] which is input)
  del blocks[0]
  for row in blocks:
    del row[0]
    
  hess = u.concat_blocks(blocks)
  ihess = u.pseudo_inverse(hess)

  # create dW's
  dW = [0]*(n+1)
  for i in range(n+1):
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
  del dW[0]  # get rid of W[0] update
  
  dWf = tf.concat([u.vectorize(grad) for grad in dW], axis=0)
  # TODO, get rid of vectors, make everything a column
  Wf_new = c2v(v2c(Wf) - lr*(ihess @ v2c(dWf)))  # Newton step
  
  train_op1 = Wf_copy.assign(c2v(Wf_new))
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/rotations_simple_losses_newton.csv",
                               delimiter= ",")
  expected_hess = np.loadtxt("data/rotations_simple_hess.csv",
                             delimiter= ",")
  observed_hess = sess.run(hess)
  u.check_equal(expected_hess, observed_hess)

  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  print(observed_losses)

  u.check_equal(observed_losses, expected_losses)

def simple2_test():
  X0 = np.genfromtxt('data/rotations_simple2_X0.csv',
                     delimiter= ",")
  W0f = np.genfromtxt('data/rotations_simple2_W0f.csv',
                      delimiter= ",")
  fs = np.genfromtxt('data/rotations_simple2_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  u.check_equal(fs, [10,2,2,2])

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  def Wdim(i): return f(i), f(i-1)  # dimension of W[i]
  
  dsize = X0.shape[1]
  assert f(-1) == dsize
  
  # load W0f and do shape checks (can remove)
  W0s = u.unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)
  Wdims = [Wdim(i) for i in range(1, n+1)]
  sizes = [dim[0] * dim[1] for dim in Wdims]
  Wf_size = np.sum(sizes)
  Wf_holder = tf.placeholder(dtype, shape=(Wf_size,))
  Wf = tf.Variable(Wf_holder, name="Wf")
  Wf_copy = tf.Variable(Wf_holder, name="Wf_copy")
  assert Wf.shape == W0f.shape
  init_dict = {Wf_holder: W0f}

  # Create W's
  W = u.unflatten(Wf, fs[1:])
  X = tf.constant(X0)
  W.insert(0, X)
  for (numpy_W, tf_W) in zip(W0s, W):
    u.check_equal(numpy_W.shape, u.fix_shape(tf_W.shape))

  # Create A's
  # A[1] == X
  # A[n+1] == Y
  A = [0]*(n+2)
  A[0] = u.Identity(dsize)
  for i in range(n+1):
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
  assert len(A) == n+2

  assert W[0].get_shape() == X0.shape
  assert A[n+1].get_shape() == X0.shape
  assert A[1].get_shape() == X0.shape

  err = X - A[n+1]
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(1.0, dtype=dtype)
  
  # Create B's
  B = [0]*(n+1)
  B[n] = -err/dsize
  Bn = [0]*(n+1)            # Newton-modified backprop
  Bn[n] = u.Identity(fs[-1])
  for i in range(n-1, -1, -1):
    B[i] = t(W[i+1]) @ B[i+1]
    Bn[i] = t(W[i+1]) @ Bn[i+1]

  # Create U's
  U = [list(range(n+1)) for _ in range(n+1)]
  for bottom in range(n+1):
    for top in range(n+1):
      prod = u.Identity(fs[top+1])
      for i in range(top, bottom-1, -1):
        prod = prod @ W[i]
      U[bottom][top] = prod

  # Block i, j gives hessian block between layer i and layer j
  blocks = [list(range(n+1)) for _ in range(n+1)]
  for i in range(1, n+1):
    for j in range(1, n+1):
      if i == j:
        blocks[i][j] = kr(A[i]@t(A[i]), Bn[i]@t(Bn[i]))/dsize
      elif i < j:
        blocks[i][j] = (kr(A[i]@t(A[j]), Bn[i]@t(Bn[j])) -
                       kr((A[i]@t(B[j])), U[i+1][j-1]) @ Kmat(f(j),f(j-1)))
      else:
        blocks[i][j] = (kr(A[i]@t(A[j]), Bn[i]@t(Bn[j])) -
                       kr(t(U[j+1][i-1]), B[i]@t(A[j])) @ Kmat(f(j),f(j-1)))
        
  # remove leftmost blocks (those are with respect to W[0] which is input)
  del blocks[0]
  for row in blocks:
    del row[0]
    
  hess = u.concat_blocks(blocks)
  ihess = u.pseudo_inverse(hess)

  # create dW's
  dW = [0]*(n+1)
  for i in range(n+1):
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
  del dW[0]  # get rid of W[0] update
  
  dWf = tf.concat([u.vectorize(grad) for grad in dW], axis=0)
  # TODO, get rid of vectors, make everything a column
  Wf_new = c2v(v2c(Wf) - lr*(ihess @ v2c(dWf)))  # Newton step
  
  train_op1 = Wf_copy.assign(c2v(Wf_new))
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/rotations_simple2_losses_newton.csv",
                               delimiter= ",")
  expected_hess = np.loadtxt("data/rotations_simple2_hess.csv",
                             delimiter= ",")
  observed_hess = sess.run(hess)
  u.check_equal(expected_hess, observed_hess)

  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  print(observed_losses)

  u.check_equal(observed_losses, expected_losses)


if __name__=='__main__':
  simple_test()
  print("All tests passed")
