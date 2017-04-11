# Inverting random rotations using various optimization methods
# Accompanying notebook
# 
# rotations.py
# https://www.wolframcloud.com/objects/d285dcfb-2c89-4ea9-9877-a79bba07192f
#
# rotations_relu.py
# https://www.wolframcloud.com/objects/3a204c89-5dfe-410f-9515-f7bf55e180bc

import numpy as np
import os
import sys
import tensorflow as tf
import util as u
from util import t  # transpose
from util import c2v
from util import v2c
from util import v2c_np
from util import v2r
from util import kr  # kronecker
from util import Kmat # commutation matrix

dtype = np.float64

def simple_gradient_test():
  tf.reset_default_graph()
  X0 = np.genfromtxt('data/rotations_simple_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/rotations_simple_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/rotations_simple_W0f.csv',
                            delimiter= ","))
  assert W0f.shape == (8, 1)
  
  fs = np.genfromtxt('data/rotations_simple_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  u.check_equal(fs, [10,2,2,2])

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = X0.shape[1]
  assert f(-1) == dsize
  
  # load W0f and do shape checks (can remove)
  W0s = u.unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)
  Wf_holder = tf.placeholder(dtype, shape=W0f.shape)
  Wf = tf.Variable(Wf_holder, name="Wf")
  Wf_copy = tf.Variable(Wf_holder, name="Wf_copy")
  init_dict = {Wf_holder: W0f}

  # Create W's
  W = u.unflatten(Wf, fs[1:])
  X = tf.constant(X0)
  Y = tf.constant(Y0)
  W.insert(0, X)
  for (numpy_W, tf_W) in zip(W0s, W):
    u.check_equal(numpy_W.shape, u.fix_shape(tf_W.shape))

  # Create A's
  # A[1] == X
  A = [0]*(n+2)
  A[0] = u.Identity(dsize)
  for i in range(n+1):
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))


  assert W[0].get_shape() == X0.shape
  assert A[n+1].get_shape() == X0.shape
  assert A[1].get_shape() == X0.shape

  err = Y - A[n+1]
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(1.0, dtype=dtype)
  
  # Create B's
  B = [0]*(n+1)
  B[n] = -err/dsize
  for i in range(n-1, -1, -1):
    B[i] = t(W[i+1]) @ B[i+1]

  # create dW's
  dW = [0]*(n+1)
  for i in range(n+1):
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
  del dW[0]  # get rid of W[0] update
  
  dWf = tf.concat([u.vec(dWi) for dWi in dW], axis=0)
  Wf_new = Wf - lr * dWf 

  train_op1 = Wf_copy.assign(Wf_new)
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/rotations_simple_gradient_losses.csv",
                               delimiter= ",")
  observed_losses = []
  # from accompanying notebook
  # {0.0111498, 0.00694816, 0.00429464, 0.00248228, 0.00159361,
  #  0.000957424, 0.000651653, 0.000423802, 0.000306749, 0.00021772,
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  u.check_equal(observed_losses, expected_losses)


def simple_newton_test():
  tf.reset_default_graph()
  X0 = np.genfromtxt('data/rotations_simple_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/rotations_simple_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/rotations_simple_W0f.csv',
                            delimiter= ","))
  assert W0f.shape == (8, 1)
  
  fs = np.genfromtxt('data/rotations_simple_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  u.check_equal(fs, [10,2,2,2])

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = X0.shape[1]
  assert f(-1) == dsize
  
  # load W0f and do shape checks (can remove)
  W0s = u.unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)
  Wf_holder = tf.placeholder(dtype, shape=W0f.shape)
  Wf = tf.Variable(Wf_holder, name="Wf")
  Wf_copy = tf.Variable(Wf_holder, name="Wf_copy")
  init_dict = {Wf_holder: W0f}
  
  # Create W's
  W = u.unflatten(Wf, fs[1:])
  X = tf.constant(X0)
  Y = tf.constant(Y0)
  W.insert(0, X)
  for (numpy_W, tf_W) in zip(W0s, W):
    u.check_equal(numpy_W.shape, u.fix_shape(tf_W.shape))

  # Create A's
  # A[1] == X
  A = [0]*(n+2)
  A[0] = u.Identity(dsize)
  for i in range(n+1):
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))

  assert W[0].get_shape() == X0.shape
  assert A[n+1].get_shape() == X0.shape
  assert A[1].get_shape() == X0.shape

  err = Y - A[n+1]
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(1.0, dtype=dtype, name="learning_rate")
  
  # Create B's
  B = [0]*(n+1)
  B[n] = -err/dsize
  Bn = [0]*(n+1)            # Newton-modified backprop
  Bn[n] = u.Identity(f(n))
  for i in range(n-1, -1, -1):
    B[i] = t(W[i+1]) @ B[i+1]
    Bn[i] = t(W[i+1]) @ Bn[i+1]

  # Create U's
  U = [list(range(n+1)) for _ in range(n+1)]
  for bottom in range(n+1):
    for top in range(n+1):
      if bottom > top:
        prod = u.Identity(f(top))
      else:
        prod = u.Identity(f(bottom-1))
        for i in range(bottom, top+1):
          prod = prod@t(W[i])
      U[bottom][top] = prod

  # Block i, j gives hessian block between layer i and layer j
  blocks = [list(range(n+1)) for _ in range(n+1)]
  for i in range(1, n+1):
    for j in range(1, n+1):
      term1 = kr(A[i] @ t(A[j]), Bn[i] @ t(Bn[j])) / dsize;
      if i == j:
        term2 = tf.zeros((f(i)*f(i-1), f(i)*f(i-1)), dtype=dtype)
      elif i < j:
        term2 = kr(A[i] @ t(B[j]), U[i+1][j-1])
      else:
        term2 = kr(t(U[j+1][i-1]), B[i] @ t(A[j]))
        
      blocks[i][j]=term1 + term2 @ Kmat(f(j), f(j-1))

        
  # remove leftmost blocks (those are with respect to W[0] which is input)
  del blocks[0]
  for row in blocks:
    del row[0]
    
  hess = u.concat_blocks(blocks)
  ihess = u.pseudo_inverse(hess)
  
  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

  expected_hess = np.loadtxt("data/rotations_simple_newton_hess0.csv",
                              delimiter= ",")
  observed_hess = sess.run(hess)
  np.savetxt("data/rotations_observed_hess.csv", observed_hess,
             fmt="%.10f", delimiter=',')
  expected_ihess = np.loadtxt("data/rotations_simple_newton_ihess0.csv",
                              delimiter= ",")

  observed_ihess = sess.run(ihess)
  u.check_equal(expected_hess, observed_hess)


  # create dW's
  dW = [0]*(n+1)
  for i in range(n+1):
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
  del dW[0]  # get rid of W[0] update
  
  dWf = tf.concat([u.vec(dWi) for dWi in dW], axis=0)
  Wf_new = Wf - lr * ihess @ dWf 

  train_op1 = Wf_copy.assign(Wf_new)
  train_op2 = Wf.assign(Wf_copy)

  
  expected_losses = np.loadtxt("data/rotations_simple_newton_losses.csv",
                               delimiter= ",")
  observed_losses = []
  
  # from accompanying notebook
  # 0.0111498, 0.0308658, 0.00462571, 0.0000251229, 1.38508*10^-9, 
  # 1.32383*10^-17, 2.39119*10^-31
  
  for i in range(10):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  u.check_equal(observed_losses, expected_losses)


def relu_gradient_test():
  tf.reset_default_graph()
  X0 = np.genfromtxt('data/rotations_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/rotations_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/rotations_W0f.csv',
                            delimiter= ","))
  assert W0f.shape == (8, 1)
  
  fs = np.genfromtxt('data/rotations_relu_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  u.check_equal(fs, [4,2,2,2])

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = X0.shape[1]
  assert f(-1) == dsize
  
  # load W0f and do shape checks (can remove)
  W0s = u.unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)
  Wf_holder = tf.placeholder(dtype, shape=W0f.shape)
  Wf = tf.Variable(Wf_holder, name="Wf")
  Wf_copy = tf.Variable(Wf_holder, name="Wf_copy")
  init_dict = {Wf_holder: W0f}

  # Create W's
  W = u.unflatten(Wf, fs[1:])
  X = tf.constant(X0, name="X0")
  Y = tf.constant(Y0, name="Y0")
  W.insert(0, X)
  for (numpy_W, tf_W) in zip(W0s, W):
    u.check_equal(numpy_W.shape, u.fix_shape(tf_W.shape))

  # Create A's
  # A[1] == X
  A = [0]*(n+2)
  A[0] = u.Identity(dsize)
  for i in range(n+1):
    if i == 0:
      A[i+1] = X
    else:
      A[i+1] = tf.nn.relu(tf.matmul(W[i], A[i], name="A"+str(i+1)))

  assert W[0].get_shape() == X0.shape
  assert A[n+1].get_shape() == X0.shape
  assert A[1].get_shape() == X0.shape

  err = Y - A[n+1]
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.1, dtype=dtype)
  
  # Create B's
  B = [0]*(n+1)
  B[n] = (-err/dsize)*u.relu_mask(A[n+1])
  for i in range(n-1, -1, -1):
    B[i] = t(W[i+1]) @ B[i+1]
    if i > 0:  # there's no relu on first matrix
      B[i] = B[i]*u.relu_mask(A[i+1])

  # create dW's
  dW = [0]*(n+1)
  for i in range(n+1):
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
  del dW[0]  # get rid of W[0] update
  
  dWf = tf.concat([u.vec(dWi) for dWi in dW], axis=0)
  Wf_new = Wf - lr * dWf 

  train_op1 = Wf_copy.assign(Wf_new)
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/rotations_relu_gradient_losses.csv",
                               delimiter= ",")
  observed_losses = []
  
  # From accompanying notebook
  #  {0.407751, 0.0683822, 0.0138657, 0.0039221, 0.00203637, 0.00164892,
  #    0.00156137, 0.00153857, 0.00153051, 0.00152593}
  for i in range(10):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  u.check_equal(observed_losses, expected_losses)
  
def relu_newton_test():
  tf.reset_default_graph()
  X0 = np.genfromtxt('data/rotations_relu_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/rotations_relu_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/rotations_relu_W0f.csv',
                            delimiter= ","))
  assert W0f.shape == (8, 1)
  
  fs = np.genfromtxt('data/rotations_relu_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  u.check_equal(fs, [4,2,2,2])

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = X0.shape[1]
  assert f(-1) == dsize
  
  # load W0f and do shape checks (can remove)
  W0s = u.unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)
  Wf_holder = tf.placeholder(dtype, shape=W0f.shape)
  Wf = tf.Variable(Wf_holder, name="Wf")
  Wf_copy = tf.Variable(Wf_holder, name="Wf_copy")
  init_dict = {Wf_holder: W0f}
  
  # Create W's
  W = u.unflatten(Wf, fs[1:])
  X = tf.constant(X0)
  Y = tf.constant(Y0)
  W.insert(0, X)
  for (numpy_W, tf_W) in zip(W0s, W):
    u.check_equal(numpy_W.shape, u.fix_shape(tf_W.shape))

  # Create A's
  # A[1] == X
  A = [0]*(n+2)
  A[0] = u.Identity(dsize)
  for i in range(n+1):
    if i == 0:
      A[i+1] = X
    else:
      A[i+1] = tf.nn.relu(tf.matmul(W[i], A[i], name="A"+str(i+1)))

  assert W[0].get_shape() == X0.shape
  assert A[n+1].get_shape() == X0.shape
  assert A[1].get_shape() == X0.shape

  err = Y - A[n+1]
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(1.0, dtype=dtype, name="learning_rate")
  
  # Create B's
  B = [0]*(n+1)
  B[n] = (-err/dsize)*u.relu_mask(A[n+1])
  Bn = [0]*(n+1)            # Newton-modified backprop
  Bn[n] = u.Identity(f(n))
  for i in range(n-1, -1, -1):
    B[i] = t(W[i+1]) @ B[i+1]
    Bn[i] = t(W[i+1]) @ Bn[i+1]
    if i > 0:  # there's no relu on first matrix
      B[i] = B[i]*u.relu_mask(A[i+1])
      #  Bn[i] = Bn[i]*u.relu_mask(A[i+1]) wrong shape for relu mask

  # Create U's
  U = [list(range(n+1)) for _ in range(n+1)]
  for bottom in range(n+1):
    for top in range(n+1):
      if bottom > top:
        prod = u.Identity(f(top))
      else:
        prod = u.Identity(f(bottom-1))
        for i in range(bottom, top+1):
          prod = prod@t(W[i])   # TODO: exclude cols which don't propagate
      U[bottom][top] = prod

  # Block i, j gives hessian block between layer i and layer j
  blocks = [list(range(n+1)) for _ in range(n+1)]
  for i in range(1, n+1):
    for j in range(1, n+1):
      term1 = kr(A[i] @ t(A[j]), Bn[i] @ t(Bn[j])) / dsize;
      if i == j:
        term2 = tf.zeros((f(i)*f(i-1), f(i)*f(i-1)), dtype=dtype)
      elif i < j:
        term2 = kr(A[i] @ t(B[j]), U[i+1][j-1])
      else:
        term2 = kr(t(U[j+1][i-1]), B[i] @ t(A[j]))
        
      blocks[i][j]=term1 + term2 @ Kmat(f(j), f(j-1))

        
  # remove leftmost blocks (those are with respect to W[0] which is input)
  del blocks[0]
  for row in blocks:
    del row[0]
    
  hess = u.concat_blocks(blocks)
  ihess = u.pseudo_inverse(hess)
  
  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

  expected_hess = np.loadtxt("data/rotations_relu_newton_hess0.csv",
                              delimiter= ",")
  observed_hess = sess.run(hess)
  np.savetxt("data/rotations_relu_newton_hess1.csv", observed_hess,
             fmt="%.10f", delimiter=',')
  expected_ihess = np.loadtxt("data/rotations_simple_newton_ihess0.csv",
                              delimiter= ",")

  observed_ihess = sess.run(ihess)
  u.check_equal(expected_hess, observed_hess)


  # create dW's
  dW = [0]*(n+1)
  for i in range(n+1):
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
  del dW[0]  # get rid of W[0] update
  
  dWf = tf.concat([u.vec(dWi) for dWi in dW], axis=0)
  Wf_new = Wf - lr * ihess @ dWf 

  train_op1 = Wf_copy.assign(Wf_new)
  train_op2 = Wf.assign(Wf_copy)

  
  expected_losses = np.loadtxt("data/rotations_simple_newton_losses.csv",
                               delimiter= ",")
  observed_losses = []
  
  # from accompanying notebook
  # 0.0111498, 0.0308658, 0.00462571, 0.0000251229, 1.38508*10^-9, 
  # 1.32383*10^-17, 2.39119*10^-31
  
  for i in range(10):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  u.check_equal(observed_losses, expected_losses)
  
# Test block-diagonal Newton approximation
def simple_newton_bd_test():
  tf.reset_default_graph()
  X0 = np.genfromtxt('data/rotations_simple_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/rotations_simple_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/rotations_simple_W0f.csv',
                            delimiter= ","))
  assert W0f.shape == (8, 1)
  
  fs = np.genfromtxt('data/rotations_simple_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  u.check_equal(fs, [10,2,2,2])

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = X0.shape[1]
  assert f(-1) == dsize
  
  # load W0f and do shape checks (can remove)
  W0s = u.unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)
  Wf_holder = tf.placeholder(dtype, shape=W0f.shape)
  Wf = tf.Variable(Wf_holder, name="Wf")
  Wf_copy = tf.Variable(Wf_holder, name="Wf_copy")
  init_dict = {Wf_holder: W0f}
  
  # Create W's
  W = u.unflatten(Wf, fs[1:])
  X = tf.constant(X0)
  Y = tf.constant(Y0)
  W.insert(0, X)
  for (numpy_W, tf_W) in zip(W0s, W):
    u.check_equal(numpy_W.shape, u.fix_shape(tf_W.shape))

  # Create A's
  # A[1] == X
  A = [0]*(n+2)
  A[0] = u.Identity(dsize)
  for i in range(n+1):
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))

  assert W[0].get_shape() == X0.shape
  assert A[n+1].get_shape() == X0.shape
  assert A[1].get_shape() == X0.shape

  err = Y - A[n+1]
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.5, dtype=dtype, name="learning_rate")
  
  # Create B's
  B = [0]*(n+1)
  B[n] = -err/dsize
  Bn = [0]*(n+1)            # Newton-modified backprop
  Bn[n] = u.Identity(f(n))
  for i in range(n-1, -1, -1):
    B[i] = t(W[i+1]) @ B[i+1]
    Bn[i] = t(W[i+1]) @ Bn[i+1]

  # Create U's
  U = [list(range(n+1)) for _ in range(n+1)]
  for bottom in range(n+1):
    for top in range(n+1):
      if bottom > top:
        prod = u.Identity(f(top))
      else:
        prod = u.Identity(f(bottom-1))
        for i in range(bottom, top+1):
          prod = prod@t(W[i])
      U[bottom][top] = prod

  # Block i, j gives hessian block between layer i and layer j
  blocks = [list(range(n+1)) for _ in range(n+1)]
  for i in range(1, n+1):
    for j in range(1, n+1):
      term1 = kr(A[i] @ t(A[j]), Bn[i] @ t(Bn[j])) / dsize;
      if i == j:
        term2 = tf.zeros((f(i)*f(i-1), f(i)*f(i-1)), dtype=dtype)
      elif i < j:
        term2 = kr(A[i] @ t(B[j]), U[i+1][j-1])
      else:
        term2 = kr(t(U[j+1][i-1]), B[i] @ t(A[j]))
        
      blocks[i][j]=term1 + term2 @ Kmat(f(j), f(j-1))

        
  # remove leftmost blocks (those are with respect to W[0] which is input)
  del blocks[0]
  for row in blocks:
    del row[0]
    
  #hess = u.concat_blocks(blocks)
  ihess = u.concat_blocks(u.block_diagonal_inverse(blocks))
  #  ihess = u.pseudo_inverse(hess)
  
  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

  # create dW's
  dW = [0]*(n+1)
  for i in range(n+1):
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
  del dW[0]  # get rid of W[0] update
  
  dWf = tf.concat([u.vec(dWi) for dWi in dW], axis=0)
  Wf_new = Wf - lr * ihess @ dWf 

  train_op1 = Wf_copy.assign(Wf_new)
  train_op2 = Wf.assign(Wf_copy)

  
  expected_losses = np.loadtxt("data/rotations_simple_newtonbd_losses.csv",
                               delimiter= ",")
  observed_losses = []
  
  # from accompanying notebook
  # 0.0111498, 0.0000171591, 4.11445*10^-11, 2.33652*10^-22, 
  # 1.21455*10^-32,
 
  for i in range(10):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  u.check_equal(observed_losses, expected_losses)

  
# Test Kfac Newton approximation
def simple_newton_kfac_test():
  tf.reset_default_graph()
  X0 = np.genfromtxt('data/rotations_simple_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/rotations_simple_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/rotations_simple_W0f.csv',
                            delimiter= ","))
  assert W0f.shape == (8, 1)
  
  fs = np.genfromtxt('data/rotations_simple_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  u.check_equal(fs, [10,2,2,2])

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = X0.shape[1]
  assert f(-1) == dsize
  
  # load W0f and do shape checks (can remove)
  W0s = u.unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)
  Wf_holder = tf.placeholder(dtype, shape=W0f.shape)
  Wf = tf.Variable(Wf_holder, name="Wf")
  Wf_copy = tf.Variable(Wf_holder, name="Wf_copy")
  init_dict = {Wf_holder: W0f}
  
  # Create W's
  W = u.unflatten(Wf, fs[1:])
  X = tf.constant(X0)
  Y = tf.constant(Y0)
  W.insert(0, X)
  for (numpy_W, tf_W) in zip(W0s, W):
    u.check_equal(numpy_W.shape, u.fix_shape(tf_W.shape))

  # Create A's
  # A[1] == X
  A = [0]*(n+2)
  A[0] = u.Identity(dsize)
  for i in range(n+1):
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))

  assert W[0].get_shape() == X0.shape
  assert A[n+1].get_shape() == X0.shape
  assert A[1].get_shape() == X0.shape

  err = Y - A[n+1]
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.5, dtype=dtype, name="learning_rate")
  
  # Create B's
  B = [0]*(n+1)
  B[n] = -err/dsize
  Bn = [0]*(n+1)            # Newton-modified backprop
  Bn[n] = u.Identity(f(n))
  for i in range(n-1, -1, -1):
    B[i] = t(W[i+1]) @ B[i+1]
    Bn[i] = t(W[i+1]) @ Bn[i+1]
    
  # inverse Hessian blocks
  iblocks = u.empty_grid(n+1, n+1)
  for i in range(1, n+1):
    for j in range(1, n+1):
      # reuse Hess tensor calculation in order to get off-diag block sizes
      dummy_term = kr(A[i] @ t(A[j]), Bn[i] @ t(Bn[j])) / dsize;
      if i == j:
        acov = A[i] @ t(A[j])
        bcov = Bn[i] @ t(Bn[j]) / dsize;
        term = kr(u.pseudo_inverse(acov), u.pseudo_inverse(bcov))
      else:
        term = tf.zeros(shape=dummy_term.get_shape(), dtype=dtype)
      iblocks[i][j]=term
        
  # remove leftmost blocks (those are with respect to W[0] which is input)
  del iblocks[0]
  for row in iblocks:
    del row[0]
    
  ihess = u.concat_blocks(iblocks)
  
  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

  # create dW's
  dW = [0]*(n+1)
  for i in range(n+1):
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
  del dW[0]  # get rid of W[0] update
  
  dWf = tf.concat([u.vec(dWi) for dWi in dW], axis=0)
  Wf_new = Wf - lr * ihess @ dWf 

  train_op1 = Wf_copy.assign(Wf_new)
  train_op2 = Wf.assign(Wf_copy)

  
  expected_losses = np.loadtxt("data/rotations_simple_newtonkfac_losses.csv",
                               delimiter= ",")
  observed_losses = []

  # from accompanying notebook
  #  {0.0111498, 0.0000171591, 4.11445*10^-11, 2.33653*10^-22, 
  # 6.88354*10^-33,
 
  for i in range(10):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  u.check_equal(observed_losses, expected_losses)


if __name__=='__main__':
  simple_gradient_test()
  simple_newton_test()
  relu_gradient_test()
  
  # Relu Newton test assumes all examples have same relu pattern, so
  # 12% of Hessian entries are wrong, skip for now.
  #  # relu_newton_test()

  simple_newton_bd_test()
  simple_newton_kfac_test()
  print("%s tests passed" %(sys.argv[0]))

