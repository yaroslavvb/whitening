# Inverting random rotations using various optimization methods
# Accompanying notebook

import time
import os
import sys
import numpy as np
import tensorflow as tf
import util as u
from util import t  # transpose
from util import c2v
from util import v2c
from util import v2c_np
from util import v2r
from util import kr  # kronecker
from util import Kmat # commutation matrix
from util import vec

dtype = np.float64

import numpy as np

dtype = np.float64

def rotations1_gradient_test():
  #  https://www.wolframcloud.com/objects/ff6ecaf0-fccd-44e3-b26f-970d8fc2a57c
  tf.reset_default_graph()
  X0 = np.genfromtxt('data/large_rotations1_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/large_rotations1_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/large_rotations1_W0f.csv',
                            delimiter= ","))
  
  fs = np.genfromtxt('data/large_rotations1_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers

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
  lr0 = np.genfromtxt('data/large_rotations1_gradient_lr.csv')
  lr = tf.Variable(lr0, dtype=dtype)
  
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
  
  expected_losses = np.loadtxt("data/large_rotations1_gradient_losses.csv",
                               delimiter= ",")
  observed_losses = []
  # from accompanying notebook
  # {0.102522, 0.028124, 0.00907214, 0.00418929, 0.00293379,
  for i in range(10):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  u.check_equal(observed_losses, expected_losses)

def rotations2_gradient():
  tf.reset_default_graph()
  X0 = np.genfromtxt('data/large_rotations2_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/large_rotations2_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/large_rotations2_W0f.csv',
                            delimiter= ","))
  fs = np.genfromtxt('data/large_rotations2_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers

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
  lr0 = 0.01
  lr = tf.Variable(lr0, dtype=dtype)
  
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
  
  expected_losses = np.loadtxt("data/large_rotations1_gradient_losses.csv",
                               delimiter= ",")
  observed_losses = []
  # from accompanying notebook
  # {0.102522, 0.028124, 0.00907214, 0.00418929, 0.00293379,
  for i in range(100):
    loss0 = sess.run([loss])[0]
    print(loss0)
    observed_losses.append(loss0)
    sess.run(train_op1)
    sess.run(train_op2)

def rotations2_newton():
  # override kr with no-shape-inferring version
  def kr(A, B):
    return u.kronecker(A, B, do_shape_inference=False)
      
  tf.reset_default_graph()
  X0 = np.genfromtxt('data/large_rotations2_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/large_rotations2_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/large_rotations2_W0f.csv',
                            delimiter= ","))
  fs = np.genfromtxt('data/large_rotations2_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers

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
  lr0 = 1.0
  lr = tf.Variable(lr0, dtype=dtype, name="learning_rate")
  
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

  time0 = time.time()
  observed_hess = sess.run(hess)
  print("Time to compute hess %.2f"%(time.time()-time0))

  time0 = time.time()
  observed_ihess = sess.run(ihess)
  print("Time to compute ihess %.2f"%(time.time()-time0))


  # create dW's
  dW = [0]*(n+1)
  for i in range(n+1):
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
  del dW[0]  # get rid of W[0] update
  
  dWf = tf.concat([u.vec(dWi) for dWi in dW], axis=0)
  Wf_new = Wf - lr * ihess @ dWf 

  train_op1 = Wf_copy.assign(Wf_new)
  train_op2 = Wf.assign(Wf_copy)

  print("Model size: %d"%(len(tf.get_default_graph().get_operations())))
  observed_losses = []
  for i in range(20):
    loss0 = sess.run([loss])[0]
    print(loss0)
    observed_losses.append(loss0)
    sess.run(train_op1)
    sess.run(train_op2)

def rotations2_newton_bd():
  # override kr with no-shape-inferring version
  def kr(A, B):
    return u.kronecker(A, B, do_shape_inference=False)
  
  tf.reset_default_graph()
  X0 = np.genfromtxt('data/large_rotations2_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/large_rotations2_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/large_rotations2_W0f.csv',
                            delimiter= ","))
  fs = np.genfromtxt('data/large_rotations2_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
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
  lr = tf.Variable(0.1, dtype=dtype, name="learning_rate")
  
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
    
  ihess = u.concat_blocks(u.block_diagonal_inverse(blocks))
  
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

  observed_losses = []
  u.reset_time()
  for i in range(20):
    loss0 = sess.run([loss])[0]
    print(loss0)
    observed_losses.append(loss0)
    sess.run(train_op1)
    sess.run(train_op2)
    u.record_time()

  u.summarize_time()
  u.summarize_graph()


def rotations2_newton_kfac():
  tf.reset_default_graph()
  
  # override kr with no-shape-inferring version
  def kr(A, B):
    return u.kronecker(A, B, do_shape_inference=False)

  X0 = np.genfromtxt('data/large_rotations2_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/large_rotations2_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/large_rotations2_W0f.csv',
                            delimiter= ","))
  fs = np.genfromtxt('data/large_rotations2_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = X0.shape[1]
  assert f(-1) == dsize

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
  lr = tf.Variable(0.1, dtype=dtype, name="learning_rate")
  
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
        bcov = (Bn[i] @ t(Bn[j]))/dsize
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

  observed_losses = []
  elapsed_times = []
  u.reset_time()
  for i in range(10):
    loss0 = sess.run([loss])[0]
    print(loss0)
    observed_losses.append(loss0)
    sess.run(train_op1)
    sess.run(train_op2)
    u.record_time()
    
    
  u.summarize_time()
  u.summarize_graph()

def rotations2_natural_empirical():
  tf.reset_default_graph()
  
  # override kr with no-shape-inferring version
  def kr(A, B):
    return u.kronecker(A, B, do_shape_inference=False)

  X0 = np.genfromtxt('data/large_rotations2_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/large_rotations2_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/large_rotations2_W0f.csv',
                            delimiter= ","))
  fs = np.genfromtxt('data/large_rotations2_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
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
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  # Create W's
  W = u.unflatten(Wf, fs[1:])
  X = tf.constant(X0)
  Y = tf.constant(Y0)
  W.insert(0, X)

  A = [0]*(n+2)
  A[0] = u.Identity(dsize)
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.000001, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B[n] = -err/dsize
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))

  # Create gradient update. Make copy of variables and split update into
  # two run calls. Using single set of variables will gives updates that 
  # occasionally produce wrong results/NaN's because of data race
  
  dW = [0]*(n+1)
  updates1 = [0]*(n+1)  # compute updated value into Wcopy
  updates2 = [0]*(n+1)  # copy value back into W
  Wcopy = [0]*(n+1)
  for i in range(n+1):
    Wi_name = "Wcopy"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_init = tf.zeros(dtype=dtype, shape=Wi_shape, name=Wi_name+"_init")
    Wcopy[i] = tf.Variable(Wi_init, name=Wi_name, trainable=False)
    
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))


  del dW[0]  # get rid of W[0] update
  
  # construct flattened gradient update vector
  dWf = tf.concat([vec(grad) for grad in dW], axis=0)

  # inverse fisher preconditioner
  grads = tf.concat([u.khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  ifisher = u.pseudo_inverse(fisher)

  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = Wf - lr*(ifisher @ dWf)
  train_op1 = Wf_copy.assign(new_val_matrix)
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  observed_losses = []
  u.reset_time()
  for i in range(10):
    loss0 = sess.run(loss)
    print(loss0)
    observed_losses.append(loss0)
    sess.run(train_op1)
    sess.run(train_op2)
    u.record_time()

  u.summarize_time()
  u.summarize_graph()

def rotations2_natural_sampled(num_samples=1):
  tf.reset_default_graph()
  np.random.seed(0)
  tf.set_random_seed(0)
  
  # override kr with no-shape-inferring version
  def kr(A, B):
    return u.kronecker(A, B, do_shape_inference=False)

  X0 = np.genfromtxt('data/large_rotations2_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/large_rotations2_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/large_rotations2_W0f.csv',
                            delimiter= ","))
  fs = np.genfromtxt('data/large_rotations2_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
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
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  # Create W's
  W = u.unflatten(Wf, fs[1:])
  X = tf.constant(X0)
  Y = tf.constant(Y0)
  W.insert(0, X)

  A = [0]*(n+2)
  A2 = [0]*(n+2)  # augmented forward props for natural gradient
  A[0] = u.Identity(dsize)
  A2[0] =  u.Identity(dsize*num_samples)
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    if i == 0:
      A2[i+1] = tf.concat([W[0]]*num_samples, axis=1)
    else:
      A2[i+1] = tf.matmul(W[i], A2[i], name="A2"+str(i+1))
      
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.1, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B2 = [0]*(n+1)
  B[n] = -err/dsize
  B2[n] = tf.random_normal((f(n), dsize*num_samples), 0, 1, dtype=dtype)
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    B2[i] = tf.matmul(tf.transpose(W[i+1]), B2[i+1], name="B2"+str(i))

  # Create gradient update. Make copy of variables and split update into
  # two run calls. Using single set of variables will gives updates that 
  # occasionally produce wrong results/NaN's because of data race
  
  dW = [0]*(n+1)
  dW2 = [0]*(n+1)
  updates1 = [0]*(n+1)  # compute updated value into Wcopy
  updates2 = [0]*(n+1)  # copy value back into W
  Wcopy = [0]*(n+1)
  for i in range(n+1):
    Wi_name = "Wcopy"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_init = tf.zeros(dtype=dtype, shape=Wi_shape, name=Wi_name+"_init")
    Wcopy[i] = tf.Variable(Wi_init, name=Wi_name, trainable=False)
    
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
    dW2[i] = tf.matmul(B2[i], tf.transpose(A2[i]), name="dW2"+str(i))


  del dW[0]  # get rid of W[0] update
  del dW2[0]  # get rid of W[0] update
  
  # construct flattened gradient update vector
  dWf = tf.concat([vec(grad) for grad in dW], axis=0)

  # inverse fisher preconditioner
  grads = tf.concat([u.khatri_rao(A2[i], B2[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / (dsize*num_samples)
  ifisher = u.pseudo_inverse(fisher)

  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = Wf - lr*(ifisher @ dWf)
  train_op1 = Wf_copy.assign(new_val_matrix)
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  observed_losses = []
  u.reset_time()
  for i in range(20):
    loss0 = sess.run(loss)
    print(loss0)
    observed_losses.append(loss0)
    sess.run(train_op1)
    sess.run(train_op2)
    u.record_time()

  u.summarize_time()
  u.summarize_graph()

def rotations2_natural_sampled_bd(num_samples=1):
  tf.reset_default_graph()
  np.random.seed(0)
  tf.set_random_seed(0)
  
  # override kr with no-shape-inferring version
  def kr(A, B):
    return u.kronecker(A, B, do_shape_inference=False)

  X0 = np.genfromtxt('data/large_rotations2_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/large_rotations2_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/large_rotations2_W0f.csv',
                            delimiter= ","))
  fs = np.genfromtxt('data/large_rotations2_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
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
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  # Create W's
  W = u.unflatten(Wf, fs[1:])
  X = tf.constant(X0)
  Y = tf.constant(Y0)
  W.insert(0, X)

  A = [0]*(n+2)
  A2 = [0]*(n+2)  # augmented forward props for natural gradient
  A[0] = u.Identity(dsize)
  A2[0] =  u.Identity(dsize*num_samples)
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    if i == 0:
      A2[i+1] = tf.concat([W[0]]*num_samples, axis=1)
    else:
      A2[i+1] = tf.matmul(W[i], A2[i], name="A2"+str(i+1))
      
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)

  # lower learning rate by 10x
  lr = tf.Variable(0.01, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B2 = [0]*(n+1)
  B[n] = -err/dsize
  B2[n] = tf.random_normal((f(n), dsize*num_samples), 0, 1, dtype=dtype)
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    B2[i] = tf.matmul(tf.transpose(W[i+1]), B2[i+1], name="B2"+str(i))

  # Create gradient update. Make copy of variables and split update into
  # two run calls. Using single set of variables will gives updates that 
  # occasionally produce wrong results/NaN's because of data race
  
  dW = [0]*(n+1)
  dW2 = [0]*(n+1)
  updates1 = [0]*(n+1)  # compute updated value into Wcopy
  updates2 = [0]*(n+1)  # copy value back into W
  Wcopy = [0]*(n+1)
  for i in range(n+1):
    Wi_name = "Wcopy"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_init = tf.zeros(dtype=dtype, shape=Wi_shape, name=Wi_name+"_init")
    Wcopy[i] = tf.Variable(Wi_init, name=Wi_name, trainable=False)
    
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
    dW2[i] = tf.matmul(B2[i], tf.transpose(A2[i]), name="dW2"+str(i))


  del dW[0]  # get rid of W[0] update
  del dW2[0]  # get rid of W[0] update
  
  # construct flattened gradient update vector
  dWf = tf.concat([vec(grad) for grad in dW], axis=0)

  # inverse fisher preconditioner
  grads = tf.concat([u.khatri_rao(A2[i], B2[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / (dsize*num_samples)
  blocks = u.partition_matrix_evenly(fisher, 10)
  #  ifisher = u.pseudo_inverse(fisher)
  ifisher = u.concat_blocks(u.block_diagonal_inverse(blocks))

  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = Wf - lr*(ifisher @ dWf)
  train_op1 = Wf_copy.assign(new_val_matrix)
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  observed_losses = []
  u.reset_time()
  for i in range(20):
    loss0 = sess.run(loss)
    print(loss0)
    observed_losses.append(loss0)
    sess.run(train_op1)
    sess.run(train_op2)
    u.record_time()

  u.summarize_time()
  u.summarize_graph()

def rotations2_natural_sampled_kfac(num_samples=1):
  tf.reset_default_graph()
  np.random.seed(0)
  tf.set_random_seed(0)
  
  # override kr with no-shape-inferring version
  def kr(A, B):
    return u.kronecker(A, B, do_shape_inference=False)

  X0 = np.genfromtxt('data/large_rotations2_X0.csv',
                     delimiter= ",")
  Y0 = np.genfromtxt('data/large_rotations2_Y0.csv',
                     delimiter= ",")
  W0f = v2c_np(np.genfromtxt('data/large_rotations2_W0f.csv',
                            delimiter= ","))
  fs = np.genfromtxt('data/large_rotations2_fs.csv',
                      delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
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
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  # Create W's
  W = u.unflatten(Wf, fs[1:])
  X = tf.constant(X0)
  Y = tf.constant(Y0)
  W.insert(0, X)

  A = [0]*(n+2)
  A2 = [0]*(n+2)  # augmented forward props for natural gradient
  A[0] = u.Identity(dsize)
  A2[0] =  u.Identity(dsize*num_samples)
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    if i == 0:
      # replicate dataset multiple times corresponding to number of samples
      A2[i+1] = tf.concat([W[0]]*num_samples, axis=1)
    else:
      A2[i+1] = tf.matmul(W[i], A2[i], name="A2"+str(i+1))
      
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)

  # lower learning rate by 10x
  lr = tf.Variable(0.01, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B2 = [0]*(n+1)
  B[n] = -err/dsize
  B2[n] = tf.random_normal((f(n), dsize*num_samples), 0, 1, dtype=dtype)
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    B2[i] = tf.matmul(tf.transpose(W[i+1]), B2[i+1], name="B2"+str(i))

  # Create gradient update. Make copy of variables and split update into
  # two run calls. Using single set of variables will gives updates that 
  # occasionally produce wrong results/NaN's because of data race
  
  dW = [0]*(n+1)
  dW2 = [0]*(n+1)
  updates1 = [0]*(n+1)  # compute updated value into Wcopy
  updates2 = [0]*(n+1)  # copy value back into W
  Wcopy = [0]*(n+1)
  for i in range(n+1):
    Wi_name = "Wcopy"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_init = tf.zeros(dtype=dtype, shape=Wi_shape, name=Wi_name+"_init")
    Wcopy[i] = tf.Variable(Wi_init, name=Wi_name, trainable=False)
    
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
    dW2[i] = tf.matmul(B2[i], tf.transpose(A2[i]), name="dW2"+str(i))

  del dW[0]  # get rid of W[0] update
  del dW2[0]  # get rid of W[0] update
  
  # construct flattened gradient update vector
  dWf = tf.concat([vec(grad) for grad in dW], axis=0)

  # todo: divide both activations and backprops by size for cov calc
  
  # Kronecker factored covariance blocks
  iblocks = u.empty_grid(n+1, n+1)
  for i in range(1, n+1):
    for j in range(1, n+1):
      if i == j:
        acov = A2[i] @ t(A2[j]) / (dsize*num_samples)
        bcov = B2[i] @ t(B2[j]) / (dsize*num_samples);
        term = kr(u.pseudo_inverse(acov), u.pseudo_inverse(bcov))
      else:
        term = tf.zeros(shape=(f(i)*f(i-1), f(j)*f(j-1)), dtype=dtype)
      iblocks[i][j]=term
        
  # remove leftmost blocks (those are with respect to W[0] which is input)
  del iblocks[0]
  for row in iblocks:
    del row[0]

  ifisher = u.concat_blocks(iblocks)
  
  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = Wf - lr*(ifisher @ dWf)
  train_op1 = Wf_copy.assign(new_val_matrix)
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  observed_losses = []
  u.reset_time()
  for i in range(20):
    loss0 = sess.run(loss)
    print(loss0)
    observed_losses.append(loss0)
    sess.run(train_op1)
    sess.run(train_op2)
    u.record_time()

  u.summarize_time()
  u.summarize_graph()


  
if __name__=='__main__':

  rotations2_natural_sampled_kfac(num_samples=10)
  sys.exit()
  # had to lower learning rate 0.1 -> 0.01
  # 8.90237442254e-05
  # 6.18130536891e-05
  # 4.96606271923e-05
  # 7.84590336526e-05
  # 5.92008979491e-05
  # 4.79501913281e-05
  # 3.88183551121e-05
  # 2.78729639502e-05
  # 1.92286298642e-05
  # 1.45717278319e-05
  # 1.18689032188e-05
  # 9.42530653255e-06
  # 8.48717716437e-06
  # 7.21245609911e-06
  # 6.49958263101e-06
  # 7.67340663869e-06
  # 6.19110712621e-06
  # 4.7796906074e-06
  # 3.58722944835e-06
  # 3.35051072823e-06
  # Times: min: 53.70, median: 54.86, times: 2227.11,55.40,54.90,54.48,53.70,54.63,54.65,54.01,54.86,54.61,56.28,53.92,54.16,54.86,59.86,56.55,55.31,55.55,54.40,55.37
  # Graph: 25908 ops, 9 MBs
  #  rotations2_natural_sampled_bd(num_samples=5)
  #  sys.exit()

  
  # This one doesn't like to converge
  # rotations2_natural_empirical()
  #  sys.exit()

  # 8.90237442254e-05
  # 0.000121080497064
  # 5.16870109539e-05
  # 3.93844900114e-05
  # 4.07540534586e-05
  # 2.85543555423e-05
  # 1.06994454845e-05
  # 6.26537506601e-06
  # 4.64665082377e-06
  # 3.10120033165e-06
  # 1.84422510204e-06
  # 1.7874049467e-06
  # 1.45818008467e-06
  # 1.16271404754e-06
  # 1.02104047911e-06
  # 9.6619013919e-07
  # 8.40683575721e-07
  # 8.23710554658e-07
  # 8.04916977793e-07
  # 7.80346517484e-07
  # Times: min: 614.73, median: 634.24, times: 1080.51,631.78,634.09,642.24,652.97,625.01,634.39,633.93,630.21,632.55,625.56,630.36,663.54,638.67,614.73,636.27,671.54,654.75,639.44,631.07
  # Graph: 5781 ops, 2 MBs
#  rotations2_natural_sampled(num_samples=1)
#  sys.exit()
  
  # 8.90237442254e-05
  # 7.45484431658e-05
  # 5.36263658941e-05
  # 4.94970415368e-05
  # 5.26316276865e-05
  # 3.70637864395e-05
  # 3.22365853897e-05
  # 2.9095776698e-05
  # 2.66559914658e-05
  # 2.23236225278e-05
  # 1.93340650931e-05
  # 1.80233464052e-05
  # 1.71216172002e-05
  # 1.44982778379e-05
  # 1.2080789836e-05
  # 1.03813967235e-05
  # 9.0094424551e-06
  # 8.68403546819e-06
  # 7.83529832147e-06
  # 7.60852440119e-06
  # Times: min: 644.11, median: 676.36, times: 2746.84,689.03,680.05,654.87,664.90,684.73,662.71,678.19,675.54,697.81,684.55,674.66,683.50,672.87,680.99,677.19,659.26,666.27,644.11,645.98

  rotations2_natural_sampled(num_samples=5)
  sys.exit()
  
  #  rotations1_gradient_test()
  
  #  rotations2_gradient()
  #  8.90237442254e-05
  #  3.26578279099e-05
  #  1.20416052473e-05
  #  4.50086666858e-06
  #  1.74259310596e-06
  #  7.335446992e-07
  #  3.64289774785e-07

  # Time to compute hess 0.93
  # Time to compute ihess 1.64
  # Model size: 20128
  # 6.81262769334e-07
  # 0.00391577472813
  # 1.55111254048e-05
  # 2.32230833013e-10
  # 7.44823949424e-20

  #  rotations2_newton()
  
  #  8.90237442254e-05
  #  3.56141425886e-06
  #  1.42451943222e-07
  #  5.69802736553e-09
  #  2.27920670486e-10
  #  9.11682339804e-12
  rotations2_newton_bd()

  #  8.90237442254e-05
  #  3.56141425548e-06
  #  1.42451943136e-07
  #  5.69802735848e-09
  #  2.27920670137e-10
  #  9.11682337579e-12
  rotations2_newton_kfac()
