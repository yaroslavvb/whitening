#!/bin/env python -u
#
# Apply natural gradient (using empirical Fisher) to a multilayer problem
#
# TODO: c2v operator
# vectorize operator
#
#
import numpy as np
import sys
import tensorflow as tf
import traceback

dtype = np.float64

def pseudo_inverse(mat):
  s, u, v = tf.svd(mat)
  eps = 1e-10   # zero threshold for eigenvalues
  si = tf.where(tf.less(s, eps), s, 1./s)
  return u @ tf.diag(si) @ tf.transpose(v)

def identity(n):
  return tf.diag(tf.ones((n,), dtype=dtype))


# partitions numpy array into sublists of given sizes
def partition(vec, sizes):
  assert np.sum(sizes) == len(vec)
  splits = []
  current_idx = 0
  for i in range(len(sizes)):
    splits.append(vec[current_idx: current_idx+sizes[i]])
    current_idx += sizes[i]
  assert current_idx == len(vec)
  return splits

def partition_tf(vec, sizes):
  assert len(vec.shape) == 1
  assert np.sum(sizes) == vec.shape[0]
  splits = []
  current_idx = 0
  for i in range(len(sizes)):
    splits.append(vec[current_idx: current_idx+sizes[i]])
    current_idx += sizes[i]
  return splits

def partition_tf_test():
  vec = tf.constant([1,2,3,4,5])
  sess = tf.Session()
  result = sess.run(partition_tf(vec, [3, 2]))
  check_equal(result[0], [1,2,3])
  assert (result[1] == [4,5]).all()

  
def v2c_tf(vec):
  """Converts vector to column matrix."""
  return tf.expand_dims(vec, 1)

def v2r_tf(vec):
  """Converts vector into row matrix."""
  return tf.expand_dims(vec, 0)
  
def c2v_tf(col):
  """Converts vector into row matrix."""
  return tf.reshape(col, [-1])
  
def unvectorize(vec, rows):
  """Turns vectorized version of tensor into original matrix with given
  number of rows."""
  assert len(vec)%rows==0
  cols = len(vec)//rows;
  return np.array(np.split(vec, cols)).T

def unvectorize_tf(vec, rows):
  assert len(vec.shape) == 1
  assert vec.shape[0]%rows == 0
  cols = int(vec.shape[0]//rows) 
  cols = [v2r_tf(v) for v in tf.split(vec, cols)]
  return tf.transpose(tf.concat(cols, 0))

def unvectorize_tf_test():
  vec = tf.constant([1,2,3,4,5,6])
  sess = tf.Session()
  result = sess.run(unvectorize_tf(vec, 2))
  assert (result==[[1,3,5],[2,4,6]]).all()

def vectorize_tf(mat):
  return tf.reshape(tf.transpose(mat), [-1,])

def vectorize_tf_test():
  mat = tf.constant([[1, 3, 5], [2, 4, 6]])
  sess = tf.Session()
  check_equal(sess.run(vectorize_tf(mat)), [1,2,3,4,5,6])
  

# turns flattened representation into list of matrices with given matrix
# sizes
def unflatten(Wf, fs):
  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  assert np.sum(sizes)==len(Wf)
  Wsf = partition(Wf, sizes)
  Ws = [unvectorize(Wsf[i], dims[i][0]) for i in range(len(sizes))]
  return Ws

# Turns flattened Tensor into list of rank-2 tensors with given sizes
def unflatten_tf(Wf, fs):
  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  assert len(Wf.shape) == 1
  assert np.sum(sizes)==Wf.shape[0]
  Wsf = partition_tf(Wf, sizes)
  Ws = [unvectorize_tf(Wsf[i], dims[i][0]) for i in range(len(sizes))]
  return Ws

def unflatten_tf_test():
  vec = tf.constant(list(range(1, 11)))
  sess = tf.Session()
  fs = [2,2,2,1]
  result = sess.run(unflatten_tf(vec, fs))
  check_equal(result[0], [[1,3],[2,4]])
  check_equal(result[1], [[5,7],[6,8]])
  check_equal(result[2], [[9, 10]])


def check_equal(a, b, rtol=1e-12, atol=1e-12):
  try:
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
  except Exception as e:
    print("Error" + "-"*60)
    for line in traceback.format_stack():
      print(line.strip())
        
    # exc_type, exc_value, exc_traceback = sys.exc_info()
    # print("*** print_tb:")
    # traceback.print_tb(exc_traceback, limit=10, file=sys.stdout)
    # efmt = traceback.format_exc()
    # print(efmt)


# convention, X0 is numpy, X is Tensor
def regular_test():
  """Test SGD, using tf.gradients for backprop."""
  
  tf.reset_default_graph()

  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_multilayer_XY0.csv',
                      delimiter= ",")
  
  fs = np.genfromtxt('data/natural_gradient_multilayer_fs.csv',
                     delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  W0f = np.genfromtxt('data/natural_gradient_multilayer_W0f.csv',
                     delimiter= ",")
  W0s = unflatten(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)

  
  # initialize data + layers
  W = []   # list of "W" matrices. W[0] is input matrix (X), W[n] is last matrix
  Wi_holders = []
  A = [identity(dsize)]   # activation matrices
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    Wi_name = "W"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_holder = tf.placeholder(dtype, shape=Wi_shape, name=Wi_name+"_h")
    Wi_holders.append(Wi_holder)  # TODO: delete
    Wi = tf.Variable(Wi_holder, name=Wi_name, trainable=(i>0))
    Ai_name = "A"+str(i+1)
    print("Multiplying %s and %s " %(Wi.shape, A[-1].shape))
    Ai = tf.matmul(Wi, A[-1], name=Ai_name)
    A.append(Ai)
    W.append(Wi)
    
    init_dict[Wi_holder] = W0s[i]

  assert len(A) == n+2
  
  assert W[0].shape == (2, 10)
  assert W[1].shape == (2, 2)
  assert W[2].shape == (2, 2)
  assert W[3].shape == (1, 2)

  assert X0.shape == (2, 10)
  assert W0s[1].shape == (2, 2)
  assert W0s[2].shape == (2, 2)
  assert W0s[3].shape == (1, 2)
  
  assert A[0].shape == (10, 10)
  assert A[1].shape == (2, 10)
  assert A[2].shape == (2, 10)
  assert A[3].shape == (2, 10)
  assert A[4].shape == (1, 10)

  
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = 0.5
  
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
  grads_and_vars = opt.compute_gradients(loss, tf.trainable_variables())
  train_op = opt.apply_gradients(grads_and_vars)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/natural_gradient_multilayer_losses_regular.csv")
  
  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op)

  check_equal(observed_losses, expected_losses)

# convention, X0 is numpy, X is Tensor
def regular_manual_test():
  """Train network, without using tf.gradients"""
  
  tf.reset_default_graph()

  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_multilayer_XY0.csv',
                      delimiter= ",")
  
  fs = np.genfromtxt('data/natural_gradient_multilayer_fs.csv',
                     delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  W0f = np.genfromtxt('data/natural_gradient_multilayer_W0f.csv',
                     delimiter= ",")
  W0s = unflatten(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)

  
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = identity(dsize)
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    Wi_name = "W"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_holder = tf.placeholder(dtype, shape=Wi_shape, name=Wi_name+"_h")
    W[i] = tf.Variable(Wi_holder, name=Wi_name, trainable=(i>0))
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    init_dict[Wi_holder] = W0s[i]

  assert len(A) == n+2
  
  assert W[0].shape == (2, 10)
  assert W[1].shape == (2, 2)
  assert W[2].shape == (2, 2)
  assert W[3].shape == (1, 2)

  assert X0.shape == (2, 10)
  assert W0s[1].shape == (2, 2)
  assert W0s[2].shape == (2, 2)
  assert W0s[3].shape == (1, 2)
  
  assert A[0].shape == (10, 10)
  assert A[1].shape == (2, 10)
  assert A[2].shape == (2, 10)
  assert A[3].shape == (2, 10)
  assert A[4].shape == (1, 10)

  
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.5, dtype=dtype)
  
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
    updates1[i] = Wcopy[i].assign(W[i]-lr*dW[i])
    updates2[i] = W[i].assign(Wcopy[i])

  del updates1[0]  # don't update input matrices
  del updates2[0]
  
  train_op1 = tf.group(*updates1)
  train_op2 = tf.group(*updates2)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/natural_gradient_multilayer_losses_regular.csv")
  
  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  check_equal(observed_losses, expected_losses)

# convention, X0 is numpy, X is Tensor
def regular_manual_vectorized_test():
  """Train network, with manual backprop, in vectorized form"""
  
  tf.reset_default_graph()

  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_multilayer_XY0.csv',
                      delimiter= ",")
  
  fs = np.genfromtxt('data/natural_gradient_multilayer_fs.csv',
                     delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  W0f = np.genfromtxt('data/natural_gradient_multilayer_W0f.csv',
                     delimiter= ",")
  W0s = unflatten(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)

  
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=(Wf_size,))
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten_tf(Wf, fs[1:])
  W.insert(0, tf.constant(X0))
  assert W[0].shape == [2, 10]
  assert W[1].shape == [2, 2]
  assert W[2].shape == [2, 2]
  assert W[3].shape == [1, 2]
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    
  assert len(A) == n+2
  assert A[0].shape == (10, 10)
  assert A[1].shape == (2, 10)
  assert A[2].shape == (2, 10)
  assert A[3].shape == (2, 10)
  assert A[4].shape == (1, 10)

  
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.5, dtype=dtype)
  
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
  dWf = tf.concat([vectorize_tf(grad) for grad in dW], axis=0)

  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  train_op1 = Wf_copy.assign(Wf - lr*dWf)
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/natural_gradient_multilayer_losses_regular.csv")
  
  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  check_equal(observed_losses, expected_losses)

def kronecker_cols(a, b):
  """Treats rank-1 vectors a, b as columns, returns Kronecker product a x b."""
  
  assert len(a.get_shape())==1, "Input a must be rank-1, got shape %s" %(a.get_shape(),)
  assert len(b.get_shape())==1, "Input b must be rank-1, got shape %s"%(a.get_shape(),)
  segments = []
  for i in range(a.get_shape()[0]):
    segments.append(a[i]*b)
  result_vec = tf.concat(segments, axis=0)
  result_col = tf.expand_dims(result_vec, 1)
  return result_col

def kronecker_cols_test():
  a = tf.constant([1,2])
  b = tf.constant([3,4])
  c = tf.transpose(tf.constant([[3,4,6,8]]))
  sess = tf.Session()
  assert sess.run(tf.equal(kronecker_cols(a, b), c)).all()

# def kronecker(A, B):
#   pass



# def merge_mats(mats):
#   """Merges mxn grid of mats into single matrix."""
#   m = len(mats)
#   n = len(mats[0])
#   for i in range(m):
#     for j in range(n):
#       pass

def col(A,i):
  """Extracts i'th column of matrix A"""
  assert len(A.get_shape())==2
  assert i>=0 and i < A.get_shape()[1]
  return tf.expand_dims(A[:,i], 1)
  
def khatri_rao(A, B):
  """Khatri rao product of matrices A,B"""

  cols = []
  assert len(A.get_shape()) == 2, "A must be rank-1, got shape %s" %(a.get_shape(),)
  assert A.get_shape()[1] == B.get_shape()[1]
  for i in range(A.get_shape()[1]):
    cols.append(kronecker_cols(A[:, i], B[:,i]))
  return tf.concat(cols, axis=1)

def khatri_rao_test():
  A = tf.constant([[1, 2], [3, 4]])
  B = tf.constant([[5, 6], [7, 8]])
  C = tf.constant([[5,12], [7,16], [15,24], [21,32]])
  sess = tf.Session()
  assert sess.run(tf.equal(khatri_rao(A, B), C)).all()
  
# convention, X0 is numpy, X is Tensor
def fisher_test():
  """Test computation of empirical Fisher matrix."""
  
  tf.reset_default_graph()

  tf.reset_default_graph()

  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_multilayer_XY0.csv',
                      delimiter= ",")
  fs = np.genfromtxt('data/natural_gradient_multilayer_fs.csv',
                     delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  W0f = np.genfromtxt('data/natural_gradient_multilayer_W0f.csv',
                     delimiter= ",")
  W0s = unflatten(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)

  
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = identity(dsize)
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    Wi_name = "W"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_holder = tf.placeholder(dtype, shape=Wi_shape, name=Wi_name+"_h")
    W[i] = tf.Variable(Wi_holder, name=Wi_name, trainable=(i>0))
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    init_dict[Wi_holder] = W0s[i]

  assert len(A) == n+2
  
  assert W[0].shape == (2, 10)
  assert W[1].shape == (2, 2)
  assert W[2].shape == (2, 2)
  assert W[3].shape == (1, 2)

  assert X0.shape == (2, 10)
  assert W0s[1].shape == (2, 2)
  assert W0s[2].shape == (2, 2)
  assert W0s[3].shape == (1, 2)
  
  assert A[0].shape == (10, 10)
  assert A[1].shape == (2, 10)
  assert A[2].shape == (2, 10)
  assert A[3].shape == (2, 10)
  assert A[4].shape == (1, 10)

  
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.5, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B[n] = -err/dsize
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/natural_gradient_multilayer_losses_regular.csv")

  expected_fisher = np.loadtxt("data/natural_gradient_multilayer_fisher0.csv",
                               delimiter= ",")

  block11 = khatri_rao(A[1], B[1]) @ tf.transpose(khatri_rao(A[1], B[1]))/dsize
  expected_block11 = expected_fisher[:4, :4]
  check_equal(sess.run(block11), expected_block11)

  # construct fisher matrix out of individual mats
  grads = tf.concat([khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  check_equal(sess.run(fisher), expected_fisher)

  # create vectorized parameter vector
  
  
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
    updates1[i] = Wcopy[i].assign(W[i]-lr*dW[i])
    updates2[i] = W[i].assign(Wcopy[i])

  del updates1[0]  # don't update input matrices
  del updates2[0]
  
  train_op1 = tf.group(*updates1)
  train_op2 = tf.group(*updates2)

  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  check_equal(observed_losses, expected_losses)

# convention, X0 is numpy, X is Tensor
def natural_gradient_test():
  """Train network, with manual backprop, in vectorized form"""
  
  tf.reset_default_graph()

  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_multilayer_XY0.csv',
                      delimiter= ",")
  
  fs = np.genfromtxt('data/natural_gradient_multilayer_fs.csv',
                     delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  W0f = np.genfromtxt('data/natural_gradient_multilayer_W0f.csv',
                     delimiter= ",")
  W0s = unflatten(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)

  
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=(Wf_size,))
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten_tf(Wf, fs[1:])
  W.insert(0, tf.constant(X0))
  assert W[0].shape == [2, 10]
  assert W[1].shape == [2, 2]
  assert W[2].shape == [2, 2]
  assert W[3].shape == [1, 2]
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    
  assert len(A) == n+2
  assert A[0].shape == (10, 10)
  assert A[1].shape == (2, 10)
  assert A[2].shape == (2, 10)
  assert A[3].shape == (2, 10)
  assert A[4].shape == (1, 10)

  
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.001, dtype=dtype)
  
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
  dWf = tf.concat([vectorize_tf(grad) for grad in dW], axis=0)


  # inverse fisher preconditioner
  grads = tf.concat([khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  ifisher = pseudo_inverse(fisher)

  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = v2c_tf(Wf) - lr*(ifisher @ v2c_tf(dWf))
  train_op1 = Wf_copy.assign(c2v_tf(new_val_matrix))
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/natural_gradient_multilayer_losses_fisher.csv")
  
  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  check_equal(observed_losses, expected_losses)


if __name__ == '__main__':
  natural_gradient_test()
  sys.exit()
  regular_test()
  regular_manual_test()
  regular_manual_vectorized_test()
  kronecker_cols_test()
  khatri_rao_test()
  partition_tf_test()
  unvectorize_tf_test()
  unflatten_tf_test()
  vectorize_tf_test()
  fisher_test()
