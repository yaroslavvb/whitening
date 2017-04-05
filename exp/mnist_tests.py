import numpy as np
import sys
import tensorflow as tf
import traceback

dtype = np.float64

DISABLE_RELU = True     # hack to turn off all relus

from util import *

def train_tiny():
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
  W0s = unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
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
  
  W = unflatten(Wf, fs[1:])
  W.insert(0, tf.constant(X0))
  assert W[0].shape == [2, 10]
  assert W[1].shape == [2, 2]
  assert W[2].shape == [2, 2]
  assert W[3].shape == [1, 2]
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    if i == 0:
      A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    else:
      A[i+1] = tf.nn.relu(tf.matmul(W[i], A[i], name="A"+str(i+1)))
    
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
  lr = tf.Variable(0.0001, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B[n] = (-err/dsize)*relu_mask(A[n+1])
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    if i == 0:
      pass
    else:
      B[i] = B[i]*relu_mask(A[i+1])

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
  dWf = tf.concat([vectorize(grad) for grad in dW], axis=0)

  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")

  # inverse fisher preconditioner
  grads = tf.concat([khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  ifisher = pseudo_inverse(fisher)
  new_val_matrix = v2c(Wf) - lr*(ifisher @ v2c(dWf))
  train_op1 = Wf_copy.assign(c2v(new_val_matrix))
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  observed_losses = []
  for i in range(100):
    loss0 = sess.run([loss])[0]
    observed_losses.append(loss0)
    print(loss0)
    sess.run(train_op1)
    sess.run(train_op2)

  print(observed_losses)

def mnist1000_test():
  do_relu = False
  
  XY0 = np.genfromtxt('data/mnist_small.csv', delimiter= ",")
  fs = [1000, 50, 10]
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-fs[-1],:]  # 50 x d
  Y0 = XY0[-fs[-1]:,:]  # 10 x d
  dsize = X0.shape[1]
  f1 = X0.shape[0]
  f2 = fs[-1]
  assert fs[1] == 50
  assert f1 == 50

  X_ = tf.placeholder(dtype, shape=X0.shape)  # remove
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  # initial W0 predicts 0.1 for every instance for every class
  # W0 = np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T
  #  W = tf.Variable(W0.astype(dtype))

  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  print("shapes are ", dims)

  W0f = vectorize_np(np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T)
  W0s = unflatten_np(W0f, fs[1:])
  W0s.insert(0, X0)
  
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=(Wf_size,))
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten(Wf, fs[1:])
  print("W vars are %s"%([v.get_shape() for v in W]))
  W.insert(0, tf.constant(X0))
  print("W vars are now %s"%([v.get_shape() for v in W]))
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    if i == 0:
      A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    else:
      A[i+1] = relu(tf.matmul(W[i], A[i], name="A"+str(i+1)))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = (Y - A[n+1])
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/dsize
  lr = tf.Variable(1e-6, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  # add factor of 2 because our loss doesn't have 1/2
  B[n] = (-2*err/dsize)*relu_mask(A[n+1]) 
  print("Created B[%d] of shape %s"%(n, B[n].get_shape()))
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    print("Created B[%d] of shape %s"%(i, B[i].get_shape()))
    if i == 0:
      pass
    else:
      B[i] = B[i]*relu_mask(A[i+1])

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
  dWf = tf.concat([vectorize(grad) for grad in dW], axis=0)

  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  train_op1 = Wf_copy.assign(Wf - lr*dWf)
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  # from mnist_linear.nb
  # {0.9, 0.888968, 0.878518, 0.868525, 0.858921, 0.849668, ...
  expected_losses = np.loadtxt("mnist_linear_losses0.csv")

  
  observed_losses = []
  for i in range(51):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  np.testing.assert_allclose(observed_losses, expected_losses)
  check_equal(observed_losses, expected_losses)
  print("mnist1000 passed")

USE_TINY = True
def mnist1000_fisher():
  do_relu = False
  
  XY0 = np.genfromtxt('data/mnist_small.csv', delimiter= ",")
  fs = [1000, 50, 10]

  if USE_TINY:
    fs = [20, 50, 10]
    
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-fs[-1],:fs[0]]  # 50 x d
  Y0 = XY0[-fs[-1]:,:fs[0]]  # 10 x d
  dsize = X0.shape[1]
  f1 = X0.shape[0]
  f2 = fs[-1]
  assert fs[1] == 50
  assert f1 == 50

  X_ = tf.placeholder(dtype, shape=X0.shape)  # remove
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  # initial W0 predicts 0.1 for every instance for every class
  # W0 = np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T
  #  W = tf.Variable(W0.astype(dtype))

  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  print("shapes are ", dims)

  W0f = vectorize_np(np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T)
  #  W0f = np.ones((500,), dtype=dtype)
  W0s = unflatten_np(W0f, fs[1:])
  W0s.insert(0, X0)
  
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=(Wf_size,))
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten(Wf, fs[1:])
  print("W vars are %s"%([v.get_shape() for v in W]))
  W.insert(0, tf.constant(X0))
  print("W vars are now %s"%([v.get_shape() for v in W]))
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    if i == 0:
      A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    else:
      A[i+1] = relu(tf.matmul(W[i], A[i], name="A"+str(i+1)))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = (Y - A[n+1])
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/dsize
  lr = tf.Variable(0.00002, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  # add factor of 2 because our loss doesn't have 1/2
  B[n] = (-2*err/dsize)*relu_mask(A[n+1]) 
  print("Created B[%d] of shape %s"%(n, B[n].get_shape()))
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    print("Created B[%d] of shape %s"%(i, B[i].get_shape()))
    if i == 0:
      pass
    else:
      B[i] = B[i]*relu_mask(A[i+1])

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
  dWf = tf.concat([vectorize(grad) for grad in dW], axis=0)

  # inverse fisher preconditioner
  grads = tf.concat([khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  ifisher = pseudo_inverse(fisher)
  print("Matrix size ", ifisher.get_shape())


  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = v2c(Wf) - lr*(ifisher @ v2c(dWf))
  train_op1 = Wf_copy.assign(c2v(new_val_matrix))
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  # from mnist_linear.nb
  # {0.9, 0.888968, 0.878518, 0.868525, 0.858921, 0.849668, ...
  expected_losses = np.loadtxt("mnist_linear_losses0.csv")

  
  observed_losses = []
  for i in range(151):
    loss0 = sess.run([loss])[0]
    observed_losses.append(loss0)
    print("Loss is ", loss0)
    sess.run(train_op1)
    sess.run(train_op2)


  np.savetxt("data/mnist_fisher.csv", observed_losses,
             fmt="%.30f", delimiter=',')

  check_equal(observed_losses, expected_losses)
  print("mnist1000 passed")

USE_TINY = True
def mnist1000_fisher2():
  do_relu = True
  
  XY0 = np.genfromtxt('data/mnist_small.csv', delimiter= ",")
  fs = [1000, 50, 10]

  if USE_TINY:
    fs = [20, 50, 10]
    
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-fs[-1],:fs[0]]  # 50 x d
  Y0 = XY0[-fs[-1]:,:fs[0]]  # 10 x d
  dsize = X0.shape[1]
  f1 = X0.shape[0]
  f2 = fs[-1]
  assert fs[1] == 50
  assert f1 == 50

  X_ = tf.placeholder(dtype, shape=X0.shape)  # remove
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  # initial W0 predicts 0.1 for every instance for every class
  # W0 = np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T
  #  W = tf.Variable(W0.astype(dtype))

  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  print("shapes are ", dims)

  W0f = vectorize_np(np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T)
  #  W0f = np.ones((500,), dtype=dtype)
  W0s = unflatten_np(W0f, fs[1:])
  W0s.insert(0, X0)
  
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=(Wf_size,))
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten(Wf, fs[1:])
  print("W vars are %s"%([v.get_shape() for v in W]))
  W.insert(0, tf.constant(X0))
  print("W vars are now %s"%([v.get_shape() for v in W]))
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    if i == 0:
      A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    else:
      A[i+1] = relu(tf.matmul(W[i], A[i], name="A"+str(i+1)))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = (Y - A[n+1])
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/dsize
  lr = tf.Variable(0.0001, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  # add factor of 2 because our loss doesn't have 1/2
  B[n] = (-2*err/dsize)*relu_mask(A[n+1]) 
  print("Created B[%d] of shape %s"%(n, B[n].get_shape()))
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    print("Created B[%d] of shape %s"%(i, B[i].get_shape()))
    if i == 0:
      pass
    else:
      B[i] = B[i]*relu_mask(A[i+1])

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
  dWf = tf.concat([vectorize(grad) for grad in dW], axis=0)

  # inverse fisher preconditioner
  grads = tf.concat([khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  ifisher = pseudo_inverse(fisher)
  print("Matrix size ", ifisher.get_shape())


  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = v2c(Wf) - lr*(ifisher @ v2c(dWf))
  train_op1 = Wf_copy.assign(c2v(new_val_matrix))
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  # from mnist_linear.nb
  # {0.9, 0.888968, 0.878518, 0.868525, 0.858921, 0.849668, ...
  expected_losses = np.loadtxt("mnist_linear_losses0.csv")

  
  observed_losses = []
  for i in range(50):
    loss0 = sess.run([loss])[0]
    observed_losses.append(loss0)
    print("Loss is ", loss0)
    sess.run(train_op1)
    sess.run(train_op2)


  np.savetxt("data/mnist_fisher2.csv", observed_losses,
             fmt="%.30f", delimiter=',')

  check_equal(observed_losses, expected_losses)
  print("mnist1000 passed")

USE_TINY = True
def mnist1000_fisher3():
  do_relu = True
  
  XY0 = np.genfromtxt('data/mnist_small.csv', delimiter= ",")
  fs = [1000, 50, 10]

  if USE_TINY:
    fs = [20, 50, 10]
    
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-fs[-1],:fs[0]]  # 50 x d
  Y0 = XY0[-fs[-1]:,:fs[0]]  # 10 x d
  dsize = X0.shape[1]
  f1 = X0.shape[0]
  f2 = fs[-1]
  assert fs[1] == 50
  assert f1 == 50

  X_ = tf.placeholder(dtype, shape=X0.shape)  # remove
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  # initial W0 predicts 0.1 for every instance for every class
  # W0 = np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T
  #  W = tf.Variable(W0.astype(dtype))

  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  print("shapes are ", dims)

  W0f = vectorize_np(np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T)
  #  W0f = np.ones((500,), dtype=dtype)
  W0s = unflatten_np(W0f, fs[1:])
  W0s.insert(0, X0)
  
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=(Wf_size,))
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten(Wf, fs[1:])
  print("W vars are %s"%([v.get_shape() for v in W]))
  W.insert(0, tf.constant(X0))
  print("W vars are now %s"%([v.get_shape() for v in W]))
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    if i == 0:
      A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    else:
      A[i+1] = relu(tf.matmul(W[i], A[i], name="A"+str(i+1)))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = (Y - A[n+1])
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/dsize
  lr = tf.Variable(0.0001, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  # add factor of 2 because our loss doesn't have 1/2
  B[n] = (-2*err/dsize)*relu_mask(A[n+1]) 
  print("Created B[%d] of shape %s"%(n, B[n].get_shape()))
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    print("Created B[%d] of shape %s"%(i, B[i].get_shape()))
    if i == 0:
      pass
    else:
      B[i] = B[i]*relu_mask(A[i+1])

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
  dWf = tf.concat([vectorize(grad) for grad in dW], axis=0)

  # inverse fisher preconditioner
  grads = tf.concat([khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  ifisher = pseudo_inverse_sqrt(fisher)
  print("Matrix size ", ifisher.get_shape())


  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = v2c(Wf) - lr*(ifisher @ v2c(dWf))
  train_op1 = Wf_copy.assign(c2v(new_val_matrix))
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  # from mnist_linear.nb
  # {0.9, 0.888968, 0.878518, 0.868525, 0.858921, 0.849668, ...
  expected_losses = np.loadtxt("mnist_linear_losses0.csv")

  
  observed_losses = []
  for i in range(50):
    loss0 = sess.run([loss])[0]
    observed_losses.append(loss0)
    print("Loss is ", loss0)
    sess.run(train_op1)
    sess.run(train_op2)


  np.savetxt("data/mnist_fisher3.csv", observed_losses,
             fmt="%.30f", delimiter=',')

  check_equal(observed_losses, expected_losses)
  print("mnist1000 passed")

USE_TINY = True
def mnist1000_fisher4():
  do_relu = True
  
  XY0 = np.genfromtxt('data/mnist_small.csv', delimiter= ",")
  fs = [1000, 50, 10]

  if USE_TINY:
    fs = [20, 50, 10]
    
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-fs[-1],:fs[0]]  # 50 x d
  Y0 = XY0[-fs[-1]:,:fs[0]]  # 10 x d
  dsize = X0.shape[1]
  f1 = X0.shape[0]
  f2 = fs[-1]
  assert fs[1] == 50
  assert f1 == 50

  X_ = tf.placeholder(dtype, shape=X0.shape)  # remove
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  # initial W0 predicts 0.1 for every instance for every class
  # W0 = np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T
  #  W = tf.Variable(W0.astype(dtype))

  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  print("shapes are ", dims)

  W0f = vectorize_np(np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T)
  W0f = np.ones((500,), dtype=dtype)/500.
  W0s = unflatten_np(W0f, fs[1:])
  W0s.insert(0, X0)
  
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=(Wf_size,))
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten(Wf, fs[1:])
  print("W vars are %s"%([v.get_shape() for v in W]))
  W.insert(0, tf.constant(X0))
  print("W vars are now %s"%([v.get_shape() for v in W]))
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    if i == 0:
      A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    else:
      A[i+1] = relu(tf.matmul(W[i], A[i], name="A"+str(i+1)))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = (Y - A[n+1])
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/dsize
  lr = tf.Variable(.0001, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  # add factor of 2 because our loss doesn't have 1/2
  B[n] = (-2*err/dsize)*relu_mask(A[n+1]) 
  print("Created B[%d] of shape %s"%(n, B[n].get_shape()))
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    print("Created B[%d] of shape %s"%(i, B[i].get_shape()))
    if i == 0:
      pass
    else:
      B[i] = B[i]*relu_mask(A[i+1])

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
  dWf = tf.concat([vectorize(grad) for grad in dW], axis=0)

  # inverse fisher preconditioner
  grads = tf.concat([khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  ifisher = pseudo_inverse_sqrt(fisher)
  print("Matrix size ", ifisher.get_shape())


  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = v2c(Wf) - lr*(ifisher @ v2c(dWf))
  train_op1 = Wf_copy.assign(c2v(new_val_matrix))
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  # from mnist_linear.nb
  # {0.9, 0.888968, 0.878518, 0.868525, 0.858921, 0.849668, ...
  expected_losses = np.loadtxt("mnist_linear_losses0.csv")

  
  observed_losses = []
  for i in range(150):
    loss0 = sess.run([loss])[0]
    observed_losses.append(loss0)
    print("Loss is ", loss0)
    sess.run(train_op1)
    sess.run(train_op2)


  np.savetxt("data/mnist_fisher4.csv", observed_losses,
             fmt="%.30f", delimiter=',')

  check_equal(observed_losses, expected_losses)
  print("mnist1000 passed")

USE_TINY = True
def mnist1000_fisher5():
  do_relu = True
  
  XY0 = np.genfromtxt('data/mnist_small.csv', delimiter= ",")
  fs = [1000, 50, 10]

  if USE_TINY:
    fs = [20, 50, 10]
    
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-fs[-1],:fs[0]]  # 50 x d
  Y0 = XY0[-fs[-1]:,:fs[0]]  # 10 x d
  dsize = X0.shape[1]
  f1 = X0.shape[0]
  f2 = fs[-1]
  assert fs[1] == 50
  assert f1 == 50

  X_ = tf.placeholder(dtype, shape=X0.shape)  # remove
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  # initial W0 predicts 0.1 for every instance for every class
  # W0 = np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T
  #  W = tf.Variable(W0.astype(dtype))

  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  print("shapes are ", dims)

  W0f = vectorize_np(np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T)
  W0f = np.ones((500,), dtype=dtype)/500.
  W0s = unflatten_np(W0f, fs[1:])
  W0s.insert(0, X0)
  
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=(Wf_size,))
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten(Wf, fs[1:])
  print("W vars are %s"%([v.get_shape() for v in W]))
  W.insert(0, tf.constant(X0))
  print("W vars are now %s"%([v.get_shape() for v in W]))
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    if i == 0:
      A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    else:
      A[i+1] = relu(tf.matmul(W[i], A[i], name="A"+str(i+1)))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = (Y - A[n+1])
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/dsize
  lr = tf.Variable(.001, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  # add factor of 2 because our loss doesn't have 1/2
  B[n] = (-2*err/dsize)*relu_mask(A[n+1]) 
  print("Created B[%d] of shape %s"%(n, B[n].get_shape()))
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    print("Created B[%d] of shape %s"%(i, B[i].get_shape()))
    if i == 0:
      pass
    else:
      B[i] = B[i]*relu_mask(A[i+1])

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
  dWf = tf.concat([vectorize(grad) for grad in dW], axis=0)

  # inverse fisher preconditioner
  grads = tf.concat([khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  #  ifisher = pseudo_inverse_sqrt(fisher)
  ifisher = pseudo_inverse(fisher)
  print("Matrix size ", ifisher.get_shape())


  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = v2c(Wf) - lr*(ifisher @ v2c(dWf))
  train_op1 = Wf_copy.assign(c2v(new_val_matrix))
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  # from mnist_linear.nb
  # {0.9, 0.888968, 0.878518, 0.868525, 0.858921, 0.849668, ...
  expected_losses = np.loadtxt("mnist_linear_losses0.csv")

  
  observed_losses = []
  for i in range(500):
    loss0 = sess.run([loss])[0]
    observed_losses.append(loss0)
    print("Loss is ", loss0)
    sess.run(train_op1)
    sess.run(train_op2)


  np.savetxt("data/mnist_fisher5.csv", observed_losses,
             fmt="%.30f", delimiter=',')

  check_equal(observed_losses, expected_losses)
  print("mnist1000 passed")


USE_TINY = True
def mnist1000_fisher6():
  do_relu = True
  
  XY0 = np.genfromtxt('data/mnist_small.csv', delimiter= ",")
  fs = [1000, 50, 10, 10]

  if USE_TINY:
    fs[0] = 20
    
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-fs[-1],:fs[0]]  # 50 x d
  Y0 = XY0[-fs[-1]:,:fs[0]]  # 10 x d
  dsize = X0.shape[1]
  f1 = X0.shape[0]
  f2 = fs[-1]
  assert fs[1] == 50
  assert f1 == 50

  X_ = tf.placeholder(dtype, shape=X0.shape)  # remove
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  # initial W0 predicts 0.1 for every instance for every class
  # W0 = np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T
  #  W = tf.Variable(W0.astype(dtype))

  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  print("shapes are ", dims)

  W0f = vectorize_np(np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T)

  W0f = np.concatenate([W0f, vectorize_np(np.identity(10))])
  #  W0f = np.ones((500,), dtype=dtype)
  W0s = unflatten_np(W0f, fs[1:])
  W0s.insert(0, X0)
  
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=(Wf_size,))
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten(Wf, fs[1:])
  print("W vars are %s"%([v.get_shape() for v in W]))
  W.insert(0, tf.constant(X0))
  print("W vars are now %s"%([v.get_shape() for v in W]))
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    if i == 0:
      A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    else:
      A[i+1] = relu(tf.matmul(W[i], A[i], name="A"+str(i+1)))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = (Y - A[n+1])
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/dsize
  lr = tf.Variable(0.00002, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  # add factor of 2 because our loss doesn't have 1/2
  B[n] = (-2*err/dsize)*relu_mask(A[n+1]) 
  print("Created B[%d] of shape %s"%(n, B[n].get_shape()))
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    print("Created B[%d] of shape %s"%(i, B[i].get_shape()))
    if i == 0:
      pass
    else:
      B[i] = B[i]*relu_mask(A[i+1])

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
  dWf = tf.concat([vectorize(grad) for grad in dW], axis=0)

  # inverse fisher preconditioner
  grads = tf.concat([khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  ifisher = pseudo_inverse(fisher)
  print("Matrix size ", ifisher.get_shape())


  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = v2c(Wf) - lr*(ifisher @ v2c(dWf))
  train_op1 = Wf_copy.assign(c2v(new_val_matrix))
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  # from mnist_linear.nb
  # {0.9, 0.888968, 0.878518, 0.868525, 0.858921, 0.849668, ...
  expected_losses = np.loadtxt("mnist_linear_losses0.csv")

  
  observed_losses = []
  for i in range(150):
    loss0 = sess.run([loss])[0]
    observed_losses.append(loss0)
    print("Loss is ", loss0)
    sess.run(train_op1)
    sess.run(train_op2)


  np.savetxt("data/mnist_fisher6.csv", observed_losses,
             fmt="%.30f", delimiter=',')

  check_equal(observed_losses, expected_losses)
  print("mnist1000 passed")

USE_TINY = True
def mnist1000_fisher7():
  do_relu = True
  
  XY0 = np.genfromtxt('data/mnist_small.csv', delimiter= ",")
  fs = [1000, 50, 10, 10, 10]

  if USE_TINY:
    fs[0] = 20
    
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-fs[-1],:fs[0]]  # 50 x d
  Y0 = XY0[-fs[-1]:,:fs[0]]  # 10 x d
  dsize = X0.shape[1]
  f1 = X0.shape[0]
  f2 = fs[-1]
  assert fs[1] == 50
  assert f1 == 50

  X_ = tf.placeholder(dtype, shape=X0.shape)  # remove
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  # initial W0 predicts 0.1 for every instance for every class
  # W0 = np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T
  #  W = tf.Variable(W0.astype(dtype))

  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  print("shapes are ", dims)

  W0f = vectorize_np(np.concatenate((np.zeros([f1-1, f2]), np.ones([1, f2])/f2)).T)

  W0f = np.concatenate([W0f, vectorize_np(np.identity(10))])
  W0f = np.concatenate([W0f, vectorize_np(np.identity(10))])
  #  W0f = np.ones((500,), dtype=dtype)
  W0s = unflatten_np(W0f, fs[1:])
  W0s.insert(0, X0)
  
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=(Wf_size,))
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten(Wf, fs[1:])
  print("W vars are %s"%([v.get_shape() for v in W]))
  W.insert(0, tf.constant(X0))
  print("W vars are now %s"%([v.get_shape() for v in W]))
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    if i == 0:
      A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    else:
      A[i+1] = relu(tf.matmul(W[i], A[i], name="A"+str(i+1)))
      print("Created A[%d] of shape %s"%(i+1, A[i+1].get_shape()))
    
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = (Y - A[n+1])
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/dsize
  lr = tf.Variable(0.00002, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  # add factor of 2 because our loss doesn't have 1/2
  B[n] = (-2*err/dsize)*relu_mask(A[n+1]) 
  print("Created B[%d] of shape %s"%(n, B[n].get_shape()))
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    print("Created B[%d] of shape %s"%(i, B[i].get_shape()))
    if i == 0:
      pass
    else:
      B[i] = B[i]*relu_mask(A[i+1])

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
  dWf = tf.concat([vectorize(grad) for grad in dW], axis=0)

  # inverse fisher preconditioner
  grads = tf.concat([khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  ifisher = pseudo_inverse(fisher)
  print("Matrix size ", ifisher.get_shape())


  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = v2c(Wf) - lr*(ifisher @ v2c(dWf))
  train_op1 = Wf_copy.assign(c2v(new_val_matrix))
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  # from mnist_linear.nb
  # {0.9, 0.888968, 0.878518, 0.868525, 0.858921, 0.849668, ...
  expected_losses = np.loadtxt("mnist_linear_losses0.csv")

  
  observed_losses = []
  for i in range(150):
    loss0 = sess.run([loss])[0]
    observed_losses.append(loss0)
    print("Loss is ", loss0)
    sess.run(train_op1)
    sess.run(train_op2)


  np.savetxt("data/mnist_fisher7.csv", observed_losses,
             fmt="%.30f", delimiter=',')

  check_equal(observed_losses, expected_losses)
  print("mnist1000 passed")


if __name__ == '__main__':
  train_tiny()
  #  mnist1000_test()
  #  mnist1000_fisher()
  #  mnist1000_fisher2()

  # simple working example
  #  mnist1000_fisher2()

  # try with symmetric sqrt of fisher
  #  mnist1000_fisher3()

  # bad initial conditions (all ones)
  # mnist1000_fisher4()

  # bad initial conditions, using Fisher normalization
  #  mnist1000_fisher5()
  #  mnist1000_fisher6()

  # add another hidden layer (4 matmuls total)
  #  mnist1000_fisher7()
