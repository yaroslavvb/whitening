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

def kr(A, B):
  return u.kronecker(A, B, do_shape_inference=False)


def gradient(lr0):
  init_dict[lr_holder] = lr0

  # gradient update rule
  train_op = grad_update(Wf - lr * dWf)

  return do_run(train_op)

def newton(lr0):
  init_dict[lr_holder] = lr0

  # todo, get rid of B's
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
  train_op = grad_update(Wf - lr * ihess @ dWf)
  return do_run(train_op)


def newton_bd(lr0):
  init_dict[lr_holder] = lr0

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

  # todo -- figure out why this is not the same as block inversion
  # grads = tf.concat([u.khatri_rao(A[i], Bn[i]) for i in range(1, n+1)], axis=0)
  # hess = grads @ tf.transpose(grads) / dsize
  # blocks = u.partition_matrix_evenly(hess, 10)
  ihess = u.concat_blocks(u.block_diagonal_inverse(blocks))
  train_op = grad_update(Wf - lr * ihess @ dWf)
  return do_run(train_op)


def newton_kfac(lr0):
  init_dict[lr_holder] = lr0

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
  
  train_op = grad_update(Wf - lr * ihess @ dWf)
  return do_run(train_op)


def natural_empirical(lr0):
  init_dict[lr_holder] = lr0

  grads = tf.concat([u.khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  ifisher = u.pseudo_inverse(fisher)
  
  train_op = grad_update(Wf - lr * ifisher @ dWf)
  return do_run(train_op)

def natural_sampled(lr0, num_samples=1):
  def kr(A, B):
    return u.kronecker(A, B, do_shape_inference=False)
  init_dict[lr_holder] = lr0
  np.random.seed(0)
  tf.set_random_seed(0)

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

  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B2 = [0]*(n+1)
  B[n] = -err/dsize
  B2[n] = tf.random_normal((f(n), dsize*num_samples), 0, 1, dtype=dtype, seed=0)
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    B2[i] = tf.matmul(tf.transpose(W[i+1]), B2[i+1], name="B2"+str(i))

  grads = tf.concat([u.khatri_rao(A2[i], B2[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / (dsize*num_samples)
  ifisher = u.pseudo_inverse(fisher)
  train_op = grad_update(Wf - lr * ifisher @ dWf)
  return do_run(train_op)


def natural_bd(lr0, num_samples=1):
  init_dict[lr_holder] = lr0
  np.random.seed(0)
  tf.set_random_seed(0)

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

  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B2 = [0]*(n+1)
  B[n] = -err/dsize
  B2[n] = tf.random_normal((f(n), dsize*num_samples), 0, 1, seed=0,
                           dtype=dtype)
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    B2[i] = tf.matmul(tf.transpose(W[i+1]), B2[i+1], name="B2"+str(i))

  grads = tf.concat([u.khatri_rao(A2[i], B2[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / (dsize*num_samples)
  blocks = u.partition_matrix_evenly(fisher, 10)
  #  ifisher = u.pseudo_inverse(fisher)
  ifisher = u.concat_blocks(u.block_diagonal_inverse(blocks))
  train_op = grad_update(Wf - lr * ifisher @ dWf)
  return do_run(train_op)

def natural_bd_sqrt(lr0, num_samples=1):
  init_dict[lr_holder] = lr0
  np.random.seed(0)
  tf.set_random_seed(0)

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

  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B2 = [0]*(n+1)
  B[n] = -err/dsize
  B2[n] = tf.random_normal((f(n), dsize*num_samples), 0, 1, seed=0,
                           dtype=dtype)
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    B2[i] = tf.matmul(tf.transpose(W[i+1]), B2[i+1], name="B2"+str(i))

  grads = tf.concat([u.khatri_rao(A2[i], B2[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / (dsize*num_samples)
  blocks = u.partition_matrix_evenly(fisher, 10)
  #  ifisher = u.pseudo_inverse(fisher)
  ifisher = u.concat_blocks(u.block_diagonal_inverse_sqrt(blocks))
  train_op = grad_update(Wf - lr * ifisher @ dWf)
  return do_run(train_op)


def natural_kfac(lr0, num_samples=1):
  init_dict[lr_holder] = lr0
  np.random.seed(0)
  tf.set_random_seed(0)

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

  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B2 = [0]*(n+1)
  B[n] = -err/dsize
  B2[n] = tf.random_normal((f(n), dsize*num_samples), 0, 1, seed=0,
                           dtype=dtype)
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    B2[i] = tf.matmul(tf.transpose(W[i+1]), B2[i+1], name="B2"+str(i))

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
  train_op = grad_update(Wf - lr * ifisher @ dWf)
  return do_run(train_op)


do_run_iters = 5
def do_run(train_op):
  sess = setup_session()
  observed_losses = []
  u.reset_time()
  for i in range(do_run_iters):
    loss0 = sess.run(loss)
    print(loss0)
    observed_losses.append(loss0)
    sess.run(train_op)
    u.record_time()
  u.summarize_time()
  return observed_losses

  
def setup_session():
  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  return sess

def grad_update(new_val):
  copy_op = Wf_copy.assign(new_val)
  with tf.control_dependencies([copy_op]):
    train_op = Wf.assign(Wf_copy)
  return train_op

def newton_test():
  global do_run_iters
  do_run_iters = 5
  result = newton(1.0)
  expected_result = [8.9023744225439743e-05, 0.060120791316053412, 0.0059295249954177918, 1.9856240803246437e-05, 2.7125563957575423e-10]
  
  u.check_equal(result, expected_result)
  # min: 733.18, median: 773.05, times: 1738.55,733.18,773.05 ,761.35,790.06
  assert min(u.global_time_list)>0.1 and min(u.global_time_list)<2.0

def natural_empirical_test():
  global do_run_iters
  do_run_iters = 5
  result = natural_empirical(0.000000002)
  expected_result = [8.9023744225439743e-05, 8.8836514644521851e-05, 8.8656278865734025e-05, 8.8480133369324773e-05, 8.830467671675067e-05]
  u.check_equal(result, expected_result)

  #  min: 674.83, median: 707.58, times: 700.63,674.83,721.99,707.58,714.83
  assert min(u.global_time_list)>0.1 and min(u.global_time_list)<1.0

def natural_sampled_test():
  result = natural_sampled(lr0=0.1, num_samples=5)
  expected_result = [8.90237442254e-05,
                     8.84573144923e-05,
                     7.38962710111e-05,
                     5.92464713306e-05,
                     4.34063312257e-05]
  u.check_equal(result, expected_result)
  # min: 626.93, median: 637.17, times: 639.94,637.17,631.02,626.93,650.61
  assert min(u.global_time_list)>0.1 and min(u.global_time_list)<2.0


def natural_bd_test():
  result = natural_bd(lr0=0.01, num_samples=5)
  expected_result = [8.90237442254e-05,
                     8.85624107242e-05,
                     8.16307998223e-05,
                     5.99453602166e-05,
                     4.25738898355e-05]
  u.check_equal(result, expected_result)
  # min: 14.54, median: 14.96, times: 89.05,14.96,14.62,15.37,14.54
  assert min(u.global_time_list)>0.002 and min(u.global_time_list)<0.1

def natural_kfac_test():
  result = natural_kfac(lr0=0.01, num_samples=5)
  expected_result = [8.90237442254e-05,
                     7.02310673526e-05,
                     5.89887049651e-05,
                     5.10372615014e-05,
                     4.20940214038e-05]
  u.check_equal(result, expected_result)
  # min: 8.46, median: 9.01, times: 140.04,9.89,8.49,8.46,9.01
  assert min(u.global_time_list)>0.001 and min(u.global_time_list)<0.1

def newton_bd_test():
  result = newton_bd(0.1)
  expected_result = [8.90237442254e-05,
                     3.56141425886e-06,
                     1.42451943222e-07,
                     5.69802736553e-09,
                     2.27920670486e-10]
  u.check_equal(result, expected_result)
  #  min: 15.64, median: 16.62, times: 275.65,17.32,16.25,15.64,16.62
  assert min(u.global_time_list)>0.001 and min(u.global_time_list)<0.1

def newton_kfac_test():
  result = newton_kfac(0.1)
  expected_result = [8.90237442254e-05,
                     3.56141425548e-06,
                     1.42451943136e-07,
                     5.69802735848e-09,
                     2.27920670137e-10]
  u.check_equal(result, expected_result)
  # min: 8.05, median: 8.24, times: 209.63,8.86,8.05,8.24,8.18
  assert min(u.global_time_list)>0.001 and min(u.global_time_list)<0.1

if __name__ == '__main__':
  # Compare a set of algorithms on rotations problem

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
  lr_holder = tf.placeholder(dtype=dtype, shape=())
  lr = tf.Variable(lr_holder, dtype=dtype)

  # run tests
  newton_test()
  natural_empirical_test()

  natural_sampled_test()
  natural_bd_test()
  natural_kfac_test()
  newton_bd_test()
  newton_kfac_test()
  
  print("%s tests passed" %(sys.argv[0]))
