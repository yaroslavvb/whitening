# Mac iteration time: 1606 ms
# Linux 1080 TI iteration time: 132 ms

# conventions. "_op things are ops"
# "x0" means numpy
# _live means it's used to update a variable value
# experiment prefixes
prefix = "small_final" # for checkin


# for line profiling
try:
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x   # if it's not defined simply ignore the decorator.

from tensorflow.python.ops import variables
def passthrough(obj, value): return value
try:
  variables.Variable._build_initializer_expr=passthrough
except: # older versions of TF don't have this
  pass


import util
import util as u

drop_l2 = True               # drop L2 term
drop_sparsity = True         # drop KL term
use_gpu = False
do_line_search = False       # line-search and dump values at each iter

import sys
whitening_mode = 4                 # 0 for gradient, 4 for full whitening
whiten_every_n_steps = 1           # how often to whiten
report_frequency = 1               # how often to print loss

num_steps = 20000 if whitening_mode==0 else 20
util.USE_MKL_SVD=True                   # Tensorflow vs MKL SVD

purely_linear = False  # convert sigmoids into linear nonlinearities
use_tikhonov = True    # use Tikhonov reg instead of Moore-Penrose pseudo-inv
Lambda = 1e-3          # magic lambda value from Jimmy Ba for Tikhonov

# adaptive line search
adaptive_step = False     # adjust step length based on predicted decrease
adaptive_step_frequency = 1 # how often to adjust
adaptive_step_burn_in = 0 # let optimization go for a bit before adjusting
local_quadratics = False  # use quadratic approximation to predict loss drop

import networkx as nx
import load_MNIST
import numpy as np
import scipy.io           # for loadmat
from scipy import linalg  # for svd
import math
import time

import os, sys
if use_gpu:
  os.environ['CUDA_VISIBLE_DEVICES']='0'
else:
  os.environ['CUDA_VISIBLE_DEVICES']=''

import tensorflow as tf
from util import t  # transpose


def W_uniform(s1, s2): # uniform weight init from Ng UFLDL
  r = np.sqrt(6) / np.sqrt(s1 + s2 + 1)
  result = np.random.random(2*s2*s1)*2*r-r
  return result

@profile
def main():
  np.random.seed(0)
  tf.set_random_seed(0)
  
  dtype = np.float32
  # 64-bit doesn't help much, search for 64-bit in
  # https://www.wolframcloud.com/objects/5f297f41-30f7-4b1b-972c-cac8d1f8d8e4
  u.default_dtype = dtype
  machine_epsilon = np.finfo(dtype).eps # 1e-7 or 1e-16
  train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
  dsize = 10000
  patches = train_images[:,:dsize];
  fs = [dsize, 28*28, 196, 28*28]

  # values from deeplearning.stanford.edu/wiki/index.php/UFLDL_Tutorial
  X0=patches
  lambda_=3e-3
  rho=tf.constant(0.1, dtype=dtype)
  beta=3
  W0f = W_uniform(fs[2],fs[3])

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = f(-1)
  n = len(fs) - 2

  # helper to create variables with numpy or TF initial value
  init_dict = {}     # {var_placeholder: init_value}
  vard = {}          # {var: util.VarInfo}
  def init_var(val, name, trainable=False, noinit=False):
    if isinstance(val, tf.Tensor):
      collections = [] if noinit else None
      var = tf.Variable(val, name=name, collections=collections)
    else:
      val = np.array(val)
      assert u.is_numeric, "Unknown type"
      holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
      var = tf.Variable(holder, name=name, trainable=trainable)
      init_dict[holder] = val
    var_p = tf.placeholder(var.dtype, var.shape)
    var_setter = var.assign(var_p)
    vard[var] = u.VarInfo(var_setter, var_p)
    return var

  lr = init_var(0.2, "lr")
  if purely_linear:   # need lower LR without sigmoids
    lr = init_var(.02, "lr")
    
  Wf = init_var(W0f, "Wf", True)
  Wf_copy = init_var(W0f, "Wf_copy", True)
  W = u.unflatten(Wf, fs[1:])   # perftodo: this creates transposes
  X = init_var(X0, "X")
  W.insert(0, X)

  def sigmoid(x):
    if not purely_linear:
      return tf.sigmoid(x)
    else:
      return tf.identity(x)
      
  def d_sigmoid(y):
    if not purely_linear:
      return y*(1-y)
    else:
      return 1
    
  def kl(x, y):
    return x * tf.log(x / y) + (1 - x) * tf.log((1 - x) / (1 - y))
  def d_kl(x, y):
    return (1-x)/(1-y) - x/y
  
  # A[i] = activations needed to compute gradient of W[i]
  # A[n+1] = network output
  A = [None]*(n+2)

  # A[0] is just for shape checks, assert fail on run
  # tf.assert always fails because of static assert
  # fail_node = tf.assert_equal(1, 0, message="too huge")
  fail_node = tf.Print(0, [0], "fail, this must never run")
  with tf.control_dependencies([fail_node]):
    A[0] = u.Identity(dsize, dtype=dtype)
  A[1] = W[0]
  for i in range(1, n+1):
    A[i+1] = sigmoid(W[i] @ A[i])
    
  # reconstruction error and sparsity error
  err = (A[3] - A[1])
  rho_hat = tf.reduce_sum(A[2], axis=1, keep_dims=True)/dsize

  # B[i] = backprops needed to compute gradient of W[i]
  # B2[i] = backprops from sampled labels needed for natural gradient
  B = [None]*(n+1)
  B2 = [None]*(n+1)
  B[n] = err*d_sigmoid(A[n+1])
  sampled_labels_live = tf.random_normal((f(n), f(-1)), dtype=dtype, seed=0)
  sampled_labels = init_var(sampled_labels_live, "sampled_labels", noinit=True)
  B2[n] = sampled_labels*d_sigmoid(A[n+1])
  for i in range(n-1, -1, -1):
    backprop = t(W[i+1]) @ B[i+1]
    backprop2 = t(W[i+1]) @ B2[i+1]
    if i == 1 and not drop_sparsity:
      backprop += beta*d_kl(rho, rho_hat)
      backprop2 += beta*d_kl(rho, rho_hat)
    B[i] = backprop*d_sigmoid(A[i+1])
    B2[i] = backprop2*d_sigmoid(A[i+1])

  # dW[i] = gradient of W[i]
  dW = [None]*(n+1)
  pre_dW = [None]*(n+1)  # preconditioned dW
  pre_dW_stable = [None]*(n+1)  # preconditioned stable dW

  cov_A = [None]*(n+1)    # covariance of activations[i]
  cov_B2 = [None]*(n+1)   # covariance of synthetic backprops[i]
  vars_svd_A = [None]*(n+1)
  vars_svd_B2 = [None]*(n+1)
  for i in range(1,n+1):
    cov_A[i] = init_var(A[i]@t(A[i])/dsize, "cov_A%d"%(i,))
    cov_B2[i] = init_var(B2[i]@t(B2[i])/dsize, "cov_B2%d"%(i,))
    vars_svd_A[i] = u.SvdWrapper(cov_A[i],"svd_A_%d"%(i,))
    vars_svd_B2[i] = u.SvdWrapper(cov_B2[i],"svd_B2_%d"%(i,))
    if use_tikhonov:
      whitened_A = u.regularized_inverse2(vars_svd_A[i],L=Lambda) @ A[i]
    else:
      whitened_A = u.pseudo_inverse2(vars_svd_A[i]) @ A[i]
    if use_tikhonov:
      whitened_B2 = u.regularized_inverse2(vars_svd_B2[i],L=Lambda) @ B[i]
    else:
      whitened_B2 = u.pseudo_inverse2(vars_svd_B2[i]) @ B[i]
    whitened_A_stable = u.pseudo_inverse_sqrt2(vars_svd_A[i]) @ A[i]
    whitened_B2_stable = u.pseudo_inverse_sqrt2(vars_svd_B2[i]) @ B[i]
    pre_dW[i] = (whitened_B2 @ t(whitened_A))/dsize
    pre_dW_stable[i] = (whitened_B2_stable @ t(whitened_A_stable))/dsize
    dW[i] = (B[i] @ t(A[i]))/dsize

  # Loss function
  reconstruction = u.L2(err) / (2 * dsize)
  sparsity = beta * tf.reduce_sum(kl(rho, rho_hat))
  L2 = (lambda_ / 2) * (u.L2(W[1]) + u.L2(W[1]))

  loss = reconstruction
  if not drop_l2:
    loss = loss + L2
  if not drop_sparsity:
    loss = loss + sparsity

  grad_live = u.flatten(dW[1:])
  pre_grad_live = u.flatten(pre_dW[1:]) # fisher preconditioned gradient
  pre_grad_stable_live = u.flatten(pre_dW_stable[1:]) # sqrt fisher preconditioned grad
  grad = init_var(grad_live, "grad")
  pre_grad = init_var(pre_grad_live, "pre_grad")
  pre_grad_stable = init_var(pre_grad_stable_live, "pre_grad_stable")

  update_params_op = Wf.assign(Wf-lr*pre_grad).op
  update_params_stable_op = Wf.assign(Wf-lr*pre_grad_stable).op
  save_params_op = Wf_copy.assign(Wf).op
  pre_grad_dot_grad = tf.reduce_sum(pre_grad*grad)
  pre_grad_stable_dot_grad = tf.reduce_sum(pre_grad*grad)
  grad_norm = tf.reduce_sum(grad*grad)
  pre_grad_norm = u.L2(pre_grad)
  pre_grad_stable_norm = u.L2(pre_grad_stable)

  def dump_svd_info(step):
    """Dump singular values and gradient values in those coordinates."""
    for i in range(1, n+1):
      svd = vars_svd_A[i]
      s0, u0, v0 = sess.run([svd.s, svd.u, svd.v])
      util.dump(s0, "A_%d_%d"%(i, step))
      A0 = A[i].eval()
      At0 = v0.T @ A0
      util.dump(A0 @ A0.T, "Acov_%d_%d"%(i, step))
      util.dump(At0 @ At0.T, "Atcov_%d_%d"%(i, step))
      util.dump(s0, "As_%d_%d"%(i, step))

    for i in range(1, n+1):
      svd = vars_svd_B2[i]
      s0, u0, v0 = sess.run([svd.s, svd.u, svd.v])
      util.dump(s0, "B2_%d_%d"%(i, step))
      B0 = B[i].eval()
      Bt0 = v0.T @ B0
      util.dump(B0 @ B0.T, "Bcov_%d_%d"%(i, step))
      util.dump(Bt0 @ Bt0.T, "Btcov_%d_%d"%(i, step))
      util.dump(s0, "Bs_%d_%d"%(i, step))      
    
  def advance_batch():
    sess.run(sampled_labels.initializer)  # new labels for next call

  def update_covariances():
    ops_A = [cov_A[i].initializer for i in range(1, n+1)]
    ops_B2 = [cov_B2[i].initializer for i in range(1, n+1)]
    sess.run(ops_A+ops_B2)

  def update_svds():
    if whitening_mode>1:
      vars_svd_A[2].update()
    if whitening_mode>2:
      vars_svd_B2[2].update()
    if whitening_mode>3:
      vars_svd_B2[1].update()

  def init_svds():
    """Initialize our SVD to identity matrices."""
    ops = []
    for i in range(1, n+1):
      ops.extend(vars_svd_A[i].init_ops)
      ops.extend(vars_svd_B2[i].init_ops)
    sess = tf.get_default_session()
    sess.run(ops)
      
  init_op = tf.global_variables_initializer()
  #  tf.get_default_graph().finalize()

  from tensorflow.core.protobuf import rewriter_config_pb2
  
  rewrite_options = rewriter_config_pb2.RewriterConfig(
    disable_model_pruning=True,
    constant_folding=rewriter_config_pb2.RewriterConfig.OFF,
    memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL)
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  graph_options=tf.GraphOptions(optimizer_options=optimizer_options,
                                rewrite_options=rewrite_options)
  config = tf.ConfigProto(graph_options=graph_options)
  #sess = tf.Session(config=config)
  sess = tf.InteractiveSession(config=config)
  sess.run(Wf.initializer, feed_dict=init_dict)
  sess.run(X.initializer, feed_dict=init_dict)
  advance_batch()
  update_covariances()
  init_svds()
  sess.run(init_op, feed_dict=init_dict)  # initialize everything else
  
  print("Running training.")
  u.reset_time()

  step_lengths = []     # keep track of learning rates
  losses = []
  ratios = []           # actual loss decrease / expected decrease
  grad_norms = []       
  pre_grad_norms = []   # preconditioned grad norm squared
  pre_grad_stable_norms = [] # sqrt preconditioned grad norms squared
  target_delta_list = []     # predicted decrease linear approximation
  target_delta2_list = []    # predicted decrease quadratic appromation
  actual_delta_list = []      # actual decrease
  
  # adaptive line search parameters
  alpha=0.3   # acceptable fraction of predicted decrease
  beta=0.8    # how much to shrink when violation
  growth_rate=1.05  # how much to grow when too conservative
    
  def update_cov_A(i):
    sess.run(cov_A[i].initializer)
  def update_cov_B2(i):
    sess.run(cov_B2[i].initializer)

  # only update whitening matrix of input activations in the beginning
  if whitening_mode>0:
    vars_svd_A[1].update()

  # compute t(delta).H.delta/2
  def hessian_quadratic(delta):
    #    update_covariances()
    W = u.unflatten(delta, fs[1:])
    W.insert(0, None)
    total = 0
    for l in range(1, n+1):
      decrement = tf.trace(t(W[l])@cov_B2[l]@W[l]@cov_A[l])
      total+=decrement
    return (total/2).eval()
    
  # compute t(delta).H^-1.delta/2
  def hessian_quadratic_inv(delta):
    #    update_covariances()
    W = u.unflatten(delta, fs[1:])
    W.insert(0, None)
    total = 0
    for l in range(1, n+1):
      invB2 = u.pseudo_inverse2(vars_svd_B2[l])
      invA = u.pseudo_inverse2(vars_svd_A[l])
      decrement = tf.trace(t(W[l])@invB2@W[l]@invA)
      total+=decrement
    return (total/2).eval()

  # do line search, dump values as csv
  def line_search(initial_value, direction, step, num_steps):
    saved_val = tf.Variable(Wf)
    sess.run(saved_val.initializer)
    pl = tf.placeholder(dtype, shape=(), name="linesearch_p")
    assign_op = Wf.assign(initial_value - direction*step*pl)
    vals = []
    for i in range(num_steps):
      sess.run(assign_op, feed_dict={pl: i})
      vals.append(loss.eval())
    sess.run(Wf.assign(saved_val)) # restore original value
    return vals
    
  for step in range(num_steps): 
    update_covariances()
    if step % whiten_every_n_steps==0:
      update_svds()

    sess.run(grad.initializer)
    sess.run(pre_grad.initializer)
    
    lr0, loss0 = sess.run([lr, loss])
    save_params_op.run()

    # regular inverse becomes unstable when grad norm exceeds 1
    stabilized_mode = grad_norm.eval()<1

    if stabilized_mode and not use_tikhonov:
      update_params_stable_op.run()
    else:
      update_params_op.run()

    loss1 = loss.eval()
    advance_batch()

    # line search stuff
    target_slope = (-pre_grad_dot_grad.eval() if stabilized_mode else
                    -pre_grad_stable_dot_grad.eval())
    target_delta = lr0*target_slope
    target_delta_list.append(target_delta)

    # second order prediction of target delta
    # TODO: the sign is wrong, debug this
    # https://www.wolframcloud.com/objects/8f287f2f-ceb7-42f7-a599-1c03fda18f28
    if local_quadratics:
      x0 = Wf_copy.eval()
      x_opt = x0-pre_grad.eval()
      # computes t(x)@H^-1 @(x)/2
      y_opt = loss0 - hessian_quadratic_inv(grad)
      # computes t(x)@H @(x)/2
      y_expected = hessian_quadratic(Wf-x_opt)+y_opt
      target_delta2 = y_expected - loss0
      target_delta2_list.append(target_delta2)
      
    
    actual_delta = loss1 - loss0
    actual_slope = actual_delta/lr0
    slope_ratio = actual_slope/target_slope  # between 0 and 1.01
    actual_delta_list.append(actual_delta)

    if do_line_search:
      vals1 = line_search(Wf_copy, pre_grad, lr/100, 40)
      vals2 = line_search(Wf_copy, grad, lr/100, 40)
      u.dump(vals1, "line1-%d"%(i,))
      u.dump(vals2, "line2-%d"%(i,))
      
    losses.append(loss0)
    step_lengths.append(lr0)
    ratios.append(slope_ratio)
    grad_norms.append(grad_norm.eval())
    pre_grad_norms.append(pre_grad_norm.eval())
    pre_grad_stable_norms.append(pre_grad_stable_norm.eval())

    if step % report_frequency == 0:
      print("Step %d loss %.2f, target decrease %.3f, actual decrease, %.3f ratio %.2f grad norm: %.2f pregrad norm: %.2f"%(step, loss0, target_delta, actual_delta, slope_ratio, grad_norm.eval(), pre_grad_norm.eval()))
    
    if adaptive_step_frequency and adaptive_step and step>adaptive_step_burn_in:
      # shrink if wrong prediction, don't shrink if prediction is tiny
      if slope_ratio < alpha and abs(target_delta)>1e-6 and adaptive_step:
        print("%.2f %.2f %.2f"%(loss0, loss1, slope_ratio))
        print("Slope optimality %.2f, shrinking learning rate to %.2f"%(slope_ratio, lr0*beta,))
        sess.run(vard[lr].setter, feed_dict={vard[lr].p: lr0*beta})
        
      # grow learning rate, slope_ratio .99 worked best for gradient
      elif step>0 and i%50 == 0 and slope_ratio>0.90 and adaptive_step:
          print("%.2f %.2f %.2f"%(loss0, loss1, slope_ratio))
          print("Growing learning rate to %.2f"%(lr0*growth_rate))
          sess.run(vard[lr].setter, feed_dict={vard[lr].p:
                                               lr0*growth_rate})

    u.record_time()

  # check against expected loss
  if 'Apple' in sys.version:
    pass
    #    u.dump(losses, "kfac_small_final_mac.csv")
    targets = np.loadtxt("data/kfac_small_final_mac.csv", delimiter=",")
  else:
    pass
    #    u.dump(losses, "kfac_small_final_linux.csv")
    targets = np.loadtxt("data/kfac_small_final_linux.csv", delimiter=",")

  u.check_equal(targets, losses[:len(targets)], rtol=1e-1)
  u.summarize_time()
  print("Test passed")


if __name__=='__main__':
  main()
