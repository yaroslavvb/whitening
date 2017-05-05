do_early_init = False  # True if we want to initialize model vars as part of constructor
regularize_covariances = True  # add identity*lambda to covariance matrices

inverse_methods = ["pseudo_inverse", "inverse"]
inverse_method = "inverse"
assert inverse_method in inverse_methods

adaptive_step = False     # adjust step length based on predicted decrease
step_rejection = False    # reject bad steps

import sys
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from collections import defaultdict

dtype = np.float32   # TODO: factor out dtype to allow switching
eps = np.finfo(dtype).eps # 1e-7 or 1e-16
one = tf.ones((), dtype=dtype)

import util as u
import util
from util import t  # transpose
from util import VarStruct

class Model:
  def __init__(self):
    self.loss = None            # loss
    self.loss2 = None           # loss wrt synthetic labels
    self.vloss = None           # validation loss (fixed data baked into model)
    self.advance_batch = None   # function that advances batch
    self.global_vars = []       # all global variables of the model
    self.local_vars = []        # all local variables of the model
    self.trainable_vars = []    # trainable vars
    self.initialize_local_vars = None  # initialize local variables
    self.initialize_global_vars = None # initialize global variables (variables
                                # shared across all instances of model)
    self.extra = {}             # extra data


class IndexedGrad:
  """OrderedDict-like object representing a set of cached gradients indexed by
  variables wrt to which the grad was taken. Values of gradient are stored in
  variables and must be explictly updated using "update" method.

  Implements Python dictionary interface delegating to underlying
  {var: cached_grad} dict

  g = IndexedGrad(vars_=vars_, grads=grads)  # create from existing grad tensors
  g = IndexedGrad(loss=loss, vars_=vars_) # use tf.gradients to get grad tensors
  g.f                          # concated vector version of all cached grads
  g.update()                   # updates cached value of grads
  g.live                       # live version of gradients [tensor1,tensor2...
  g.cached                     # cached version of grads [var1,var2...
  g[param]                     # cached version of grad for param

  # TODO: need better way to get live gradients out
  """
  
  def __init__(self, *forbidden, grads=None, vars_=None, loss=None):
    assert len(forbidden) == 0  # force params to be keyword-only
    
    assert vars_ is not None
    if grads==None:
      assert loss is not None
      grads = tf.gradients(loss, vars_)
      
    self.vars_ = vars_
    self.grads_dict = OrderedDict()
    self.cached = []
    self.live = []
    update_ops = []
    for grad, var in zip(grads, vars_):
      self.live.append(grad)
      cached = tf.Variable(grad, var.name+"_grad_cached")
      self.cached.append(cached)
      self.grads_dict[var] = cached
      update_ops.append(cached.initializer)
    self.update_op = tf.group(*update_ops)
    self.f = u.flatten(self.cached)
    self.cached = VarList(self.cached)
    assert list(self.cached) == list(self.grads_dict.values())

  def update(self):
    """Upgrade cached gradients using from their live values using
    default session."""
    sess = tf.get_default_session()
    sess.run(self.update_op)

  def __len__(self):
    return self.grads_dict.__len__()
  def __length_hint__(self):
    return self.grads_dict.__length_hint__()
  def __getitem__(self, key):
    return self.grads_dict.__getitem__(key)
  def __missing__(self, key):
    return self.grads_dict.__missing__(key)
  def __iter__(self):
    return self.grads_dict.__iter__()
  def __reversed__(self):
    return self.grads_dict.__reversed__()
  def __contains__(self, item):
    return self.grads_dict.__contains__(item)

  def from_grads_and_vars(grads_and_vars):
    grads = [g[0] for g in grads_and_vars]
    vars_ = [g[1] for g in grads_and_vars]
    return IndexedGrad(grads=grads, vars_=vars_)

  def to_grads_and_vars(self):
    grads = list(self.cached)
    vars_ = list(self.vars_)
    return zip(grads, vars_)


class VarList:
  """Class to simplify dealing with groups of variables. Acts like a regular
  TensorFlow variable, with helper methods to extract sub-components.

  Implements Python list interface delegating to underlying list of
  variables.

  a = VarList([tf.Variable(),tf.Variable()])
  b = a.copy()            # new VarList of same sizes
  a.f                     # flattened representation of variable list
  a[0]                    # tf.Variable object, first
  sess.run(b.assign(a))
  """

  def __init__(self, vars_, name=None):
    for v in vars_:
      assert isinstance(v, tf.Variable)
    assert vars_
    
    self.vars_ = vars_
    self.f = u.flatten(vars_)
    if name is None:  # take name from first variable
      self.name = "VarList_"+vars_[0].op.name
    else:
      self.name = name
      
  def copy(self):
    """Create a copy of given VarList. New vars_ depend on existing vars for
    initialization."""
    
    var_copies = [tf.Variable(var.initialized_value, name=var.op.name+"_copy")
                  for var in self.vars_]
    return VarList(var_copies, name="copy_"+self.name)
  
  def assign(self, other):
    """Creates group of assign ops that copies value of current VarList to
    other VarList."""

    assert isinstance(other, VarList)
    assert len(self) == len(other)
    ops = []
    for (my_var, other_var) in zip(self.vars_, other.vars_):
      ops.append(my_var.assign(other_var))
    return tf.group(*ops, name="assign_"+self.name)

  def sub(self, other, weight=one):
    """Returns an op that subtracts other from current VarList.

    sess.run(a.sub(b))   # subtracts b from a
    """
    assert isinstance(other, VarList)
    assert len(self) == len(other)
    
    ops = []
    if isinstance(weight, VarStruct):
      weight = weight.var
      
    for (my_var, other_var) in zip(self.vars_, other.vars_):
      ops.append(my_var.assign_sub(weight*other_var))
    return tf.group(*ops, name="sub_"+self.name)

  def __iter__(self):
    return self.vars_.__iter__()
  def __getitem__(self, key):
    return self.vars_.__getitem__(key)
  def __len__(self):
    return self.vars_.__len__()

def get_batch_size(data):
  if isinstance(data, IndexedGrad):
    return int(data.live[0].shape[1])
  else:
    return int(data.shape[1])

class Covariance():
  """Convenience structure to keep covariance of data tensor as well
  as decomposition of this covariance.
  """
  
  def __init__(self, data, var, prefix, Lambda):
    dsize = get_batch_size(data)
    cov_op = data @ t(data) / dsize
    if regularize_covariances:
      cov_op = cov_op + Lambda*u.Identity(cov_op.shape[0])
      
    cov_name = "%s_cov_%s" %(prefix, var.op.name)
    svd_name = "%s_svd_%s" %(prefix, var.op.name)
    self.cov = u.get_variable(name=cov_name, initializer=cov_op)
    self.svd = u.SvdWrapper(target=self.cov, name=svd_name,
                            do_inverses=(inverse_method=='inverse'))
    self.cov_update_op = self.cov.initializer


class KfacCorrectionInfo():
  """Contains information needed to correct a single parameter."""
  def __init__(self):
    self.A = None  # Covariance object
    self.B2 = None # Covariance object


class Kfac():
  """Singleton class controlling gradient correction."""
  
  # TODO: assert singletonness
  def __init__(self, model_creator, dsize):
    """
    model_creator: function that creates Model object
    stats_dsize is used to control the batch size of model used to collect
      kfac statistics."""
    
    s = self       # use for private members, ie, s.some_internal_val

    s.model = model_creator(dsize)
    if do_early_init:
      s.model.initialize_local_vars()
      s.model.initialize_global_vars()
      
    s.log = OrderedDict()

    # regular gradient
    s.grad = IndexedGrad(loss=s.model.loss, vars_=s.model.trainable_vars)
    
    # gradient with synthetic backprops
    s.grad2 = IndexedGrad(loss=s.model.loss2, vars_=s.model.trainable_vars)

    s.lr = VarStruct(initial_value=-np.inf, name="lr", dtype=dtype)
    s.Lambda = VarStruct(initial_value=-np.inf, name="Lambda", dtype=dtype)
    
    # covariance and SVD ops for all correctable ops, mapped to parameter
    # variable to correct
    s.kfac_correction_dict = OrderedDict()
    
    for var in s.model.trainable_vars:
      if not s.needs_correction(var):
        continue
      A = s.extract_A(s.grad2, var)
      B2 = s.extract_B(s.grad2, var)  # todo: change to extract_B
      s.register_correction(var)
      dsize_op = tf.constant(dsize, dtype=dtype)
      s[var].A = Covariance(A, var, "A", s.Lambda.var)
      # dsize is already incorporated as part of backprop, so must
      # multiply to get B's on the same scale as 
      s[var].B2 = Covariance(B2*dsize_op, var, "B2", s.Lambda.var)

      
    s.grad_new = s.correct(s.grad)
    s.grad_dot_grad_new_op = tf.reduce_sum(s.grad.f * s.grad_new.f)
    s.grad_norm_op = u.L2(s.grad.f)
    s.grad_new_norm_op = u.L2(s.grad_new.f)

    # create parameter save and parameter restore ops
    s.param = VarList(s.model.trainable_vars)
    s.param_copy = s.param.copy()
    s.param_save_op = s.param_copy.assign(s.param)
    s.param_restore_op = s.param.assign(s.param_copy)
    s.param_update_op = s.param.sub(s.grad_new.cached, weight=s.lr)
    assert s.param.vars_ == s.grad_new.vars_
    
    s.sess = tf.get_default_session()

  def register_correction(self, var):
    self.kfac_correction_dict[var] = KfacCorrectionInfo()
    
  # cheat for now and get those values from manual gradients
  def extract_A(self, grad, var):
    global matmul_registry
    #    i = self.model.extra['W'].index(var)
    
    assert var in matmul_registry
    ii = list(grad).index(var)  # todo: refactor this to fetch from grad
    grad_live = grad.live[ii]
    op = grad_live.op
    
    # handle the case of matmul being wrapped in tf.Identity
    if op.op_def.name == 'Identity':
      op = op.inputs[0].op
    assert op.op_def.name == 'MatMul'
    assert op.get_attr("transpose_a") == False
    assert op.get_attr("transpose_b") == True
    # tf.gradients inserts extra var/read, so below assert fails
    #    assert op.inputs[1] == self.model.extra['A'][i]
    return op.inputs[1]

  def extract_B(self, grad, var):
    #    i = self.model.extra['W'].index(var)

    assert var in matmul_registry
    ii = list(grad).index(var)  # todo: refactor this to fetch from grad
    grad_live = grad.live[ii]
    op = grad_live.op
    # handle the case of matmul being wrapped in tf.Identity
    if op.op_def.name == 'Identity':
      op = op.inputs[0].op
    assert op.op_def.name == 'MatMul'
    assert op.get_attr("transpose_a") == False
    assert op.get_attr("transpose_b") == True
    return op.inputs[0]


  def extract_dW(self, grad, var):
    i = self.model.extra['W'].index(var)
    return self.model.extra['dW'][i]


   # https://docs.python.org/3/reference/datamodel.html?emulating-container-types#emulating-container-types
  def __len__(self):
    return self.kfac_correction_dict.__len__()
  def __length_hint__(self):
    return self.kfac_correction_dict.__length_hint__()
  def __getitem__(self, key):
    return self.kfac_correction_dict.__getitem__(key)
  def __missing__(self, key):
    return self.kfac_correction_dict.__missing__(key)
  def __setitem__(self, key, value):
    return self.kfac_correction_dict.__setitem__(key, value)
  def __delitem__(self, key):
    return self.kfac_correction_dict.__delitem__(key)
  def __iter__(self):
    return self.kfac_correction_dict.__iter__()
  def __reversed__(self):
    return self.kfac_correction_dict.__reversed__()
  def __contains__(self, item):
    return self.kfac_correction_dict.__contains__(item)

  
  def update_stats(self):  # todo: split into separate stats/svd updates
    """Updates all covariance/SVD info of correctable factors."""
    s = self
    ops = []

    # update covariances
    s.model.advance_batch()
    s.grad.update()   # TODO: not needed
    s.grad2.update()
    
    for var in s:
      ops.append(s[var].A.cov_update_op)
      ops.append(s[var].B2.cov_update_op)
    s.run(ops)

    # update SVDs
    corrected_vars = list(s)
    with u.timeit("svd"):
      for var in s:
        s[var].A.svd.update()
        s[var].B2.svd.update()

  def needs_correction(self, var):
    """Returns whether given variable is getting kfac corrected."""
    global matmul_registry
    if var in matmul_registry:
      return True
    else:
      return False

  def correct(self, grad):
    """Accepts IndexedGrad object, produces corrected version."""
    s = self

    vars_ = []
    grads_new = []

    assert list(grad) == self.model.trainable_vars

    dsize = get_batch_size(grad)
    
    for var in grad:
      vars_.append(var)
      A = s.extract_A(grad, var)    # extract activations
      B = s.extract_B(grad, var)*dsize    # extract backprops
      if s.needs_correction(var):
        # correct the gradient. Assume op is left matmul
        A_svd = s[var].A.svd
        B2_svd = s[var].B2.svd
        if inverse_method == 'pseudo_inverse':
          A_new = u.pseudo_inverse2(A_svd) @ A
          B_new = u.pseudo_inverse2(B2_svd) @ B
        elif inverse_method == 'inverse':
          A_new = A_svd.inv @ A
          B_new = B2_svd.inv @ B
        else:
          assert False
          
        dW_new = (B_new @ t(A_new)) / dsize
        grads_new.append(dW_new)
      else:  
        dW = B@t(A)/dsize   
        grads_new.append(dW)

    return IndexedGrad(grads=grads_new, vars_=vars_)

  
  def reset(self):
    """Initialize/reset all optimization related variables.

    This method initializes correction factors to identity and resets history
    of the optimization.
    """
    
    s = self
    s.step_counter = 0
    
    # TODO: initialize first layer activations here, and not everywhere else
    #    self.model.initialize_local_vars()
    #    self.model.initialize_global_vars()

    ops = []

    for var in self.model.trainable_vars:
      if self.needs_correction(var):
        A_svd = s[var].A.svd
        B2_svd = s[var].B2.svd 
        ops.extend(A_svd.init_ops)
        ops.extend(B2_svd.init_ops)
    s.run(ops)


  def adaptive_step(self):
    """Performs a single KFAC step with adaptive learning rate."""

    s = self
    
    # adaptive line search parameters
    down_adjustment_frequency = 1
    up_adjustment_frequency = 50
    alpha=0.3         # acceptable fraction of predicted decrease
    beta=0.8          # how much to shrink when violation
    gamma=1.05  # how much to grow when too conservative
    report_frequency = 1
    

    s.model.advance_batch()
    s.update_stats()       # update cov matrices and svds

    s.grad.update()      # gradient (ptodo, already updated in stats)
    s.grad2.update()     # gradient from synth labels (don't need?)
    s.grad_new.update()  # corrected gradient

    # TODO: decide on s vs s.
    s.run(s.param_save_op)   # TODO: insert lr somewhere
    lr0, loss0 = s.run(s.lr, s.model.loss)

    s.run(s.param_update_op)
    loss1 = s.run(s.model.loss)
    
    target_slope = -s.run(s.grad_dot_grad_new_op)
    target_delta = lr0*target_slope
    actual_delta = loss1 - loss0
    actual_slope = actual_delta/lr0
    slope_ratio = actual_slope/target_slope  # between 0 and 1.01
    
    s.record('loss', loss0)
    s.record('step_length', lr0)
    s.record('grad_norm', s.run(s.grad_norm_op))
    s.record('grad_new_norm', s.run(s.grad_new_norm_op))
    s.record('target_delta', target_delta)

    if step_rejection and actual_delta > 0:
      print('Observed increase in loss %.2f, rejecting step'%(actual_delta,))
      s.run(s.param_restore_op)

    if s.step_counter % report_frequency == 0:
      print('NStep %d loss %.2f, target decrease %.3f, actual decrease, %.3f ratio %.2f'%(self.step_counter, loss0, target_delta, actual_delta, slope_ratio))

    if (adaptive_step and s.step_counter % down_adjustment_frequency == 0 and
        slope_ratio < alpha and abs(target_delta)>eps):
      print('%.2f %.2f %.2f'%(loss0, loss1, slope_ratio))
      print('Slope optimality %.2f, shrinking learning rate to %.2f'%(slope_ratio, lr0*beta,))
      s.lr.set(lr0*beta)
    elif (adaptive_step and s.step_counter % up_adjustment_frequency == 0 and
          slope_ratio>0.90):
      print('%.2f %.2f %.2f'%(loss0, loss1, slope_ratio))
      print('Growing learning rate to %.2f'%(lr0*gamma))
      s.lr.set(lr0*gamma)
      
    s.step_counter+=1

  def record(self, key, value):
    self.log.setdefault(key, []).append(value)

  def set(self, var, value):
    s.run(var.setter, feed_dict={var.val_: value})

  def run(self, *ops):
    new_ops = []
    for op in ops:
      if isinstance(op, VarStruct):
        new_ops.append(op.var)
      else:
        new_ops.append(op)
    if len(new_ops) == 1:
      return self.sess.run(new_ops[0])
    return self.sess.run(new_ops)
    

def vargroup_test():
  sess = tf.InteractiveSession()
  v1 = tf.Variable(1.)
  v2 = tf.Variable(1.)
  a = VarList([v1, v2])
  b = a.copy()
  sess.run(tf.global_variables_initializer())
  sess.run(a.sub(b, weight=1))
  u.check_equal(v1.eval(), 0)


matmul_registry = OrderedDict()
def matmul(a, b, name=None):
  global matmul_registry
  assert a.__class__.__name__=='Variable'
  op = tf.matmul(a, b, name)
  matmul_registry[a] = op
  return op

if __name__=='__main__':
  u.run_all_tests(sys.modules[__name__])

