# recovering backprop matrices from graph created by tf.gradients

import tensorflow as tf
import util as u
from util import t

if __name__=='__main__':
  dtype = tf.float32
  fs = [4, 2, 3, 4, 5]
  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = f(-1)
  n = len(fs) - 2

  W = [None]*(n+1)
  for layer in range(0, n+1):
    W[layer] = tf.Variable(tf.ones((f(layer), f(layer-1))))

  # todo: add nonlinearity
  A = [None]*(n+2)
  A[0] = u.Identity(dsize, dtype=dtype)
  A[1] = W[0]
  for i in range(1, n+1):
    A[i+1] = W[i] @ A[i]

  Y = tf.ones((f(n), f(-1)))
  err = (Y - A[n+1])
  loss = u.L2(err) / (2 * dsize)

  B = [None]*(n+1)
  B[n] = err
  for i in range(n-1, -1, -1):
    B[i] = t(W[i+1]) @ B[i+1]
    
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  # manually created gradients
  dW = [None]*(n+1)
  for i in range(n+1):
    dW[i] = -(B[i] @ t(A[i]))/dsize


  # automatic gradients
  grads = tf.gradients(loss, W)

  for i in range(n+1):
    u.check_equal(dW[i].eval(), grads[i].eval())

  # first backprop is special since it's right input of matmul
  op = grads[0].op
  assert op.op_def.name == 'MatMul'
  assert op.get_attr("transpose_a") == True
  assert op.get_attr("transpose_b") == False
  u.check_equal(op.inputs[0], W[1])
  u.check_equal(op.inputs[1], -B[1]/dsize)
    
  for i in range(1, n+1):
    op = grads[i].op
    assert op.op_def.name == 'MatMul'
    assert op.get_attr("transpose_a") == False
    assert op.get_attr("transpose_b") == True
    u.check_equal(op.inputs[0], -B[i]/dsize)
    u.check_equal(op.inputs[1], A[i])
    
  print("All backprops match")
