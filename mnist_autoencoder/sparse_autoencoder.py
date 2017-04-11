import numpy as np
import tensorflow as tf

dtype = np.float64
#import pdb; pdb.set_trace()
import os, sys


# add exp directory since it has util
def up(path): return os.path.dirname(path)
cwd = os.path.realpath(__file__)
sys.path.append(up(up(cwd))+"/exp")

from util import t
import util as u

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def KL_divergence(x, y):
    print("Numpy shapes: %s, %s"%(x.shape, y.shape))
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

def KL_divergence_tf(x, y):
    print("TF shapes: %s, %s"%(x.shape, y.shape))
    return x * tf.log(x / y) + (1 - x) * tf.log((1 - x) / (1 - y))


def initialize(hidden_size, visible_size):
    # we'll choose weights uniformly from the interval [-r, r]
    r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    W1 = np.random.random((hidden_size, visible_size)) * 2 * r - r
    W2 = np.random.random((visible_size, hidden_size)) * 2 * r - r

    b1 = np.zeros(hidden_size, dtype=np.float64)
    b2 = np.zeros(visible_size, dtype=np.float64)

    theta = np.concatenate((W1.reshape(hidden_size * visible_size),
                            W2.reshape(hidden_size * visible_size),
                            b1.reshape(hidden_size),
                            b2.reshape(visible_size)))

    return theta


# visible_size: the number of input units (probably 64)
# hidden_size: the number of hidden units (probably 25)
# lambda_: weight decay parameter
# sparsity_param: The desired average activation for the hidden units (denoted in the lecture
#                            notes by the greek alphabet rho, which looks like a lower-case "p").
# beta: weight of sparsity penalty term
# data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.
#
# The input theta is a vector (because minFunc expects the parameters to be a vector).
# We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
# follows the notation convention of the lecture notes.
# Returns: (cost,gradient) tuple
def sparse_autoencoder_cost(theta, visible_size, hidden_size,
                            lambda_, sparsity_param, beta, data):
    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    W2 = theta[hidden_size * visible_size:2 * hidden_size * visible_size].reshape(visible_size, hidden_size)
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    b2 = theta[2 * hidden_size * visible_size + hidden_size:]

    # Number of training examples
    m = data.shape[1]

    # Forward propagation
    z2 = W1.dot(data) + np.tile(b1, (m, 1)).transpose()
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + np.tile(b2, (m, 1)).transpose()
    h = sigmoid(z3)

    # Sparsity
    rho_hat = np.sum(a2, axis=1) / m
    rho = np.tile(sparsity_param, hidden_size)

    # Cost function
    cost = np.sum((h - data) ** 2) / (2 * m) + \
           (lambda_ / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2)) + \
           beta * np.sum(KL_divergence(rho, rho_hat))

    # Backprop
    sparsity_delta = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).transpose()

    delta3 = -(data - h) * sigmoid_prime(z3)
    delta2 = (W2.transpose().dot(delta3) + beta * sparsity_delta) * sigmoid_prime(z2)
    W1grad = delta2.dot(data.transpose()) / m + lambda_ * W1
    W2grad = delta3.dot(a2.transpose()) / m + lambda_ * W2
    b1grad = np.sum(delta2, axis=1) / m
    b2grad = np.sum(delta3, axis=1) / m

    # After computing the cost and gradient, we will convert the gradients back
    # to a vector format (suitable for minFunc).  Specifically, we will unroll
    # your gradient matrices into a vector.
    grad = np.concatenate((W1grad.reshape(hidden_size * visible_size),
                           W2grad.reshape(hidden_size * visible_size),
                           b1grad.reshape(hidden_size),
                           b2grad.reshape(visible_size)))

    return cost, grad

# visible_size: the number of input units (probably 64)
# hidden_size: the number of hidden units (probably 25)
# lambda_: weight decay parameter
# sparsity_param: The desired average activation for the hidden units (denoted in the lecture
#                            notes by the greek alphabet rho, which looks like a lower-case "p").
# beta: weight of sparsity penalty term
# data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.
#
# The input theta is a vector (because minFunc expects the parameters to be a vector).
# We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
# follows the notation convention of the lecture notes.
# Returns: (cost,gradient) tuple
def sparse_autoencoder_cost_matlab(theta, visible_size, hidden_size,
                            lambda_, sparsity_param, beta, data):
    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size, order='F')
    W2 = theta[hidden_size * visible_size:2 * hidden_size * visible_size].reshape(visible_size, hidden_size, order='F')
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    b2 = theta[2 * hidden_size * visible_size + hidden_size:]

    # Number of training examples
    m = data.shape[1]
    a1 = data

    # Forward propagation
    z2 = W1.dot(a1) + np.tile(b1, (m, 1)).transpose()
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + np.tile(b2, (m, 1)).transpose()
    a3 = sigmoid(z3)

    # Sparsity
    rho_hat = np.sum(a2, axis=1) / m
    rho = np.tile(sparsity_param, hidden_size)

    # Cost function
    cost = np.sum((a3 - a1) ** 2) / (2 * m) + \
           (lambda_ / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2)) + \
           beta * np.sum(KL_divergence(rho, rho_hat))

    # Backprop
    # tile adds extra dimensions on the left, produces m x l2 shape
    sparsity_delta = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).T / m

    # data is A1 in my notation, h is A3
    # sigmoid_prime(z3) = sigmoid(z3)(1-sigmoid(z3)) = h(1-h)
    # deltas are transposed from matlab code

    # delta3 gives sensitivity of z3
    delta3 = (a3 - a1) * a3 * (1 - a3) / m
    # delta2 is sensitivity of z2
    delta2 = (W2.T.dot(delta3) + beta * sparsity_delta) * a2 * (1 - a2)
    W1grad = delta2.dot(a1.T)  + lambda_ * W1
    W2grad = delta3.dot(a2.T) + lambda_ * W2
    b1grad = np.sum(delta2, axis=1)
    b2grad = np.sum(delta3, axis=1)

    # After computing the cost and gradient, we will convert the gradients back
    # to a vector format (suitable for minFunc).  Specifically, we will unroll
    # your gradient matrices into a vector.
    grad = np.concatenate((W1grad.reshape(hidden_size * visible_size, order='F'),
                           W2grad.reshape(hidden_size * visible_size, order='F'),
                           b1grad.reshape(hidden_size),
                           b2grad.reshape(visible_size)))

    return cost, grad

def sparse_autoencoder_cost_tf(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):
  
    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size, order='F')
    W2 = theta[hidden_size * visible_size:2 * hidden_size * visible_size].reshape(visible_size, hidden_size, order='F')
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    b2 = theta[2 * hidden_size * visible_size + hidden_size:]

    init_dict = {}
    def init_var(val, name):
      holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
      var = tf.Variable(holder, name=name+"_var")
      init_dict[holder] = val
      return var

    W1_ = init_var(W1, "W1")
    W2_ = init_var(W2, "W2")
    b1_ = init_var(np.expand_dims(b1, 0), "b1")
    b2_ = init_var(np.expand_dims(b2, 0), "b2")
    data_ = init_var(data, "data")
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
    
    # Number of training examples
    m = data.shape[1]
    a1 = data
    a1_ = data_

    # Forward propagation
    z2 = W1.dot(a1) + np.tile(b1, (m, 1)).transpose()
    # ValueError: Dimensions must be equal, but are 10000 and 1960000 for 'add' (op: 'Add') with input shapes: [196,10000], [1,1960000].
    z2_ = tf.matmul(W1_, a1_) + t(tf.tile(b1_, (m, 1)))

    ## STOP HERE
    a2 = sigmoid(z2)
    a2_ = tf.sigmoid(z2_)
    z3 = W2.dot(a2) + np.tile(b2, (m, 1)).transpose()
    z3_ = tf.matmul(W2_, a2_) + t(tf.tile(b2_, (m, 1)))
    a3 = sigmoid(z3)
    a3_ = tf.sigmoid(z3_)

    # Sparsity
    rho_hat = np.sum(a2, axis=1) / m
    rho_hat_ = tf.reduce_sum(a2_, axis=1, keep_dims=True)/m
    rho = np.tile(sparsity_param, hidden_size)
    # ValueError: Shape must be rank 1 but is rank 0 for 'Tile_2' (op: 'Tile') with input shapes: [], [].
    rho_ = tf.constant(sparsity_param, dtype=dtype)
    #tf.ones((hidden_size, 1), dtype=dtype)*sparsity_param

    u.check_equal(sess.run(a3_), a3)
    u.check_equal(sess.run(a2_), a2)
    u.check_equal(sess.run(a1_), a1)
    u.check_equal(tf.reduce_sum(KL_divergence_tf(rho_, rho_hat_)).eval(),
                  np.sum(KL_divergence(rho, rho_hat)))
    
    # Cost function
    cost = np.sum((a3 - a1) ** 2) / (2 * m) + \
           (lambda_ / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2)) + \
           beta * np.sum(KL_divergence(rho, rho_hat))
    cost_ = tf.reduce_sum((a3_ - a1_) ** 2) / (2 * m) + \
           (lambda_ / 2) * (tf.reduce_sum(W1_ ** 2) + \
                            tf.reduce_sum(W2_ ** 2)) + \
           beta * tf.reduce_sum(KL_divergence_tf(rho_, rho_hat_))
    return sess.run(cost_)



def sparse_autoencoder(theta, hidden_size, visible_size, data):
    """
    :param theta: trained weights from the autoencoder
    :param hidden_size: the number of hidden units (probably 25)
    :param visible_size: the number of input units (probably 64)
    :param data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example.
    """

    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.
    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]

    # Number of training examples
    m = data.shape[1]

    # Forward propagation
    z2 = W1.dot(data) + np.tile(b1, (m, 1)).transpose()
    a2 = sigmoid(z2)

    return a2


# visible_size: the number of input units (probably 64)
# hidden_size: the number of hidden units (probably 25)
# lambda_: weight decay parameter
# sparsity_param: The desired average activation for the hidden units (denoted in the lecture
#                            notes by the greek alphabet rho, which looks like a lower-case "p").
# beta: weight of sparsity penalty term
# data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.
#
# The input theta is a vector (because minFunc expects the parameters to be a vector).
# We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
# follows the notation convention of the lecture notes.
# Returns: (cost,gradient) tuple
def sparse_autoencoder_linear_cost(theta, visible_size, hidden_size,
                                   lambda_, sparsity_param, beta, data):
    # The input theta is a vector (because minFunc expects the parameters to be a vector).
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
    # follows the notation convention of the lecture notes.

    W1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size)
    W2 = theta[hidden_size * visible_size:2 * hidden_size * visible_size].reshape(visible_size, hidden_size)
    b1 = theta[2 * hidden_size * visible_size:2 * hidden_size * visible_size + hidden_size]
    b2 = theta[2 * hidden_size * visible_size + hidden_size:]

    # Number of training examples
    m = data.shape[1]

    # Forward propagation
    z2 = W1.dot(data) + np.tile(b1, (m, 1)).transpose()
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + np.tile(b2, (m, 1)).transpose()
    h = z3

    # Sparsity
    rho_hat = np.sum(a2, axis=1) / m
    rho = np.tile(sparsity_param, hidden_size)


    # Cost function
    cost = np.sum((h - data) ** 2) / (2 * m) + \
           (lambda_ / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2)) + \
           beta * np.sum(KL_divergence(rho, rho_hat))



    # Backprop
    sparsity_delta = np.tile(- rho / rho_hat + (1 - rho) / (1 - rho_hat), (m, 1)).transpose()

    delta3 = -(data - h)
    delta2 = (W2.transpose().dot(delta3) + beta * sparsity_delta) * sigmoid_prime(z2)
    W1grad = delta2.dot(data.transpose()) / m + lambda_ * W1
    W2grad = delta3.dot(a2.transpose()) / m + lambda_ * W2
    b1grad = np.sum(delta2, axis=1) / m
    b2grad = np.sum(delta3, axis=1) / m

    # After computing the cost and gradient, we will convert the gradients back
    # to a vector format (suitable for minFunc).  Specifically, we will unroll
    # your gradient matrices into a vector.
    grad = np.concatenate((W1grad.reshape(hidden_size * visible_size),
                           W2grad.reshape(hidden_size * visible_size),
                           b1grad.reshape(hidden_size),
                           b2grad.reshape(visible_size)))

    return cost, grad

