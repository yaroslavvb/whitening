import load_MNIST
import sparse_autoencoder
import scipy.optimize
import softmax
import numpy as np
import display_network
from IPython.display import Image
import scipy.io # for loadmat
import matplotlib # for matplotlib.cm.gray
from matplotlib.pyplot import imshow
import math
import time

def transposeImageCols(mat):
    """Treats columns of mat as flattened square images, transposes them"""
    result = np.zeros(mat.shape, dtype=mat.dtype)
    w2=mat.shape[0]
    w = int(math.sqrt(w2))
    assert w**2==w2
    for i in range(mat.shape[1]):
        patch=mat[:,i].reshape((w,w))
        result[:,i]=patch.transpose().flatten()
    return result

def asciiPlot(mat, threshold=0.1):
    """Returns printable representation of matrix thresholded to binary, use "print" to display """
    binary_array=np.piecewise(mat, [mat < threshold, mat >= threshold], [0, 1]).astype("uint8");
    def stringConvert(arr):
        return ''.join([str(s) for s in arr])
    return '\n'.join([stringConvert(row) for row in list(binary_array)])



train_images = load_MNIST.load_MNIST_images('data/train-images-idx3-ubyte')
train_labels = load_MNIST.load_MNIST_labels('data/train-labels-idx1-ubyte')

# implement MNIST sparse autoencoder

visibleSize = 28*28;
hiddenSize = 196;
sparsityParam = 0.1;
lambda_ = 3e-3;
beta = 3;
patches = train_images[:,:10000];

opttheta = scipy.io.loadmat("opttheta.mat")["opttheta"].flatten()
W1 = opttheta[:hiddenSize*visibleSize].reshape((hiddenSize, visibleSize),order='F')
# also need to transpose individual columns
display_network.display_network(transposeImageCols(W1.transpose()), "sample2.jpg")

patchesTransposed = transposeImageCols(patches)
J1 = lambda x: sparse_autoencoder.sparse_autoencoder_cost_matlab(x, visibleSize, hiddenSize, lambda_, sparsityParam, beta, patchesTransposed)
cost,grad=J1(opttheta)
print("My cost1 %.3f" %(cost))

J2 = lambda x: sparse_autoencoder.sparse_autoencoder_cost(x, visibleSize, hiddenSize, lambda_, sparsityParam, beta, patchesTransposed)
cost,grad=J2(opttheta)
print("My cost2 %.3f" %(cost))

cost3 = sparse_autoencoder.sparse_autoencoder_cost_tf(opttheta, visibleSize, hiddenSize, lambda_, sparsityParam, beta, patchesTransposed)
print("My cost3 %.3f" %(cost3))

matlab_vars = scipy.io.loadmat("opttheta_vars.mat")
print("Matlab cost %.3f" %(matlab_vars["cost"]),)
