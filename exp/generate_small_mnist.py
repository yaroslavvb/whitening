import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf
import os
import sys

from PIL import Image

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64

NUM_EPOCHS = 10
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

FLAGS = None

dtype = tf.float32

def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def main(_):
  # Get the data.
  train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
  train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')

  dsize = 1000
  
  # Extract it into numpy arrays.
  train_data = extract_data(train_data_filename, dsize)
  train_labels = extract_labels(train_labels_filename, dsize)

  train_data2 = []
  for i in range(train_data.shape[0]):
    arr = (255*(train_data[i,:,:,:]+0.5)).astype(np.uint8)
    im = Image.fromarray(arr[:,:,0])
    im.thumbnail((7,7), Image.ANTIALIAS)
    arr2 = numpy.array(im)
    # Use these if need back in original format
    #    arr2 = arr.astype(np.float32)
    #    arr2 = (arr2/255.)-0.5

    train_data2.append(arr2)

  train_data2_ = numpy.array(train_data2)

  # Use this if need back in original format
  #  train_data2_ = np.expand_dims(train_data2_, 3) # insert dim into the back

  # reshape to have
  # 49 + 10 rows, dsize columns

  # train_data2_ is (dsize, 7, 7)
  # row-major order so can just use reshape
  train_features = train_data2_.reshape((dsize, 49)).T  # (49, dsize)

  train_labels_one_hot = np.zeros((dsize, 10))
  train_labels_one_hot[np.arange(dsize), train_labels] = 1
  train_labels = train_labels_one_hot.T # (10, dsize)
  
  train_data = np.concatenate((train_features, train_labels)) # (59, dsize)
  np.savetxt("mnist_small.csv", train_data, fmt="%d", delimiter=',')


if __name__ == '__main__':
  main(None)
  

