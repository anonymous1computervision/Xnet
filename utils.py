from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.layers.core import dense, flatten
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from box import Box
from mpl_toolkits.axes_grid1 import ImageGrid

def load_config(fp):
    return Box(yaml.load(open(fp, 'r').read()))

def my_eigen(x):
    return np.linalg.eigh(x)

def my_svd(x):
    return np.linalg.svd(x, compute_uv=False)

def attention_estimator(x, g):
    with tf.variable_scope(None, 'attention_estimator'):
        _, height, width, x_dim = x.get_shape().as_list()
        g_dim = g.get_shape().as_list()[-1]

        if not x_dim == g_dim:
            x = dense(x, g_dim, use_bias=False)

        c = tf.reduce_sum(x * tf.expand_dims(tf.expand_dims(g, 1), 1), axis=-1)
        a = tf.nn.softmax(flatten(c))
        a = tf.reshape(a, (-1, height, width))
        g_out = x * tf.expand_dims(a, -1)
        g_out = tf.reduce_sum(g_out, axis=[1, 2])
        return g_out, a


def attention_module(ls, g):
    gs = g
    as_ = []
    _g, a = attention_estimator(ls, g)
    gs = tf.add(gs, _g)
    return g, as_   

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    plt.show()


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig('dann_tsne.png')
