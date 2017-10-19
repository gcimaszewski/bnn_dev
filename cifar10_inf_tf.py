import sys
import os
import time

import numpy as np
np.random.seed(1234) # for reproducibility?

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')

import tensorflow as tf
import hardware_net_tf

import cPickle as pickle
import gzip

import hardware_net
import FixedPoint
import Printer

from pylearn2.datasets.cifar10 import CIFAR10

def cnn_build(features, labels, mode):

	cnn = tf.layers.conv2d(
      inputs=input_layer,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=hardware_net_tf.SignTheano)

	bn_1 = hardware_net_tf.batch_norm(cnn, h, k)

	conv2 = tf.layers.conv2d(
      inputs=,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=hardware_net_tf.SignTheano)

	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2])

	bn2 = hardware_net_tf.batch_norm(pool2, h, k)

	conv3 = tf.layers.conv2d(
      inputs=bn2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=hardware_net_tf.SignTheano)

	bn3 = hardware_net_tf.batch_norm(conv3, h, k)

	conv4 = tf.layers.conv2d(
      inputs=bn3,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=hardware_net_tf.SignTheano)

	pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2])

	bn4 = hardware_net_tf.batch_norm(pool4, h, k)

	conv5 = tf.layers.conv2d(
      inputs=input_layer,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=hardware_net_tf.SignTheano)

	bn5 = hardware_net_tf.batch_norm(conv5, h, k)

	conv6 = tf.layers.conv2d(
      inputs=input_layer,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=hardware_net_tf.SignTheano)

	pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2])

	bn6 = hardware_net_tf.batch_norm(conv6, h, k)

	bn6 = tf.reshape(bn6, [-1, 4*4*512])
    dense1 = tf.layers.dense(inputs=bn6, units=1024, activation = hardware_net_tf.SignTheano)

    bn7 = hardware_net_tf.batch_norm(dense1, h, k)

    dense2 = tf.layers.dense(inputs=bn7, units=1024, activation=hardware_net_tf.SignTheano)

    bn8 = hardware_net_tf.batch_norm(dense2, h, k)

    dense3 = tf.layers.dense(inputs=bn8, units=10, activation=hardware_net_tf.SignTheano)

    






if __name__ == "__main__":
	tf.app.run()