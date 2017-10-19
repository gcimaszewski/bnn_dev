import layer_conv1
import layer_conv1_tf
import cifar10_inference_tf
import layer_conv2_tf
import layer_conv2

import sys
import os
import time

import numpy as np
np.random.seed(1234) # for reproducibility?

import cPickle as pickle
import gzip

import theano
import theano.tensor as T
import lasagne

import hardware_net
import FixedPoint
import Printer

import tensorflow as tf
from pylearn2.datasets.cifar10 import CIFAR10


if __name__ == "__main__":

    # BN parameters
    # alpha is the exponential moving average factor
    alpha = .1
    epsilon = 1e-4

    # Parameters directory
    if not os.environ.has_key('CRAFT_BNN_ROOT'):
        print "CRAFT_BNN_ROOT not set!"
        exit(-1)
    top_dir = os.environ['CRAFT_BNN_ROOT']
    params_dir = top_dir + '/params'

    # BinaryOut
    activation = hardware_net.SignTheano
    print("activation = sign(x)")

    no_bias = True
    print("no_bias = " + str(no_bias))

    # BinaryConnect
    H = 1.
    print('Loading CIFAR-10 dataset...')

    test_set = CIFAR10(which_set="test")
    print("Test set size = "+str(len(test_set.X)))
    test_instances = 1
    print("Using instances 0 .. "+str(test_instances))

    # bc01 format
    # Inputs in the range [-1,+1]
    test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))
    # flatten targets
    test_set.y = np.hstack(test_set.y)
    # Onehot the targets
    test_set.y = np.float32(np.eye(10)[test_set.y])
    # for hinge loss
    test_set.y = 2* test_set.y - 1.

    print('Quantizing the input...')
    X = test_set.X[0:test_instances]
    X = FixedPoint.FixedPoint(32,31).convert(X)
    X_tf = np.transpose(X, [0, 2, 3, 1])

    y = test_set.y[0:test_instances]
    print y
    conv_otpt_l = layer_conv1.conv_otpt()[0]
    conv_otpt_l = np.array(conv_otpt_l)
    conv_otpt_l = np.around(conv_otpt_l, 2)

    fin_otpt_1 = layer_conv1.conv_otpt()[1]
    fin_otpt_l = np.array(fin_otpt_1)

    #conv_otpt_l = layer_conv1.conv_fn(X)

    conv_otpt_tf = layer_conv1_tf.test_otpt()[0]
    conv_otpt_tf = np.array(conv_otpt_tf)
    conv_otpt_tf = np.transpose(conv_otpt_tf, [0, 3, 1, 2])
    conv_otpt_tf = np.around(conv_otpt_tf, 2)



    fin_otpt_tf = layer_conv1_tf.test_otpt()[1]
    fin_otpt_tf = np.array(fin_otpt_tf)
    fin_otpt_tf = np.transpose(fin_otpt_tf, [0, 3, 1, 2])

    # inf_tf = cifar10_inference_tf.conv_check()
    # print inf_tf
    # inf_tf = np.asarray(inf_tf)
    # print inf_tf.shape
    # inf_tf = np.transpose(inf_tf, [0, 3, 1, 2])

    conv2_l = layer_conv2.conv2_out()
    conv2_l = np.array(conv2_l)

    conv2_tf = layer_conv2_tf.conv2_out()
    conv2_tf = np.array(conv2_tf)
    conv2_tf = np.transpose(conv2_tf, [0,3,1,2])
    print "print convolution 2 results\n\n"

    print np.sum(conv2_tf == conv2_l)


    # print (conv_otpt_l.shape)
    # print (conv_otpt_tf.shape)
    print conv_otpt_tf[0][0][0]
    print conv_otpt_l[0][0][0]
    print np.sum(conv_otpt_tf == conv_otpt_l)
    print np.sum(fin_otpt_tf == fin_otpt_l)

