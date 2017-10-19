import sys
import os
import time

import numpy as np
np.random.seed(1234) # for reproducibility?

#import cPickle as pickle
import gzip

import hardware_net_tf
import FixedPoint
import Printer

import tensorflow as tf

from pylearn2.datasets.cifar10 import CIFAR10

#if __name__ == "__main__":

def test_otpt():

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
    activation = hardware_net_tf.SignTheano
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
 #   test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))
    test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1, 3, 32, 32))
    # flatten targets
    test_set.y = np.hstack(test_set.y)
    # Onehot the targets
    test_set.y = np.float32(np.eye(10)[test_set.y])
    # for hinge loss
    test_set.y = 2* test_set.y - 1.

    print('Quantizing the input...')
    X = test_set.X[0:test_instances]
    X = FixedPoint.FixedPoint(32,31).convert(X)
    X = np.transpose(X, [0, 2, 3, 1])
    y = test_set.y[0:test_instances]

    print "shape of \n"
    print np.shape(X)

    print('Building the CNN...')

    #the input layer
    #tensorflow doesn't have an OO format for layers - layers are just functions
    inputs = tf.placeholder(tf.float32, [None, 32, 32, 3])
    inputs_tr = tf.placeholder(tf.float32, [None, 32, 32, 3])
    #target = T.matrix(X, 'targets')

    #final output from the nonlinearity layer
    target = tf.placeholder(tf.float32, [None, 32, 32, 128])
    conv_result = tf.placeholder(tf.float32, [None, 32, 32, 128])
    weights = tf.Variable(tf.zeros([128, 3, 3, 3]))
    

    k = tf.Variable(tf.zeros([128]))
    h = tf.Variable(tf.zeros([128]))



    #--------------------------------------------
    # CNN start
    #--------------------------------------------

    def cnn(x, weight, h, k):
        #filter size and strides?
        #filter is only weight
     #   x = tf.nn.conv2d(x, tf.random_uniform([128, 3, 3, 3], -1.0, 1.0), [1,1,1,1], "SAME")
        weight = tf.transpose(weight, [2, 3, 1, 0])
        y = tf.nn.conv2d(x, weight, [1,1,1,1], "SAME")
        #overwrite batch_norm
        x = hardware_net_tf.batch_norm(y, h, k)
        x = hardware_net_tf.SignTheano(x)
    #    return tf.nn.relu(x)
        return [y,x]


    print("Loading the trained parameters and binarizing the weights...")



    with np.load(params_dir + '/cifar10_parameters_nb.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(3)]

    k_fix = FixedPoint.FixedPoint(16,15)
    h_fix = FixedPoint.FixedPoint(16,12)


    # Binarize the weights
    num_params = 3
    l = 1
    lout = 1
    param_values_mod = [None for i in range(num_params)]
    for param in range(num_params):
        #W
        if param == 0:
          #  param.set_value(hardware_net.SignNumpy(param_values[param])
            param_values_mod[param] = hardware_net_tf.SignNumpy(param_values[param])
            if l == lout:
                print "kernel 0 for layer", l
                Printer.print_3d(param_values_mod[param][0,:,:,:], 3, 3, 3, 'b')
        #k
        elif param == 1:
          #  param.set_value(k_fix.convert(param_values[param])
            param_values_mod[param] = k_fix.convert(param_values[param])
            if l == lout:
                print "k =", param_values_mod[param][0:4]
        #h
        elif param == 2:
            param_values_mod[param] = h_fix.convert(param_values[param])
            if l == lout:
                print "h =", param_values_mod[param][0:4]
            l = l + 1
        else:
            print "Incorrect param name", param.name
            exit(-1)


    for num_input_maps in range(len(param_values_mod[0])):
        for num_output_maps in range(len(param_values_mod[0][num_input_maps])):
                 #   print  weights_arr[conv][num_input_maps][num_output_maps]
            param_values_mod[0][num_input_maps][num_output_maps] = np.flip(param_values_mod[0][num_input_maps][num_output_maps], 1)
            param_values_mod[0][num_input_maps][num_output_maps] = np.flip(param_values_mod[0][num_input_maps][num_output_maps], 0)


    print('Running...')

    # Load parameters

    conv_output = cnn(inputs, weights, h, k )

    init = tf.initialize_all_variables()

    start_time = time.time()


    with tf.Session() as sess:
        sess.run(init)
        #load data into the placeholders
        output = sess.run(conv_output, feed_dict={weights: param_values_mod[0], k:param_values_mod[1], \
         h: param_values_mod[2], inputs: X})
        print "executed"

#    output = np.reshape(output, (1, 128, 32, 32))


    # input_ = input_fn(X)
    # output = cnn_fn(X)
    # conv_out = conv_fn(X)
    # print "input shape=", input_.shape
    # Printer.print_2d(input_[0,0,:,:], 8, 8, 'f')
    # print " ##"
    # Printer.print_2d(input_[0,1,:,:], 8, 8, 'f')

   # print "\nconv shape=", conv_out.shape
  #  Printer.print_2d(output[0][0,0,:,:], 8, 8, 'f')

    # print "\noutput shape=", output.shape
    # Printer.print_2d(output[0,:,:,0], 8, 8, 'b')
    # print " ##"
    # Printer.print_2d(output[0,:,:,1], 8, 8, 'b')
    # print " ##"
    # Printer.print_2d(output[0,:,:,2], 8, 8, 'b')

    # #np.savez("py_conv1_maps.npz", output);

    run_time = time.time() - start_time
    print("run_time = "+str(run_time)+"s")

    return output