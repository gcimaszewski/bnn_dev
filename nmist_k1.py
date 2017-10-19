from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# unsure of batch size
# size of train data = 60,000
# size of test data = 10,000
# with test batch of 1000 and train of 20000: ~91.5%
# with test batch of 200 and train of 20000: ~96.5%
# with test batch of 200 and train of 30000: ~98%
batch_xtr, batch_ytr = mnist.train.next_batch(30000) #training batch
batch_xte, batch_yte = mnist.test.next_batch(200)  #testing batch

x_train = tf.placeholder("float", [None, 784])
x_test = tf.placeholder("float", [784])

distance = tf.reduce_sum(tf.abs(tf.add(x_train, tf.neg(x_test))), reduction_indices=1)
pred = tf.arg_min(distance, 0)

accuracy = 0

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(batch_xte)):
        nn_index = sess.run(pred, feed_dict={x_train: batch_xtr, x_test: batch_xte[i, :]})
#        print("Test", i, "Prediction:", np.argmax(batch_ytr[nn_index]), \
#            "True Class:", np.argmax(batch_yte[i]))
       
        if np.argmax(batch_ytr[nn_index]) == np.argmax(batch_yte[i]):
            accuracy += 1/float(len(batch_xte))
    print("Accuracy:", accuracy)