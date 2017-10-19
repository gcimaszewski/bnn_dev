import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#dimension of NMist data is 784 (28x28 pixels), with a total of 10
#classes or letters represented.  


def main():

  #number of nearest neighbors.
  nkNN = 1

  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  mnist_data = tf.placeholder(tf.float32, [None, 784])
  one_hot = tf.placeholder(tf.float32, [None, 10])

  #the letter being sorted at the time.
  test_letter = tf.placeholder(tf.float32, [1, 784])

  distances = calc_distance(mnist, test_letter, scale_frac, scales)

  rest_mnist_data = tf.identity(mnist_data)
  rest_one_hot = tf.identity(one_hot)
  rest_distances = tf.identity(distances)

  for i in range(nkNN):
    # Gets the location of training entry currently closest to the test
    # entry.
    min_slice = tf.to_int64(tf.concat(0, [tf.argmin(remaining_distances, 0), [-1]]))

    # Cuts the nearest neighbour out of the training set.
    start = tf.slice(remaining_training, tf.to_int64([0, 0]), min_slice)
    end = tf.slice(remaining_training, min_slice + [1, 1], [-1, -1])
    remaining_training = tf.concat(0, [start, end])
    # Cuts the nearest neighbour out of the distances set.
    start = tf.slice(remaining_distances, tf.to_int64([0, 0]), min_slice)
    end = tf.slice(remaining_distances, min_slice + [1, 1], [-1, -1])
    rest_distances = tf.concat(0, [start, end])

    # Cuts the nearest neighbour's class and records it.
    start = tf.slice(remaining_one_hot, tf.to_int64([0, 0]), min_slice)
    end = tf.slice(remaining_one_hot, min_slice + [1, 1], [-1, -1])
    class_slice = tf.slice(remaining_one_hot, min_slice + [0, 1], [1, -1])
    remaining_one_hot = tf.concat(0, [start, end])
    if i == 0:
        neighbour_one_hot = class_slice
    else:
        neighbour_one_hot = tf.concat(0, [neighbour_one_hot, class_slice])

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

  #try displaying the actual image
  # 

  return training, one_hot, test, tf.reduce_sum(neighbour_one_hot, reduction_indices=0)


def cal_distance(training, test, scale_frac, scales):
    """Calculates the distance between a training and test instance."""
    if scale_frac == 0:
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(training, test)),
                                         reduction_indices=1, keep_dims=True))
    else:
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.div(tf.sub(training, test), scales)),
            reduction_indices=1, keep_dims=True))
    return distance

if name===__main():
	main()