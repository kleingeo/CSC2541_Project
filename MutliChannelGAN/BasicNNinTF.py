'''
(c) A.Martel lab, 2018
Author: Grey Kuling
'''

from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop, Adadelta, Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import keras.backend as K
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

if __name__ == '__main__':

    ### Pull in data set and organize test and training data
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #
    # onehot_target = pd.get_dummies(y_train)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, onehot_target,
    #                                                   test_size=0.1,
    #                                                   random_state=20)
    # x_train = x_train.reshape(x_train.shape[0],
    #                           x_train.shape[1]*x_train.shape[2] )
    #
    # x_val = x_val.reshape(x_val.shape[0],
    #                       x_val.shape[1] * x_val.shape[2])

    ### Build Model
    learning_rate = 0.5
    epochs = 10
    batch_size = 100
    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(tf.float32, [None, 784])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10])

    # x = tf.constant(x_train, dtype=tf.float32)
    # y = tf.constant(y_train.values, dtype=tf.float32)

    # now declare the weights connecting the input to the hidden layer
    W1 = tf.Variable(tf.random_normal([784, 1568], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([1568]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    W2 = tf.Variable(tf.random_normal([1568, 1568], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([1568]), name='b2')
    # and the weights connecting the hidden layer to the output layer
    W3 = tf.Variable(tf.random_normal([1568, 10], stddev=0.03), name='W3')
    b3 = tf.Variable(tf.random_normal([10]), name='b3')

    # # now declare the weights connecting the input to the hidden layer
    # W1 = tf.Variable(tf.random_normal([784, 1568], stddev=0.03), name='W1')
    # b1 = tf.Variable(tf.random_normal([1568]), name='b1')
    # # and the weights connecting the hidden layer to the output layer
    # W2 = tf.Variable(tf.random_normal([1568, 10], stddev=0.03), name='W2')
    # b2 = tf.Variable(tf.random_normal([10]), name='b2')

    h_o1 = tf.add(tf.matmul(x, W1), b1)
    h_o1 = tf.nn.sigmoid(h_o1)

    h_o2 = tf.add(tf.matmul(h_o1, W2), b2)
    h_o2 = tf.nn.sigmoid(h_o2)

    y_ = tf.nn.softmax(tf.add(tf.matmul(h_o2, W3), b3))

    # Implement a cross entropy cost function
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.99999999)
    # cross_entropy = -tf.reduce_mean(tf.reduce_sum(y*tf.log(y_clipped) + (1 -
    #                                                                      y) *
    #                                               tf.log(1-y_clipped), axis=1))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                               logits=y_clipped)

    # declare an optimizer
    # optimiser =  tf.train.GradientDescentOptimizer(
    #     learning_rate=learning_rate).minimize(cross_entropy)
    # optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
    #     cross_entropy)
    optimiser = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(
        cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ### Training

    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(mnist.train.labels) / batch_size)
        # total_batch = int(len(y_train) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                # batch_x = x_train[int(i*100):int(i*100+100),...].astype(np.float32)
                # batch_y = y_train.values[int(i*100):int(i*100+100)].astype(
                #     np.float32)
                _, c = sess.run([optimiser, cross_entropy],
                                feed_dict={x: batch_x, y: batch_y})
                avg_cost += np.mean(c)
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(
                avg_cost/total_batch))
        print(sess.run(accuracy,
                   feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    print('done')
