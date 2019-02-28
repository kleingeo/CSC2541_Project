import tensorflow as tf
import numpy as np
import csv

def build_model(features, labels, mode):

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    return loss


def data_gen(filename):

    csv_file = open(filename)
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:

        label = int(row[0])

        image = np.array(row[1:])

        yield image, label

def input_func_gen(data_gen_fn):

    dataset = tf.data.Dataset.from_generator(data_gen_fn, output_types=(tf.float32, tf.int32))

    dataset.batch(4)
    iterator = dataset.make_initializable_iterator()
    image, label = iterator.get_next()

    features = {'x': image}
    return features, labels


def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys
   params):
    pass



if __name__ == '__main__':

    data_gen('mnist_train.csv')
