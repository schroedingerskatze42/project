from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import cv2

from six.moves import xrange
from sklearn.model_selection import train_test_split

from lib import tf_ssim
from lib import read_images_from_directory

IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
NUM_CHANNELS = 1
PIXEL_DEPTH = 255

VALIDATION_SIZE = 50  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 4
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 1  # Number of steps between evaluations.


def error_rate(predictions, labels):
    result = 0

    for p, l in zip(predictions, labels):
        result += 1 - tf_ssim(p, l)

    return result / len(predictions)


def at_generate_weights(inputs, outputs):
    return tf.Variable(
        tf.truncated_normal([3, 3, inputs, outputs], stddev=10, seed=SEED, dtype=np.float32)
    )


def at_generate_biases(channels):
    return tf.Variable(tf.zeros([channels], dtype=np.float32))


def model_layers(data):
    conv_1 = tf.layers.conv2d(data, filters=256, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    conv_2 = tf.layers.conv2d(max_pool_1, filters=256, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
    max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    conv_3 = tf.layers.conv2d(max_pool_2, filters=512, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
    conv_4 = tf.layers.conv2d(conv_3, filters=512, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
    upsample_1 = tf.layers.conv2d_transpose(conv_4, filters=512, kernel_size=[2, 2], strides=[2, 2], padding='SAME')
    conv_5 = tf.layers.conv2d(upsample_1, filters=256, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
    upsample_2 = tf.layers.conv2d_transpose(conv_5, filters=256, kernel_size=[2, 2], strides=[2, 2], padding='SAME')
    conv_6 = tf.layers.conv2d(upsample_2, filters=1, kernel_size=[3, 3], activation=tf.nn.sigmoid, padding='SAME')
    return conv_6


def main(_):
    # Extract data into np arrays.
    X, y = read_images_from_directory()

    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=64/len(X))

    train_data = np.array(train_data, dtype=np.float32)
    test_data = np.array(test_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.float32)

    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]

    train_size = train_labels.shape[0]

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        np.float32,
        shape=(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS)
    )

    train_labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))

    eval_data = tf.placeholder(np.float32, shape=(EVAL_BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS))

    # Training computation: logits + cross-entropy loss.
    logits = model_layers(train_data_node)

    loss = tf.reduce_mean(tf.square(train_labels_node - logits))
    # loss = 1 - tf_ssim(train_labels_node, logits)

    batch = tf.Variable(0, dtype=np.float32)

    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = 0.05
    # learning_rate = tf.train.exponential_decay(
    #     0.01,  # Base learning rate.
    #     batch * BATCH_SIZE,  # Current index into the dataset.
    #     train_size,  # Decay step.
    #     0.95,  # Decay rate.
    #     staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.01).minimize(loss, global_step=batch)

    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = model_layers(eval_data)

    # Small utility function to evaluate a data set by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = np.ndarray(shape=(size, IMAGE_WIDTH, IMAGE_HEIGHT, 1), dtype=np.float32)
        for begin in xrange(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    # Create a local session to run the training.
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.global_variables_initializer().run()

        print('Initialized!')

        print('Train: %d' % len(train_data))
        print('Test: %d' % len(test_data))

        # Loop through training steps.
        saver = tf.train.Saver()
        start_time = time.time()

        # if os.path.isfile('./model.ckpt')
        # saver.restore(sess, './model_2.ckpt')
        # else:
        range = int(NUM_EPOCHS * train_size) // BATCH_SIZE
        for step in xrange(range):
            print('Step: %d/%d' % (step + 1, range))
            start_time = time.time()

            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]

            # This dictionary maps the batch data (as a np array) to the
            # node in the graph it should be fed to.
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}

            # Run the optimizer to update weights.
            tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=optimizer)
            sess.run(optimizer, feed_dict=feed_dict)

            print('Duration: %ds' % int(time.time() - start_time))

            if 0 == step % 10:
                if 0 == step % 500:
                    save_path = saver.save(sess, './checkpoints/model_layers_%d.ckpt' % int(step / 500))
                    print("Model saved in path: %s" % save_path)

                # write result image
                predictions = eval_in_batches(test_data, sess)
                img = np.array(np.multiply(predictions[0].reshape([48, 48]), 255), dtype=np.int)
                filename = './results_tensorflow/result_%d.ppm' % int(step / 10)
                cv2.imwrite(filename, img)


if __name__ == '__main__':
    tf.app.run(main=main)
