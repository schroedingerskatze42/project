from lib import data_reader_2 as data_reader
import time
import os
import numpy as np
from PIL import Image

import tensorflow as tf

try:
    xrange
except:
    xrange = range


# THE MODEL TO BE IMPLEMENTED
#
# input_img = Input(shape=(*target_size, 1))
#
# x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(input_img)
# x = MaxPooling2D((2, 2), border_mode='same')(x)
#
# x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
# x = MaxPooling2D((2, 2), border_mode='same')(x)
#
# x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
# x = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(x)
# x = UpSampling2D((2, 2))(x)
#
# x = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(x)
# x = UpSampling2D((2, 2))(x)
#
# x = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)
#
#
# model = Model(input_img, x)
# model.compile(optimizer='adadelta', loss=gradient_importance)
#
# END MODEL


class SRCNN(object):

    def __init__(self,
                 sess,
                 image_height=33,
                 image_width=33,
                 label_height=21,
                 label_width=21,
                 batch_size=128,
                 c_dim=1,
                 checkpoint_dir=None,
                 sample_dir=None):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_height = image_height
        self.image_width = image_width
        self.label_height = label_height
        self.label_width = label_width
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_height, self.label_width, self.c_dim], name='labels')

        self.weights_1 = {
            'w1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=1e-3), name='w3'),
            'w4': tf.Variable(tf.random_normal([3, 3, 128, 64], stddev=1e-3), name='w4'),
            'up1': tf.Variable(tf.random_normal([3, 3, 128, 64], stddev=1e-3), name='up1'),
            'w5': tf.Variable(tf.random_normal([3, 3, 128, 64], stddev=1e-3), name='w5'),
            'up2': tf.Variable(tf.random_normal([3, 3, 128, 64], stddev=1e-3), name='up2'),
            'w6': tf.Variable(tf.random_normal([3, 3, 128, 1], stddev=1e-3), name='w6')
        }

        self.biases_1 = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([64]), name='b2'),
            'b3': tf.Variable(tf.zeros([128]), name='b3'),
            'b4': tf.Variable(tf.zeros([64]), name='b4'),
            'b5': tf.Variable(tf.zeros([64]), name='b5'),
            'b6': tf.Variable(tf.zeros([1]), name='b6')
        }

        self.weights_orig = {
            'w1': tf.Variable(tf.random_normal([3, 3, 1, 256], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=1e-3), name='w3'),
            'w4': tf.Variable(tf.random_normal([3, 3, 512, 256], stddev=1e-3), name='w4'),
            'up1': tf.Variable(tf.random_normal([3, 3, 512, 256], stddev=1e-3), name='up1'),
            'w5': tf.Variable(tf.random_normal([3, 3, 512, 256], stddev=1e-3), name='w5'),
            'up2': tf.Variable(tf.random_normal([3, 3, 512, 256], stddev=1e-3), name='up2'),
            'w6': tf.Variable(tf.random_normal([3, 3, 512, 1], stddev=1e-3), name='w6')
        }

        self.biases_orig = {
            'b1': tf.Variable(tf.zeros([256]), name='b1'),
            'b2': tf.Variable(tf.zeros([256]), name='b2'),
            'b3': tf.Variable(tf.zeros([512]), name='b3'),
            'b4': tf.Variable(tf.zeros([256]), name='b4'),
            'b5': tf.Variable(tf.zeros([256]), name='b5'),
            'b6': tf.Variable(tf.zeros([1]), name='b6')
        }

        self.weights = {
            'w1': tf.Variable(tf.random_normal([1, 1, 1, 64], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, 64, 128], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([1, 1, 128, 1], stddev=1e-3), name='w3')
        }

        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([128]), name='b2'),
            'b3': tf.Variable(tf.zeros([1]), name='b3')
        }

        # this work's
        # self.pred = self.model()

        # this won't work probably
        # self.pred = self.model_1()

        # this neither :)
        self.pred = self.model_2()

        # new try
        # self.loss = l.gradient_importance(self.labels, self.pred)

        # Loss function (MSE)
        self.loss = tf.reduce_mean(self.labels - self.pred)

        self.saver = tf.train.Saver()

    def train(self, config):

        train_data, train_label = data_reader.read_images_from_directory()

        train_data = np.reshape(train_data, [-1, 96, 64, 1])
        train_label = np.reshape(train_label, [-1, 96, 64, 1])

        # Bilder bis hierhin okay!

        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)

        # self.train_op = tf.train.AdadeltaOptimizer(config.learning_rate).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if config.is_train:
            print("Training...")

            for ep in xrange(config.epoch):
                # Run by batch images
                batch_idxs = len(train_data) // config.batch_size
                for idx in xrange(0, batch_idxs):
                    batch_images = train_data[idx * config.batch_size: (idx + 1) * config.batch_size]
                    batch_labels = train_label[idx * config.batch_size: (idx + 1) * config.batch_size]

                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss],
                                           feed_dict={self.images: batch_images, self.labels: batch_labels})

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" % (
                            (ep + 1), counter, time.time() - start_time, err))

                        test_input = batch_images[:1]
                        test_result = self.sess.run(self.pred, feed_dict={self.images: test_input})
                        print("test_result:")
                        print(test_result)
                        test_result = test_result.reshape(96, 64)
                        test_result = test_result * 255
                        print("test_result_2:")
                        print(test_result)
                        Image.fromarray(test_result).show()

                    if counter % 100 == 0:
                        self.save(config.checkpoint_dir, counter)

        else:
            print("Testing... # to be implemented")

            train_data = train_data[:1]

            print(train_data)
            tst = train_data.reshape(224, 224) * 255
            tst = Image.fromarray(tst)
            tst.show()

            result = self.pred.eval({self.images: train_data})
            result = np.multiply(result, 255)
            result = result.reshape(224, 224)
            print(result)
            img = Image.fromarray(result)
            img.show()

    def model(self):
        conv1 = tf.nn.relu(
            tf.nn.conv2d(self.images, self.weights['w1'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b1'])
        conv2 = tf.nn.relu(
            tf.nn.conv2d(conv1, self.weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b3']
        return conv3

    def model_1(self):
        conv_1 = tf.nn.relu(
            tf.nn.conv2d(self.images, self.weights_1['w1'], strides=[1, 1, 1, 1], padding='SAME') + self.biases_1['b1'])
        max_pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv_2 = tf.nn.relu(
            tf.nn.conv2d(max_pool_1, self.weights_1['w2'], strides=[1, 1, 1, 1], padding='SAME') + self.biases_1['b2'])
        max_pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv_3 = tf.nn.relu(
            tf.nn.conv2d(max_pool_2, self.weights_1['w3'], strides=[1, 1, 1, 1], padding='SAME') + self.biases_1['b3'])
        conv_4 = tf.nn.relu(
            tf.nn.conv2d(conv_3, self.weights_1['w4'], strides=[1, 1, 1, 1], padding='SAME') + self.biases_1['b4'])
        upsample_1 = tf.nn.conv2d_transpose(conv_4, filter=self.weights_1['up1'],
                                            output_shape=[self.batch_size, 48, 32, 128], strides=[1, 2, 2, 1],
                                            padding='SAME')
        conv_5 = tf.nn.relu(
            tf.nn.conv2d(upsample_1, self.weights_1['w5'], strides=[1, 1, 1, 1], padding='SAME') + self.biases_1['b5'])
        upsample_2 = tf.nn.conv2d_transpose(conv_5, filter=self.weights_1['up2'],
                                            output_shape=[self.batch_size, 96, 64, 128], strides=[1, 2, 2, 1],
                                            padding='SAME')
        conv_6 = tf.nn.sigmoid(
            tf.nn.conv2d(upsample_2, self.weights_1['w6'], strides=[1, 1, 1, 1], padding='SAME') + self.biases_1['b6'])
        return conv_6

    def model_2(self):
        conv_1 = tf.layers.conv2d(self.images, filters=64, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME',
                                  use_bias=True)
        max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        conv_2 = tf.layers.conv2d(max_pool_1, filters=64, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME',
                                  use_bias=True)
        max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=[2, 2], strides=[2, 2], padding='SAME')
        conv_3 = tf.layers.conv2d(max_pool_2, filters=128, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME',
                                  use_bias=True)
        conv_4 = tf.layers.conv2d(conv_3, filters=128, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME',
                                  use_bias=True)
        upsample_1 = tf.layers.conv2d_transpose(conv_4, filters=128, kernel_size=[2, 2], strides=[2, 2], padding='SAME',
                                                use_bias=True)
        conv_5 = tf.layers.conv2d(upsample_1, filters=64, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME',
                                  use_bias=True)
        upsample_2 = tf.layers.conv2d_transpose(conv_5, filters=64, kernel_size=[2, 2], strides=[2, 2], padding='SAME',
                                                use_bias=True)
        conv_6 = tf.layers.conv2d(upsample_2, filters=1, kernel_size=[3, 3], activation=tf.nn.sigmoid, padding='SAME',
                                  use_bias=True)
        return conv_6

    def save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_height)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("srcnn", self.label_height)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
