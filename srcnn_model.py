import data_reader
import time
import os
import numpy as np

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

        self.pred = self.model()

        # Loss function (MSE)
        print("labels: ", self.labels)
        print("images: ", self.images)
        print("pred: ", self.pred)
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.saver = tf.train.Saver()

    def train(self, config):
        # if config.is_train:
        #     input_setup(self.sess, config)
        # else:
        #     nx, ny = input_setup(self.sess, config)

        # if config.is_train:
        #     data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
        # else:
        #     data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

        train_data, train_label = data_reader.read_images_from_directory()  # read_data(data_dir)

        # maybe move data manipulation to reader?

        train_data = np.divide(train_data, 255)
        train_data = np.reshape(train_data, [-1, 96, 64, 1])
        train_label = np.divide(train_label, 255)
        train_label = np.reshape(train_label, [-1, 96, 64, 1])

        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)

        tf.initialize_all_variables().run()

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
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                              % ((ep + 1), counter, time.time() - start_time, err))

                    if counter % 50 == 0:
                        self.save(config.checkpoint_dir, counter)

        else:
            print("Testing... # to be implemented")

            # result = self.pred.eval({self.images: train_data, self.labels: train_label})
            #
            # result = merge(result, [nx, ny])
            # result = result.squeeze()
            # image_path = os.path.join(os.getcwd(), config.sample_dir)
            # image_path = os.path.join(image_path, "test_image.png")
            # imsave(result, image_path)

    def model(self):
        print("model_images: ", self.images)
        conv1 = tf.nn.relu(
            tf.nn.conv2d(self.images, self.weights['w1'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b1'])
        print("model_conv1: ", conv1)
        conv2 = tf.nn.relu(
            tf.nn.conv2d(conv1, self.weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b2'])
        print("model_conv2: ", conv2)
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b3']
        print("model_conv3: ", conv3)
        return conv3

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
