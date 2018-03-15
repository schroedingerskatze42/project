from srcnn_model import SRCNN
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 250, "Number of epoch [15000]")
flags.DEFINE_integer("batch_size", 85, "The size of batch images [128]")
flags.DEFINE_integer("image_height", 96, "The height of image to use [96]")
flags.DEFINE_integer("image_size", 33, "The height of image to use [96]")
flags.DEFINE_integer("image_width", 64, "The width of image to use [64]")
flags.DEFINE_integer("label_height", 96, "The height of label to produce [96]")
flags.DEFINE_integer("label_size", 21, "The height of label to produce [96]")
flags.DEFINE_integer("label_width", 64, "The width of label to produce [64]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        srcnn = SRCNN(sess,
                      image_height=FLAGS.image_height,
                      image_width=FLAGS.image_width,
                      label_height=FLAGS.label_height,
                      label_width=FLAGS.label_width,
                      batch_size=FLAGS.batch_size,
                      c_dim=FLAGS.c_dim,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir)

        srcnn.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
