import tensorflow as tf


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    x = tf.layers.conv2d(features, filters=256, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    x = tf.layers.conv2d(x, filters=256, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
    x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2, 2], padding='SAME')
    x = tf.layers.conv2d(x, filters=512, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
    x = tf.layers.conv2d(x, filters=512, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
    x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=[2, 2], strides=[2, 2], padding='SAME')
    x = tf.layers.conv2d(x, filters=256, kernel_size=[3, 3], activation=tf.nn.relu, padding='SAME')
    x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=[2, 2], strides=[2, 2], padding='SAME')
    logits = tf.layers.conv2d(x, filters=1, kernel_size=[3, 3], activation=tf.nn.sigmoid, padding='SAME')

    predictions = {
      "classes": logits,
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.reduce_mean(tf.square(labels - logits))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions)}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
