from model_tensorflow_2 import cnn_model_fn
from sklearn.model_selection import train_test_split

from lib import read_images_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image

if __name__ == '__main__':
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        warm_start_from="./checkpoints"
    )

    tensors_to_log = {"probabilities": "tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    X, y = read_images_from_directory()

    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=64/len(X))

    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.float32)
    test_data = np.array(test_data, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.float32)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )

    print('start training')
    estimator.train(
        input_fn=train_input_fn,
        steps=20000
    )
    print('end training')

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )

    eval_results = estimator.evaluate(input_fn=eval_input_fn)
    print(eval_results)

