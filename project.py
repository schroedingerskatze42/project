#!/usr/bin/env python

import re
import numpy
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import math

PATH_ORIGINAL = 'images/thumbnails/'
PATH_BLURRED_03 = 'images/blurred_03/'
PATH_BLURRED_06 = 'images/blurred_06/'
PATH_BLURRED_09 = 'images/blurred_09/'
PATH_BLURRED_12 = 'images/blurred_12/'
PATH_BLURRED_15 = 'images/blurred_15/'
PATH_MOSAIC_05 = 'images/mosaic_05/'
PATH_MOSAIC_10 = 'images/mosaic_10/'
PATH_MOSAIC_15 = 'images/mosaic_15/'
PATH_MOSAIC_20 = 'images/mosaic_20/'
PATH_MOSAIC_25 = 'images/mosaic_25/'

TRAIN_RATIO = 0.2


def image_path_to_array(path):
    return numpy.array(Image.open(path)).flatten().tolist()


if __name__ == '__main__':
    X = y = []

    for filename in os.listdir(PATH_BLURRED_06):
        X.append(image_path_to_array(os.path.join(PATH_ORIGINAL, filename)))

    for filename in os.listdir(PATH_ORIGINAL):
        y.append(image_path_to_array(os.path.join(PATH_ORIGINAL, filename)))

    print(X[0])
    exit()
    print('Dataset loaded...')

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=TRAIN_RATIO)

    # classifier =
    classifier = MLPRegressor(solver='lbfgs', alpha=1e-5,
                              hidden_layer_sizes=(5, 2), random_state=1)

    print('Start training...')
    classifier.fit(train_x, train_y)
    print('Training finished...')

    print(classifier.score(test_x, test_y))
