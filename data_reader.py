
import numpy
import os
from PIL import Image

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


def image_path_to_array(path):
    return numpy.array(Image.open(path)).flatten().tolist()


def read_images_from_directory(target=PATH_BLURRED_06):
    X = y = []

    for filename in os.listdir(PATH_ORIGINAL):
        X.append(image_path_to_array(os.path.join(target, filename)))
        y.append(image_path_to_array(os.path.join(PATH_ORIGINAL, filename)))

    return X, y
