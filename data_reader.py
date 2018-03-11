import numpy as np
import os
from skimage import io

IMAGE_SOURCE = 'images_50_grayscale/'

PATH_ORIGINAL = IMAGE_SOURCE + 'thumbnails/'
PATH_BLURRED_03 = IMAGE_SOURCE + 'blurred_03/'
PATH_BLURRED_06 = IMAGE_SOURCE + 'blurred_06/'
PATH_BLURRED_09 = IMAGE_SOURCE + 'blurred_09/'
PATH_BLURRED_12 = IMAGE_SOURCE + 'blurred_12/'
PATH_BLURRED_15 = IMAGE_SOURCE + 'blurred_15/'
PATH_MOSAIC_05 = IMAGE_SOURCE + 'mosaic_05/'
PATH_MOSAIC_10 = IMAGE_SOURCE + 'mosaic_10/'
PATH_MOSAIC_15 = IMAGE_SOURCE + 'mosaic_15/'
PATH_MOSAIC_20 = IMAGE_SOURCE + 'mosaic_20/'
PATH_MOSAIC_25 = IMAGE_SOURCE + 'mosaic_25/'


PIXEL_DEPTH = 255
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 96
NUM_CHANNELS = 1


def image_to_np(path):
    return np.array(io.imread(path)).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS)


def read_images_from_directory(target=PATH_BLURRED_06):
    X = []
    y = []

    for path in os.listdir(PATH_ORIGINAL):
        if path.endswith('.ppm'):
            X.append(image_to_np(PATH_ORIGINAL + path))
            y.append(image_to_np(PATH_BLURRED_06 + path))

    return np.array(X), np.array(y)
