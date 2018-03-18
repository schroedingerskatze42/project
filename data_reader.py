import numpy as np
import os
from skimage import io
import cv2
from PIL import Image

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

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 96
NUM_CHANNELS = 1


def image_path_to_array(path):
    # print(np.asarray(io.imread(path)))
    # exit(1)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS)


def read_images_from_directory(target=PATH_BLURRED_03):
    X = []
    y = []

    for filename in os.listdir(PATH_ORIGINAL):
        if filename.endswith('.ppm'):
            X.append(image_path_to_array(os.path.join(target, filename)))
            y.append(image_path_to_array(os.path.join(PATH_ORIGINAL, filename)))

    return X, y


if __name__ == '__main__':
    x, y = read_images_from_directory()
    print("x0:")
    print(x[0].shape)
    print(x[0])
    print("y0")
    print(y[0].shape)
    print(y[0])
    img = Image.fromarray(x[0], 'L')
    img.show()
    img = Image.fromarray(y[0], 'L')
    img.show()
