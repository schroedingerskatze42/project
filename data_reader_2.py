import os

import numpy as np
from PIL import Image

IMAGE_SOURCE = 'images_50_grayscale/'

PATH_ORIGINAL = IMAGE_SOURCE + 'original/'
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
    with Image.open(path) as img:
        img = img.convert('L')
        return np.asarray(img) / 255


def read_images_from_directory(source=PATH_BLURRED_03, target=PATH_ORIGINAL):
    X = []
    y = []

    for source_file in os.listdir(source):
        if source_file.endswith('.ppm'):
            X.append(image_path_to_array(os.path.join(source, source_file)))

    for target_file in os.listdir(target):
        if target_file.endswith('.ppm'):
            y.append(image_path_to_array(os.path.join(target, target_file)))

    return X, y


if __name__ == '__main__':
    x, y = read_images_from_directory()
    print(x[0].shape)
    print(y[0].shape)
    img_in = Image.fromarray(x[0] * 255)
    img_out = Image.fromarray(y[0] * 255)
    img_in.show()
    img_out.show()
