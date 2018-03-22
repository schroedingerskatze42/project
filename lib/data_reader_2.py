import os

import numpy as np
from PIL import Image

__all__ = ['read_images_from_directory_2']

IMAGE_SOURCE = 'databases/colorferet_testimages/'

PATH_ORIGINAL = IMAGE_SOURCE + 'original/'
PATH_BLURRED_03 = IMAGE_SOURCE + 'grayscale/blur_03/'
PATH_MOSAIC_05 = IMAGE_SOURCE + 'grayscale/mosaic_05/'

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 96
NUM_CHANNELS = 1


def image_path_to_array(path):
    with Image.open(path) as img:
        img = img.convert('L')
        return np.asarray(img) / 255


def read_images_from_directory_2(source=PATH_BLURRED_03, target=PATH_ORIGINAL):
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
