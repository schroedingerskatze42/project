import os
import cv2
from PIL import Image

__all__ = ['image_path_to_array', 'read_images_from_directory']

IMAGE_SOURCE = 'databases/facescrub_testimages/'

PATH_ORIGINAL = IMAGE_SOURCE + 'original/'
PATH_BLURRED_03 = IMAGE_SOURCE + 'grayscale/blur_03/'
PATH_MOSAIC_05 = IMAGE_SOURCE + 'grayscale/mosaic_05/'

IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
NUM_CHANNELS = 1


def image_path_to_array(path):
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
    x[0] = x[0].reshape([48, 48])
    # print(x[0])
    print("y0")
    print(y[0].shape)
    # print(y[0])
    img = Image.fromarray(x[0], 'L')
    img.show()
    # img = Image.fromarray(y[0], 'L')
    # img.show()
