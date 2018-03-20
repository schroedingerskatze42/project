import os
import random
from glob import glob

import numpy as np
from PIL import Image

ORIGINAL = 'images/'
RESULT_INPUT = 'test_data_updated/input/'
RESULT_OUTPUT = 'test_data_updated/output/'


def create_test_data(target_size, source_rescale, shuffle=True, ):
    originals = glob(os.path.join(ORIGINAL, '*.ppm'))

    if shuffle:
        random.shuffle(originals)

    source = np.ndarray((len(originals), target_size[0], target_size[1], 1), dtype=np.float32)
    target = np.ndarray((len(originals), target_size[0], target_size[1], 1), dtype=np.float32)

    for i in range(len(originals)):
        with Image.open(originals[i]) as img:
            img = img.convert('L').resize(target_size)
            source_img = img.resize(source_rescale).resize(target_size)

            source_scaled = np.asarray(source_img) / 255
            target_scaled = np.asarray(img) / 255

            source[i] = source_scaled.reshape(target_size[0], target_size[1], 1)
            target[i] = target_scaled.reshape(target_size[0], target_size[1], 1)

    return source, target


if __name__ == '__main__':
    x, y = create_test_data(source_rescale=(56, 56), target_size=(224, 224))

    for i in range(len(x)):
        img_in = Image.fromarray(x[i].reshape(224, 224) * 255)
        img_in = img_in.convert('RGB')
        img_out = Image.fromarray(y[i].reshape(224, 224) * 255)
        img_out = img_out.convert('RGB')
        img_in.save(RESULT_INPUT + "in_" + str(i) + ".ppm", "ppm")
        img_out.save(RESULT_OUTPUT + "out_" + str(i) + ".ppm", "ppm")
