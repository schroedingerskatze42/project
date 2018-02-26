#!/usr/bin/env python

import re
import numpy
from PIL import Image
import os

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

if __name__ == '__main__':
    # img = Image.open('images/blurred_03/00002_930831_fa.ppm')
    # print(numpy.array(img))

    for filename in os.listdir(PATH_ORIGINAL):
        if filename.endswith(".ppm"):
            print(os.path.join(PATH_ORIGINAL, filename))
            continue
        else:
            continue
