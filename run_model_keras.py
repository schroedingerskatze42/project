import model_keras
from lib import read_images_from_directory_2
import numpy as np
from PIL import Image

if __name__ == '__main__':
    m = model_keras.construct()
    # train_data, _ = data_reader.read_images_from_directory_2()  # source='test_data/input/', target='test_data/output/')
    #
    # train_data = np.reshape(train_data, [-1, 96, 64, 1])
    # test_img = train_data[:1]
    #
    # pred = m.predict(test_img)
    # pred = np.reshape(pred, [96, 64])
    # pred = np.multiply(pred, 255)
    # img = Image.fromarray(pred)
    # img. show()
