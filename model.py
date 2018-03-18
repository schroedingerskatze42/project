from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import LambdaCallback
import data_reader
import numpy as np
from PIL import Image

from loss import gradient_importance


def construct():
    input_img = Input(shape=(96, 64, 1))

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input_img, x)
    model.compile(optimizer='adadelta', loss=gradient_importance)

    train_data, train_label = data_reader.read_images_from_directory()
    train_data = np.divide(train_data, 255)
    train_data = np.reshape(train_data, [-1, 96, 64, 1])
    train_label = np.divide(train_label, 255)
    train_label = np.reshape(train_label, [-1, 96, 64, 1])

    test_img = train_data[:1]

    print_image_on_epoch_end = LambdaCallback(
        on_epoch_end=lambda epoch, logs: Image.fromarray(np.multiply(model.predict(test_img).reshape([96, 64]), 255)).show())

    model.fit(x=train_data, y=train_label, batch_size=85, epochs=15, callbacks=[print_image_on_epoch_end])
    return model
