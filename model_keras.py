from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras import losses
import data_reader
import numpy as np
from PIL import Image

from loss import gradient_importance


def construct():
    input_img = Input(shape=(224, 224, 1))

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
    # model.compile(optimizer='adadelta', loss=losses.mean_absolute_error)

    train_data, train_label = data_reader.read_images_from_directory(source='test_data_updated/input/',
                                                                     target='test_data_updated/output/')

    train_data = np.reshape(train_data, [-1, 224, 224, 1])
    train_label = np.reshape(train_label, [-1, 224, 224, 1])

    test_img = train_data[:1]

    print_image_on_batch_end = LambdaCallback(
        on_batch_end=lambda epoch, logs: Image.fromarray(
            np.multiply(model.predict(test_img).reshape([224, 224]), 255)).show())

    filepath = "keras_checkpoints/weights_images_updated_{epoch:02d}_2.hdf5"
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1)

    model.fit(x=train_data, y=train_label, batch_size=32, epochs=4,
              callbacks=[print_image_on_batch_end, checkpointer])

    return model
