from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K


def build_cae_model():
    """
    build convolutional autoencoder model
    """
    input_img = Input(shape=(32, 32, 3))

    # encoder
    net = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    net = MaxPooling2D((2, 2), padding='same')(net)
    net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
    net = MaxPooling2D((2, 2), padding='same')(net)
    net = Conv2D(4, (3, 3), activation='relu', padding='same')(net)
    encoded = MaxPooling2D((2, 2), padding='same', name='enc')(net)

    # decoder
    net = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    net = UpSampling2D((2, 2))(net)
    net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
    net = UpSampling2D((2, 2))(net)
    net = Conv2D(16, (3, 3), activation='relu', padding='same')(net)
    net = UpSampling2D((2, 2))(net)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(net)

    return Model(input_img, decoded)
