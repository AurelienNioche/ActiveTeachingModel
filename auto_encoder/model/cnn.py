import numpy as np

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model


def _create_convulational_autoencoder(size):

    input_img = Input(shape=(size, size, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (13, 13, 8)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.summary()

    # this model maps an input to its encoded representation
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('max_pooling2d_3').output)
    encoder.summary()

    encoded_input = Input(shape=autoencoder.layers[-7].input_shape[1:])
    deco = autoencoder.layers[-7](encoded_input)
    deco = autoencoder.layers[-6](deco)
    deco = autoencoder.layers[-5](deco)
    deco = autoencoder.layers[-4](deco)
    deco = autoencoder.layers[-3](deco)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)

    decoder = Model(encoded_input, deco)
    decoder.summary()

    return autoencoder, encoder, decoder


def train_autoencoder(x_train, x_test, epochs=100, batch_size=128):

    x_train = np.reshape(x_train, (len(x_train), x_train.shape[1], x_train.shape[2], 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), x_train.shape[1], x_train.shape[2], 1))  # adapt this if using `channels_first` image data format

    autoencoder, encoder, decoder = _create_convulational_autoencoder(size=x_train.shape[1])

    history = autoencoder.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, x_test))

    return (autoencoder, encoder, decoder), history
