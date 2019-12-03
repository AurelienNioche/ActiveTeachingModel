import numpy as np


def _simple_autoencoder(size, encoding_dim=32):

    from keras.layers import Input, Dense
    from keras.models import Model

    # this is the size of our encoded representations
    # encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=(size,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(size, activation='sigmoid')(encoded)

    # this learner maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this learner maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder learner
    decoder_layer = autoencoder.layers[-1]
    # create the decoder learner
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder, encoder, decoder


def train_simple(x_train, x_test, encoding_dim=32, epochs=50, batch_size=256):

    size = np.prod(x_test.shape[1:])

    x_train = x_train.reshape((len(x_train), size))
    x_test = x_test.reshape((len(x_test), size))

    autoencoder, encoder, decoder = _simple_autoencoder(size=size, encoding_dim=encoding_dim)

    train = autoencoder.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, x_test))

    return (autoencoder, encoder, decoder), train
