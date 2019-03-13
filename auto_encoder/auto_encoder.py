import numpy as np

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
# from keras import backend as K

from keras.callbacks import TensorBoard
from keras.models import load_model

import matplotlib.pyplot as plt
import os 
import pickle

from PIL import Image


def _load_images(img_size, folder="image"):

    n_images = len(os.listdir(folder))

    train_data = np.zeros((n_images*2, img_size, img_size))

    i = 0

    for img in os.listdir(folder):

        path = os.path.join(folder, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((img_size, img_size), Image.ANTIALIAS)
        train_data[i] = np.array(img)

        i += 1

        # Basic Data Augmentation - Horizontal Flipping
        flip_img = Image.open(path)
        flip_img = flip_img.convert('L')
        flip_img = flip_img.resize((img_size, img_size), Image.ANTIALIAS)
        flip_img = np.array(flip_img)
        flip_img = np.fliplr(flip_img)
        train_data[i] = flip_img

        i += 1

    train_data /= 255.
    # random.shuffle(train_data)
    return train_data


def _show_example_image(train_data):

        print(train_data)
        print(train_data[0].shape)
        plt.imshow(train_data[0], cmap='gist_gray')
        plt.show()


def load_training_data(img_size, ratio=2/3):

    train_data = _load_images(img_size=img_size)

    np.random.shuffle(train_data)

    split = int(len(train_data) * ratio)

    x_train = train_data[:split]
    x_test = train_data[split:]

    return x_train, x_test


def _simple_autoencoder(size, encoding_dim=32):

    # this is the size of our encoded representations
    # encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=(size,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(size, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
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


def _create_convulational_autoencoder(size):

    input_img = Input(shape=(size, size, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

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

    return autoencoder, encoder


def train_autoencoder(x_train, x_test, epochs=100, batch_size=128):

    # tensorboard --logdir=/tmp/autoencoder

    x_train = np.reshape(x_train, (len(x_train), x_train.shape[1], x_train.shape[2], 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), x_train.shape[1], x_train.shape[2], 1))  # adapt this if using `channels_first` image data format

    autoencoder, encoder = _create_convulational_autoencoder(size=x_train.shape[1])

    train = autoencoder.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    return (autoencoder, encoder), train


def get_encoded_decoded(encoder, autoencoder, x_test):
    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)

    return encoded_imgs, decoded_imgs


def get_decoded_only(model, x_test):
    x_test = np.reshape(x_test, (len(x_test), x_test.shape[1], x_test.shape[2], 1))
    return model.predict(x_test)


def _example_dataset():
    from keras.datasets import mnist

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    return x_train, x_test


def show_result(x_test, decoded_images):

    size = x_test.shape[1]

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(size, size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_images[i].reshape(size, size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def show_accuracy(train):

    # Plot the accuracy and loss plots between training and validation data:
    accuracy = train.history['acc']
    val_accuracy = train.history['val_acc']
    loss = train.history['loss']
    val_loss = train.history['val_loss']
    epochs = range(len(accuracy))

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(epochs, accuracy, 'bo', label='Training accuracy')
    ax1.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    ax1.set_title('Training and validation accuracy')
    ax1.legend()

    ax2.plot(epochs, loss, 'bo', label='Training loss')
    ax2.plot(epochs, val_loss, 'b', label='Validation loss')
    ax2.set_title('Training and validation loss')
    ax2.legend()

    plt.tight_layout()

    fig_file = f"fig/training_and_validation_loss.pdf"
    os.makedirs(fig_file.split("/")[0], exist_ok=True)
    plt.savefig(fig_file)


def save_model_and_history(backup_train_file, train, backup_model_files, models):

    # Backup in pickle
    os.makedirs(backup_train_file.split("/")[0], exist_ok=True)
    pickle.dump(train, file=open(backup_train_file, 'wb'))

    for bmf, m in zip(backup_model_files, models):
        m.save(bmf)


def load_model_and_history(backup_train_file, backup_model_files):

    train = pickle.load(file=open(backup_train_file, 'rb'))

    models = [load_model(i) for i in backup_model_files]

    return models, train


# def evaluate_model(model, test_X, test_Y_one_hot):
#
#     # Model evaluation on the test set
#     test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=1)
#     print('Test loss:', test_eval[0])
#     print('Test accuracy:', test_eval[1])


def main(force=False):

    # Parameters
    img_size = 100
    epochs = 200

    # Seed
    np.random.seed(123)

    # Backup file
    bkp_train_file = f'backup/backup_train.p'
    bkp_model_files = ['backup/backup_autoencoder.h5', 'backup/backup_encoder.h5']

    os.makedirs(bkp_train_file.split("/")[0], exist_ok=True)
    files_exist = np.all([os.path.exists(i) for i in bkp_model_files + [bkp_train_file]])

    x_train, x_test = load_training_data(img_size=img_size)

    if force or not files_exist:

        (autoencoder, encoder), train = train_autoencoder(x_train, x_test, epochs=epochs)

        x_test = np.reshape(x_test, (len(x_test), x_test.shape[1], x_test.shape[2], 1))

        save_model_and_history(backup_train_file=bkp_train_file, backup_model_files=bkp_model_files,
                               models=(autoencoder, encoder), train=train)
    else:
        (autoencoder, encoder), train = \
            load_model_and_history(backup_train_file=bkp_train_file, backup_model_files=bkp_model_files)

    encoded_images, decoded_images = get_encoded_decoded(x_test=x_test, encoder=encoder, autoencoder=autoencoder)
    show_result(x_test=x_test, decoded_images=decoded_images)


if __name__ == "__main__":

    main()
