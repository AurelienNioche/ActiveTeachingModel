from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
# from keras import backend as K
# from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from PIL import Image, ImageDraw, ImageFont


SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
BACKUP_FOLDER = f'{SCRIPT_FOLDER}/backup'
IMG_FOLDER = f'{SCRIPT_FOLDER}/image'
IMG_SIZE = 100
EPOCHS = 100


def _img_file(name):

    return f"{IMG_FOLDER}/{name}.png"


def _create_images(
        kanji_dic,
        font='auto_encoder/font/arialunicodems.ttf',
        text_color=(255, 255, 255),
        background_color=(0, 0, 0)):

    os.makedirs(IMG_FOLDER, exist_ok=True)
    font_size = int(IMG_SIZE*2/3)

    for name, character in kanji_dic.items():

        img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=background_color)

        fnt = ImageFont.truetype(font, font_size)
        d = ImageDraw.Draw(img)
        w, h = d.textsize(character, font=fnt)
        d.text(((IMG_SIZE-w)/2, (IMG_SIZE-h)/2 - h/10), character, font=fnt, fill=text_color)

        if name is None:
            name = ord(character)

        img.save(_img_file(name))


def _are_images_existing(kanji_dic):

    for key in kanji_dic.keys():
        if not os.path.exists(_img_file(name=key)):
            return False
    return True


# --------------------------------------- #


def _format_images():

    n_images = len(os.listdir(IMG_FOLDER))
    train_data = np.zeros((n_images * 2, IMG_SIZE, IMG_SIZE))

    i = 0

    for img in os.listdir(IMG_FOLDER):
        path = os.path.join(IMG_FOLDER, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        train_data[i] = np.array(img)

        i += 1

        # Basic Data Augmentation - Horizontal Flipping
        flip_img = Image.open(path)
        flip_img = flip_img.convert('L')
        flip_img = flip_img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        flip_img = np.array(flip_img)
        flip_img = np.fliplr(flip_img)
        train_data[i] = flip_img

        i += 1

    train_data /= 255.
    return train_data


def _training_data(kanji_dic, ratio_training_test=2/3):

    if not _are_images_existing(kanji_dic):
        _create_images(kanji_dic=kanji_dic)

    train_data = _format_images()

    np.random.shuffle(train_data)

    split = int(len(train_data) * ratio_training_test)

    x_train = train_data[:split]
    x_test = train_data[split:]

    return x_train, x_test


def _show_image_example(train_data):

    plt.imshow(train_data[0], cmap='gist_gray')
    plt.show()

# ----------------------------------------- #


def get_formatted_image_for_cnn(kanji_id):

    path = os.path.join(IMG_FOLDER, _img_file(kanji_id))
    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    a_img = np.array(img, dtype='float32')
    a_img /= 255.
    a_img = np.reshape(a_img, (1, a_img.shape[0], a_img.shape[1], 1))
    return a_img

# ----------------------------------------- #


def _example_dataset():

    from keras.datasets import mnist

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    return x_train, x_test

# ------------------------------------------------ #


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

# -------------------------------------------------------- #


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


# --------------------------------------------------------- #


def get_encoded_decoded(encoder, autoencoder, decoder, x_test, convulational=True):

    if convulational:
        x_test = np.reshape(x_test, (len(x_test), x_test.shape[1], x_test.shape[2], 1))

    else:
        size = np.prod(x_test.shape[1:])
        x_test = x_test.reshape((len(x_test), size))

    encoded_img = encoder.predict(x_test)
    decoded_img = decoder.predict(encoded_img)
    decoded_img_auto = autoencoder.predict(x_test)

    return encoded_img, decoded_img, decoded_img_auto

# -------------------------------------------------------- #


def show_result(x_test, decoded_images, decoded_images_auto,
                fig_file="fig/autoencoder_image_reconstruction.pdf"):

    size = x_test.shape[1]

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(size, size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(decoded_images[i].reshape(size, size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(decoded_images_auto[i].reshape(size, size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()

    os.makedirs(fig_file.split("/")[0], exist_ok=True)
    plt.savefig(fig_file)


def show_accuracy(train, fig_file="fig/autoencoder_validation_loss.pdf"):

    # Plot the accuracy and loss plots between training and validation data:
    # accuracy = train.history['acc']
    # val_accuracy = train.history['val_acc']
    loss = train.history['loss']
    val_loss = train.history['val_loss']
    epochs = range(len(loss))

    # fig, (ax1, ax2) = plt.subplots(2, 1)

    # ax1.plot(epochs, accuracy, 'bo', label='Training accuracy')
    # ax1.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    # ax1.set_title('Training and validation accuracy')
    # ax1.legend()

    fig, ax = plt.subplots()

    ax.plot(epochs, loss, 'bo', label='Training loss')
    ax.plot(epochs, val_loss, 'b', label='Validation loss')
    ax.set_title('Training and validation loss')
    ax.legend()

    plt.tight_layout()

    os.makedirs(fig_file.split("/")[0], exist_ok=True)
    plt.savefig(fig_file)

# -------------------------------------------------------- #


def save_model_and_history(bkp_train_file, history, bkp_model_files, models):

    # Backup in pickle
    os.makedirs(bkp_train_file.split("/")[0], exist_ok=True)
    pickle.dump(history, file=open(bkp_train_file, 'wb'))

    for bmf, m in zip(bkp_model_files, models):
        m.save(bmf)


def load_model_and_history(bkp_train_file, bkp_model_files):

    history = pickle.load(file=open(bkp_train_file, 'rb'))

    models = [load_model(i) for i in bkp_model_files]

    return models, history

# -------------------------------------------------------- #


def get(kanji_dic, force=False):

    # Seed
    np.random.seed(123)

    # Backup file
    bkp_train_file = \
        f'{BACKUP_FOLDER}/train.p'
    bkp_model_files = [
        f'{BACKUP_FOLDER}/autoencoder.h5',
        f'{BACKUP_FOLDER}/encoder.h5',
        f'{BACKUP_FOLDER}/decoder.h5']

    os.makedirs(BACKUP_FOLDER, exist_ok=True)

    # Check if bkp files already exist
    files_exist = np.all([os.path.exists(i) for i in bkp_model_files + [bkp_train_file]])

    if force or not files_exist:

        # Get the data set
        x_train, x_test = _training_data(kanji_dic=kanji_dic)

        # Create and train autoencoder
        (autoencoder, encoder, decoder), history = train_autoencoder(x_train, x_test, epochs=EPOCHS)

        # Save models and history
        save_model_and_history(bkp_train_file=bkp_train_file, bkp_model_files=bkp_model_files,
                               models=(autoencoder, encoder, decoder), history=history)
    else:
        # Load models
        (autoencoder, encoder, decoder), history = \
            load_model_and_history(bkp_train_file=bkp_train_file, bkp_model_files=bkp_model_files)

    # encoded_images, decoded_images, decoded_images_auto = \
    #     get_encoded_decoded(x_test=x_test, encoder=encoder, decoder=decoder, autoencoder=autoencoder)
    # show_result(x_test=x_test, decoded_images=decoded_images, decoded_images_auto=decoded_images_auto)
    # show_accuracy(history)

    return autoencoder, encoder, decoder
