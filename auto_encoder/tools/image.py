import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
import os


SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

BACKUP_FOLDER = f'{SCRIPT_FOLDER}/../backup'
IMG_FOLDER = f'{SCRIPT_FOLDER}/../image'

BKP_HISTORY = \
    f'{BACKUP_FOLDER}/history.p'
BKP_MODEL = [
    f'{BACKUP_FOLDER}/autoencoder.h5',
    f'{BACKUP_FOLDER}/encoder.h5',
    f'{BACKUP_FOLDER}/decoder.h5']

IMG_SIZE = 100
N_ROTATION = 3
EPOCHS = 100
SEED = 123

os.makedirs(BACKUP_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)


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

def _open_image(path):

    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    img = np.array(img)
    return img


def _format_images():

    n_images = len(os.listdir(IMG_FOLDER))
    train_data = np.zeros((n_images * N_ROTATION, IMG_SIZE, IMG_SIZE))

    i = 0

    for img in os.listdir(IMG_FOLDER):
        path = os.path.join(IMG_FOLDER, img)

        img = _open_image(path)

        train_data[i] = img
        i += 1

        # Basic Data Augmentation - Horizontal Flipping
        flip_h_img = np.fliplr(img)
        train_data[i] = flip_h_img

        i += 1

        # Basic Data Augmentation - Vertical Flipping
        flip_v_img = np.flipud(img)
        train_data[i] = flip_v_img

        i += 1

    train_data /= 255.
    return train_data


def training_data(kanji_dic, ratio_training_test=2/3):

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

# --------------------------------------- #


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


# ----------------------------------------- #


# def _example_dataset():
#
#     from keras.datasets import mnist
#
#     (x_train, _), (x_test, _) = mnist.load_data()
#
#     x_train = x_train.astype('float32') / 255.
#     x_test = x_test.astype('float32') / 255.
#
#     return x_train, x_test
