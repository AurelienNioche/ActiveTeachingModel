import os
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = f'{SCRIPT_FOLDER}/../data'
BACKUP_FOLDER = f'{SCRIPT_FOLDER}/../backup'
IMG_FOLDER = f'{SCRIPT_FOLDER}/../image'
FONT_FOLDER = f'{SCRIPT_FOLDER}/../font'

FONT_FILE = f'{FONT_FOLDER}/arialunicodems.ttf'

KANJI_DIC = f'{DATA_FOLDER}/kanji.json'

BKP_HISTORY = \
    f'{BACKUP_FOLDER}/history.p'
BKP_MODEL = [
    f'{BACKUP_FOLDER}/autoencoder.h5',
    f'{BACKUP_FOLDER}/encoder.h5',
    f'{BACKUP_FOLDER}/decoder.h5']

IMG_SIZE = 100
REL_FONT_SIZE = 0.98
N_ROTATION = 4
EPOCHS = 100
SEED = 123
RATIO_TRAINING_TEST = 5/6

TEXT_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (0, 0, 0)

os.makedirs(BACKUP_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)


kanji_dic = json.load(open(KANJI_DIC, 'r'))


def _img_file(name):

    return f"{IMG_FOLDER}/{name}.png"


def _create_images():

    font_size = int(IMG_SIZE)

    # kanji_dic = {k: i for i, k in enumerate(kanji_list)}

    for character, name in kanji_dic.items():

        img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=BACKGROUND_COLOR)

        fnt = ImageFont.truetype(FONT_FILE, font_size)
        d = ImageDraw.Draw(img)
        w, h = d.textsize(character, font=fnt)
        d.text(((IMG_SIZE-w)/2, (IMG_SIZE-h)/2 - h/10), character, font=fnt, fill=TEXT_COLOR)

        if name is None:
            name = ord(character)

        img.save(_img_file(name))


def _images_exist():

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


def formatted_images():

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

        # Vertical Flipping
        flip_v_img = np.flipud(img)
        train_data[i] = flip_v_img
        i += 1

        # Vertical and horizontal Flipping
        flip_v_img = np.flipud(flip_h_img)
        train_data[i] = flip_v_img
        i += 1

    train_data /= 255.
    return train_data


def training_and_test_data():

    if not _images_exist():
        _create_images()

    train_data = formatted_images()

    assert(len(train_data) > 0)

    np.random.shuffle(train_data)

    split = int(len(train_data) * RATIO_TRAINING_TEST)

    x_train = train_data[:split]
    x_test = train_data[split:]

    return x_train, x_test


def _show_image_example(train_data):

    plt.imshow(train_data[0], cmap='gist_gray')
    plt.show()

# ----------------------------------------- #


def get_formatted_image_for_cnn(kanji):

    kanji_id = kanji_dic[kanji]

    path = os.path.join(IMG_FOLDER, _img_file(kanji_id))
    img = Image.open(path)
    img = img.convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    a_img = np.array(img, dtype='float32')
    a_img /= 255.
    a_img = np.reshape(a_img, (1, a_img.shape[0], a_img.shape[1], 1))
    return a_img

# --------------------------------------- #


def get_encoded_decoded(encoder, autoencoder, decoder, x_test, convolutional=True):

    if convolutional:
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
