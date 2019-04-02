import os
import sys

import pickle
import numpy as np

from itertools import combinations


try:
    from auto_encoder.tools import backup
    from auto_encoder.tools import image
    from auto_encoder.model import cnn
    from auto_encoder.tools import evaluation
except ModuleNotFoundError:
    from pathlib import Path  # if you haven't already done so

    file = Path(__file__).resolve()
    parent, root = file.parent, file.parents[1]
    sys.path.append(str(root))

    # Additionally remove the current file's directory from sys.path
    try:
        sys.path.remove(str(parent))
    except ValueError:  # Already removed
        pass

    from auto_encoder.tools import backup
    from auto_encoder.tools import image
    from auto_encoder.model import cnn
    from auto_encoder.tools import evaluation

SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
BACKUP_FOLDER = f'{SCRIPT_FOLDER}/backup'
IMG_FOLDER = f'{SCRIPT_FOLDER}/image'

BKP_HISTORY = \
    f'{BACKUP_FOLDER}/history.p'
BKP_MODEL = [
    f'{BACKUP_FOLDER}/autoencoder.h5',
    f'{BACKUP_FOLDER}/encoder.h5',
    f'{BACKUP_FOLDER}/decoder.h5']

GRAPHIC_PROPERTIES = f'{BACKUP_FOLDER}/graphic_properties.p'
SIMILARITIES = f'{BACKUP_FOLDER}/similarities.p'

IMG_SIZE = 100
EPOCHS = 100
SEED = 123

os.makedirs(BACKUP_FOLDER, exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)


def _get_models(force=False):

    os.makedirs(BACKUP_FOLDER, exist_ok=True)

    # Check if bkp files already exist
    files_exist = np.all([os.path.exists(i) for i in BKP_MODEL + [BKP_HISTORY]])

    if force or not files_exist:

        # Seed
        np.random.seed(SEED)

        # Get the data set
        x_train, x_test = image.training_and_test_data()

        # Create and train autoencoder
        (autoencoder, encoder, decoder), history = cnn.train_autoencoder(x_train, x_test, epochs=EPOCHS)

        # Save models and history
        backup.save_model_and_history(models=(autoencoder, encoder, decoder), history=history)

    else:
        # Load models
        (autoencoder, encoder, decoder), history = backup.load_model_and_history()

    return (autoencoder, encoder, decoder), history


def _normalize(a):
    return np.interp(a, (np.nanmin(a), np.nanmax(a)), (0, 1))


class Similarity:

    def __init__(self):

        self.sim = self.get_similarity()

    @classmethod
    def get_similarity(cls, force=False):

        os.makedirs(BACKUP_FOLDER, exist_ok=True)

        if force or not os.path.exists(SIMILARITIES):
            sim = cls._create_similarity(force=force)
            # Save in pickle
            pickle.dump(obj=sim, file=open(SIMILARITIES, 'wb'))

        else:
            sim = pickle.load(file=open(SIMILARITIES, 'rb'))

        return sim

    @classmethod
    def _create_similarity(cls, force=False, normalize=True):

        graphic_prop = cls._get_graphic_properties(force=force)

        kanji_list = list(image.kanji_dic.keys())

        n_kanji = len(kanji_list)

        dist = np.zeros((n_kanji, n_kanji))

        for i in range(n_kanji):
            dist[i, i] = np.nan

        for i, j in combinations(range(n_kanji), 2):
            a = kanji_list[i]
            b = kanji_list[j]

            x = graphic_prop[a]
            y = graphic_prop[b]

            distance = np.abs(np.linalg.norm(x - y))

            dist[i, j] = dist[j, i] = distance

        if normalize:
            dist = _normalize(dist)
            sim_array = 1 - dist

        else:
            sim_array = - dist

        sim = {}
        for i, j in combinations(range(n_kanji), 2):
            a = kanji_list[i]
            b = kanji_list[j]

            sim[a, b] = sim[b, a] = sim_array[i, j]

        return sim

    @classmethod
    def _get_graphic_properties(cls, force=False):

        os.makedirs(BACKUP_FOLDER, exist_ok=True)

        if force or not os.path.exists(GRAPHIC_PROPERTIES):
            graphic_prop = cls._create_graphic_properties()
            # Save in pickle
            pickle.dump(obj=graphic_prop, file=open(GRAPHIC_PROPERTIES, 'wb'))

        else:
            graphic_prop = pickle.load(file=open(GRAPHIC_PROPERTIES, 'rb'))

        return graphic_prop

    @classmethod
    def _create_graphic_properties(cls, force=False):

        # Get the trained models
        (autoencoder, encoder, decoder), history = _get_models(force=force)

        kanji_list = list(image.kanji_dic.keys())

        n_kanji = len(kanji_list)

        # Dimension of the entry of the encoder
        dim_entry = encoder.layers[0].input_shape[1:]

        # Get the images formatted for the autoencoder
        a = np.zeros((n_kanji,) + dim_entry)

        for i, k in enumerate(kanji_list):
            a[i] = image.get_formatted_image_for_cnn(k)

        # Get the encoded value
        v = encoder.predict(a)

        # Create a dictionary with graphic representation of all kanji
        graphic_prop = {}

        for i, k in enumerate(kanji_list):
            encoded = v[i]
            graphic_prop[k] = encoded
        return graphic_prop

    def __call__(self, kanji_a, kanji_b):

        return self.sim[kanji_a, kanji_b]


def get_similarity():
    return Similarity()


# ----------------------------------------------------- #


def show_training_dataset():
    import matplotlib.pyplot as plt

    img = image.formatted_images()
    for i, x in enumerate(img):
        plt.imshow(x, cmap='gist_gray')
        plt.title(f'Image {i}')
        plt.show()


def evaluate_model():

    # Get the data set
    x_train, x_test = image.training_and_test_data()

    # Get the models
    (autoencoder, encoder, decoder), history = _get_models()

    encoded_images, decoded_images, decoded_images_auto = \
        image.get_encoded_decoded(x_test=x_test, encoder=encoder, decoder=decoder, autoencoder=autoencoder)
    evaluation.show_result(x_test=x_test, decoded_images=decoded_images, decoded_images_auto=decoded_images_auto)
    evaluation.show_accuracy(history)

# ------------------------------------------------- #


def demo(kanji_list=('八', '花', '虫', '中', '王', '足', '生', '力', '七', '二')):

    s = Similarity()

    for i, j in combinations(range(len(kanji_list)), 2):
        print(s(i, j))


if __name__ == "__main__":

    demo()
