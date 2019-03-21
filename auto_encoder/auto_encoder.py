import numpy as np
import os
import sys


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

IMG_SIZE = 100
EPOCHS = 100
SEED = 123


def get_models(kanji_dic=None, force=False):

    os.makedirs(BACKUP_FOLDER, exist_ok=True)

    # Check if bkp files already exist
    files_exist = np.all([os.path.exists(i) for i in BKP_MODEL + [BKP_HISTORY]])

    if force or not files_exist:

        assert kanji_dic is not None, 'You should give a kanji_dic in order to train the model!'

        # Seed
        np.random.seed(SEED)

        # Get the data set
        x_train, x_test = image.training_and_test_data(kanji_dic=kanji_dic)

        # Create and train autoencoder
        (autoencoder, encoder, decoder), history = cnn.train_autoencoder(x_train, x_test, epochs=EPOCHS)

        # Save models and history
        backup.save_model_and_history(models=(autoencoder, encoder, decoder), history=history)

    else:
        # Load models
        (autoencoder, encoder, decoder), history = backup.load_model_and_history()

    # encoded_images, decoded_images, decoded_images_auto = \
    #     get_encoded_decoded(x_test=x_test, encoder=encoder, decoder=decoder, autoencoder=autoencoder)
    # show_result(x_test=x_test, decoded_images=decoded_images, decoded_images_auto=decoded_images_auto)
    # show_accuracy(history)

    return (autoencoder, encoder, decoder), history


def get_formatted_image_for_cnn(kanji_id):

    return image.get_formatted_image_for_cnn(kanji_id=kanji_id)


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
    (autoencoder, encoder, decoder), history = get_models()

    encoded_images, decoded_images, decoded_images_auto = \
        image.get_encoded_decoded(x_test=x_test, encoder=encoder, decoder=decoder, autoencoder=autoencoder)
    evaluation.show_result(x_test=x_test, decoded_images=decoded_images, decoded_images_auto=decoded_images_auto)
    evaluation.show_accuracy(history)


if __name__ == "__main__":

    show_training_dataset()

