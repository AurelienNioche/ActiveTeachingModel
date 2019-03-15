import numpy as np
import os

from auto_encoder.tools import backup
from auto_encoder.tools import image
from auto_encoder.model import cnn


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


def get_models(kanji_dic, force=False):

    # Seed
    np.random.seed(SEED)

    os.makedirs(BACKUP_FOLDER, exist_ok=True)

    # Check if bkp files already exist
    files_exist = np.all([os.path.exists(i) for i in BKP_MODEL + [BKP_HISTORY]])

    if force or not files_exist:

        # Get the data set
        x_train, x_test = image.training_data(kanji_dic=kanji_dic)

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

    return autoencoder, encoder, decoder


def get_formatted_image_for_cnn(kanji_id):

    return image.get_formatted_image_for_cnn(kanji_id=kanji_id)
