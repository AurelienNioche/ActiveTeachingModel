import os
import pickle

from keras.models import load_model


SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
BACKUP_FOLDER = f'{SCRIPT_FOLDER}/../backup'

BKP_HISTORY = \
    f'{BACKUP_FOLDER}/history.p'

BKP_MODEL = [
    f'{BACKUP_FOLDER}/autoencoder.h5',
    f'{BACKUP_FOLDER}/encoder.h5',
    f'{BACKUP_FOLDER}/decoder.h5']

os.makedirs(BACKUP_FOLDER, exist_ok=True)


def save_model_and_history(history, models):

    # Backup in pickle
    pickle.dump(history, file=open(BKP_HISTORY, 'wb'))

    for bmf, m in zip(BKP_MODEL, models):
        m.save(bmf)


def load_model_and_history():

    history = pickle.load(file=open(BKP_HISTORY, 'rb'))
    models = [load_model(i) for i in BKP_MODEL]

    return models, history
