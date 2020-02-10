import os

PILOT_NAME = "pilot_2019_09_02"

N_POSSIBLE_REPLIES = 6

BKP_FOLDER = os.path.join("user_data", "bkp")

BKP_FIT = os.path.join(
    BKP_FOLDER, PILOT_NAME, "fit"
)

BKP_USER_LIST = \
    os.path.join(
        BKP_FOLDER, PILOT_NAME,
        "user", "user.p")

BKP_HIST = os.path.join(
        BKP_FOLDER, PILOT_NAME,
        "history")


BKP_GRAPHIC_CONNECTION = \
    os.path.join(
        BKP_FOLDER, PILOT_NAME,
        "similarity",
        "graphic_connection.p"
    )

BKP_SEMANTIC_CONNECTION = \
    os.path.join(
        BKP_FOLDER, PILOT_NAME,
        "similarity",
        "semantic_connection.p")

BKP_WORD2VEC_MODEL = \
    os.path.join(
        BKP_FOLDER,
        'word2vec', 'word_vectors',
        'word_vectors.kv')

BKP_WORD2VEC_TRAINED = \
    os.path.join(
        BKP_FOLDER,
        'word2vec',
        'GoogleNews-vectors-negative300.bin'
    )

BKP_WORD2VEC_REPLACEMENT_DIC = \
    os.path.join(
        BKP_FOLDER,
        'word2vec',
        'replacement.json')
