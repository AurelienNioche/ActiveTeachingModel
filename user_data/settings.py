import os

PILOT_NAME = "pilot_2019_09_02"

N_POSSIBLE_REPLIES = 6

BKP_FOLDER = os.path.join("user_data", "bkp")

BKP_DATA = os.path.join(
    BKP_FOLDER, PILOT_NAME,
    f"data_{PILOT_NAME}.p"
)

BKP_FIT = os.path.join(
    BKP_FOLDER, PILOT_NAME,
    f"fit_{PILOT_NAME}.p"
)

BKP_CONNECTION = \
    os.path.join(
        BKP_FOLDER, PILOT_NAME,
        "similarity")

# The source of stroke data for each character
STROKE_SOURCE = os.path.join(BKP_FOLDER,
                             "simsearch",
                             'stroke_ulrich')

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
