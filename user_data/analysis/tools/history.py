import os
import pickle


from user_data.settings import BKP_HIST


def get(user_id):

    bkp_file = os.path.join(BKP_HIST, f"user_{user_id}.p")

    if os.path.exists(bkp_file):

        print("Loading from pilot_2019_09_02 file...")
        hist_question, hist_success, seen = \
            pickle.load(open(bkp_file, 'rb'))
        return hist_question, hist_success, seen

    else:
        raise Exception('Data not found')
