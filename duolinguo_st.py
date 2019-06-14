import os
import django.db.utils
import psycopg2

# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

from duolingo.models import Item

import numpy as np
from tqdm import tqdm

import googletrans

import pickle


TR = googletrans.Translator()


def translate(word, src='ja', dest='en'):

    return TR.translate(word, src=src, dest=dest).text


def subject_selection():

    user_id = [i[0] for i in Item.objects.filter(learning_language='en').
               order_by().values_list('user_id').distinct()]

    selection = []
    n = 0
    for u_id in user_id:

        lexeme_id = [i[0] for i in Item.objects.filter(
            user_id=u_id
        ).order_by().values_list('lexeme_id').distinct()]

        all_history = True

        for l_id in lexeme_id:
            c = Item.objects.filter(history_seen=1,
                                    user_id=u_id, lexeme_id=l_id).exists()
            if not c:
                all_history = False
                break

        if all_history:
            n += 1
            selection.append(u_id)

        print(n)

    file_path = os.path.join("data", "selection.p")
    pickle.dump(selection, open(file_path, 'wb'))

    #
    # for e in Item.objects.all():
    # print(len(np.unique(a)))


def subjects_selection_method2():
    print("Loading from db...", end= " ", flush=True)
    user_id, lexeme_id, history_seen = \
        np.asarray(Item.objects.filter(learning_language='en')
                   .values_list('user_id', 'lexeme_id', 'history_seen')).T
    print("Done!")
    unq_user_id, user_idx = np.unique(user_id, return_inverse=True)
    unq_lex_id, lex_idx = np.unique(lexeme_id, return_inverse=True)

    history_seen = np.array([int(i) for i in history_seen])

    hs_first = history_seen == 1

    n = len(unq_user_id)
    print('n subjects', n)
    # idx = np.zeros(n)

    selection = []
    n_selected = 0
    for u_idx in tqdm(range(n)):

        u_bool = user_idx == u_idx

        # assert len(np.unique(user_id[u_bool])) == 1
        u_lex_idx = lex_idx[u_bool]
        unq_lex = np.unique(u_lex_idx)

        comp_hist = True
        for u_l in unq_lex:
            # For each idx of lexeme that user saw at least one...
            rel_lex = lex_idx == u_l
            cond = u_bool * hs_first * rel_lex
            if cond.sum() == 0:
                comp_hist = False
                break

        if comp_hist:
            n_selected += 1
            selection.append(unq_user_id[u_idx])
    # n = 0
    # for u_id in user_id:
    #
    #     lexeme_id = [i[0] for i in Item.objects.filter(
    #         user_id=u_id
    #     ).order_by().values_list('lexeme_id').distinct()]
    #
    #     all_history = True
    #
    #     for l_id in lexeme_id:
    #         c = Item.objects.filter(history_seen=1,
    #                                 user_id=u_id, lexeme_id=l_id).exists()
    #         if not c:
    #             all_history = False
    #             break
    #
    #     if all_history:
    #         n += 1
    #         selection.append(u_id)
    #
    #     print(n)

    file_path = os.path.join("data", "selection.p")
    pickle.dump(selection, open(file_path, 'wb'))


def main():

    print(translate('leer', 'es'))


if __name__ == "__main__":

    subjects_selection_method2()
