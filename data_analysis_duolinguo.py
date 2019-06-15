import os

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

from fit import fit
from behavior import data_structure
from learner.act_r import ActR

TR = googletrans.Translator()


def translate(word, src='ja', dest='en'):

    return TR.translate(word, src=src, dest=dest).text


def subjects_selection():

    file_path = os.path.join("data", "selection.p")
    if os.path.exists(file_path):
        return pickle.load(open(file_path, 'rb'))

    print("Loading from db...", end=" ", flush=True)
    user_id, lexeme_id, history_seen = \
        np.asarray(Item.objects.all()
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

    pickle.dump(selection, open(file_path, 'wb'))
    return selection


def count():

    selection = subjects_selection()
    for s in selection:

        lexeme_id, h_seen, h_correct, time_stamp = \
            np.asarray(Item.objects.filter(learning_language='en',
                                           user_id=s)
                       .values_list('lexeme_id', 'history_seen',
                                    'history_correct',
                                    'timestamp')).T
        print(s, 'unique', len(np.unique(lexeme_id)), 'trials', len(lexeme_id))


def fit_user(u_id='u:iOIr'):

    lexeme_id, lexeme_string = \
        np.asarray(Item.objects.filter(user_id=u_id)
                   .values_list('lexeme_id', 'lexeme_string')).T

    h_seen, h_correct, time_stamp = \
        np.asarray(Item.objects.filter(user_id=u_id)
                   .values_list('history_seen', 'history_correct',
                                'timestamp')).T

    print(h_seen)

    t_max = len(lexeme_id)
    model = ActR

    tk = data_structure.Task(t_max=t_max)
    data = data_structure.Data(t_max=t_max)

    f = fit.Fit(tk=tk, model=model, data=data, verbose=True)

    return f.evaluate()
    # print(fit_r)


def main():

    fit_user()

    # print(translate('leer', 'es'))


if __name__ == "__main__":

    main()
