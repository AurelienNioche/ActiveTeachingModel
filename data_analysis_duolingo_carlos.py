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


def subjects_selection_trials_n(thr=100):

    file_path = os.path.join("data", "selection_trials_n.p")
    if os.path.exists(file_path):
        selection = pickle.load(open(file_path, 'rb'))
        print('n selected', len(selection))
        return selection

    print("Loading from db...", end=" ", flush=True)
    user_id, lexeme_id = \
        np.asarray(Item.objects.all()
                   .values_list('user_id', 'lexeme_id')).T
    print("Done!")
    unq_user_id, user_idx = np.unique(user_id, return_inverse=True)
    unq_lex_id, lex_idx = np.unique(lexeme_id, return_inverse=True)

    n = len(unq_user_id)
    print('n subjects', n)
    # idx = np.zeros(n)

    selection = []
    n_selected = 0
    for u_idx in tqdm(range(n)):

        u_bool = user_idx == u_idx

        # assert len(np.unique(user_id[u_bool])) == 1
        u_lex_idx = lex_idx[u_bool]
        # unq_lex = np.unique(u_lex_idx)

        selected = np.sum(u_bool) > thr

        if selected:
            n_selected += 1
            selection.append(unq_user_id[u_idx])

    pickle.dump(selection, open(file_path, 'wb'))
    print('n selected', len(selection))
    return selection


def subjects_selection_history():

    file_path = os.path.join("data", "selection_full_h_all.p")
    if os.path.exists(file_path):
        selection = pickle.load(open(file_path, 'rb'))
        print('n selected', len(selection))
        return selection

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
    print('n selected', len(selection))
    return selection


def count():

    selection = subjects_selection_history()
    for s in selection:

        lexeme_id, h_seen, h_correct, time_stamp = \
            np.asarray(Item.objects.filter(user_id=s)
                       .values_list('lexeme_id', 'history_seen',
                                    'history_correct',
                                    'timestamp')).T
        print(s, 'unique', len(np.unique(lexeme_id)), 'trials', len(lexeme_id))


def fit_user(u_id='u:iOIr'):

    lexeme_id, lexeme_string = \
        np.asarray(Item.objects.filter(user_id=u_id).order_by('timestamp')
                   .values_list('lexeme_id', 'lexeme_string')).T

    h_seen, h_correct, time_stamp = \
        np.asarray(Item.objects.filter(user_id=u_id).order_by('timestamp')
                   .values_list('history_seen', 'history_correct',
                                'timestamp')).T

    unq_lex_id, lex_idx = np.unique(lexeme_id, return_inverse=True)

    n = len(lexeme_id)

    t_max = 0

    questions = []
    replies = []
    times = []

    n_unq_lex = len(unq_lex_id)

    # previous_h_seen = np.zeros(n_unq_lex, dtype=int)
    previous_h_correct = np.zeros(n_unq_lex, dtype=int)

    for i in range(n):

        l_idx = lex_idx[i]
        success = h_correct[l_idx] - previous_h_correct[l_idx]

        if success:
            replies.append(l_idx)
        else:
            replies.append(-1)

        questions.append(l_idx)

        times.append(time_stamp[i])

        # For next iteration
        previous_h_correct[l_idx] = h_correct[l_idx]
        t_max += 1

    model = ActR

    tk = data_structure.Task(t_max=t_max)
    data = data_structure.Data(t_max=t_max,
                               questions=questions,
                               replies=replies,
                               times=times)

    f = fit.Fit(tk=tk, model=model, data=data, verbose=True)

    return f.evaluate()
    # print(fit_r)


def main():

    fit_user()

    # print(translate('leer', 'es'))


if __name__ == "__main__":

    main()
