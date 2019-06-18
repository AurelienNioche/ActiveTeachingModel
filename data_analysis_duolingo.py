import os

# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

from duolingo.models import Item

import datetime

import numpy as np
from tqdm import tqdm

import googletrans

import pickle

from fit import fit
from behavior import data_structure
from learner.act_r import ActR
import plot.success

from utils.utils import dump, load

TR = googletrans.Translator()


def translate(word, src='ja', dest='en'):

    return TR.translate(word, src=src, dest=dest).text


def subjects_selection_trials_n(thr=100):

    file_path = os.path.join("data", "selection_trials_n.p")
    if os.path.exists(file_path):
        selection = load(file_path)
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

    dump(selection, file_path)
    print('n selected', len(selection))
    return selection


def subjects_selection_history(force=False):

    file_path = os.path.join("data", "selection_full_h_all.p")
    if not force and os.path.exists(file_path):
        selection = pickle.load(open(file_path, 'rb'))
        print('n selected', len(selection))
        return selection

    print("Loading from db...", end=" ", flush=True)
    user_id, lexeme_id = \
        np.asarray(Item.objects.all().order_by('timestamp')
                   .values_list('user_id', 'lexeme_id')).T
    print("Done!")
    unq_user_id, user_idx = np.unique(user_id, return_inverse=True)
    unq_lex_id, lex_idx = np.unique(lexeme_id, return_inverse=True)

    h_seen, h_correct, s_seen, s_correct, time_stamp = \
        np.asarray(Item.objects.all().order_by('timestamp')
                   .values_list('history_seen', 'history_correct',
                                'session_seen', 'session_correct',
                                'timestamp')).T

    assert h_seen.dtype == int

    hs_first = h_seen == s_seen

    n = len(unq_user_id)
    print('n subjects', n)
    # idx = np.zeros(n)

    selection = []
    # n_selected = 0
    for u_idx in tqdm(range(n)):

        u_bool = user_idx == u_idx

        # assert len(np.unique(user_id[u_bool])) == 1
        u_lex_idx = lex_idx[u_bool]
        unq_lex = np.unique(u_lex_idx)

        comp_hist = np.sum(u_bool*hs_first) == len(unq_lex)

        # comp_hist = True
        # for u_l in unq_lex:
        #     # For each idx of lexeme that user saw at least one...
        #     rel_lex = lex_idx == u_l
        #     cond = u_bool * hs_first * rel_lex
        #     if cond.sum() == 0:
        #         comp_hist = False
        #         break

        if comp_hist:
            #  n_selected += 1
            selection.append(unq_user_id[u_idx])

    pickle.dump(selection, open(file_path, 'wb'))
    print('n selected', len(selection))
    return selection


def count(selection, verbose=True):

    t_max = []
    n_item = []

    for s in selection:

        lexeme_id, = \
            np.asarray(Item.objects.filter(user_id=s)
                       .values_list('lexeme_id', )).T

        s_seen,  = \
            np.asarray(Item.objects.filter(user_id=s)
                       .values_list('session_seen',
                                    )).T
        _t_max = np.sum(s_seen)
        _n_item = len(np.unique(lexeme_id))

        t_max.append(_t_max)
        n_item.append(_n_item)

    if verbose:

        best_idx = np.argsort(t_max)[-1]
        print("*** USER WITH THE MOST ITERATIONS ***")
        print('user_id:', selection[best_idx])
        print('t_max:', t_max[best_idx])
        print('n_item:', n_item[best_idx])

    return t_max, n_item


def n_trial(u_id='u:iOIr'):
    """
    :param u_id: User id (string)
    :return: t_max and number of unique lexeme (tuple)
    """

    lexeme_id, lexeme_string = \
        np.asarray(Item.objects.filter(user_id=u_id).order_by('timestamp')
                   .values_list('lexeme_id', 'lexeme_string')).T

    # h_seen, h_correct, time_stamp = \
    #     np.asarray(Item.objects.filter(user_id=u_id).order_by('timestamp')
    #                .values_list('history_seen', 'history_correct',
    #                             'timestamp')).T

    unq_lex_id, lex_idx = np.unique(lexeme_id, return_inverse=True)
    return len(lexeme_id), len(unq_lex_id)


def fit_user(u_id='u:iOIr', ignore_times=False):

    lexeme_id, lexeme_string = \
        np.asarray(Item.objects.filter(user_id=u_id).order_by('timestamp')
                   .values_list('lexeme_id', 'lexeme_string')).T

    h_seen, h_correct, s_seen, s_correct, time_stamp = \
        np.asarray(Item.objects.filter(user_id=u_id).order_by('timestamp')
                   .values_list('history_seen', 'history_correct',
                                'session_seen', 'session_correct',
                                'timestamp')).T

    it = \
        Item.objects.filter(user_id=u_id)[0]
    lg_ui = it.ui_language
    lg_learn = it.learning_language

    ui_language, learning_language = \
        np.asarray(Item.objects.filter(user_id=u_id).values_list(
            'ui_language', 'learning_language'
        )).T

    assert len(np.unique(ui_language)) == 1, 'ui_language > 1!'
    assert len(np.unique(learning_language)) == 1, 'learning_language > 1!'

    unq_lex_id, lex_idx = np.unique(
        lexeme_id, return_inverse=True)

    n = len(lexeme_id)

    t_max = 0

    questions = []
    replies = []
    times = []

    for i in range(n):

        # Get the idx of the lexeme
        l_idx = lex_idx[i]

        # if previous_h_correct[l_idx] == -1:
        #     previous_h_correct[l_idx] = h_correct[i]
        #     if l_idx == 0:
        #         print('ignore')
        #     continue
        success = np.zeros(s_seen[i], dtype=bool)
        success[:s_correct[i]] = 1

        to_add = [-1, l_idx]

        for s in success:
            questions.append(l_idx)
            replies.append(to_add[int(s)])
            times.append(time_stamp[i])
            t_max += 1

        # if l_idx == 0:
        #     print(h_correct[i], h_seen[i], s_correct[i], s_seen[i],
        #           time_stamp[i], success)
    print('user_id:', u_id)
    print('n iterations:', t_max)
    print('n items:', len(unq_lex_id))
    print('language ui:', lg_ui)
    print('language learned:', lg_learn)
    print('Total period:',
          datetime.timedelta(seconds=int(time_stamp[-1]-time_stamp[0])))

    model = ActR

    # times = np.asarray(times)
    if ignore_times:
        times = None

    # times[:] = times/60

    tk = data_structure.Task(t_max=t_max)
    data = data_structure.Data(t_max=t_max,
                               questions=questions,
                               replies=replies,
                               times=times)

    # Plot the success curves
    success = data.questions == data.replies
    plot.success.curve(successes=success,
                       fig_name=f'success_curve_u{u_id}.pdf')
    plot.success.multi_curve(questions=questions, replies=replies,
                             fig_name=f'success_curve_u{u_id}_multi.pdf',
                             max_lines=40)

    u_q, counts = np.unique(
        questions, return_counts=True)

    plot.success.bar(sorted(counts), fig_name=f'counts_u{u_id}.pdf')

    f = fit.Fit(tk=tk, model=model, data=data, verbose=True)

    fit_r = f.evaluate()


def main():

    # subjects_selection_history()

    s = load('data/selection_full_h_all.p')
    t_max, n_item = count(s)
    best_idx = np.argsort(t_max)[-2]
    print('user_id:', s[best_idx])
    print('t_max:', t_max[best_idx])
    print('n_item:', n_item[best_idx])

    fit_user(s[best_idx])
    # fit_user('u:IY_')
    # subjects_selection_history()

    # a = subjects_selection_trials_n()
    #
    # best_length_idx = np.argsort([i for i, j in map(n_trial, a)])
    #
    # selected_u = a[best_length_idx[-2]]
    # print("selected user:", selected_u)
    #
    # # 'u:bcH_'
    # print(n_trial(selected_u))
    # fit_user(selected_u, ignore_times=True)

    # print(translate('leer', 'es'))


if __name__ == "__main__":

    main()
