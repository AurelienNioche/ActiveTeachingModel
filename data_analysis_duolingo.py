import os

# Django specific settings
import plot.duolingo

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

from duolingo.models import Item

import datetime
from time import time

import numpy as np
from tqdm import tqdm

import googletrans

import pickle

from fit import fit
from behavior import data_structure
from learner.act_r import ActR
import plot.success

from utils.utils import dump, load, print_begin, print_done

TR = googletrans.Translator()


def translate(word, src='ja', dest='en'):

    return TR.translate(word, src=src, dest=dest).text


class Data:

    file_path = os.path.join("data", "data.p")

    def __init__(self):
        pass

    @classmethod
    def load(cls):

        if not os.path.exists(cls.file_path):
            t = time()
            print("Loading data from file...", end=" ", flush=True)
            data = cls()
            print(f"Done! [time elapsed "
                  f"{datetime.timedelta(seconds=time()-t)}]")
            dump(data, cls.file_path)

        else:
            return load(cls.file_path)


class Db(Data):

    file_path = os.path.join("data", "db.p")

    def __init__(self):

        super().__init__()

        self.user_id, self.lexeme_id, self.ui_language, \
            self.learning_language = \
            np.asarray(Item.objects.all().order_by('timestamp')
                       .values_list('user_id', 'lexeme_id',
                                    'ui_language',
                                    'learning_language')).T

        self.h_seen, self.h_correct, self.s_seen, self.s_correct, \
            self.time_stamp = \
            np.asarray(Item.objects.all().order_by('timestamp')
                       .values_list('history_seen', 'history_correct',
                                    'session_seen', 'session_correct',
                                    'timestamp')).T


class UserData:

    def __init__(self, user_id):

        self.user_id = user_id

        lexeme_id, lexeme_string = \
            np.asarray(Item.objects.filter(user_id=user_id)
                       .order_by('timestamp')
                       .values_list('lexeme_id', 'lexeme_string')).T

        h_seen, h_correct, s_seen, s_correct, time_stamp = \
            np.asarray(Item.objects.filter(user_id=user_id)
                       .order_by('timestamp')
                       .values_list('history_seen', 'history_correct',
                                    'session_seen', 'session_correct',
                                    'timestamp')).T

        it = \
            Item.objects.filter(user_id=user_id)[0]
        self.ui_language = it.ui_language
        self.learning_language = it.learning_language

        ui_language, learning_language = \
            np.asarray(Item.objects.filter(user_id=user_id).values_list(
                'ui_language', 'learning_language'
            )).T

        assert len(np.unique(ui_language)) == 1, 'ui_language > 1!'
        assert len(np.unique(learning_language)) == 1, 'learning_language > 1!'

        unq_lex_id, lex_idx = np.unique(
            lexeme_id, return_inverse=True)

        self.n_item = len(lexeme_id)

        self.t_max = 0

        questions = []
        replies = []
        times = []

        for i in range(self.n_item):

            # Get the idx of the lexeme
            l_idx = lex_idx[i]

            success = np.zeros(s_seen[i], dtype=bool)
            success[:s_correct[i]] = 1

            to_add = [-1, l_idx]

            for s in success:
                questions.append(l_idx)
                replies.append(to_add[int(s)])
                times.append(time_stamp[i])
                self.t_max += 1

        self.questions = np.asarray(questions)
        self.replies = np.asarray(replies)
        self.times = np.asarray(times)

    @classmethod
    def load(cls, user_id):

        folder_path = os.path.join("data", "duolingo_user")
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, f"data_u{user_id}.p")

        if not os.path.exists(file_path):
            t = time()
            print("Loading data from file...", end=" ", flush=True)
            data = cls(user_id=user_id)
            print(f"Done! [time elapsed "
                  f"{datetime.timedelta(seconds=time()-t)}]")
            dump(data, file_path)
            return data

        else:
            return load(file_path)


def general_statistics(force=False):

    file_path = os.path.join("data", "stats.p")
    if not force and os.path.exists(file_path):
        return load(file_path)

    d = Db.load()

    unq_user_id, user_idx = np.unique(d.user_id, return_inverse=True)
    unq_lex_id, lex_idx = np.unique(d.lexeme_id, return_inverse=True)

    print("Period:",
          datetime.timedelta(seconds=int(d.time_stamp[-1] - d.time_stamp[0])))

    n_users = len(unq_user_id)
    n_lex = len(unq_lex_id)
    print("N users:", n_users)
    print("N lexeme:", n_lex)

    ui_learn = np.stack((d.ui_language, d.learning_language), axis=-1)
    unq_ui_learn, count_ui_learn = np.unique(ui_learn, return_counts=True,
                                             axis=0)

    t = print_begin('Computing the number of iterations / lexemes by user...')
    n_it = np.zeros(n_users, dtype=int)
    n_lex_ind = np.zeros(n_users, dtype=int)
    for i in tqdm(range(n_users)):

        bool_u = user_idx == i
        n_it[i] = np.sum(d.s_seen[bool_u])
        n_lex_ind[i] = len(np.unique(lex_idx[bool_u]))

    print_done(t)

    print(f"N it/ind: {np.mean(n_it)} (+/-{np.std(n_it)})")
    print(f"N lex/ind: {np.mean(n_lex_ind)} (+/-{np.std(n_lex_ind)})")

    print('UI-language & Learnt language:')
    for u in unq_ui_learn:

        u_id, = np.asarray(Item.objects.filter(ui_language=u[0],
                                               learning_language=u[1])
                                       .order_by('user_id')
                                       .values_list('user_id')).T
        n = len(np.unique(u_id))
        print(f'{u} (n={n})')


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
    # unq_lex_id, lex_idx = np.unique(lexeme_id, return_inverse=True)

    n = len(unq_user_id)
    print('n subjects', n)
    # idx = np.zeros(n)

    selection = []
    n_selected = 0
    for u_idx in tqdm(range(n)):

        u_bool = user_idx == u_idx

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

        if comp_hist:
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

        success = np.zeros(s_seen[i], dtype=bool)
        success[:s_correct[i]] = 1

        to_add = [-1, l_idx]

        for s in success:
            questions.append(l_idx)
            replies.append(to_add[int(s)])
            times.append(time_stamp[i])
            t_max += 1

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

    times = np.asarray(times)

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

    plot.duolingo.bar(sorted(counts), fig_name=f'counts_u{u_id}.pdf')

    f = fit.Fit(tk=tk, model=model, data=data, verbose=True,
                fit_param={'use_p_correct': True})

    fit_r = f.evaluate()


def info_user(user_id):

    u = UserData.load(user_id=user_id)

    print('user_id:', u.user_id)
    print('n iterations:', u.t_max)
    print('n items:', u.n_item)
    print('language ui:', u.ui_language)
    print('language learned:', u.learning_language)
    print('Total period:',
          datetime.timedelta(seconds=int(u.times[-1]-u.times[0])))

    u_q, counts = np.unique(
        u.questions, return_counts=True)

    plot.duolingo.bar(sorted(counts), fig_name=f'counts_u{user_id}.pdf')


def main():

    # info_user('u:IY_')
    general_statistics()

    # subjects_selection_history()

    # s = load('data/selection_full_h_all.p')
    # t_max, n_item = count(s)
    # best_idx = np.argsort(t_max)[-2]
    # print('user_id:', s[best_idx])
    # print('t_max:', t_max[best_idx])
    # print('n_item:', n_item[best_idx])
    #
    # fit_user(s[best_idx])
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
