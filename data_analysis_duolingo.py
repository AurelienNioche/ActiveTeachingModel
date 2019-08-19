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
from behavior.data_structure import Task
from behavior.duolingo import UserData
from learner.act_r import ActR
# from learner.act_r_duo import ActRDuo
import plot.success
import plot.memory_trace
from simulation.memory import p_recall_over_time_after_learning

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
    user_id, = \
        np.asarray(Item.objects.all()
                   .values_list('user_id', )).T
    print("Done!")
    unq_user_id, user_idx = np.unique(user_id, return_inverse=True)
    # unq_lex_id, lex_idx = np.unique(lexeme_id, return_inverse=True)

    h_seen, = \
        np.asarray(Item.objects.all().order_by('timestamp')
                   .values_list('history_seen', )).T

    hs_first = h_seen == 1

    n = len(unq_user_id)
    print('n subjects', n)
    # idx = np.zeros(n)

    selection = unq_user_id[np.unique(user_idx[hs_first])]

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


def fit_user(u, model, ignore_times=False):

    if ignore_times:
        u.times[:] = None

    # Plot the success curves
    success = u.questions == u.replies
    plot.success.curve(successes=success,
                       fig_name=f'u_{u.user_id}_success_curve.pdf')
    # plot.success.multi_curve(questions=u.questions, replies=u.replies,
    #                          fig_name=f'u_{u.user_id}_success_curve_multi.pdf',
    #                          max_lines=40)

    u_q, counts = np.unique(
        u.questions, return_counts=True)

    plot.duolingo.bar(sorted(counts), fig_name=f'u_{u.user_id}_counts.pdf')

    f = fit.Fit(tk=u.tk, model=model, data=u, verbose=True,
                use_p_correct=True)

    return f.evaluate()


def info_user(u):

    print('user_id:', u.user_id)
    print('n iterations:', u.t_max)
    print('n items:', u.n_item)
    print('language ui:', u.ui_language)
    print('language learned:', u.learning_language)
    print('Total period:',
          datetime.timedelta(seconds=int(u.times[-1]-u.times[0])))

    # u_q, counts = np.unique(
    #     u.questions, return_counts=True)
    #
    # plot.duolingo.bar(sorted(counts), fig_name=f'u{user_id}_counts.pdf')
    # plot.duolingo.scatter(u.questions, u.times,
    #                       fig_name=f'u{user_id}_time_presentation.pdf',
    #                       size=0.1)


def main(model=ActR):

    # it, = np.asarray(Item.objects.all().values_list('history_seen')).T
    # print(min(it))
    # print(max(it))

    selection = subjects_selection_history()
    for user_id in selection:

        u = UserData.load(user_id, force=True, verbose=False)
        if u.learning_language is not None and u.t_max > 1000:
            info_user(u)
            min_u = min(u.times)
            max_u = max(u.times)
            sec = max_u - min_u

            time_norm = 100 / sec

            raw_times = np.zeros(u.t_max)
            raw_times[:] = u.times
            u.times[:] = u.times * time_norm

            fit_r = fit_user(u, model=model)

            agent = model(
                tk=Task(t_max=u.t_max, n_item=u.n_item),
                # param={'d': 0.5, 'tau': 0.01, 's': 0.3, 'r': 0.001})
                param=fit_r["best_param"])

            for i in range(u.t_max):

                item, t = u.questions[i], u.times[i]
                agent.learn(item=item, time=t)

            time_sampling = \
                np.linspace(start=min_u,
                            stop=max_u, num=100)

            p_recall = p_recall_over_time_after_learning(
                agent=agent,
                t_max=u.t_max,
                n_item=u.n_item,
                time_norm=time_norm,
                time_sampling=time_sampling)

            plot.memory_trace.plot(
                p_recall_value=p_recall,
                p_recall_time=time_sampling,
                success_time=raw_times,
                success_value=u.successes,
                questions=u.questions,
                fig_name=f'u_{u.user_id}/memory_trace.pdf'
            )

    # selection = subjects_selection_history()
    # for user_id in selection:
    #     u = UserData.load(user_id=user_id, force=True)
    #     if u.t_max > 100:
    #         info_user(u)

    # info_user('u:IY_')
    # general_statistics()

    # subjects_selection_history(force=True)

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
