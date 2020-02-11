import os
import pickle
import numpy as np
import multiprocessing as mp

# from user_data.models import User
# from teaching_material.selection import kanji

from user_data import analysis
from user_data.analysis.fit.scipy import DifferentialEvolution
# from analysis.fit.pygpgo import PyGPGO

from user_data.analysis.fit.learning_model.act_r.act_r import ActR
from user_data.analysis.fit.learning_model.act_r.custom import \
    ActRMeaning, ActRGraphic
from user_data.analysis.fit.learning_model.exponential_forgetting \
    import ExponentialForgetting, ExponentialForgettingAsymmetric
from user_data.analysis.fit.learning_model.rl import QLearner

# from teaching_material.selection import kanji, meaning

from user_data.analysis.fit.degenerate import Degenerate

from user_data.analysis.fit.comparison import bic

import user_data.analysis.tools.data
import user_data.analysis.similarity.graphic.measure
import user_data.analysis.similarity.semantic.measure
import user_data.analysis.plot.human

from user_data.settings import N_POSSIBLE_REPLIES, BKP_FIT


def _mp_fit_user_data(kwargs):
    return fit_user_data(**kwargs)


def fit_user_data(user, i,
                  list_model_to_fit,
                  kanji, meaning,
                  fit_class=DifferentialEvolution,
                  normalize_sem_con=True,
                  normalize_graph_con=True,
                  force_fit=False):

    print("-" * 16)
    print(f"User {i}")
    print("-" * 16)
    print()
    print("Importing data...", end=' ', flush=False)

    n_item = len(kanji)

    task_param = {
        'n_possible_replies': N_POSSIBLE_REPLIES,
        'n_item': n_item
    }

    hist = user["hist"]
    success = user["success"]
    n_seen = [len(np.unique(hist[:t])) for t in range(1, len(success) + 1)]
    print("Done.\n")

    print(f"N iteration: {len(hist)}.")
    print(f"N kanji seen: {len(np.unique(hist))}.")
    print(f"Average success rate: {np.mean(success) * 100:.2f}%.")
    print()

    # analysis.plot.human.plot(
    #     n_seen=n_seen,
    #     success=success,
    #     fig_folder=os.path.join("fig", "pilot"),
    #     fig_name=f'hist_success_u{i}.pdf'
    # )

    os.makedirs(BKP_FIT, exist_ok=True)

    bkp_file = os.path.join(
        BKP_FIT,
        f'fit_u{i}_degenerate.p')

    if not os.path.exists(bkp_file) or force_fit:

        f = Degenerate()
        r = f.evaluate(
            hist_question=hist,
            hist_success=success,
            task_param=task_param)
        pickle.dump(r, open(bkp_file, 'wb'))
    else:
        r = pickle.load(open(bkp_file, 'rb'))

    print('Degenerate model with only success:')
    print(f'Best value: {r["best_value"]:.2f}.\n')

    for model_to_fit in list_model_to_fit:

        ext_file = f"u{i}_{model_to_fit.__name__}_{fit_class.__name__}_"
        if model_to_fit == ActRMeaning:
            ext_file += f"normalize_sem_con_{normalize_sem_con}"
            if "semantic_connection" not in task_param:
                task_param['semantic_connection'] = \
                    analysis.similarity.semantic.measure.get(
                        word_list=meaning,
                        normalize_similarity=normalize_sem_con)

        elif model_to_fit == ActRGraphic:
            ext_file += f"normalize_graph_con_{normalize_graph_con}"
            if 'graphic_connection' not in task_param:
                task_param['graphic_connection'] = \
                    analysis.similarity.graphic.measure.get(
                        kanji_list=kanji,
                        normalize_similarity=normalize_graph_con)

        bkp_file = os.path.join(
            BKP_FIT,
            f'fit_{ext_file}.p')

        if not os.path.exists(bkp_file) or force_fit:
            print(f"Running fit {model_to_fit.__name__} for user {i}...",
                  end=' ', flush=True)
            f = fit_class(model=model_to_fit)

            r = f.evaluate(
                hist_question=hist,
                hist_success=success,
                task_param=task_param
            )

            print("Done.")

            pickle.dump(r, open(bkp_file, 'wb'))

        else:
            print("Loading from pickle file...", end=' ',
                  flush=True)
            r = pickle.load(open(bkp_file, 'rb'))
            print("Done")

        bic_user = bic(lls=r["best_value"],
                       k=len(model_to_fit.bounds),
                       n=len(hist))

        print(
            f'User {i} - {model_to_fit.__name__}:'
            f'Best param: {r["best_param"]}, '
            f'best value: {r["best_value"]:.2f}.'
            f'BIC: {bic_user}'
        )
        print()


def main():

    data = analysis.tools.data.get()

    list_model_to_fit = \
        QLearner,  ActR, \
        ExponentialForgetting, ExponentialForgettingAsymmetric, \
        ActRMeaning, \
        ActRGraphic, \

    kanji = data["kanji"]
    meaning = data["meaning"]

    kwargs_list = []

    for i, user in enumerate(data["user_data"]):

        kwargs_list.append(
            {"user": user,
             "i": i,
             "list_model_to_fit": list_model_to_fit,
             "kanji": kanji,
             "meaning": meaning
             })

    pool = mp.Pool()
    with pool as p:
        p.map(_mp_fit_user_data, kwargs_list)


if __name__ == "__main__":

    main()
