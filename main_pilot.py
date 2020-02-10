import os
import pickle
import numpy as np

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

import user_data.analysis.tools.history
import user_data.analysis.tools.users
import user_data.analysis.similarity.graphic.measure
import user_data.analysis.similarity.semantic.measure
import user_data.analysis.plot.human

from user_data.settings import N_POSSIBLE_REPLIES, BKP_FIT


def main():

    force = False

    # Get similarity
    print("Computing the graphic connection...")
    graphic_connection = \
        analysis.similarity.graphic.measure.get(kanji_list=None)

    semantic_connection = \
        analysis.similarity.semantic.measure.get(word_list=None)

    fit_class = DifferentialEvolution  # or PyGPO
    list_model_to_fit = \
        QLearner,  ActR, ActRMeaning, ActRGraphic, \
        ExponentialForgetting, ExponentialForgettingAsymmetric

    task_param = {
        'n_possible_replies': N_POSSIBLE_REPLIES,
        'n_item': len(semantic_connection),
        'semantic_connection': semantic_connection,
        'graphic_connection': graphic_connection}

    list_user_id = analysis.tools.users.get(force=False)

    # list_user_id = [list_user_id[-1], ]

    n_user = len(list_user_id)
    # n_item = len(kanji)

    for i, user_id in enumerate(list_user_id):

        print("-"*16)
        print(f"User {user_id} ({i}/{n_user})")
        print("-" * 16)
        print()
        print("Importing data...", end=' ', flush=False)

        hist_question, hist_success, seen = \
            analysis.tools.history.get(user_id=user_id)
        print("Done.\n")

        print(f"N iteration: {len(hist_question)}.")
        print(f"N kanji seen: {len(np.unique(hist_question))}.")
        print(f"Average success rate: {np.mean(hist_success)*100:.2f}%.")
        print()

        analysis.plot.human.plot(
            seen=seen,
            successes=hist_success,
            fig_folder=os.path.join("fig", "pilot"),
            fig_name=f'hist_success_u{user_id}.pdf'
        )

        bkp_file = os.path.join(
            BKP_FIT,
            f'fit_u{user_id}_degenerate.p')

        if not os.path.exists(bkp_file) or force:

            f = Degenerate()
            r = f.evaluate(
                hist_question=hist_question,
                hist_success=hist_success,
                task_param=task_param)
            pickle.dump(r, open(bkp_file, 'wb'))
        else:
            r = pickle.load(open(bkp_file, 'rb'))

        print('Degenerate model with only success:')
        print(f'Best value: {r["best_value"]:.2f}.\n')

        for model_to_fit in list_model_to_fit:

            bkp_file = os.path.join(
                BKP_FIT,
                f'fit_u{user_id}_{model_to_fit.__name__}.p')

            if not os.path.exists(bkp_file) or force:
                print(f"Running fit {model_to_fit.__name__}...",
                      end=' ', flush=True)
                f = fit_class(model=model_to_fit)

                r = f.evaluate(
                    hist_question=hist_question,
                    hist_success=hist_success,
                    task_param=task_param
                )

                print("Done.")

                pickle.dump(r, open(bkp_file, 'wb'))

            else:
                print("Loading from pilot_2019_09_02 file...", end=' ', flush=True)
                r = pickle.load(open(bkp_file, 'rb'))
                print("Done")

            bic_user = bic(lls=r["best_value"],
                           k=len(model_to_fit.bounds),
                           n=len(hist_question))

            print(f'Best param: {r["best_param"]}, '
                  f'best value: {r["best_value"]:.2f}.',
                  f'BIC: {bic_user}')
            print()


if __name__ == "__main__":

    main()
