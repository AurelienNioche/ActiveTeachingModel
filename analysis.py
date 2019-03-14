import os
# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# Your application specific imports
from task.models import User

import numpy as np

import behavior.data
import fit.act_r
import fit.rl
import fit.generic

import graph.model_comparison
import graph.success


N_MODELS = 2


def model_comparison():

    users = User.objects.all().order_by('id')

    data = {
        "bic": [[] for _ in range(N_MODELS)],
        "p_choices": [[] for _ in range(N_MODELS)]
    }

    for u in users:

        # Get user id
        user_id = u.id
        print(user_id)
        print("*" * 5)

        # Get questions, replies, possible_replies, and number of different items
        questions, replies, n_items, possible_replies = behavior.data.get(user_id)

        # ---------- #
        # RL fitting
        # ---------- #

        # Get log-likelihood sum
        lls, best_param, bic_v = fit.rl.fit(questions=questions, replies=replies,
                                            n_items=n_items, possible_replies=possible_replies)
        print(f'[RL] Alpha: {best_param["alpha"]:.3f}, Tau: {best_param["tau"]:.3f}, LLS: {lls:.2f}, BIC: {bic_v:.2f}')

        data["bic"][0].append(bic_v)

        # Get probabilities with best param
        p_choices = fit.rl.get_p_choices(parameters=best_param, questions=questions, replies=replies,
                                         possible_replies=possible_replies, n_items=n_items)

        data["p_choices"][0].append(np.mean(p_choices))
        # data["p_choices"][0] += list(p_choices)

        # ---------- #
        # ACT-R fitting
        # ---------- #

        # Get log-likelihood sum
        lls, best_param, bic_v = fit.act_r.fit(questions=questions, replies=replies, n_items=n_items)
        print(f'[ACT-R] d: {best_param["d"]:.3f}, Tau: {best_param["tau"]:.3f}, s: {best_param["s"]:.3f}, '
              f'LLS: {lls:.2f}, BIC: {bic_v:.2f}')

        data["bic"][1].append(bic_v)

        # Get probabilities with best param
        p_choices = fit.act_r.get_p_choices(parameters=best_param, questions=questions, replies=replies,
                                            n_items=n_items)

        # data["p_choices"][1] += list(p_choices)
        data["p_choices"][1].append(np.mean(p_choices))

        print()

    graph.model_comparison.scatter_plot(data_list=data["bic"], colors=["C0", "C1"], x_tick_labels=["RL", "ACT-R - -"],
                                        f_name='model_comparison.pdf', y_label='BIC', invert_y_axis=True)

    graph.model_comparison.scatter_plot(data_list=data["p_choices"], colors=["C0", "C1"],
                                        x_tick_labels=["RL", "ACT-R - -"],
                                        f_name='model_probabilities.pdf', y_label='p')


def main():

    model_comparison()


if __name__ == "__main__":

    main()
