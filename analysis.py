import os
# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# Your application specific imports
from task.models import User

from fit import fit

import behavior.data


import plot.model_comparison
import plot.success

import graphic_similarity.measure
import semantic_similarity.measure

MODEL_NAMES = ["RL", "ACT-R -", "ACT-R +", "ACT-R ++"]
N_MODELS = len(MODEL_NAMES)


def model_comparison():

    users = User.objects.all().order_by('id')

    data = {
        "bic": [[] for _ in range(N_MODELS)],
        "mean_p": [[] for _ in range(N_MODELS)]
    }

    use_p_correct = False

    for u in users[::-1]:

        # Get user id
        user_id = u.id
        print(user_id)
        print("*" * 5)

        # Get questions, replies, possible_replies, and number of different items
        questions, replies, n_items, possible_replies, success = behavior.data.get(user_id, verbose=True)

        # Get task parameters for ACT-R +
        question_entries, kanjis, meanings = behavior.data.task_features(user_id=u.id)
        c_graphic = graphic_similarity.measure.get(kanjis)
        c_semantic = semantic_similarity.measure.get(meanings)

        f = fit.Fit(questions=questions, replies=replies, possible_replies=possible_replies, n_items=n_items,
                    c_graphic=c_graphic, c_semantic=c_semantic, use_p_correct=use_p_correct)

        rl_mean_p, rl_bic = f.rl()
        act_r_mean_p, act_r_bic = f.act_r()
        act_r_plus_mean_p, act_r_plus_bic = f.act_r_plus()
        act_r_pp_mean_pp, act_r_pp_bic = f.act_r_plus_plus()

        data["bic"][0].append(rl_bic)
        data["bic"][1].append(act_r_bic)
        data["bic"][2].append(act_r_plus_bic)
        data["bic"][3].append(act_r_pp_bic)

        data["mean_p"][0].append(rl_mean_p)
        data["mean_p"][1].append(act_r_mean_p)
        data["mean_p"][2].append(act_r_plus_mean_p)
        data["mean_p"][3].append(act_r_pp_mean_pp)

        print()

        plot.success.curve(successes=success, fig_name=f'success_curve_u{user_id}.pdf')
        plot.success.scatter(successes=success, fig_name=f'success_scatter_u{user_id}.pdf')

    colors = [f"C{i}" for i in range(N_MODELS)]

    plot.model_comparison.scatter_plot(data_list=data["bic"], colors=colors,
                                       x_tick_labels=MODEL_NAMES,
                                       f_name='model_comparison.pdf', y_label='BIC', invert_y_axis=True)

    plot.model_comparison.scatter_plot(data_list=data["mean_p"], colors=colors,
                                       x_tick_labels=MODEL_NAMES,
                                       f_name='model_probabilities.pdf', y_label='p')


if __name__ == "__main__":

    model_comparison()
