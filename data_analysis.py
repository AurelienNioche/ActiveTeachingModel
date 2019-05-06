import os
# Django specific settings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# Your application specific imports
from task.models import User

# import logging

from fit import fit

import behavior.data


import plot.model_comparison
import plot.success

import similarity_graphic.measure
import similarity_semantic.measure

LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)

import sys


class Tee(object):
    def __init__(self, f):

        self.files = (sys.stdout, open(f, 'w'))

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()

    def __del__(self):
        self.files[-1].close()


def fit_method_comparision(use_p_correct=True, methods=('de', 'tpe')):

    models = "rl", "act_r", "act_r_meaning", "act_r_graphic"
    models = models

    users = User.objects.all().order_by('id')

    for u in users:

        print(f'{u.id}\n{"*" * 5}\n')

        # Get questions, replies, possible_replies, and number of different items
        questions, replies, n_items, possible_replies, success = behavior.data.get(user_id=u.id, verbose=True)

        # Get task parameters for ACT-R +
        question_entries, kanjis, meanings = behavior.data.task_features(user_id=u.id)
        c_graphic = similarity_graphic.measure.get(kanjis)
        c_semantic = similarity_semantic.measure.get(meanings)

        for i, m in enumerate(models):

            for method in methods:

                f = fit.Fit(questions=questions, replies=replies, possible_replies=possible_replies, n_items=n_items,
                            c_graphic=c_graphic, c_semantic=c_semantic, use_p_correct=use_p_correct, method=method)
                getattr(f, m)()

        print('-' * 10)


def model_comparison(use_p_correct=True, fit_method='de', sim_graphic_method='sim_search',
                     models=("rl", "act_r", "act_r_meaning", "act_r_graphic")):

    # logging.basicConfig(
    #     filename=f'fit_{fit_method}_{sim_graphic_method}_{"correct" if use_p_correct else "choice"}.log',
    #     level=logging.DEBUG)
    # # noinspection PyShadowingBuiltins
    # print = logging.info

    sys.stdout = Tee(f'{LOG_DIR}/fit_{fit_method}_{sim_graphic_method}_{"correct" if use_p_correct else "choice"}.log')

    models = models[::-1]
    n_models = len(models)

    users = User.objects.all().order_by('id')

    data = {
        "bic": [[] for _ in range(n_models)],
        "mean_p": [[] for _ in range(n_models)]
    }

    print(f"Fit parameters: use_p_correct={use_p_correct}; method='{fit_method}'\n")
    print(f"Graphic similarity: method={sim_graphic_method}'\n")

    for u in users:

        print(f'{u.id}\n{"*" * 5}\n')

        # Get questions, replies, possible_replies, and number of different items
        questions, replies, n_items, possible_replies, success = behavior.data.get(user_id=u.id, verbose=True)

        # Plot the success curves
        plot.success.curve(successes=success, fig_name=f'success_curve_u{u.id}.pdf')
        plot.success.scatter(successes=success, fig_name=f'success_scatter_u{u.id}.pdf')

        # Get kanji's connections
        question_entries, kanjis, meanings = behavior.data.task_features(user_id=u.id)
        c_graphic = similarity_graphic.measure.get(kanjis, method=sim_graphic_method)
        c_semantic = similarity_semantic.measure.get(meanings)

        # Get the fit
        f = fit.Fit(questions=questions, replies=replies, possible_replies=possible_replies, n_items=n_items,
                    c_graphic=c_graphic, c_semantic=c_semantic, use_p_correct=use_p_correct, method=fit_method)

        # Keep trace of bic and mean_p
        for i, m in enumerate(models):
            best_param, mean_p, lls, bic = getattr(f, m)()
            data["bic"][i].append(bic)
            data["mean_p"][i].append(mean_p)

    # Prepare for plot
    colors = [f"C{i}" for i in range(n_models)]
    dsp_use_p_correct = "p_correct" if use_p_correct else "p_choice"

    # Plot BICs
    plot.model_comparison.scatter_plot(data_list=data["bic"], colors=colors,
                                       x_tick_labels=[i.replace('_', ' ') for i in models],
                                       f_name=f'model_comparison_{fit_method}_{dsp_use_p_correct}.pdf',
                                       y_label='BIC', invert_y_axis=True)

    # Plot average means
    plot.model_comparison.scatter_plot(data_list=data["mean_p"], colors=colors,
                                       x_tick_labels=[i.replace('_', ' ') for i in models],
                                       f_name=f'model_probabilities_{fit_method}_{dsp_use_p_correct}.pdf',
                                       y_label='p')

    print('-' * 10)


def main():

    for sim_graphic_method in ("auto_encoder", "sim_search"):
        model_comparison(models=("rl", "act_r", "act_r_meaning", "act_r_graphic"),
                         sim_graphic_method=sim_graphic_method)


if __name__ == "__main__":

    main()
