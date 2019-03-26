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

import similarity_graphic.measure
import similarity_semantic.measure


def model_comparison(use_p_correct=True, method='de'):

    models = "rl", "act_r", "act_r_meaning", "act_r_graphic"
    models = models[::-1]
    n_models = len(models)

    users = User.objects.all().order_by('id')

    data = {
        "bic": [[] for _ in range(n_models)],
        "mean_p": [[] for _ in range(n_models)]
    }

    print(f"Fit parameters: use_p_correct={use_p_correct}; method='{method}'\n")

    for u in users:

        print(f'{u.id}\n{"*" * 5}\n')

        # Get questions, replies, possible_replies, and number of different items
        questions, replies, n_items, possible_replies, success = behavior.data.get(user_id=u.id, verbose=True)

        # Get task parameters for ACT-R +
        question_entries, kanjis, meanings = behavior.data.task_features(user_id=u.id)
        c_graphic = similarity_graphic.measure.get(kanjis)
        c_semantic = similarity_semantic.measure.get(meanings)

        f = fit.Fit(questions=questions, replies=replies, possible_replies=possible_replies, n_items=n_items,
                    c_graphic=c_graphic, c_semantic=c_semantic, use_p_correct=use_p_correct, method=method)

        for i, m in enumerate(models):
            mean_p, bic = getattr(f, m)()
            data["bic"][i].append(bic)
            data["mean_p"][i].append(mean_p)

        plot.success.curve(successes=success, fig_name=f'success_curve_u{u.id}.pdf')
        plot.success.scatter(successes=success, fig_name=f'success_scatter_u{u.id}.pdf')

    colors = [f"C{i}" for i in range(n_models)]

    plot.model_comparison.scatter_plot(data_list=data["bic"], colors=colors,
                                       x_tick_labels=[i.replace('_', ' ') for i in models],
                                       f_name='model_comparison.pdf', y_label='BIC', invert_y_axis=True)

    plot.model_comparison.scatter_plot(data_list=data["mean_p"], colors=colors,
                                       x_tick_labels=[i.replace('_', ' ') for i in models],
                                       f_name='model_probabilities.pdf', y_label='p')

    print('-' * 10)


if __name__ == "__main__":

    methods = 'tpe', 'de'  # 'SLSQP', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',

    for me in methods:
        model_comparison(method=me)
