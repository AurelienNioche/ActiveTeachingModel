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

import pickle


import plot.model_comparison
import plot.success

from learner.act_r import ActR
from learner.rl import QLearner
from learner.act_r_custom import ActRMeaning, ActRGraphic, ActRPlus

LOG_DIR = "log"
BKP_FOLDER = "bkp/model_comparison"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BKP_FOLDER, exist_ok=True)

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


def _get_model_comparison_data(models, use_p_correct=True, fit_method='de'):

    sys.stdout = Tee(f'{LOG_DIR}/fit_{fit_method}_{"correct" if use_p_correct else "choice"}.log')

    models = models[::-1]
    model_labels = [m.__name__ for m in models]

    users = User.objects.all().order_by('id')

    measures = "bic", "mean_p"

    data = {
        k: {ml: [] for ml in model_labels} for k in measures
    }

    print(f"Fit parameters: use_p_correct={use_p_correct}; method='{fit_method}'\n")
    # print(f"Graphic similarity: method={sim_graphic_method}'\n")

    for u in users:

        print(f'{u.id}\n{"*" * 5}\n')

        # Get user data
        u_data = behavior.data.UserData(user_id=u.id, verbose=True)

        # Plot the success curves
        plot.success.curve(successes=u_data.success, fig_name=f'success_curve_u{u.id}.pdf')
        plot.success.scatter(successes=u_data.success, fig_name=f'success_scatter_u{u.id}.pdf')

        # Get task features
        tk = u_data.tk

        for m in models:

            # Get the fit
            f = fit.Fit(tk=tk, model=m, data=u_data, method='de',
                        fit_param={'use_p_correct': use_p_correct}, verbose=True)

            fit_r = f.evaluate()

            # Keep trace of bic and mean_p
            for ms in measures:
                data[ms][m.__name__].append(fit_r[ms])

    return data


def model_comparison(models, use_p_correct=True, fit_method='de'):

    dsp_use_p_correct = "p_correct" if use_p_correct else "p_choice"

    file_path = f"{BKP_FOLDER}/model_comparison_{fit_method}_{dsp_use_p_correct}.p"
    if not os.path.exists(file_path):
        data = _get_model_comparison_data(models=models, use_p_correct=use_p_correct, fit_method=fit_method)
        pickle.dump(data, open(file_path, 'wb'))
    else:
        data = pickle.load(open(file_path, 'rb'))

    # Prepare for plot
    colors = [f"C{i}" for i in range(len(models))]

    # Plot BICs
    plot.model_comparison.scatter_plot(data["bic"], colors=colors,
                                       f_name=f'model_comparison_{fit_method}_{dsp_use_p_correct}.pdf',
                                       y_label='BIC', invert_y_axis=True)

    # Plot average means
    plot.model_comparison.scatter_plot(data["mean_p"], colors=colors,
                                       f_name=f'model_probabilities_{fit_method}_{dsp_use_p_correct}.pdf',
                                       y_label='p')

    print('-' * 10)


def main():

    model_comparison(models=(QLearner, ActR, ActRMeaning, ActRGraphic, ActRPlus), use_p_correct=False)


if __name__ == "__main__":

    main()
