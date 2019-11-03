import os
# Django specific settings
import utils.console_output

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ActiveTeachingModel.settings")
# Ensure settings are read
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# Your application specific imports
from task.models import User

# import logging

from fit import fit

import behavior.aalto

import pickle

import numpy as np

import plot.model_comparison
import plot.success

from learner.act_r import ActR

LOG_DIR = "log"
BKP_FOLDER = "bkp/model_comparison"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BKP_FOLDER, exist_ok=True)

import sys

from utils import utils


def fit_user(user_id=7, model=ActR, verbose=True,
             normalize_similarity=True, **kwargs):
    # data
    data = behavior.aalto.UserData(
        user_id=user_id,
        normalize_similarity=normalize_similarity, verbose=verbose)
    # Get the fit
    f = fit.Fit(tk=data.tk, model=model, data=data,
                verbose=verbose, **kwargs)

    fit_r = f.evaluate()
    return fit_r


def _get_model_comparison_data(models, normalize_similarity=False,
                               verbose=False, **kwargs):

    sys.stdout = utils.console_output.Tee(f'{LOG_DIR}/fit.log')

    models = models[::-1]
    model_labels = [m.__name__ for m in models]

    users = User.objects.all().order_by('id')
    n_users = len(users)

    measures = "bic", "mean_p"

    data = {
        k: {ml: np.zeros(n_users) for ml in model_labels} for k in measures
    }

    # print(f"Fit parameters: {fit_param}'\n")
    # print(f"Graphic similarity: method={sim_graphic_method}'\n")

    for i, u in enumerate(users):

        print(f'{u.id}\n{"*" * 5}\n')

        # Get user data
        u_data = behavior.aalto.UserData(
            user_id=u.id,
            normalize_similarity=normalize_similarity, verbose=verbose)

        # Plot the success curves
        plot.success.curve(successes=u_data.success,
                           fig_name=f'success_curve_u{u.id}.pdf')
        plot.success.scatter(successes=u_data.success,
                             fig_name=f'success_scatter_u{u.id}.pdf')

        # Get task features
        tk = u_data.tk

        for m in models:

            # Get the fit
            f = fit.Fit(tk=tk, model=m, data=u_data,
                        verbose=verbose, **kwargs)

            fit_r = f.evaluate()

            # Keep trace of bic and mean_p
            for ms in measures:
                data[ms][m.__name__][i] = fit_r[ms]

    return data


def model_comparison(models, normalize_similarity=False,
                     verbose=False, **kwargs):

    file_path = f"{BKP_FOLDER}/" \
        f"model_comparison_" \
        f"_norm_{normalize_similarity}.p"
    if not os.path.exists(file_path):
        data = _get_model_comparison_data(
            models=models,
            normalize_similarity=normalize_similarity, verbose=verbose,
            **kwargs
        )
        pickle.dump(data, open(file_path, 'wb'))
    else:
        data = pickle.load(open(file_path, 'rb'))

    # Prepare for plot
    colors = [f"C{i}" for i in range(len(models))]

    # Plot BICs
    plot.model_comparison.scatter_plot(
        data["bic"], colors=colors,
        f_name=f'model_comparison.pdf',
        y_label='BIC', invert_y_axis=True)

    # Plot average means
    plot.model_comparison.scatter_plot(
        data["mean_p"], colors=colors,
        f_name=f'model_probabilities.pdf',
        y_label='p')

    print('-' * 10)


def main():

    # model_comparison(models=
    # (QLearner, ActR, ActRMeaning, ActRGraphic, ActRPlus),
    # use_p_correct=False)
    # model_comparison(models=
    # (QLearner, ActR, ActRMeaning, ActRGraphic, ActRPlus),
    # fit_param={'use_p_correct': True},
    #                  normalize_similarity=True, verbose=True)
    fit_user()


if __name__ == "__main__":

    main()
