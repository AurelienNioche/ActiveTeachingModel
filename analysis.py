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
import fit.act_r_plus
import fit.act_r_pp

import plot.model_comparison
import plot.success

import graphic_similarity.measure
import semantic_similarity.measure

MODEL_NAMES = ["RL", "ACT-R -", "ACT-R +", "ACT-R ++"]
N_MODELS = len(MODEL_NAMES)


def model_stats(p_choices, questions, best_param):

    mean_p = np.mean(p_choices)

    lls = fit.generic.log_likelihood_sum(p_choices)

    n = len(questions)
    k = len(best_param)
    bic = fit.generic.bic(lls=lls, n=n, k=k)

    return mean_p, lls, bic


def fit_rl(questions, replies, n_items, possible_replies, use_p_correct):

    # Get log-likelihood sum
    best_param = fit.rl.fit(questions=questions, replies=replies,
                            n_items=n_items, possible_replies=possible_replies, use_p_correct=use_p_correct)

    # Get probabilities with best param
    p_choices = fit.rl.get_p_choices(parameters=best_param, questions=questions, replies=replies,
                                     possible_replies=possible_replies, n_items=n_items, use_p_correct=use_p_correct)

    mean_p, lls, bic = model_stats(p_choices=p_choices, questions=questions, best_param=best_param)

    print(f'[RL] Alpha: {best_param["alpha"]:.3f}, Tau: {best_param["tau"]:.3f}, LLS: {lls:.2f}, '
          f'BIC: {bic:.2f}, mean(P): {mean_p:.3f}')

    return mean_p, bic


def fit_act_r(questions, replies, n_items, use_p_correct):
    # Get log-likelihood sum
    best_param = fit.act_r.fit(questions=questions, replies=replies, n_items=n_items, use_p_correct=use_p_correct)

    # Get probabilities with best param
    p_choices = fit.act_r.get_p_choices(parameters=best_param, questions=questions, replies=replies,
                                        n_items=n_items, use_p_correct=use_p_correct)

    mean_p, lls, bic = model_stats(p_choices=p_choices, questions=questions, best_param=best_param)

    print(f'[ACT-R -] d: {best_param["d"]:.3f}, Tau: {best_param["tau"]:.3f}, s: {best_param["s"]:.3f}, '
          f'LLS: {lls:.2f}, BIC: {bic:.2f}, mean(P): {mean_p:.3f}')

    return mean_p, bic


def fit_act_r_plus(questions, replies, n_items, c_semantic, c_graphic, use_p_correct):

    # Get log-likelihood sum
    best_param = fit.act_r_plus.fit(
        questions=questions, replies=replies, n_items=n_items,
        c_semantic=c_semantic, c_graphic=c_graphic, use_p_correct=use_p_correct)

    # Get probabilities with best param
    p_choices = fit.act_r_plus.get_p_choices(
        parameters=best_param, questions=questions, replies=replies,
        c_graphic=c_graphic, c_semantic=c_semantic, n_items=n_items, use_p_correct=use_p_correct)

    mean_p, lls, bic = model_stats(p_choices=p_choices, questions=questions, best_param=best_param)

    print(f'[ACT-R +] '
          f'd: {best_param["d"]:.3f}, Tau: {best_param["tau"]:.3f}, s: {best_param["s"]:.3f}, '
          f'm: {best_param["m"]:.3f}, g: {best_param["g"]:.3f}, '
          f'LLS: {lls:.2f}, BIC: {bic:.2f}, mean(P): {mean_p:.3f}')

    return mean_p, bic


def fit_act_r_pp(questions, replies, n_items, c_semantic, c_graphic, use_p_correct):

    # Get log-likelihood sum
    best_param = fit.act_r_pp.fit(
        questions=questions, replies=replies, n_items=n_items,
        c_semantic=c_semantic, c_graphic=c_graphic, use_p_correct=use_p_correct)

    # Get probabilities with best param
    p_choices = fit.act_r_pp.get_p_choices(
        parameters=best_param, questions=questions, replies=replies,
        c_graphic=c_graphic, c_semantic=c_semantic, n_items=n_items, use_p_correct=use_p_correct)

    mean_p, lls, bic = model_stats(p_choices=p_choices, questions=questions, best_param=best_param)

    print(f'[ACT-R ++] '
          f'd: {best_param["d"]:.3f}, Tau: {best_param["tau"]:.3f}, s: {best_param["s"]:.3f}, '
          f'm: {best_param["m"]:.3f}, g: {best_param["g"]:.3f}, '
          f'g_mu: {best_param["g_mu"]:.3f}, g_sigma: {best_param["g_sigma"]:.3f}, '
          f'm_mu: {best_param["m_mu"]:.3f}, m_sigma: {best_param["m_sigma"]:.3f}, '
          f'LLS: {lls:.2f}, BIC: {bic:.2f}, mean(P): {mean_p:.3f}')

    return mean_p, bic


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

        act_r_pp_mean_pp, act_r_pp_bic = fit_act_r_pp(
            questions=questions, replies=replies, n_items=n_items, c_semantic=c_semantic, c_graphic=c_graphic,
            use_p_correct=use_p_correct
        )

        rl_mean_p, rl_bic = fit_rl(
            questions=questions, replies=replies, n_items=n_items, possible_replies=possible_replies,
            use_p_correct=use_p_correct)
        act_r_mean_p, act_r_bic = fit_act_r(questions=questions, replies=replies, n_items=n_items,
                                            use_p_correct=use_p_correct)
        act_r_plus_mean_p, act_r_plus_bic = fit_act_r_plus(
            questions=questions, replies=replies, n_items=n_items, c_semantic=c_semantic, c_graphic=c_graphic,
            use_p_correct=use_p_correct
        )

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
