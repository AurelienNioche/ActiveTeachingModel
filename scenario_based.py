from adaptive_teaching.constants import POST_MEAN, POST_SD, P, P_SEEN, FR_SEEN, \
    N_SEEN
from adaptive_teaching.plot import fig_parameter_recovery, fig_p_recall_item, \
    fig_p_recall, fig_n_seen
from adaptive_teaching.simplified.learner import \
    ExponentialForgettingAsymmetric
from adaptive_teaching.simplified.scenario import run_n_days
from main import TEACHER, LEITNER, FIG_FOLDER


def scenario_based():

    seed = 1
    n_item = 1000

    grid_size = 20

    condition_labels = \
        TEACHER, LEITNER   # , ADAPTIVE

    learner = ExponentialForgettingAsymmetric

    # Select the parameter to use
    param = 0.02, 0.22, 0.44,
    # param = 0.01, 0.01, 0.06,

    n_day = 10

    results = {}
    for cd in condition_labels:
        results[cd] = run_n_days(
            condition=cd,
            n_item=n_item,
            n_day=n_day,
            grid_size=grid_size,
            param=param,
            seed=seed,
            learner=learner,
        )

    data_type = (POST_MEAN, POST_SD, P, P_SEEN, FR_SEEN, N_SEEN)
    data = {dt: {} for dt in data_type}

    for cd in condition_labels:
        for dt in data_type:
            d = results[cd][dt]
            data[dt][cd] = d

    str_param = learner.__name__ + "_" + "_".join([f'{p:.2f}' for p in param])

    fig_parameter_recovery(param=learner.param_labels,
                           condition_labels=condition_labels,
                           post_means=data[POST_MEAN],
                           post_sds=data[POST_SD],
                           true_param={learner.param_labels[i]: param[i]
                                       for i in range(len(param))},
                           fig_name=f'{str_param}_rc.pdf',
                           fig_folder=FIG_FOLDER)

    fig_p_recall_item(
        p_recall=data[P], condition_labels=condition_labels,
        fig_name=f'{str_param}_pi.pdf', fig_folder=FIG_FOLDER)

    fig_p_recall(data=data[P_SEEN], condition_labels=condition_labels,
                 fig_name=f'{str_param}_p_.pdf', fig_folder=FIG_FOLDER)

    fig_p_recall(
        y_label="Forgetting rates",
        data=data[FR_SEEN], condition_labels=condition_labels,
        fig_name=f'{str_param}_fr.pdf', fig_folder=FIG_FOLDER)

    fig_n_seen(
        data=data[N_SEEN], condition_labels=condition_labels,
        fig_name=f'{str_param}_n_.pdf', fig_folder=FIG_FOLDER)