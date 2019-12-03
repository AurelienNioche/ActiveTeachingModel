import numpy as np

from collections.abc import Sequence
import os

from adaptive_teaching.constants import P_RECALL, POST_MEAN, POST_SD, HIST, \
    FORGETTING_RATES

from adaptive_teaching.engine.engine import Engine

from adaptive_teaching.plot import fig_parameter_recovery, \
    fig_p_recall, fig_p_recall_item, fig_n_seen

from adaptive_teaching.teacher.leitner import Leitner
from adaptive_teaching.teacher.random import RandomTeacher
from adaptive_teaching.teacher.memorize import Memorize
from adaptive_teaching.teacher.threefold import Threefold
from adaptive_teaching.teacher.metaclass import GenericTeacher

from adaptive_teaching.learner.half_life import HalfLife

from adaptive_teaching.run import run

from utils.backup import dump, load


from utils.string import dic2string

FIG_FOLDER = os.path.join("fig", "adaptive")


def main():

    task_param = {
        "n_trial": 500,
        "n_item": 200
    }

    engine_model = Engine
    engine_param = {
        "grid_size": 20,
    }

    learner_model = HalfLife
    learner_param = {
        "alpha": 0.05,
        "beta": 0.2
        # "beta": 0.10,
        # "alpha": 0.5
    }

    force = False

    seed = 123

    labels_teacher_models_and_params = [
        ("Random", RandomTeacher, {}),
        ("Leitner", Leitner, {}),
        ("Opt. Info", GenericTeacher, {"confidence_threshold": -9999}),
        ("Adapt. Memorize", Memorize, {"confidence_threshold": 0.1,
                                       "learner_model": learner_model}),
         ("Adapt. Threefold", Threefold, {"confidence_threshold": 0.1,
                                        "learner_model": learner_model,
                                        }
          )
    ]

    labels = [i[0] for i in labels_teacher_models_and_params]

    results = {}

    # Run simulations for every design
    for i in range(len(labels_teacher_models_and_params)):

        label, teacher_model, teacher_param = \
            labels_teacher_models_and_params[i]

        bkp_file = os.path.join(
            "bkp", "adaptive",
            f"{dic2string(task_param)}_"
            f"{learner_model.__name__}_" 
            f"{dic2string(learner_param)}_"
            f"{teacher_model.__name__}_" 
            f"{Engine.__name__}_" 
            f"{dic2string(engine_param)}_"
            f".p")

        r = load(bkp_file)

        if isinstance(force, bool):
            f = force
        elif isinstance(force, Sequence):
            f = force[i]
        else:
            raise ValueError

        if not r or f:
            r = run(
                engine_model=engine_model,
                engine_param=engine_param,
                teacher_model=teacher_model,
                teacher_param=teacher_param,
                learner_model=learner_model,
                learner_param=learner_param,
                seed=seed,
                task_param=task_param)

            dump(r, bkp_file)

        results[label] = r

    post_means = {
        d: results[d][POST_MEAN] for d in labels
    }

    post_sds = {
        d: results[d][POST_SD] for d in labels
    }

    p_recall = {
        d: results[d][P_RECALL] for d in labels
    }

    # strength = {
    #     d: 1/results[d][FORGETTING_RATES] for d in design_types
    # }

    n_trial = task_param['n_trial']
    n_item = task_param['n_item']

    forgetting_rates = {
        d: {k: np.zeros(n_trial) for k in ('mean', 'sd')}
        for d in labels
    }

    for d in labels:
        t_max = results[d][FORGETTING_RATES].shape[1]
        for t in range(t_max):
            all_d = results[d][FORGETTING_RATES][:, t]
            data = all_d[all_d != np.inf]
            forgetting_rates[d]['mean'][t] = \
                np.mean(data)
            forgetting_rates[d]['sd'][t] = \
                np.std(data)

    # forgetting_rates_weighted = {
    #     d: {k: np.zeros(n_trial) for k in ('mean', 'sd')}
    #     for d in design_types
    # }

    n_seen = {
        d: np.zeros(n_trial) for d in labels
    }

    for d in labels:

        t_max = results[d][FORGETTING_RATES].shape[1]
        seen = np.zeros(n_item, dtype=bool)
        hist = results[d][HIST]

        for t in range(t_max):
            seen[int(hist[t])] = True
            n_seen[d][t] = np.sum(seen)

            # all_d = results[d][FORGETTING_RATES][:, t]
            # data = all_d[all_d != np.inf] / (np.sum(seen))
            #
            # forgetting_rates_weighted[d]['mean'][t] = \
            #     np.mean(data)
            #
            # forgetting_rates_weighted[d]['sd'][t] = \
            #     np.std(data)

    param = sorted(learner_model.bounds.keys())

    fig_ext = \
        "_" \
        f"{dic2string(task_param)}_" \
        f"{learner_model.__name__}_" \
        f"{dic2string(learner_param)}_" \
        f"{Engine.__name__}_" \
        f"{dic2string(engine_param)}_" \
        f".pdf"

    fig_name = f"param_recovery" + fig_ext
    fig_parameter_recovery(param=param, design_types=labels,
                           post_means=post_means, post_sds=post_sds,
                           true_param=learner_param, num_trial=n_trial,
                           fig_name=fig_name,
                           fig_folder=FIG_FOLDER)

    fig_name = f"p_recall" + fig_ext
    fig_p_recall(data=p_recall, design_types=labels,
                 fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"p_recall_item" + fig_ext
    fig_p_recall_item(
        p_recall=p_recall, design_types=labels,
        fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"forgetting_rates" + fig_ext
    fig_p_recall(
        y_label="Forgetting rates",
        data=forgetting_rates, design_types=labels,
        fig_name=fig_name, fig_folder=FIG_FOLDER)

    # fig_name = f"forgetting_rates_weighted" + fig_ext
    # fig_p_recall(
    #     y_label="Forgetting rates weighted",
    #     data=forgetting_rates_weighted, design_types=design_types,
    #     fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"n_seen" + fig_ext
    fig_n_seen(
        data=n_seen, design_types=labels,
        fig_name=fig_name, fig_folder=FIG_FOLDER)


if __name__ == "__main__":
    main()
