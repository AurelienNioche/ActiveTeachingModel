import numpy as np

from collections.abc import Sequence
import os

from adaptive_teaching.constants import \
    POST_MEAN, POST_SD, HIST_ITEM, \
    P, P_SEEN, FR_SEEN, N_SEEN

from adaptive_teaching.engine.engine import Engine

from adaptive_teaching.plot import fig_parameter_recovery, \
    fig_p_recall, fig_p_recall_item, fig_n_seen

from adaptive_teaching.teacher.leitner import Leitner
from adaptive_teaching.teacher.random import RandomTeacher
from adaptive_teaching.teacher.memorize import Memorize
from adaptive_teaching.teacher.adaptive import Adaptive
from adaptive_teaching.teacher.generic import GenericTeacher

from adaptive_teaching.learner.half_life import HalfLife
from adaptive_teaching.learner.half_life_asymmetric import HalfLifeAsymmetric

from adaptive_teaching.run import run

from utils.backup import dump, load


from utils.string import dic2string

FIG_FOLDER = os.path.join("fig", "adaptive")


def main():

    task_param = {
        "n_trial": 500,
        "n_item": 200
    }

    # learner_model = HalfLife
    # learner_param = {
    #     "alpha": 0.05,
    #     "beta": 0.2
    #     # "beta": 0.10,
    #     # "alpha": 0.5
    # }

    learner_model = HalfLifeAsymmetric
    learner_param = {
        "alpha": 0.05,
        "beta": -0.2,
        "gamma": 0.2,
    }

    engine_model = Engine
    engine_param = {
        "grid_size": 20,
        # "true_param": (0.05, -0.2, 0.2)
    }

    force = False, False, True

    seed = 123

    labels_teacher_models_and_params = [
        ("Random", RandomTeacher, {}),
        ("Leitner", Leitner, {}),
        ("Opt. Info", GenericTeacher, {"confidence_threshold": -9999}),
        # ("Adapt. Memorize", Memorize, {"confidence_threshold": 0.1,
        #                                "learner_model": learner_model}),
         ("Adapt. Adaptive", Adaptive,  {"confidence_threshold": 0.1,
                                         "learner_model": learner_model,
                                         "alpha": 0.999})
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

    data_type = (POST_MEAN, POST_SD, P, P_SEEN, FR_SEEN, N_SEEN)
    data = {dt: {} for dt in data_type}

    for lb in labels:
        for dt in data_type:
            data[dt][lb] = results[lb][dt]

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
                           post_means=data[POST_MEAN], post_sds=data[POST_SD],
                           true_param=learner_param,
                           num_trial=task_param['n_trial'],
                           fig_name=fig_name,
                           fig_folder=FIG_FOLDER)

    fig_name = f"p_seen" + fig_ext
    fig_p_recall(data=data[P_SEEN], labels=labels,
                 fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"p_item" + fig_ext
    fig_p_recall_item(
        p_recall=data[P], labels=labels,
        fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"fr_seen" + fig_ext
    fig_p_recall(
        y_label="Forgetting rates",
        data=data[FR_SEEN], labels=labels,
        fig_name=fig_name, fig_folder=FIG_FOLDER)

    # fig_name = f"forgetting_rates_weighted" + fig_ext
    # fig_p_recall(
    #     y_label="Forgetting rates weighted",
    #     data=forgetting_rates_weighted, design_types=design_types,
    #     fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"n_seen" + fig_ext
    fig_n_seen(
        data=data[N_SEEN], design_types=labels,
        fig_name=fig_name, fig_folder=FIG_FOLDER)


if __name__ == "__main__":
    main()
