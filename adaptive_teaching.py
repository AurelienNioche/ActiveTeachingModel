import numpy as np

from collections.abc import Sequence
import os

from adaptive_design.constants import P_RECALL, POST_MEAN, POST_SD, HIST, \
    FORGETTING_RATES
from adaptive_design.engine.teacher_half_life import TeacherHalfLife, \
    RANDOM, OPT_TEACH, OPT_INF0, ADAPTIVE, LEITNER
from adaptive_design.plot import fig_parameter_recovery, \
    fig_p_recall, fig_p_recall_item, fig_n_seen
from adaptive_design.teach import run

from utils.backup import dump, load

from learner.half_life import FastHalfLife

from utils.string import dic2string

FIG_FOLDER = os.path.join("fig", "adaptive")


def main():

    force = False, False, False, False  # True, True, True, True

    design_types = [
        LEITNER, RANDOM, OPT_INF0, ADAPTIVE]

    engine_model = TeacherHalfLife

    grid_size = 20
    n_item = 200
    n_trial = 500

    seed = 123

    learner_model = FastHalfLife

    learner_param = {
        "alpha": 0.05,
        "beta": 0.2
        # "beta": 0.10,
        # "alpha": 0.5
    }

    results = {}

    # Run simulations for every design
    for i, dt in enumerate(design_types):

        bkp_file = os.path.join(
            "bkp", "adaptive",
            f"{dt}_"
            f"{learner_model.__name__}_" 
            f"{dic2string(learner_param)}_"
            f"confidence_thr_{engine_model.confidence_threshold}_"
            f"gamma_{engine_model.gamma}_"
            f"n_trial_{n_trial}_"
            f"n_item_{n_item}"
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
                design_type=dt,
                learner_model=learner_model,
                learner_param=learner_param,
                engine_model=engine_model,
                n_item=n_item,
                n_trial=n_trial,
                grid_size=grid_size,
                seed=seed,
            )

            dump(r, bkp_file)

        results[dt] = r

    post_means = {
        d: results[d][POST_MEAN] for d in design_types
    }

    post_sds = {
        d: results[d][POST_SD] for d in design_types
    }

    p_recall = {
        d: results[d][P_RECALL] for d in design_types
    }

    # strength = {
    #     d: 1/results[d][FORGETTING_RATES] for d in design_types
    # }

    forgetting_rates = {
        d: {k: np.zeros(n_trial) for k in ('mean', 'sd')}
        for d in design_types
    }

    for d in design_types:
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
        d: np.zeros(n_trial) for d in design_types
    }

    for d in design_types:

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
        f"_{learner_model.__name__}_" \
        f"{dic2string(learner_param)}_" \
        f"confidence_thr_{engine_model.confidence_threshold}_" \
        f"gamma_{engine_model.gamma}_" \
        f"n_trial_{n_trial}_" \
        f"n_item_{n_item}" \
        f".pdf"

    fig_name = f"param_recovery" + fig_ext
    fig_parameter_recovery(param=param, design_types=design_types,
                           post_means=post_means, post_sds=post_sds,
                           true_param=learner_param, num_trial=n_trial,
                           fig_name=fig_name,
                           fig_folder=FIG_FOLDER)

    fig_name = f"p_recall" + fig_ext
    fig_p_recall(data=p_recall, design_types=design_types,
                 fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"p_recall_item" + fig_ext
    fig_p_recall_item(
        p_recall=p_recall, design_types=design_types,
        fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"forgetting_rates" + fig_ext
    fig_p_recall(
        y_label="Forgetting rates",
        data=forgetting_rates, design_types=design_types,
        fig_name=fig_name, fig_folder=FIG_FOLDER)

    # fig_name = f"forgetting_rates_weighted" + fig_ext
    # fig_p_recall(
    #     y_label="Forgetting rates weighted",
    #     data=forgetting_rates_weighted, design_types=design_types,
    #     fig_name=fig_name, fig_folder=FIG_FOLDER)

    fig_name = f"n_seen" + fig_ext
    fig_n_seen(
        data=n_seen, design_types=design_types,
        fig_name=fig_name, fig_folder=FIG_FOLDER)


if __name__ == "__main__":
    main()
