#%%
"""
    Parameters
    ----------

    num_agents : int
        Number of agents to generate a config file.
    param_labels : Iterable[str]
        Parameter names.
    is_item_specific : bool
        P recall is item specific.
    n_item : int
        Number of items.
    ss_n_iter : int
        Number of iterations in a session.
    ss_n_iter_between : int
        Number of iterations between sessions (iteration used as unit of time)
    n_ss : int
        Number of sessions.
    thr : float
        P recall threshold to consider an item learnt.
    bounds : Iterable[Iterable[float]]
        Min and max values for each parameter.
    init_guess : None
        Initial guess for psychologist.
    learner_model : str
        Learner model name.
    psychologist_model : str
        Psychologist model name.
    teachers : Iterable[str]
        Teacher names.
    omniscient : Iterable[bool]
        P recall known by the teacher/psychologist.
    time_per_iter : int
        Time per iteration.
    grid_size : int
        How fine is the parameter grid.
    seed : int
        Random number generation seed.
"""

import os
import json

import numpy as np
import pandas as pd

from typing import Iterable
from numpy.random import default_rng

import settings.paths as paths
import task_param.task_param_carlos as task_param


def generate_random_param(
    parameter_name: str, num_agents: int, bounds: Iterable, rng: object,
) -> pd.Series:
    """Get randomized parameters for all subjects within bounds"""

    assert len(bounds) == 2
    assert bounds[-1] >= bounds[0]

    return pd.Series(
        (bounds[-1] - bounds[0]) * rng.random((num_agents,)) + bounds[0],
        name=parameter_name,
    )


def get_param_df(
    num_agents: int, parameter_names: Iterable, bounds: Iterable[Iterable[float]]
) -> pd.DataFrame:
    """Get all parameter dataframe for all agents"""

    return pd.concat(
        tuple(
            generate_random_param(parameter_name, num_agents, bounds[idx], rng)
            for idx, parameter_name in enumerate(parameter_names)
        ),
        axis=1,
    )


def make_config_json(
    num_agents: int, save_path: str, **kwargs,
):
    """Make a JSON config file"""

    params_df = get_param_df(num_agents, kwargs["param_labels"], kwargs["bounds"])

    for n_agent in range(num_agents):
        param = {"param": tuple(params_df.iloc[n_agent].values)}
        seed = {"seed": n_agent}
        json_content = {**param, **kwargs, **seed}

        with open(
            os.path.join(
                save_path,
                f"{kwargs['learner_model']}-{kwargs['teachers'][0]}-{kwargs['psychologist_model']}-{n_agent}.json",
            ),
            "w",
        ) as file_path:
            json.dump(json_content, file_path, sort_keys=False, indent=4)


def main() -> None:
    """Set the parameters and generate the JSON config files"""

    num_agents = 100

    param_labels = ["tau", "s", "b", "m", "c", "x"]
    bounds = [
        [0.8, 1.2],
        [0.03, 0.05],
        [0.005, 0.20],
        [0.005, 0.20],
        [0.1, 0.1],
        [0.6, 0.6],
    ]

    mcts = {"iter_limit": 500, "time_limit": None, "horizon": 50}

    leitner = {"delay_factor": 2, "delay_min": 2}

    is_item_specific = False
    n_item = 100
    ss_n_iter = 100
    ss_n_iter_between = task_param.N_SEC_PER_DAY - ss_n_iter * task_param.N_SEC_PER_ITER
    n_ss = 10
    thr = 0.9
    init_guess = None

    # # # Options
    # 'exp_decay': ExponentialNDelta,
    # 'act_r2005': ActR2005,
    # 'act_r2008': ActR2008,
    # 'walsh2018': Walsh2018
    # learner_model = ["walsh2018"]  # Default
    learner_models = ["walsh2018"]

    # # # Options
    # 'psy_grid': PsychologistGrid,
    # 'psy_gradient': PsychologistGradient
    psychologist_model = "psy_grid"  # Default

    # # # Options
    # 'threshold': Threshold,
    # 'leitner': Leitner,
    # 'mcts': MCTSTeacher,
    # 'sampling': Sampling
    # ! Must be the list of three
    # teachers = ["leitner", "threshold", "sampling"]  # Default
    teachers = ["sampling"]

    # # # Options
    # omniscient = [False, True, True]  # Default
    omniscient = [False]
    time_per_iter = 2
    grid_size = 10

    for learner_model in learner_models:
        make_config_json(
            num_agents=num_agents,
            save_path=paths.AUTO_JSON_DIR,
            # kwargs
            param_labels=param_labels,
            is_item_specific=is_item_specific,
            n_item=n_item,
            ss_n_iter=ss_n_iter,
            ss_n_iter_between=ss_n_iter_between,
            n_ss=n_ss,
            thr=thr,
            bounds=bounds,
            init_guess=init_guess,
            learner_model=learner_model,
            psychologist_model=psychologist_model,
            teachers=teachers,
            omniscient=omniscient,
            time_per_iter=time_per_iter,
            grid_size=grid_size,
            leitner=leitner,
            mcts=mcts,
        )


if __name__ == "__main__":
    # Change parameters inside `main`
    rng = default_rng(123)
    main()

#%%

# param_labels: Iterable[str],
# is_item_specific: bool,
# n_item: int,
# ss_n_iter: int,
# ss_n_iter_between: int,
# n_ss: int,
# thr: float,
# bounds: Iterable[Iterable[float]],
# init_guess: None,
# learner_model: str,
# psychologist_model: str,
# teachers: Iterable[str],
# omniscient: Iterable[bool],
# time_per_iter: int,
# grid_size: int,
# seed: int,
