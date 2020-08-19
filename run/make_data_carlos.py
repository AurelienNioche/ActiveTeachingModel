#%%
import os
import sys

from itertools import product

import pandas as pd
import numpy as np

from numpy.random import default_rng
from tqdm import tqdm

import model.parameters
from task_param.task_param_carlos import TaskParam
from model.teacher.generic import Teacher


# COMMENTED FOR CELL USAGE!!!!!!!!!!!
# SCRIPT_NAME = os.path.basename(__file__).split(".")[0]
# PICKLE_FOLDER = os.path.join("pickle", SCRIPT_NAME)

# os.makedirs(PICKLE_FOLDER, exist_ok=True)


def make_data(tk: TaskParam, agent_id: int, human: bool) -> dict:
    """Runs task for every teacher"""

    num_iterations = tk.n_ss * tk.ss_n_iter * len(tk.teachers)
    range_iterations = range(num_iterations)

    item_history = {}
    param_recovery = {}

    agent_id = pd.Series((agent_id for _ in range_iterations), name=("Agent", "ID"))

    agent_human = pd.Series((human for _ in range_iterations), name=("Agent", "Human"))

    model_learner = pd.Series(
        (tk.learner_model.__name__.lower() for _ in range_iterations),
        name=("Model", "Learner"),
    )

    model_psychologist = pd.Series(
        (tk.psychologist_model.__name__.lower() for _ in range_iterations),
        name=("Model", "Psychologist"),
    )

    session_seed = pd.Series(
        (tk.seed for _ in range_iterations), name=("Session", "Seed"),
    )

    learner_name = tk.learner_model.__name__.lower()
    learner_param_names = model.parameters.make_param_learners_df()[
        learner_name
    ].dropna()

    # Bounds to df (originally array)
    bounds_to_df = {}
    for idx, param_name in enumerate(learner_param_names):
        bounds_to_df[param_name] = tuple(tk.bounds[idx] for _ in range_iterations)
    bounds = pd.DataFrame(
        bounds_to_df, columns=learner_param_names,
    )
    bounds.columns = pd.MultiIndex.from_product(({"Bounds"}, learner_param_names))


# !! For cell usage !!
config = os.path.join("config", f"config.json")  # Protyping

task_param = TaskParam.get(config)
data = make_data(tk=task_param, agent_id=23, human=False)
print(data)
#%%
    for teacher_class, omniscient in zip(tk.teachers, tk.omniscient):

        cond_label = teacher_class.__name__
        tqdm.write(f"Simulating '{cond_label}'...")
        teacher = teacher_class.create(tk=tk, omniscient=omniscient)
        r = run(teacher=teacher, tk=tk, omniscient=omniscient)

        if omniscient:
            cond_label += "-omniscient"

        if isinstance(r, tuple):
            item_history[cond_label], param_recovery[cond_label] = r
        else:
            item_history[cond_label] = r

    # !! this has to return the series
    return {"kwarg1": tk, "kwarg2": item_history}


def run(teacher: Teacher, tk: TaskParam, omniscient: bool):
    """Run for every teacher by make_data"""

    num_iterations = tk.n_ss * tk.ss_n_iter
    range_iterations = range(num_iterations)

    # teacher_name = teacher.__class__.__name__.lower()
    learner_name = tk.learner_model.__name__.lower()

    use_teacher_psy = hasattr(teacher, "psychologist") and not omniscient
    learner_param_names = None
    if use_teacher_psy:
        psychologist = teacher.psychologist
        # inferred_param = np.zeros((num_iterations, len(tk.param)))
        learner_param_names = model.parameters.make_param_learners_df()[
            learner_name
        ].dropna()
        inferred_param = pd.DataFrame(
            np.zeros((num_iterations, len(learner_param_names))),
            columns=learner_param_names,
        )

    else:
        # Psychologist needed to output a probability of recall
        psychologist = tk.psychologist_model.create(tk=tk, omniscient=True)

    learner_true_parameters = pd.Series(
        tk.param, index=learner_param_names, name="Parameter value",
    )

    # np.random.seed(tk.seed)
    rng = default_rng(tk.seed)
    # time_stamps = np.zeros(num_iterations, dtype=int)  # Unit is second
    # time_delta = pd.Series(
    #     (
    #         pd.Timedelta(n_iter * tk.time_per_iter, unit="s")
    #         for n_iter in range_iterations
    #     ),
    #     name="Time delta",
    # )
    session_time_delta = pd.Series(
        (0 for _ in range_iterations), name=("Session", "Time delta")
    )

    session_iteration = pd.Series(tk.ss_n_iter, name=("Session", "Iteration"))

    session_number = pd.Series(
        (0 for _ in range_iterations), name=("Session", "Number")
    )
    # item_history = np.zeros(num_iterations, dtype=int)
    item_history = pd.Series(
        (0 for _ in range_iterations), dtype=int, name=("Session", "Item ID"),
    )

    session_p_recall_error = pd.Series(
        (0 for _ in range_iterations), dtype=int, name=("Session", "P recall error"),
    )

    session_success = pd.Series(
        (False for _ in range_iterations), dtype=bool, name=("Session", "Success"),
    )

    now = 0

    item_id = None
    timestamp = None
    was_success = None

    itr = 0

    with tqdm(total=tk.ss_n_iter * tk.n_ss, file=sys.stdout) as pbar:
        for n_session in range(tk.n_ss):
            for n_iter in range(tk.ss_n_iter):
                # print(n_session)
                # print(n_iter)

                # Decide item to present
                item_id = teacher.ask(
                    now=now,
                    last_was_success=was_success,
                    last_time_reply=timestamp,
                    idx_last_q=item_id,
                )

                # Save presented item
                item_history.iloc[itr] = item_id

                timestamp = now
                p = psychologist.p(item=item_id, param=tk.param, now=timestamp)

                # was_success = np.random.random() < p
                was_success = rng.random() < p
                session_success.iloc[itr] = p
                print(p)

                # print("itr", itr, "item", item, "p", p, "success", was_success)

                # history[itr] = item

                if use_teacher_psy:
                    # Save inferred parameters
                    # inferred_param[itr] = teacher.psychologist.inferred_learner_param()
                    inferred_param.iloc[
                        itr
                    ] = teacher.psychologist.inferred_learner_param()
                    # inferred_param_error =
                    # print(inferred_param.iloc[itr])

                    # TODO: Compute the error of prediction
                    # (average of sum over all the items)
                else:
                    psychologist.learner.update(item=item_id, timestamp=timestamp)
                    # You don't need to do it here

                session_time_delta.iloc[itr] = pd.Timedelta(now, unit="s")
                now += tk.time_per_iter
                itr += 1
                pbar.update()

            now += tk.time_per_iter * tk.ss_n_iter_between
    if use_teacher_psy:
        return item_history, inferred_param

    else:
        return item_history


