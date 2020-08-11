#%%
import os
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product, combinations

import string

from typing import Hashable, Iterable, Mapping

from numpy.random import default_rng

import paths
from plot.subplot import chocolate, swarm, efficiency_cost, error


rng = default_rng()

EPSILON = np.finfo(float).eps

MODELS = frozenset({"Learner", "Psychologist", "Teacher"})

LEARNERS = frozenset({"exponential", "walsh2018"})
PSYCHOLOGISTS = frozenset({"bayesian", "black_box"})
TEACHERS = frozenset({"leitner", "threshold", "sampling"})

BKP_DIR = paths.BKP_DIR

NUM_AGENTS = 30
NUM_TEACHERS = len(TEACHERS)
NUM_LEARNERS = len(LEARNERS)
NUM_PSYCHOLOGISTS = len(PSYCHOLOGISTS)

NUM_CONDITIONS = NUM_TEACHERS + NUM_LEARNERS + NUM_PSYCHOLOGISTS
NUM_OBSERVATIONS = NUM_CONDITIONS * NUM_AGENTS

T_TOTAL = 300

models = (LEARNERS, PSYCHOLOGISTS, TEACHERS)


# Gen fake data
def get_trial_conditions(models: Iterable) -> np.ndarray:
    """Makes combinations of all models."""

    return np.array(tuple(product(*models)))


def make_fake_df(models: Iterable, num_agents: int) -> pd.DataFrame:
    product_len = len(get_trial_conditions(models))
    """Make the big array of cumulative items learnt"""

    print("Making the cumulative items learnt dataframe")
    # Reduce
    to_df = get_trial_conditions(models)
    for _ in range(NUM_AGENTS - 1):
        to_df = np.vstack((to_df, get_trial_conditions(models)))
    df = pd.DataFrame.from_records(to_df)

    df.columns = ("Learner", "Psychologist", "Teacher")

    items_learnt = pd.Series(
        abs(rng.normal(100, 50, size=product_len * num_agents)),
    ).astype(int)

    t_computation = pd.Series(
        abs(rng.normal(1000, 1000, size=product_len * num_agents)),
    ).astype(int)

    df["Items learnt"] = items_learnt
    df["Computation time"] = t_computation
    df["Agent ID"] = np.hstack(np.mgrid[:NUM_AGENTS, :product_len][0])
    print("Done!")

    return df


def get_multiplying_factors_fake(
    factor: float,
    factor_of_factor: float
    ) -> tuple:
    """
    Make factors to multiply fake data and
    make it more realistic.
    """

    teacher_factors = pd.Series(
        {
            "leitner": 1,
            "threshold": factor,
            "sampling": factor * factor_of_factor,
        },
        name="Teacher factor",
        dtype=float,
    )

    learner_factors = pd.Series(
        {
            "exponential": 1,
            "walsh2018": factor,
        },
        name="Learner factor",
        dtype=float,
    )

    psychologist_factors = pd.Series(
        {
            "bayesian": 1,
            "black_box": factor,
        },
        name="Psychologist factor",
        dtype=float,
    )

    return teacher_factors, learner_factors, psychologist_factors


def df_times_factors(df: pd.DataFrame, fake_factors: tuple) -> pd.DataFrame:
    """Add model factors as column and multiply by the values"""

    for fake_factor in fake_factors:
        assert "factor" not in df.columns
        assert " " in fake_factor.name
        model = next(iter(set(df.columns).intersection(fake_factor.name.split(" "))))
        #print(fake_factor.name, model)
        df = pd.merge(df, fake_factor, how="left", left_on=model, right_index=True)
        #print(df.columns)
        df["Items learnt"] = df["Items learnt"] * df[fake_factor.name]
        df["Computation time"] = df["Computation time"] * df[fake_factor.name]
    return df


def get_plot_values(df, sort_column: str, groupby_columns: Iterable, value_column: str) -> dict:
    """Get values to plot from a groupby"""

    dict_plot_values = {}
    for index, sub_df in df.sort_values(sort_column).groupby(groupby_columns):
        dict_plot_values[frozenset(index)] = sub_df[value_column].values
    return dict_plot_values


def map_teacher_colors() -> dict:
    """
    Map each of the teachers to a color
    Warning: assumes the teachers
    """

    return {
        "leitner": "blue",
        "threshold": "orange",
        "sampling": "green",
    }



def prepare_data_chocolate(
    fake_factor: float,
    fake_factor_of_factor: float,
    models: Iterable,
    num_agents: int,
    bkp_path: str,
    force_save: bool="False",
    ) -> pd.DataFrame:
    """Group functions to make fake data"""

    bkp_file_path = os.path.join(bkp_path, "cumulative_df.p")
    if not os.path.exists(bkp_file_path) or force_save:

        factors = get_multiplying_factors_fake(fake_factor, fake_factor_of_factor)
        df = make_fake_df(models, num_agents)
        df = df_times_factors(df, factors)

        print("Saving object as pickle file...")
        pickle.dump(df, open(bkp_file_path, "wb"))
        print("Done!")

    else:
        print("Loading from pickle file...")
        df = pickle.load(open(bkp_file_path, "rb"))
        print("Done!")

    return df


# Time evolution fake data
def make_fake_p_recall_agent(t_total: int) -> np.ndarray:
    """Probability of recall fake data after some time"""

    noise = abs(rng.normal(0.1, 0.03, size=t_total))

    p_recall = np.arange(t_total) / t_total
    #p_recall = np.array(tuple(map(lambda x: x if x < 1 else 1 - EPSILON, p_recall)))

    p_recall_error = np.array(1 - p_recall, dtype=float)
    # Use values to define std of noise
    p_recall_error *= noise
    # Adjust the final value with lower mid p_value
    p_recall_error += 0.25
    return p_recall_error

# # Reduce
# plt.close("all")
# error, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey=True)
# plt.tight_layout()
# for i in range(4):
#     for j in range(2):
#         axes[i,j].set_xlabel("Time")
#         axes[i,j].set_ylabel("Error")
#         #sns.lineplot(data=p_recall_error, ax=axes[i, j])
#         axes[i,j].plot(p_recall_error)
# #axes[0, 0].text(0,0,"dfsaf")#-0.1, 1.1, string.ascii_uppercase[1], transform=axes[0,0].transAxes, size=20)
# error.savefig(os.path.join(paths.FIG_DIR, "error.pdf"))

def make_fake_primary_data_df(
    models: Iterable,
    num_agents: int,
    t_total: int,
    bkp_path: str,
    force_save: bool="False",
    ) -> pd.DataFrame:
    """Fake data one entry per iteration"""

    # Pickle management
    bkp_file_path = os.path.join(bkp_path, "primary_df.p")
    if not os.path.exists(bkp_file_path) or force_save:
        product_len = len(get_trial_conditions(models))
        p_recall_error = make_fake_p_recall_agent(t_total)
        for _ in range(num_agents * product_len - 1):
            p_recall_error = np.hstack((p_recall_error, make_fake_p_recall_agent(t_total)))
        p_recall_error_all = pd.Series(p_recall_error, name="p recall error", dtype=float)

        df = get_trial_conditions(models)
        for _ in range(NUM_AGENTS - 1):
            df = np.vstack((df, get_trial_conditions(models)))
        df = pd.DataFrame.from_records(df)
        df.columns = ("Learner", "Psychologist", "Teacher")
        df = df.loc[df.index.repeat(t_total)]

        assert df.shape[0] == p_recall_error_all.shape[0]

        df["p recall error"] = p_recall_error_all
        df["Agent ID"] = np.hstack(np.mgrid[:num_agents, :product_len * t_total][0])

        print("Saving object as pickle file...")
        pickle.dump(df, open(bkp_file_path, "wb"))
        print("Done!")

    else:
        print("Loading from pickle file...")
        df = pickle.load(open(bkp_file_path, "rb"))
        print("Done!")

    return df

def main():
    models = (LEARNERS, PSYCHOLOGISTS, TEACHERS)
    performance = prepare_data_chocolate(1.3, 1.2, models, NUM_AGENTS, paths.BKP_DIR, force_save=False)
    dict_cond_scores = get_plot_values(performance, "Agent ID", ["Teacher", "Learner", "Psychologist"], "Items learnt")
    chocolate.plot_chocolate(TEACHERS, LEARNERS, PSYCHOLOGISTS, performance, map_teacher_colors(), paths.FIG_DIR, dict_cond_scores)
    swarm.plot_swarm(performance, paths.FIG_DIR)
    efficiency_cost.plot_efficiency_and_cost(performance, paths.FIG_DIR)
    conditions = make_fake_primary_data_df(models, NUM_AGENTS, T_TOTAL, paths.BKP_DIR, force_save=False)
    dict_cond_scores = get_plot_values(conditions, "Agent ID", ["Teacher", "Learner", "Psychologist"], "p recall error")
    error.plot_error(TEACHERS, LEARNERS, PSYCHOLOGISTS, conditions, map_teacher_colors(), paths.FIG_DIR, dict_cond_scores)


if __name__ == "__main__":
    main()

