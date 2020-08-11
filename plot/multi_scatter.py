#%%
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product, combinations

import string

from typing import Hashable, Iterable, Mapping

from matplotlib import cm
from numpy.random import default_rng

import paths


rng = default_rng()

EPSILON = np.finfo(float).eps

MODELS = frozenset({"Learner", "Psychologist", "Teacher"})

LEARNERS = frozenset({"exponential", "walsh2018"})
PSYCHOLOGISTS = frozenset({"bayesian", "black_box"})
TEACHERS = frozenset({"leitner", "threshold", "sampling"})

NUM_AGENTS = 30
NUM_TEACHERS = len(TEACHERS)
NUM_LEARNERS = len(LEARNERS)
NUM_PSYCHOLOGISTS = len(PSYCHOLOGISTS)

NUM_CONDITIONS = NUM_TEACHERS + NUM_LEARNERS + NUM_PSYCHOLOGISTS
NUM_OBSERVATIONS = NUM_CONDITIONS * NUM_AGENTS

T_TOTAL = 300

# Gen fake data
def get_trial_conditions():
    return np.array(tuple(product(LEARNERS, PSYCHOLOGISTS, TEACHERS)))

product_len = len(get_trial_conditions())

# Reduce. Thanks GVR
performance = get_trial_conditions()
for _ in range(NUM_AGENTS - 1):
    performance = np.vstack((performance, get_trial_conditions()))
performance = pd.DataFrame.from_records(performance)
performance.columns= ("Learner", "Psychologist", "Teacher")


items_learnt = pd.Series(
    abs(rng.normal(100, 50, size=product_len * NUM_AGENTS)),
).astype(int)

t_computation = pd.Series(
    abs(rng.normal(1000, 1000, size=product_len * NUM_AGENTS)),
).astype(int)

performance["Items learnt"] = items_learnt
performance["Computation time"] = t_computation
performance["Agent ID"] = np.hstack(np.mgrid[:NUM_AGENTS, :product_len][0])


# Adding more realistic data
FAKE_SIM_FACTOR = 1.3
FACTOR_OF_FACTOR = 1.2

teacher_factors = pd.Series(
    {
        "leitner": 1,
        "threshold": FAKE_SIM_FACTOR,
        "sampling": FAKE_SIM_FACTOR * FACTOR_OF_FACTOR,
    },
    name="Teacher factor",
    dtype=float,
)

learner_factors = pd.Series(
    {
        "exponential": 1,
        "walsh2018": FAKE_SIM_FACTOR,
    },
    name="Learner factor",
    dtype=float,
)

psychologist_factors = pd.Series(
    {
        "bayesian": 1,
        "black_box": FAKE_SIM_FACTOR,
    },
    name="Psychologist factor",
    dtype=float,
)

fake_factors = (teacher_factors, learner_factors, psychologist_factors)


# Add model factors as column and multiply by the values
for fake_factor in fake_factors:
    #print(fake_factor)
    assert "factor" not in performance.columns
    assert " " in fake_factor.name
    model = next(iter(set(performance.columns).intersection(fake_factor.name.split(" "))))
    performance = pd.merge(performance, fake_factor, how="left", left_on=model, right_index=True)
    performance["Items learnt"] = performance["Items learnt"] * performance[fake_factor.name]
    performance["Computation time"] = performance["Computation time"] * performance[fake_factor.name]


# # # Chocolate

# Get values to plot from a groupby
dict_cond_scores = {}
for index, df in performance.sort_values("Agent ID").groupby(["Teacher", "Learner", "Psychologist"]):
    dict_cond_scores[frozenset(index)] = df["Items learnt"].values

#%%

# Make the combinations of all teachers
teachers_combos = tuple(combinations(TEACHERS, 2))
# Make the combinations of all learners with all psychologists
learners_psychologists_combos = tuple(product(LEARNERS, PSYCHOLOGISTS))
# Make the combinations of all learners with all teachers
learners_teachers_combos_no_leitner = set(product(LEARNERS, TEACHERS.symmetric_difference(frozenset({"leitner"}))))

# Start the multiscatter plot
chocolate, axs = plt.subplots(
    len(learners_psychologists_combos),
    len(teachers_combos),
    sharex=True,
    sharey=True,
    #subplot_kw=dict(alpha=0.1),
    #gridspec_kw=dict(),
    figsize=(10,10)
)

# Colors

teacher_colors = {
    "leitner": "blue",
    "threshold": "orange",
    "sampling": "green",
}

def get_color_sequence(
    teacher_0: str,
    sequence_0: Iterable,
    teacher_1: str,
    sequence_1: Iterable,
    color_mapping: Mapping,
    ) -> tuple:
    """Get the color of each dot"""

    assert len(sequence_0) == len(sequence_1)
    epsilon = np.finfo(float).eps
    sequence_0 += epsilon
    sequence_1 += epsilon
    sequence = sequence_0 / sequence_1
    return tuple(map(lambda x: color_mapping[teacher_0] if x > 1 else color_mapping[teacher_1], sequence))
    #return tuple(map(lambda x: "blue" if x > 1 else "orange", sequence))

for i, learner_psychologist_combo in enumerate(learners_psychologists_combos):
    for j, teachers_combo in enumerate(teachers_combos):
        x = dict_cond_scores[frozenset({teachers_combo[0] , *learner_psychologist_combo})]
        y = dict_cond_scores[frozenset({teachers_combo[1] , *learner_psychologist_combo})]
        colors = get_color_sequence(teachers_combo[0], x, teachers_combo[1], y, teacher_colors)

        #axs[i,j].plot( x, y, "o",)
        learner = next(iter(LEARNERS.intersection(learner_psychologist_combo)))
        psychologist = next(iter(PSYCHOLOGISTS.intersection(learner_psychologist_combo)))
        num_rows = len(learners_psychologists_combos)
        num_columns = len(teachers_combos)
        if j == 0:
            i_pos_factor = i * (1 / num_rows) + (1 / num_rows / 2)
            chocolate.text(coord_min, i_pos_factor, (learner + ", " + psychologist), va="center", rotation="vertical")
        if i == 0:
            j_pos_factor = j * (1 / num_columns) + (1 / num_columns / 2)
            chocolate.text(j_pos_factor, coord_top , teachers_combo[0] + ", " + teachers_combo[1], ha="center")
        axs[i,j].scatter( x, y, c=colors, alpha=0.9, zorder=1)
        axs[i,j].plot([0, 1], [0, 1], "-k", transform=axs[i,j].transAxes, alpha=0.5, zorder=0)
        axs[i,j].set_xlabel("Items learnt " + teachers_combo[0])
        axs[i,j].set_ylabel("Items learnt " + teachers_combo[1])

axs.legend()
plt.tight_layout()
chocolate.savefig(os.path.join(paths.FIG_DIR, "chocolate.pdf"))

#%% Box
box = sns.catplot(x="Teacher", y="Items learnt", kind="swarm", data=performance)
#for ax in g.axes.flat:
#    ax.plot((0, 50), (0, .2 * 50), c=".2", ls="--")
box.savefig(os.path.join(paths.FIG_DIR, "box.pdf"))

#%%
# Efficiency vs. computational cost
eff_cost_data = performance.groupby("Teacher").sum()
eff_cost = sns.relplot(x="Items learnt", y="Computation time", hue=eff_cost_data.index,  data=eff_cost_data)
eff_cost.savefig(os.path.join(paths.FIG_DIR, "eff_cost.pdf"))

#%%
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

# Another reduce in ugly loop, thanks GVR
p_recall_error= make_fake_p_recall_agent(T_TOTAL)
plt.close("all")
error, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey=True)
plt.tight_layout()
for i in range(4):
    for j in range(2):
        axes[i,j].set_xlabel("Time")
        axes[i,j].set_ylabel("Error")
        #sns.lineplot(data=p_recall_error, ax=axes[i, j])
        axes[i,j].plot(p_recall_error)
#axes[0, 0].text(0,0,"dfsaf")#-0.1, 1.1, string.ascii_uppercase[1], transform=axes[0,0].transAxes, size=20)
error.savefig(os.path.join(paths.FIG_DIR, "error.pdf"))
#%%
for _ in range(NUM_AGENTS * product_len - 1):
    p_recall_error = np.hstack((p_recall_error, make_fake_p_recall_agent(T_TOTAL)))
p_recall_error_all = pd.Series(p_recall_error, name="p recall error", dtype=float)


conditions = get_trial_conditions()
for _ in range(NUM_AGENTS - 1):
    conditions = np.vstack((conditions, get_trial_conditions()))
conditions = pd.DataFrame.from_records(conditions)
conditions.columns = ("Learner", "Psychologist", "Teacher")
conditions = conditions.loc[conditions.index.repeat(T_TOTAL)]

assert conditions.shape[0] == p_recall_error_all.shape[0]

#trials = conditions.append(p_recall_error_all)
conditions["p recall error"] = p_recall_error_all
conditions["Agent ID"] = np.hstack(np.mgrid[:NUM_AGENTS, :product_len * T_TOTAL][0])

#%%
dict_cond_scores = {}
for index, df in conditions.sort_values("Agent ID").groupby(["Teacher", "Learner", "Psychologist"]):
    #print(df)
    dict_cond_scores[frozenset(index)] = df["p recall error"].values

#%%
plt.close("all")
error, axs = plt.subplots(
    len(learners_teachers_combos_no_leitner),
    len(PSYCHOLOGISTS),
    sharex=True,
    sharey=True,
    #subplot_kw=dict(alpha=0.9),
    #gridspec_kw=dict(),
    figsize=(10,10)
)

# Text positions
coord_min = 0.005
coord_max = 0.5
coord_top = 0.989

for i, learner_teacher_combo in enumerate(learners_teachers_combos_no_leitner):
    for j, psychologist in enumerate(PSYCHOLOGISTS):
        y = dict_cond_scores[frozenset({psychologist, *learner_teacher_combo})]
        # colors = get_color_sequence(psychologists_combo[0], x, psychologists_combo[1], y, teacher_colors)
        learner = next(iter(LEARNERS.intersection(learner_teacher_combo)))
        teacher = next(iter(TEACHERS.intersection(learner_teacher_combo)))
        color = teacher_colors[teacher]
        #print((str(learner_teacher_combo)))
        #print((str(psychologist)))
        #axs[i,j].plot( x, y, "o",)
        #axs[i,j].scatter( x, y, c=colors)
        # sns.lineplot(x, y=np.arange(len(x)), ax=axes[i, j])
        axs[i,j].plot(y, color=color)
        axs[i,j].fill_between(range(len(y)), y * 1.3, y * 0.7, alpha=0.3, color=color)
        # Text psychologist name to column
        # Text learner and teacher combo to row
        num_rows = len(learners_teachers_combos_no_leitner)
        num_columns = len(PSYCHOLOGISTS)
        if j == 0:
            i_pos_factor = i * (1 / num_rows) + (1 / num_rows / 2)
            error.text(coord_min, i_pos_factor, (learner + ", " + teacher), va="center", rotation="vertical")
        if i == 0:
            j_pos_factor = j * (1 / num_columns) + (1 / num_columns / 2)
            error.text(j_pos_factor, coord_top , psychologist, ha="center")
        #axs[i,j].plot([0, 1], [0, 1], "--k", transform=axs[i,j].transAxes)
        #axs[i,j].set_xlabel(psychologist)
        #axs[i,j].set_ylabel()
#sns.lineplot(data=p_recall_error)
# Text bottom
error.text(coord_max, coord_min, "Time", ha="center")
# Text left
error.text(coord_min, coord_max, "Items learnt", va="center", rotation="vertical")
plt.tight_layout()
plt.margins(0.9)
error.savefig(os.path.join(paths.FIG_DIR, "error.pdf"))

