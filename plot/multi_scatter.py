#%%
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product, combinations

from typing import Hashable, Iterable, Mapping

from matplotlib import cm
from numpy.random import default_rng

import paths


rng = default_rng()

EPSILON = np.finfo(float).eps

MODELS = frozenset({"learner", "psychologist", "teacher"})

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
performance.columns= ("learner", "psychologist", "teacher")


items_learnt = pd.Series(
    abs(rng.normal(100, 50, size=product_len * NUM_AGENTS)),
).astype(int)

t_computation = pd.Series(
    abs(rng.normal(1000, 1000, size=product_len * NUM_AGENTS)),
).astype(int)

performance["Items learnt"] = items_learnt
performance["Computation time"] = t_computation
performance["agent"] = np.hstack(np.mgrid[:NUM_AGENTS, :product_len][0])


# Adding more realistic data
FAKE_SIM_FACTOR = 1.3
FACTOR_OF_FACTOR = 1.2

teacher_factors = pd.Series(
    {
        "leitner": 1,
        "threshold": FAKE_SIM_FACTOR,
        "sampling": FAKE_SIM_FACTOR * FACTOR_OF_FACTOR,
    },
    name="teacher_factor",
    dtype=float,
)

learner_factors = pd.Series(
    {
        "exponential": 1,
        "walsh2018": FAKE_SIM_FACTOR,
    },
    name="learner_factor",
    dtype=float,
)

psychologist_factors = pd.Series(
    {
        "bayesian": 1,
        "black_box": FAKE_SIM_FACTOR,
    },
    name="psychologist_factor",
    dtype=float,
)

fake_factors = (teacher_factors, learner_factors, psychologist_factors)


# Add model factors as column and multiply by the values
for fake_factor in fake_factors:
    assert "factor" not in performance.columns
    assert "_" in fake_factor.name
    model = next(iter(set(performance.columns).intersection(fake_factor.name.split("_"))))
    performance = pd.merge(performance, fake_factor, how="left", left_on=model, right_index=True)
    performance["Items learnt"] = performance["Items learnt"] * performance[fake_factor.name]
    performance["Computation time"] = performance["Computation time"] * performance[fake_factor.name]


# # # Chocolate

# Get values to plot from a groupby
dict_cond_scores = {}
for index, df in performance.sort_values("agent").groupby(["teacher", "learner", "psychologist"]):
    dict_cond_scores[frozenset(index)] = df["Items learnt"].values

#%%
# Make the combinations of all teachers
teacher_combos = tuple(combinations(TEACHERS, 2))
# Make the combinations of all learners with all psychologists
learner_psy_combos = tuple(product(LEARNERS, PSYCHOLOGISTS))

# Start the multiscatter plot
chocolate, axs = plt.subplots(
    len(learner_psy_combos),
    len(teacher_combos),
    sharex=True,
    sharey=True,
    subplot_kw=dict(),
    gridspec_kw=dict(),
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

for i, learner_psy_combo in enumerate(learner_psy_combos):
    for j, teacher_combo in enumerate(teacher_combos):
        x = dict_cond_scores[frozenset({teacher_combo[0] , *learner_psy_combo})]
        y = dict_cond_scores[frozenset({teacher_combo[1] , *learner_psy_combo})]
        colors = get_color_sequence(teacher_combo[0], x, teacher_combo[1], y, teacher_colors)

        #axs[i,j].plot( x, y, "o",)
        axs[i,j].scatter( x, y, c=colors)
        axs[i,j].plot([0, 1], [0, 1], "--k", transform=axs[i,j].transAxes)
        axs[i,j].set_xlabel(teacher_combo[0])
        axs[i,j].set_ylabel(teacher_combo[1])

chocolate.savefig(os.path.join(paths.FIG_DIR, "chocolate.pdf"))

#%% Box
box = sns.catplot(x="teacher", y="Items learnt", kind="box", data=performance)
#for ax in g.axes.flat:
#    ax.plot((0, 50), (0, .2 * 50), c=".2", ls="--")
box.savefig(os.path.join(paths.FIG_DIR, "box.pdf"))

#%%
# Efficiency vs. computational cost
eff_cost_data = performance.groupby("teacher").sum()
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
for _ in range(NUM_AGENTS * product_len - 1):
    p_recall_error = np.hstack((p_recall_error, make_fake_p_recall_agent(T_TOTAL)))
p_recall_error_all = pd.Series(p_recall_error, name="p recall error", dtype=float)

conditions = get_trial_conditions()
for _ in range(NUM_AGENTS - 1):
    conditions = np.vstack((conditions, get_trial_conditions()))
conditions = pd.DataFrame.from_records(conditions)
conditions.columns = ("learner", "psychologist", "teacher")
conditions = conditions.loc[conditions.index.repeat(T_TOTAL)]

assert conditions.shape[0] == p_recall_error_all.shape[0]

#trials = conditions.append(p_recall_error_all)
conditions["p recall error"] = p_recall_error_all
conditions["agent"] = np.hstack(np.mgrid[:NUM_AGENTS, :product_len * T_TOTAL][0])

#%%
plt.close("all")
sns.lineplot(data=p_recall_error)
plt.savefig(os.path.join(paths.FIG_DIR, "test.pdf"))

