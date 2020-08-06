#%%
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product, combinations

from typing import Hashable

from numpy.random import default_rng

import paths

FIG_DIR = paths.FIG_DIR

rng = default_rng()


NUM_AGENTS = 30
NUM_TEACHERS = 3
NUM_LEARNERS = 2
NUM_PSYCHOLOGISTS = 2

NUM_CONDITIONS = NUM_TEACHERS + NUM_LEARNERS + NUM_PSYCHOLOGISTS
NUM_OBSERVATIONS = NUM_CONDITIONS * NUM_AGENTS

models = {"learner", "psychologist", "teacher"}

learners = {"exponential", "walsh2018"}
psychologists = {"bayesian", "black_box"}
teachers = {"leitner", "threshold", "sampling"}


#%% Gen fake data
def get_trial_conditions():
    return np.array(tuple(product(learners, psychologists, teachers)))

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

performance["items_learnt"] = items_learnt
performance["t_computation"] = t_computation
performance["agent"] = np.hstack(np.mgrid[:NUM_AGENTS, :product_len][0])

#%% Chocolate
dict_cond_scores = {}
for index, df in performance.sort_values("agent").groupby(["teacher", "learner", "psychologist"]):
    dict_cond_scores[frozenset(index)] = df["items_learnt"].values

teacher_combos = tuple(combinations(teachers, 2))
learner_psy_combos = tuple(product(learners, psychologists))

chocolate, axs = plt.subplots(
    len(learner_psy_combos),
    len(teacher_combos),
    sharex=True,
    sharey=True,
    subplot_kw=dict(),
    gridspec_kw=dict(),
    figsize=(10,10)
)


for i, learner_psy_combo in enumerate(learner_psy_combos):
    for j, teacher_combo in enumerate(teacher_combos):

        axs[i,j].plot(
            dict_cond_scores[frozenset({teacher_combo[0] , *learner_psy_combo})],
            dict_cond_scores[frozenset({teacher_combo[1] , *learner_psy_combo})],
            "o",
        )
        axs[i,j].plot([0, 1], [0, 1], "--k", transform=axs[i,j].transAxes)
        axs[i,j].set_xlabel(teacher_combo[0])
        axs[i,j].set_ylabel(teacher_combo[1])

chocolate.savefig(os.path.join(paths.FIG_DIR, "chocolate.pdf"))


#%% Efficiency vs. computational cost
eff_cost_data = performance.groupby("teacher").sum()
eff_cost = sns.relplot(x="items_learnt", y="t_computation", hue=eff_cost_data.index,  data=eff_cost_data)
eff_cost.savefig(os.path.join(paths.FIG_DIR, "eff_cost.pdf"))

#%% Other proposals for the same data
box = sns.catplot(x="teacher", y="items_learnt", kind="box", data=performance)
box.savefig(os.path.join(paths.FIG_DIR, "box.pdf"))

swarm = sns.catplot( x="learner", y="items_learnt",hue="teacher", kind="swarm", data=performance)
swarm.savefig(os.path.join(paths.FIG_DIR, "swarm.pdf"))

multibox = sns.catplot( x="learner", y="items_learnt",hue="teacher", kind="box", data=performance)
multibox.savefig(os.path.join(paths.FIG_DIR, "multibox.pdf"))
