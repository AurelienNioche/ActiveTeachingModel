import numpy as np
from tqdm import tqdm

import plot.memory_trace
import plot.success
import plot.n_seen
import plot.n_learnt
from learner.act_r import ActR
from learner.act_r_custom import ActRMeaning, ActRGraphic, ActRPlus
from learner.rl import QLearner
from simulation.memory import p_recall_over_time_after_learning
from teacher.avya import AvyaTeacher
# from teacher.avya_leitner import AvyaLeitTeacher
from teacher.leitner import LeitnerTeacher
from teacher.random import RandomTeacher

import matplotlib.pyplot as plt
from plot.generic import save_fig

from utils.utils import dic2string, load, dump

from plot.generic import save_fig
from simulation.data import SimulatedData, Data
from fit.fit import Fit

import os


def run(student_model, teacher_model, student_param,
        n_item, grades, t_max, normalize_similarity):

    teacher = teacher_model(t_max=t_max, n_item=n_item,
                            normalize_similarity=normalize_similarity,
                            grades=grades)

    learner = student_model(param=student_param, tk=teacher.tk)

    iterator = tqdm(range(t_max))  # if verbose else range(t_max)

    questions = np.zeros(t_max, dtype=int)
    replies = np.zeros(t_max, dtype=int)
    successes = np.zeros(t_max, dtype=bool)
    possible_replies = np.zeros((t_max, teacher.tk.n_possible_replies),
                                dtype=int)
    seen = np.zeros((n_item, t_max), dtype=bool)

    model_learner = student_model(
        tk=teacher.tk,
        param=student_model.generate_random_parameters()
    )

    teacher.agent = model_learner

    print(learner.d)
    raise Exception

    for t in iterator:

        question, poss_replies = teacher.ask()

        reply = learner.decide(
            question=question,
            possible_replies=poss_replies)
        learner.learn(question=question)

        # For backup
        questions[t] = question
        replies[t] = reply
        successes[t] = reply == question
        possible_replies[t] = poss_replies

        # Update the count of item seen
        if t > 0:
            seen[:, t] = seen[:, t - 1]
        seen[question, t] = True

        # Update the model of the learner

        data_view = Data(n_items=n_item,
                         questions=questions[:t+1],
                         replies=replies[:t+1],
                         possible_replies=possible_replies[:t+1, :])

        f = Fit(model=student_model, tk=teacher.tk, data=data_view)
        fit_r = f.evaluate()

        model_learner.set_parameters(fit_r['best_param'])
        # We assume that the matching is (0,0), (1, 1), (n, n)
        # print(model_learner.d)
        # print("IIII")
    return questions, replies, successes


def main():

    run(student_model=ActRMeaning, teacher_model=AvyaTeacher,
        student_param={"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02},
        n_item=40, grades=(1, ), t_max=500, normalize_similarity=True
        )


if __name__ == '__main__':
    main()
