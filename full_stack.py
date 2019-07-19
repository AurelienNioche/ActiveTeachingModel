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

import os


def run(student_model, teacher_model, student_param,
        n_item, grade, t_max, verbose=False):

    teacher = teacher_model(t_max=t_max, n_item=n_item, grade=grade)

    learner = student_model(param=student_param, tk=teacher.tk)

    iterator = tqdm(range(t_max)) \
        if verbose else range(t_max)

    questions = np.zeros(t_max, dtype=int)
    replies = np.zeros(t_max, dtype=int)

    for t in iterator:
        question, possible_replies = teacher.ask()

        reply = learner.decide(
            question=question,
            possible_replies=possible_replies)
        reply.learn(question=question)

        # Update the count of item seen
        if t > 0:
            seen[:, t] = seen[:, t - 1]
        seen[question, t] = True

        # For backup
        questions[t] = question
        replies[t] = reply
        successes[t] = reply == question
        # We assume that the matching is (0,0), (1, 1), (n, n)

        return questions, replies, successes


def main():

    run(student_model=ActRMeaning, teacher_model=AvyaTeacher,
        student_param={"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.1},
        n_item=40, grade=1, t_max=500
        )


if __name__ == '__main__':
    main()
