import plot.memory_trace
from learner.act_r import ActR
from learner.act_r_custom import ActRMeaning
from teacher.random import RandomTeacher

from behavior.data_structure import Task

import datetime

import numpy as np


def main():

    total_time = datetime.timedelta(days=10)
    sec = int(total_time.total_seconds())

    time_norm = 100/sec

    n_iteration = 100
    n_item = 9

    time = np.random.randint(low=0, high=sec, size=n_iteration)
    time.sort()

    questions = np.random.randint(low=0, high=n_item, size=n_iteration)

    success = np.zeros(n_iteration, dtype=bool)

    agent = ActR(
        param={"d": 0.5, "tau": 0.01, "s": 0.06},
        tk=Task(t_max=n_iteration, n_item=n_item))

    for i in range(n_iteration):

        item, t = questions[i], time[i]

        p = agent.p_recall(item=item, time=t/sec*100)
        r = np.random.random()

        success[i] = p > r

        agent.learn(question=item, time=t/sec*100)

    samp_size = 100
    time_sampling = np.linspace(start=0, stop=sec, num=samp_size)
    p_recall = np.zeros((n_item, samp_size))

    for t_idx, t in enumerate(time_sampling):

        for item in range(n_item):
            p_recall[item, t_idx] = agent.p_recall(item=item, time=t)

    # # questions, replies, successes = teacher.teach(agent=agent)
    # teacher.teach(agent=agent)
    #
    plot.memory_trace.plot(p_recall_value=p_recall,
                           p_recall_time=time_sampling,
                           success_time=time,
                           success_value=success,
                           questions=questions)


if __name__ == "__main__":
    main()
