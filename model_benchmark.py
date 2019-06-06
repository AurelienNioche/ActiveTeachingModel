import time

from learner.act_r_custom import ActRPlus
from teacher.random import RandomTeacher


def main(t_max=300, n_items=6, model=ActRPlus, parameters=None, track_p_recall=False):

    parameters = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.3, "g": 0.7} if parameters is None else parameters

    teacher = RandomTeacher(t_max=t_max, n_items=n_items, handle_similarities=False)
    agent = model(parameters=parameters, task_features=teacher.task_features, track_p_recall=track_p_recall)

    a = time.time()
    teacher.teach(agent=agent)
    b = time.time()
    print(f'Time needed (ms): {(b - a) * 1000:.0f}')


if __name__ == "__main__":
    main()

