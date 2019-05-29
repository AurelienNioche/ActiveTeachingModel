from learner.act_r_custom import ActRPlus
from teacher.random_teacher import RandomTeacher

import plot.p_recall


def main(t_max=300, n_item=6):

    teacher = RandomTeacher(t_max=t_max, n_item=n_item, handle_similarities=False)
    agent = ActRPlus(param={"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.3, "g": 0.7},
                     tk=teacher.tk, track_p_recall=True)
    # questions, replies, successes = teacher.teach(agent=agent)
    teacher.teach(agent=agent)
    plot.p_recall.curve(agent.p)


if __name__ == "__main__":
    main()
