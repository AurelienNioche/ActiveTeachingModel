from learner.act_r_custom import ActRPlus
from teacher.random_teacher import RandomTeacher
import matplotlib.pyplot as plt


def main(t_max=300, n_items=6):

    teacher = RandomTeacher(t_max=t_max, n_items=n_items, handle_similarities=False)
    agent = ActRPlus(parameters={"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.3, "g": 0.7},
                     task_features=teacher.task_features, track_p_recall=True)
    questions, replies, successes = teacher.teach(agent=agent)

    plt.plot(agent.p)
    plt.show()


if __name__ == "__main__":
    main()
