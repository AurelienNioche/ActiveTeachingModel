# import plot.p_recall
from learner.act_r_custom import ActRMeaning
from teacher.random import RandomTeacher


def main(t_max=300, n_item=30):

    teacher = RandomTeacher(t_max=t_max, n_item=n_item,
                            handle_similarities=True,
                            normalize_similarity=True,
                            verbose=True)

    agent = ActRMeaning(param={"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02},
                        tk=teacher.tk)
    # questions, replies, successes = teacher.teach(agent=agent)
    teacher.teach(agent=agent)
    # plot.p_recall.curve(agent.p)


if __name__ == "__main__":
    for i in range(6, 80):
        main(t_max=5, n_item=i)
