from learner.carlos_exponential import Exponential
from teacher.random import RandomTeacher

import plot.p_recall


def main(t_max=300, n_item=30):

    teacher = RandomTeacher(t_max=t_max, n_item=n_item, handle_similarities=True, normalize_similarity=True,
                            verbose=False, seed=10)
    agent = Exponential(param={"alpha": 0.5, "beta": 0.01, "n_0": 0.06},
                        tk=teacher.tk)
    # questions, replies, successes = teacher.teach(agent=agent)
    teacher.teach(agent=agent)
    # plot.p_recall.curve(agent.p)
    print(agent.success)
    print(agent.hist)


if __name__ == "__main__":
    main()
