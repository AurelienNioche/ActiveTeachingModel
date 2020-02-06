# import plot.p_recall
from teacher.random import RandomTeacher
from learner.carlos_ann import Network


def main(n_iteration=300, n_item=32):

    teacher = RandomTeacher(n_iteration=n_iteration, n_item=n_item,
                            handle_similarities=True,
                            normalize_similarity=True,
                            verbose=False)

    # agent = ActRMeaning(param={"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02},
    #                    tk=teacher.tk, track_p_recall=True)
    agent = Network(param={"n_epoch": 5, "n_iteration": n_iteration, "n_hidden": 5},
                    tk=teacher.tk)
    # questions, replies, successes = teacher.teach(agent=agent)
    teacher.teach(agent=agent)
    # plot.p_recall.curve(agent.p)


if __name__ == "__main__":
    main()
