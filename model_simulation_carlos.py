# from learner.carlos_power import Power
# from learner.carlos_exponential import Exponential
# from learner.carlos_decay import Decay
from teacher.random import RandomTeacher
from learner.act_r import ActR

import plot.p_recall
import plot.success


def run(model, parameters, t_max=3000, n_item=30):

    teacher = RandomTeacher(t_max=t_max, n_item=n_item,
                            handle_similarities=True,
                            normalize_similarity=True,
                            verbose=False, seed=10)

    agent = model(param=parameters, tk=teacher.tk)
    # questions, replies, successes = teacher.teach(agent=agent)
    questions, replies, successes = teacher.teach(agent=agent)
    # plot.p_recall.curve(agent.p)
    # print(agent.success)
    print(agent.hist)

    print("Done.\n")

    # Figures
    extension = f'{model.__name__}_{RandomTeacher.__name__}'
    plot.success.curve(successes,
                       fig_name=f"success_curve_{extension}.pdf")
    plot.success.scatter(successes,
                         fig_name=f"success_scatter_{extension}.pdf")


def main():
    # run(Exponential, {"alpha": 0.9, "beta": 0.5, "n_0": 0.9})
    # run(Power, {"alpha": 0.9, "beta": 0.5, "n_0": 0.9, "w": 5})
    # run(Decay, {"difficulty": 1})
    run(ActR, {"d": 1.1, "tau": 1.1, "s": 1.1}, n_item=30)


if __name__ == "__main__":
    main()
