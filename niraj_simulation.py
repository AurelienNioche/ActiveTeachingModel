from learner.rl import QLearner
from learner.act_r import ActR, ActRMeaning, ActRGraphic, ActRPlus

import plot.success

from tqdm import tqdm

from teacher.niraj_teacher import NirajTeacher


def run_simulation(model, parameters, teacher, verbose=False):

    task_features = teacher.task_features
    agent = model(parameters=parameters, task_features=task_features, verbose=verbose)

    t_max = task_features['t_max']

    if verbose:
        print(f"Generating data with model {model.__name__}...")
        gen = tqdm(range(t_max))
    else:
        gen = range(t_max)

    questions = []
    replies = []
    successes = []

    for _ in gen:
        question, possible_replies = teacher.ask(questions=questions, successes=successes, agent=agent)

        reply = agent.decide(question=question, possible_replies=possible_replies)
        agent.learn(question=question)

        # For backup
        questions.append(question)
        replies.append(reply)
        successes.append(reply == question)  # We assume that the matching is (0,0), (1, 1), (n, n)

    return questions, replies, successes


def demo_all(t_max=100):

    """
    parameter mapping between paper (left) and code (right):
    lambda: d
    tau: tau
    theta: s
    gamma: g
    psi: m
    """

    for m, p in (
            (QLearner, {"alpha": 0.05, "tau": 0.01}),
            (ActR, {"d": 0.5, "tau": 0.05, "s": 0.4}),
            (ActRGraphic, {"d": 0.5, "tau": 0.05, "s": 0.4, "g": 0.5}),
            (ActRMeaning, {"d": 0.5, "tau": 0.05, "s": 0.5, "m": 0.7}),
            (ActRPlus, {"d": 0.5, "tau": 0.05, "s": 0.4, "m": 0.3, "g": 0.7})
    ):
        teacher = NirajTeacher(t_max=t_max)
        run_simulation(model=m, parameters=p, teacher=teacher)


def demo(t_max=150, n_items=70):

    """
    parameter mapping between paper (left) and code (right):
    lambda: d
    tau: tau
    theta: s
    gamma: g
    psi: m
    """

    import time

    model, parameters = ActRPlus, {"d": 0.5, "tau": 0.05, "s": 0.4, "m": 0.3, "g": 0.7}
    teacher = NirajTeacher(t_max=t_max, n_items=n_items, grade=1)
    a = time.time()
    questions, replies, successes = run_simulation(model=model, parameters=parameters, teacher=teacher)
    b = time.time()
    print(f'Time needed (ms): {(b-a) * 1000:.0f}')

    plot.success.curve(successes, fig_name=f'niraj_simulated_{model.__name__}_curve.pdf')
    plot.success.scatter(successes, fig_name=f'niraj_simulated_{model.__name__}_scatter.pdf')


if __name__ == "__main__":

    demo()

