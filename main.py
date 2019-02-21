import numpy as np

from graph import plot

from model.learner import QLearner, ActRLearner, ActRCogLearner
from model.teacher import Teacher
from model.task import TaskConnect


def run_exercise(task, teacher, learner):

    for t in range(task.t_max):

        question = teacher.ask_question(t=t)
        reply = learner.decide(question=question)
        correct_answer, success = teacher.evaluate(t=t, reply=reply)
        learner.learn(question=question, reply=reply, correct_answer=correct_answer)

    return teacher.summarize()


def plot_results(success, condition='test'):

    plot.success_scatter_plot(success, fig_name=f'{condition}_scatter.pdf')
    plot.success_curve(success, fig_name=f'{condition}_curve.pdf')


def main():

    np.random.seed(123)

    task = TaskConnect()
    teacher = Teacher(task=task, plan_questions_in_advance=True)

    for learner in (
        QLearner(task=task),
        ActRLearner(task=task),
        # ActRCogLearner(task=task)
    ):

        success = run_exercise(task, teacher, learner)
        plot_results(success, condition=learner.name)


if __name__ == "__main__":

    main()
