import plot.success
import plot.p_recall

from learner.rl import QLearner
from learner.act_r import ActR
from learner.act_r_custom import ActRMeaning, ActRGraphic, ActRPlus
from teacher.niraj import NirajTeacher
from teacher.avya import AvyaTeacher
from teacher.tugce import TugceTeacher
from teacher.random import RandomTeacher


def run(student_model, student_param=None, teacher_model=None,  n_item=25, grade=1, t_max=150,
        track_p_recall=False):

    """
        :param teacher_model: Can be one of those:
            * NirajTeacher
            * RandomTeacher
        :param grade: Level of difficulty of the kanji selected (1: easiest)
        :param n_item: Positive integer (above the number of possible answers displayed)

        :param t_max: Positive integer (zero excluded). If grade=1, should be between 6 and 79.

        :param student_model: Class to use for creating the learner. Can be:
            * ActR
            * ActRMeaning
            * ActRGraphic
            * ActRPlus

        :param student_param: dictionary containing the parameters when creating the instance of the learner.

            For Act-R models:
            * d: decay rate
            * tau: recalling threshold
            * s: noise/stochasticity
            * g: weight for graphic connection
            * m: weight for semantic connection

            For RL models:
            * alpha: learning rate
            * tau: exploration-exploitation ratio (softmax temperature)

        :param track_p_recall: If true, can represent the evolution of the probabilities of recall for each item

        :return: None
        """

    if student_param is None:

        if student_model == QLearner:
            student_param = {"alpha": 0.1, "tau": 0.05}

        if student_model == ActR:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06}

        elif student_model == ActRMeaning:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.1}

        elif student_model == ActRGraphic:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "g": 0.1}

        elif student_model == ActRPlus:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.1, "g": 0.1}

    assert student_model in (ActR, ActRMeaning, ActRGraphic, ActRPlus, QLearner), "Student model not recognized."
    assert teacher_model in (NirajTeacher, AvyaTeacher, TugceTeacher, RandomTeacher), "Teacher model not recognized."

    teacher = teacher_model(t_max=t_max, n_item=n_item, grade=grade)
    learner = student_model(param=student_param, tk=teacher.tk, track_p_recall=track_p_recall)

    print(f"\nSimulating data with a student {student_model.__name__} (parameters={student_param}), "
          f"and a teacher {teacher_model.__name__} "
          f"using {n_item} kanji of grade {grade} for {t_max} time steps...", end=" ", flush=True)

    questions, replies, successes = teacher.teach(agent=learner)

    print("Done.\n")

    # Figures
    extension = f'{student_model.__name__}_{teacher_model.__name__}'
    plot.success.curve(successes, fig_name=f"success_curve_{extension}.pdf")
    plot.success.scatter(successes, fig_name=f"success_scatter_{extension}.pdf")
    if track_p_recall:
        plot.p_recall.curve(p_recall=learner.p, fig_name=f'p_recall_{extension}.pdf')


def main():

    for teacher_model in (NirajTeacher, RandomTeacher):
        run(student_model=ActRMeaning, teacher_model=teacher_model)


if __name__ == "__main__":

    main()
