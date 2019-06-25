import plot.success
import plot.p_recall
import plot.memory_trace

from learner.rl import QLearner
from learner.act_r import ActR
from learner.act_r_custom import ActRMeaning, ActRGraphic, ActRPlus
from teacher.niraj import NirajTeacher
from teacher.avya import AvyaTeacher
from teacher.tugce import TugceTeacher
from teacher.random import RandomTeacher
from teacher.leitner import LeitnerTeacher


from simulation.memory import p_recall_over_time_after_learning


def run(student_model, teacher_model,
        student_param=None, n_item=25, grade=1, t_max=150):

    """
        :param teacher_model: Can be one of those:
            * NirajTeacher
            * RandomTeacher
            * AvyaTeacher
            * TugceTeacher
            * RandomTeacher
            * LeitnerTeacher
        :param grade: Level of difficulty of the kanji selected (1: easiest)
        :param n_item: Positive integer
        (above the number of possible answers displayed)

        :param t_max: Positive integer (zero excluded).
        If grade=1, should be between 6 and 79.

        :param student_model: Class to use for creating the learner. Can be:
            * ActR
            * ActRMeaning
            * ActRGraphic
            * ActRPlus

        :param student_param: dictionary containing the parameters
        when creating the instance of the learner.

            For Act-R models:
            * d: decay rate
            * tau: recalling threshold
            * s: noise/stochasticity
            * g: weight for graphic connection
            * m: weight for semantic connection

            For RL models:
            * alpha: learning rate
            * tau: exploration-exploitation ratio (softmax temperature)

        :return: None
        """

    if student_param is None:

        if student_model == QLearner:
            student_param = {"alpha": 0.1, "tau": 0.05}

        if student_model == ActR:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06}

        elif student_model == ActRMeaning:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06,
                             "m": 0.1}

        elif student_model == ActRGraphic:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06,
                             "g": 0.1}

        elif student_model == ActRPlus:
            student_param = {"d": 0.5, "tau": 0.01, "s": 0.06,
                             "m": 0.1,
                             "g": 0.1}

    assert student_model in \
        (ActR, ActRMeaning, ActRGraphic, ActRPlus, QLearner), \
        "Student model not recognized."
    assert teacher_model in \
        (NirajTeacher, AvyaTeacher, TugceTeacher,
         RandomTeacher, LeitnerTeacher), \
        "Teacher model not recognized."

    teacher = teacher_model(t_max=t_max, n_item=n_item, grade=grade)
    learner = student_model(param=student_param, tk=teacher.tk)

    print(f"\nSimulating data with a student {student_model.__name__} "
          f"(parameters={student_param}), "
          f"and a teacher {teacher_model.__name__} "
          f"using {n_item} kanji of grade {grade} for {t_max} time steps...",
          end=" ", flush=True)

    questions, replies, successes = teacher.teach(agent=learner)

    print("Done.\n")

    # Figures for success
    extension = f'{student_model.__name__}_{teacher_model.__name__}'
    plot.success.curve(successes,
                       fig_name=f"success_curve_{extension}.pdf")
    plot.success.scatter(successes,
                         fig_name=f"success_scatter_{extension}.pdf")

    # Figure combining probability of recall and actual successes
    p_recall = p_recall_over_time_after_learning(
        agent=learner,
        t_max=t_max,
        n_item=n_item)

    plot.memory_trace.plot(p_recall_value=p_recall,
                           success_value=successes,
                           questions=questions,
                           fig_name=f"memory_trace_{extension}.pdf")


def main():

    for teacher_model in (AvyaTeacher, RandomTeacher):
        run(student_model=ActRMeaning, teacher_model=teacher_model)


if __name__ == "__main__":

    main()
