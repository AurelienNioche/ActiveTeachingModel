import plot.success
import plot.p_recall

# noinspection PyUnresolvedReferences
from learner.rl import QLearner
# noinspection PyUnresolvedReferences
from learner.act_r import ActR
# noinspection PyUnresolvedReferences
from learner.act_r_custom import ActRMeaning, ActRGraphic, ActRPlus
from teacher.niraj_teacher import NirajTeacher
from teacher.random_teacher import RandomTeacher


def run(n_items=25, t_max=150, model=ActRPlus, parameters=None, track_p_recall=False):

    """
    :param n_items: Positive integer (above the number of possible answers displayed)

    :param t_max: Positive integer (zero excluded)

    :param model: Class to use for creating the learner. Can be:
        * ActR
        * ActRMeaning
        * ActRGraphic
        * ActRPlus

    :param parameters: dictionary containing the parameters when creating the instance of the learner.

        Parameter mapping between paper (left) and code (right):
        lambda: d
        tau: tau
        theta: s
        gamma: g
        psi: m

    :param track_p_recall: If true, can represent the evolution of the probabilities of recall for each item

    :return: None
    """

    parameters = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.3, "g": 0.7} if parameters is None else parameters

    # Create data with Niraj's solver
    teacher = NirajTeacher(t_max=t_max, n_items=n_items, grade=1)
    learner = model(parameters=parameters, task_features=teacher.task_features, track_p_recall=track_p_recall)

    questions, replies, successes = teacher.teach(agent=learner)

    plot.success.curve(successes, fig_name=f'niraj_opt_teaching_{model.__name__}_success_curve.pdf')
    plot.success.scatter(successes, fig_name=f'niraj_opt_teaching_{model.__name__}_success_scatter.pdf')

    if track_p_recall:
        plot.p_recall.curve(p_recall=learner.p, fig_name=f'niraj_opt_teaching_{model.__name__}_p_recall.pdf')

    # Create data with random selection of questions
    teacher = RandomTeacher(t_max=t_max, n_items=n_items, grade=1)
    learner = model(parameters=parameters, task_features=teacher.task_features, track_p_recall=track_p_recall)
    questions, replies, successes = teacher.teach(agent=learner)

    plot.success.curve(successes, fig_name=f'niraj_rdm_teaching_{model.__name__}_success_curve.pdf')
    plot.success.scatter(successes, fig_name=f'niraj_rdm_teaching_{model.__name__}_success_scatter.pdf')

    if track_p_recall:
        plot.p_recall.curve(p_recall=learner.p, fig_name=f'niraj_rdm_teaching_{model.__name__}_p_recall.pdf')


if __name__ == "__main__":

    run(track_p_recall=False)

