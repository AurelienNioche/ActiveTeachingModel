import os
import numpy as np
from tqdm import tqdm

from learner.act_r_custom import ActRMeaning
from simulation.memory import p_recall_over_time_after_learning
from teacher.avya import AvyaTeacher

from simulation.data import Data
# from fit.bayesian import BayesianFit
# from fit.bayesian_gpyopt import BayesianGPyOptFit
# from fit.bayesian_pygpgo import BayesianPYGPGOFit
from fit.fit import Fit

# import matplotlib.pyplot as plt

# from plot.generic import save_fig
# import plot.memory_trace
# import plot.success
# import plot.n_seen
# import plot.n_learnt

from utils.utils import dic2string, dump, load

import warnings


def run(student_model, teacher_model, student_param,
        n_item, grades, t_max, normalize_similarity,
        max_iter):

    teacher = teacher_model(t_max=t_max, n_item=n_item,
                            normalize_similarity=normalize_similarity,
                            grades=grades)

    learner = student_model(param=student_param, tk=teacher.tk)

    iterator = tqdm(range(t_max))  # if verbose else range(t_max)

    param_values = student_model.generate_random_parameters()

    model_learner = student_model(
        tk=teacher.tk,
        param=param_values
    )

    hist_param_values = np.zeros((len(student_model.bounds), t_max))

    for t in iterator:

        question, possible_replies = teacher.ask(
            agent=model_learner,
            make_learn=False)

        reply = learner.decide(
            question=question,
            possible_replies=possible_replies)

        learner.learn(question=question)

        teacher.register_question_and_reply(question=question, reply=reply,
                                            possible_replies=possible_replies)

        # Update the model of the learner
        data_view = Data(n_items=n_item,
                         questions=teacher.questions[:t+1],
                         replies=teacher.replies[:t+1],
                         possible_replies=teacher.possible_replies[:t+1, :])

        f = Fit(model=student_model, tk=teacher.tk,
                data=data_view)
        fit_r = f.evaluate(max_iter=max_iter)
        if fit_r is not None:
            param_values = fit_r['best_param']
            model_learner.set_parameters(param_values)
        else:
            warnings.warn("Fit did not end up successfully :/")

        model_learner.learn(question=question)

        hist_param_values[:, t] = [param_values[tup[0]]
                                   for tup in student_model.bounds]

    p_recall = p_recall_over_time_after_learning(
        agent=learner,
        t_max=t_max,
        n_item=n_item)

    return {
        'seen': teacher.seen,
        'p_recall': p_recall,
        'questions': teacher.questions,
        'replies': teacher.replies,
        'successes': teacher.successes,
        'student_param': student_param,
        'hist_param_values': hist_param_values
    }


def main(force=True):

    student_model = ActRMeaning
    teacher_model = AvyaTeacher

    student_param = {"d": 0.5, "tau": 0.01, "s": 0.06, "m": 0.02}

    normalize_similarity = True

    t_max = 1000
    n_item = 30
    grades = (1, )

    max_iter = 50

    extension = f'{teacher_model.__name__}_{student_model.__name__}_' \
        f'{dic2string(student_param)}_' \
        f'ni_{n_item}_grade_{grades}_tmax_{t_max}_norm_{normalize_similarity}'

    bkp_file = os.path.join('bkp', 'full_stack', f'{extension}.p')

    r = load(bkp_file)
    if r is not None and not force:
        return r
    else:
        r = run(
            student_model=student_model, teacher_model=teacher_model,
            student_param=student_param,
            n_item=n_item, grades=grades, t_max=t_max,
            normalize_similarity=normalize_similarity,
            max_iter=max_iter)

        dump(r, bkp_file)
        return r


if __name__ == '__main__':
    main()
