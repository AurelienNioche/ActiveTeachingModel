from tqdm import tqdm

from simulation.memory import p_recall_over_time_after_learning
from simulation.data import Data

from fit.bayesian_pygpgo import BayesianPYGPGOFit


def with_bayesian_opt(teacher_model, t_max, grades, n_item,
                      normalize_similarity, student_model,
                      student_param, n_cpu,
                      max_iter, init_eval,
                      verbose):

    teacher = teacher_model(t_max=t_max, n_item=n_item,
                            normalize_similarity=normalize_similarity,
                            grades=grades)

    learner = student_model(param=student_param, tk=teacher.tk)

    iterator = tqdm(range(t_max))

    model_learner = student_model(
        tk=teacher.tk,
        param=student_model.generate_random_parameters()
    )

    f = BayesianPYGPGOFit(
        model=student_model, tk=teacher.tk,
        data=None, n_jobs=n_cpu, verbose=verbose
    )

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

        data_view = Data(n_items=n_item,
                         questions=teacher.questions[:t + 1],
                         replies=teacher.replies[:t + 1],
                         possible_replies=teacher.possible_replies[:t + 1, :])

        f.evaluate(max_iter=max_iter, init_evals=init_eval,
                   data=data_view)

        model_learner.set_parameters(f.best_param)

        model_learner.learn(question=question)

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
        'history_best_fit_param': f.history_best_fit_param,
        'history_best_fit_value': f.history_best_fit_value,
    }
