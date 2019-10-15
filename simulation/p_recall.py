import numpy as np
from tqdm import tqdm


def _compute(student_model,
             student_param,
             hist,
             n_item,
             time_sampling=None,
             time_norm=None,
             ):

    n_iteration = len(hist)

    p_recall = np.zeros((n_item, n_iteration))

    learner = student_model(param=student_param)

    for t in tqdm(range(n_iteration)):

        item_presented = hist[t]

        if time_sampling:
            time = time_sampling[t] * time_norm

        for item in range(n_item):

            if time_sampling:
                # noinspection PyUnboundLocalVariable
                p = \
                    learner.p_recall(item=item, time=time)
            else:
                print(item)
                p = \
                    learner.p_recall(item=item)

            p_recall[item, t] = p

        if time_sampling:
            learner.learn(item=item_presented, time=time)
        else:
            learner.learn(item=item_presented)

        # p_recall[:, i] = pool.map(
        #     _run,
        #     ((agent, item, i) for item in range(n_item)))

    tqdm.write('\n')

    return p_recall


def p_recall_over_time_after_learning(
        student_model,
        student_param,
        hist,
        n_item,
        time_norm=None,
        time_sampling=None):

    tqdm.write("Computing the probabilities of recall...")

    assert time_sampling is not None or time_norm is None, \
        "If continuous time, then " \
        "'time_norm' and 'time_sampling' have to be defined"

    p_recall = _compute(student_model=student_model,
                        student_param=student_param,
                        hist=hist,
                        n_item=n_item,
                        time_sampling=time_sampling,
                        time_norm=time_norm
                        )

    tqdm.write('\n')

    return p_recall
