import numpy as np


def objective(e, learnt_thr):

    param = e.param_values
    timestamps = e.timestamp_array
    hist = e.hist_array

    t = timestamps[-1]

    p_seen = eval(e.learner_model).p_seen_at_t(
        hist=hist, timestamps=timestamps, param=param, t=t
    )

    return np.sum(p_seen[:] > learnt_thr)