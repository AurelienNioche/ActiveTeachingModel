import numpy as np
from model.learner import ExponentialForgetting


def objective(e, learnt_thr):

    param = e.param_values
    timestamps = e.timestamp_array
    hist = e.hist_array

    t = timestamps[-1]

    if e.learner_model == ExponentialForgetting.__name__:
        pass
    else:
        raise ValueError("Model not recognized")

    p_seen = ExponentialForgetting.p_seen_at_t(
        hist=hist, timestamps=timestamps, param=param, t=t
    )

    return np.sum(p_seen[:] > learnt_thr)