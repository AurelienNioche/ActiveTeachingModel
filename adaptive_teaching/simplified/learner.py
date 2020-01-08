import numpy as np
from abc import ABC, abstractmethod


EPS = np.finfo(np.float).eps

# np.seterr(all='raise')


class GenericLearner:

    bounds = ()
    param_labels = ()

    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def _log_p_grid(cls, grid_param, delta_i, n_pres_i, n_success_i, i,
                    hist, timestamps, t):
        pass

    @classmethod
    @abstractmethod
    def p(cls, param, n_pres_i, delta_i, n_success_i, i, hist, timestamps, t):
        pass

    @classmethod
    @abstractmethod
    def p_seen(cls, param, n_pres, delta, n_success, hist,
               timestamps, t):
        pass

    @classmethod
    def log_lik(cls,
                grid_param, delta, n_pres, n_success, hist,
                timestamps, t):

        n_item = len(n_pres)
        n_param_set = len(grid_param)

        log_lik = np.zeros((n_item, n_param_set, 2))

        for i in range(n_item):
            log_lik[i, :, :] = cls._log_p_grid(
                grid_param=grid_param,
                delta_i=delta[i],
                n_pres_i=n_pres[i],
                n_success_i=n_success[i],
                i=i,
                hist=hist,
                timestamps=timestamps,
                t=t
            )

        return log_lik


class ExponentialForgetting(GenericLearner):

    bounds = (0., 0.25), (0., 0.50),
    param_labels = "alpha", "beta"

    def __init__(self):
        super().__init__()

    @classmethod
    def _log_p_grid(cls, grid_param, delta_i, n_pres_i, n_success_i, i,
                    hist, timestamps, t):

        n_param_set, n_param = grid_param.shape
        p = np.zeros((n_param_set, 2))

        if n_pres_i == 0:
            pass

        elif delta_i == 0:
            p[:, 1] = 1

        else:

            fr = grid_param[:, 0] * (1 - grid_param[:, 1]) ** (n_pres_i-1)

            p[:, 1] = np.exp(- fr * delta_i)

        p[:, 0] = 1 - p[:, 1]
        return np.log(p + EPS)

    @classmethod
    def p(cls, param, n_pres_i, delta_i, n_success_i, i, hist, timestamps, t):

        if n_pres_i:

            fr = param[0] * (1 - param[1]) ** (n_pres_i - 1)

            p = np.exp(- fr * delta_i)

        else:
            p = 0

        return p

    @classmethod
    def fr_p_seen(cls, n_success, param, n_pres, delta):

        seen = n_pres[:] > 0

        fr = param[0] * (1 - param[1]) ** (n_pres[seen] - 1)

        p = np.exp(-fr * delta[seen])
        return fr, p

    @classmethod
    def p_seen(cls, param, n_pres, delta, n_success, hist,
               timestamps, t):

        return cls.fr_p_seen(n_success=n_success,
                             param=param, n_pres=n_pres, delta=delta)[1]


class ExponentialForgettingAsymmetric(GenericLearner):

    bounds = (0., 0.25), (0., 0.50), (0., 0.50),
    param_labels = "alpha", "beta_minus", "beta_plus"

    def __init__(self):
        super().__init__()

    @classmethod
    def _log_p_grid(cls, grid_param, delta_i, n_pres_i, n_success_i, i,
                    hist, timestamps, t):

        n_param_set, n_param = grid_param.shape
        p = np.zeros((n_param_set, 2))

        if n_pres_i == 0:
            pass

        elif delta_i == 0:
            p[:, 1] = 1

        else:

            if n_pres_i == 1:
                fr = grid_param[:, 0]
            else:
                fr = grid_param[:, 0] \
                    * (1 - grid_param[:, 1]) ** (n_pres_i - n_success_i - 1) \
                    * (1 - grid_param[:, 2]) ** n_success_i

            p[:, 1] = np.exp(- fr * delta_i)

        p[:, 0] = 1 - p[:, 1]
        return np.log(p + EPS)

    @classmethod
    def p(cls, param, n_pres_i, delta_i, n_success_i, i, hist, timestamps, t):

        if n_pres_i:

            fr = param[0] \
                * (1 - param[1]) ** (n_pres_i - n_success_i - 1) \
                * (1 - param[2]) ** n_success_i

            p = np.exp(- fr * delta_i)

        else:
            p = 0

        return p

    @classmethod
    def fr_p_seen(cls, n_success, param, n_pres, delta):

        seen = n_pres[:] > 0

        fr = param[0] \
            * (1 - param[1]) ** (n_pres[seen] - n_success[seen] - 1) \
            * (1 - param[2]) ** n_success[seen]

        p = np.exp(-fr * delta[seen])
        return fr, p

    @classmethod
    def p_seen(cls, param, n_pres, delta, n_success, hist,
               timestamps, t):

        return cls.fr_p_seen(n_success=n_success,
                             param=param, n_pres=n_pres, delta=delta)[1]


class ActR(GenericLearner):

    bounds = (
        (0.001, 1.0),
        (-20, 20),
        (0.001, 5.0)
    )
    param_labels = "d", "tau", "s",

    def __init__(self):
        super().__init__()

    @classmethod
    def _log_p_grid(cls,
                    grid_param, delta_i, n_pres_i, n_success_i, i,
                    hist, timestamps, t):

        n_param_set, n_param = grid_param.shape

        assert n_param == 3

        p = np.zeros((n_param_set, 2))

        if n_pres_i == 0:
            p[:, 1] = 0

        else:

            time_presentation = timestamps[(hist[:] == i).nonzero()[0]]

            time_elapsed = t - time_presentation
            # print('time_elapsed', time_elapsed)

            # Presentation effect
            pe = np.zeros((n_pres_i, n_param_set))
            for pres in range(n_pres_i):
                pe[pres] = np.power(time_elapsed[pres], -grid_param[:, 0])

            # Activation
            a = np.log(pe.sum(axis=0))

            # Sigmoid
            x = (grid_param[:, 1] - a) / (grid_param[:, 2] * np.square(2))

            small = x[:] < -10 ** 2  # 1 / (1+exp(-1000)) equals approx 1.

            big = x[:] > 700  # 1 / (1+exp(700)) equals approx 0.

            neither_small_or_big = np.logical_not(small + big)
            p[neither_small_or_big, 1] = \
                1 / (1 + np.exp(x[neither_small_or_big]))
            p[small, 1] = 1
            p[big, 1] = 0

        p[:, 0] = 1 - p[:, 1]
        return np.log(p + EPS)

    @classmethod
    def p(cls, param, n_pres_i, delta_i, n_success_i, i, hist, timestamps, t):

        time_presentation = timestamps[(hist[:] == i).nonzero()[0]]
        if not len(time_presentation):
            return 0

        time_elapsed = t - time_presentation

        # Presentation effect
        pe = np.power(time_elapsed, -param[0]).sum()

        # Activation
        a = np.log(pe)

        # Sigmoid
        x = (param[1] - a) / (param[2] * np.square(2))

        return 1 / (1 + np.exp(x))

    @classmethod
    def p_seen(cls, n_success, param, n_pres, delta, hist, timestamps, t):

        seen = n_pres[:] > 0

        n_seen = np.sum(seen)
        n_item = len(n_pres)

        p = np.zeros(n_seen)

        for i, item in enumerate(np.arange(n_item)[seen]):
            n_pres_i = n_pres[item]
            p[i] = cls.p(
                param=param,
                n_pres_i=n_pres_i,
                delta_i=None,
                n_success_i=None,
                i=item,
                hist=hist,
                timestamps=timestamps,
                t=t)

        return p

#
# class ExponentialForgetting(GenericLearner):
#
#     bounds = (0., 0.25), (0., 0.25),
#     param_labels = "alpha", "beta"
#
#     def __init__(self):
#         super().__init__()
#
#     @classmethod
#     def _log_p_grid(cls, grid_param, delta_i, n_pres_i, n_success_i, i,
#                     hist, timestamps, t):
#
#         n_param_set, n_param = grid_param.shape
#         p = np.zeros((n_param_set, 2))
#
#         if n_pres_i == 0:
#             pass
#
#         elif delta_i == 0:
#             p[:, 1] = 1
#
#         else:
#
#             if n_param == 2:
#                 fr = grid_param[:, 0] * (1 - grid_param[:, 1]) ** (n_pres_i-1)
#
#             elif n_param == 3:
#                 if n_pres_i == 1:
#                     fr = grid_param[:, 0]
#                 else:
#                     fr = grid_param[:, 0] \
#                         * (1 - grid_param[:, 1]) ** (n_pres_i - n_success_i - 1) \
#                         * (1 - grid_param[:, 2]) ** n_success_i
#             else:
#                 fr = grid_param[:, i] * (1 - grid_param[: -1]) ** (n_pres_i - 1)
#
#             p[:, 1] = np.exp(- fr * delta_i)
#
#         p[:, 0] = 1 - p[:, 1]
#         return np.log(p + EPS)
#
#     @classmethod
#     def p(cls, param, n_pres_i, delta_i, n_success_i, i, hist, timestamps, t):
#
#         n_param = len(param)
#
#         if n_pres_i:
#
#             if n_param == 2:
#                 fr = param[0] * (1 - param[1]) ** (n_pres_i - 1)
#
#             elif n_param == 3:
#                 fr = param[0] \
#                     * (1 - param[1]) ** (n_pres_i - n_success_i - 1) \
#                     * (1 - param[2]) ** n_success_i
#             else:
#                 fr = param[i] * (1 - param[-1]) ** (n_pres_i - 1)
#
#             p = np.exp(- fr * delta_i)
#
#         else:
#             p = 0
#
#         return p
#
#     @classmethod
#     def fr_p_seen(cls, n_success, param, n_pres, delta):
#
#         n_param = len(param)
#
#         seen = n_pres[:] > 0
#
#         if n_param == 2:
#             fr = param[0] * (1 - param[1]) ** (n_pres[seen] - 1)
#
#         elif n_param == 3:
#             fr = param[0] \
#                  * (1 - param[1]) ** (n_pres[seen] - n_success[seen] - 1) \
#                  * (1 - param[2]) ** n_success[seen]
#         else:
#
#             fr = param[np.arange(len(seen))[seen]] \
#                  * (1 - param[-1]) ** (n_pres[seen] - 1)
#
#         p = np.exp(-fr * delta[seen])
#         return fr, p
#
#     @classmethod
#     def p_seen(cls, param, n_pres, delta, n_success, hist,
#                timestamps, t):
#
#         return cls.fr_p_seen(n_success=n_success,
#                              param=param, n_pres=n_pres, delta=delta)[1]