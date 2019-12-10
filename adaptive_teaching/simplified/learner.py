import numpy as np


EPS = np.finfo(np.float).eps

# np.seterr(all='raise')


def log_p_grid(grid_param, delta_i, n_pres_i, n_success_i):

    n_param_set, n_param = grid_param.shape
    p = np.zeros((n_param_set, 2))

    if n_pres_i == 0:
        pass

    else:

        if n_param == 2:
            fr = grid_param[:, 0] * (1 - grid_param[:, 1]) ** (n_pres_i-1)

        elif n_param == 3:
            if n_pres_i == 1:
                fr = grid_param[:, 0]
            else:
                fr = grid_param[:, 0] \
                    * (1 - grid_param[:, 1]) ** (n_pres_i - n_success_i - 1) \
                    * (1 - grid_param[:, 2]) ** n_success_i
        else:
            raise Exception
        p[:, 1] = np.exp(- fr * delta_i)

    p[:, 0] = 1 - p[:, 1]
    return np.log(p + EPS)


def learner_p(param, n_pres_i, delta_i, n_success_i):

    n_param = len(param)

    if n_pres_i:
        if n_param == 2:
            fr = param[0] * (1 - param[1]) ** (n_pres_i - 1)
        elif n_param == 3:
            fr = param[0] \
                * (1 - param[1]) ** (n_pres_i - n_success_i - 1) \
                * (1 - param[2]) ** n_success_i
        else:
            raise Exception
        p = np.exp(- fr * delta_i)

    else:
        p = 0

    return p


def learner_fr_p_seen(n_success, param, n_pres, delta):

    n_param = len(param)

    seen = n_pres[:] > 0

    if n_param == 2:
        fr = param[0] * (1 - param[1]) ** (n_pres[seen] - 1)

    elif n_param == 3:
        fr = param[0] \
             * (1 - param[1]) ** (n_pres[seen] - n_success[seen] - 1) \
             * (1 - param[2]) ** n_success[seen]
    else:
        raise Exception

    p = np.exp(-fr * delta[seen])
    return fr, p
