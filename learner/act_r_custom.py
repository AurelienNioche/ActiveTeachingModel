import numpy as np
from learner.act_r import ActR


class ActRMeaningParam:

    def __init__(self, d, tau, s, m):

        # Decay parameter
        self.d = d
        # Retrieval threshold
        self.tau = tau
        # Noise in the activation levels
        self.s = s

        self.m = m


class ActRGraphicParam:

    def __init__(self, d, tau, s, g):

        # Decay parameter
        self.d = d
        # Retrieval threshold
        self.tau = tau
        # Noise in the activation levels
        self.s = s

        self.g = g


class ActRPlusParam:

    def __init__(self, d, tau, s, g, m):

        # Decay parameter
        self.d = d
        # Retrieval threshold
        self.tau = tau
        # Noise in the activation levels
        self.s = s

        self.g = g
        self.m = m


class ActRPlusPlusParam:

    def __init__(self, d, tau, s, g, m, g_mu, g_sigma, m_mu, m_sigma):
        # Decay parameter
        self.d = d
        # Retrieval threshold
        self.tau = tau
        # Noise in the activation levels
        self.s = s

        self.g = g
        self.m = m

        self.g_mu = g_mu
        self.g_sigma = g_sigma

        self.m_mu = m_mu
        self.m_sigma = m_sigma


class ActRMeaning(ActR):

    bounds = ('d', 0.0000001, 1.0), ('tau', -5, 5), ('s', 0.0000001, 1), ('m', -2, 2)

    def __init__(self, tk, param=None, verbose=False, track_p_recall=False):

        if param is not None:

            if type(param) == dict:
                self.pr = ActRMeaningParam(**param)

            elif type(param) in (tuple, list, np.ndarray):
                self.pr = ActRMeaningParam(*param)

            else:
                raise Exception(f"Type {type(param)} is not handled for parameters")

        super().__init__(tk, verbose=verbose, track_p_recall=track_p_recall)

        if param is not None:
            self.x = self.pr.m
            self.c_x = self.tk.c_semantic

    def p_recall(self, item):

        a_i = self._base_level_learning_activation(item)
        if a_i > 0:
            x_i = self._x(item)
        else:
            x_i = 0

        p_r = self._sigmoid_function(
            a_i + self.x * x_i
        )
        if self.verbose:
            print(f"t={self.t}: a_i={a_i:.3f}; x_i={x_i:.3f};  p={p_r:.3f}")

        return p_r

    def _x(self, i):

        x_i = 0

        list_j = list(range(self.tk.n_item))
        list_j.remove(i)

        for j in list_j:

            b_j = self._base_level_learning_activation(j)
            if b_j > 0:
                x_i += self.c_x[i, j] * b_j

        x_i /= (self.tk.n_item - 1)
        return x_i


class ActRGraphic(ActRMeaning):

    bounds = ('d', 0.0000001, 1.0), ('tau', -5, 5), ('s', 0.0000001, 1), ('g', -2, 2)

    def __init__(self, param, tk, verbose=False, track_p_recall=False):

        if type(param) == dict:
            self.pr = ActRGraphicParam(**param)
        elif type(param) in (tuple, list, np.ndarray):
            self.pr = ActRGraphicParam(*param)
        else:
            raise Exception(f"Type {type(param)} is not handled for parameters")

        super().__init__(tk=tk, verbose=verbose, track_p_recall=track_p_recall)

        self.c_x = self.tk.c_graphic
        self.x = self.pr.g


class ActRPlus(ActR):

    bounds = ('d', 0, 1.0), ('tau', 0.00, 5), ('s', 0.0000001, 10), ('g', -10, 10), ('m', -10, 10)

    def __init__(self, tk, param, verbose=False, track_p_recall=False):

        if type(param) == dict:
            self.pr = ActRPlusParam(**param)
        elif type(param) in (tuple, list, np.ndarray):
            self.pr = ActRPlusParam(*param)
        else:
            raise Exception(f"Type {type(param)} is not handled for parameters")

        super().__init__(tk=tk, verbose=verbose, track_p_recall=track_p_recall)

    def p_recall(self, item):

        a_i = self._base_level_learning_activation(item)
        g_i, m_i = self._g_and_m(item)

        p_r = self._sigmoid_function(
            a_i + self.pr.g * g_i + self.pr.m * m_i
        )
        if self.verbose:
            print(f"t={self.t}: a_i={a_i:.3f}; g_i={g_i:.3f}; m_i={m_i:.3f};  p={p_r:.3f}")

        return p_r

    def _g_and_m(self, i):

        g_i = 0
        m_i = 0

        list_j = list(range(self.tk.n_item))
        list_j.remove(i)

        for j in list_j:

            b_j = self._base_level_learning_activation(j)
            if b_j > 1:
                np.seterr(all='raise')

                try:
                    g_i += self.tk.c_graphic[i, j] * self._sigmoid_function(b_j)
                except Exception:
                    print(f'b_j: {b_j}, cg: {self.tk.c_graphic[i, j]}')

                try:
                    m_i += self.tk.c_semantic[i, j] * self._sigmoid_function(b_j)
                except Exception:
                    print(f'b_j: {b_j}, cs: {self.tk.c_semantic[i, j]}')

                np.seterr(all='ignore')
        g_i /= (self.tk.n_item - 1)
        m_i /= (self.tk.n_item - 1)
        return g_i, m_i


class ActRPlusPlus(ActRPlus):

    bounds = ('d', 0, 1.0), ('tau', 0.00, 5), ('s', 0.0000001, 10), ('g', -10, 10), \
        ('m', -10, 10), ('g_mu', 0, 1), ('g_sigma', 0.01, 5), ('m_mu', 0, 1), ('m_sigma', 0.01, 5)

    def __init__(self, param, tk, verbose=False, track_p_recall=False):

        if type(param) == dict:
            self.pr = ActRPlusPlusParam(**param)
        elif type(param) in (tuple, list, np.ndarray):
            self.pr = ActRPlusPlusParam(*param)
        else:
            raise Exception(f"Type {type(param)} is not handled for parameters")

        super().__init__(tk=tk, verbose=verbose, track_p_recall=track_p_recall)

    def _g_and_m(self, i):

        g_i = 0
        m_i = 0

        list_j = list(range(self.tk.n_item))
        list_j.remove(i)

        for j in list_j:
            b_j = self._base_level_learning_activation(j)

            g_ij = self._normal_pdf(self.tk.c_graphic[i, j], mu=self.pr.g_mu, sigma=self.pr.g_sigma)
            m_ij = self._normal_pdf(self.tk.c_semantic[i, j], mu=self.pr.m_mu, sigma=self.pr.m_sigma)

            g_i += g_ij * b_j
            m_i += m_ij * b_j

        return g_i, m_i

    @classmethod
    def _normal_pdf(cls, x, mu, sigma):

        sigma_squared = sigma ** 2

        a = (x - mu) ** 2
        b = 2 * sigma_squared
        c = -a / b
        if c < -700:  # exp(-700) equals approx 0
            return 0

        try:
            d = np.exp(c)
        except FloatingPointError as e:
            print(f'x={x}; mu={mu}; sigma={sigma}')
            raise e
        e = (2 * np.pi * sigma_squared) ** (1 / 2)
        f = 1 / e
        return f * d
