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

    def __init__(self,  task_features, parameters=None, verbose=False, track_p_recall=False):

        if parameters is not None:

            if type(parameters) == dict:
                self.pr = ActRMeaningParam(**parameters)

            elif type(parameters) in (tuple, list, np.ndarray):
                self.pr = ActRMeaningParam(*parameters)

            else:
                raise Exception(f"Type {type(parameters)} is not handled for parameters")

        super().__init__(task_features=task_features, verbose=verbose, track_p_recall=track_p_recall)

        if parameters is not None:
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

        list_j = list(range(self.tk.n_items))
        list_j.remove(i)

        for j in list_j:

            b_j = self._base_level_learning_activation(j)
            if b_j > 0:
                x_i += self.c_x[i, j] * b_j

        x_i /= (self.tk.n_items - 1)
        return x_i


class ActRGraphic(ActRMeaning):

    def __init__(self,  parameters, task_features, verbose=False, track_p_recall=False):

        if type(parameters) == dict:
            self.pr = ActRGraphicParam(**parameters)
        elif type(parameters) in (tuple, list, np.ndarray):
            self.pr = ActRGraphicParam(*parameters)
        else:
            raise Exception(f"Type {type(parameters)} is not handled for parameters")

        super().__init__(task_features=task_features, verbose=verbose, track_p_recall=track_p_recall)

        self.c_x = self.tk.c_graphic
        self.x = self.pr.g


class ActRPlus(ActR):

    def __init__(self, task_features, parameters, verbose=False, track_p_recall=False):

        if type(parameters) == dict:
            self.pr = ActRPlusParam(**parameters)
        elif type(parameters) in (tuple, list, np.ndarray):
            self.pr = ActRPlusParam(*parameters)
        else:
            raise Exception(f"Type {type(parameters)} is not handled for parameters")

        super().__init__(task_features=task_features, verbose=verbose, track_p_recall=track_p_recall)

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

        list_j = list(range(self.tk.n_items))
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
        g_i /= (self.tk.n_items - 1)
        m_i /= (self.tk.n_items - 1)
        return g_i, m_i


class ActRPlusPlus(ActRPlus):

    def __init__(self, parameters, task_features, verbose=False, track_p_recall=False):

        if type(parameters) == dict:
            self.pr = ActRPlusPlusParam(**parameters)
        elif type(parameters) in (tuple, list, np.ndarray):
            self.pr = ActRPlusPlusParam(*parameters)
        else:
            raise Exception(f"Type {type(parameters)} is not handled for parameters")

        super().__init__(task_features=task_features, verbose=verbose, track_p_recall=track_p_recall)

    def _g_and_m(self, i):

        g_i = 0
        m_i = 0

        list_j = list(range(self.tk.n_items))
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
