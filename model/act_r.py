import numpy as np

from model.generic import Learner, Task


class ActRParam:

    def __init__(self, d, tau, s, g=None, m=None, g_mu=None, g_sigma=None,
                 m_mu=None, m_sigma=None):

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


class ActRLearner(Learner):

    """
    A chunk is composed of:
    * a type (here: means)
    * several slots (here: slot 1: kanji, slot2: meaning)
    """

    def __init__(self, parameters, task_features):

        super().__init__()

        if type(parameters) == dict:
            self.pr = ActRParam(**parameters)
        else:
            self.pr = ActRParam(*parameters)

        self.tk = Task(**task_features)

        self.p_random = 1/self.tk.n_possible_replies

        # Time recording of presentations of chunks
        self.time_presentation = [[] for _ in range(self.tk.n_items)]

        # Time counter
        self.t = 0

        # self.square_root_2 = 2**(1/2)

    def _activation_function(self, i):

        """The activation of a chunk is the sum of its base-level activation and some noise
        IN FUTURE: ...and the activations it receives from elements attended to. """

        # noise = np.random.normal()
        # print(self.base_level_learning_activation(i))
        return self._base_level_learning_activation(i)  # + noise

    def _base_level_learning_activation(self, i):

        """The base-level activation measures how much time has elapsed since the jth use:"""

        # noinspection PyTypeChecker
        sum_a = np.sum([
            ((self.t - t_presentation) / self.tk.t_max)**(-self.pr.d)
            for t_presentation in self.time_presentation[i]
        ])

        b = np.log(sum_a) if sum_a > 0 else -np.inf
        return b

    def _probability_of_retrieval_equation(self, a):

        """The probability of a chunk being above some retrieval threshold τ is"""

        x = (self.pr.tau - a) / self.pr.s

        # Avoid overflow
        if x < -10**2:  # 1 / (1+exp(-1000)) equals approx 1.
            return 1

        elif x > 700:  # 1 / (1+exp(700)) equals approx 0.
            return 0

        else:
            try:
                return 1 / (1 + np.exp(x))
            except FloatingPointError as e:
                print(f'x={x}, tau = {self.pr.tau}, a = {a}, s = {self.pr.s}')
                raise e

    def _update_time_presentation(self, question):

        # noinspection PyTypeChecker
        self.time_presentation[question].append(self.t)
        self.t += 1

    def _p_retrieve(self, item):

        a = self._activation_function(item)
        return self._probability_of_retrieval_equation(a)

    def _p_choice(self, question, reply, possible_replies=None):

        p_retrieve = self._p_retrieve(question)
        p = self.p_random + p_retrieve*(1 - self.p_random)

        success = question == reply

        if success:
            return p
        else:
            return (1-p) / (self.tk.n_possible_replies - 1)

    def _p_correct(self, question, reply, possible_replies=None):

        p_correct = self._p_choice(question=question, reply=question)

        if question == reply:
            return p_correct
        else:
            return 1-p_correct

    def decide(self, question, possible_replies):

        p_r = self._p_retrieve(question)
        r = np.random.random()

        if p_r > r:
            reply = question
        else:
            reply = np.random.choice(possible_replies)

        return reply

    def learn(self, question, reply):

        self._update_time_presentation(question)  # We suppose the response to be always correct if recalled


class ActRPlusLearner(ActRLearner):

    def __init__(self,  parameters, task_features):

        super().__init__(parameters=parameters, task_features=task_features)

    def _p_retrieve(self, item):

        a_i = self._base_level_learning_activation(item)
        g_i, m_i = self._g_and_m(item)
        print(f"t={self.t}: a_i={a_i}; g_i={g_i}; m_i={m_i}")

        p_r = self._probability_of_retrieval_equation(
            a_i + self.pr.g * g_i + self.pr.m * m_i
        )
        print(f"t={self.t}: p={p_r}")

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
                    g_i += self.tk.c_graphic[i, j] * self._probability_of_retrieval_equation(b_j)
                except:
                    print(f'b_j: {b_j}, cg: {self.tk.c_graphic[i, j]}')

                try:
                    m_i += self.tk.c_semantic[i, j] * self._probability_of_retrieval_equation(b_j)
                except:
                    print(f'b_j: {b_j}, cs: {self.tk.c_semantic[i, j]}')

                np.seterr(all='ignore')
        g_i /= (self.tk.n_items - 1)
        m_i /= (self.tk.n_items - 1)
        return g_i, m_i


class ActRPlusPlusLearner(ActRPlusLearner):

    def __init__(self, parameters, task_features):

        super().__init__(parameters=parameters, task_features=task_features)

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
