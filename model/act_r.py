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

        if sum_a > 0:
            return np.log(sum_a)
        else:
            return -np.inf

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

    def _p_choice(self, question, reply, possible_replies=None):

        success = question == reply

        a = self._activation_function(question)
        p = self._probability_of_retrieval_equation(a)

        p = self.p_random + p*(1 - self.p_random)

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

        a = self._activation_function(question)
        p = self._probability_of_retrieval_equation(a)
        r = np.random.random()
        if p > r:
            reply = question
        else:
            reply = np.random.randint(self.tk.n_items)

        return reply

    def learn(self, question, reply):

        self._update_time_presentation(question)  # We suppose the response to be always correct if recalled


class ActRPlusLearner(ActRLearner):

    def __init__(self,  parameters, task_features):

        super().__init__(parameters=parameters, task_features=task_features)

    def _activation_function(self, i):

        return self._base_level_learning_activation(i) + self._sum_source_activation(i)

    def _sum_source_activation(self, i):

        sum_source = 0

        list_j = list(range(self.tk.n_items))
        list_j.remove(i)

        for j in list_j:

            s_j = self._probability_of_retrieval_equation(self._base_level_learning_activation(j))
            assert 0 <= s_j <= 1
            # s_j = self.base_level_learning_activation(j)

            sum_source += \
                self.pr.g * self.tk.c_graphic[i, j] * s_j + \
                self.pr.m * self.tk.c_semantic[i, j] * s_j

        return (1/2) * (1/(self.tk.n_items-1)) * sum_source
        # return sum_source


class ActRPlusPlusLearner(ActRPlusLearner):

    def __init__(self, parameters, task_features):

        super().__init__(parameters=parameters, task_features=task_features)

    def _sum_source_activation(self, i):

        sum_source = 0

        list_j = list(range(self.tk.n_items))
        list_j.remove(i)

        for j in list_j:

            s_j = self._probability_of_retrieval_equation(self._base_level_learning_activation(j))
            assert 0 <= s_j <= 1

            if s_j > 0.001:

                g_ij = self.tk.c_graphic[i, j]
                m_ij = self.tk.c_semantic[i, j]

                g_v = self.normal_pdf(g_ij, mu=self.pr.g_mu, sigma=self.pr.g_sigma) * self.pr.g_sigma

                m_v = self.normal_pdf(m_ij, mu=self.pr.m_mu, sigma=self.pr.m_sigma) * self.pr.m_sigma

                try:
                    sum_source += \
                        self.pr.g * g_v * s_j + \
                        self.pr.m * m_v * s_j
                except FloatingPointError as e:
                    print(f'g={self.pr.g}; g_v= {g_v} ; s_j = {s_j}, m={self.pr.m}, m_v={m_v}')
                    raise e

        return (1/2) * (1/(self.tk.n_items-1)) * sum_source

    @classmethod
    def normal_pdf(cls, x, mu, sigma):

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



# class ActRPlusLearner(ActRLearner):
#
#     def __init__(self,  parameters, task_features):
#
#         super().__init__(parameters=parameters, task_features=task_features)
#
#     def _activation_function(self, i):
#
#         return self.pr.s + self._normalized_activation(i) + self._sum_source_activation(i)
#
#     def _sum_source_activation(self, i):
#
#         sum_source = 0
#
#         list_j = list(range(self.tk.n_items))
#         list_j.remove(i)
#
#         for j in list_j:
#
#             s_j = self._normalized_activation(j)
#             assert 0 <= s_j <= 1
#             # s_j = self.base_level_learning_activation(j)
#
#             sum_source += \
#                 self.pr.g * self.tk.c_graphic[i, j] * s_j + \
#                 self.pr.m * self.tk.c_semantic[i, j] * s_j
#
#         return (1/2) * (1/(self.tk.n_items-1)) * sum_source
#         # return sum_source
#
#     def _normalized_activation(self, item):
#
#         a = self._base_level_learning_activation(item)
#
#         x = (self.pr.tau - a) / self.pr.s
#
#         # Avoid overflow
#         if x < -10**2:  # 1 / (1+exp(-1000)) equals approx 1.
#             return 1
#
#         elif x > 700:  # 1 / (1+exp(700)) equals approx 0.
#             return 0
#
#         else:
#             try:
#                 return 1 / (1 + np.exp(x))
#             except FloatingPointError as e:
#                 print(f'x={x}, tau = {self.pr.tau}, a = {a}, s = {self.pr.s}')
#                 raise e
#
#     def _probability_of_retrieval_equation(self, question):
#
#         a = self._activation_function(question)
#
#         """The probability of a chunk being above some retrieval threshold τ is"""
#
#         # x = (self.pr.tau - a) / self.pr.s
#         #
#         # # Avoid overflow
#         # if x < -10**2:  # 1 / (1+exp(-1000)) equals approx 1.
#         #     return 1
#         #
#         # elif x > 700:  # 1 / (1+exp(700)) equals approx 0.
#         #     return 0
#         #
#         # else:
#         #     try:
#         #         return 1 / (1 + np.exp(x))
#         #     except FloatingPointError as e:
#         #         print(f'x={x}, tau = {self.pr.tau}, a = {a}, s = {self.pr.s}')
#         #         raise e
#
#
#     def decide(self, question, possible_replies):
#
#         a = self._activation_function(question)
#         p = self._probability_of_retrieval_equation(a)
#         r = np.random.random()
#         if p > r:
#             reply = question
#         else:
#             reply = np.random.randint(self.tk.n_items)
#
#         return reply
#
#
#
# class ActRPlusPlusLearner(ActRPlusLearner):
#
#     def __init__(self, parameters, task_features):
#
#         super().__init__(parameters=parameters, task_features=task_features)
#
#     def _sum_source_activation(self, i):
#
#         sum_source = 0
#
#         list_j = list(range(self.tk.n_items))
#         list_j.remove(i)
#
#         for j in list_j:
#
#             s_j = self._probability_of_retrieval_equation(self._base_level_learning_activation(j))
#             assert 0 <= s_j <= 1
#
#             if s_j > 0.001:
#
#                 g_ij = self.tk.c_graphic[i, j]
#                 m_ij = self.tk.c_semantic[i, j]
#
#                 g_v = self.normal_pdf(g_ij, mu=self.pr.g_mu, sigma=self.pr.g_sigma) * self.pr.g_sigma
#
#                 m_v = self.normal_pdf(m_ij, mu=self.pr.m_mu, sigma=self.pr.m_sigma) * self.pr.m_sigma
#
#                 try:
#                     sum_source += \
#                         self.pr.g * g_v * s_j + \
#                         self.pr.m * m_v * s_j
#                 except FloatingPointError as e:
#                     print(f'g={self.pr.g}; g_v= {g_v} ; s_j = {s_j}, m={self.pr.m}, m_v={m_v}')
#                     raise e
#
#         return (1/2) * (1/(self.tk.n_items-1)) * sum_source
#
#     @classmethod
#     def normal_pdf(cls, x, mu, sigma):
#
#         sigma_squared = sigma ** 2
#
#         a = (x - mu) ** 2
#         b = 2 * sigma_squared
#         c = -a / b
#         if c < -700:  # exp(-700) equals approx 0
#             return 0
#
#         try:
#             d = np.exp(c)
#         except FloatingPointError as e:
#             print(f'x={x}; mu={mu}; sigma={sigma}')
#             raise e
#         e = (2 * np.pi * sigma_squared) ** (1 / 2)
#         f = 1 / e
#         return f * d
#
