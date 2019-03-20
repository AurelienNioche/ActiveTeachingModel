import numpy as np
from scipy.stats import norm

debug = False
np.seterr(all='raise')


def softmax(x, temp):
    try:
        return np.exp(x / temp) / np.sum(np.exp(x / temp))
    except (Warning, FloatingPointError) as w:
        print(x, temp)
        raise Exception(f'{w} [x={x}, temp={temp}]')


def temporal_difference(v, obs, alpha):

    return v + alpha*(obs-v)


class Learner:

    def __init__(self):
        pass

    def decide(self, question):
        return 0

    def learn(self, question, reply):
        pass

    @property
    def name(self):
        return type(self).__name__

    def _p_choice(self, question, reply, possible_replies=None):
        return -1

    def _p_correct(self, question, reply, possible_replies=None):
        return -1

    def get_p_choices(self, exp):

        questions, replies, possible_replies, use_p_correct = exp

        t_max = len(questions)

        p_choices = np.zeros(t_max)

        for t in range(t_max):

            question, reply = questions[t], replies[t]
            possible_rep = None if possible_replies is None else possible_replies[t]

            if use_p_correct:
                p = self._p_correct(question=question, reply=reply, possible_replies=possible_rep)

            else:
                p = self._p_choice(question=question, reply=reply, possible_replies=possible_rep)
            if p == 0:
                return None

            p_choices[t] = p
            self.learn(question=question, reply=reply)

        return p_choices


class QLearner(Learner):

    def __init__(self, parameters, task_features):

        super().__init__()

        alpha, tau = parameters
        n_items, = task_features

        self.n = n_items
        self.q = np.zeros((self.n, self.n))
        self.alpha = alpha
        self.tau = tau

    def _softmax_unique(self, x_i, x):

        try:
            p = np.exp(x_i / self.tau) / np.sum(np.exp(x / self.tau))
            return p

        except (Warning, RuntimeWarning, FloatingPointError) as w:
            # print(w, x, self.tau)
            raise Exception(f'{w} [x={x}, temp={self.tau}]')

    def _p_choice(self, question, reply, possible_replies=None):

        x_i = self.q[question, reply]
        x = self.q[question, possible_replies]

        return self._softmax_unique(x_i, x)

    def _p_correct(self, question, reply, possible_replies=None):

        x_correct = self.q[question, question]
        x = self.q[question, possible_replies]

        p_correct = self._softmax_unique(x_correct, x)
        if question == reply:
            return p_correct
        else:
            return 1-p_correct

    def decide(self, question):

        p = softmax(x=self.q[question, :], temp=self.tau)
        reply = np.random.choice(np.arange(self.n), p=p)

        if debug:
            print(f'Question is: {question}')
            print(f'P values are: {[f"{p_i:.02f}" for p_i in p]}')
            print(f'Reply is {reply}')

        return reply

    def learn(self, question, reply):

        success = question == reply  # We suppose matching (0,0), (1,1) ... (n,n)

        old_q_value = self.q[question, reply]
        new_q_value = temporal_difference(v=old_q_value, obs=success, alpha=self.alpha)

        self.q[question, reply] = new_q_value

        if not success:
            self.q[question, question] = temporal_difference(v=self.q[question, question], obs=1, alpha=self.alpha)

        if debug:
            print(f'Old q value is {old_q_value}; New q value is {new_q_value}')


class ActRLearner(Learner):

    """
    A chunk is composed of:
    * a type (here: means)
    * several slots (here: slot 1: kanji, slot2: meaning)
    """

    def __init__(self, parameters, task_features):

        super().__init__()
        d, tau, s = parameters
        n_items, t_max, n_possible_replies = task_features

        self.n_items = n_items

        self.n_possible_replies = n_possible_replies if n_possible_replies is not None else n_items
        self.p_random = 1/self.n_possible_replies

        # Time recording of presentations of chunks
        self.time_presentation = [[] for _ in range(self.n_items)]

        # Decay parameter
        self._d = d

        # Retrieval threshold
        self._tau = tau

        # Noise in the activation levels
        self._s = s

        # Time counter
        self.t = 0
        self.t_max = t_max

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
            ((self.t - t_presentation) / self.t_max)**(-self._d)
            for t_presentation in self.time_presentation[i]
        ])

        if sum_a > 0:
            return np.log(sum_a)
        else:
            return -np.inf

    def _probability_of_retrieval_equation(self, a):

        """The probability of a chunk being above some retrieval threshold τ is"""

        x = (self._tau - a) / self._s

        # Avoid overflow
        if x < -10**2:  # 1 / (1+exp(-1000)) equals approx 1.
            return 1

        elif x > 700:  # 1 / (1+exp(700)) equals approx 0.
            return 0

        else:
            try:
                return 1 / (1 + np.exp(x))
            except FloatingPointError as e:
                print(f'x={x}, tau = {self._tau}, a = {a}, s = {self._s}')
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
            return (1-p) / (self.n_possible_replies - 1)

    def _p_correct(self, question, reply, possible_replies=None):

        p_correct = self._p_choice(question=question, reply=question)

        if question == reply:
            return p_correct
        else:
            return 1-p_correct

    def decide(self, question):

        a = self._activation_function(question)
        p = self._probability_of_retrieval_equation(a)
        r = np.random.random()
        if p > r:
            reply = question
        else:
            reply = np.random.randint(self.n_items)

        return reply

    def learn(self, question, reply):

        self._update_time_presentation(question)  # We suppose the response to be always correct if recalled


class ActRPlusLearner(ActRLearner):

    def __init__(self,  parameters, task_features):

        n_items, t_max, n_possible_replies, c_graphic, c_semantic = task_features
        d, tau, s, g, m = parameters

        super().__init__(parameters=(d, tau, s), task_features=(n_items, t_max, n_possible_replies, ))

        self.c_graphic = c_graphic
        self.c_semantic = c_semantic

        self.g = g
        self.m = m

    def _activation_function(self, i):

        return self._base_level_learning_activation(i) + self._sum_source_activation(i)

    def _sum_source_activation(self, i):

        sum_source = 0

        list_j = list(range(self.n_items))
        list_j.remove(i)

        for j in list_j:

            s_j = self._probability_of_retrieval_equation(self._base_level_learning_activation(j))
            assert 0 <= s_j <= 1
            # s_j = self.base_level_learning_activation(j)

            sum_source += \
                self.g * self.c_graphic[i, j] * s_j + \
                self.m * self.c_semantic[i, j] * s_j

        return (1/2) * (1/(self.n_items-1)) * sum_source
        # return sum_source


class ActRPlusPlusLearner(ActRPlusLearner):

    def __init__(self, parameters, task_features):

        d, tau, s, g, m, g_mu, g_sigma, m_mu, m_sigma = parameters
        n_items, t_max, n_possible_replies, c_graphic, c_semantic = task_features

        super().__init__(
            parameters=(d, tau, s, g, m),
            task_features=(n_items, t_max, n_possible_replies, c_graphic, c_semantic))

        self.g_mu = g_mu
        self.g_sigma = g_sigma
        self.m_mu = m_mu
        self.m_sigma = m_sigma

    def _sum_source_activation(self, i):

        sum_source = 0

        list_j = list(range(self.n_items))
        list_j.remove(i)

        for j in list_j:

            s_j = self._probability_of_retrieval_equation(self._base_level_learning_activation(j))
            assert 0 <= s_j <= 1

            if s_j > 0.001:

                g_ij = self.c_graphic[i, j]
                m_ij = self.c_semantic[i, j]

                g_v = self.normal_pdf(g_ij, mu=self.g_mu, sigma=self.g_sigma) * self.g_sigma

                m_v = self.normal_pdf(m_ij, mu=self.m_mu, sigma=self.m_sigma) * self.m_sigma

                try:
                    sum_source += \
                        self.g * g_v * s_j + \
                        self.m * m_v * s_j
                except FloatingPointError as e:
                    print(f'g={self.g}; g_v= {g_v} ; s_j = {s_j}, m={self.m}, m_v={m_v}')
                    raise e

        return (1/2) * (1/(self.n_items-1)) * sum_source

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

