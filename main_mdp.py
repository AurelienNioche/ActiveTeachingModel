import numpy as np
from itertools import product

"""
Adapted from http://aima.cs.berkeley.edu/python/mdp.html
"""

class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of T(s, a, s') being  probability number for each
    state/action/state triplet, we instead have T(s, a) return a list of (p, s')
    pairs.  We also keep track of the possible states, terminal states, and
    actions for each state. [page 615]"""

    def __init__(self, reward, transition, terminal, possible_action, gamma=.9):
        self.reward = reward
        self.transition = transition
        self.terminal = terminal
        self.gamma = gamma

        self.n_state, self.n_action, _ = transition.shape
        self.states = np.arange(self.n_state)

        self.possible_action = possible_action

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(self, state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        return self.transition[state, action]

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if self.terminal[state]:
            return None
        else:
            return np.arange(self.n_action)[self.possible_action[state]]

    def value_iteration(self, epsilon=0.001):
        """Solving an MDP by value iteration. [Fig. 17.4]"""
        U1 = np.arange(self.n_state)
        while True:
            U = U1.copy()
            delta = 0
            for s in self.states:
                U1[s] = self.R(s) + \
                        self.gamma * np.max([np.sum([p * U[s1] for (p, s1) in self.T(s, a)])
                                          for a in self.actions(s)])
                delta = np.max(delta, np.abs(U1[s] - U[s]))
            if delta < epsilon * (1 - self.gamma) / self.gamma:
                 return U

    def best_policy(self, U):
        """Given an MDP and a utility function U, determine the best policy,
        as a mapping from state to action. (Equation 17.4)"""
        pi = np.arange(self.n_state)
        for s in self.states:
            pi[s] = np.argmax([self.expected_utility(a, s, U)
                               for a in self.actions(s)])
        return pi

    def expected_utility(self, a, s, U):
        """The expected utility of doing a in state s,
        according to the MDP and U"""
        return np.sum([p * U[s1] for (p, s1) in self.T(s, a)])

    def policy_iteration(self):
        """Solve an MDP by policy iteration [Fig. 17.7]"""
        U = dict([(s, 0) for s in self.states])
        pi = dict([(s, np.random.choice(self.actions(s))) for s in self.states])
        while True:
            U = self.policy_evaluation(pi, U)
            unchanged = True
            for s in self.states:
                a = np.argmax([self.expected_utility(a, s, U)
                               for a in self.actions(s)])
                if a != pi[s]:
                    pi[s] = a
                    unchanged = False
            if unchanged:
                return pi

    def policy_evaluation(self, pi, U, k=20):
        """Return an updated utility mapping U from each state in the MDP to its
        utility, using an approximation (modified policy iteration)."""
        for i in range(k):
            for s in self.states:
                U[s] = self.R(s) + self.gamma * np.sum([p * U[s] for (p, s1)
                                                in self.T(s, pi[s])])
        return U


def main():
    n_item = 5

    states = list(product(range(2), repeat=n_item))
    reward = np.array([np.sum(s) for s in states])


if __name__ == "__main__":
    main()



# # P, R = mdptoolbox.example.forest()
# vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
# for _ in range(10000):
#     vi.run()
# print(vi.policy)