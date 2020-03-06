import numpy as np
from itertools import product
import datetime
import mdp

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

        self.n_action, self.n_state,  _, = transition.shape
        self.states = np.arange(self.n_state)

        self.possible_action = possible_action

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(self, state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        return self.transition[action, state]

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
                        self.gamma * np.max([np.sum([p * U[s1] for (s1, p) in enumerate(self.T(s, a))]) for a in self.actions(s)])
                delta = np.max([delta, np.abs(U1[s] - U[s])])
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
        return np.sum([p * U[s1] for (s1, p) in enumerate(self.T(s, a))])

    def policy_iteration(self):
        """Solve an MDP by policy iteration [Fig. 17.7]"""
        U = np.zeros(self.n_state)
        pi = np.array([np.random.choice(self.actions(s)) for s in self.states])
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
        """Return an updated utility mapping U from each state in the MDP
        to its
        utility, using an approximation (modified policy iteration)."""
        for i in range(k):
            for s in self.states:
                U[s] = self.R(s) \
                       + self.gamma * np.sum([p * U[s] for (p, s1)
                                              in self.T(s, pi[s])])
        return U


def main():
    n_level = 3
    n_item = 2

    states = np.array(list(product(range(n_level), repeat=n_item)))
    state_list = [list(i) for i in states]
    print(states)
    print()

    reward = np.array([np.sum(s) for s in states])
    n_state = len(states)
    n_action = n_item
    transition = np.zeros((n_action, n_state, n_state))

    for s in range(n_state):
        for a in range(n_action):

            s_desc = states[s].copy()
            s_desc[a] = np.min((s_desc[a]+1, n_level-1))

            where_to_put_one = state_list.index(list(s_desc))
            # np.where(states == s_desc)[0][0]

            #transition[s, a, :] = 0
            transition[a, s, where_to_put_one] = 1

    # print(transition)
    #
    # for s in range(n_state):
    #     for s_prime in range(n_state):
    #         for a in range(n_action):
    #             print(f"If s = {states[s]} and pick action {a}, p(s' = {states[s_prime]}) = {transition[s, a, s_prime]}")
    # print("run MDP")
    # a = datetime.datetime.utcnow()
    # basic_mdp = MDP(reward=reward, transition=transition, terminal=np.zeros(n_state, dtype=bool),
    #                 possible_action=np.ones((n_state, n_action), dtype=bool))
    # U = basic_mdp.value_iteration()
    # print("U", U)
    # print("best policy", basic_mdp.best_policy(U))
    # b = datetime.datetime.utcnow()
    # print(b-a)

    print("run MDP v2")
    a = datetime.datetime.utcnow()
    mdp2 = mdp.ValueIteration(transitions=transition, reward=reward)
    mdp2.run()
    b = datetime.datetime.utcnow()
    print(b-a)
    print("U", mdp2.V)
    best_policy = mdp2.policy
    print("best policy", best_policy)

    s = 0
    t = 0
    while True:
        a = best_policy[s]
        print(f"t={t}, s={states[s]}, action={a},", end=" ")
        s = np.where(transition[a, s] == 1)[0][0]
        print(f"new state = {states[s]}\n")
        if np.sum(states[s]) == n_item * (n_level-1):
            break
        else:
            t+=1


if __name__ == "__main__":
    main()



# # P, R = mdptoolbox.example.forest()
# vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
# for _ in range(10000):
#     vi.run()
# print(vi.policy)