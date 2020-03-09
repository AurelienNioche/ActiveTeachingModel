import numpy as np
from itertools import product
import datetime
import mdp


def simulate_path(states, best_policy, transition, terminal):

    s = 0
    t = 0
    while True:
        a = best_policy[s]
        print(f"t={t}, s={states[s]}, action={a},", end=" ")
        s = np.where(transition[a, s] == 1)[0][0]
        print(f"new state = {states[s]}")
        if terminal[s]:
            break
        else:
            t += 1


def value_iteration(transition, reward):
    m = mdp.ValueIteration(transitions=transition, reward=reward)
    m.run()
    return m.policy


def q_learning(transition, reward):
    m = mdp.QLearning(transitions=transition, reward=reward, discount=0.96)
    m.run()
    return m.policy

def policy_iteration(transition, reward):
    m = mdp.PolicyIteration(transitions=transition, reward=reward, discount=0.96)
    m.run()
    return m.policy


# def finite_horizon(transition, reward, horizon=4):
#     m = mdp.FiniteHorizon(transitions=transition, reward=reward, N=horizon,
#                           discount=1)
#     m.run()
#     return m.policy

def basic_example():

    n_level = 3
    n_item = 5

    states = np.array(list(product(range(n_level), repeat=n_item)))
    state_list = [list(i) for i in states]
    print(f"States: {states}")
    print(f'N states: {len(states)}')

    reward = np.array([np.sum(s) for s in states])
    n_state = len(states)
    n_action = n_item
    transition = np.zeros((n_action, n_state, n_state))

    terminal = np.array([np.sum(states[s]) == n_item * (n_level-1)
                         for s in range(n_state)])

    print("Computing transition matrix")

    for s in range(n_state):
        for a in range(n_action):

            s_desc = states[s].copy()
            s_desc[a] = np.min((s_desc[a]+1, n_level-1))

            where_to_put_one = state_list.index(list(s_desc))
            # np.where(states == s_desc)[0][0]

            #transition[s, a, :] = 0
            transition[a, s, where_to_put_one] = 1

    return states, reward, transition, terminal

def main():

    # horizon = 2
    # n_item = 2
    #
    # possible_path = list(product(range(n_item), repeat=horizon))

    states, reward, transition, terminal = basic_example()



    # print(transition)
    #
    # for s in range(n_state):
    #     for s_prime in range(n_state):
    #         for a in range(n_action):
    #             print(f"If s = {states[s]} and pick action {a}, p(s' = {states[s_prime]}) = {transition[s, a, s_prime]}")
    # print("run MDP")


    print("run MDP")
    a = datetime.datetime.utcnow()

    best_policy = value_iteration(transition=transition, reward=reward)
    b = datetime.datetime.utcnow()
    print(b-a)
    print(f"best policy: {best_policy}")
    print()

    simulate_path(best_policy=best_policy, transition=transition,
                  terminal=terminal, states=states)


if __name__ == "__main__":
    main()



# # P, R = mdptoolbox.example.forest()
# vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
# for _ in range(10000):
#     vi.run()
# print(vi.policy)