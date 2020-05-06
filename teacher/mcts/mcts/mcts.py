"""
Adapted from: https://github.com/pbsinclair42/MCTS/blob/master/mcts.py
"""
import time
import numpy as np


class Node:
    def __init__(self, state, parent):

        self.state = state

        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal

        self.parent = parent
        self.children = {}

        self.num_visits = 0
        self.total_reward = 0

    def __str__(self):
        return str(self.state)


class MCTS:
    def __init__(self, time_limit=None, iteration_limit=None,
                 exploration_constant=1 / np.sqrt(2),
                 verbose=False):

        """
        :param time_limit: int (seconds)
        :param iteration_limit: int
        :param exploration_constant: float
        :param verbose: bool
        """

        if time_limit is not None:
            if iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.time_limit = time_limit
            self.limit_type = 'time'
        else:
            if iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iteration_limit
            self.limit_type = 'iterations'
        self.exploration_constant = exploration_constant

        self.verbose = verbose

        self.root = None

    def run(self, initial_state):
        """
        :param initial_state: Node
        :return: int (action)
        """
        self.root = Node(initial_state, None)

        if self.limit_type == 'time':
            time_limit = time.time() + self.time_limit
            while time.time() < time_limit:
                self._execute_round()
        else:
            for i in range(self.searchLimit):
                self._execute_round()

        best_child = self._get_best_child(self.root, 0)
        return self._get_action(self.root, best_child)

    def _execute_round(self):
        node = self._select_node(self.root)
        reward = self._rollout(node.state)
        self._backpropogate(node, reward)

    def _select_node(self, node):
        """
        :param node: Node
        :return: Node
        """

        if self.verbose:
            print(f"selecting starting from node {node}")
        while not node.is_terminal:

            if node.is_fully_expanded:
                if self.verbose:
                    print(f"Node '{node}' is fully expanded")
                node = self._get_best_child(node, self.exploration_constant)
            else:
                node = self._expand(node)
                break
        if self.verbose:
            print(f"Selected node is {node}")
        return node

    def _expand(self, node):
        """
        :param node: Node
        :return: Node
        """
        if self.verbose:
            print(f"Expand node {node}")

        actions = node.state.get_possible_actions()
        # print(str(node))
        for action in actions:
            if action not in node.children:
                new_node = Node(node.state.take_action(action), node)
                node.children[action] = new_node
                if len(actions) == len(node.children):
                    node.is_fully_expanded = True

                if self.verbose:
                    print(f"New node: {new_node}")
                return new_node

        raise Exception("Should never reach here")

    def _get_best_child(self, node, exploration_value):

        if self.verbose:
            print(f"Get best child of node {node} in childs: "
                  f"{node.children.values()}")
        best_value = - np.inf
        best_nodes = []
        for child in node.children.values():
            node_value = \
                child.total_reward / child.num_visits \
                + exploration_value * np.sqrt(
                    2 * np.log(node.num_visits) / child.num_visits)
            # print(node_value)
            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)
        # print("best nodes", best_nodes)
        best_node = np.random.choice(best_nodes)
        if self.verbose:
            print(f"Best node is: {best_node}")
        return best_node

    def _rollout(self, state):

        """
        Using random policy
        :param state:
        :return:
        """

        if self.verbose:
            print(f"Rollout for state {state}")

        while not state.is_terminal():
            try:
                action = np.random.choice(state.get_possible_actions())
            except IndexError:
                raise Exception(
                    "Non-terminal state has no possible actions: " + str(
                        state))
            state = state.take_action(action)
        return state.get_reward()

    @classmethod
    def _backpropogate(cls, node, reward):
        while node is not None:
            node.num_visits += 1
            node.total_reward += reward
            node = node.parent

    @classmethod
    def _get_action(cls, root, best_child):
        for action, node in root.children.items():
            if node is best_child:
                return action
