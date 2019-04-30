import random


def get_next_node(questions, successes, agent, n_items):
    """
    :param questions: list of integers (index of questions). Empty at first iteration
    :param successes: list of booleans (True: success, False: failure) for every question
    :param agent: agent object (RL, ACT-R, ...) that implements at least the following methods:
        * p_recall(item): takes index of a question and gives the probability of recall for the agent in actual state
        * learn(item): strengthen the association between a kanji and its meaning
    :param n_items:
        * Number of items included
    :return: integer (index of the question to ask)
    """
    new_question = random.randint(0, n_items-1)
    return new_question
