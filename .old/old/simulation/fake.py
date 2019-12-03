import numpy as np


def _fake_connections(n_item):

    c = np.zeros((n_item, n_item))

    for i in range(n_item):
        for j in range(n_item):
            if i == j:
                c[i, j] = np.nan
            elif i < j:
                c[i, j] = np.random.random()
            else:
                c[i, j] = c[j, i]

    return c


def generate_fake_task_param(n_item, n_possible_replies=6):

    task_param = {
        "semantic_connections": _fake_connections(n_item=n_item),
        "graphic_connections": _fake_connections(n_item=n_item),
        "n_possible_replies": n_possible_replies,
        "n_item": n_item,
    }

    return task_param
