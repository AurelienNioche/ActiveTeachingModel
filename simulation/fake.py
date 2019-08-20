import numpy as np


def generate_fake_task_param(n_item):
    semantic_connections = np.zeros((n_item, n_item))

    for i in range(n_item):
        for j in range(n_item):
            if i == j:
                semantic_connections[i, j] = np.nan
            elif i < j:
                semantic_connections[i, j] = np.random.random()
            else:
                semantic_connections[i, j] = semantic_connections[j, i]

    task_param = {
        "semantic_connections": semantic_connections
    }

    return task_param
