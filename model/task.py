import numpy as np
import json
import itertools as it


class Task:

    def __init__(self, config_file='model/parameters/parameters.json'):

        self.parameters = self.import_parameters(config_file)

        self.t_max = int(self.parameters['t_max'])
        self.n = self.parameters['n']

        self.questions = np.arange(self.n)
        self.replies = np.arange(self.n)

    @staticmethod
    def import_parameters(config_file):

        try:
            parameters = json.load(open(config_file))

        except FileNotFoundError:
            config_file = 'model/parameters/default.json'
            parameters = json.load(open(config_file))

        finally:
            print(f"Using the config file '{config_file}'")

        return parameters

    def get_reply(self, question):

        reply = self.replies[question]

        return reply


class TaskConnect(Task):

    def __init__(self, config_file='model/parameters/parameters.json',
                 n_graphic_dimension=2, n_semantic_dimension=2):

        super().__init__(config_file)

        self.c_graphic = np.zeros((self.n, self.n))
        self.c_semantic = np.zeros((self.n, self.n))

        # Let assume that semantic is 2d and graphic is 2d
        self.compute_distances(n_graphic_dimension, n_semantic_dimension)

    def compute_distances(self, n_graphic_dimension, n_semantic_dimension):

        p_graphic = np.random.random(size=(self.n, n_graphic_dimension))
        p_semantic = np.random.random(size=(self.n, n_semantic_dimension))

        for i, j in it.combinations(range(self.n), r=2):

            dist_graph = np.abs(np.linalg.norm(p_graphic[i] - p_graphic[j]))
            dist_semantic = np.abs(np.linalg.norm(p_semantic[i] - p_semantic[j]))

            for x, y in [(i, j), (j, i)]:

                self.c_graphic[x, y] = 1 - dist_graph
                self.c_semantic[x, y] = 1 - dist_semantic
